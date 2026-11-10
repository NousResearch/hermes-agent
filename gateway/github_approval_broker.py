"""Durable, fail-closed Telegram authority for GitHub approvals."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_CHOICES = ("Approve", "Do not approve")
_ITEM_KEY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*\Z")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovalProposal:
    item_key: str
    head_sha: str
    contract_digest_sha256: str
    executor: str = "hermes"

    def canonical_key(self) -> str:
        self.validate()
        body = json.dumps(
            {"contract_digest_sha256": self.contract_digest_sha256, "executor": self.executor,
             "head_sha": self.head_sha, "item_key": self.item_key},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def validate(self) -> None:
        if not isinstance(self.item_key, str) or _ITEM_KEY_RE.fullmatch(self.item_key) is None:
            raise ValueError("item_key must be owner/repo#number with a positive canonical number")
        for name, value, length in (
            ("head_sha", self.head_sha, 40),
            ("contract_digest_sha256", self.contract_digest_sha256, 64),
        ):
            if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be lowercase hexadecimal ({length} chars)")
        if self.executor != "hermes":
            raise ValueError("GitHub approval executor must be hermes")


@dataclass(frozen=True)
class RequestOutcome:
    action: str
    proposal_key: str
    clarify_id: Optional[str] = None
    message_id: Optional[str] = None
    response: Optional[str] = None


@dataclass(frozen=True)
class ApprovalDecision:
    decision_id: int
    proposal_key: str
    response: str
    approver: str


async def _await(value):
    return await value if inspect.isawaitable(value) else value


class GitHubApprovalBroker:
    """SQLite-backed authority. Telegram and executor ambiguity always stop closed."""

    def __init__(self, path: Path, *, instance_id: str, delivery_lease_seconds: float = 30.0):
        self.path = Path(path)
        self.instance_id = str(instance_id)
        if not self.instance_id:
            raise ValueError("instance_id is required")
        self.delivery_lease_seconds = max(1.0, float(delivery_lease_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS github_approval_proposals (
                    proposal_key TEXT PRIMARY KEY,
                    item_key TEXT NOT NULL,
                    head_sha TEXT NOT NULL,
                    contract_digest_sha256 TEXT NOT NULL,
                    executor TEXT NOT NULL CHECK(executor = 'hermes'),
                    state TEXT NOT NULL CHECK(state IN (
                        'pending_notification','dispatching','delivery_unknown','active','answered','superseded')),
                    clarify_id TEXT NOT NULL UNIQUE,
                    session_key TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    authorized_user_id TEXT NOT NULL,
                    destination_id TEXT,
                    question TEXT NOT NULL,
                    message_id TEXT,
                    callback_bound INTEGER NOT NULL DEFAULT 0 CHECK(callback_bound IN (0,1)),
                    owner_instance_id TEXT,
                    lease_expires_at REAL,
                    response TEXT,
                    approver TEXT,
                    origin_status TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS github_approval_item_state
                    ON github_approval_proposals(item_key, state);
                CREATE TABLE IF NOT EXISTS github_approval_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_key TEXT NOT NULL UNIQUE REFERENCES github_approval_proposals(proposal_key),
                    response TEXT NOT NULL,
                    approver TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','claimed','resumed','stale','blocked')),
                    claim_token TEXT,
                    claim_owner TEXT,
                    claim_expires_at REAL,
                    execution_key TEXT,
                    execution_receipt TEXT,
                    decided_at REAL NOT NULL,
                    resumed_at REAL
                );
                COMMIT;
            """)
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(github_approval_proposals)")}
            if "destination_id" not in columns:
                db.execute("ALTER TABLE github_approval_proposals ADD COLUMN destination_id TEXT")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _proposal_from_row(row: sqlite3.Row | dict) -> ApprovalProposal:
        return ApprovalProposal(str(row["item_key"]), str(row["head_sha"]),
                                str(row["contract_digest_sha256"]), str(row["executor"]))

    @staticmethod
    def _callback_is_bound(adapter: Any, clarify_id: str, session_key: str) -> bool:
        probe = getattr(adapter, "approval_callback_binding", None)
        if callable(probe):
            return bool(probe(clarify_id, session_key))
        state = getattr(adapter, "_clarify_state", None)
        return isinstance(state, dict) and state.get(clarify_id) == session_key

    @staticmethod
    def _destination_identity(adapter: Any, clarify_id: str) -> Optional[str]:
        identity = getattr(adapter, "approval_destination_identity", None)
        value = identity(clarify_id) if callable(identity) else None
        return value if isinstance(value, str) and value else None

    def _claim(self, proposal: ApprovalProposal, *, chat_id: str, question: str, session_key: str,
               thread_id: Optional[str], authorized_user_id: str,
               destination_id: Optional[str]) -> tuple[RequestOutcome, list[str]]:
        key, now = proposal.canonical_key(), time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exact = db.execute("SELECT * FROM github_approval_proposals WHERE proposal_key=?", (key,)).fetchone()
            if exact is not None:
                requested_binding = (chat_id, thread_id, authorized_user_id, question)
                stored_binding = (
                    str(exact["chat_id"]),
                    str(exact["thread_id"]) if exact["thread_id"] is not None else None,
                    str(exact["authorized_user_id"]),
                    str(exact["question"]),
                )
                if requested_binding != stored_binding:
                    db.commit()
                    return RequestOutcome(
                        "binding_conflict", key, str(exact["clarify_id"]), exact["message_id"]), []
                stored_destination = exact["destination_id"]
                if stored_destination is not None and destination_id != str(stored_destination):
                    db.commit()
                    return RequestOutcome(
                        "binding_conflict", key, str(exact["clarify_id"]), exact["message_id"]), []
                if exact["state"] == "superseded":
                    db.commit()
                    return RequestOutcome("superseded", key, str(exact["clarify_id"])), []
            if exact is not None and exact["state"] == "answered":
                db.commit()
                return RequestOutcome("answered", key, str(exact["clarify_id"]), exact["message_id"],
                                      str(exact["response"])), []
            if exact is not None and exact["state"] == "active":
                db.commit()
                return RequestOutcome("reused", key, str(exact["clarify_id"]), str(exact["message_id"])), []
            if exact is not None and exact["state"] in {"dispatching", "delivery_unknown"}:
                db.commit()
                return RequestOutcome("operator_reconciliation", key, str(exact["clarify_id"])), []
            old = db.execute(
                "SELECT clarify_id FROM github_approval_proposals WHERE item_key=? AND proposal_key!=? "
                "AND state IN ('pending_notification','dispatching','active','answered')", (proposal.item_key, key)).fetchall()
            retired = [str(row["clarify_id"]) for row in old]
            db.execute("UPDATE github_approval_proposals SET state='superseded',callback_bound=0,updated_at=? "
                       "WHERE item_key=? AND proposal_key!=? AND state IN "
                       "('pending_notification','dispatching','active','answered')", (now, proposal.item_key, key))
            clarify_id = str(exact["clarify_id"]) if exact else f"gha-{key[:24]}"
            db.execute("""INSERT INTO github_approval_proposals(
                    proposal_key,item_key,head_sha,contract_digest_sha256,executor,state,clarify_id,
                    session_key,chat_id,thread_id,authorized_user_id,question,message_id,callback_bound,
                    destination_id,owner_instance_id,lease_expires_at,last_error,created_at,updated_at)
                VALUES(?,?,?,?,?,'dispatching',?,?,?,?,?,?,NULL,0,?,?,NULL,NULL,?,?)
                ON CONFLICT(proposal_key) DO UPDATE SET state='dispatching',session_key=excluded.session_key,
                    chat_id=excluded.chat_id,thread_id=excluded.thread_id,
                    authorized_user_id=excluded.authorized_user_id,question=excluded.question,message_id=NULL,
                    callback_bound=0,owner_instance_id=excluded.owner_instance_id,lease_expires_at=NULL,
                    last_error=NULL,updated_at=excluded.updated_at""",
                (key, proposal.item_key, proposal.head_sha, proposal.contract_digest_sha256, proposal.executor,
                 clarify_id, session_key, chat_id, thread_id, authorized_user_id, question,
                 destination_id, self.instance_id, now, now))
            db.commit()
        return RequestOutcome("send", key, clarify_id), retired

    async def request(self, proposal: ApprovalProposal, *, adapter: Any, chat_id: str, question: str,
                      session_key: str, authorized_user_id: str, thread_id: Optional[str] = None) -> RequestOutcome:
        required = (chat_id, question, session_key, authorized_user_id)
        if not all(isinstance(value, str) and value for value in required):
            raise ValueError("chat_id, question, session_key, and authorized_user_id are required")
        key = proposal.canonical_key()
        destination_id = self._destination_identity(adapter, f"gha-{key[:24]}")
        claim, retired = self._claim(proposal, chat_id=chat_id, question=question, session_key=session_key,
                                     thread_id=str(thread_id) if thread_id is not None else None,
                                     authorized_user_id=authorized_user_id, destination_id=destination_id)
        if claim.action != "send":
            return claim
        retire = getattr(adapter, "retire_clarify_card", None)
        if callable(retire):
            for clarify_id in retired:
                try:
                    await retire(clarify_id, "Superseded by a newer GitHub contract")
                except Exception:
                    logger.exception("Could not retire superseded GitHub approval card")
        return await self._deliver_claim(claim.proposal_key, adapter)

    async def _deliver_claim(self, proposal_key: str, adapter: Any) -> RequestOutcome:
        with self._connect() as db:
            row = db.execute("SELECT * FROM github_approval_proposals WHERE proposal_key=? AND state='dispatching'",
                             (proposal_key,)).fetchone()
        if row is None:
            observed = self.observe_key(proposal_key)
            return RequestOutcome("reused", proposal_key, observed.get("clarify_id") if observed else None,
                                  observed.get("message_id") if observed else None)
        destination_id = str(row["destination_id"]) if row["destination_id"] is not None else None
        metadata = {"is_approval_prompt": True, "github_approval": True,
                    "authorized_user_id": str(row["authorized_user_id"])}
        if row["thread_id"] is not None:
            metadata["thread_id"] = str(row["thread_id"])
        explicit_failure = False
        try:
            result = await adapter.send_clarify(
                chat_id=str(row["chat_id"]), question=str(row["question"]), choices=list(_CHOICES),
                clarify_id=str(row["clarify_id"]), session_key=str(row["session_key"]), metadata=metadata)
            if not result.success:
                explicit_failure, message_id = True, None
                error = result.error or "provider rejected delivery"
                state = "pending_notification"
            else:
                message_id = str(result.message_id) if result.message_id is not None else None
                bound = self._callback_is_bound(adapter, str(row["clarify_id"]), str(row["session_key"]))
                destination_id = self._destination_identity(adapter, str(row["clarify_id"]))
                if message_id and bound and destination_id:
                    state, error = "active", None
                else:
                    state, error = "delivery_unknown", "successful send lacked durable destination/message/callback binding"
        except Exception as exc:
            logger.exception("GitHub approval card delivery outcome is ambiguous")
            state, message_id, error = "delivery_unknown", None, f"ambiguous delivery: {type(exc).__name__}"
        now = time.time()
        with self._connect() as db:
            changed = db.execute("""UPDATE github_approval_proposals SET state=?,message_id=?,callback_bound=?,destination_id=?,
                    last_error=?,updated_at=? WHERE proposal_key=? AND state='dispatching' AND owner_instance_id=?""",
                (state, message_id if state == "active" else None, 1 if state == "active" else 0,
                 destination_id if state == "active" else row["destination_id"],
                 error, now, proposal_key, self.instance_id)).rowcount
        if changed != 1:
            return RequestOutcome("operator_reconciliation", proposal_key, str(row["clarify_id"]))
        action = "sent" if state == "active" else ("pending_retry" if explicit_failure else "operator_reconciliation")
        return RequestOutcome(action, proposal_key, str(row["clarify_id"]), message_id if state == "active" else None)

    async def retry_pending(self, *, adapter: Any, limit: int = 16) -> list[RequestOutcome]:
        destination_id = self._destination_identity(adapter, "")
        if destination_id is None:
            return []
        now, claimed = time.time(), []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT proposal_key FROM github_approval_proposals "
                              "WHERE state='pending_notification' AND destination_id=? "
                              "ORDER BY updated_at LIMIT ?", (destination_id, max(1, int(limit)))).fetchall()
            for row in rows:
                key = str(row["proposal_key"])
                if db.execute("UPDATE github_approval_proposals SET state='dispatching',owner_instance_id=?,"
                              "updated_at=? WHERE proposal_key=? AND state='pending_notification'",
                              (self.instance_id, now, key)).rowcount == 1:
                    claimed.append(key)
            db.commit()
        return [await self._deliver_claim(key, adapter) for key in claimed]

    def recover_callbacks(self, adapter: Any) -> list[dict[str, str]]:
        restore = getattr(adapter, "recover_approval_callback", None)
        state = getattr(adapter, "_clarify_state", None)
        if not callable(restore) and not isinstance(state, dict):
            return []
        destination_id = self._destination_identity(adapter, "")
        if destination_id is None:
            return []
        with self._connect() as db:
            rows = db.execute("SELECT clarify_id,session_key,message_id,chat_id,thread_id FROM github_approval_proposals "
                              "WHERE state='active' AND callback_bound=1 AND message_id IS NOT NULL "
                              "AND destination_id=?", (destination_id,)).fetchall()
        recovered = []
        for row in rows:
            if callable(restore):
                restore(
                    str(row["clarify_id"]), str(row["session_key"]), str(row["chat_id"]),
                    str(row["message_id"]), str(row["thread_id"]) if row["thread_id"] is not None else None)
            else:
                state[str(row["clarify_id"])] = str(row["session_key"])
            recovered.append({key: str(row[key]) for key in ("clarify_id", "session_key", "message_id")})
        return recovered

    def has_callback(self, clarify_id: str, destination_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM github_approval_proposals WHERE clarify_id=? AND destination_id=?",
                              (str(clarify_id), str(destination_id))).fetchone() is not None

    def callback_choice(self, clarify_id: str, choice_index: int, destination_id: str) -> Optional[str]:
        if type(choice_index) is not int or not 0 <= choice_index < len(_CHOICES):
            return None
        with self._connect() as db:
            row = db.execute("SELECT state,response FROM github_approval_proposals "
                             "WHERE clarify_id=? AND destination_id=?",
                             (str(clarify_id), str(destination_id))).fetchone()
        if row is None or row["state"] not in {"active", "answered"}:
            return None
        return str(row["response"]) if row["state"] == "answered" else _CHOICES[choice_index]

    def resolve_callback(self, clarify_id: str, choice_index: int, *, approver: str, user_id: str,
                         chat_id: str, thread_id: Optional[str], message_id: str,
                         destination_id: str) -> Optional[ApprovalDecision]:
        if type(choice_index) is not int or not 0 <= choice_index < len(_CHOICES):
            return None
        response, now = _CHOICES[choice_index], time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM github_approval_proposals WHERE clarify_id=? AND destination_id=?",
                             (str(clarify_id), str(destination_id))).fetchone()
            context = (str(user_id), str(chat_id), str(thread_id) if thread_id is not None else None, str(message_id))
            expected = ((str(row["authorized_user_id"]), str(row["chat_id"]),
                         str(row["thread_id"]) if row["thread_id"] is not None else None,
                         str(row["message_id"]))) if row is not None else None
            if row is None or row["state"] not in {"active", "answered"} or context != expected:
                db.commit()
                return None
            if row["state"] == "active":
                if approver != f"telegram:{user_id}":
                    db.commit()
                    return None
                db.execute("UPDATE github_approval_proposals SET state='answered',response=?,approver=?,"
                           "callback_bound=0,updated_at=? WHERE clarify_id=? AND state='active'",
                           (response, approver, now, str(clarify_id)))
                db.execute("INSERT INTO github_approval_decisions(proposal_key,response,approver,state,decided_at) "
                           "VALUES(?,?,?,'pending',?) ON CONFLICT(proposal_key) DO NOTHING",
                           (str(row["proposal_key"]), response, approver, now))
            decision = db.execute("SELECT decision_id,proposal_key,response,approver FROM github_approval_decisions "
                                  "WHERE proposal_key=?", (str(row["proposal_key"]),)).fetchone()
            db.commit()
        return ApprovalDecision(int(decision["decision_id"]), str(decision["proposal_key"]),
                                str(decision["response"]), str(decision["approver"])) if decision else None

    def claim_resumptions(self, *, owner: str, lease_seconds: float = 30.0, limit: int = 16) -> list[dict]:
        now, items = time.time(), []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT decision_id FROM github_approval_decisions WHERE state='pending' OR "
                              "(state='claimed' AND claim_expires_at<=?) ORDER BY decision_id LIMIT ?",
                              (now, max(1, int(limit)))).fetchall()
            for row in rows:
                token = secrets.token_hex(16)
                if db.execute("UPDATE github_approval_decisions SET state='claimed',claim_token=?,claim_owner=?,"
                              "claim_expires_at=? WHERE decision_id=? AND (state='pending' OR "
                              "(state='claimed' AND claim_expires_at<=?))",
                              (token, owner, now + max(1.0, float(lease_seconds)), int(row["decision_id"]), now)).rowcount == 1:
                    claimed = db.execute("SELECT d.*,p.session_key,p.item_key,p.head_sha,p.contract_digest_sha256,"
                                         "p.executor,p.clarify_id FROM github_approval_decisions d JOIN "
                                         "github_approval_proposals p USING(proposal_key) WHERE d.decision_id=?",
                                         (int(row["decision_id"]),)).fetchone()
                    item = dict(claimed)
                    item["claim_token"] = token
                    items.append(item)
            db.commit()
        return items

    def _finish_claim(self, claim: dict, state: str, receipt: Any = None) -> bool:
        receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":")) if receipt is not None else None
        with self._connect() as db:
            changed = db.execute("UPDATE github_approval_decisions SET state=?,execution_receipt=?,resumed_at=?,"
                                 "claim_expires_at=NULL WHERE decision_id=? AND state='claimed' AND claim_token=?",
                                 (state, receipt_json, time.time() if state == "resumed" else None,
                                  int(claim["decision_id"]), str(claim["claim_token"]))).rowcount
        return changed == 1

    async def resume_approved(self, *, owner: str, github: Any, executor: Any, adapter: Any = None,
                              limit: int = 16) -> list[str]:
        """Revalidate and continue only through a receipt-readable idempotent executor."""
        outcomes = []
        for claim in self.claim_resumptions(owner=owner, limit=limit):
            if claim["response"] != "Approve":
                self._finish_claim(claim, "resumed", {"decision": "declined"})
                outcomes.append("declined")
                continue
            reader_contract = inspect.getattr_static(github, "read_contract", None)
            reader_receipt = inspect.getattr_static(executor, "read_receipt", None)
            executor_method = inspect.getattr_static(executor, "execute_idempotent", None)
            read_contract = getattr(github, "read_contract", None) if callable(reader_contract) else None
            read_receipt = getattr(executor, "read_receipt", None) if callable(reader_receipt) else None
            execute = getattr(executor, "execute_idempotent", None) if callable(executor_method) else None
            if not all(callable(value) for value in (read_contract, read_receipt, execute)):
                self._finish_claim(claim, "blocked")
                outcomes.append("blocked_executor")
                continue
            try:
                live = await _await(read_contract(str(claim["item_key"])))
                if not isinstance(live, ApprovalProposal):
                    raise ValueError("GitHub reader returned a non-canonical contract")
                live.validate()
            except Exception:
                logger.exception("Could not re-read live GitHub approval contract")
                self._finish_claim(claim, "stale")
                outcomes.append("stale")
                continue
            expected = self._proposal_from_row(claim)
            if live != expected:
                with self._connect() as db:
                    db.execute("UPDATE github_approval_proposals SET state='superseded',callback_bound=0,updated_at=? "
                               "WHERE proposal_key=?", (time.time(), str(claim["proposal_key"])))
                if adapter is not None and callable(retire := getattr(adapter, "retire_clarify_card", None)):
                    try:
                        await retire(str(claim["clarify_id"]), "Stale GitHub head or contract")
                    except Exception:
                        logger.exception("Could not retire stale GitHub approval card")
                self._finish_claim(claim, "stale")
                outcomes.append("stale")
                continue
            execution_key = f"gha-exec-{claim['proposal_key']}"
            with self._connect() as db:
                db.execute("UPDATE github_approval_decisions SET execution_key=? WHERE decision_id=? AND claim_token=?",
                           (execution_key, int(claim["decision_id"]), str(claim["claim_token"])))
            receipt = await _await(read_receipt(execution_key))
            if receipt is None:
                receipt = await _await(execute(dict(claim), idempotency_key=execution_key))
            if (not isinstance(receipt, dict) or set(receipt) != {"id"}
                    or receipt.get("id") != execution_key):
                outcomes.append("operator_reconciliation")
                continue
            if not self._finish_claim(claim, "resumed", receipt):
                outcomes.append("operator_reconciliation")
                continue
            outcomes.append("executed")
        return outcomes

    def record_origin_status(self, proposal: ApprovalProposal, status: str) -> bool:
        with self._connect() as db:
            changed = db.execute("UPDATE github_approval_proposals SET origin_status=?,updated_at=? WHERE proposal_key=?",
                                 (str(status), time.time(), proposal.canonical_key())).rowcount
        return changed == 1

    def observe_key(self, proposal_key: str) -> Optional[dict]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM github_approval_proposals WHERE proposal_key=?", (str(proposal_key),)).fetchone()
        return dict(row) if row is not None else None

    def observe(self, proposal: ApprovalProposal) -> Optional[dict]:
        return self.observe_key(proposal.canonical_key())


def default_broker(*, instance_id: str, create: bool = True) -> Optional[GitHubApprovalBroker]:
    from hermes_constants import get_process_hermes_home
    path = get_process_hermes_home() / "state" / "github-approval-broker.db"
    if not create and not path.exists():
        return None
    return GitHubApprovalBroker(path, instance_id=instance_id)
