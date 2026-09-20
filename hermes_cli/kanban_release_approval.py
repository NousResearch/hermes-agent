"""Task/release-bound, persisted and idempotent release-approval saga.

The SQLite transactions claim/finalize a local saga; they do not pretend the
external promotion is globally ACID.  The adapter is retried with one stable
operation key and must be idempotent.  Every subprocess call is argv-only.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

from hermes_cli.kanban_db_connect import write_txn

_APPROVAL_TEXT = "freigegeben"
_REVIEW_STATUS = "Auf Dev zur Prüfung"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_LEASE_SECONDS = 60
_RESULT_WORDS = {"success", "failed", "not_started", "skipped", "active", "rolled_back"}


@dataclass(frozen=True)
class ApprovalContext:
    platform: str
    chat_id: str
    thread_id: str
    actor_id: str
    reply_to_message_id: str


@dataclass(frozen=True)
class PresentedRelease:
    gate_id: str
    task_id: str
    release_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class ApprovalResult:
    ok: bool
    classification: str
    release_id: str | None = None
    dev_result: str = "not_started"
    test_result: str = "not_started"
    production_result: str = "not_started"
    active_release_id: str | None = None
    previous_release_id: str | None = None
    rollback_available: bool = False
    detail: str = ""

    def user_message(self) -> str:
        release = self.release_id or "unbekannt"
        active = self.active_release_id or "unbekannt"
        rollback = "ja" if self.rollback_available else "nein"
        prefix = "Freigabe verarbeitet" if self.ok else "Fehler: Freigabe konnte nicht verarbeitet werden"
        return (
            f"{prefix}. Release-ID: {release}. Dev: {self.dev_result}. "
            f"Test: {self.test_result}. Prod: {self.production_result}. "
            f"Aktiv: {active}. Vorgänger: {self.previous_release_id or 'keiner'}. "
            f"Rollback verfügbar: {rollback}."
        )


def _required(value: str, name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def present_release(
    conn,
    *,
    task_id: str,
    release_id: str,
    manifest_sha256: str,
    manual_test_cases_digest: str,
    workflow_status: str,
    active_dev_release_id: str,
    platform: str,
    chat_id: str,
    thread_id: str,
    actor_id: str,
    presented_message_id: str,
    previous_release_id: str | None,
    rollback_available: bool,
    presented_at: int | None = None,
) -> PresentedRelease:
    """Persist the exact visible approval target and supersede older route gates."""
    fields = {
        "task_id": task_id, "release_id": release_id, "platform": platform,
        "chat_id": chat_id, "actor_id": actor_id, "presented_message_id": presented_message_id,
    }
    fields = {name: _required(value, name) for name, value in fields.items()}
    manifest_sha256 = str(manifest_sha256 or "").lower()
    manual_test_cases_digest = str(manual_test_cases_digest or "").lower()
    if not _DIGEST.fullmatch(manifest_sha256) or not _DIGEST.fullmatch(manual_test_cases_digest):
        raise ValueError("manifest and manual-test digests must be lowercase SHA-256 values")
    if workflow_status != _REVIEW_STATUS:
        raise ValueError(f"workflow_status must be {_REVIEW_STATUS!r}")
    if active_dev_release_id != fields["release_id"]:
        raise ValueError("active Dev release must equal the presented release")
    gate_id = "rg_" + secrets.token_hex(8)
    now = int(time.time()) if presented_at is None else int(presented_at)
    thread_id = str(thread_id or "")
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id=?", (fields["task_id"],)).fetchone() is None:
            raise ValueError("task does not exist")
        if conn.execute(
            "SELECT 1 FROM kanban_release_gates WHERE gate_status='consuming' AND "
            "(task_id=? OR (platform=? AND chat_id=? AND thread_id=? AND actor_id=?))",
            (fields["task_id"], fields["platform"], fields["chat_id"], thread_id,
             fields["actor_id"]),
        ).fetchone() is not None:
            raise RuntimeError("release promotion is in progress for this task or approval route")
        conn.execute(
            "INSERT INTO kanban_release_state "
            "(task_id,workflow_status,active_dev_release_id,manifest_sha256,updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
            "workflow_status=excluded.workflow_status,active_dev_release_id=excluded.active_dev_release_id,"
            "manifest_sha256=excluded.manifest_sha256,updated_at=excluded.updated_at",
            (fields["task_id"], workflow_status, active_dev_release_id, manifest_sha256, now),
        )
        conn.execute(
            "UPDATE kanban_release_gates SET gate_status='superseded' "
            "WHERE platform=? AND chat_id=? AND thread_id=? AND actor_id=? "
            "AND gate_status IN ('active','consuming')",
            (fields["platform"], fields["chat_id"], thread_id, fields["actor_id"]),
        )
        conn.execute(
            "INSERT INTO kanban_release_gates "
            "(id,task_id,release_id,manifest_sha256,manual_test_cases_digest,workflow_status,"
            "active_dev_release_id,platform,chat_id,thread_id,actor_id,presented_message_id,"
            "presented_at,previous_release_id,rollback_available,gate_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active')",
            (gate_id, fields["task_id"], fields["release_id"], manifest_sha256,
             manual_test_cases_digest, workflow_status, active_dev_release_id,
             fields["platform"], fields["chat_id"], thread_id, fields["actor_id"],
             fields["presented_message_id"], now, previous_release_id,
             int(bool(rollback_available))),
        )
    return PresentedRelease(gate_id, fields["task_id"], fields["release_id"], manifest_sha256)


def update_release_state(
    conn, *, task_id: str, workflow_status: str, active_dev_release_id: str,
    manifest_sha256: str, updated_at: int | None = None,
) -> None:
    """Update approval-time release truth without mutating the visible gate snapshot."""
    task_id = _required(task_id, "task_id")
    active_dev_release_id = _required(active_dev_release_id, "active_dev_release_id")
    manifest_sha256 = str(manifest_sha256 or "").lower()
    if not _DIGEST.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 value")
    now = int(time.time()) if updated_at is None else int(updated_at)
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
            raise ValueError("task does not exist")
        if conn.execute(
            "SELECT 1 FROM kanban_release_gates WHERE task_id=? AND gate_status='consuming'",
            (task_id,),
        ).fetchone() is not None:
            raise RuntimeError("release promotion is in progress for this task")
        conn.execute(
            "INSERT INTO kanban_release_state "
            "(task_id,workflow_status,active_dev_release_id,manifest_sha256,updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
            "workflow_status=excluded.workflow_status,active_dev_release_id=excluded.active_dev_release_id,"
            "manifest_sha256=excluded.manifest_sha256,updated_at=excluded.updated_at",
            (task_id, str(workflow_status or ""), active_dev_release_id, manifest_sha256, now),
        )


def _result(classification: str, gate=None, **kwargs) -> ApprovalResult:
    return ApprovalResult(
        ok=False,
        classification=classification,
        release_id=gate["release_id"] if gate is not None else None,
        active_release_id=gate["active_dev_release_id"] if gate is not None else None,
        previous_release_id=gate["previous_release_id"] if gate is not None else None,
        rollback_available=bool(gate["rollback_available"]) if gate is not None else False,
        **kwargs,
    )


def _operation_key(gate) -> str:
    bound = "\0".join((gate["id"], gate["task_id"], gate["release_id"], gate["manifest_sha256"],
                        gate["actor_id"], gate["presented_message_id"]))
    return "release-approval:" + hashlib.sha256(bound.encode()).hexdigest()


def _claim(conn, text: str, context: ApprovalContext, now: int):
    if text != _APPROVAL_TEXT:
        return None, _result("not_approval")
    with write_txn(conn):
        rows = conn.execute(
            "SELECT g.*, t.status AS task_status, s.workflow_status AS current_workflow_status, "
            "s.active_dev_release_id AS current_active_dev_release_id, "
            "s.manifest_sha256 AS current_manifest_sha256 FROM kanban_release_gates g "
            "JOIN tasks t ON t.id=g.task_id "
            "LEFT JOIN kanban_release_state s ON s.task_id=g.task_id "
            "WHERE g.platform=? AND g.chat_id=? "
            "AND g.thread_id=? AND g.actor_id=? AND g.presented_message_id=? "
            "ORDER BY g.presented_at DESC, g.id DESC",
            (context.platform, context.chat_id, context.thread_id or "", context.actor_id,
             context.reply_to_message_id),
        ).fetchall()
        if len(rows) != 1:
            return None, _result("stale_approval")
        gate = rows[0]
        if gate["gate_status"] == "approved":
            return None, _result("replay", gate)
        if (gate["gate_status"] not in {"active", "consuming"}
                or gate["workflow_status"] != _REVIEW_STATUS
                or gate["active_dev_release_id"] != gate["release_id"]
                or gate["current_workflow_status"] != _REVIEW_STATUS
                or gate["current_active_dev_release_id"] != gate["release_id"]
                or gate["current_manifest_sha256"] != gate["manifest_sha256"]
                or gate["task_status"] in {"done", "archived"}
                or not _DIGEST.fullmatch(gate["manifest_sha256"])
                or not _DIGEST.fullmatch(gate["manual_test_cases_digest"])):
            return None, _result("stale_approval", gate)
        op_key = _operation_key(gate)
        saga = conn.execute("SELECT * FROM kanban_release_sagas WHERE gate_id=?", (gate["id"],)).fetchone()
        if saga is not None and saga["state"] == "succeeded":
            return None, _result("replay", gate)
        if (saga is not None and saga["state"] == "promoting"
                and saga["lease_expires"] is not None and saga["lease_expires"] > now):
            # The current owner may still be mutating the external targets.
            # Once its lease expires, retry enters the adapter's explicit
            # resume/reconcile action with this same operation key.
            return None, _result("in_progress", gate)
        owner = secrets.token_hex(16)
        action = "promote" if saga is None else "resume"
        if saga is None:
            conn.execute(
                "INSERT INTO kanban_release_sagas "
                "(operation_key,gate_id,state,owner_token,lease_expires,created_at,updated_at) "
                "VALUES (?,?,'promoting',?,?,?,?)",
                (op_key, gate["id"], owner, now + _LEASE_SECONDS, now, now),
            )
        else:
            conn.execute(
                "UPDATE kanban_release_sagas SET state='promoting',owner_token=?,lease_expires=?,"
                "error_class=NULL,updated_at=? WHERE operation_key=?",
                (owner, now + _LEASE_SECONDS, now, op_key),
            )
        conn.execute("UPDATE kanban_release_gates SET gate_status='consuming' WHERE id=?", (gate["id"],))
        return (gate, op_key, owner, action), None


def _adapter_receipt(
    promotion_argv: Sequence[str], gate, operation_key: str, *, action: str = "promote",
) -> dict:
    if not promotion_argv or not all(isinstance(value, str) and value for value in promotion_argv):
        raise ValueError("promotion adapter argv is not configured")
    command = [*promotion_argv, action, "--task-id", gate["task_id"],
               "--release-id", gate["release_id"], "--manifest-sha256", gate["manifest_sha256"],
               "--operation-key", operation_key]
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300, check=True)
    receipt = json.loads(completed.stdout)
    if not isinstance(receipt, dict) or receipt.get("ok") is not True or receipt.get("release_id") != gate["release_id"]:
        raise ValueError("promotion adapter returned an invalid release receipt")
    for target in ("dev", "test", "production"):
        state = receipt.get(target)
        if not isinstance(state, dict) or state.get("result") not in _RESULT_WORDS:
            raise ValueError(f"promotion adapter omitted {target} state")
    if receipt["dev"].get("active_release_id") != gate["release_id"]:
        raise ValueError("promotion adapter did not read back the approved Dev release")
    return receipt


def _from_receipt(ok: bool, classification: str, gate, receipt: dict, detail: str = "") -> ApprovalResult:
    return ApprovalResult(
        ok=ok, classification=classification, release_id=gate["release_id"],
        dev_result=receipt["dev"]["result"], test_result=receipt["test"]["result"],
        production_result=receipt["production"]["result"],
        active_release_id=receipt["production"].get("active_release_id")
                          or receipt["test"].get("active_release_id")
                          or receipt["dev"].get("active_release_id"),
        previous_release_id=receipt.get("previous_release_id"),
        rollback_available=bool(receipt.get("rollback_available")), detail=detail,
    )


def process_approval(
    conn,
    *,
    text: str,
    context: ApprovalContext,
    promotion_argv: Sequence[str],
    now: int | None = None,
) -> ApprovalResult:
    """Claim, invoke and finalize one exact approval without holding a DB lock over I/O."""
    timestamp = int(time.time()) if now is None else int(now)
    claim, early = _claim(conn, text, context, timestamp)
    if early is not None:
        return early
    if claim is None:  # defensive: _claim returns either a claim or a result
        return _result("stale_approval")
    gate, operation_key, owner, action = claim
    try:
        receipt = _adapter_receipt(promotion_argv, gate, operation_key, action=action)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        with write_txn(conn):
            conn.execute(
                "UPDATE kanban_release_sagas SET error_class='adapter',lease_expires=?,"
                "updated_at=? WHERE operation_key=? AND owner_token=?",
                (timestamp, timestamp, operation_key, owner),
            )
        return ApprovalResult(
            ok=False,
            classification="adapter_failed",
            release_id=gate["release_id"],
            previous_release_id=gate["previous_release_id"],
            rollback_available=bool(gate["rollback_available"]),
            detail="Promotion outcome is unknown; retry resumes with the same operation key.",
        )

    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    with write_txn(conn):
        current = conn.execute(
            "SELECT g.*, t.status AS task_status, s.workflow_status AS current_workflow_status, "
            "s.active_dev_release_id AS current_active_dev_release_id, "
            "s.manifest_sha256 AS current_manifest_sha256 FROM kanban_release_gates g "
            "JOIN tasks t ON t.id=g.task_id LEFT JOIN kanban_release_state s ON s.task_id=g.task_id "
            "WHERE g.id=?",
            (gate["id"],),
        ).fetchone()
        saga = conn.execute("SELECT * FROM kanban_release_sagas WHERE operation_key=?", (operation_key,)).fetchone()
        if (current is None or current["gate_status"] != "consuming"
                or current["current_workflow_status"] != _REVIEW_STATUS
                or current["current_active_dev_release_id"] != gate["release_id"]
                or current["current_manifest_sha256"] != gate["manifest_sha256"]
                or current["task_status"] in {"done", "archived"} or saga is None
                or saga["owner_token"] != owner or saga["state"] != "promoting"):
            if saga is not None and saga["owner_token"] == owner:
                conn.execute("UPDATE kanban_release_sagas SET state='failed',error_class='stale_after_promotion',"
                             "adapter_receipt=?,lease_expires=NULL,updated_at=? WHERE operation_key=?",
                             (encoded, timestamp, operation_key))
            return _from_receipt(False, "stale_after_promotion", gate, receipt)
        conn.execute("UPDATE kanban_release_sagas SET state='succeeded',adapter_receipt=?,"
                     "lease_expires=NULL,updated_at=? WHERE operation_key=? AND owner_token=?",
                     (encoded, timestamp, operation_key, owner))
        conn.execute("UPDATE kanban_release_gates SET gate_status='approved' WHERE id=?", (gate["id"],))
        from hermes_cli.kanban_db import _append_event
        _append_event(conn, gate["task_id"], "release_approved", {
            "operation_key": operation_key, "release_id": gate["release_id"],
            "manifest_sha256": gate["manifest_sha256"], "result": receipt,
        })
    return _from_receipt(True, "success", gate, receipt)


def process_current_board_approval(
    *, text: str, context: ApprovalContext, promotion_argv: Sequence[str], now: int | None = None,
) -> ApprovalResult:
    """Thread-safe gateway entry point: open and close the current board connection."""
    from hermes_cli.kanban_db_connect import connect_closing
    with connect_closing() as conn:
        return process_approval(conn, text=text, context=context,
                                promotion_argv=promotion_argv, now=now)
