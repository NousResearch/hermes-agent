"""Durable, restart-safe, owner-private store for Sol food proposals.

Storage is a single JSON document under the plugin's private state
directory (directory mode 0700, file mode 0600), written atomically
(temp file + ``os.replace``) so a crash can never leave a torn state.
Consumption of a callback token and deduplication of its Telegram update
are recorded in the SAME atomic write, before any side effect runs —
which is what makes confirm-effects exactly-once across restarts.

Retention:

- Proposal content (candidates, labels) is deleted at terminal state or
  expiry — it never outlives the workflow.
- Only value-free linkage survives, for 48 hours: consumed update ids,
  consumed action records, and the opaque receipt reference. No food
  text, labels, items, or callback payloads are retained or logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from plugins.sol_food.limits import (
    FOOD_CACHE_DIR_MODE,
    FOOD_CACHE_FILE_MODE,
    FOOD_DEDUP_RETENTION_SECONDS,
)
from plugins.sol_food.proposal import (
    ACTION_CONFIRM,
    Candidate,
    FoodProposal,
    ProposalError,
    ProposalState,
    candidates_hash,
    make_proposal,
    validate_candidates,
)

__all__ = ["CallbackOutcome", "FoodProposalStore"]

# Stable value-free reason codes.
REASON_ACTIVE_PROPOSAL_EXISTS = "food_store_active_proposal_exists"
REASON_UNKNOWN_TOKEN = "food_store_unknown_token"
REASON_FOREIGN_ORIGIN = "food_store_foreign_origin"
REASON_STALE_VERSION = "food_store_stale_version"
REASON_EXPIRED = "food_store_expired"
REASON_ALREADY_RESOLVED = "food_store_already_resolved"
REASON_NOT_PENDING = "food_store_not_pending"
REASON_BAD_PRESENTATION = "food_store_bad_presentation"


class CallbackOutcome:
    """Result of resolving one callback update against the store."""

    __slots__ = ("kind", "reason_code", "action", "proposal_id", "receipt_ref")

    KIND_ACTION = "action"
    KIND_REPLAY = "replay"
    KIND_DENIED = "denied"

    def __init__(
        self,
        kind: str,
        *,
        reason_code: Optional[str] = None,
        action: Optional[str] = None,
        proposal_id: Optional[str] = None,
        receipt_ref: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.reason_code = reason_code
        self.action = action
        self.proposal_id = proposal_id
        self.receipt_ref = receipt_ref

    @classmethod
    def denied(cls, reason_code: str) -> "CallbackOutcome":
        return cls(cls.KIND_DENIED, reason_code=reason_code)

    @classmethod
    def replay(cls, receipt_ref: Optional[str]) -> "CallbackOutcome":
        return cls(cls.KIND_REPLAY, receipt_ref=receipt_ref)

    @classmethod
    def act(cls, action: str, proposal_id: str) -> "CallbackOutcome":
        return cls(cls.KIND_ACTION, action=action, proposal_id=proposal_id)


class FoodProposalStore:
    """Async-safe durable proposal store. All mutation under one lock."""

    def __init__(
        self,
        state_dir: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._dir = Path(state_dir)
        self._file = self._dir / "sol-food-proposals.json"
        self._clock = clock
        self._lock = asyncio.Lock()
        self._proposals: Dict[str, FoodProposal] = {}
        # update_id(str) -> {"ts": float, "receipt_ref": str|None,
        #                    "proposal_id": str, "action": str}
        self._consumed_updates: Dict[str, Dict[str, Any]] = {}
        self._ensure_dirs()
        self._load()

    # ── persistence ─────────────────────────────────────────────────────
    def _ensure_dirs(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, FOOD_CACHE_DIR_MODE)

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A torn/corrupt file fails closed to empty state: proposals
            # are re-creatable transient state; dedup loss only risks a
            # denied replay (safe direction: DENY, never double-commit)…
            # except that losing consumed-linkage could re-enable a
            # token. To stay fail-closed we keep the corrupt file for
            # forensics and start with NO proposals (all tokens dead).
            backup = self._file.with_suffix(".corrupt")
            try:
                os.replace(self._file, backup)
            except OSError:
                pass
            return
        for entry in raw.get("proposals", []):
            try:
                proposal = FoodProposal.from_json(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self._proposals[proposal.proposal_id] = proposal
        consumed = raw.get("consumed_updates", {})
        if isinstance(consumed, dict):
            for key, value in consumed.items():
                if isinstance(value, dict):
                    self._consumed_updates[str(key)] = dict(value)

    def _persist(self) -> None:
        payload = {
            "proposals": [p.to_json() for p in self._proposals.values()],
            "consumed_updates": self._consumed_updates,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        tmp = self._file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FOOD_CACHE_FILE_MODE)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, self._file)
        os.chmod(self._file, FOOD_CACHE_FILE_MODE)

    # ── internal upkeep (call under lock) ───────────────────────────────
    def _sweep_locked(self, now: float) -> bool:
        changed = False
        for proposal in list(self._proposals.values()):
            if proposal.awaiting_commit:
                # A durably consumed Confirm owns this proposal until its
                # verified receipt lands: it must never expire out from
                # under the frozen envelope / reconcile().
                continue
            if proposal.state is ProposalState.PENDING and proposal.expired(now):
                proposal.state = ProposalState.EXPIRED
                proposal.invalidate_all_tokens()
                proposal.scrub_content()
                changed = True
        # Drop terminal proposals whose dedup window has fully lapsed.
        horizon = now - FOOD_DEDUP_RETENTION_SECONDS
        for pid, proposal in list(self._proposals.items()):
            if proposal.state.terminal and proposal.issued_at < horizon:
                del self._proposals[pid]
                changed = True
        for key, record in list(self._consumed_updates.items()):
            if float(record.get("ts", 0)) < horizon:
                del self._consumed_updates[key]
                changed = True
        return changed

    def _active_for_origin_locked(
        self, owner_chat_id: str, thread_id: int, now: float
    ) -> Optional[FoodProposal]:
        for proposal in self._proposals.values():
            if (
                proposal.owner_chat_id == owner_chat_id
                and proposal.thread_id == thread_id
                and proposal.state is ProposalState.PENDING
                and not proposal.expired(now)
            ):
                return proposal
        return None

    # ── public API ──────────────────────────────────────────────────────
    async def has_active_proposal(self, owner_chat_id: str, thread_id: int) -> bool:
        async with self._lock:
            now = self._clock()
            if self._sweep_locked(now):
                self._persist()
            return self._active_for_origin_locked(owner_chat_id, thread_id, now) is not None

    async def create(
        self,
        *,
        bot_id: str,
        owner_chat_id: str,
        thread_id: int,
        origin_update_id: int,
        origin_message_id: int,
        candidates: Sequence[Candidate],
    ) -> tuple[FoodProposal, Dict[str, str]]:
        """Create + persist a fresh proposal; returns (proposal, tokens).

        Refuses while ANY non-terminal proposal exists for the same
        owner+thread (origins are never merged), and refuses a second
        proposal for an already-seen origin update.
        """
        async with self._lock:
            now = self._clock()
            self._sweep_locked(now)
            if self._active_for_origin_locked(owner_chat_id, thread_id, now) is not None:
                raise ProposalError(REASON_ACTIVE_PROPOSAL_EXISTS)
            if str(origin_update_id) in self._consumed_updates:
                raise ProposalError(REASON_ALREADY_RESOLVED)
            proposal = make_proposal(
                bot_id=bot_id,
                owner_chat_id=owner_chat_id,
                thread_id=thread_id,
                origin_update_id=origin_update_id,
                origin_message_id=origin_message_id,
                candidates=candidates,
                now=now,
            )
            tokens = proposal.mint_action_tokens()
            self._proposals[proposal.proposal_id] = proposal
            self._persist()
            return proposal, tokens

    async def record_presentation(
        self, proposal_id: str, presentation_message_id: int
    ) -> None:
        async with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise ProposalError(REASON_UNKNOWN_TOKEN)
            proposal.presentation_message_id = int(presentation_message_id)
            self._persist()

    async def edit_new_version(
        self, proposal_id: str, candidates: Sequence[Candidate]
    ) -> tuple[FoodProposal, Dict[str, str]]:
        """Create a new immutable candidate version inside the same TTL.

        Invalidates every earlier-version token immediately.
        """
        async with self._lock:
            now = self._clock()
            self._sweep_locked(now)
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise ProposalError(REASON_UNKNOWN_TOKEN)
            if proposal.state is not ProposalState.PENDING:
                raise ProposalError(REASON_NOT_PENDING)
            if proposal.expired(now):
                raise ProposalError(REASON_EXPIRED)
            validate_candidates(candidates)
            proposal.invalidate_all_tokens()
            proposal.version += 1
            proposal.candidates = list(candidates)
            proposal.version_hash = candidates_hash(candidates)
            # NOTE: expires_at intentionally unchanged — same lifetime.
            tokens = proposal.mint_action_tokens()
            self._persist()
            return proposal, tokens

    def _validate_callback_locked(
        self,
        *,
        token: str,
        update_id: int,
        bot_id: str,
        chat_id: str,
        thread_id: Optional[int],
        callback_message_id: Optional[int],
        now: float,
        mutate_expiry: bool,
    ):
        """Shared validation. Returns either a terminal CallbackOutcome
        (replay/denied) or the ``(proposal, record, action)`` triple that a
        consumer may act on. Mutates nothing except (optionally) the
        expiry transition when ``mutate_expiry`` is set."""
        update_key = str(update_id)
        prior = self._consumed_updates.get(update_key)
        if prior is not None:
            return CallbackOutcome.replay(prior.get("receipt_ref"))

        record = None
        proposal = None
        for candidate_proposal in self._proposals.values():
            if token in candidate_proposal.tokens:
                proposal = candidate_proposal
                record = candidate_proposal.tokens[token]
                break
        if proposal is None or record is None:
            return CallbackOutcome.denied(REASON_UNKNOWN_TOKEN)

        # Origin binding: owner chat, thread, bot, presentation message.
        if (
            str(chat_id) != proposal.owner_chat_id
            or str(bot_id) != proposal.bot_id
            or thread_id is None
            or int(thread_id) != proposal.thread_id
        ):
            return CallbackOutcome.denied(REASON_FOREIGN_ORIGIN)
        if (
            proposal.presentation_message_id is not None
            and callback_message_id is not None
            and int(callback_message_id) != proposal.presentation_message_id
        ):
            return CallbackOutcome.denied(REASON_BAD_PRESENTATION)

        if proposal.state is not ProposalState.PENDING:
            if proposal.state is ProposalState.EXPIRED:
                return CallbackOutcome.denied(REASON_EXPIRED)
            return CallbackOutcome.denied(REASON_ALREADY_RESOLVED)
        if proposal.expired(now):
            if mutate_expiry:
                proposal.state = ProposalState.EXPIRED
                proposal.invalidate_all_tokens()
                proposal.scrub_content()
                self._persist()
            return CallbackOutcome.denied(REASON_EXPIRED)
        if int(record.get("version", -1)) != proposal.version:
            return CallbackOutcome.denied(REASON_STALE_VERSION)
        if record.get("consumed"):
            return CallbackOutcome.denied(REASON_ALREADY_RESOLVED)
        return proposal, record, str(record.get("action"))

    async def peek_callback(
        self,
        *,
        token: str,
        update_id: int,
        bot_id: str,
        chat_id: str,
        thread_id: Optional[int],
        callback_message_id: Optional[int],
    ) -> CallbackOutcome:
        """Validate WITHOUT consuming. Same outcome kinds as
        ``resolve_callback``, but nothing is mutated or persisted.

        Callers use this to prepare durable side-effect state (e.g. freeze
        a commit envelope) BEFORE the atomic consume, so a crash between
        the two can always be reconciled. A peek result is advisory: the
        authoritative decision is the subsequent ``resolve_callback``.
        """
        async with self._lock:
            now = self._clock()
            if self._sweep_locked(now):
                self._persist()
            result = self._validate_callback_locked(
                token=token,
                update_id=update_id,
                bot_id=bot_id,
                chat_id=chat_id,
                thread_id=thread_id,
                callback_message_id=callback_message_id,
                now=now,
                mutate_expiry=False,
            )
            if isinstance(result, CallbackOutcome):
                return result
            proposal, _record, action = result
            return CallbackOutcome.act(action, proposal.proposal_id)

    async def resolve_callback(
        self,
        *,
        token: str,
        update_id: int,
        bot_id: str,
        chat_id: str,
        thread_id: Optional[int],
        callback_message_id: Optional[int],
    ) -> CallbackOutcome:
        """Validate + atomically consume one callback action.

        Ordering guarantees:
        - exact-duplicate update replay is detected FIRST and returns the
          original receipt reference with no state change;
        - all origin/version/expiry validation happens BEFORE consumption;
        - consumption + update-dedup (and, for Confirm, the durable
          ``awaiting_commit`` flag that pins the proposal until its
          verified receipt lands) are persisted in ONE atomic write
          BEFORE the outcome is returned (so the caller's side effect can
          never run twice, even across a crash+restart).
        """
        async with self._lock:
            now = self._clock()
            if self._sweep_locked(now):
                self._persist()
            result = self._validate_callback_locked(
                token=token,
                update_id=update_id,
                bot_id=bot_id,
                chat_id=chat_id,
                thread_id=thread_id,
                callback_message_id=callback_message_id,
                now=now,
                mutate_expiry=True,
            )
            if isinstance(result, CallbackOutcome):
                return result
            proposal, record, action = result

            # Atomic consume + dedup, durably, before any side effect.
            record["consumed"] = True
            if action == ACTION_CONFIRM:
                # Confirm freezes the whole proposal against further
                # actions; other tokens die with it. Only a COMMITTABLE
                # confirm (exactly one candidate) pins the proposal
                # (no expiry) until the verified receipt lands — a
                # multi-candidate confirm is answered with "select
                # first" and never commits.
                proposal.invalidate_all_tokens()
                if len(proposal.candidates) == 1:
                    proposal.awaiting_commit = True
            self._consumed_updates[str(update_id)] = {
                "ts": now,
                "receipt_ref": None,
                "proposal_id": proposal.proposal_id,
                "action": action,
            }
            self._persist()
            return CallbackOutcome.act(action, proposal.proposal_id)

    async def record_receipt(
        self, proposal_id: str, update_id: int, receipt_ref: str
    ) -> None:
        """Bind the (value-free) Health receipt reference to the consumed
        confirm action and mark the proposal terminal-confirmed."""
        async with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise ProposalError(REASON_UNKNOWN_TOKEN)
            proposal.state = ProposalState.CONFIRMED
            proposal.receipt_ref = str(receipt_ref)
            proposal.awaiting_commit = False
            proposal.scrub_content()
            entry = self._consumed_updates.get(str(update_id))
            if entry is not None:
                entry["receipt_ref"] = str(receipt_ref)
            self._persist()

    async def mark_terminal(
        self, proposal_id: str, state: ProposalState
    ) -> None:
        if not state.terminal:
            raise ProposalError(REASON_NOT_PENDING)
        async with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise ProposalError(REASON_UNKNOWN_TOKEN)
            proposal.state = state
            proposal.awaiting_commit = False
            proposal.invalidate_all_tokens()
            proposal.scrub_content()
            self._persist()

    async def get(self, proposal_id: str) -> Optional[FoodProposal]:
        async with self._lock:
            return self._proposals.get(proposal_id)

    async def sweep(self) -> None:
        async with self._lock:
            if self._sweep_locked(self._clock()):
                self._persist()
