"""The delivery-obligation verbs of the reference coordination repository.

Rows live in the scope store's ``delivery_obligations`` table keyed by obligation id. The sweeps take the owner probe as data: a row is processed only when its stamp is in ``dead_owners`` (or, for the runtime sweep, when the helper passed it in ``due``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Optional

from persistence.authority.coordination import (
    OBLIGATION_FLOOD_MARKER,
    OBLIGATION_MAX_ATTEMPTS,
    OBLIGATION_MAX_ROWS,
    OBLIGATION_RECONNECTED_MARKER,
    OBLIGATION_RETENTION_SECONDS,
    OBLIGATION_STALE_AFTER_SECONDS,
    ClaimedObligation,
    FailedObligationRow,
    OwnerStamp,
    RecordedOwner,
    RetryCandidate,
    RetryDue,
    flood_not_before,
    is_flood_error,
)

__all__ = ["ObligationLedger"]

_PRUNE_ORDER = {"delivered": 0, "abandoned": 1}


def _profile(adapter_profile: Optional[str]) -> str:
    return str(adapter_profile).strip() if adapter_profile else "default"


def _exhausted(row: dict[str, Any], now: float) -> bool:
    return bool(row["attempts"] >= OBLIGATION_MAX_ATTEMPTS or (now - row["created_at"]) > OBLIGATION_STALE_AFTER_SECONDS)


def _stamped(row: dict[str, Any], owner: OwnerStamp) -> bool:
    return bool(row["owner_pid"] == owner.pid and row["owner_started_at"] == owner.started_at)


def _claimed_row(row: dict[str, Any], profile: str, *, needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None) -> ClaimedObligation:
    """The claimed obligation built from the row as it stood before the claim, its attempt counted and its marker chosen."""
    marker = OBLIGATION_FLOOD_MARKER if flood else (OBLIGATION_RECONNECTED_MARKER if runtime else None)
    return ClaimedObligation(
        obligation_id=row["obligation_id"], session_key=row["session_key"], platform=row["platform"], chat_id=row["chat_id"],
        thread_id=row["thread_id"], content=row["content"], profile=profile, attempts=row["attempts"] + 1,
        needs_marker=needs_marker, marker=marker if needs_marker and marker else None, runtime_recovery=runtime,
        last_error=last_error or None)


class ObligationLedger:
    """The obligation verbs; the host class provides ``_table``."""

    def _table(self, name: str) -> dict[Any, Any]:
        raise NotImplementedError

    def _obligations(self) -> dict[str, dict[str, Any]]:
        return self._table("delivery_obligations")

    def record_delivery_obligation(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str, thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None, owner: OwnerStamp) -> None:
        now = time.time()
        rows = self._obligations()
        rows[obligation_id] = {
            "obligation_id": obligation_id, "session_key": session_key, "platform": platform, "chat_id": str(chat_id),
            "thread_id": str(thread_id) if thread_id else None, "content": content, "state": "pending", "attempts": 0,
            "created_at": now, "updated_at": now, "owner_pid": owner.pid, "owner_started_at": owner.started_at,
            "last_error": None, "adapter_profile": _profile(adapter_profile)}
        for key in [key for key, row in rows.items() if row["state"] in ("delivered", "abandoned") and row["updated_at"] < now - OBLIGATION_RETENTION_SECONDS]:
            del rows[key]
        if len(rows) > OBLIGATION_MAX_ROWS:
            ordered = sorted(rows.values(), key=lambda row: (_PRUNE_ORDER.get(row["state"], 2), row["updated_at"]))
            for row in ordered[:len(rows) - OBLIGATION_MAX_ROWS]:
                del rows[row["obligation_id"]]

    def record_crash_left_reply(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str, thread_id: Optional[str], content: str, since: float, adapter_profile: Optional[str] = None) -> None:
        now = time.time()
        rows = self._obligations()
        if obligation_id in rows or any(row["session_key"] == session_key and row["content"] == content and row["created_at"] >= since for row in rows.values()):
            return
        rows[obligation_id] = {
            "obligation_id": obligation_id, "session_key": session_key, "platform": platform, "chat_id": str(chat_id),
            "thread_id": str(thread_id) if thread_id else None, "content": content, "state": "attempting", "attempts": 0,
            "created_at": now, "updated_at": now, "owner_pid": None, "owner_started_at": None,
            "last_error": None, "adapter_profile": _profile(adapter_profile)}

    def _update_state(self, obligation_id: str, state: str, error: str = "") -> None:
        row = self._obligations().get(obligation_id)
        if row is not None:
            row.update(state=state, updated_at=time.time(), last_error=error[:500] if error else None)

    def mark_obligation_attempting(self, *, obligation_id: str) -> None:
        self._update_state(obligation_id, "attempting")

    def mark_obligation_delivered(self, *, obligation_id: str) -> None:
        self._update_state(obligation_id, "delivered")

    def mark_obligation_failed(self, *, obligation_id: str, error: str = '') -> None:
        self._update_state(obligation_id, "failed", error)

    def release_runtime_obligation_claim(self, *, obligation_id: str, error: str = '', owner: OwnerStamp) -> bool:
        if owner.started_at is None:
            return False
        row = self._obligations().get(obligation_id)
        if row is None or row["state"] != "attempting" or not _stamped(row, owner):
            return False
        row.update(state="failed", attempts=max(0, row["attempts"] - 1), updated_at=time.time(), last_error=error[:500] if error else None)
        return True

    def sweep_recoverable_obligations(self, *, now: Optional[float] = None, deliverable_platforms: Optional[frozenset[str]] = None, deliverable_targets: Optional[frozenset[tuple[str, str]]] = None, owner: OwnerStamp, dead_owners: frozenset[RecordedOwner]) -> tuple[ClaimedObligation, ...]:
        now = now if now is not None else time.time()
        claimed: list[ClaimedObligation] = []
        for row in [row for row in self._obligations().values() if row["state"] in ("pending", "attempting", "failed")]:
            if RecordedOwner(row["owner_pid"], row["owner_started_at"]) not in dead_owners:
                continue
            if _exhausted(row, now):
                row.update(state="abandoned", updated_at=now)
                continue
            if ((deliverable_platforms is not None and row["platform"] not in deliverable_platforms)
                    or (deliverable_targets is not None and (row["platform"], row["adapter_profile"]) not in deliverable_targets)):
                continue
            claimed.append(self._recover_row(row, now, owner))
        return tuple(claimed)

    @staticmethod
    def _recover_row(row: dict[str, Any], now: float, owner: OwnerStamp) -> ClaimedObligation:
        """One dead owner's row: adopted inside a flood wait, else claimed to ``attempting``. The claim's owner guard always holds here: the backend lock admits no writer between the read and this write."""
        flood_row = row["state"] == "failed" and is_flood_error(row["last_error"])
        profile = row["adapter_profile"] or "default"
        if flood_row and now < flood_not_before(row["updated_at"], row["last_error"]):
            row.update(owner_pid=owner.pid, owner_started_at=owner.started_at, adapter_profile=profile)
            return ClaimedObligation(
                obligation_id=row["obligation_id"], session_key=row["session_key"], platform=row["platform"], chat_id=row["chat_id"],
                thread_id=row["thread_id"], content=row["content"], profile=profile, attempts=row["attempts"], adopted=True,
                not_before=flood_not_before(row["updated_at"], row["last_error"]))
        claimed = _claimed_row(row, profile, needs_marker=row["state"] != "pending" or row["attempts"] > 0, flood=flood_row)
        row.update(owner_pid=owner.pid, owner_started_at=owner.started_at, attempts=row["attempts"] + 1, updated_at=now,
                   adapter_profile=profile, state="attempting", last_error=None if flood_row else row["last_error"])
        return claimed

    def sweep_failed_obligations_for_runtime(self, platform: str, *, now: Optional[float] = None, profile: Optional[str] = None, owner: OwnerStamp, due: Sequence[RetryDue]) -> tuple[ClaimedObligation, ...]:
        now = now if now is not None else time.time()
        if owner.started_at is None:
            return ()
        expected_profile = "default" if not profile or profile == "default" else str(profile)
        rows = self._obligations()
        claimed: list[ClaimedObligation] = []
        for item in due:
            row = rows.get(item.obligation_id)
            if (row is None or row["state"] != "failed" or row["platform"] != platform or row["adapter_profile"] != expected_profile
                    or not _stamped(row, owner) or row["updated_at"] != item.updated_at or row["last_error"] != item.last_error
                    or row["attempts"] != item.attempts):
                continue
            if _exhausted(row, now):
                row.update(state="abandoned", updated_at=now)
                continue
            if now < item.not_before:
                continue
            last_error = row["last_error"]
            claimed.append(_claimed_row(row, row["adapter_profile"], needs_marker=True, runtime=True, flood=is_flood_error(last_error), last_error=last_error))
            row.update(state="attempting", attempts=row["attempts"] + 1, updated_at=now, last_error=None)
        return tuple(claimed)

    # ── reads ────────────────────────────────────────────────────────────

    def list_owned_failed_obligations(self, *, owner: OwnerStamp) -> tuple[FailedObligationRow, ...]:
        if owner.started_at is None:
            return ()
        return tuple(FailedObligationRow(platform=row["platform"], adapter_profile=row["adapter_profile"], updated_at=row["updated_at"],
                                         last_error=row["last_error"], attempts=row["attempts"], created_at=row["created_at"])
                     for row in self._obligations().values() if row["state"] == "failed" and _stamped(row, owner))

    def list_obligation_owners(self) -> frozenset[RecordedOwner]:
        return frozenset(RecordedOwner(row["owner_pid"], row["owner_started_at"]) for row in self._obligations().values()
                         if row["state"] in ("pending", "attempting", "failed"))

    def list_runtime_retry_candidates(self, platform: str, *, profile: Optional[str] = None, owner: OwnerStamp) -> tuple[RetryCandidate, ...]:
        if owner.started_at is None:
            return ()
        expected_profile = "default" if not profile or profile == "default" else str(profile)
        return tuple(RetryCandidate(obligation_id=row["obligation_id"], updated_at=row["updated_at"], last_error=row["last_error"], attempts=row["attempts"])
                     for row in self._obligations().values()
                     if row["state"] == "failed" and row["platform"] == platform and row["adapter_profile"] == expected_profile and _stamped(row, owner))
