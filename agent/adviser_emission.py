"""Admission policy for adviser ``advise`` calls.

Port of oh-my-pi's ``advisor/emission-guard.ts``: everything the adviser model may
emit goes through one gate that (in order) drops empty notes and content-free
self-talk, applies rank-aware session dedupe, and charges a per-update budget for
non-blocker notes. A ``blocker`` is exempt from the budget so it can always
interrupt. Dedupe is rank-aware: re-raising the same text at a strictly higher
severity is a real escalation and is admitted; equal or lower repeats stay
suppressed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

SEVERITIES = ("nit", "concern", "blocker")
SEVERITY_RANK = {"nit": 1, "concern": 2, "blocker": 3}

#: Hard cap on non-blocker notes admitted per update (config clamps into [1, this]).
MAX_BUDGET_PER_UPDATE = 32
DEFAULT_BUDGET_PER_UPDATE = 4
DEFAULT_HISTORY_CAPACITY = 4096

_SUPPRESSED_PHRASES = frozenset({
    "stop", "stop here", "stop now", "halt", "abort",
    "done", "task done", "task complete", "complete", "finished", "ok", "okay", "ok done",
    "no issue", "no issues", "no issue continue", "no concerns", "no concern",
    "nothing to add", "nothing to flag", "nothing to report", "no notes",
    "no further input", "no further input needed", "no further input required",
    "no further watcher input", "no further watcher input needed",
    "no further advice", "no further advice needed",
    "lgtm", "looks good", "all good", "agent is on track", "agent on track", "on track",
    "continue", "carry on",
})

_NORM_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def normalize_note(note: str) -> str:
    """Case-insensitive, punctuation-folded key (``"Stop."`` → ``"stop"``)."""
    text = unicodedata.normalize("NFKC", str(note or "")).lower()
    return _NORM_STRIP.sub(" ", text).strip()


def severity_rank(severity: Optional[str]) -> int:
    return SEVERITY_RANK.get(str(severity or "nit"), 1)


def meets_severity(severity: Optional[str], minimum: str) -> bool:
    return severity_rank(severity) >= severity_rank(minimum)


def is_interrupting(severity: Optional[str]) -> bool:
    return severity in ("concern", "blocker")


@dataclass
class Admission:
    accepted: bool
    reason: str = ""
    displaced_key: Optional[str] = None


class EmissionGuard:
    """Single admission authority: noise/empty filter, rank-aware dedupe, per-update budget."""

    def __init__(self, *, capacity: int = DEFAULT_HISTORY_CAPACITY,
                 budget_per_update: int = DEFAULT_BUDGET_PER_UPDATE) -> None:
        self._capacity = max(1, int(capacity))
        self._budget = min(MAX_BUDGET_PER_UPDATE, max(1, int(budget_per_update)))
        self._seen: dict[str, int] = {}
        self._seen_order: list[str] = []
        self._slots: list[dict] = []

    @property
    def budget_per_update(self) -> int:
        return self._budget

    def reset(self) -> None:
        self._seen.clear()
        self._seen_order.clear()
        self._slots = []

    def begin_update(self) -> None:
        self._slots = []

    def _record_rank(self, key: str, rank: int) -> None:
        is_new = key not in self._seen
        self._seen[key] = rank
        if not is_new:
            return
        self._seen_order.append(key)
        if len(self._seen_order) > self._capacity:
            stale = self._seen_order.pop(0)
            self._seen.pop(stale, None)

    def escalate_pending(self, note: str, rank: int) -> None:
        key = normalize_note(note)
        if not key or rank <= self._seen.get(key, 0):
            return
        self._record_rank(key, rank)
        for slot in self._slots:
            if slot["key"] == key and slot["rank"] < rank:
                slot["rank"] = rank

    def mark_routed(self, note: str) -> None:
        key = normalize_note(note)
        for slot in self._slots:
            if slot["key"] == key:
                slot["pending"] = False

    def admit(self, note: str, *, rank: int, pending: bool) -> Admission:
        key = normalize_note(note)
        if not key:
            return Admission(False, "empty")
        if key in _SUPPRESSED_PHRASES:
            return Admission(False, "noise")
        seen_rank = self._seen.get(key, 0)
        if rank <= seen_rank:
            return Admission(False, "duplicate")
        displaced_key: Optional[str] = None
        own_slot = next((slot for slot in self._slots if slot["key"] == key), None)
        if rank >= 3:
            # Blockers are unlimited; a blocker escalation of a still-pending note
            # releases its slot because the note now routes live.
            if own_slot is not None and own_slot["pending"]:
                self._slots.remove(own_slot)
        elif own_slot is not None:
            own_slot["rank"] = rank
        elif len(self._slots) < self._budget:
            self._slots.append({"key": key, "rank": rank, "pending": pending})
        else:
            pending_slots = [slot for slot in self._slots if slot["pending"]]
            if pending_slots:
                lowest = min(pending_slots, key=lambda slot: slot["rank"])
                if rank > lowest["rank"]:
                    displaced_key = lowest["key"]
                    self._slots[self._slots.index(lowest)] = {"key": key, "rank": rank, "pending": pending}
                else:
                    return Admission(False, "rate-limit")
            else:
                return Admission(False, "rate-limit")
        self._record_rank(key, rank)
        return Admission(True, "", displaced_key)


__all__ = [
    "DEFAULT_BUDGET_PER_UPDATE",
    "DEFAULT_HISTORY_CAPACITY",
    "MAX_BUDGET_PER_UPDATE",
    "SEVERITIES",
    "SEVERITY_RANK",
    "Admission",
    "EmissionGuard",
    "is_interrupting",
    "meets_severity",
    "normalize_note",
    "severity_rank",
]
