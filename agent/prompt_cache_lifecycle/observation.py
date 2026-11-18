"""Bounded process-local warming observations; no agents, requests, clients or credentials.

Writers canonicalize homes at lifecycle boundaries. Readers use the manager's already canonical
scope key and take only this lock (never a controller lock). Losing a record is unavailable, not
permission to claim that another process owns a budget.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Literal

Reason = Literal["scheduled", "refreshing", "expiring", "unsupported", "disabled", "unknown", "unavailable"]
_MANAGED = frozenset({"scheduled", "refreshing", "expiring"})


@dataclass(frozen=True)
class Observation:
    owner: object
    reason: Reason = "unknown"
    next_warm_at: float | None = None
    route: str = ""
    turn: int | None = None
    ended: bool = False


class WarmObservations:
    """One-way lock ordering: controller -> observations. Reads never touch a controller."""

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], Observation] = OrderedDict()

    def claim(self, key: tuple[str, str], owner: object) -> Observation:
        with self._lock:
            prior = self._entries.get(key)
            item = Observation(owner) if prior is None else replace(
                prior, owner=owner, reason="expiring" if prior.reason in _MANAGED else prior.reason,
                next_warm_at=None,
            )
            self._entries[key] = item
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
            return item

    def update(self, key: tuple[str, str], item: Observation) -> None:
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current.owner is item.owner:
                self._entries[key] = item
                self._entries.move_to_end(key)

    def clear(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def carry(self, old: tuple[str, str], new: tuple[str, str]) -> None:
        with self._lock:
            prior = self._entries.get(old)
            if prior is None or prior.reason not in _MANAGED:
                return
            if old != new and new in self._entries:
                return  # the child already made a current decision; an ancestor cannot replace it
            owner = prior.owner if old == new else object()
            self._entries[new] = replace(prior, owner=owner, reason="expiring", next_warm_at=None)
            self._entries.move_to_end(new)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def snapshot(self, home_key: str, session_id: str) -> dict:
        with self._lock:
            item = self._entries.get((home_key, session_id))
        reason: Reason = "unavailable" if item is None else item.reason
        due = None if item is None else item.next_warm_at
        if reason == "scheduled" and (due is None or not math.isfinite(due) or due <= time.time()):
            reason, due = "expiring", None
        return {"managed": reason in _MANAGED,
                "next_warm_at": float(due) if reason == "scheduled" and due is not None else None,
                "reason": reason}


observations = WarmObservations()
