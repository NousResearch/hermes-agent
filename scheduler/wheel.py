"""Sparse hierarchical timing wheel over a *logical* clock.

Three levels of ``buckets`` slots (default 256). Placement is by remaining distance, so
memory scales with scheduled nodes, never with sessions × slots. A tick drains one L0 slot;
crossing a 256 / 65 536 boundary cascades the parent slot down. Insert and advance are O(1)
amortized, FIFO order inside a slot is preserved, and nothing is ever wall-clock based.

Design: website/docs/developer-guide/scheduled-tasks.md § Timing wheel.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable

from scheduler.tasks import TaskSpec


class WheelNode:
    """One scheduled task instance in a session's wheel."""

    __slots__ = ("cancelled", "due", "seq", "spec")

    def __init__(self, spec: TaskSpec, due: int, seq: int) -> None:
        self.spec = spec
        self.due = int(due)
        self.cancelled = False
        self.seq = seq

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"WheelNode({self.spec.name}@{self.due}{' cancelled' if self.cancelled else ''})"


class TimingWheel:
    """Logical-clock wheel; ``advance_to(n)`` fires every node due at or before ``n``."""

    def __init__(self, *, buckets: int = 256, levels: int = 3, now: int = 0) -> None:
        if buckets < 2 or levels < 1:
            raise ValueError("buckets >= 2 and levels >= 1 required")
        self.buckets = int(buckets)
        self.levels = int(levels)
        self.now = int(now)
        self._slots: list[dict[int, list[WheelNode]]] = [dict() for _ in range(self.levels)]
        self._due: dict[int, list[WheelNode]] = {}
        self._seq = itertools.count()

    # ── placement ─────────────────────────────────────────────────────────
    def _level_for(self, due: int) -> int:
        distance = due - self.now
        for level in range(self.levels - 1):
            if distance < self.buckets ** (level + 1):
                return level
        return self.levels - 1

    def _slot_index(self, due: int, level: int) -> int:
        return (due // (self.buckets ** level)) % self.buckets

    def insert(self, spec: TaskSpec, due: int) -> WheelNode:
        node = WheelNode(spec, due, next(self._seq))
        self._place(node)
        return node

    def _place(self, node: WheelNode) -> None:
        level = self._level_for(node.due)
        self._slots[level].setdefault(self._slot_index(node.due, level), []).append(node)
        self._due.setdefault(node.due, []).append(node)

    def cancel(self, node: WheelNode) -> None:
        """Cancel lazily: the slot is cleaned when it drains."""
        node.cancelled = True

    def reschedule(self, node: WheelNode, due: int) -> None:
        """Move a node that was just fired (its slot entry is already popped) to *due*."""
        node.due = int(due)
        self._place(node)

    def _drop_due_ref(self, node: WheelNode) -> None:
        bucket = self._due.get(node.due)
        if not bucket:
            return
        try:
            bucket.remove(node)
        except ValueError:
            return
        if not bucket:
            self._due.pop(node.due, None)

    # ── queries ───────────────────────────────────────────────────────────
    def due_nodes(self, seq: int) -> list[WheelNode]:
        """Live nodes due exactly at ``seq`` (pre-phase probe; never advances the clock)."""
        return [node for node in self._due.get(seq, ()) if not node.cancelled]

    def pending(self) -> int:
        return sum(len(bucket) for bucket in self._due.values())

    # ── advance ───────────────────────────────────────────────────────────
    def advance_to(self, target: int, on_due: Callable[[WheelNode], None]) -> None:
        """Advance to logical ``target``, calling ``on_due`` for each node that comes due.

        ``on_due`` owns rescheduling: the wheel never mutates a fired node.
        """
        if target <= self.now:
            return
        while self.now < target:
            self.now += 1
            if self.now % (self.buckets ** 2) == 0 and self.levels > 2:
                self._cascade(2, on_due)
            if self.now % self.buckets == 0:
                self._cascade(1, on_due)
            self._drain(0, self.now % self.buckets, on_due)

    def _cascade(self, level: int, on_due: Callable[[WheelNode], None] | None = None) -> None:
        if level >= self.levels:
            return
        slot = self._slot_index(self.now, level)
        nodes = self._slots[level].pop(slot, [])
        for node in nodes:
            self._drop_due_ref(node)
            if node.cancelled:
                continue
            if node.due <= self.now and on_due is not None:
                on_due(node)
            else:
                self._place(node)

    def _drain(self, level: int, slot: int, on_due: Callable[[WheelNode], None]) -> None:
        nodes = self._slots[level].pop(slot, [])
        for node in nodes:
            self._drop_due_ref(node)
            if node.cancelled:
                continue
            if node.due <= self.now:
                on_due(node)
            else:  # contract violation guard: never lose a node
                self._place(node)

    def live_nodes(self) -> Iterable[WheelNode]:
        for bucket in list(self._due.values()):
            yield from (node for node in bucket if not node.cancelled)
