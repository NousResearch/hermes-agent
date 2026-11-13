"""Kanban board workflow: the one definition of board columns and manual moves.

Groundwork for user-defined columns and workflow per board (issue #54818). Target
model: ``tasks.status`` holds a board-defined column key.

Phase 0 (this module) holds only what is consumed or checked today: the column list
(order, label, CLI icon, drag target) and the manual move matrix. The scattered status
copies (``VALID_STATUSES``, ``BOARD_COLUMNS``, the agent-tool enum, CLI icons) derive
from it, and ``tests/plugins/test_kanban_workflow_matrix.py`` pins ``manual`` to the live
dashboard API in both directions.

Kernel semantics are deliberately NOT modelled here — neither what a column means to
the dispatcher (claimable, terminal, parent gate) nor kernel-driven transitions (~20
conditional status writes: block routing, crash/reclaim return, parent reopen, import).
Both are designed together with the kernel switch, so this module never publishes a
partial contract.

Pure data + stdlib only: ``kanban_db`` imports this module, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping, Optional

# ``archived`` is outside every workflow: a filter toggle, never a board column.
ARCHIVED = "archived"


@dataclass(frozen=True)
class Column:
    """One board column. ``key`` is stable (stored in ``tasks.status``); ``label`` is display-only."""

    key: str
    label: str
    icon: str = "?"  # single-glyph CLI marker
    # Offered as a drag/menu target in board UIs. False for columns a card normally
    # reaches only through a verb with extra input (reviewer, wake time) or the kernel.
    drag_target: bool = True


@dataclass(frozen=True)
class Workflow:
    """Columns (board order) and the manual move allow-list."""

    columns: tuple[Column, ...]
    # Manual moves (drag, PATCH status) a human may request: ``src -> {dst}``. Archiving
    # (dst) is always allowed and not listed; ``archived`` may appear as a source (restore). Allowed != guaranteed: verbs still apply their own
    # gates (parents open, completion evidence).
    manual: Mapping[str, frozenset[str]]

    def __iter__(self) -> Iterator[Column]:
        return iter(self.columns)

    def keys(self) -> tuple[str, ...]:
        """Column keys in board order (never includes ``archived``)."""
        return tuple(c.key for c in self.columns)

    def column(self, key: str) -> Optional[Column]:
        return next((c for c in self.columns if c.key == key), None)

    def _require(self, key: str) -> Column:
        col = self.column(key)
        if col is None:
            raise ValueError(f"unknown kanban workflow column {key!r}")
        return col

    def can_move(self, src: str, dst: str) -> bool:
        """True when a human may request ``src -> dst``. Archiving is always allowed;
        ``archived`` is a valid source (restore). ``src == dst`` is not a move."""
        if src != ARCHIVED:
            self._require(src)
        if src == dst:
            return False
        if dst == ARCHIVED:
            return True
        self._require(dst)
        return dst in self.manual.get(src, frozenset())

    def to_dict(self) -> dict:
        """JSON shape served by the dashboard's ``GET /workflow``. ``manual`` keys are the
        column keys plus ``archived`` (restore source), which is never a column."""
        return {
            "columns": [
                {"key": c.key, "label": c.label, "icon": c.icon, "drag_target": c.drag_target}
                for c in self.columns
            ],
            "manual": {src: sorted(dsts) for src, dsts in self.manual.items()},
            "archived": ARCHIVED,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent workflow (duplicate or unknown keys)."""
        keys = self.keys()
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate column keys: {keys}")
        if ARCHIVED in keys:
            raise ValueError(f"{ARCHIVED!r} is reserved and cannot be a column")
        known = set(keys)
        for src, dsts in self.manual.items():
            if src not in known | {ARCHIVED} or not set(dsts) <= known:
                raise ValueError(f"manual moves from {src!r} reference unknown columns")


def _frozen_manual(table: Mapping[str, tuple[str, ...]]) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({src: frozenset(dsts) for src, dsts in table.items()})


# Today's board, written down. Column order = dashboard order. ``drag_target=False``
# mirrors the desktop's LOCKED_COLUMNS (review needs a reviewer, scheduled a wake time,
# running is kernel-only). The manual table is the measured PATCH /tasks/{id} matrix on
# a parentless task; ``test_kanban_workflow_matrix.py`` re-measures it against the live API.
DEFAULT_WORKFLOW = Workflow(
    columns=(
        Column("triage", "Triage", "◇"),
        Column("todo", "Todo", "◻"),
        Column("scheduled", "Scheduled", "⏱", drag_target=False),
        Column("ready", "Ready", "▶"),
        Column("running", "Running", "●", drag_target=False),
        Column("blocked", "Blocked", "⊘"),
        Column("review", "Review", "◎", drag_target=False),
        Column("done", "Done", "✓"),
    ),
    manual=_frozen_manual({
        "triage": ("todo", "ready"),
        "todo": ("triage", "scheduled", "ready"),
        "scheduled": ("triage", "todo", "ready"),
        "ready": ("triage", "todo", "scheduled", "blocked", "review", "done"),
        "running": ("triage", "todo", "scheduled", "ready", "blocked", "review", "done"),
        "blocked": ("triage", "todo", "scheduled", "ready", "done"),
        # Known quirk kept for zero behavior change: review -> todo lands in ``ready``
        # (reopen_review_task ignores the requested target). Phase 1 fixes it.
        "review": ("triage", "todo", "ready", "done"),
        "done": ("triage", "todo", "ready"),
        ARCHIVED: ("triage", "todo", "ready"),  # restore from the archive filter
    }),
)
DEFAULT_WORKFLOW.validate()

# Every value ``tasks.status`` can hold under the default workflow.
DEFAULT_STATUSES = frozenset(DEFAULT_WORKFLOW.keys()) | {ARCHIVED}
