"""Kanban board workflow: the one definition of columns, traits and transitions.

Groundwork for user-defined columns and workflow per board (issue #54818). Target model —
``tasks.status`` holds a board-defined column key; kernel behavior attaches to
column *traits*, and kernel events resolve to a column through per-column *edges*
with a board-wide default, so an automatic transition never has to guess.

Phase 0 (this module): ``DEFAULT_WORKFLOW`` writes today's hardcoded behavior down
as data, and the scattered status copies (``VALID_STATUSES``, ``BOARD_COLUMNS``, the
agent-tool enum, CLI icons) derive from it. The kernel does not consult traits or
edges yet; ``tests/hermes_cli/test_kanban_workflow.py`` pins them to the kernel's real
behavior so the later phases that switch the kernel over cannot drift silently.

Pure data + stdlib only: ``kanban_db`` imports this module, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator, Mapping, Optional

# ``archived`` is outside every workflow: a filter toggle, never a board column.
ARCHIVED = "archived"

# --- Traits: kernel behavior a column opts into ------------------------------------------
DECOMPOSE = "decompose"                    # specifier/decomposer input (today: triage)
WAIT_PARENTS = "wait_parents"              # parked until every parent is satisfied (todo)
HOLD_TIME = "hold_time"                    # time gate, not dispatchable (scheduled)
DISPATCH_IMPLEMENT = "dispatch_implement"  # dispatcher claims for the assignee (ready)
CLAIMED = "claimed"                        # a worker holds the card (running) — Phase 1 turns
                                           # this into a claim fact instead of a column
HOLD_HUMAN = "hold_human"                  # waiting on a person / external blocker (blocked)
DISPATCH_REVIEW = "dispatch_review"        # dispatcher claims for the reviewer (review)
TERMINAL = "terminal"                      # satisfies a child's parent gate (done)

TRAITS = frozenset({
    DECOMPOSE, WAIT_PARENTS, HOLD_TIME, DISPATCH_IMPLEMENT, CLAIMED, HOLD_HUMAN, DISPATCH_REVIEW, TERMINAL,
})

# --- Kernel events that move a card without a human choosing the target -----------------
EV_CLAIM = "claim"                # dispatcher claims the card
EV_COMPLETE = "complete"          # worker/human completes
EV_BLOCK = "block"                # worker/human blocks
EV_SCHEDULE = "schedule"          # parked on a time gate
EV_REVIEW = "review"              # implementer hands off for review
EV_CHANGES = "changes"            # reviewer requests changes
EV_PARENTS_DONE = "parents_done"  # last open parent is satisfied

EVENTS = frozenset({EV_CLAIM, EV_COMPLETE, EV_BLOCK, EV_SCHEDULE, EV_REVIEW, EV_CHANGES, EV_PARENTS_DONE})


@dataclass(frozen=True)
class Column:
    """One board column. ``key`` is stable (stored in ``tasks.status``); ``label`` is display-only."""

    key: str
    label: str
    icon: str = "?"                  # single-glyph CLI marker
    traits: frozenset = frozenset()
    # Offered as a drag/menu target in board UIs. False for columns a card normally
    # reaches only through a verb with extra input (reviewer, wake time) or the kernel.
    drag_target: bool = True
    # Per-column event edges; an event missing here falls back to ``Workflow.defaults``.
    on: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Workflow:
    """Columns (board order), default event edges, and the manual move allow-list."""

    columns: tuple
    defaults: Mapping[str, str]
    # Manual moves (drag, PATCH status) a human may request: ``src -> {dst}``. Archive is
    # always allowed and not listed. Allowed != guaranteed: verbs still apply their own
    # gates (parents open, completion evidence).
    manual: Mapping[str, frozenset]

    def __iter__(self) -> Iterator[Column]:
        return iter(self.columns)

    def keys(self) -> tuple:
        """Column keys in board order (never includes ``archived``)."""
        return tuple(c.key for c in self.columns)

    def column(self, key: str) -> Optional[Column]:
        return next((c for c in self.columns if c.key == key), None)

    def keys_with(self, trait: str) -> tuple:
        """Keys of every column carrying ``trait``, in board order."""
        if trait not in TRAITS:
            raise ValueError(f"unknown kanban workflow trait {trait!r}")
        return tuple(c.key for c in self.columns if trait in c.traits)

    def on_event(self, key: str, event: str) -> str:
        """Column a card in ``key`` moves to when the kernel emits ``event``."""
        if event not in EVENTS:
            raise ValueError(f"unknown kanban workflow event {event!r}")
        col = self.column(key)
        if col is not None and event in col.on:
            return col.on[event]
        return self.defaults[event]

    def can_move(self, src: str, dst: str) -> bool:
        """True when a human may request ``src -> dst`` (archive always allowed)."""
        if dst == ARCHIVED:
            return True
        return dst in self.manual.get(src, frozenset())

    def to_dict(self) -> dict:
        """JSON shape served by the dashboard's ``GET /workflow``."""
        return {
            "columns": [
                {"key": c.key, "label": c.label, "icon": c.icon, "traits": sorted(c.traits),
                 "drag_target": c.drag_target, "on": dict(c.on)}
                for c in self.columns
            ],
            "defaults": dict(self.defaults),
            "manual": {src: sorted(dsts) for src, dsts in self.manual.items()},
            "archived": ARCHIVED,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent workflow (unknown keys, dead-end events)."""
        keys = self.keys()
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate column keys: {keys}")
        if ARCHIVED in keys:
            raise ValueError(f"{ARCHIVED!r} is reserved and cannot be a column")
        known = set(keys)
        for c in self.columns:
            unknown = c.traits - TRAITS
            if unknown:
                raise ValueError(f"column {c.key!r}: unknown traits {sorted(unknown)}")
            for ev, dst in c.on.items():
                if ev not in EVENTS or dst not in known:
                    raise ValueError(f"column {c.key!r}: bad edge {ev!r} -> {dst!r}")
        missing = EVENTS - set(self.defaults)
        if missing:
            raise ValueError(f"no default column for events {sorted(missing)}")
        for ev, dst in self.defaults.items():
            if dst not in known:
                raise ValueError(f"default edge {ev!r} -> unknown column {dst!r}")
        for src, dsts in self.manual.items():
            if src not in known or not set(dsts) <= known:
                raise ValueError(f"manual moves from {src!r} reference unknown columns")
        if not self.keys_with(TERMINAL):
            raise ValueError("workflow needs at least one terminal column")


def _col(key: str, label: str, icon: str, *traits: str, drag_target: bool = True) -> Column:
    return Column(key=key, label=label, icon=icon, traits=frozenset(traits), drag_target=drag_target)


def _frozen_manual(table: Mapping[str, tuple]) -> Mapping[str, frozenset]:
    return MappingProxyType({src: frozenset(dsts) for src, dsts in table.items()})


# Today's board, written down. Column order = dashboard order. ``drag_target=False``
# mirrors the desktop's LOCKED_COLUMNS (review needs a reviewer, scheduled a wake time,
# running is kernel-only). The manual table is the measured PATCH /tasks/{id} matrix on
# a parentless task; ``test_kanban_workflow.py`` re-measures it against the live code.
DEFAULT_WORKFLOW = Workflow(
    columns=(
        _col("triage", "Triage", "◇", DECOMPOSE),
        _col("todo", "Todo", "◻", WAIT_PARENTS),
        _col("scheduled", "Scheduled", "⏱", HOLD_TIME, drag_target=False),
        _col("ready", "Ready", "▶", DISPATCH_IMPLEMENT),
        _col("running", "Running", "●", CLAIMED, drag_target=False),
        _col("blocked", "Blocked", "⊘", HOLD_HUMAN),
        _col("review", "Review", "◎", DISPATCH_REVIEW, drag_target=False),
        _col("done", "Done", "✓", TERMINAL),
    ),
    defaults=MappingProxyType({
        EV_CLAIM: "running",
        EV_COMPLETE: "done",
        EV_BLOCK: "blocked",
        EV_SCHEDULE: "scheduled",
        EV_REVIEW: "review",
        EV_CHANGES: "ready",
        EV_PARENTS_DONE: "ready",
    }),
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
    }),
)
DEFAULT_WORKFLOW.validate()

# Every value ``tasks.status`` can hold under the default workflow.
DEFAULT_STATUSES = frozenset(DEFAULT_WORKFLOW.keys()) | {ARCHIVED}
