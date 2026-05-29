"""Client-side guards before projecting Hermes todo state to Kynver plan progress."""

from __future__ import annotations

from typing import Any


class PreTransitionError(ValueError):
    """Todo or focus transition rejected before calling Kynver."""


HERMES_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
KYNVER_ROW_STATUSES = frozenset({"todo", "in_progress", "running", "partial", "blocked", "done"})


def normalize_hermes_status(status: str) -> str:
    clean = (status or "pending").strip().lower()
    return clean if clean in HERMES_STATUSES else "pending"


def hermes_row_key(todo_id: str) -> str:
    tid = (todo_id or "").strip() or "?"
    return f"hermes-todo:{tid}" if not tid.startswith("hermes-todo:") else tid


def assert_single_in_progress(todos: list[dict[str, Any]]) -> None:
    active = [t for t in todos if normalize_hermes_status(str(t.get("status", ""))) == "in_progress"]
    if len(active) > 1:
        raise PreTransitionError("only one todo may be in_progress at a time")


def assert_focus_allowed(*, row_status: str | None, next_hermes_status: str) -> None:
    if next_hermes_status != "in_progress":
        return
    if row_status == "running":
        raise PreTransitionError(
            "cannot set Hermes todo in_progress while Kynver row has executor lease (running)"
        )
    if row_status == "done":
        raise PreTransitionError("cannot set focus on a done plan row")
