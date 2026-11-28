"""Stop gate for a text answer that leaves ``todo_list`` items open.

A model often ends a turn on "Next I'll commit and check CI." while its own task list still has
pending or in-progress items, and the user is left with a half-done job reported as an answer.
With ``agent.todo_stop_gate: true`` such a stop gets ONE nudge per turn naming the open items;
the model either finishes them, updates the list, or answers again and the turn ends. Off by
default (legacy behaviour): every nudge is an extra model call.

Two stops are intentional and never nudged: a turn that handed work to a notifying background
process or a background ``delegate_task`` (the result wakes the agent later), and a reply that asks
the user for a decision, approval or input. Salvaged from #80155 and restacked onto
``agent.turn_stop_gates``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from agent.completion_evidence import current_turn

_MAX_NUDGES_PER_TURN = 1
_MAX_LABELS = 6
_ACTIVE = frozenset({"pending", "in_progress"})

_WAITS_FOR_USER_RE = re.compile(
    r"\b(?:need|require|await|waiting for)\s+(?:your|the user'?s)\s+"
    r"(?:approval|permission|input|confirmation|choice|decision|credentials?)\b"
    r"|\bplease\s+(?:provide|confirm|choose|approve)\b"
    r"|(?:需要|等待|请)(?:你|您|用户)(?:的)?(?:批准|授权|许可|确认|输入|选择|决定|凭证)"
    r"|请(?:提供|确认|选择|批准|授权)",
    re.IGNORECASE,
)


def todo_stop_gate_enabled() -> bool:
    """``agent.todo_stop_gate`` from the active profile's config; only an explicit ``true``."""
    from hermes_cli.config import load_config_readonly

    agent_cfg = (load_config_readonly() or {}).get("agent")
    return isinstance(agent_cfg, dict) and agent_cfg.get("todo_stop_gate") is True


def _active_items(agent: Any) -> list[dict[str, Any]]:
    store = getattr(agent, "_todo_store", None)
    return [i for i in (store.read() if store is not None else []) if i.get("status") in _ACTIVE]


def _handed_off_to_background(turn_rows: list[Any]) -> bool:
    """This turn started work that reports back on its own (notify_on_complete process or a
    background delegate_task), so stopping now is waiting, not abandoning."""
    for row in turn_rows:
        if not isinstance(row, dict) or row.get("role") != "tool" or not isinstance(row.get("content"), str):
            continue
        try:
            payload = json.loads(row["content"])
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("notify_on_complete") is True and payload.get("output") == "Background process started":
            return True
        if payload.get("status") == "dispatched" and payload.get("mode") == "background":
            return True
    return False


def _labels(items: list[dict[str, Any]]) -> str:
    ids = ", ".join(str(i.get("id", "?")) for i in items[:_MAX_LABELS])
    return ids + (", …" if len(items) > _MAX_LABELS else "")


def build_todo_stop_nudge(agent: Any, final_response: Any, messages: list[Any]) -> Optional[str]:
    """The nudge for a stop that leaves todos open, or None when the gate is off, already used
    this turn, the list is settled, or the stop is an intentional pause."""
    if getattr(agent, "_todo_stop_nudges", 0) >= _MAX_NUDGES_PER_TURN:
        return None
    active = _active_items(agent)
    if not active or not todo_stop_gate_enabled():
        return None
    if _WAITS_FOR_USER_RE.search(str(final_response or "")):
        return None
    if _handed_off_to_background(current_turn(messages)):
        return None
    return (
        f"[System note: your task list still has open items ({_labels(active)}). If the work is not "
        "done, continue it now with the next tool call. If it is done, mark those items completed "
        "or cancelled with todo_list. If you need the user's approval, input or a decision, ask that "
        "question. Then give your final answer.]"
    )
