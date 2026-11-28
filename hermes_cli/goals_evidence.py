"""Host-observed evidence for the ``/goal`` judge.

The judge used to read only the agent's reply, so an invented "all 42 tests pass" and a real one
looked the same. This block gives it what the host saw: the tool calls of the turn being judged
(name, key argument, exit code), whether a passing test/build/CI command ran after the last file
edit anywhere in the recent session, and the verification ledger's verdict when that ledger is on
(``agent.verify_on_stop``). Read from ``state.db`` so CLI, TUI, Desktop and gateway feed the judge
the same evidence with no call-site changes; the judge's system prompt says how to weigh it.
First version: #117222.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agent.completion_evidence import ToolEvent, check_after_last_edit, current_turn, tool_events

logger = logging.getLogger(__name__)

# Bounded so a tool-heavy turn cannot bloat the judge prompt (~1 KB of lines at most).
_TOOL_ACTIVITY_MAX_ITEMS = 12
# How far back "after the last edit" looks: enough for a multi-turn goal, bounded for one read.
_EVIDENCE_WINDOW_ROWS = 400

JUDGE_TOOL_ACTIVITY_BLOCK_TEMPLATE = (
    "Tool activity the host observed (fact, not part of the agent's response):\n"
    "This turn:\n{tool_lines}\n{evidence_lines}\n\n"
)


def _event_line(event: ToolEvent) -> str:
    line = f"- {event.name} {event.detail}".rstrip()
    if event.edit_failed:
        return line + " → FAILED (nothing written)"
    return line + (f" → exit {event.exit_code}" if event.exit_code is not None else "")


def _ledger_line(session_id: str, events: list[ToolEvent]) -> str:
    """The verification ledger's verdict for the edited workspace, or "" when it is off or knows
    nothing about it."""
    from agent.verification_evidence import verification_status

    seen: set[str] = set()
    for event in events:
        for raw in event.paths:
            cwd = str(Path(raw).expanduser().parent)
            if cwd in seen:
                continue
            seen.add(cwd)
            status = verification_status(session_id=session_id, cwd=cwd)
            state = str(status.get("status") or "")
            if state in ("disabled", "not_applicable"):
                continue
            command = (status.get("evidence") or {}).get("canonical_command")
            return f"Verification ledger for {status.get('root') or cwd}: {state}" + (f" ({command})" if command else "")
    return ""


def gather_tool_activity(session_id: Optional[str], *, max_items: int = _TOOL_ACTIVITY_MAX_ITEMS) -> str:
    """The judge's evidence block for ``session_id``'s latest turn, or "" when the session shows no
    tool activity at all (the judge prompt is then byte-identical to the no-evidence case).
    Fails open: a store read error yields "" and the judge falls back to the reply alone."""
    if not session_id:
        return ""
    from hermes_cli.goals import _get_session_db

    db = _get_session_db()
    if db is None:
        return ""
    try:
        rows = db.get_messages(str(session_id), limit=_EVIDENCE_WINDOW_ROWS, latest=True) or []
    except Exception as exc:  # the judge path is fail-open by design
        logger.debug("gather_tool_activity: messages read failed: %s", exc, exc_info=True)
        return ""
    window = tool_events(rows)
    if not window:
        return ""
    turn = tool_events(current_turn(rows))
    shown = turn[-max_items:]
    tool_lines = [_event_line(e) for e in shown] or ["- (no tool calls)"]
    if len(turn) > len(shown):
        tool_lines.insert(0, f"- ({len(turn) - len(shown)} earlier calls this turn omitted)")
    verdict = check_after_last_edit(window)
    evidence = {
        "no_edits": "No file edits observed in this session.",
        "verified": f"Checks after the last edit: {verdict['reason']}.",
    }.get(verdict["state"], f"Checks after the last edit: NOT VERIFIED: {verdict['reason']}.")
    try:
        ledger = _ledger_line(str(session_id), window)
    except Exception as exc:  # a ledger I/O error must not cost the judge its other evidence
        logger.debug("gather_tool_activity: ledger read failed: %s", exc, exc_info=True)
        ledger = ""
    return JUDGE_TOOL_ACTIVITY_BLOCK_TEMPLATE.format(
        tool_lines="\n".join(tool_lines), evidence_lines="\n".join(filter(None, (evidence, ledger))),
    )


__all__ = ["JUDGE_TOOL_ACTIVITY_BLOCK_TEMPLATE", "gather_tool_activity"]
