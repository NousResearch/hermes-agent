"""One-shot model context for sessions restored across process boundaries."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from hermes_cli.timefmt import coerce_epoch

# Imported during CLI / TUI gateway startup.  Session creation timestamps are
# deliberately not used: an in-process session switch is not a process restart.
PROCESS_START = time.time()

SESSION_RESUME_NOTE = (
    "[Session resumed after a process restart. "
    "This conversation history was restored from a prior Hermes process. "
    "Tool calls shown in the restored history have already been executed — "
    "do NOT re-execute them merely because they appear above. "
    "Continue with the user's current message.]"
)


def history_predates_process(
    history: Iterable[Mapping[str, Any]] | None,
    *,
    process_start: float | None = None,
) -> bool:
    """Whether the newest timestamped history row predates this process.

    Restored projections can end in synthetic rows without timestamps, so walk
    backward to the newest trustworthy timestamp instead of assuming the final
    dict has one.  Missing or corrupt timestamps fail closed (no note).
    """
    started_at = PROCESS_START if process_start is None else float(process_start)
    for message in reversed(list(history or ())):
        if not isinstance(message, Mapping):
            continue
        timestamp = coerce_epoch(
            message.get("timestamp"), field="session resume message timestamp"
        )
        if timestamp is not None:
            return timestamp < started_at
    return False


def stage_session_resume_note(
    agent: Any,
    history: Iterable[Mapping[str, Any]] | None,
    *,
    process_start: float | None = None,
) -> bool:
    """Stage the restart note on *agent* for its next user turn only.

    ``agent.turn_context`` consumes this attribute into the current user
    message's ``api_content`` sidecar.  That keeps the cached system prompt
    byte-stable, adds no synthetic role, and makes later replay use the exact
    bytes sent on the first post-resume turn.
    """
    if agent is None or getattr(agent, "_session_resume_note", ""):
        return False
    if not history_predates_process(history, process_start=process_start):
        return False
    agent._session_resume_note = SESSION_RESUME_NOTE
    return True
