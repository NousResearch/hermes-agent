"""Lean compaction appendix: background processes this session still has running.

A backgrounded command's handle lives in its tool result, which compaction can summarise
away; this section re-lists the live handles so the model can poll, wait on or kill them
instead of losing track or starting duplicates. Rebuilt from the registry every compaction.
"""

import contextlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_LEAN_BACKGROUND_HEADING = "## Background Processes (still running)"
_LEAN_BACKGROUND_MAX_ENTRIES = 10
_LEAN_BACKGROUND_BUDGET_CHARS = 1_500
_LEAN_BACKGROUND_COMMAND_CHARS = 80
_LEAN_BACKGROUND_MORE_RESERVE = 48  # room for the "+N more" line inside the budget
_BACKGROUND_SECTION = re.compile(
    rf"(?ms)^{re.escape(_LEAN_BACKGROUND_HEADING)}\r?$\n?.*?(?=^## |\Z)"
)


def _strip_background_section(summary: str) -> str:
    """Remove only whole-line process sections, leaving other summary sections intact."""
    return _BACKGROUND_SECTION.sub("", summary)


def _session_owner_ids(session_id: str, session_db: Any) -> set[str]:
    """This session's id plus its compression ancestors: agent turns spawn processes with the
    session id as owner, and compaction rotates that id while the process keeps running."""
    owners = {session_id}
    lineage = getattr(session_db, "get_compression_lineage", None)
    if callable(lineage):
        with contextlib.suppress(Exception):  # the current id alone still scopes the list
            owners.update(lineage(session_id) or ())
    return owners


def _short_command(command: str) -> str:
    text = " ".join((command or "").split())
    limit = _LEAN_BACKGROUND_COMMAND_CHARS
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _build_background_section(compressor: Any) -> str:
    """Running processes owned by the compressor's session, or "" when none (or no session)."""
    session_id = getattr(compressor, "_session_id", "") or ""
    if not session_id:
        return ""
    owners = _session_owner_ids(session_id, getattr(compressor, "_session_db", None))
    try:  # optional appendix: an unavailable registry omits it and never fails the compaction
        from tools.process_registry import process_registry

        # list_sessions refreshes recovered detached entries, so an exited PID is not reported running.
        procs = sorted((p for p in process_registry.list_sessions()
                        if p["owner_task_id"] in owners and p["status"] == "running"),
                       key=lambda p: p["started_at"])
    except Exception:
        logger.debug("background-process appendix omitted: registry unavailable", exc_info=True)
        return ""
    if not procs:
        return ""
    lines = [
        "\n\n" + _LEAN_BACKGROUND_HEADING,
        "Use process(action='poll'|'log'|'wait'|'kill', session_id=...); do not start duplicates.",
    ]
    used = sum(len(line) + 1 for line in lines)
    shown = 0
    for proc in procs[:_LEAN_BACKGROUND_MAX_ENTRIES]:
        started = str(proc["started_at"])[:16].replace("T", " ")
        line = f"- {proc['session_id']} (started {started}): {_short_command(proc['command'])}"
        if used + len(line) + 1 > _LEAN_BACKGROUND_BUDGET_CHARS - _LEAN_BACKGROUND_MORE_RESERVE:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(procs):
        lines.append(f"- (+{len(procs) - shown} more: process(action='list'))")
    return "\n".join(lines)
