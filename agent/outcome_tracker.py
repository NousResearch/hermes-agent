"""Wire up the dead bump_use counter by scanning session messages for skill_view calls.

Called once at session end from AIAgent.shutdown_memory_provider. Best-effort:
failures are logged at DEBUG and never propagate to the caller.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def record_session_outcome(messages: list[dict[str, Any]]) -> None:
    """Increment use_count for every skill viewed during the session."""
    skill_names = _extract_skill_views(messages)
    if not skill_names:
        return
    try:
        from tools.skill_usage import bump_use
        for name in skill_names:
            bump_use(name)
    except Exception as exc:
        logger.debug("outcome_tracker: bump_use failed: %s", exc)


def _extract_skill_views(messages: list[dict[str, Any]]) -> set[str]:
    """Return the set of skill names loaded via skill_view in this session.

    Parses OpenAI-format assistant messages (role='assistant', tool_calls=[...]).
    Malformed entries are silently skipped.
    """
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if func.get("name") != "skill_view":
                continue
            try:
                args = json.loads(func.get("arguments", "{}"))
                name = args.get("name") or args.get("skill_name")
                if name:
                    seen.add(str(name))
            except Exception:
                pass
    return seen
