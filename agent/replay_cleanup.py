"""Replay-history sanitization shared across resume code paths.

When a session's last turn dies mid-tool-loop — the process is killed by a
restart/shutdown command, a stale-timeout fires, or an interrupt lands before
the tool result is written — the persisted transcript can end with a dangling
``assistant(tool_calls)`` (no matching ``tool`` answer) or an interrupted
``assistant→tool`` block.  On resume the model sees that broken tail and
re-issues the unanswered call, producing an endless "thinking"/reboot loop
(#49201, #29086).

These pure helpers strip those tails before the history is replayed to the
model.  They were originally local to ``gateway/run.py`` (which fixed the
messaging-gateway path) and are extracted here so every resume surface — the
messaging gateway AND the TUI/WebUI gateway — shares the same cleanup instead
of the WebUI path silently skipping it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def is_interrupted_tool_result(content: Any) -> bool:
    """Return True if a tool result indicates the tool was interrupted."""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    if "[command interrupted]" in lowered:
        return True
    if "exit_code" in lowered and ("130" in lowered or "-1" in lowered):
        return "interrupt" in lowered
    return False


def strip_interrupted_tool_tails(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip interrupted assistant→tool sequences from replay history.

    Older interrupted gateway turns can be followed by a queued real user
    message, so the interrupted assistant/tool block is not necessarily the
    final tail by the time we rebuild replay history.  Remove any contiguous
    assistant(tool_calls) + tool-result block that contains an interrupted tool
    result, while preserving successful tool-call sequences intact.
    """
    if not agent_history:
        return agent_history

    cleaned: List[Dict[str, Any]] = []
    i = 0
    n = len(agent_history)
    while i < n:
        msg = agent_history[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            j = i + 1
            tool_results: List[Dict[str, Any]] = []
            while j < n and agent_history[j].get("role") == "tool":
                tool_results.append(agent_history[j])
                j += 1
            if tool_results and any(
                is_interrupted_tool_result(m.get("content", ""))
                for m in tool_results
            ):
                logger.debug(
                    "Stripping interrupted assistant→tool replay block "
                    "(indices %d–%d, tool_results=%d)",
                    i, j - 1, len(tool_results),
                )
                i = j
                continue
        if msg.get("role") == "tool" and is_interrupted_tool_result(msg.get("content", "")):
            logger.debug("Stripping orphan interrupted tool result from replay history")
            i += 1
            continue
        cleaned.append(msg)
        i += 1

    return cleaned


def strip_dangling_tool_call_tail(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip a trailing ``assistant(tool_calls)`` block left with NO answers.

    When a tool call itself kills the gateway process (``docker restart``,
    ``systemctl restart``, ``kill``, ``hermes gateway restart``), the process
    is terminated by SIGKILL *mid-call* — before the tool result is ever
    written and before the orderly shutdown rewind
    (``_drop_trailing_empty_response_scaffolding``) can run.  The last thing
    persisted is the ``assistant`` message that issued the ``tool_calls``,
    with zero matching ``tool`` rows.

    On resume the model sees an unanswered tool call at the tail and naturally
    re-issues it — which restarts the gateway again, producing the infinite
    reboot loop in #49201.  ``strip_interrupted_tool_tails`` does not catch
    this because there is no tool result to inspect for an interrupt marker.

    This strips that dangling tail at the source so there is nothing for the
    model to re-execute.  It only acts when the tail is an
    ``assistant(tool_calls)`` whose calls have NO corresponding ``tool``
    results — a completed assistant→tool pair (any tool answers present) is
    left untouched so genuine mid-progress tool loops still resume.
    """
    if not agent_history:
        return agent_history

    last = agent_history[-1]
    if not (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and last.get("tool_calls")
    ):
        return agent_history

    logger.debug(
        "Stripping dangling unanswered assistant(tool_calls) tail "
        "(%d call(s)) — process likely killed mid-tool-call by a "
        "restart/shutdown command (#49201)",
        len(last.get("tool_calls") or []),
    )
    return agent_history[:-1]


def sanitize_replay_history(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply both replay-tail strippers in the canonical order.

    Convenience entry point for resume code paths: removes interrupted
    assistant→tool blocks anywhere in the history, then removes a dangling
    unanswered ``assistant(tool_calls)`` tail.  Returns the same list object
    when there is nothing to strip.
    """
    if not agent_history:
        return agent_history
    return strip_dangling_tool_call_tail(strip_interrupted_tool_tails(agent_history))
