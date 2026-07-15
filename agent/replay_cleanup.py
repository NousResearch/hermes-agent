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

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_result_classification import tool_may_have_side_effect
from utils import safe_json_loads

logger = logging.getLogger(__name__)


_TERMINAL_INTERRUPT_MARKER = "[Command interrupted]"


def is_interrupted_tool_result(content: Any, tool_name: str = "") -> bool:
    """Recognize only the terminal tool's exact interruption envelope.

    Tool results are otherwise opaque model-visible data.  A business tool is
    free to return fields such as ``status`` or ``exit_code`` without granting
    replay cleanup authority over its transcript row.  The caller therefore
    has to bind the result's ``tool_call_id`` to the exact assistant-declared
    tool name before invoking this helper.

    Current terminal results use a JSON envelope with an integer POSIX 130
    exit code and an exact final interrupt-marker line.  The exact marker by
    itself remains supported only for legacy terminal transcripts.
    """
    if tool_name != "terminal" or not isinstance(content, str):
        return False
    stripped = content.strip()
    if stripped == _TERMINAL_INTERRUPT_MARKER:
        return True
    data = safe_json_loads(stripped)
    if not isinstance(data, dict):
        return False
    if type(data.get("exit_code")) is not int or data["exit_code"] != 130:
        return False
    output = data.get("output")
    if not isinstance(output, str):
        return False
    output_lines = output.rstrip().splitlines()
    return bool(output_lines) and output_lines[-1].strip() == _TERMINAL_INTERRUPT_MARKER


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
            calls = msg.get("tool_calls") or []
            call_names = {
                str(call.get("id") or call.get("call_id") or ""): str(
                    (call.get("function") or {}).get("name") or ""
                )
                for call in calls
                if call.get("id") or call.get("call_id")
            }

            def _result_is_interrupted(tool_result: Dict[str, Any]) -> bool:
                call_id = str(tool_result.get("tool_call_id") or "")
                return is_interrupted_tool_result(
                    tool_result.get("content", ""),
                    call_names.get(call_id, ""),
                )

            if tool_results and any(
                _result_is_interrupted(tool_result)
                for tool_result in tool_results
            ):
                if any(
                    tool_may_have_side_effect(
                        str((call.get("function") or {}).get("name") or "")
                    )
                    for call in calls
                ):
                    cleaned.append(msg)
                    for tool_result in tool_results:
                        if not _result_is_interrupted(tool_result):
                            cleaned.append(tool_result)
                            continue
                        recovered = dict(tool_result)
                        name = call_names.get(str(tool_result.get("tool_call_id") or ""), "")
                        recovered["effect_disposition"] = (
                            "unknown" if tool_may_have_side_effect(name) else "none"
                        )
                        recovered["content"] = (
                            "[Orphan recovery: interrupted side-effecting tool may have "
                            "executed; its effect is UNKNOWN. Inspect state before retrying.]"
                            if recovered["effect_disposition"] == "unknown"
                            else "[Orphan recovery: interrupted read-only tool did not complete.]"
                        )
                        cleaned.append(recovered)
                    i = j
                    continue
                logger.debug(
                    "Stripping interrupted read-only assistant→tool replay block "
                    "(indices %d–%d, tool_results=%d)",
                    i, j - 1, len(tool_results),
                )
                i = j
                continue
        # An orphan tool result has no assistant-owned call envelope from
        # which to establish the exact tool name.  Keep it opaque instead of
        # guessing from arbitrary result content.
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

    tool_calls = last.get("tool_calls") or []
    if any(
        tool_may_have_side_effect(
            str((call.get("function") or {}).get("name") or "")
        )
        for call in tool_calls
    ):
        recovered = list(agent_history)
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "unknown")
            call_id = str(call.get("id") or call.get("call_id") or "")
            disposition = "unknown" if tool_may_have_side_effect(name) else "none"
            content = (
                "[Orphan recovery: this tool may have executed before Hermes stopped; "
                "its effect is UNKNOWN. Inspect current state before retrying.]"
                if disposition == "unknown"
                else "[Orphan recovery: this read-only tool did not complete and had no effect.]"
            )
            recovered.append(make_tool_result_message(
                name, content, call_id, effect_disposition=disposition,
            ))
        logger.warning(
            "Recovered dangling side-effecting tool call(s) as UNKNOWN instead of erasing them"
        )
        return recovered

    logger.debug(
        "Stripping dangling unanswered read-only assistant(tool_calls) tail (%d call(s))",
        len(tool_calls),
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
