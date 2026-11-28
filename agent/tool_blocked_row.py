"""Show a tool call that a guard stopped before it ran.

A call blocked by a ``pre_tool_call`` hook returns before the executor reaches
``_begin_tool_execution``, so it used to emit neither a start nor a completion signal: the user
saw a spinner and then nothing, and could not tell "the agent never tried" from "a guard stopped
it" — which hides the moment a guard plugin worked (#151276).

These helpers emit the pair a real call emits, minus the checkpoint preflight (a call that will
not run must not snapshot the workspace), and remember which rows were opened so a blocked call is
closed iff it was announced. That boundary is deliberate: a guardrail/scope/pruned-args block keeps
its contract of emitting nothing at all, pinned by
``tests/agent/test_tool_call_guardrail_runtime.py::test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution``.
"""

import json
import logging

from agent.display import build_tool_preview as _build_tool_preview
from agent.display import redact_tool_args_for_display as _redact_tool_args_for_display

logger = logging.getLogger(__name__)


def blocked_row_open(agent, tool_call_id) -> bool:
    """True when :func:`emit_blocked_tool_start` announced a row for this blocked call."""
    return tool_call_id in getattr(agent, "_visible_blocked_tool_calls", ())


def _safe(callback, label: str, *args) -> None:
    """Invoke a UI callback; a failing bridge is logged with its traceback, never fatal."""
    if not callback:
        return
    try:
        callback(*args)
    except Exception:  # a UI bridge must never break the executor
        logger.debug("%s callback error", label, exc_info=True)


def emit_blocked_tool_start(agent, ref) -> None:
    """Announce ``ref`` as a started tool row on both start lanes.

    ``tool.started`` covers progress-only consumers (CLI scrollback, ACP, the API server);
    ``tool_start_callback`` is the ID-keyed lane TUI/Desktop open their row from — they drop the
    progress ``tool.started`` as an id-less duplicate, so a completion frame alone would land on a
    row that never opened. No checkpoint preflight rides along: a blocked call never reaches
    ``_begin_tool_execution``, so nothing snapshots a workspace it will not change."""
    announced = getattr(agent, "_visible_blocked_tool_calls", None)
    if announced is None:
        announced = agent._visible_blocked_tool_calls = set()
    announced.add(ref.call_id)
    display_args = _redact_tool_args_for_display(ref.name, ref.args) or ref.args
    preview = None
    try:
        preview = _build_tool_preview(ref.name, display_args)
    except Exception:  # a preview must never cost the row
        logger.debug("blocked-tool preview failed for %s", ref.name, exc_info=True)
    _safe(agent.tool_progress_callback, "Tool progress", "tool.started", ref.name, preview, display_args)
    _safe(agent.tool_start_callback, "Tool start", ref.call_id, ref.name, display_args)


def blocked_tool_result(agent, ref, *, block_body: dict | None, block_error_type: str, guardrail_decision) -> str:
    """Synthesize the result for a blocked call (the JSON the model sees), announce a hook block as
    a tool row and emit the call's one terminal ``post_tool_call``."""
    if block_body is not None:
        result = json.dumps(block_body, ensure_ascii=False)
        error_type, error_message = block_error_type, block_body.get("message") or block_body["error"]
    else:
        result = agent._guardrail_block_result(guardrail_decision)
        error_type = "guardrail_block"
        error_message = getattr(guardrail_decision, "message", None) or "Tool blocked by guardrail policy"
    # A hook block carries its own message in ``block_body``; a scope/pruned-args block carries a
    # different tag and a guardrail block none at all, so both keep their silent contract.
    if block_body is not None and block_error_type == "plugin_block":
        emit_blocked_tool_start(agent, ref)
    ref.emit_post(agent, result, status="blocked", error_type=error_type, error_message=error_message)
    return result
