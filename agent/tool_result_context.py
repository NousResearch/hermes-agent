"""Append plugin context only at fresh top-level tool-result boundaries."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def inject_tool_result_context(agent, messages):
    """Return delivery callbacks; the caller acknowledges only after canonical persistence."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    if (getattr(agent, "_interrupt_requested", False)
            or getattr(agent, "_persist_disabled", False)
            or not messages or messages[-1].get("role") != "tool"):
        return []
    message = messages[-1]
    if message.get(_DB_PERSISTED_MARKER) or message.get("_plugin_context_injected"):
        return []
    content = message.get("content")
    if not isinstance(content, (str, list)) and content is not None:
        return []
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not has_hook("tool_result_context"):
            return []
        from tools.hook_output_spill import get_spill_config, spill_if_oversized
        results = invoke_hook(
            "tool_result_context", session_id=agent.session_id,
            platform=getattr(agent, "platform", "") or "",
            parent_session_id=getattr(agent, "_parent_session_id", "") or "",
            tool_name=message.get("name", ""), tool_call_id=message.get("tool_call_id", ""),
        )
        pieces, callbacks = [], []
        for result in results:
            piece = result.get("context") if isinstance(result, dict) else result
            if not isinstance(piece, str) or not piece.strip():
                continue
            try:
                piece = spill_if_oversized(piece, session_id=agent.session_id,
                            source="tool_result_context", config=get_spill_config(),
                            raise_on_failure=True)
            except Exception:
                # Do not acknowledge unavailable context; other plugins can still deliver.
                continue
            pieces.append(piece)
            callback = result.get("on_delivery") if isinstance(result, dict) else None
            if callable(callback):
                callbacks.append(callback)
        if not pieces:
            return []
        text = "\n\n[PLUGIN CONTEXT — not a user message]\n" + "\n\n".join(pieces)
        if isinstance(content, str):
            message["content"] = content + text
        else:
            message["content"] = [*(content or []), {"type": "text", "text": text}]
        # Aggregate output limiting must not spill a checkpoint after delivery acknowledgement.
        message["_plugin_context_injected"] = True
        return callbacks
    except Exception:
        logger.warning("Tool-result context hook failed", exc_info=True)
        return []


def acknowledge_tool_result_context(callbacks, persisted):
    for callback in callbacks:
        try:
            from functools import partial
            from hermes_cli.plugins import get_plugin_manager, _resolve_hook_callback_timeout
            timeout = _resolve_hook_callback_timeout()
            if timeout > 0:
                # Reuse the hook worker cap and copied profile/parent context. Positional
                # binding also supports built-in callbacks such as list.append.
                get_plugin_manager()._run_hook_callback_bounded(
                    "tool_result_context_receipt", partial(callback, persisted), {}, timeout,
                )
            else:
                callback(persisted)
        except Exception:
            # A missing receipt may replay context, but must never lose a tool's real result.
            logger.warning("Tool-result context receipt failed", exc_info=True)
