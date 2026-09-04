from __future__ import annotations

import sys
from typing import Any, Callable

from . import codex_runtime_impl as _impl


_original_consume_codex_event_stream = _impl._consume_codex_event_stream


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _codex_event_has_result_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return any(_codex_event_has_result_content(item) for item in value)
    if isinstance(value, dict):
        return any(
            _codex_event_has_result_content(value.get(key))
            for key in (
                "text",
                "content",
                "delta",
                "output_text",
                "result",
                "output",
                "arguments",
            )
            if key in value
        )
    if value is None:
        return False
    for key in (
        "text",
        "content",
        "delta",
        "output_text",
        "result",
        "output",
        "arguments",
    ):
        if hasattr(value, key) and _codex_event_has_result_content(getattr(value, key)):
            return True
    return False


def _normalized_phase(value: Any) -> str | None:
    phase = _field(value, "phase", None)
    if not isinstance(phase, str):
        return None
    normalized = phase.strip().lower()
    return normalized or None


def _codex_event_advances_aux_result(
    event: Any,
    active_message_phase: str | None = None,
) -> bool:
    """Return whether a Codex SSE frame advances the usable aux result.

    Reasoning/commentary remains transport activity, but it must not refresh
    the outer compression inactivity fence. Only result-bearing text or a
    function/tool result advances summary progress.
    """
    typ = str(_field(event, "type", "") or "")
    if not typ:
        return False

    if "reasoning" in typ:
        return False

    if typ in {
        "response.output_text.delta",
        "response.output_text.done",
        "response.text.delta",
        "response.text.done",
    }:
        if active_message_phase in {"analysis", "commentary"}:
            return False
        return _codex_event_has_result_content(_field(event, "delta", None)) or _codex_event_has_result_content(
            _field(event, "text", None)
        )

    if typ in {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }:
        return _codex_event_has_result_content(_field(event, "delta", None)) or _codex_event_has_result_content(
            _field(event, "arguments", None)
        )

    if typ == "response.output_item.added":
        item = _field(event, "item", None)
        item_type = str(_field(item, "type", "") or "")
        if "function_call" in item_type or item_type in {"tool_call", "function_call_output"}:
            return any(
                _codex_event_has_result_content(_field(item, key, None))
                for key in ("id", "call_id", "name", "arguments", "output")
            )
        return False

    if typ == "response.output_item.done":
        item = _field(event, "item", None)
        item_type = str(_field(item, "type", "") or "")
        if "reasoning" in item_type:
            return False
        if item_type == "message":
            phase = _normalized_phase(item) or active_message_phase
            if phase in {"analysis", "commentary"}:
                return False
            return _codex_event_has_result_content(_field(item, "content", None)) or _codex_event_has_result_content(
                _field(item, "text", None)
            )
        if "function_call" in item_type or item_type in {"tool_call", "function_call_output"}:
            return True
        return False

    if typ == "response.completed":
        response = _field(event, "response", None)
        if _codex_event_has_result_content(_field(response, "output_text", None)):
            return True
        output = _field(response, "output", None)
        if not isinstance(output, (list, tuple)):
            return False
        for item in output:
            item_type = str(_field(item, "type", "") or "")
            if "reasoning" in item_type:
                continue
            if item_type == "message":
                if _normalized_phase(item) in {"analysis", "commentary"}:
                    continue
                if _codex_event_has_result_content(_field(item, "content", None)) or _codex_event_has_result_content(
                    _field(item, "text", None)
                ):
                    return True
            elif "function_call" in item_type or item_type in {"tool_call", "function_call_output"}:
                return True
        return False

    return False


def _dispatch_event_without_aux_progress(
    event: Any,
    on_event: Callable[[Any], None],
) -> None:
    """Preserve the event callback while muting only the outer aux heartbeat.

    The Codex auxiliary callback owns the transport watchdog and cancellation
    checks, so it must still receive every SSE event with the event argument.
    During non-result provider activity, temporarily replace only the current
    thread's auxiliary-progress hook with a no-op. That leaves transport
    liveness intact while preventing reasoning/commentary from extending the
    compression summary inactivity fence.
    """
    aux_module = sys.modules.get("agent.auxiliary_client")
    aux_progress_hook = getattr(aux_module, "aux_progress_hook", None)
    if not callable(aux_progress_hook):
        on_event(event)
        return
    with aux_progress_hook(lambda: None):
        on_event(event)


def _consume_codex_event_stream(
    *args: Any,
    on_event: Callable[[Any], None] | None = None,
    **kwargs: Any,
):
    if on_event is None:
        return _original_consume_codex_event_stream(*args, on_event=None, **kwargs)

    active_message_phase: str | None = None

    def _route_event(event: Any) -> None:
        nonlocal active_message_phase
        typ = str(_field(event, "type", "") or "")
        if typ == "response.output_item.added":
            item = _field(event, "item", None)
            if str(_field(item, "type", "") or "") == "message":
                active_message_phase = _normalized_phase(item)
            else:
                active_message_phase = None

        if _codex_event_advances_aux_result(event, active_message_phase):
            on_event(event)
        else:
            _dispatch_event_without_aux_progress(event, on_event)

        if typ == "response.output_item.done":
            active_message_phase = None

    return _original_consume_codex_event_stream(
        *args,
        on_event=_route_event,
        **kwargs,
    )


# Preserve the original implementation module as the single mutable namespace.
# Existing runtime functions retain that module's globals, so monkeypatches
# against ``agent.codex_runtime`` still target the code that executes.
_impl._codex_event_has_result_content = _codex_event_has_result_content
_impl._codex_event_advances_aux_result = _codex_event_advances_aux_result
_impl._consume_codex_event_stream = _consume_codex_event_stream
sys.modules[__name__] = _impl
