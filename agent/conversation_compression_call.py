"""Engine-call binding for context compression, moved verbatim from
``agent.conversation_compression`` so that file keeps going down toward its
code-health line cap (moved code keeps its cap; see root AGENTS.md). Callers
import the name from :mod:`agent.conversation_compression`, which re-exports it.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _tc_get(obj: Any, key: str, default: Any = "") -> Any:
    """Field of a dict- or object-shaped tool call (or its ``function`` sub-object)."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _tool_calls_by_id(messages: list[dict[str, Any]]) -> dict[str, tuple]:
    """Map ``tool_call_id -> (tool_name, raw_arguments)`` over every assistant tool call."""
    out: dict[str, tuple] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = _tc_get(tc, "function", {})
            out[_tc_get(tc, "id") or ""] = (_tc_get(fn, "name", "unknown"), _tc_get(fn, "arguments"))
    return out


def _supported_compression_kwargs(
    compress_fn: Any, *, current_tokens: Optional[int], focus_topic: Optional[str], force: bool,
    memory_context: str, bypass_cooldown: bool = False, protect_last_n_override: Optional[int] = None,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.
    Inspecting first keeps older plugin signatures compatible without catching ``TypeError`` and running a
    stateful compressor twice."""
    candidates = {"current_tokens": current_tokens, "focus_topic": focus_topic, "force": force}
    if bypass_cooldown:
        candidates["bypass_cooldown"] = True
    if memory_context:
        candidates["memory_context"] = memory_context
    # A caller-chosen verbatim boundary (/compress here N) disables the tail window for
    # its head compress (#144423); None means "not a boundary call, keep the default".
    if protect_last_n_override is not None:
        candidates["protect_last_n_override"] = protect_last_n_override
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # current_tokens has always been in the ContextEngine ABC; use the oldest call
        # shape when the callable has no inspectable signature.
        return {"current_tokens": current_tokens}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


def _resolve_compress_call(
    agent: Any, *, approx_tokens: Optional[int], focus_topic: Optional[str], force: bool, memory_context: str,
    bypass_cooldown: bool, protect_last_n_override: Optional[int] = None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Bind ``compress()`` and only the kwargs its signature accepts."""
    compress_fn = agent.context_compressor.compress
    compress_kwargs = _supported_compression_kwargs(
        compress_fn, current_tokens=approx_tokens, focus_topic=focus_topic, force=force, memory_context=memory_context,
        bypass_cooldown=bypass_cooldown, protect_last_n_override=protect_last_n_override,
    )
    if memory_context.strip() and "memory_context" not in compress_kwargs:
        engine_name = getattr(agent.context_compressor, "name", type(agent.context_compressor).__name__)
        if getattr(agent, "_last_memory_context_unsupported_engine", None) != engine_name:
            agent._last_memory_context_unsupported_engine = engine_name
            logger.warning(
                "context engine %s does not accept memory_context; continuing without provider-supplied summary context",
                engine_name,
            )
    return compress_fn, compress_kwargs
