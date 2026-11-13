"""Bind the summary engine's supported compression arguments."""

import inspect
import logging
from typing import Any, Callable, Optional

from agent.context_compressor_cache import summary_prefix_for_agent

logger = logging.getLogger(__name__)


def _supported_compression_kwargs(
    compress_fn: Any, *, current_tokens: Optional[int], focus_topic: Optional[str], force: bool,
    memory_context: str, bypass_cooldown: bool = False, summary_prefix: Any = None,
) -> dict:
    """Inspect first: never retry a stateful engine after catching TypeError."""
    candidates = {"current_tokens": current_tokens, "focus_topic": focus_topic, "force": force}
    if bypass_cooldown:
        candidates["bypass_cooldown"] = True
    if memory_context:
        candidates["memory_context"] = memory_context
    if summary_prefix is not None:
        candidates["summary_prefix"] = summary_prefix
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        return {"current_tokens": current_tokens}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


def _resolve_compress_call(
    agent: Any, *, approx_tokens: Optional[int], focus_topic: Optional[str], force: bool, memory_context: str,
    bypass_cooldown: bool,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Bind compress() and only the kwargs its signature accepts."""
    compress_fn = agent.context_compressor.compress
    compress_kwargs = _supported_compression_kwargs(
        compress_fn, current_tokens=approx_tokens, focus_topic=focus_topic, force=force, memory_context=memory_context,
        bypass_cooldown=bypass_cooldown, summary_prefix=summary_prefix_for_agent(agent),
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
