"""LM Studio runtime/preload logic — 層級：Provider Logic Layer

Extracts LM Studio-specific logic from run_agent.py:
- Model preload with minimum context enforcement
- Reasoning options probing with caching
- Reasoning effort resolution

Forwarder pattern: actual logic lives here; run_agent.py delegates.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


def ensure_lmstudio_runtime_loaded(
    agent,
    config_context_length: Optional[int] = None,
) -> None:
    """
    Preload the LM Studio model with at least Hermes' minimum context.

    Args:
        agent: The AIAgent instance (used to access model, base_url, api_key,
               provider, api_mode, and context_compressor).
        config_context_length: Override the configured context length. If None,
            falls back to ``agent._config_context_length``.
    """
    if (agent.provider or "").strip().lower() != "lmstudio":
        return
    try:
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        from hermes_cli.models import ensure_lmstudio_model_loaded

        if config_context_length is None:
            config_context_length = getattr(agent, "_config_context_length", None)
        target_ctx = max(config_context_length or 0, MINIMUM_CONTEXT_LENGTH)
        loaded_ctx = ensure_lmstudio_model_loaded(
            agent.model,
            agent.base_url,
            getattr(agent, "api_key", ""),
            target_ctx,
        )
        if loaded_ctx:
            # Push into the live compressor so the status bar reflects the
            # real loaded ctx the moment the load resolves, instead of
            # holding the previous model's value (or "ctx --") through the
            # next render tick.
            cc: Optional["ContextCompressor"] = getattr(agent, "context_compressor", None)
            if cc is not None:
                cc.update_model(
                    model=agent.model,
                    context_length=loaded_ctx,
                    base_url=agent.base_url,
                    api_key=getattr(agent, "api_key", ""),
                    provider=agent.provider,
                    api_mode=agent.api_mode,
                )
    except Exception as err:
        logger.debug("LM Studio preload skipped: %s", err)


def lmstudio_reasoning_options_cached(agent) -> List[str]:
    """Probe LM Studio's published reasoning ``allowed_options`` once per
    (model, base_url). The list (e.g. ``["off","on"]`` or
    ``["off","minimal","low"]``) is needed both for the supports-reasoning
    gate and for clamping the emitted ``reasoning_effort`` so toggle-style
    models don't 400 on ``high``. Cache is keyed on (model, base_url) so
    ``/model`` swaps and base-URL changes don't reuse a stale list.
    Non-empty results are cached permanently (model capabilities don't
    change). Empty results (transient probe failure OR genuinely
    non-reasoning model) are cached with a 60-second TTL to avoid an
    HTTP round-trip on every turn while still retrying reasonably soon.

    Args:
        agent: The AIAgent instance (used to access model, base_url, api_key,
               and the cache attribute ``_lm_reasoning_opts_cache``).

    Returns:
        List of supported reasoning effort options (e.g. ``["off","on"]``).
    """
    cache = getattr(agent, "_lm_reasoning_opts_cache", None)
    if cache is None:
        cache = agent._lm_reasoning_opts_cache = {}
    key = (agent.model, agent.base_url)
    cached = cache.get(key)
    if cached is not None:
        opts, ts = cached
        # Non-empty → permanent. Empty → 60s TTL.
        if opts or (_time.monotonic() - ts) < 60:
            return opts
    try:
        from hermes_cli.models import lmstudio_model_reasoning_options

        opts = lmstudio_model_reasoning_options(
            agent.model,
            agent.base_url,
            getattr(agent, "api_key", ""),
        )
    except Exception:
        opts = []
    cache[key] = (opts, _time.monotonic())
    return opts


def resolve_lmstudio_summary_reasoning_effort(
    reasoning_config: Optional[dict],
    allowed_options: List[str],
) -> Optional[str]:
    """Resolve a safe top-level ``reasoning_effort`` for LM Studio.

    The iteration-limit summary path calls ``chat.completions.create()``
    directly, bypassing the transport. Share the helper so the two paths
    can't drift on effort resolution and clamping.

    Args:
        reasoning_config: The agent's ``reasoning_config`` dict.
        allowed_options: Output of ``lmstudio_reasoning_options_cached(agent)``.

    Returns:
        The reasoning effort string to send, or ``None`` to omit the field.
    """
    from agent.lmstudio_reasoning import resolve_lmstudio_effort

    return resolve_lmstudio_effort(reasoning_config, allowed_options)
