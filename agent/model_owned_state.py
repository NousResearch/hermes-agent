"""Model-owned derived state for a model switch, shared by every switch path.

This module owns the *model-owned* half of a route switch: prompt-cache flags
(``_use_prompt_caching`` / ``_use_native_cache_layout``), the per-model
context-length re-point of ``agent.context_compressor``, and the per-model
``reasoning_config``.  ``agent_runtime_helpers.switch_model`` projects exactly
this state after its identity/transport swap, and any other model-only switch
(a runtime override, a later routing phase) must project the same set, so the
two can never drift apart.

The context-length and compressor helpers stay in
``agent.agent_runtime_helpers``; they are imported inside the function so this
module remains importable from that facade without an import cycle.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_model_owned_state(
    agent, new_model, snapshot=None, *, provider=None, api_mode=None,
) -> None:
    """Project the model-owned derived state for ``new_model`` onto ``agent``.

    Runs after the agent's identity fields already describe the destination
    route (``agent.model``/``provider``/``base_url``/``api_mode``), which is how
    the destination context length resolves.  ``agent._config_context_length``
    and ``agent._custom_providers`` are refreshed as part of the projection.

    ``provider``/``api_mode`` default to the agent fields; ``switch_model``
    passes its resolved destination route explicitly (the transport rebuild may
    have overridden ``agent.api_mode`` -- MoA pins ``chat_completions``).

    ``snapshot`` is the caller's rollback authority for context-length /
    compressor resolution failures: ``switch_model`` passes its full switch
    snapshot (restore + re-raise aborts the switch); a caller without one passes
    ``None`` and owns the failure itself.  The reasoning-config re-resolve is
    best-effort on every path.
    """
    # Deferred: agent_runtime_helpers imports this module at module load.
    from agent.agent_runtime_helpers import (
        _resolve_switch_context_length,
        _update_switch_compressor,
    )

    custom_providers, effective_context_length = _resolve_switch_context_length(agent, snapshot)
    # Refresh the custom-provider snapshot from the config just loaded so the
    # prompt_caching lookup sees flags added to config.yaml after session start.
    if custom_providers is not None:
        agent._custom_providers = custom_providers
    agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
        provider=provider if provider is not None else agent.provider,
        base_url=agent.base_url,
        api_mode=api_mode if api_mode is not None else agent.api_mode,
        model=new_model,
    )
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        _update_switch_compressor(agent, custom_providers, effective_context_length, snapshot)
    # Re-read the per-model reasoning_effort override so it applies immediately
    # (per-model > global; YAML False = disabled).
    try:
        from hermes_constants import resolve_reasoning_config
        from hermes_cli.config import load_config as _sm_load_config
        agent.reasoning_config = resolve_reasoning_config(_sm_load_config() or {}, agent.model)
        logger.info(
            "model switch: reasoning_config resolved for %s: %s", agent.model, agent.reasoning_config
        )
    except Exception as _reasoning_err:
        logger.debug("switch_model: could not re-resolve reasoning_config: %s", _reasoning_err, exc_info=True)
