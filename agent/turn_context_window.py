"""Host-owned bounds for overflow retries when a context engine is uncalibrated."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _runtime_key(agent: Any) -> tuple:
    return (agent.model, agent.provider, str(agent.base_url or "").rstrip("/"), agent.api_mode)


def remember_provider_context_limit(agent: Any, limit: int) -> None:
    # Plugin update_model() implementations may ignore the host's correction. Keep the
    # provider's measurement independently, bound to the route that actually rejected it.
    agent._overflow_context_limit = (_runtime_key(agent), limit)


def overflow_recheck_threshold(agent: Any, compressor: Any) -> int:
    """Fallback threshold for an engine that publishes none; zero still fails closed."""
    window = int(getattr(compressor, "context_length", 0) or 0)
    if window <= 0:
        from agent.model_metadata import get_model_context_length

        try:
            window = int(get_model_context_length(
                agent.model, base_url=agent.base_url, provider=agent.provider,
                api_key=agent.api_key if isinstance(agent.api_key, str) else "",
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            ) or 0)
        except Exception:
            logger.debug("Context window unavailable for overflow recheck", exc_info=True)
            window = 0
    reported = getattr(agent, "_overflow_context_limit", None)
    if reported is not None and reported[0] == _runtime_key(agent):
        window = min(window, reported[1]) if window > 0 else reported[1]
    percent = getattr(compressor, "threshold_percent", 0.75)
    if not isinstance(percent, (int, float)) or isinstance(percent, bool) or not 0 < percent <= 1:
        percent = 0.75
    return int(window * percent) if window > 0 else 0
