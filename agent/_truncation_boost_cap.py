"""Output-token boost ceiling for truncation retries.

The conversation loop retries a truncated response (``finish_reason='length'``)
by progressively raising ``max_tokens``.  A hard-coded ceiling of 32 768 tokens
used to cap that boost — well below the output limits of modern models
(GLM-4.5-Flash: 98 304, GPT-5.x: 131 072, Claude 4.x: 64 000).

When the request already carried an explicit ``max_tokens`` (``requested_cap``)
we honour it as a floor (the boost must always have room to grow).  When it
did not, we fall back to the model's declared ``limit.output`` from the
models.dev cache, then to 32 768 only when even that is unavailable.  This
lets long code blocks / tool arguments finish on the model that already
started them, instead of every retry hitting the same artificial wall and
forcing the agent into the fallback chain.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_FALLBACK_CAP = 32768


def resolve_truncation_boost_cap(
    *,
    requested_cap: Optional[int],
    provider: Optional[str],
    model: Optional[str],
) -> int:
    """Return the highest ``max_tokens`` a truncation-retry may request.

    Parameters
    ----------
    requested_cap:
        The ``max_tokens`` value carried by the original API call, if any.
        When present it is used as the minimum ceiling — the boost is
        allowed to grow above it so retries have room to finish.
    provider, model:
        Used to look up ``limit.output`` from the models.dev cache.
    """
    if requested_cap is not None:
        return max(_FALLBACK_CAP, requested_cap)

    if provider and model:
        try:
            from agent.models_dev import get_model_capabilities

            caps = get_model_capabilities(provider, model)
            if caps is not None:
                model_output = getattr(caps, "max_output_tokens", None)
                if isinstance(model_output, (int, float)) and model_output > 0:
                    return max(_FALLBACK_CAP, int(model_output))
        except Exception:
            logger.warning(
                "Could not resolve model output limit for %s/%s",
                provider,
                model,
                exc_info=True,
            )

    return _FALLBACK_CAP
