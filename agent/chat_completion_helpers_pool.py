"""Fallback credential-pool exhaustion detail, split out of ``chat_completion_helpers``.

Extracted so ``agent.chat_completion_helpers`` stays under its code-health
FILE_LINES cap: the helper is re-imported back into the facade for its internal
caller and for the tests that import it from there.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from agent.retry_utils import RETRY_AFTER_CAP_S

logger = logging.getLogger("agent.chat_completion_helpers")


def _pool_exhaustion_detail(agent, fb_provider: str, fb_model: str) -> Optional[str]:
    """None when the candidate's credential pool is usable. Otherwise a short reason
    token distinguishing the two ways the pool can be unusable: "cooldown" when every
    entry sits in an exhaustion cooldown longer than the retry loop's longest wait
    (the 600s Retry-After cap), versus "no-wait-info" when nothing reports a recovery
    time at all — an unfilled borrowed row or empty-looking pool, where no entry is
    actually in cooldown (#131993). The skip decision is the same either way; only
    the log message differs."""
    pool = getattr(agent, "_credential_pool", None)
    if pool is None or (getattr(pool, "provider", "") or "").strip().lower() != fb_provider:
        try:
            from agent.credential_pool import load_pool
            pool = load_pool(fb_provider)
        except Exception:
            logger.debug("fallback pool read failed", exc_info=True)
            return None
    if pool is None or not pool.has_credentials() or pool.has_available(model=fb_model):
        return None
    until = pool.next_available_at(model=fb_model)
    if until is None:
        return "no-wait-info"
    if until - time.time() > RETRY_AFTER_CAP_S:
        return "cooldown"
    return None
