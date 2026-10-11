"""Model-scoped credential-pool selection for auxiliary clients.

Extracted from ``agent.auxiliary_client`` so that facade stays under its
code-health cap: the scoped-selection funnel added by #130053 lives here.
``auxiliary_client`` calls it as ``_aux_scope._select_pool_entry`` /
``_peek_pool_entry``; the old ``agent.auxiliary_client._select_pool_entry`` names
were removed, not re-exported, so patch this module instead.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# Moved code keeps its log source: the warning below was asserted via
# ``agent.auxiliary_client.logger`` (see
# tests/agent/test_auxiliary_codex_model_scoped_selection.py), so log there
# rather than under this module's name.
logger = logging.getLogger("agent.auxiliary_client")


def _select_pool_entry(provider: str, model: Optional[str] = None) -> tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    from agent.auxiliary_client import _load_pool_with_credentials

    pool = _load_pool_with_credentials(provider)
    if pool is None:
        return False, None
    try:
        return True, pool.select(model=model)
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc, exc_info=True)
        return True, None


def _peek_pool_entry(provider: str, pool: Any = None) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order.

    ``pool`` skips the disk re-read when the caller already loaded it.
    """
    from agent.auxiliary_client import _load_pool_with_credentials

    if pool is None:
        pool = _load_pool_with_credentials(provider, " (peek)")
    if pool is None:
        return None
    try:
        current_fn = getattr(pool, "current", None)
        current = current_fn() if callable(current_fn) else None
        if current is not None:
            return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc, exc_info=True)
    return None
