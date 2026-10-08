"""Model-scoped credential-pool selection for auxiliary clients.

Extracted from ``agent.auxiliary_client`` so that facade stays under its
code-health cap: the scoped-selection funnel added by #130053 lives here and
is re-exported back into ``auxiliary_client`` so behavior and public call
sites are unchanged.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional, Tuple

# Moved code keeps its log source: the warning below was asserted via
# ``agent.auxiliary_client.logger`` (see
# tests/agent/test_auxiliary_codex_model_scoped_selection.py), so log there
# rather than under this module's name.
logger = logging.getLogger("agent.auxiliary_client")


def _accepts_model_scope(fn: Any) -> bool:
    """Whether ``fn`` accepts a ``model`` keyword (or swallows ``**kwargs``).

    Unknown signatures (C builtins without introspectable parameters) count as
    accepting: attempt the scoped call and let genuine errors surface.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD or p.name == "model" for p in params)


def _call_scoped_or_unscoped(fn: Any, *args: Any, model: Optional[str] = None, **kwargs: Any) -> Any:
    """Call ``fn`` scoped by ``model`` when its signature allows it.

    Legacy callees without a ``model`` parameter fall back to an unscoped call
    with a warning. Any other ``TypeError`` is a real bug: it propagates to the
    caller's outer handler (logged, no credential) instead of silently
    downgrading to the credential the scoped path refused (#130053).
    """
    if _accepts_model_scope(fn):
        return fn(*args, model=model, **kwargs)
    if model is not None:
        logger.warning("Auxiliary client: %s does not accept a model scope; falling back unscoped",
                       getattr(fn, "__qualname__", repr(fn)))
    return fn(*args, **kwargs)


def _select_pool_entry(provider: str, model: Optional[str] = None) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    from agent.auxiliary_client import _load_pool_with_credentials

    pool = _load_pool_with_credentials(provider)
    if pool is None:
        return False, None
    try:
        return True, _call_scoped_or_unscoped(pool.select, model=model)
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
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
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None
