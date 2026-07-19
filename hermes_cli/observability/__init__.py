"""First-party Hermes observability integrations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def prepare_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Prepare subscribers that must observe a session's start event."""
    from . import relay_runtime, relay_shared_metrics

    if hook_name in relay_runtime.SESSION_START_HOOKS:
        try:
            relay_shared_metrics.prepare_session_start()
        except Exception:
            logger.warning("Built-in observability preparation failed", exc_info=True)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Dispatch a Hermes lifecycle event to built-in observability features."""
    from . import relay_runtime, relay_shared_metrics

    # Session-start plugin callbacks register optional per-session subscribers
    # before this completion step opens the shared core scope. On teardown,
    # metrics finish child LLM scopes before the neutral host closes the owner.
    if hook_name not in relay_runtime.SESSION_CLOSE_HOOKS:
        _safe_observe(relay_runtime.observe_lifecycle, hook_name, kwargs)
    _safe_observe(relay_shared_metrics.observe_lifecycle, hook_name, kwargs)
    if hook_name in relay_runtime.SESSION_CLOSE_HOOKS:
        _safe_observe(relay_runtime.observe_lifecycle, hook_name, kwargs)


def handles_hook(hook_name: str) -> bool:
    """Return whether any built-in observability feature handles a hook."""
    from . import relay_runtime, relay_shared_metrics

    return relay_runtime.handles_hook(hook_name) or relay_shared_metrics.handles_hook(
        hook_name
    )


def _safe_observe(callback: Any, hook_name: str, kwargs: dict[str, Any]) -> None:
    try:
        callback(hook_name, **kwargs)
    except Exception:
        logger.warning(
            "Built-in observability hook failed: %s", hook_name, exc_info=True
        )
