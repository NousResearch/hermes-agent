"""First-party Hermes observability integrations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Dispatch a Hermes lifecycle event to built-in observability features."""
    from . import relay_shared_metrics
    from hermes_cli import background_activity

    for observer in (relay_shared_metrics.observe_lifecycle, background_activity.observe_lifecycle):
        try:
            observer(hook_name, **kwargs)
        except Exception:
            logger.warning("Built-in observability hook failed: %s", hook_name, exc_info=True)


def handles_hook(hook_name: str) -> bool:
    """Return whether any built-in observability feature handles a hook."""
    from . import relay_shared_metrics

    return hook_name in {
        "on_session_start", "on_session_end", "on_session_finalize", "on_session_reset",
        "subagent_start", "subagent_stop",
    } or relay_shared_metrics.handles_hook(hook_name)
