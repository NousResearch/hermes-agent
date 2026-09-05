"""First-party Hermes observability integrations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Dispatch a Hermes lifecycle event to built-in observability features."""
    from . import relay_shared_metrics

    try:
        relay_shared_metrics.observe_lifecycle(hook_name, **kwargs)
    except Exception:
        from hermes_cli.lifecycle import log_observer_failure

        log_observer_failure(logger, "Built-in observability hook failed: %s", hook_name)


def handles_hook(hook_name: str) -> bool:
    """Return whether any built-in observability feature handles a hook."""
    from . import relay_shared_metrics

    return relay_shared_metrics.handles_hook(hook_name)
