"""First-party Hermes observability integrations."""

from __future__ import annotations

import logging
from typing import Any

# The trace recorder registers its Relay session initializer at import, so it must load with the
# package: every session opened afterwards is recorded from its first scope event.
from . import relay_traces

logger = logging.getLogger(__name__)


def _features() -> tuple[Any, ...]:
    from . import relay_shared_metrics

    return relay_shared_metrics, relay_traces


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Dispatch a Hermes lifecycle event to built-in observability features."""
    for feature in _features():
        try:
            feature.observe_lifecycle(hook_name, **kwargs)
        except Exception:
            logger.warning("Built-in observability hook failed: %s", hook_name, exc_info=True)


def handles_hook(hook_name: str) -> bool:
    """Return whether any built-in observability feature handles a hook."""
    return any(feature.handles_hook(hook_name) for feature in _features())
