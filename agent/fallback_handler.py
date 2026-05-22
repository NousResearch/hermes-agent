"""Fallback handler extracted from AIAgent for modularity."""

import logging

logger = logging.getLogger(__name__)


def try_activate_fallback(runner, reason: "FailoverReason | None" = None) -> bool:
    """Forwarder — see ``agent.chat_completion_helpers.try_activate_fallback``."""
    from agent.chat_completion_helpers import try_activate_fallback as _impl
    return _impl(runner, reason)
