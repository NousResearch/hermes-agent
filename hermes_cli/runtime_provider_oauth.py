"""OAuth runtime failures that preserve a renewable session."""

import time

from hermes_cli.auth_constants import AuthError
from hermes_cli.auth_plugin_providers import plugin_refresh_hook


def raise_for_plugin_pool_cooldown(provider, pool, *, model=None):
    """An eligible account waiting for cooldown does not need another sign-in."""
    if plugin_refresh_hook(provider) is None:
        return
    available_at = pool.next_available_at(model=model)
    delay = (available_at or 0) - time.time()
    if delay > 0:
        raise AuthError(
            f"{provider} credentials are temporarily rate-limited; retry after {delay:.0f}s.",
            provider=provider, code="rate_limit_exceeded", retry_after=delay, retryable=True,
        )
