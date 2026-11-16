"""Publication of persisted profile credentials into active runtime layers."""

import os
from typing import Optional

from hermes_constants import is_launch_only_env_key


def publish_env_value(key: str, value: Optional[str]) -> None:
    """Refresh the local bootstrap layer without turning a store value into an overlay."""
    from hermes_cli import config
    from agent.secret_scope import current_secret_scope, serves_routed_profile
    from hermes_secret_routing import live_bootstrap_env, routed_provider, stays_local

    try:
        scope, routed = current_secret_scope(), serves_routed_profile()
    except Exception:  # health: allow BLE001 -- base fallback: a scope that cannot resolve publishes ambient
        scope, routed = None, False
    provider = routed_provider()
    if provider is not None and stays_local(provider, key):
        live_bootstrap_env(config.get_hermes_home(), provider)
    targets = [scope] if isinstance(scope, dict) else []
    # Launch-only keys never reach the process env from a .env write (#135200). A bound non-dict
    # scope keeps os.environ untouched with the local store, as before; a store's live scope does not.
    if not routed and not is_launch_only_env_key(key) and (
            scope is None or isinstance(scope, dict) or provider is not None):
        targets.append(os.environ)
    for target in targets:
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value
