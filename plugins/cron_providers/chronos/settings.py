"""Where the ``cron.chronos.*`` keys live: the ROOT Hermes home's config, for every profile.

The keys describe the machine, not a profile: this host's public dashboard URL (``callback_url``),
its NAS instance (``expected_audience``), the portal and its JWKS. NAS seeds them into the root
``config.yaml`` only (hosted: ``HERMES_HOME=/opt/data``) and checks them there with ``load_config()``.
A named profile's ``config.yaml`` keeps whatever it held when it was cloned or imported, possibly
another agent's keys or ones from before a host move, so it is never read for them: arming with it
presents a callback URL NAS rejects (400), and verifying with it trusts another instance's identity.

The root comes from ``HERMES_HOME`` (``get_default_hermes_root``), never from the context-local
home override, so a multiplexed ``/p/<profile>/`` request and a process launched as a profile
(``-p <name>``, ``HERMES_HOME=<root>/profiles/<name>``) both read the same file, with the root's
own secrets behind any ``${VAR}`` in it.
"""

from __future__ import annotations

from typing import Any


def chronos_settings() -> dict[str, Any]:
    """The root config's ``cron.chronos`` section (``DEFAULT_CONFIG`` included), whatever profile
    scope or launch profile the caller runs in."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli.config import cfg_get, load_config
    from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override
    from tui_gateway.launch_profile_policy import owner_secret_scope

    root = get_default_hermes_root()
    home_token = set_hermes_home_override(str(root))
    # A ${VAR} in the root config resolves against the root's secrets first, not the caller's
    # profile .env (a multiplexed request's scope, or a process launched as that profile).
    secret_token = set_secret_scope(owner_secret_scope(root), profile_home=str(root))
    try:
        section = cfg_get(load_config(), "cron", "chronos", default=None)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    return section if isinstance(section, dict) else {}
