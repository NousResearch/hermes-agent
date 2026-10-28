"""The provider-profile entries of ``OPTIONAL_ENV_VARS``, added on first read instead of at import.

Listing provider profiles runs provider discovery, and discovery reads config: ``plugins.enabled``
gates pip-installed provider plugins, ``plugins.isolation`` decides how ``$HERMES_HOME`` ones load.
Done while ``hermes_cli.config`` imports, that read is the launch profile's config in a process that
serves several profiles, and a cycle with a config backend that imports this code while it loads the
config (the remote backend's Nous credential imports ``hermes_cli.auth``, which imports
``hermes_cli.config``). So importing ``hermes_cli.config`` reads no config; the first read of the
table (or of ``CORE_DECLARED_ENV_NAMES``) adds the entries.
"""
from __future__ import annotations

_STATIC_NAMES: frozenset[str] = frozenset()  # declared in core (config_defaults), known at import
_PROVIDER_NAMES: set[str] = set()            # added from provider profiles by the fill


def arm(table, static_names) -> None:
    """Called once by ``hermes_cli.config`` at the end of its import."""
    global _STATIC_NAMES
    _STATIC_NAMES = frozenset(static_names)
    table.arm(_add_provider_entries)


def _add_provider_entries(table) -> None:
    """Expose the env vars of every ``auth_type="api_key"`` provider profile. A core entry wins;
    a provider entry replaces one a platform plugin manifest added (it was added first only
    because this runs late)."""
    try:
        from providers import list_providers
        profiles = list_providers()
    except Exception:  # health: allow BLE001 -- as before: a broken provider layer must not break config readers
        # (ConfigBackendUnavailable is a SystemExit: it propagates, and the table re-arms)
        return
    for pp in profiles:
        if pp.auth_type != "api_key":
            continue
        for var in pp.env_vars:
            if var in _STATIC_NAMES or var in _PROVIDER_NAMES:
                continue
            is_key = not var.endswith(("_BASE_URL", "_URL"))
            label = pp.display_name or pp.name
            dict.__setitem__(table, var, {
                "description": f"{label} {'API key' if is_key else 'base URL override'}",
                "prompt": f"{label} {'API key' if is_key else 'base URL (leave empty for default)'}",
                "url": pp.signup_url or None,
                "password": is_key,
                "category": "provider",
                "advanced": True})
            _PROVIDER_NAMES.add(var)


def core_declared_env_names() -> frozenset[str]:
    """Names declared in core: config_defaults' plus the provider profiles'. A platform manifest
    never reclassifies one."""
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
    OPTIONAL_ENV_VARS.get("")  # runs the fill once
    return _STATIC_NAMES | frozenset(_PROVIDER_NAMES)
