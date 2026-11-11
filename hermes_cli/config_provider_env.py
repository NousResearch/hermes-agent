"""The provider-profile entries of ``OPTIONAL_ENV_VARS``: descriptive metadata that reads no config.

They used to come from ``list_providers()`` while ``hermes_cli.config`` imported, and provider
discovery reads config (``plugins.enabled``, ``plugins.isolation``): the launch profile's config in
a process serving several profiles, and a deadlock with a config backend whose own load imports
``hermes_cli.auth``. Now the bundled providers' entries are built from :func:`providers.bundled_profiles`
(no config, no discovery) on the table's first read, and discovery's completion hook adds the
process-wide plugin providers it registered.
"""
from __future__ import annotations

_STATIC_NAMES: frozenset[str] = frozenset()  # declared in core (config_defaults), known at import
_PROVIDER_NAMES: set[str] = set()            # added from provider profiles by the fill


def arm(table, static_names) -> None:
    """Called once by ``hermes_cli.config`` at the end of its import."""
    global _STATIC_NAMES
    _STATIC_NAMES = frozenset(static_names)
    table.arm(_provider_entries, _PROVIDER_NAMES.update)


_CATALOG: dict | None = None  # the bundled providers' entries: config-free, computed once


def _entries_for(profiles, skip) -> dict:
    """Entries for the ``auth_type="api_key"`` profiles' env vars, minus ``skip``; the first profile
    to declare a name wins."""
    entries: dict = {}
    for pp in profiles:
        if pp.auth_type != "api_key":
            continue
        for var in pp.env_vars:
            if var in skip or var in entries:
                continue
            is_key = not var.endswith(("_BASE_URL", "_URL"))
            label = pp.display_name or pp.name
            entries[var] = {
                "description": f"{label} {'API key' if is_key else 'base URL override'}",
                "prompt": f"{label} {'API key' if is_key else 'base URL (leave empty for default)'}",
                "url": pp.signup_url or None,
                "password": is_key,
                "category": "provider",
                "advanced": True}
    return entries


def _provider_entries() -> dict:
    """The bundled provider profiles' entries. Reads no config and runs no discovery (so it has no
    partial state and nothing to wait for): a plain idempotent cache, and concurrent first callers
    just compute the same result. A core entry wins; a provider entry replaces one a platform plugin
    manifest added (it was added first only because this runs late)."""
    global _CATALOG
    if _CATALOG is None:
        try:
            from providers import bundled_profiles
            profiles = bundled_profiles()
        except Exception:  # health: allow BLE001 -- as before: a broken provider layer must not break config readers
            profiles = []
        _CATALOG = _entries_for(profiles, _STATIC_NAMES)
    return dict(_CATALOG)


def sync_registered_providers() -> None:
    """Provider discovery's completion hook: add the env vars of process-wide plugin providers it
    registered (entry points enabled in config). Reads the process-wide registry and its source
    record as they stand: never config, never discovery. Not ``provider_source()``: that one is
    profile-aware and scans the bound home's plugin dirs, whose isolation check reads config.
    ``$HERMES_HOME`` plugins (per profile) are not process-wide and are left out."""
    import providers
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
    sources = dict(providers._SOURCES)
    profiles = [p for name, p in list(providers._REGISTRY.items()) if sources.get(name) != "bundled"]
    entries = _entries_for(profiles, _STATIC_NAMES | _PROVIDER_NAMES | set(_CATALOG or ()))
    for var, entry in entries.items():
        dict.__setitem__(OPTIONAL_ENV_VARS, var, entry)
    _PROVIDER_NAMES.update(entries)


def core_declared_env_names() -> frozenset[str]:
    """Names declared in core: config_defaults' plus the provider profiles'. A platform manifest
    never reclassifies one."""
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
    OPTIONAL_ENV_VARS.get("")  # runs the fill once
    return _STATIC_NAMES | frozenset(_PROVIDER_NAMES)
