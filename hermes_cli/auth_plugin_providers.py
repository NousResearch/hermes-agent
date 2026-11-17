"""Model-provider plugins as first-class citizens of ``hermes auth`` and the credential pool.

A ``plugins/model-providers/<name>/`` profile (``providers.base.ProviderProfile``) is mirrored into
``hermes_cli.auth.PROVIDER_REGISTRY`` with the ``auth_type`` it declares, so ``resolve_provider()``
accepts it whatever its auth shape. Non-api-key plugins own their login through two optional profile
callables that core consults BEFORE any built-in, name-keyed path:

- ``auth_handler(action, args) -> bool`` for ``hermes auth add|status|logout|refresh <name>``
  (``args`` is the parsed CLI namespace; truthy = the plugin owned the action, falsy = built-in path).
- ``refresh_credential(entry) -> Mapping | None`` for the credential pool: given the pooled
  ``PooledCredential`` it returns the rotated fields (``access_token``, ``refresh_token``,
  ``expires_at_ms`` …) or raises. A separate hook rather than ``auth_handler("refresh", …)`` because
  the pool has a credential row, not an argparse namespace, and needs tokens back, not a bool.

Every function here late-imports ``hermes_cli.auth`` names: this module is imported by the auth facade
right after ``PROVIDER_REGISTRY`` exists, and by ``agent.credential_pool``.
"""
from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Providers handled outside the registry: kimi/zai have bespoke token refresh in auth.py;
# openrouter/custom are aggregator/user-supplied and runtime_provider relies on
# ``openrouter not in PROVIDER_REGISTRY``.
_REGISTRY_PLUGIN_SKIP = frozenset({"kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"})

PLUGIN_AUTH_ACTIONS = ("add", "status", "logout", "refresh")

# Names whose PROVIDER_REGISTRY row came from a plugin profile (not the built-in rows). Only these
# can be "OAuth-shaped with nobody to log them in": a bundled OAuth provider (nous, openai-codex …)
# declares the same auth_type but its login lives in core.
PLUGIN_MIRRORED_PROVIDERS: set[str] = set()


_SYNCING = threading.local()  # .active: this thread runs the pending mirror


class HomeScopedRegistry(dict):
    """``PROVIDER_REGISTRY``: process-wide storage, each row read as the bound profile home sees it.

    One process serves many profile homes (multiplex gateway, Desktop ``serve``) and each home
    installs its own provider plugins, so no read returns a row another home rewrote: a mirrored row
    is hidden from a home without that provider and built from the home's own profile when only the
    home declares it; a ``$HERMES_HOME`` plugin's declared fields (:func:`override_registry_row`) land
    on a per-home copy; an alias key follows the bound home's plugin that claims it. Built-in rows a
    home does not override are returned as stored; writes are plain dict writes.
    """

    def _view(self, key: str, row: Any, resolve: Callable[[str], tuple]) -> Any:
        """*row* (stored under *key*) for the bound home; None = hidden there."""
        if key != row.id:
            claimant = resolve(key)[0]
            if claimant is not None and claimant.name != row.id:
                row = dict.get(self, claimant.name)
                if row is None or row.id != claimant.name:
                    return None
        user, shared = resolve(row.id)
        mirrored = row.id in PLUGIN_MIRRORED_PROVIDERS
        if user is None:
            return row if shared is not None or not mirrored else None
        if row.id in _REGISTRY_PLUGIN_SKIP or not (mirrored or row.id in _builtin_ids()):
            return row
        return _home_row(row, user, from_profile=mirrored and shared is None)

    _sync_pending = False

    def sync_before_first_read(self) -> None:
        """Mirror plugin profiles (:func:`sync_plugin_provider_registry`) before the first read."""
        self._sync_pending = True

    def _ready(self) -> None:
        """Run the pending mirror on this read. No lock is held across it (it lists providers,
        which reads config): concurrent first readers each run the idempotent sync. Never while
        ``providers`` is mid-discovery: a partial profile list would mirror a provider that a
        later, higher-priority registration replaces (discovery's completion sync runs then)."""
        if not self._sync_pending or getattr(_SYNCING, "active", False):
            return
        import providers
        if getattr(providers, "_discovering", False):
            return
        _SYNCING.active = True  # the sync reads the registry
        try:
            sync_plugin_provider_registry()
        finally:
            _SYNCING.active = False
        self._sync_pending = False

    def __getitem__(self, key: str) -> Any:
        self._ready()
        row = self._view(key, dict.__getitem__(self, key), _resolver())
        if row is None:
            raise KeyError(key)
        return row

    def get(self, key: str, default: Any = None) -> Any:
        self._ready()
        row = dict.get(self, key)
        row = None if row is None else self._view(key, row, _resolver())
        return default if row is None else row

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None

    def items(self) -> list[tuple[str, Any]]:
        self._ready()
        resolve = _resolver()
        views = ((key, self._view(key, row, resolve)) for key, row in dict.items(self))
        return [(key, row) for key, row in views if row is not None]

    def keys(self) -> list[str]:
        return [key for key, _ in self.items()]

    def values(self) -> list[Any]:
        return [row for _, row in self.items()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.items())


def _resolver() -> Callable[[str], tuple]:
    from providers import bound_profile_resolver
    return bound_profile_resolver()


def _builtin_ids() -> frozenset:
    from hermes_cli.auth import BUILTIN_PROVIDER_IDS
    return BUILTIN_PROVIDER_IDS


# (id(stored row), id(home profile)) -> (stored row, home profile, that home's row). Holding both
# keeps the ids from being reused; a re-imported plugin is a new profile object, so a new entry.
_HOME_ROWS: dict[tuple[int, int], tuple[Any, Any, Any]] = {}


def _home_row(row: Any, pp: Any, *, from_profile: bool) -> Any:
    """The row a home whose own plugin *pp* declares *row*'s provider reads: built from *pp* when no
    process-wide profile backs the row (``from_profile``), else a copy carrying *pp*'s declared fields."""
    cached = _HOME_ROWS.get((id(row), id(pp)))
    if cached is not None:
        return cached[2]
    home = _profile_row(pp) if from_profile else None
    if home is None:
        home = copy.copy(row)
        override_registry_row(home, pp)
    if len(_HOME_ROWS) >= 512:
        _HOME_ROWS.clear()
    _HOME_ROWS[(id(row), id(pp))] = (row, pp, home)
    return home


def _api_key_env_fields(pp: Any) -> tuple[tuple, str]:
    """Split a profile's ``env_vars`` into (api-key vars, base-URL var); the URL var may be ""."""
    is_url = lambda v: v.endswith("_BASE_URL") or v.endswith("_URL")
    return (tuple(v for v in pp.env_vars if not is_url(v)) or pp.env_vars,
            next((v for v in pp.env_vars if is_url(v)), None) or "")


def register_plugin_provider(pp: Any) -> None:
    """Mirror one profile into ``PROVIDER_REGISTRY`` under the ``auth_type`` it declares.

    Registering is what lets an out-of-tree provider pass ``resolve_provider()``'s known-provider
    gate ("Unknown provider"); OAuth-shaped plugins carry their own login via ``auth_handler``.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = _profile_row(pp)
    if pconfig is None:
        return
    _insert_row(PROVIDER_REGISTRY, pconfig, pp.registry_after)
    PLUGIN_MIRRORED_PROVIDERS.add(pp.name)
    mirror_aliases(pconfig, pp)


def _profile_row(pp: Any) -> Optional[Any]:
    """The registry row *pp* declares, or None (a skipped name, an api-key profile without env vars)."""
    from hermes_cli.auth import ProviderConfig, _api_key_provider

    if pp.name in _REGISTRY_PLUGIN_SKIP:
        return None
    auth_type = registry_auth_type(pp)
    if auth_type == "api_key":
        if not pp.env_vars:
            return None
        key_vars, url_var = _api_key_env_fields(pp)
        return _api_key_provider(pp.name, pp.display_name or pp.name, pp.base_url, key_vars,
                                 pp.base_url_env_var or url_var)
    return ProviderConfig(pp.name, pp.display_name or pp.name, auth_type, inference_base_url=pp.base_url,
                          base_url_env_var=pp.base_url_env_var)


def registry_auth_type(pp: Any) -> str:
    """The ``auth_type`` *pp*'s registry row carries (``registry_auth_type`` wins over ``auth_type``)."""
    return pp.registry_auth_type or pp.auth_type


def _insert_row(registry: dict, pconfig: Any, after: str) -> None:
    """Add *pconfig* right after the *after* row (in place: callers hold the dict), else append.

    Reads the raw storage: rows hidden from the bound home must survive the rebuild."""
    if not dict.__contains__(registry, after):
        registry[pconfig.id] = pconfig
        return
    rows = list(dict.items(registry))
    at = [key for key, _ in rows].index(after) + 1
    registry.clear()
    registry.update(rows[:at] + [(pconfig.id, pconfig)] + rows[at:])


def _user_owns_alias(pp: Any, alias: str) -> bool:
    """True when *alias* resolves to the ``$HERMES_HOME`` plugin *pp* in the ``providers`` layer."""
    try:
        from providers import get_provider_profile, provider_source
    except Exception:
        return False
    owner = get_provider_profile(alias)
    return owner is not None and owner.name == pp.name and provider_source(pp.name) == "user"


def mirror_aliases(pconfig: Any, pp: Any) -> None:
    """Point ``pp.aliases`` at *pconfig* so ``resolve_provider()`` resolves them too.

    A bundled plugin never steals an alias another row already holds; a ``$HERMES_HOME`` plugin
    whose alias the ``providers`` layer already resolves to it does — the same ownership rule as
    :func:`override_registry_row`, otherwise the alias kept resolving to the built-in row while
    ``providers.get_provider_profile`` followed the user's profile (#116668).
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    for alias in pp.aliases:
        if alias not in PROVIDER_REGISTRY or _user_owns_alias(pp, alias):
            PROVIDER_REGISTRY[alias] = pconfig


def override_registry_row(pconfig: Any, pp: Any) -> None:
    """A ``$HERMES_HOME`` plugin re-registering a name that already has a row wins for the fields
    it declares — ``display_name``, ``base_url`` and, on api-key rows, ``env_vars`` (#48450, #116668). ``register_provider()``
    is last-writer-wins for the profile; without this the runtime kept reading the built-in
    endpoint. A ``$HERMES_HOME`` layer plugin's fields land on that home's own copy of the row
    (:class:`HomeScopedRegistry`), so another served home keeps the built-in endpoint; a process-wide
    registration (every home reads it) rewrites the stored row in place.
    """
    if pp.display_name:
        pconfig.name = pp.display_name
    if pp.base_url:
        pconfig.inference_base_url = pp.base_url
    if registry_auth_type(pp) == "api_key" == pconfig.auth_type and pp.env_vars:
        pconfig.api_key_env_vars, url_var = _api_key_env_fields(pp)
        if url_var:
            pconfig.base_url_env_var = url_var


def sync_plugin_provider_registry() -> int:
    """Mirror provider-plugin profiles into ``PROVIDER_REGISTRY``; return how many were added.

    Idempotent (existing entries are never replaced — a user plugin re-registering a bundled name
    wins for the fields it declares in its own home only, see :class:`HomeScopedRegistry`), so it is safe from
    resolution paths. It runs at
    auth import and again whenever a name is missing (:func:`registry_lookup`) or when ``providers``
    finishes discovery, because the import-time pass can observe a *partial* profile list: a plugin
    whose own imports pull ``hermes_cli.auth`` in mid-``_discover_providers()`` sees only what was
    registered so far, and every later plugin would otherwise fail with "Unknown provider" (#102123).
    """
    from hermes_cli.auth import BUILTIN_PROVIDER_IDS, PROVIDER_REGISTRY

    try:
        from providers import list_providers, provider_source
        profiles = list_providers()
        home_profile = lambda name: _resolver()(name)[0]
    except Exception:
        return 0
    added = 0
    for pp in profiles:
        if pp.name in PROVIDER_REGISTRY:
            # Only rows core wrote (built-in or mirrored) — a row the plugin injected itself is its
            # own, more specific declaration and stays as written.
            core_row = pp.name in BUILTIN_PROVIDER_IDS or pp.name in PLUGIN_MIRRORED_PROVIDERS
            if core_row and pp.name not in _REGISTRY_PLUGIN_SKIP and provider_source(pp.name) == "user":
                row = dict.__getitem__(PROVIDER_REGISTRY, pp.name)
                if home_profile(pp.name) is not pp:  # registered process-wide: every home reads it
                    override_registry_row(row, pp)
                mirror_aliases(row, pp)
            continue
        register_plugin_provider(pp)
        added += pp.name in PROVIDER_REGISTRY
    return added


def canonical_registry_rows() -> list[tuple[str, Any]]:
    """``PROVIDER_REGISTRY`` items in order, minus the alias keys :func:`mirror_aliases` points at a
    plugin's row, so a scan or listing sees each provider once, under its own id."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    return [(pid, pconfig) for pid, pconfig in PROVIDER_REGISTRY.items() if pid == pconfig.id]


def registry_lookup(provider_id: str) -> Optional[Any]:
    """``PROVIDER_REGISTRY.get`` that re-syncs plugin profiles on a miss."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if pconfig is None:
        # Re-read whatever the count says: a home's first plugin scan runs inside the sync's own
        # ``list_providers()`` and mirrors its rows in that nested pass.
        sync_plugin_provider_registry()
        pconfig = PROVIDER_REGISTRY.get(provider_id)
    return pconfig


def plugin_profile(provider: str) -> Optional[Any]:
    """The registered ``ProviderProfile`` for *provider*, or None (also when the layer is unavailable)."""
    try:
        from providers import get_provider_profile
    except Exception:
        return None
    return get_provider_profile(provider)


def _profile_hook(provider: str, name: str) -> Optional[Callable[..., Any]]:
    hook = getattr(plugin_profile(provider), name, None)
    return hook if callable(hook) else None


def plugin_auth_handler(provider: str) -> Optional[Callable[[str, Any], Any]]:
    return _profile_hook(provider, "auth_handler")


def plugin_refresh_hook(provider: str) -> Optional[Callable[[Any], Any]]:
    """The profile's ``refresh_credential`` hook, i.e. whether its pooled OAuth rows are refreshable."""
    return _profile_hook(provider, "refresh_credential")


def is_refreshable_oauth_provider(provider: str) -> bool:
    """Built-in refreshable set OR a plugin profile shipping ``refresh_credential``."""
    from agent.credential_pool import REFRESHABLE_OAUTH_PROVIDERS

    return provider in REFRESHABLE_OAUTH_PROVIDERS or plugin_refresh_hook(provider) is not None


def dispatch_plugin_auth(action: str, args: Any, provider: str) -> bool:
    """Offer ``hermes auth <action> <provider>`` to the provider's ``auth_handler``.

    True = the plugin owned the action (core prints nothing more). False = run the built-in path. A
    handler exception becomes a readable ``SystemExit`` naming provider and action.
    ``<plugin>:<id>`` names an OAuth client a plugin declares under ``oauth:`` (:mod:`hermes_cli.plugin_oauth`).
    """
    from hermes_cli.plugin_oauth import handle_auth_action

    owned = handle_auth_action(action, args, provider)
    if owned is not None:
        return owned
    handler = plugin_auth_handler(provider)
    if handler is None:
        return False
    try:
        return bool(handler(action, args))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"{provider} auth handler failed for `{action}`: {type(exc).__name__}: {exc}") from exc


def plugin_missing_auth_handler_error(provider: str, action: str) -> Optional[SystemExit]:
    """Fail loud for a registered non-api-key plugin that ships no ``auth_handler``.

    Its login is not something core can perform (there is no token endpoint to call), and silently
    running the api-key prompt or reporting "Unknown provider" both hide the plugin bug.
    """
    if provider not in PLUGIN_MIRRORED_PROVIDERS:
        return None
    profile = plugin_profile(provider)
    if profile is None or registry_auth_type(profile) == "api_key" or plugin_auth_handler(provider) is not None:
        return None
    return SystemExit(
        f"Provider '{provider}' declares auth_type '{profile.auth_type}' but its plugin ships no "
        f"auth_handler, so `hermes auth {action} {provider}` cannot be handled. Add "
        "`auth_handler=` to its ProviderProfile (see the model-provider plugin guide).")


def _pool_entry_expired(entry: Any) -> bool:
    """A pooled OAuth row is expired when its ``expires_at_ms`` / ISO ``expires_at`` is in the past."""
    import time
    from hermes_cli.auth import _parse_iso_timestamp

    if entry.expires_at_ms is not None:
        return int(entry.expires_at_ms) <= int(time.time() * 1000)
    if entry.expires_at:
        epoch = _parse_iso_timestamp(entry.expires_at)
        return epoch is not None and epoch <= time.time()
    return False


def get_plugin_oauth_auth_status(provider_id: str) -> dict[str, Any]:
    """Status for an OAuth-shaped PLUGIN provider, read from the credential pool its ``auth_handler`` fills.

    ``configured`` = the profile is registered; ``logged_in`` = a pool row carries a live token;
    ``needs_refresh`` = every token is expired but refresh material (and a ``refresh_credential`` hook)
    exists. Bundled OAuth providers keep their bespoke builders (``_BESPOKE_STATUS_FUNCTIONS``) — this
    one is gated on the plugin-mirrored set so their status bytes never change.
    """
    if provider_id not in PLUGIN_MIRRORED_PROVIDERS:
        return {"logged_in": False}
    from agent.credential_pool import load_pool

    entries = [e for e in load_pool(provider_id).entries() if (e.access_token or e.agent_key or "").strip()]
    live = [e for e in entries if not _pool_entry_expired(e)]
    refreshable = [e for e in entries if e.refresh_token] if not live and plugin_refresh_hook(provider_id) else []
    return {
        "configured": True, "provider": provider_id, "logged_in": bool(live),
        "needs_refresh": bool(refreshable), "accounts": len(entries),
        "base_url": next((e.base_url for e in live + refreshable if e.base_url), "") or "",
        "hint": "" if live else f"Run `hermes auth add {provider_id}` to sign in."}
