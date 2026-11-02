"""Provider module registry.

Provider profiles can live in three places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``
3. Pip-installed plugins: distributions exposing a ``hermes_agent.plugins``
   entry point (``module:func`` callable or a self-registering ``module``)

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first call to ``get_provider_profile()`` or
``list_providers()`` imports the bundled and pip-installed plugins once per
process; the ``$HERMES_HOME`` plugins of the profile home bound at lookup time
load into that home's own layer, so one process serving several profiles
(multiplex gateway, Desktop ``serve``) resolves each profile's installs. User
plugins override bundled plugins on name collision, so third parties can
monkey-patch or replace any built-in profile without editing the repo.

For backward compatibility, ``providers/*.py`` files (other than ``base.py``
and ``__init__.py``) are still discovered via ``pkgutil.iter_modules``.
This lets out-of-tree users drop a single-file profile into an editable
install without the plugin dir structure. New profiles should prefer the
plugin layout.

Usage::

    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# Process-wide layer: bundled plugins, pip entry points, legacy ``providers/<name>.py``.
_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
# Where the CURRENT registration of each name came from: "bundled" / "user" (a
# ``$HERMES_HOME`` plugin dir) / "runtime" (entry point, legacy module, direct call).
_SOURCES: dict[str, str] = {}
_current_source: str | None = None
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
# Bumped by every registration (process-wide or per-home layer): memos over the registry key on it.
_REGISTRY_GENERATION = 0
_discovered = False
_discovering = False
_PLUGIN_DIR_STAMP_TTL_SECONDS = 1.0
# Widest directory mtime granularity in use (FAT rounds to 2 s, HFS+ to 1 s; a kernel without
# fine-grained timestamps stamps every change within one ~4 ms clock tick alike).
_PLUGIN_DIR_MTIME_TICK_NS = 2_000_000_000


@dataclass
class _HomeLayer:
    """The ``$HERMES_HOME/plugins`` providers of ONE profile home.

    One process serves many profiles (multiplex gateway, Desktop ``serve``) and each profile installs
    its own plugins, so user plugins are keyed by the home bound at lookup time instead of the home
    that happened to be bound at first discovery — that made a plugin installed in a secondary
    profile ``Unknown provider`` in Desktop while the same profile worked in a terminal (#88143).
    ``stamps`` are the plugin dirs' mtimes: a directory added by ``hermes plugins install`` while the
    process runs changes them, and the next lookup imports it without a restart.
    """
    registry: dict[str, ProviderProfile] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    stamps: tuple = ()
    stamp_checked_at: float | None = None
    # name -> when its miss last forced a stamp check (see ``_miss_recheck_due``)
    misses: dict[str, float] = field(default_factory=dict)
    # plugin dir name -> the error its import raised (``plugin_load_error``), until an import succeeds
    errors: dict[str, str] = field(default_factory=dict)
    # module names of plugins whose profiles this layer already holds as plugin-host proxies: the
    # hosted path's ``sys.modules`` check, so a rescan never re-imports, re-warns or swaps proxies
    hosted: set[str] = field(default_factory=set)


_HOME_LAYERS: dict[str, _HomeLayer] = {}
_HOME_LAYERS_LOCK = threading.Lock()
# The layer a ``$HERMES_HOME`` plugin import registers into. A ContextVar, not a module global:
# two turn threads scanning two profile homes at once must not cross-register. Never a lock held
# across the import itself — a thread mid-``import hermes_cli.auth`` (whose import calls
# ``list_providers()``) would block on it while the scanning thread waits on that module's import lock.
_REGISTRATION_TARGET: ContextVar[_HomeLayer | None] = ContextVar("_provider_registration_target", default=None)
# Set by :func:`bundled_profiles`: registrations made in this context are collected, nothing else.
_COLLECT: ContextVar[list | None] = ContextVar("_provider_collect", default=None)
# Profile names an in-process import may register (None = all). Set while a hosted plugin's
# profiles that cannot cross the host boundary are imported here: its hosted siblings stay hosted.
_REGISTRATION_ONLY: ContextVar[frozenset[str] | None] = ContextVar("_provider_registration_only", default=None)

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)


def _sync_auth_registry() -> None:
    """Mirror profiles into the ``hermes_cli`` snapshots (auth registry, picker catalog) that are loaded.

    ``hermes_cli.auth`` takes its own snapshot of ``list_providers()`` when it is imported. If a
    plugin's imports pull that module in while :func:`_discover_providers` is still running, the
    snapshot is partial and later plugins never reach the auth registry ("Unknown provider",
    #102123). Calling back into auth once discovery is complete closes that window. Looked up via
    ``sys.modules`` on purpose: this layer must never import ``hermes_cli`` (that would run auth's
    top-level code mid-scan and risk a circular import). Never raises: registration must not fail
    because of the auth mirror.
    """
    for module, attr in (
        ("hermes_cli.auth", "sync_plugin_provider_registry"),
        ("hermes_cli.models_catalog_static", "sync_plugin_provider_catalog"),
        ("hermes_cli.config_provider_env", "sync_registered_providers"),
    ):
        sync = getattr(sys.modules.get(module), attr, None)
        if sync is None:
            continue
        try:
            sync()
        except Exception as exc:  # pragma: no cover — never break discovery
            logger.debug("%s sync skipped: %s", module, exc)


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code. A registration made while a
    ``$HERMES_HOME`` plugin is being imported lands in that home's layer.
    """
    global _PROVIDER_LIST_CACHE, _REGISTRY_GENERATION
    collect = _COLLECT.get()
    if collect is not None:
        collect.append(profile)
        return
    only = _REGISTRATION_ONLY.get()
    if only is not None and profile.name not in only:
        return
    _REGISTRY_GENERATION += 1
    layer = _REGISTRATION_TARGET.get()
    if layer is not None:
        layer.registry[profile.name] = profile
        for alias in profile.aliases:
            layer.aliases[alias] = profile.name
    else:
        _REGISTRY[profile.name] = profile
        _SOURCES[profile.name] = _current_source or "runtime"
        for alias in profile.aliases:
            _ALIASES[alias] = profile.name
        _PROVIDER_LIST_CACHE = None
    if _discovered and not _discovering:  # post-discovery registration: mirror it immediately
        _sync_auth_registry()


def provider_source(name: str) -> str | None:
    """Discovery source of the profile currently registered under *name* (see ``_SOURCES``), or None.

    ``"user"`` is what lets a ``$HERMES_HOME`` plugin re-registering a bundled name win in
    ``hermes_cli.auth.PROVIDER_REGISTRY`` too — a bundled profile never rewrites a built-in row.
    """
    layer = _home_layer()
    canonical = layer.aliases.get(name) or _ALIASES.get(name, name)
    if canonical in layer.registry:
        return "user"
    return _SOURCES.get(canonical)


# ``has_named_custom_provider`` walks the provider registry (``_shadowed_by_builtin`` →
# ``resolve_provider``, which formats an AuthError for unregistered names) and loads config
# on every call — ~2.5 ms per miss (Enough1122's profile on #120901, 40-50x the old miss
# path). This sits on the profile-miss path that ``model_metadata._strip_provider_prefix``
# reaches per model/prefix, so it must not be paid per iteration. Memo per home keyed on
# the config file's ``file_signature`` — the same invalidation signal ``load_config`` uses —
# so an edit re-arms the next lookup, and multiplex homes never borrow each other's answers.
_NAMED_CUSTOM_MEMO: dict[str, dict[str, bool]] = {}
_NAMED_CUSTOM_MEMO_SIG: dict[str, Any] = {}
_NAMED_CUSTOM_MEMO_NO_SIG = object()


def _has_named_custom_provider(name: str, home: Path | None, hkey: str) -> bool:
    """``has_named_custom_provider`` memoized on the home's config cache version."""
    from hermes_cli.config_backend import config_cache_version

    try:
        sig = config_cache_version(home / "config.yaml") if home is not None else None
    except OSError:
        sig = None  # no readable config: nothing configured, stable until one appears
    memo = _NAMED_CUSTOM_MEMO.get(hkey)
    if memo is None or _NAMED_CUSTOM_MEMO_SIG.get(hkey, _NAMED_CUSTOM_MEMO_NO_SIG) != sig:
        memo = {}
        _NAMED_CUSTOM_MEMO[hkey] = memo
        _NAMED_CUSTOM_MEMO_SIG[hkey] = sig
    if name in memo:
        return memo[name]
    from hermes_cli.runtime_provider_custom import has_named_custom_provider

    memo[name] = has_named_custom_provider(name)
    return memo[name]


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return None
    if not _discovered:
        _discover_providers()
    layer, home, key = _bound_home_layer()
    checked = _refresh_home_layer(layer, home, key)

    def lookup(n: str) -> ProviderProfile | None:
        canonical = layer.aliases.get(n) or _ALIASES.get(n, n)
        return layer.registry.get(canonical) or _REGISTRY.get(canonical)

    profile = lookup(name)
    # A newly installed provider is normally first requested by its new name, so a miss
    # re-checks the plugin dirs now (unless this call just did) instead of waiting for the
    # periodic check. ``custom:<route>`` misses resolve to the generic profile below and the
    # picker asks for them once per model, so they wait for the periodic check.
    is_custom_route = isinstance(name, str) and name.lower().startswith("custom:")
    if profile is None and not is_custom_route and not checked and _miss_recheck_due(layer, name):
        if _refresh_home_layer(layer, home, key, force=True):
            profile = lookup(name)
    # Named custom routes share the generic wire policy unless a plugin
    # explicitly registered that route. Other names retain exact lookup.
    if profile is None and is_custom_route:
        profile = lookup("custom")
    # Bare named custom providers (not ``custom:<name>``) that are configured in
    # ``providers:`` / ``custom_providers:`` but lack an explicit profile entry
    # also share the generic wire policy. The helper is memoized (see
    # ``_has_named_custom_provider``) so the miss path stays cheap for per-model loops.
    if profile is None and isinstance(name, str) and not is_custom_route:
        if _has_named_custom_provider(name, home, key):
            profile = lookup("custom")
    return profile


def _miss_recheck_due(layer: _HomeLayer, name: str) -> bool:
    """Whether a miss for *name* may force a plugin-dir check now: once per name per stamp TTL.

    Registry ids with no profile (``openai-api``, ``xai-oauth`` …) are looked up on every request;
    re-statting for each defeated the TTL. A new name still forces at once, so a just-installed
    provider is found on its first request; a repeat miss waits for the periodic check.
    """
    now = time.monotonic()
    last = layer.misses.get(name)
    if last is not None and now - last < _PLUGIN_DIR_STAMP_TTL_SECONDS:
        return False
    if len(layer.misses) >= 512:
        layer.misses.clear()
    layer.misses[name] = now
    return True


def routed_model_rejects_vision_tool_messages(provider: str, model: str) -> bool:
    """Whether an active route or its aggregator-targeted model rejects image tool parts.

    Routing aggregators such as ``openrouter`` send vendor-prefixed model IDs
    (for example, ``xiaomi/mimo-v2.5``), but their own profile cannot describe
    every routed provider's tool-message compatibility. Preserve the transport
    profile as the default and consult a registered target profile only for
    routing aggregators. Missing or unrecognized identities deliberately fail open.
    """
    provider_name = str(provider or "").strip().lower()
    profile = get_provider_profile(provider_name)
    if profile is not None and profile.supports_vision_tool_messages is False:
        return True
    # Routing aggregators accept a ``vendor/model`` identifier while the request is sent
    # to the aggregator; the target provider can have stricter message-shape support than
    # the aggregator's generic OpenAI-compatible transport profile.
    from hermes_cli.providers import is_routing_aggregator
    if not is_routing_aggregator(provider_name):
        return False

    target_name, separator, _ = str(model or "").strip().partition("/")
    if not separator or not target_name:
        return False
    target_profile = get_provider_profile(target_name.strip().lower())
    return target_profile is not None and target_profile.supports_vision_tool_messages is False


_HOST_PROFILE_MEMO: dict[tuple, tuple[tuple, ProviderProfile | None]] = {}


def registry_generation() -> int:
    """Counter bumped by every provider registration; key memos of profile answers on it."""
    return _REGISTRY_GENERATION


def bound_registry_key() -> tuple[str, int]:
    """``(bound home key, registry generation)`` after the periodic plugin-dir check.

    Key for per-home memos of profile-derived answers: each profile home has its own plugins, and
    any registration (a plugin installed while the process runs) moves the generation.
    """
    if not _discovered:
        _discover_providers()
    layer, home, key = _bound_home_layer()
    _refresh_home_layer(layer, home, key)
    return key, _REGISTRY_GENERATION


def _host_claims(host: str, claimed: str) -> bool:
    claimed = str(claimed or "").strip().lower().rstrip(".")
    return bool(claimed) and (host == claimed or host.endswith("." + claimed))


def profile_for_host(url: str, *, metadata: bool = False) -> ProviderProfile | None:
    """The registered profile whose declared ``hostnames`` match *url*'s host (exact or subdomain).

    *metadata* also matches ``metadata_hostnames``. Resolved against the bound home's providers, so a
    ``$HERMES_HOME`` plugin installed after import (or in another profile) is seen; memoized per
    (home, host) until the registry changes, because wire paths ask on every request.
    """
    from agent.safe_worker_policy import safe_worker_enabled
    from utils import base_url_hostname

    host = base_url_hostname(url) if url else ""
    if not host or safe_worker_enabled():
        return None
    if not _discovered:
        _discover_providers()
    layer, home, key = _bound_home_layer()
    _refresh_home_layer(layer, home, key)
    token = (_REGISTRY_GENERATION, id(_REGISTRY), len(_REGISTRY), len(layer.registry))
    memo_key = (key, host, metadata)
    cached = _HOST_PROFILE_MEMO.get(memo_key)
    if cached is not None and cached[0] == token:
        return cached[1]
    found = None
    for profile in list_providers():
        hosts = (*profile.hostnames, *profile.metadata_hostnames) if metadata else profile.hostnames
        if any(_host_claims(host, claimed) for claimed in hosts):
            found = profile
            break
    _HOST_PROFILE_MEMO[memo_key] = (token, found)
    return found


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name); the bound home's
    ``$HERMES_HOME`` plugins shadow process-wide profiles of the same name."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return []
    global _PROVIDER_LIST_CACHE
    if not _discovered:
        _discover_providers()
    layer = _home_layer()
    if _PROVIDER_LIST_CACHE is None:
        # Deduplicate: _REGISTRY has canonical names; _ALIASES points to same objects
        seen: set[int] = set()
        cache: list[ProviderProfile] = []
        for profile in _REGISTRY.values():
            if id(profile) not in seen:
                seen.add(id(profile))
                cache.append(profile)
        _PROVIDER_LIST_CACHE = cache
    result = [p for p in _PROVIDER_LIST_CACHE if p.name not in layer.registry]
    result.extend({id(p): p for p in layer.registry.values()}.values())
    return result


def unlisted_provider_names() -> set[str]:
    """Registered profiles that discovery surfaces (provider pickers, setup lists, the accounts tab)
    must not offer: pre-release ones the user has not opted into.

    Resolution never consults this: an unlisted profile still resolves, authenticates and serves
    turns by name. A raising ``listed()`` counts as unlisted, because a broken gate must never
    advertise a dark launch.
    """
    hidden: set[str] = set()
    for profile in list_providers():
        try:
            if not profile.listed():
                hidden.add(profile.name)
        except Exception:
            logger.debug("provider %s listed() raised; hiding it", profile.name, exc_info=True)
            hidden.add(profile.name)
    return hidden


def bound_profile_resolver():
    """``resolve(name) -> (home plugin profile, process-wide profile)`` for the home bound now.

    *name* may be an alias. For process-wide mirrors read per home (``hermes_cli.auth``'s
    ``PROVIDER_REGISTRY``), so cheap enough per lookup: no discovery, no miss re-scan; only the
    periodic plugin-dir stamp check, skipped mid-scan so a plugin reading the mirror while it imports
    cannot re-enter its own scan.
    """
    layer = _bound_home_layer()[0] if _discovering else _home_layer()

    def resolve(name: str) -> tuple[ProviderProfile | None, ProviderProfile | None]:
        shared = _ALIASES.get(name, name)
        return layer.registry.get(layer.aliases.get(name) or shared), _REGISTRY.get(shared)
    return resolve


def _home_layer(*, force_stamp_check: bool = False) -> _HomeLayer:
    """The layer for the home bound right now, importing plugin dirs it has not seen yet."""
    layer, home, key = _bound_home_layer()
    _refresh_home_layer(layer, home, key, force=force_stamp_check)
    return layer


def refresh_bound_home() -> None:
    """Re-scan the bound home's plugin dirs now and forget its remembered misses: a plugin was just
    installed there, and the periodic check (or a name's miss back-off) would hide it for up to a TTL."""
    _home_layer(force_stamp_check=True).misses.clear()


def _bound_home_layer() -> tuple[_HomeLayer, Path | None, str]:
    try:
        from hermes_constants import get_hermes_home, hermes_home_key

        home = Path(get_hermes_home())
        key = hermes_home_key(home)
    except Exception:
        home, key = None, ""
    with _HOME_LAYERS_LOCK:
        layer = _HOME_LAYERS.get(key)
        if layer is None:
            layer = _HOME_LAYERS[key] = _HomeLayer()
    return layer, home, key


def _refresh_home_layer(layer: _HomeLayer, home: Path | None, key: str, *, force: bool = False) -> bool:
    """Re-stat the layer's plugin dirs when due (or *force*d); True when it stat'ed this call.

    Stamps are read before the scan: a plugin that lands mid-scan changes them and the next
    check picks it up. Checking on a short cadence keeps a newly installed plugin discoverable
    without making every model lookup perform two filesystem stats.
    """
    now = time.monotonic()
    from agent.safe_worker_policy import safe_worker_enabled

    # Safe mode fires zero user customization: same gate as ``_discover_providers``.
    if home is None or safe_worker_enabled() or not (
        force
        or layer.stamp_checked_at is None
        or now - layer.stamp_checked_at >= _PLUGIN_DIR_STAMP_TTL_SECONDS
    ):
        return False
    stamps = _plugin_dir_stamps(home)
    if stamps != layer.stamps:
        # Judge the stamps' age before scanning: a slow scan must not let a stamp read before it
        # look settled afterwards and get remembered over an install that landed mid-scan.
        settled = _plugin_dir_stamps_settled(stamps)
        _scan_home_layer(layer, key)
        # A stamp still inside its mtime tick may not move for an install landing in that same
        # tick, which would hide the plugin until some later change: keep it unremembered so the
        # next periodic check scans again, until the stamps settle.
        layer.stamps = stamps if settled else ()
        # Publish the completed layer -- stamps AND check time -- before auth
        # sync: it calls list_providers(), which re-enters this function. An
        # unpublished stamp rescanned forever; an unpublished check time
        # re-stats the plugin dirs inside the TTL.
        layer.stamp_checked_at = now
        if _discovered and not _discovering:
            _sync_auth_registry()
    layer.stamp_checked_at = now
    return True


def _plugin_dir_stamps(home: Path) -> tuple:
    """mtimes of ``plugins/`` and ``plugins/model-providers/``: they change when a child is added."""
    def stamp(path: Path):
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return None
    return (stamp(home / "plugins"), stamp(home / "plugins" / "model-providers"))


def _plugin_dir_stamps_settled(stamps: tuple) -> bool:
    """Whether every stamp is at least one mtime tick away from now (either side: clock skew)."""
    now = time.time_ns()
    return all(s is None or abs(now - s) >= _PLUGIN_DIR_MTIME_TICK_NS for s in stamps)


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _installed_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/`` if it exists.

    This is where ``hermes plugins install`` clones a plugin — flat, one
    directory per plugin, NOT under ``model-providers/``. See
    :func:`_discover_installed_provider_plugins`.
    """
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _declares_model_provider_kind(plugin_dir: Path) -> bool:
    """Whether ``plugin_dir``'s manifest declares ``kind: model-provider``.

    Only that kind is imported from the flat install directory — every other
    plugin there belongs to ``PluginManager``, which owns its lifecycle and
    consent flow. Parsed with ruamel.yaml when available, falling back to a line
    scan so provider discovery never hard-depends on it.
    """
    for filename in ("plugin.yaml", "plugin.yml"):
        manifest = plugin_dir / filename
        if not manifest.is_file():
            continue
        try:
            text = manifest.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        try:
            from utils import fast_safe_load

            data = fast_safe_load(text)
            if isinstance(data, dict):
                return str(data.get("kind", "")).strip() == "model-provider"
        except Exception:
            pass
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            if key.strip() == "kind":
                return value.strip().strip("\"'") == "model-provider"
        return False
    return False


def _scan_home_layer(layer: _HomeLayer, key: str) -> None:
    """Import the bound home's not-yet-imported provider plugins into *layer*.

    ``$HERMES_HOME/plugins/model-providers/<name>/`` first, then plugins cloned flat by
    ``hermes plugins install`` into ``$HERMES_HOME/plugins/<name>/`` that declare
    ``kind: model-provider`` (PluginManager owns every other kind there). Per-home module names
    let two profiles carry the same plugin without aliasing each other's registrations.
    """
    global _discovering
    token, prior_discovering = _REGISTRATION_TARGET.set(layer), _discovering
    _discovering = True
    try:
        user_dir = _user_plugins_dir()
        if user_dir is not None:
            for child in sorted(user_dir.iterdir()):
                if child.is_dir() and not child.name.startswith(("_", ".")):
                    _import_plugin_dir(child, "user", home_key=key)
        installed_dir = _installed_plugins_dir()
        if installed_dir is not None:
            for child in sorted(installed_dir.iterdir()):
                if not child.is_dir() or child.name.startswith(("_", ".")) or child.name == "model-providers":
                    continue
                if _declares_model_provider_kind(child):
                    _import_plugin_dir(child, "user", home_key=key)
    finally:
        _REGISTRATION_TARGET.reset(token)
        _discovering = prior_discovering


def _user_module_name(plugin_dir: Path, home_key: str) -> str:
    digest = hashlib.sha1(home_key.encode("utf-8")).hexdigest()[:10]
    return f"_hermes_user_provider_{digest}_{plugin_dir.name.replace('-', '_')}"


def _import_plugin_dir(plugin_dir: Path, source: str, *, home_key: str = "") -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user"; it is recorded per registered profile (``_SOURCES``).
    """
    if not (plugin_dir / "__init__.py").exists():
        return
    if source != "bundled":
        from hermes_cli.plugin_isolation import ISOLATION_HOST, isolation_mode
        if isolation_mode() == ISOLATION_HOST:  # the plugin's code runs in the plugin host
            _import_hosted_plugin_dir(plugin_dir, source, _user_module_name(plugin_dir, home_key))
            return

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` under a per-home module name so
    # multiple HERMES_HOME profiles don't alias each other.
    if source == "bundled":
        module_name = f"plugins.model_providers.{plugin_dir.name.replace('-', '_')}"
    else:
        module_name = _user_module_name(plugin_dir, home_key)
    _exec_plugin_module(plugin_dir, source, module_name)


def _import_hosted_plugin_dir(plugin_dir: Path, source: str, module_name: str) -> None:
    """Register a plugin's profiles as plugin-host proxies; a profile that cannot cross the host
    boundary (e.g. it overrides ``create_client``) is imported in-process ALONE, with a warning,
    so it never takes its hosted siblings down with it."""
    global _current_source
    from hermes_cli.plugin_host_profiles import load_hosted_profiles
    layer = _REGISTRATION_TARGET.get()
    if layer is not None and module_name in layer.hosted:
        return  # already imported into this home's layer
    _current_source = source
    try:
        hosted, in_process = load_hosted_profiles(plugin_dir, module_name)
        for profile in hosted:
            register_provider(profile)
    except Exception as exc:  # health: allow BLE001 -- plugin load boundary (moved from _import_plugin_dir): one broken plugin never breaks discovery
        logger.warning("Failed to load user provider plugin %s in the plugin host: %s", plugin_dir.name, exc)
        _note_load_result(plugin_dir, exc)
        return
    finally:
        _current_source = None
    if in_process:
        for name, reason in in_process.items():
            logger.warning("Provider plugin %s: profile %r cannot run in the plugin host (%s); loading it "
                           "in-process, NOT isolated by plugins.isolation: host", plugin_dir.name, name, reason)
        token = _REGISTRATION_ONLY.set(frozenset(in_process))
        try:
            _exec_plugin_module(plugin_dir, source, module_name)
        finally:
            _REGISTRATION_ONLY.reset(token)
        if module_name not in sys.modules:
            return  # the in-process import failed: the next scan retries the plugin
    if layer is not None:
        layer.hosted.add(module_name)


def _exec_plugin_module(plugin_dir: Path, source: str, module_name: str) -> None:
    global _current_source
    if module_name in sys.modules:
        return  # already imported

    _current_source = source
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _note_load_result(plugin_dir, None)
    except Exception as exc:  # health: allow BLE001 -- plugin import boundary (moved from _import_plugin_dir): one broken plugin never breaks discovery
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)
        _note_load_result(plugin_dir, exc)
    finally:
        _current_source = None


def _note_load_result(plugin_dir: Path, exc: BaseException | None) -> None:
    """Record (or clear) a ``$HERMES_HOME`` plugin's import error on the home layer importing it: the
    general plugin manager lists a model-provider dir with no error, so this is the only record of why it
    registered nothing."""
    layer = _REGISTRATION_TARGET.get()
    if layer is None:  # bundled: process-wide, never a home's plugin
        return
    if exc is None:
        layer.errors.pop(plugin_dir.name, None)
    else:
        layer.errors[plugin_dir.name] = f"{type(exc).__name__}: {exc}"


def plugin_load_error(plugin: str) -> str:
    """The error the bound home's provider plugin dir *plugin* raised on its last import, else ``""``."""
    return _home_layer().errors.get(plugin, "")


def _discover_entry_point_providers() -> None:
    """Import pip-installed provider plugins via the ``hermes_agent.plugins``
    entry-point group so they self-register.

    A distribution ships::

        [project.entry-points."hermes_agent.plugins"]
        acme-inference = "acme_hermes_plugin:register"

    The target may be either a **callable** (``module:func`` — invoked with no
    args; typically calls ``register_provider(profile)``) or a **module**
    (``module`` — imported for its module-level ``register_provider`` side
    effect, mirroring the directory-plugin ``__init__.py`` contract).

    Gating and safety:

    * **Opt-in.** Entry-point plugins are subject to the same
      ``plugins.enabled`` allow-list (and ``plugins.disabled`` deny-list) the
      general PluginManager enforces — a pip package is never imported just
      because it is installed. An entry point whose name is not enabled is
      skipped without loading.
    * **Provider targets only.** The ``hermes_agent.plugins`` group is shared
      with general plugins whose target is ``register(ctx)``. Callables that
      require arguments are skipped here (the PluginManager owns them);
      provider registration hooks take no arguments by contract.

    Failures are swallowed per-entry (a broken third-party package must not
    break provider discovery) and logged at warning level. This scan runs
    first, so filesystem plugins (bundled + ``$HERMES_HOME``) keep their
    documented override precedence via last-writer-wins in
    ``register_provider()`` — a pip package cannot hijack a first-party
    provider name.
    """
    try:
        import importlib.metadata as _md
    except Exception:  # pragma: no cover — importlib.metadata always present ≥3.8
        return

    # Same opt-in gate as the general PluginManager: only entry points named
    # in ``plugins.enabled`` load, and ``plugins.disabled`` always wins.
    try:
        from hermes_cli.plugins import _get_disabled_plugins, _get_enabled_plugins

        enabled = _get_enabled_plugins()  # None = nothing enabled yet (opt-in default)
        disabled = _get_disabled_plugins()
    except Exception:  # pragma: no cover — config layer unavailable
        enabled, disabled = None, set()
    if not enabled:
        return

    group = "hermes_agent.plugins"
    try:
        eps = _md.entry_points()
        # Python 3.10+ exposes .select(); older returns a dict-like mapping.
        if hasattr(eps, "select"):
            group_eps = list(eps.select(group=group))
        else:  # pragma: no cover — legacy interpreters
            group_eps = list(eps.get(group, []))  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("entry-point provider scan skipped: %s", exc)
        return

    for ep in group_eps:
        if ep.name not in enabled or ep.name in disabled:
            logger.debug(
                "entry-point provider %r skipped: not enabled in config", ep.name
            )
            continue
        from hermes_cli.plugin_isolation import in_process_import_refusal
        refusal = in_process_import_refusal(f"pip-installed model-provider plugin {ep.name!r}")
        if refusal:
            logger.warning("%s", refusal)
            continue
        try:
            loaded = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load entry-point provider plugin %r: %s", ep.name, exc
            )
            continue
        # ``module:func`` → callable we invoke; bare ``module`` → import side
        # effect already happened during load(). Only call when it's callable
        # AND zero-arg: general plugins in this shared group expose
        # ``register(ctx)`` (requires an argument) and belong to the
        # PluginManager, not the provider registry.
        if callable(loaded):
            if _requires_arguments(loaded):
                logger.debug(
                    "entry-point %r skipped by provider scan: target requires "
                    "arguments (general plugin owned by PluginManager)",
                    ep.name,
                )
                continue
            try:
                loaded()
            except Exception as exc:
                logger.warning(
                    "Entry-point provider plugin %r raised on invocation: %s",
                    ep.name,
                    exc,
                )


def _requires_arguments(fn) -> bool:
    """True when ``fn`` cannot be called with zero arguments.

    Used to distinguish provider registration hooks (zero-arg by contract)
    from general plugin hooks (``register(ctx)``) sharing the same entry-point
    group. Unintrospectable callables (C extensions) are treated as zero-arg
    and left to the per-entry exception guard.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover — builtins/C callables
        return False
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) and param.default is inspect.Parameter.empty:
            return True
    return False


def bundled_profiles() -> list[ProviderProfile]:
    """Every bundled provider profile, one per name (the last registration wins, in discovery
    order), WITHOUT config and without touching the registry: each bundled plugin is executed into
    a private module whose registrations are collected. For descriptive catalogs, which must not
    depend on config or on discovery having run."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():  # a safe worker loads no plugin code, bundled included (as list_providers)
        return []
    collected: list[ProviderProfile] = []
    token = _COLLECT.set(collected)
    try:
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()) if _BUNDLED_PLUGINS_DIR.is_dir() else ():
            init = child / "__init__.py"
            if not child.is_dir() or child.name.startswith(("_", ".")) or not init.exists():
                continue
            name = f"_hermes_provider_catalog_{child.name.replace('-', '_')}"
            spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(child)])
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # relative imports inside the plugin
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # health: allow BLE001 -- one broken plugin never breaks the catalog
                logger.debug("provider catalog skipped %s: %s", child.name, exc)
    finally:
        _COLLECT.reset(token)
    return list({p.name: p for p in collected}.values())


def _discover_providers() -> None:
    """Populate the process-wide registry by importing every provider plugin.

    Order:
      1. Bundled plugins at ``<repo>/plugins/model-providers/<name>/``
      2. Legacy per-file modules at ``providers/<name>.py`` (back-compat)

    Each step imports its plugins, which call ``register_provider()`` at
    module-level. Later steps win on name collision. ``$HERMES_HOME`` plugins are
    per profile home and load through :func:`_home_layer` at lookup time.
    """
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return
    global _discovered, _discovering
    if _discovered:
        return
    _discovered = True
    _discovering = True
    try:
        _run_discovery_steps()
    finally:
        _discovering = False
        # hermes_cli.auth may have been imported by a plugin during discovery and snapshotted a
        # partial profile list — hand it the complete one (no-op unless auth is already loaded).
        _sync_auth_registry()


def _run_discovery_steps() -> None:
    """The discovery passes, in precedence order (see :func:`_discover_providers`)."""
    # 0. Pip-installed plugins — entry points in the ``hermes_agent.plugins``
    #    group (the same group the general PluginManager uses). The manager
    #    records model-provider manifests for introspection but deliberately
    #    does NOT import them — provider lifecycle is owned here — so without
    #    this step a ``pip install``ed provider never calls
    #    ``register_provider()`` and is never selectable.
    #
    #    Discovered FIRST, i.e. lowest precedence: because
    #    ``register_provider()`` is last-writer-wins, running this before the
    #    filesystem steps means a bundled or ``$HERMES_HOME`` profile of the
    #    same name always overrides a pip-installed one. That prevents a
    #    third-party package from silently hijacking a first-party provider
    #    name (e.g. ``openrouter``) while still letting pip packages add
    #    genuinely new providers.
    _discover_entry_point_providers()

    # 1. Bundled plugins — shipped with hermes-agent.
    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    # 2. Legacy single-file profiles at providers/<name>.py. Kept for
    #    back-compat — if someone drops a ``providers/foo.py`` into an
    #    editable install, it still works without the plugin layout.
    try:
        import pkgutil

        import providers as _pkg

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                importlib.import_module(f"providers.{modname}")
            except ImportError as exc:
                logger.warning(
                    "Failed to import legacy provider module %s: %s", modname, exc
                )
    except Exception:
        pass

    # (Pip entry-point providers are discovered in step 0, before the
    # filesystem plugins, so first-party profiles always win on name
    # collision — see _discover_entry_point_providers.)
