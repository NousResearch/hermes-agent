"""Provider module registry.

Provider profiles can live in three places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``
3. Opt-in project plugins: ``./.hermes/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — identity plus non-executable ``requires_env`` metadata

Discovery is lazy. Manifest identity, credential names, and activation are
evaluated before any plugin code is imported. Bundled profiles are on by
default; user and project profiles must be listed in ``plugins.enabled``.
Explicit disable always wins, and safe mode imports bundled profiles only.
Active user/project plugins override bundled plugins on canonical-key collision.

For backward compatibility, explicitly enabled ``providers/*.py`` files
(other than ``base.py`` and ``__init__.py``) are still discovered via
``pkgutil.iter_modules``. This lets out-of-tree users keep a single-file
profile in an editable install without making it execute by default. New
profiles should prefer the plugin layout.

Usage::

    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from hermes_cli.plugin_activation import PluginActivationState
from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401
from providers._core_identities import (
    PROFILELESS_CORE_PROVIDER_ALIASES,
    PROFILELESS_CORE_PROVIDER_IDS,
)
from utils import atomic_json_write, env_var_enabled, fast_safe_load

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
_discovered = False
_discovering = False
_ACTIVATION_STATE: PluginActivationState | None = None
_DISCOVERY_FINGERPRINT: tuple[object, ...] | None = None
_DISCOVERY_LOCK = threading.RLock()
_IMPORTED_PROVIDER_MODULES: set[str] = set()
_PROVIDER_REFRESH_HOOKS: list[Callable[[], None]] = []
_PLUGIN_MANAGED_PROVIDER_IDS: set[str] = set()
_PROVIDER_PROFILE_ORIGINS: dict[str, tuple[str, str]] = {}


@dataclass(frozen=True)
class ProviderCatalogSnapshot:
    """One immutable provider catalog for one profile/project scope.

    Provider discovery used to publish into the module globals above.  That is
    unsafe in the TUI gateway, where concurrent turns bind different
    ``HERMES_HOME`` values with a ContextVar: one turn could read another
    profile's endpoint after the second turn refreshed the process-global
    registry.  Runtime readers now resolve this snapshot from their current
    scope.  The old globals remain only as a compatibility mirror for tests
    and third-party code that inspected private attributes.
    """

    scope_identity: tuple[str, str]
    fingerprint: tuple[object, ...]
    activation: PluginActivationState
    registry: MappingProxyType
    aliases: MappingProxyType
    origins: MappingProxyType
    profiles: tuple[ProviderProfile, ...]
    plugin_managed_provider_ids: frozenset[str]
    active_plugin_provider_ids: frozenset[str]
    bundled_provider_ids: frozenset[str]
    known_provider_ids: frozenset[str]


@dataclass
class _ProviderBuildState:
    registry: dict[str, ProviderProfile]
    aliases: dict[str, str]
    origins: dict[str, tuple[str, str]]


_BUILD_LOCAL = threading.local()
_SNAPSHOT_CACHE: dict[tuple[object, ...], ProviderCatalogSnapshot] = {}
_LAST_SNAPSHOT_FINGERPRINT: tuple[object, ...] | None = None
_NOTIFIED_SNAPSHOT_FINGERPRINTS: set[tuple[object, ...]] = set()
_MODULE_REGISTRATIONS: dict[str, tuple[ProviderProfile, ...]] = {}
_MODULE_PATHS: dict[str, str] = {}
_OBSERVED_PROFILES: dict[tuple[str, str, str], ProviderProfile] = {}
_OBSERVED_PROVIDER_ENV_VARS: set[str] = set()
_OBSERVED_PROVIDER_IDS_BY_SCOPE: dict[tuple[str, str], set[str]] = {}
_RUNTIME_REGISTRY: dict[str, ProviderProfile] = {}
_RUNTIME_ALIASES: dict[str, str] = {}
_RUNTIME_REGISTRATION_GENERATION = 0

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)

# Exact operational names that a provider must never claim as credentials.
# Provider metadata is allowed to use unconventional secret names (for example
# ``LC_VENDOR_ACCESS``), so this must remain an exact-name policy rather than a
# prefix allowlist.  Compare through ``upper()`` because Windows environment
# names are case-insensitive even when a plugin manifest preserves mixed case.
_RESERVED_PROVIDER_ENV_VARS = frozenset({
    # POSIX process/runtime essentials.
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "OLDPWD",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    # Windows process/runtime essentials. Keep this aligned with the exact
    # operational allowlist in tools.code_execution_tool.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "OS",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PUBLIC",
    "ALLUSERSPROFILE",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "USERDOMAIN",
    "USERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
    "COMPUTERNAME",
    # Hermes child-process routing/location invariants.
    "HERMES_HOME",
    "HERMES_PROFILE",
    "HERMES_CONFIG",
    "HERMES_ENV",
    "HERMES_DELEGATED_CHILD_CONTEXT",
})


def is_reserved_provider_env_var(name: object) -> bool:
    """Return whether *name* is operational and cannot be a credential."""
    return (
        isinstance(name, str)
        and bool(name.strip())
        and name.strip().upper() in _RESERVED_PROVIDER_ENV_VARS
    )


def _provider_credential_env_names(values: object) -> tuple[str, ...]:
    """Validate credential names while preserving their declared order."""
    if isinstance(values, str):
        entries = (values,)
    elif isinstance(values, (list, tuple, set, frozenset)):
        entries = values
    else:
        return ()

    names: list[str] = []
    seen: set[str] = set()
    for value in entries:
        if not isinstance(value, str):
            continue
        name = value.strip()
        # NUL and '=' cannot name environment variables on supported hosts.
        # Reserved operational names are rejected fail-closed so declarative
        # metadata cannot turn child-process credential scrubbing into a DoS.
        if (
            not name
            or "\x00" in name
            or "=" in name
            or is_reserved_provider_env_var(name)
        ):
            continue
        # Windows environment names are case-insensitive. Canonicalizing the
        # security metadata prevents a mixed-case manifest name from missing
        # the differently-cased key spelling preserved by ``os.environ``.
        if sys.platform == "win32":
            name = name.upper()
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return tuple(names)


def _record_observed_profile_locked(
    source: str,
    path: str,
    profile: ProviderProfile,
) -> None:
    """Record monotonic non-executable security metadata under discovery lock."""
    _OBSERVED_PROFILES[(source, path, profile.name)] = profile
    _OBSERVED_PROVIDER_ENV_VARS.update(
        _provider_credential_env_names(profile.env_vars)
    )


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    global _PROVIDER_LIST_CACHE, _RUNTIME_REGISTRATION_GENERATION
    # ProviderProfile is mutable by design. Normalize at the registration
    # boundary so active plugin metadata cannot reach auth/config indexes under
    # an OS-essential name through a path other than the observation set.
    profile.env_vars = _provider_credential_env_names(profile.env_vars)
    build_state = getattr(_BUILD_LOCAL, "state", None)
    if isinstance(build_state, _ProviderBuildState):
        build_state.registry[profile.name] = profile
        for alias in profile.aliases:
            build_state.aliases[alias] = profile.name
        return

    with _DISCOVERY_LOCK:
        _RUNTIME_REGISTRY[profile.name] = profile
        for alias in profile.aliases:
            _RUNTIME_ALIASES[alias] = profile.name
        _RUNTIME_REGISTRATION_GENERATION += 1
        _REGISTRY[profile.name] = profile
        for alias in profile.aliases:
            _ALIASES[alias] = profile.name
        _record_observed_profile_locked("runtime", "", profile)
        _PROVIDER_LIST_CACHE = None


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    build_state = getattr(_BUILD_LOCAL, "state", None)
    if isinstance(build_state, _ProviderBuildState):
        canonical = build_state.aliases.get(name, name)
        return build_state.registry.get(canonical)

    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            canonical = _ALIASES.get(name, name)
            return _REGISTRY.get(canonical)
    canonical = snapshot.aliases.get(name, name)
    return snapshot.registry.get(canonical)


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    global _PROVIDER_LIST_CACHE
    build_state = getattr(_BUILD_LOCAL, "state", None)
    if isinstance(build_state, _ProviderBuildState):
        return _dedupe_profiles(build_state.registry.values())

    snapshot = _ensure_providers_discovered()
    if snapshot is not None:
        return list(snapshot.profiles)

    # Private-registry compatibility for tests that deliberately install a
    # hand-built registry and mark it discovered without a fingerprint.
    with _DISCOVERY_LOCK:
        if _PROVIDER_LIST_CACHE is not None:
            return list(_PROVIDER_LIST_CACHE)
        result = _dedupe_profiles(_REGISTRY.values())
        _PROVIDER_LIST_CACHE = list(result)
        return result


def _dedupe_profiles(profiles) -> list[ProviderProfile]:
    seen: set[int] = set()
    result: list[ProviderProfile] = []
    for profile in profiles:
        profile_identity = id(profile)
        if profile_identity not in seen:
            seen.add(profile_identity)
            result.append(profile)
    return result


def get_provider_catalog_snapshot() -> ProviderCatalogSnapshot:
    """Return the immutable provider catalog for the current ContextVar scope."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            profiles = tuple(_dedupe_profiles(_REGISTRY.values()))
            return ProviderCatalogSnapshot(
                scope_identity=("compat", ""),
                fingerprint=("compat",),
                activation=PluginActivationState(),
                registry=MappingProxyType(dict(_REGISTRY)),
                aliases=MappingProxyType(dict(_ALIASES)),
                origins=MappingProxyType(dict(_PROVIDER_PROFILE_ORIGINS)),
                profiles=profiles,
                plugin_managed_provider_ids=frozenset(
                    _PLUGIN_MANAGED_PROVIDER_IDS
                ),
                active_plugin_provider_ids=frozenset(_REGISTRY),
                bundled_provider_ids=frozenset(),
                known_provider_ids=frozenset(_REGISTRY),
            )
    return snapshot


def get_provider_scope_identity() -> tuple[str, str]:
    """Return the profile/project identity without importing plugin code."""
    try:
        from hermes_constants import get_hermes_home

        home = _path_identity(get_hermes_home())
    except Exception:
        home = ""
    project = _path_identity(Path.cwd()) if env_var_enabled(
        "HERMES_ENABLE_PROJECT_PLUGINS"
    ) else ""
    return home, project


def register_provider_refresh_hook(callback: Callable[[], None]) -> None:
    """Register an in-process index derived from provider discovery."""
    with _DISCOVERY_LOCK:
        if callback not in _PROVIDER_REFRESH_HOOKS:
            _PROVIDER_REFRESH_HOOKS.append(callback)


def get_provider_discovery_identity() -> tuple[object, ...]:
    """Return the active discovery identity for downstream cache keys."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            return _DISCOVERY_FINGERPRINT or ()
    state = snapshot.activation
    return (
        *snapshot.scope_identity,
        state.safe_mode,
        None if state.enabled is None else tuple(sorted(state.enabled)),
        tuple(sorted(state.disabled)),
        *snapshot.fingerprint[3:],
        tuple(sorted(snapshot.origins.items())),
    )


def get_provider_profile_origin(name: str) -> tuple[str, str] | None:
    """Return ``(source, path)`` for the active provider profile."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            canonical = _ALIASES.get(name, name)
            return _PROVIDER_PROFILE_ORIGINS.get(canonical)
    canonical = snapshot.aliases.get(name, name)
    return snapshot.origins.get(canonical)


def invalidate_provider_discovery() -> None:
    """Rebuild providers and notify indexes after activation changes."""
    global _discovered, _ACTIVATION_STATE, _DISCOVERY_FINGERPRINT
    global _LAST_SNAPSHOT_FINGERPRINT
    scope_identity = get_provider_scope_identity()
    with _DISCOVERY_LOCK:
        stale = [
            fingerprint
            for fingerprint, snapshot in _SNAPSHOT_CACHE.items()
            if snapshot.scope_identity == scope_identity
        ]
        for fingerprint in stale:
            _SNAPSHOT_CACHE.pop(fingerprint, None)
            _NOTIFIED_SNAPSHOT_FINGERPRINTS.discard(fingerprint)

        # Explicit invalidation also means plugin code may have changed on
        # disk.  Existing snapshots retain their already-created objects, so
        # it is safe to evict module/cache entries before the next build.
        for prefix in tuple(_IMPORTED_PROVIDER_MODULES):
            for module_name in tuple(sys.modules):
                if module_name == prefix or module_name.startswith(f"{prefix}."):
                    sys.modules.pop(module_name, None)
        _IMPORTED_PROVIDER_MODULES.clear()
        _MODULE_REGISTRATIONS.clear()
        _MODULE_PATHS.clear()
        _discovered = False
        _ACTIVATION_STATE = None
        _DISCOVERY_FINGERPRINT = None
        _LAST_SNAPSHOT_FINGERPRINT = None
    _ensure_providers_discovered()


def is_plugin_managed_provider_id(provider_id: str) -> bool:
    """Return whether any model-provider plugin declares *provider_id*."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            return provider_id in _PLUGIN_MANAGED_PROVIDER_IDS
    return provider_id in snapshot.plugin_managed_provider_ids


def is_provider_plugin_active(provider_id: str) -> bool:
    """Return whether a plugin-managed provider is active and registered."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            if provider_id not in _PLUGIN_MANAGED_PROVIDER_IDS:
                return True
            return provider_id in _REGISTRY
    if provider_id not in snapshot.plugin_managed_provider_ids:
        return True
    return provider_id in snapshot.active_plugin_provider_ids


def is_bundled_provider_id(provider_id: str) -> bool:
    """Return whether the trusted bundled catalog owns *provider_id*."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        return False
    return provider_id in snapshot.bundled_provider_ids


def get_known_provider_ids() -> frozenset[str]:
    """Return non-executable identities suitable for cleanup validation."""
    snapshot = _ensure_providers_discovered()
    if snapshot is None:
        with _DISCOVERY_LOCK:
            return frozenset(_REGISTRY)
    return snapshot.known_provider_ids


def get_observed_provider_profiles() -> tuple[ProviderProfile, ...]:
    """Return a process-lifetime union used only for secret-name hardening."""
    with _DISCOVERY_LOCK:
        return tuple(_OBSERVED_PROFILES.values())


def get_observed_provider_env_vars() -> frozenset[str]:
    """Return provider env names observed before catalog publication.

    Provider refresh hooks run after a newly built catalog is placed in the
    snapshot cache.  A concurrent subprocess launch must not depend on those
    callbacks having completed: otherwise a newly discovered provider key can
    briefly escape the child-env scrubber.  Executable profiles and static
    ``plugin.yaml`` metadata both update this monotonic view synchronously
    while discovery still holds ``_DISCOVERY_LOCK``, so it is the security
    publication barrier for spawn-time checks.
    """
    with _DISCOVERY_LOCK:
        # Re-filter on read as defense in depth for a long-lived process whose
        # monotonic set was populated before this policy became active.
        return frozenset(
            _provider_credential_env_names(_OBSERVED_PROVIDER_ENV_VARS)
        )


def is_observed_provider_env_var(name: str) -> bool:
    """Check the provider-owned spawn-security index without hook latency."""
    if is_reserved_provider_env_var(name):
        return False
    with _DISCOVERY_LOCK:
        return name in _OBSERVED_PROVIDER_ENV_VARS


def _current_activation_state() -> PluginActivationState:
    """Late-bind config.py to avoid its eager provider-injection cycle."""
    try:
        from hermes_cli.config import load_plugin_activation_state

        return load_plugin_activation_state()
    except Exception:
        return PluginActivationState(
            safe_mode=env_var_enabled("HERMES_SAFE_MODE"),
        )


def _ensure_providers_discovered() -> ProviderCatalogSnapshot | None:
    """Resolve the immutable catalog for the caller's current scope."""
    global _LAST_SNAPSHOT_FINGERPRINT

    # A small private compatibility seam used by provider-registry unit tests.
    if _discovered and _DISCOVERY_FINGERPRINT is None:
        return None

    state = _current_activation_state()
    scope_identity = get_provider_scope_identity()
    user_dir = _user_plugins_dir()
    project_dir = _project_plugins_dir()
    fingerprint = (
        *scope_identity,
        state,
        _RUNTIME_REGISTRATION_GENERATION,
        _path_identity(_BUNDLED_PLUGINS_DIR),
        _path_identity(user_dir),
        _path_identity(project_dir),
    )
    callbacks: tuple[Callable[[], None], ...] = ()
    with _DISCOVERY_LOCK:
        snapshot = _SNAPSHOT_CACHE.get(fingerprint)
        if snapshot is None:
            snapshot = _discover_providers(
                state,
                scope_identity=scope_identity,
                user_dir=user_dir,
                project_dir=project_dir,
                fingerprint=fingerprint,
            )
            _SNAPSHOT_CACHE[fingerprint] = snapshot

        if _LAST_SNAPSHOT_FINGERPRINT != fingerprint:
            _publish_compatibility_mirror(snapshot)
            _LAST_SNAPSHOT_FINGERPRINT = fingerprint

        # Derived registries are context-scoped/lazy, while the remaining
        # hooks only warm or monotonically harden metadata. Notify once per
        # immutable snapshot rather than every A/B profile switch; otherwise
        # concurrent multiplex traffic turns a compatibility-mirror change
        # into a refresh storm.
        if fingerprint not in _NOTIFIED_SNAPSHOT_FINGERPRINTS:
            _NOTIFIED_SNAPSHOT_FINGERPRINTS.add(fingerprint)
            callbacks = tuple(_PROVIDER_REFRESH_HOOKS)

    callback_failed = False
    for callback in callbacks:
        try:
            callback()
        except Exception:
            callback_failed = True
            logger.warning(
                "Failed to refresh a provider-derived registry",
                exc_info=True,
            )
    if callback_failed:
        # Best-effort hooks may be retried by the next reader. Runtime routing
        # and subprocess security do not depend on hook completion.
        with _DISCOVERY_LOCK:
            _NOTIFIED_SNAPSHOT_FINGERPRINTS.discard(fingerprint)
    return snapshot


def _publish_compatibility_mirror(snapshot: ProviderCatalogSnapshot) -> None:
    """Publish a non-authoritative mirror for legacy private-attribute users."""
    global _discovered, _ACTIVATION_STATE, _DISCOVERY_FINGERPRINT
    global _PROVIDER_LIST_CACHE

    _REGISTRY.clear()
    _REGISTRY.update(snapshot.registry)
    _ALIASES.clear()
    _ALIASES.update(snapshot.aliases)
    _PROVIDER_PROFILE_ORIGINS.clear()
    _PROVIDER_PROFILE_ORIGINS.update(snapshot.origins)
    _PLUGIN_MANAGED_PROVIDER_IDS.clear()
    _PLUGIN_MANAGED_PROVIDER_IDS.update(snapshot.plugin_managed_provider_ids)
    _PROVIDER_LIST_CACHE = list(snapshot.profiles)
    _ACTIVATION_STATE = snapshot.activation
    _DISCOVERY_FINGERPRINT = snapshot.fingerprint
    _discovered = True


def _path_identity(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(path.absolute())


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _project_plugins_dir() -> Path | None:
    """Return the opt-in project model-provider directory when present."""
    if not env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        return None
    directory = Path.cwd() / ".hermes" / "plugins" / "model-providers"
    return directory if directory.is_dir() else None


@dataclass(frozen=True)
class _ProviderPlugin:
    path: Path
    source: str
    key: str
    name: str
    provider_ids: frozenset[str]
    credential_env_vars: frozenset[str]


_PROVIDER_ENV_CACHE_VERSION = 1
_PROVIDER_ENV_CACHE_SOURCES = frozenset({"user", "project"})


def _manifest_env_names(value: object) -> frozenset[str]:
    """Extract simple/rich ``requires_env`` entries without executing code."""
    if isinstance(value, (str, dict)):
        entries = (value,)
    elif isinstance(value, list):
        entries = value
    else:
        return frozenset()

    names: list[str] = []
    for entry in entries:
        raw_name = entry.get("name") if isinstance(entry, dict) else entry
        if isinstance(raw_name, str):
            names.append(raw_name)
    # Provider credentials do not have to follow API_KEY/TOKEN naming
    # conventions (for example LC_VENDOR_ACCESS); only invalid and exact
    # operational names are excluded.
    return frozenset(_provider_credential_env_names(names))


def _provider_env_cache_path(
    scope_identity: tuple[str, str],
    plugin: _ProviderPlugin,
) -> Path | None:
    """Return a profile-local security cache path without exposing plugin paths."""
    home = scope_identity[0]
    if not home or plugin.source not in _PROVIDER_ENV_CACHE_SOURCES:
        return None
    identity = f"{plugin.source}\x00{_path_identity(plugin.path)}"
    digest = hashlib.sha256(
        identity.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return Path(home) / "cache" / "provider-env-names" / f"{digest}.json"


def _load_cached_provider_env_names(path: Path | None) -> frozenset[str]:
    """Read previously observed names; malformed cache data grants nothing."""
    if path is None:
        return frozenset()
    try:
        if not path.is_file():
            return frozenset()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != _PROVIDER_ENV_CACHE_VERSION:
            return frozenset()
        return _manifest_env_names(data.get("env_vars"))
    except (OSError, ValueError, TypeError):
        logger.debug("Could not read provider credential-name cache: %s", path)
        return frozenset()


def _persist_provider_env_names(
    path: Path | None,
    names: set[str],
) -> None:
    """Persist a monotonic name-only blocklist, never credentials or activation."""
    if path is None:
        return
    merged = set(_load_cached_provider_env_names(path))
    merged.update(_provider_credential_env_names(names))
    # Avoid creating empty cache files, but rewrite an existing cache even when
    # only rejected names remain so an older poisoned file is self-healing.
    if not merged and not path.is_file():
        return
    try:
        atomic_json_write(
            path,
            {
                "version": _PROVIDER_ENV_CACHE_VERSION,
                "env_vars": sorted(merged),
            },
            mode=0o600,
        )
    except OSError:
        # The current process is already protected by the in-memory monotonic
        # set. A read-only profile should not make provider discovery unusable.
        logger.warning(
            "Could not persist provider credential-name security cache: %s",
            path,
            exc_info=True,
        )


def _provider_plugin(plugin_dir: Path, source: str) -> _ProviderPlugin:
    """Read provider identities without importing executable plugin code."""
    key = f"model-providers/{plugin_dir.name}"
    name = key
    provider_ids = {plugin_dir.name}
    credential_env_vars: frozenset[str] = frozenset()
    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = plugin_dir / "plugin.yml"
    if manifest_file.exists():
        try:
            data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                manifest_name = data.get("name")
                if isinstance(manifest_name, str) and manifest_name.strip():
                    name = manifest_name.strip()
                declared_ids = data.get("provider_ids")
                if isinstance(declared_ids, list):
                    normalized_ids = {
                        value.strip()
                        for value in declared_ids
                        if isinstance(value, str) and value.strip()
                    }
                    if normalized_ids:
                        provider_ids = normalized_ids
                credential_env_vars = _manifest_env_names(
                    data.get("requires_env")
                )
        except Exception:
            logger.debug(
                "Could not parse provider plugin manifest identity: %s",
                plugin_dir,
                exc_info=True,
            )
    return _ProviderPlugin(
        path=plugin_dir,
        source=source,
        key=key,
        name=name,
        provider_ids=frozenset(provider_ids),
        credential_env_vars=credential_env_vars,
    )


def _provider_module_name(plugin_dir: Path, source: str) -> str:
    safe_name = plugin_dir.name.replace("-", "_")
    if source == "bundled":
        return f"plugins.model_providers.{safe_name}"
    path_digest = hashlib.sha256(
        _path_identity(plugin_dir).encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:16]
    return f"_hermes_{source}_provider_{safe_name}_{path_digest}"


def _module_provider_profiles(module) -> tuple[ProviderProfile, ...]:
    """Best-effort replay for a provider module imported before discovery."""
    profiles: list[ProviderProfile] = []
    seen: set[int] = set()
    for value in vars(module).values():
        if isinstance(value, ProviderProfile) and id(value) not in seen:
            seen.add(id(value))
            profiles.append(value)
    return tuple(profiles)


def _record_plugin_profiles(
    profiles: tuple[ProviderProfile, ...],
    *,
    source: str,
    plugin_dir: Path,
) -> None:
    build_state = getattr(_BUILD_LOCAL, "state", None)
    if not isinstance(build_state, _ProviderBuildState):
        raise RuntimeError("provider plugin imported outside discovery build")
    origin = (source, _path_identity(plugin_dir))
    for profile in profiles:
        register_provider(profile)
        build_state.origins[profile.name] = origin
        _record_observed_profile_locked(source, origin[1], profile)


def _import_plugin_dir(plugin_dir: Path, source: str) -> tuple[ProviderProfile, ...]:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled", "user", or "project".
    """
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return ()

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` with a unique module name so
    # multiple HERMES_HOME profiles don't alias each other.
    module_name = _provider_module_name(plugin_dir, source)

    if module_name in sys.modules:
        profiles = _MODULE_REGISTRATIONS.get(module_name)
        if profiles is None:
            profiles = _module_provider_profiles(sys.modules[module_name])
            _MODULE_REGISTRATIONS[module_name] = profiles
        _record_plugin_profiles(profiles, source=source, plugin_dir=plugin_dir)
        _IMPORTED_PROVIDER_MODULES.add(module_name)
        return profiles

    build_state = getattr(_BUILD_LOCAL, "state", None)
    if not isinstance(build_state, _ProviderBuildState):
        raise RuntimeError("provider discovery build state is unavailable")
    registry_snapshot = dict(build_state.registry)
    aliases_snapshot = dict(build_state.aliases)
    origins_snapshot = dict(build_state.origins)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return ()
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        profiles = tuple(
            profile
            for provider_id, profile in build_state.registry.items()
            if registry_snapshot.get(provider_id) is not profile
        )
        origin = (source, _path_identity(plugin_dir))
        for profile in profiles:
            build_state.origins[profile.name] = origin
            _record_observed_profile_locked(source, origin[1], profile)
        _MODULE_REGISTRATIONS[module_name] = profiles
        _MODULE_PATHS[module_name] = origin[1]
        _IMPORTED_PROVIDER_MODULES.add(module_name)
        return profiles
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        build_state.registry.clear()
        build_state.registry.update(registry_snapshot)
        build_state.aliases.clear()
        build_state.aliases.update(aliases_snapshot)
        build_state.origins.clear()
        build_state.origins.update(origins_snapshot)
        _MODULE_REGISTRATIONS.pop(module_name, None)
        _MODULE_PATHS.pop(module_name, None)
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
                sys.modules.pop(imported_name, None)
        return ()


def _discover_providers(
    state: PluginActivationState,
    *,
    scope_identity: tuple[str, str],
    user_dir: Path | None,
    project_dir: Path | None,
    fingerprint: tuple[object, ...],
) -> ProviderCatalogSnapshot:
    """Build one provider snapshot without publishing process-global state."""
    global _discovering
    if _discovering:
        raise RuntimeError("recursive provider discovery")

    _discovering = True
    build_state = _ProviderBuildState(
        registry=dict(_RUNTIME_REGISTRY),
        aliases=dict(_RUNTIME_ALIASES),
        origins={},
    )
    _BUILD_LOCAL.state = build_state
    try:
        candidates: list[_ProviderPlugin] = []
        for directory, source in (
            (_BUNDLED_PLUGINS_DIR, "bundled"),
            (user_dir, "user"),
            (project_dir, "project"),
        ):
            if directory is None or not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                if not child.is_dir() or child.name.startswith(("_", ".")):
                    continue
                candidates.append(_provider_plugin(child, source))

        grouped: dict[str, list[_ProviderPlugin]] = {}
        # Static core implementations and profiles registered outside plugin
        # discovery are trusted owners.  An inactive external manifest may
        # declare their IDs for an override, but cannot turn the declaration
        # itself into an activation deny for the underlying implementation.
        # Bundled manifests remain activation-managed even when auth also has
        # static implementation metadata for the same ID.
        bundled_declared_provider_ids = {
            provider_id
            for plugin in candidates
            if plugin.source == "bundled"
            for provider_id in plugin.provider_ids
        }
        trusted_implementation_ids = set(PROFILELESS_CORE_PROVIDER_IDS)
        trusted_implementation_ids.update(build_state.registry)
        activation_exempt_ids = (
            trusted_implementation_ids - bundled_declared_provider_ids
        )
        trusted_aliases = dict(PROFILELESS_CORE_PROVIDER_ALIASES)
        trusted_aliases.update(build_state.aliases)
        activation_exempt_ids.update(
            alias
            for alias, canonical in trusted_aliases.items()
            if canonical in activation_exempt_ids
        )
        managed_provider_ids: set[str] = set(
            _OBSERVED_PROVIDER_IDS_BY_SCOPE.get(scope_identity, ())
        ) - activation_exempt_ids
        known_provider_ids: set[str] = set(managed_provider_ids)
        bundled_provider_ids: set[str] = set(bundled_declared_provider_ids)
        active_provider_ids: set[str] = set()
        for plugin in candidates:
            # Credential names are declarative security metadata, not an
            # activation grant. Read them for inactive/disabled plugins too so
            # a fresh process can scrub their secrets without importing
            # untrusted ``__init__.py`` code. This union is process-local and
            # never changes routing or plugin activation.
            cache_path = _provider_env_cache_path(scope_identity, plugin)
            cached_env_vars = _load_cached_provider_env_names(cache_path)
            _OBSERVED_PROVIDER_ENV_VARS.update(
                _provider_credential_env_names(cached_env_vars)
            )
            _OBSERVED_PROVIDER_ENV_VARS.update(
                _provider_credential_env_names(plugin.credential_env_vars)
            )
            # Rewrite legacy cache files through the current validator even
            # when the plugin is inactive or disabled. Otherwise a reserved
            # name removed from the in-memory index would remain poisoned on
            # disk indefinitely (and could affect an older binary later).
            _persist_provider_env_names(cache_path, set(cached_env_vars))
            grouped.setdefault(plugin.key, []).append(plugin)
            known_provider_ids.update(plugin.provider_ids)
            known_provider_ids.update((plugin.path.name, plugin.name))
            if plugin.source == "bundled":
                managed_provider_ids.update(plugin.provider_ids)
            else:
                managed_provider_ids.update(
                    plugin.provider_ids - activation_exempt_ids
                )

        for key, plugins in grouped.items():
            statuses = [
                (
                    plugin,
                    state.status(
                        name=plugin.name,
                        key=plugin.key,
                        source=plugin.source,
                        kind="model-provider",
                    ),
                )
                for plugin in plugins
            ]
            # A deny on any identity in a canonical-key collision is a
            # fail-closed deny for the whole key.  Otherwise inactive external
            # candidates simply fall back to active bundled candidates.
            if any(status == "disabled" for _plugin, status in statuses):
                logger.debug(
                    "Skipping explicitly disabled provider plugin group '%s'",
                    key,
                )
                continue
            active_plugins = [
                plugin for plugin, status in statuses if status == "enabled"
            ]
            if not active_plugins:
                logger.debug(
                    "Skipping inactive provider plugin group '%s' (%s)",
                    key,
                    ", ".join(plugin.source for plugin in plugins),
                )
                continue

            # Load active sources in precedence order. Bundled profiles load
            # first so an enabled user/project override can replace selected
            # IDs without deleting sibling profiles from a multi-ID bundle.
            for plugin in active_plugins:
                profiles = _import_plugin_dir(plugin.path, plugin.source)
                observed_plugin_env_vars: set[str] = set()
                for profile in profiles:
                    observed_plugin_env_vars.update(
                        _provider_credential_env_names(profile.env_vars)
                    )
                    active_provider_ids.add(profile.name)
                    active_provider_ids.update(profile.aliases)
                    registered_ids = {profile.name, *profile.aliases}
                    if plugin.source != "bundled":
                        registered_ids.difference_update(activation_exempt_ids)
                    managed_provider_ids.update(registered_ids)
                    known_provider_ids.add(profile.name)
                    known_provider_ids.update(profile.aliases)
                    if plugin.source == "bundled":
                        bundled_provider_ids.add(profile.name)
                        bundled_provider_ids.update(profile.aliases)
                _persist_provider_env_names(
                    _provider_env_cache_path(scope_identity, plugin),
                    observed_plugin_env_vars,
                )

        # Legacy single-file profiles are a compatibility extension path. Safe
        # mode must not execute them because their provenance is unknowable.
        if not state.safe_mode:
            try:
                import pkgutil

                import providers as _pkg

                for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
                    if modname.startswith("_") or modname == "base":
                        continue
                    known_provider_ids.add(modname)
                    if not state.is_active(
                        name=modname,
                        key=f"model-providers/{modname}",
                        source="legacy",
                        kind="model-provider",
                    ):
                        continue
                    module_name = f"providers.{modname}"
                    registry_snapshot = dict(build_state.registry)
                    aliases_snapshot = dict(build_state.aliases)
                    origins_snapshot = dict(build_state.origins)
                    try:
                        if module_name in sys.modules:
                            module = sys.modules[module_name]
                            profiles = _MODULE_REGISTRATIONS.get(module_name)
                            if profiles is None:
                                profiles = _module_provider_profiles(module)
                        else:
                            module = importlib.import_module(module_name)
                            profiles = tuple(
                                profile
                                for provider_id, profile in build_state.registry.items()
                                if registry_snapshot.get(provider_id) is not profile
                            )
                        _MODULE_REGISTRATIONS[module_name] = profiles
                        origin = (
                            "legacy",
                            _path_identity(
                                Path(getattr(module, "__file__", None) or modname)
                            ),
                        )
                        _MODULE_PATHS[module_name] = origin[1]
                        for profile in profiles:
                            register_provider(profile)
                            build_state.origins[profile.name] = origin
                            _record_observed_profile_locked(
                                "legacy", origin[1], profile
                            )
                            active_provider_ids.add(profile.name)
                            active_provider_ids.update(profile.aliases)
                            managed_provider_ids.update(
                                {profile.name, *profile.aliases}
                                - activation_exempt_ids
                            )
                            known_provider_ids.add(profile.name)
                            known_provider_ids.update(profile.aliases)
                        _IMPORTED_PROVIDER_MODULES.add(module_name)
                    except Exception as exc:
                        build_state.registry.clear()
                        build_state.registry.update(registry_snapshot)
                        build_state.aliases.clear()
                        build_state.aliases.update(aliases_snapshot)
                        build_state.origins.clear()
                        build_state.origins.update(origins_snapshot)
                        _MODULE_REGISTRATIONS.pop(module_name, None)
                        _MODULE_PATHS.pop(module_name, None)
                        for imported_name in tuple(sys.modules):
                            if imported_name == module_name or imported_name.startswith(
                                f"{module_name}."
                            ):
                                sys.modules.pop(imported_name, None)
                        logger.warning(
                            "Failed to import legacy provider module %s: %s",
                            modname,
                            exc,
                        )
            except Exception:
                pass

        observed_ids = _OBSERVED_PROVIDER_IDS_BY_SCOPE.setdefault(
            scope_identity, set()
        )
        observed_ids.difference_update(activation_exempt_ids)
        observed_ids.update(managed_provider_ids)
        managed_provider_ids.update(observed_ids)
        known_provider_ids.update(observed_ids)

        profiles = tuple(_dedupe_profiles(build_state.registry.values()))
        return ProviderCatalogSnapshot(
            scope_identity=scope_identity,
            fingerprint=fingerprint,
            activation=state,
            registry=MappingProxyType(dict(build_state.registry)),
            aliases=MappingProxyType(dict(build_state.aliases)),
            origins=MappingProxyType(dict(build_state.origins)),
            profiles=profiles,
            plugin_managed_provider_ids=frozenset(managed_provider_ids),
            active_plugin_provider_ids=frozenset(active_provider_ids),
            bundled_provider_ids=frozenset(bundled_provider_ids),
            known_provider_ids=frozenset(known_provider_ids),
        )
    finally:
        try:
            del _BUILD_LOCAL.state
        except AttributeError:
            pass
        _discovering = False
