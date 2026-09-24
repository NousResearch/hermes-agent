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
_discovered = False
_discovering = False
_PLUGIN_DIR_STAMP_TTL_SECONDS = 1.0


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


_HOME_LAYERS: dict[str, _HomeLayer] = {}
_HOME_LAYERS_LOCK = threading.Lock()
# The layer a ``$HERMES_HOME`` plugin import registers into. A ContextVar, not a module global:
# two turn threads scanning two profile homes at once must not cross-register. Never a lock held
# across the import itself — a thread mid-``import hermes_cli.auth`` (whose import calls
# ``list_providers()``) would block on it while the scanning thread waits on that module's import lock.
_REGISTRATION_TARGET: ContextVar[_HomeLayer | None] = ContextVar("_provider_registration_target", default=None)

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
    global _PROVIDER_LIST_CACHE
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
    if profile is None and not is_custom_route and not checked:
        if _refresh_home_layer(layer, home, key, force=True):
            profile = lookup(name)
    # Named custom routes share the generic wire policy unless a plugin
    # explicitly registered that route. Other names retain exact lookup.
    if profile is None and is_custom_route:
        profile = lookup("custom")
    return profile


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


def _home_layer(*, force_stamp_check: bool = False) -> _HomeLayer:
    """The layer for the home bound right now, importing plugin dirs it has not seen yet."""
    layer, home, key = _bound_home_layer()
    _refresh_home_layer(layer, home, key, force=force_stamp_check)
    return layer


def _bound_home_layer() -> tuple[_HomeLayer, Path | None, str]:
    try:
        from hermes_constants import get_hermes_home, hermes_home_key

        home = get_hermes_home()
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
    if home is None or not (
        force
        or layer.stamp_checked_at is None
        or now - layer.stamp_checked_at >= _PLUGIN_DIR_STAMP_TTL_SECONDS
    ):
        return False
    stamps = _plugin_dir_stamps(home)
    if stamps != layer.stamps:
        _scan_home_layer(layer, key)
        layer.stamps = stamps
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
    consent flow. Parsed with PyYAML when available, falling back to a line
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
    if _discovered and not _discovering:
        _sync_auth_registry()


def _user_module_name(plugin_dir: Path, home_key: str) -> str:
    digest = hashlib.sha1(home_key.encode("utf-8")).hexdigest()[:10]
    return f"_hermes_user_provider_{digest}_{plugin_dir.name.replace('-', '_')}"


def _import_plugin_dir(plugin_dir: Path, source: str, *, home_key: str = "") -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user"; it is recorded per registered profile (``_SOURCES``).
    """
    global _current_source
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` under a per-home module name so
    # multiple HERMES_HOME profiles don't alias each other.
    if source == "bundled":
        module_name = f"plugins.model_providers.{plugin_dir.name.replace('-', '_')}"
    else:
        module_name = _user_module_name(plugin_dir, home_key)

    if module_name in sys.modules:
        return  # already imported

    _current_source = source
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)
    finally:
        _current_source = None


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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'OMIT_TEMPERATURE': ('providers.base', 'OMIT_TEMPERATURE'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
