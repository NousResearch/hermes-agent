"""Provider discovery mechanics for the canonical registry."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from providers import registry as _registry

logger = logging.getLogger(__name__)

_current_source: str | None = None
_discovered = False
_discovering = False
_PLUGIN_DIR_STAMP_TTL_SECONDS = 1.0
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)

def current_registration_source() -> str | None:
    return _current_source

def home_layer(*, force_stamp_check: bool = False) -> _registry._HomeLayer:
    """The layer for the home bound right now, importing plugin dirs it has not seen yet."""
    layer, home, key = bound_home_layer()
    refresh_home_layer(layer, home, key, force=force_stamp_check)
    return layer

def bound_home_layer() -> tuple[_registry._HomeLayer, Path | None, str]:
    try:
        from hermes_constants import get_hermes_home, hermes_home_key

        home = get_hermes_home()
        key = hermes_home_key(home)
    except Exception:
        home, key = None, ""
    return _registry._get_or_create_home_layer(key), home, key

def refresh_home_layer(layer: _registry._HomeLayer, home: Path | None, key: str, *, force: bool = False) -> bool:
    """Re-stat the layer's plugin dirs when due (or *force*d); True when it stat'ed this call."""
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
    """Return ``$HERMES_HOME/plugins/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None

def _declares_model_provider_kind(plugin_dir: Path) -> bool:
    """Whether ``plugin_dir``'s manifest declares ``kind: model-provider``."""
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

def _scan_home_layer(layer: _registry._HomeLayer, key: str) -> None:
    """Import the bound home's not-yet-imported provider plugins into *layer*."""
    global _discovering
    token, prior_discovering = _registry._REGISTRATION_TARGET.set(layer), _discovering
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
        _registry._REGISTRATION_TARGET.reset(token)
        _discovering = prior_discovering

def _user_module_name(plugin_dir: Path, home_key: str) -> str:
    digest = hashlib.sha1(home_key.encode("utf-8")).hexdigest()[:10]
    return f"_hermes_user_provider_{digest}_{plugin_dir.name.replace('-', '_')}"

def _import_plugin_dir(plugin_dir: Path, source: str, *, home_key: str = "") -> None:
    """Import a single plugin directory so it self-registers."""
    global _current_source
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return

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
    """Import pip-installed provider plugins via the ``hermes_agent.plugins``"""
    try:
        import importlib.metadata as _md
    except Exception:  # pragma: no cover — importlib.metadata always present ≥3.8
        return

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
    """True when ``fn`` cannot be called with zero arguments."""
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

def ensure_process_discovered() -> None:
    """Populate the process-wide registry by importing every provider plugin."""
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

def _run_discovery_steps() -> None:
    """The discovery passes, in precedence order (see :func:`ensure_process_discovered`)."""
    _discover_entry_point_providers()

    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    try:
        import pkgutil

        import providers as _pkg

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname.startswith("_") or modname in {"base", "identity", "registry", "discovery"}:
                continue
            try:
                importlib.import_module(f"providers.{modname}")
            except ImportError as exc:
                logger.warning(
                    "Failed to import legacy provider module %s: %s", modname, exc
                )
    except Exception:
        pass

@contextmanager
def isolated_plugin_import(plugin_dir: Path, source: str = "user", *, home_key: str = ""):
    """Load one provider plugin through real discovery and restore registry state on exit."""
    module_name = (
        f"plugins.model_providers.{plugin_dir.name.replace('-', '_')}"
        if source == "bundled"
        else _user_module_name(plugin_dir, home_key)
    )
    prior_module = sys.modules.pop(module_name, None)
    snapshot = _registry._snapshot_state()
    before = snapshot[0]
    try:
        _import_plugin_dir(plugin_dir, source, home_key=home_key)
        registered = tuple(
            sorted(
                name
                for name, profile in _registry._REGISTRY.items()
                if before.get(name) is not profile
            )
        )
        yield registered
    finally:
        _registry._restore_state(snapshot)
        sys.modules.pop(module_name, None)
        if prior_module is not None:
            sys.modules[module_name] = prior_module
