"""Process- and profile-scoped lifecycle for plugin managers."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_constants import get_hermes_home, hermes_home_key
from plugin_runtime.loading import (
    _BARE_MODULE_SCOPE,
    _MODULE_NAMESPACE_LOCK,
    _NS_PARENT,
    _evict_modules,
    in_plugin_load_worker,
)
from plugin_runtime.manager import PluginManager
from plugin_runtime.debug import install_plugin_debug_handler
from plugin_runtime.host_bindings import get_plugin_host_callback

logger = logging.getLogger("hermes_cli.plugins")

# Runtime-owned process state. Public callers use get_plugin_manager().
_plugin_manager: Optional[PluginManager] = None

# Resolved Hermes home -> manager. Profiles never share registrations.
_plugin_managers_by_home: dict[Path, PluginManager] = {}
_plugin_managers_lock = threading.RLock()

# Process-wide Ink TUI / desktop host, stamped onto every profile manager.
_published_tui_message_injector: tuple[object, Callable, Optional[Callable]] | None = None
_published_tui_host_lock = threading.Lock()

# One process-wide warm-start discovery worker.
_background_discovery_thread: Optional[threading.Thread] = None
_background_discovery_lock = threading.Lock()


def _plugin_home_key() -> Path:
    """Resolved active Hermes home used to key the manager cache."""
    try:
        return get_hermes_home().expanduser().resolve()
    except Exception:
        return get_hermes_home().expanduser()


def _clear_plugin_submodules(manager: Optional[PluginManager]) -> None:
    """Purge directory-plugin modules owned by one manager/profile."""
    if manager is None:
        return
    for loaded in getattr(manager, "_plugins", {}).values():
        module_name = getattr(getattr(loaded, "module", None), "__name__", None)
        if not module_name or not module_name.startswith(f"{_NS_PARENT}."):
            continue
        _evict_modules(module_name)
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.get(module_name) == manager.scope_key:
                _BARE_MODULE_SCOPE.pop(module_name, None)


def _known_plugin_managers() -> list[PluginManager]:
    with _plugin_managers_lock:
        managers = list(dict.fromkeys(_plugin_managers_by_home.values()))
        if _plugin_manager is not None and _plugin_manager not in managers:
            managers.append(_plugin_manager)
    return managers


def publish_tui_message_host(
    owner: object,
    injector: Callable[..., bool],
    refresher: Optional[Callable[[Path, str], None]] = None,
) -> None:
    """Publish the process TUI/desktop host and stamp managers that already exist."""
    global _published_tui_message_injector
    with _published_tui_host_lock:
        _published_tui_message_injector = (owner, injector, refresher)
    for manager in _known_plugin_managers():
        manager.set_tui_message_injector(owner, injector)


def refresh_tui_plugin_sessions(note: str) -> bool:
    """Refresh open TUI/desktop sessions after a plugin goes live in the active profile."""
    with _published_tui_host_lock:
        host = _published_tui_message_injector
    if host is None or host[2] is None:
        return False
    host[2](Path(get_hermes_home()), note)
    return True


def clear_published_tui_message_host(owner: object) -> None:
    """Clear the process TUI host only when still owned by the supplied owner."""
    global _published_tui_message_injector
    with _published_tui_host_lock:
        if (
            _published_tui_message_injector is not None
            and _published_tui_message_injector[0] is owner
        ):
            _published_tui_message_injector = None
    for manager in _known_plugin_managers():
        manager.clear_tui_message_injector(owner)


def _attach_published_tui_host(manager: PluginManager) -> None:
    with _published_tui_host_lock:
        host = _published_tui_message_injector
    if host is not None and manager._tui_message_injector is None:
        manager._tui_message_injector = (host[0], host[1])


def get_plugin_manager() -> PluginManager:
    """Return the manager for the active Hermes profile/home."""
    install_plugin_debug_handler()
    global _plugin_manager
    current_home = _plugin_home_key()
    with _plugin_managers_lock:
        if _plugin_manager is not None and _plugin_manager not in _plugin_managers_by_home.values():
            _plugin_managers_by_home[current_home] = _plugin_manager
            manager = _plugin_manager
        else:
            manager = _plugin_managers_by_home.get(current_home)
            if manager is None:
                manager = PluginManager(scope_key=hermes_home_key(current_home))
                _plugin_managers_by_home[current_home] = manager
            _plugin_manager = manager
    _attach_published_tui_host(manager)
    return manager


def reset_plugin_managers_for_tests() -> None:
    """Drop cached managers, owned plugin modules, and process-global host state."""
    global _plugin_manager, _published_tui_message_injector
    with _plugin_managers_lock:
        managers = list(dict.fromkeys(_plugin_managers_by_home.values()))
        if _plugin_manager is not None and _plugin_manager not in managers:
            managers.append(_plugin_manager)
        for manager in managers:
            _clear_plugin_submodules(manager)
            try:
                manager.unload()
            except Exception:
                logger.debug("test plugin-manager unload failed", exc_info=True)
        _plugin_managers_by_home.clear()
        _plugin_manager = None
    with _published_tui_host_lock:
        _published_tui_message_injector = None

    clear_providers = get_plugin_host_callback("dashboard_auth_clear")
    if clear_providers is not None:
        try:
            clear_providers()
        except Exception:
            logger.debug("dashboard-auth registry clear failed", exc_info=True)


def has_enabled_agent_plugin_mcp(raw_config: Mapping[str, Any]) -> bool:
    """Probe portable MCP manifests without mutating the live registry."""
    return PluginManager().has_enabled_portable_mcp(raw_config)


def discover_plugins(force: bool = False) -> None:
    """Discover/load the active profile after joining warm-start discovery."""
    if PluginManager._plugin_dispatch_safe_worker_enabled():
        return
    join_background_discovery()
    get_plugin_manager().discover_and_load(force=force)


def start_background_plugin_discovery() -> None:
    """Start one daemon discovery worker for the active profile when needed."""
    if PluginManager._plugin_dispatch_safe_worker_enabled():
        return

    global _background_discovery_thread
    manager = get_plugin_manager()
    if manager._discovered:
        return

    with _background_discovery_lock:
        if _background_discovery_thread is not None and _background_discovery_thread.is_alive():
            return

        def _run() -> None:
            try:
                manager.discover_and_load()
                _persist_plugin_toolset_keys()
            except Exception:
                logger.warning("background plugin discovery failed", exc_info=True)

        _background_discovery_thread = threading.Thread(
            target=_run,
            name="plugin-discovery",
            daemon=True,
        )
        _background_discovery_thread.start()


def join_background_discovery(timeout: float = 30.0) -> None:
    """Join the warm-start worker unless doing so would self-deadlock."""
    thread = _background_discovery_thread
    if (
        thread is None
        or not thread.is_alive()
        or thread is threading.current_thread()
        or in_plugin_load_worker()
    ):
        return
    thread.join(timeout=timeout)


def _plugin_toolset_keys_cache_path() -> Path:
    return get_hermes_home() / "cache" / "plugin_toolset_keys.json"


def _live_plugin_toolset_keys(manager: PluginManager) -> set[str]:
    if not manager._plugin_tool_names:
        return set()
    try:
        from tools.registry import registry
    except Exception:
        return set()

    keys: set[str] = set()
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            keys.add(entry.toolset)
    return keys


def _persist_plugin_toolset_keys() -> None:
    """Persist plugin toolset keys and portable MCP names for nonblocking startup probes."""
    try:
        from utils import atomic_json_write

        manager = get_plugin_manager()
        keys = sorted(_live_plugin_toolset_keys(manager))
        try:
            portable = sorted(manager.get_portable_mcp_servers())
        except Exception:
            portable = []
        atomic_json_write(
            _plugin_toolset_keys_cache_path(),
            {"toolset_keys": keys, "portable_mcp": portable},
            indent=None,
            mode=0o600,
        )
    except Exception:
        logger.debug("plugin toolset key persist failed", exc_info=True)


def _nowait_plugin_set(
    cache_field: str,
    live: Callable[[PluginManager], set[str]],
) -> set[str]:
    manager = get_plugin_manager()
    thread = _background_discovery_thread
    in_flight = thread is not None and thread.is_alive()
    if manager._discovered and not in_flight:
        return live(manager)
    if in_flight:
        with suppress(Exception):
            blob = json.loads(_plugin_toolset_keys_cache_path().read_text(encoding="utf-8-sig"))
            values = blob.get(cache_field) if isinstance(blob, dict) else None
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                return set(values)
    discover_plugins()
    return live(manager)


def get_plugin_toolset_keys_nowait() -> set[str]:
    """Return plugin toolset keys without waiting when a warm-start cache is available."""
    return _nowait_plugin_set("toolset_keys", _live_plugin_toolset_keys)


def get_portable_mcp_server_names_nowait() -> set[str]:
    """Return portable MCP names without waiting when a warm-start cache is available."""
    return _nowait_plugin_set(
        "portable_mcp",
        lambda manager: set(manager.get_portable_mcp_servers()),
    )


def ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the active manager after idempotent discovery."""
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def delivery_manager() -> PluginManager:
    """Return the active manager, discovering before hook/middleware delivery if needed."""
    manager = get_plugin_manager()
    if not getattr(manager, "_discovered", True):
        join_background_discovery()
        manager.discover_and_load()
    return manager