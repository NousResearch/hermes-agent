"""Selection snapshots for Matrix administration before a homeserver write."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hermes_cli import config, env_loader, managed_scope, plugins, tools_config
from hermes_cli.config_effective import load_user_config_effective

_SELECTION_ERROR = "Enable matrix_admin for Matrix in both the runtime and transport profiles in hermes tools"


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    link: tuple[int, ...] | None
    target: tuple[int, ...] | None

    @classmethod
    def capture(cls, path: Path) -> _FileIdentity:
        def identity(follow: bool) -> tuple[int, ...] | None:
            try:
                info = path.stat(follow_symlinks=follow)
            except FileNotFoundError:
                return None
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )

        return cls(path, identity(False), identity(True))


@dataclass(frozen=True)
class _ResolverInputs:
    registry: tuple[int, int]
    core_tools: frozenset[str]
    toolsets: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...]
    platforms: tuple[tuple[str, str], ...]
    configurable: frozenset[str]
    default_off: frozenset[str]
    config_only: frozenset[str]
    recently_shipped: frozenset[str]
    platform_restrictions: tuple[tuple[str, frozenset[str]], ...]
    plugin_manager: tuple[int, bool, tuple[str, ...]]
    discovery_thread: tuple[int, bool]
    injected_manager: int

    @classmethod
    def capture(cls, home: Path) -> _ResolverInputs:
        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
        from tools.registry import registry

        toolsets = tuple(
            (
                name,
                tuple(definition.get("tools", ())),
                tuple(definition.get("includes", ())),
                bool(definition.get("posture")),
            )
            for name, definition in TOOLSETS.items()
        )
        platforms = tuple(
            (name, definition["default_toolset"])
            for name, definition in tools_config.PLATFORMS.items()
        )
        restrictions = tuple(
            (name, frozenset(platforms))
            for name, platforms in tools_config._TOOLSET_PLATFORM_RESTRICTIONS.items()
        )
        manager = plugins._plugin_managers_by_home.get(home.resolve())
        manager_inputs = (
            id(manager),
            bool(getattr(manager, "_discovered", False)),
            tuple(sorted(getattr(manager, "_plugin_tool_names", ()))),
        )
        thread = plugins._background_discovery_thread
        thread_inputs = (id(thread), thread is not None and thread.is_alive())
        injected = plugins._plugin_manager
        injected_id = (
            id(injected)
            if injected is not None
            and injected not in plugins._plugin_managers_by_home.values()
            else 0
        )
        return cls(
            (id(registry), registry._generation),
            frozenset(_HERMES_CORE_TOOLS),
            toolsets,
            platforms,
            frozenset(key for key, _, _ in tools_config.CONFIGURABLE_TOOLSETS),
            frozenset(tools_config._DEFAULT_OFF_TOOLSETS),
            frozenset(tools_config._CONFIG_ONLY_TOOLSETS),
            frozenset(tools_config._RECENTLY_SHIPPED_TOOLSETS),
            restrictions,
            manager_inputs,
            thread_inputs,
            injected_id,
        )


@dataclass(frozen=True)
class _ProfileInputs:
    home: Path
    resolved_home: Path
    managed_directory: Path | None
    files: tuple[_FileIdentity, ...]
    process_environment: tuple[tuple[str, str], ...] = field(repr=False)
    secret_source_values: tuple[tuple[str, str], ...] = field(repr=False)
    routing_process_home: Path
    multiplex_active: bool
    global_secret_names: frozenset[str]
    global_secret_prefixes: tuple[str, ...]
    bridged_allow_all_users: str | None
    resolver: _ResolverInputs

    @classmethod
    def capture(cls, home: Path) -> _ProfileInputs:
        from agent.secret_scope import (
            _GLOBAL_ENV_EXACT,
            _GLOBAL_ENV_PREFIXES,
            is_multiplex_active,
        )
        from gateway.config_loader import bridged_allow_all_users
        from hermes_constants import (
            get_default_hermes_root,
            get_routing_process_hermes_home,
        )

        managed = managed_scope.get_managed_dir()
        default_root = get_default_hermes_root()
        paths = [
            home / "config.yaml",
            home / ".env",
            home / "auth.json",
            default_root / "auth.json",
            home / "cache" / "plugin_toolset_keys.json",
        ]
        candidate = os.environ.get("HERMES_MANAGED_DIR", "").strip()
        if candidate:
            paths.append(Path(candidate))
        elif not managed_scope._under_pytest():
            paths.append(managed_scope._DEFAULT_MANAGED_DIR)
        if managed is not None:
            paths.extend((managed / "config.yaml", managed / ".env"))
        return cls(
            home,
            home.resolve(),
            managed,
            tuple(_FileIdentity.capture(path) for path in dict.fromkeys(paths)),
            tuple(sorted(os.environ.items())),
            tuple(sorted(env_loader.get_secret_source_values(home).items())),
            get_routing_process_hermes_home(),
            is_multiplex_active(),
            frozenset(_GLOBAL_ENV_EXACT),
            tuple(_GLOBAL_ENV_PREFIXES),
            bridged_allow_all_users(),
            _ResolverInputs.capture(home),
        )

    def current(self) -> bool:
        try:
            return self == self.capture(self.home)
        except (OSError, RuntimeError):
            return False


@dataclass(frozen=True)
class _ProfileSelection:
    inputs: _ProfileInputs
    selected: bool
    stable: bool
    user_references: tuple[tuple[str, str | None], ...] = field(repr=False)
    managed_references: tuple[tuple[str, str | None], ...] = field(repr=False)
    process_reference_values: tuple[tuple[str, str | None], ...] = field(repr=False)

    @classmethod
    def capture(cls, home: Path) -> _ProfileSelection:
        from gateway.run import _profile_runtime_scope

        for _attempt in range(2):
            before = _ProfileInputs.capture(home)
            with _profile_runtime_scope(home, hydrate_secrets=False):
                effective = load_user_config_effective(fail_closed=True)
                path = str(config.get_config_path())
                with config._CONFIG_LOCK:
                    user_hit = config._RAW_CONFIG_CACHE.get(path)
                    user_sig, _ = config._load_config_cache_sig(
                        config.get_config_path()
                    )
                    raw = (
                        user_hit[4]
                        if user_sig is not None
                        and user_hit is not None
                        and user_hit[:4] == user_sig
                        else {}
                    )
                    user_references = tuple(
                        sorted(config._env_ref_snapshot(raw).items())
                    )
                    managed = managed_scope.load_managed_config()
                    managed_references = tuple(
                        sorted(config._env_ref_snapshot(managed).items())
                    )
                keys = set(dict(user_references)) | set(dict(managed_references))
                process_values = tuple(
                    (key, os.environ.get(key)) for key in sorted(keys)
                )
                selected = "matrix_admin" in tools_config._get_platform_tools(
                    effective,
                    "matrix",
                    include_default_mcp_servers=False,
                )
            after = _ProfileInputs.capture(home)
            stable = before == after
            if stable:
                break
        return cls(
            after, selected, stable, user_references, managed_references, process_values
        )

    def current(self) -> bool:
        return self.stable and self.inputs.current()


@dataclass(frozen=True)
class MatrixAdminSelection:
    profiles: tuple[_ProfileSelection, ...]

    @classmethod
    def capture(cls, homes: tuple[Path, ...]) -> MatrixAdminSelection:
        result = cls(
            tuple(_ProfileSelection.capture(home) for home in dict.fromkeys(homes))
        )
        if not all(profile.selected for profile in result.profiles):
            raise ValueError(_SELECTION_ERROR)
        return result

    def current(self) -> bool:
        return all(profile.current() for profile in self.profiles)

    def require_current(self) -> None:
        if not self.current():
            raise ValueError(_SELECTION_ERROR)
