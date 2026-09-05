"""Public, profile-scoped child-process routing for trusted session integrations.

This is cooperative execution routing, not a sandbox or an approval bypass.
Registrations are immutable snapshots; consumers keep a revocable lease, never a
mutable shared environment. No implicit task/delegate inheritance is performed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import stat
import sys
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

from hermes_constants import get_hermes_home


class SessionExecutionError(RuntimeError):
    """An execution context cannot safely be used; never fall back to the host."""


def _command_prefix(value) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("command_prefix must be an argv sequence, not shell text")
    value = tuple(value)
    if any(not isinstance(arg, str) or "\0" in arg for arg in value):
        raise ValueError("command_prefix arguments must be NUL-free strings")
    if value and not os.path.isabs(value[0]):
        raise ValueError("command_prefix executable must be an absolute path")
    return value


def _runtime_identity(path: str) -> tuple:
    try:
        info = Path(path).stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SessionExecutionError("runtime_dir must be an owned private directory")
        return (info.st_dev, info.st_ino)
    except OSError as exc:
        raise SessionExecutionError("runtime_dir unavailable") from exc


@dataclass(frozen=True)
class ComputerUseLaunchContext:
    driver_command: str | None = None
    private_daemon: bool = False
    no_overlay: bool | None = None
    session_name: str | None = None
    theme: str | None = None
    desktop_only: bool = False
    allow_input: Callable[[], bool] | None = field(default=None, repr=False)
    command_prefix: tuple[str, ...] = ()
    runtime_dir: str | None = None
    _runtime: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "command_prefix", _command_prefix(self.command_prefix))
        if type(self.private_daemon) is not bool or type(self.desktop_only) is not bool:
            raise ValueError("desktop policy flags must be booleans")
        if self.runtime_dir is not None:
            if not isinstance(self.runtime_dir, str) or not os.path.isabs(self.runtime_dir) or "\0" in self.runtime_dir:
                raise ValueError("runtime_dir must be an absolute NUL-free path")
            if not self.private_daemon:
                raise ValueError("runtime_dir requires a private daemon")
            object.__setattr__(self, "_runtime", _runtime_identity(self.runtime_dir))
        if self.allow_input is not None and not callable(self.allow_input):
            raise TypeError("allow_input must be callable")
        if self.no_overlay is not None and type(self.no_overlay) is not bool:
            raise ValueError("no_overlay must be a boolean or None")
        if self.desktop_only and not self.private_daemon:
            raise ValueError("desktop-only scope requires a private daemon")
        for value in (self.driver_command, self.session_name, self.theme):
            if value is not None and (not isinstance(value, str) or not value or "\0" in value):
                raise ValueError("launch values must be nonempty NUL-free strings")
        if self.driver_command is not None and not os.path.isabs(self.driver_command):
            raise ValueError("driver_command must be an absolute executable path")


def _desktop_endpoint(env: Mapping[str, str]) -> Path:
    runtime, display = env.get("XDG_RUNTIME_DIR"), env.get("WAYLAND_DISPLAY")
    if not runtime or not display or not os.path.isabs(runtime):
        raise SessionExecutionError("desktop-only context requires explicit runtime and Wayland display")
    return (Path(runtime) / display).resolve()


def _desktop_identity(env: Mapping[str, str]) -> tuple:
    endpoint = _desktop_endpoint(env)
    try:
        runtime = Path(env["XDG_RUNTIME_DIR"]).resolve()
        info, directory = endpoint.stat(), runtime.stat()
        if (endpoint.parent != runtime or not stat.S_ISSOCK(info.st_mode)
                or directory.st_uid != os.getuid() or info.st_uid != os.getuid()
                or directory.st_mode & 0o077):
            raise SessionExecutionError("desktop endpoint must be in an owned private runtime directory")
        return (str(endpoint), info.st_dev, info.st_ino)
    except OSError as exc:
        raise SessionExecutionError("desktop endpoint unavailable") from exc


@dataclass(frozen=True)
class SessionExecutionContext:
    env_set: Mapping[str, str] = field(default_factory=dict)
    env_unset: frozenset[str] = field(default_factory=frozenset)
    computer_use: ComputerUseLaunchContext | None = None
    validate: Callable[[], bool] | None = field(default=None, repr=False)
    command_prefix: tuple[str, ...] = ()
    _desktop: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "command_prefix", _command_prefix(self.command_prefix))
        values, removed = dict(self.env_set), frozenset(self.env_unset)
        for key in set(values) | removed:
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("execution environment keys must be shell variable names")
        if set(values) & removed:
            raise ValueError("execution environment cannot both set and unset a key")
        if any(not isinstance(v, str) or "\0" in v for v in values.values()):
            raise ValueError("execution environment values must be NUL-free strings")
        object.__setattr__(self, "env_set", MappingProxyType(values))
        object.__setattr__(self, "env_unset", removed)
        if self.computer_use is not None and not isinstance(self.computer_use, ComputerUseLaunchContext):
            raise TypeError("computer_use must be a ComputerUseLaunchContext")
        if self.computer_use is not None and self.computer_use.desktop_only:
            object.__setattr__(self, "_desktop", _desktop_identity(values))

    def check(self):
        prefixes = [self.command_prefix]
        if self.computer_use is not None:
            prefixes.append(self.computer_use.command_prefix)
        for prefix in prefixes:
            if prefix and not (os.path.isfile(prefix[0]) and os.access(prefix[0], os.X_OK)):
                raise SessionExecutionError("configured command_prefix executable unavailable")
        if self.computer_use is not None:
            command = self.computer_use.driver_command
            launch = self.computer_use
            if launch.runtime_dir is not None and _runtime_identity(launch.runtime_dir) != launch._runtime:
                raise SessionExecutionError("runtime_dir was replaced")
            if command is not None and (not os.path.isfile(command) or not os.access(command, os.X_OK)):
                raise SessionExecutionError("configured driver executable unavailable")
        if self._desktop is not None:
            if self.env_set.get("CUA_DRIVER_RS_ENABLE_WAYLAND") != "1":
                raise SessionExecutionError("desktop-only context requires explicit native Wayland enablement")
            for key in ("DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "HYPRLAND_INSTANCE_SIGNATURE", "YDOTOOL_SOCKET",
                        "CUA_INJECT_SOCKET", "SWAYSOCK", "AT_SPI_BUS_ADDRESS", "SESSION_MANAGER", "XAUTHORITY"):
                host = os.environ.get(key)
                if host and key not in self.env_unset and self.env_set.get(key, host) == host:
                    raise SessionExecutionError(f"desktop context retains host {key}; unset or replace it explicitly")
            if _desktop_identity(self.env_set) != self._desktop:
                raise SessionExecutionError("desktop endpoint was replaced")
            if os.environ.get("XDG_RUNTIME_DIR") and os.environ.get("WAYLAND_DISPLAY"):
                if str(_desktop_endpoint(os.environ)) == self._desktop[0]:
                    raise SessionExecutionError("desktop context points at the host display")
        if self.validate is not None:
            try:
                valid = self.validate() is True
            except Exception as exc:
                raise SessionExecutionError("execution context validation failed") from exc
            if not valid:
                raise SessionExecutionError("execution context validation failed")

    def apply_env(self, base: Mapping[str, str]) -> dict[str, str]:
        self.check()
        return {k: v for k, v in dict(base, **self.env_set).items() if k not in self.env_unset}


@dataclass(frozen=True)
class SessionExecutionLease:
    home: str
    session_id: str
    context: SessionExecutionContext
    generation: str = field(default_factory=lambda: uuid4().hex)

    @property
    def cache_key(self) -> str:
        return f"execution:{self.generation}"

    def check(self):
        with _lock:
            if _contexts.get((self.home, self.session_id)) is not self:
                raise SessionExecutionError("execution context was revoked or replaced")
        self.context.check()

    def apply_env(self, base: Mapping[str, str]) -> dict[str, str]:
        self.check()
        return self.context.apply_env(base)

    def wrap_argv(self, argv, *, computer_use: bool = False) -> list[str]:
        self.check()
        launch = self.context.computer_use if computer_use else None
        return [*self.context.command_prefix, *(launch.command_prefix if launch else ()), *argv]


_lock = threading.RLock()
_contexts: dict[tuple[str, str], SessionExecutionLease] = {}
_aliases: dict[tuple[str, str], str] = {}


def _home() -> str:
    return str(get_hermes_home().resolve())


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise ValueError("session/task identity must be a nonempty exact string")
    return value


def register_session_execution_context(session_id: str, context: SessionExecutionContext,
                                       *, task_ids=()) -> None:
    """Atomically register/replace an owner and its explicit task aliases.

    Register in the serving process. Separate CLI processes must communicate via
    their plugin's durable manager, not this process-local registry.
    """
    sid, home = _identifier(session_id), _home()
    if not isinstance(context, SessionExecutionContext):
        raise TypeError("context must be a SessionExecutionContext")
    aliases = {_identifier(task) for task in task_ids} | {sid}
    context.check()
    with _lock:
        for alias in aliases:
            if _aliases.get((home, alias), sid) != sid:
                raise SessionExecutionError(f"execution alias {alias!r} is already owned")
        previous = _drop_locked(home, sid)
        _contexts[home, sid] = SessionExecutionLease(home, sid, context)
        _aliases.update({(home, alias): sid for alias in aliases})
    _retire(previous)


def _drop_locked(home: str, sid: str) -> SessionExecutionLease | None:
    found = _contexts.pop((home, sid), None)
    for key, owner in list(_aliases.items()):
        if key[0] == home and owner == sid:
            del _aliases[key]
    return found


def remove_session_execution_context(session_id: str) -> bool:
    """Revoke a registration. Already running children retain their own env."""
    with _lock:
        previous = _drop_locked(_home(), _identifier(session_id))
    _retire(previous)
    return previous is not None


def _retire(previous: SessionExecutionLease | None) -> None:
    # No tool import at registration time: only retire transports that exist.
    # Lease identity prevents a concurrent replacement from releasing the new owner.
    if previous is not None and (tool := sys.modules.get("tools.computer_use.tool")) is not None:
        tool.release_computer_use_execution_context(previous)


def resolve_session_execution_context(*, session_id: str | None = None,
                                      task_id: str | None = None) -> SessionExecutionLease | None:
    """Resolve explicit identities only; disagreement fails closed."""
    home = _home()
    with _lock:
        owners = {_aliases[home, key] for key in (session_id, task_id) if key and (home, key) in _aliases}
        if len(owners) > 1:
            raise SessionExecutionError("conflicting execution context owners")
        lease = _contexts.get((home, next(iter(owners)))) if owners else None
    if lease is not None:
        lease.check()
    return lease


def get_session_execution_context(*, session_id: str | None = None,
                                  task_id: str | None = None) -> SessionExecutionContext | None:
    lease = resolve_session_execution_context(session_id=session_id, task_id=task_id)
    return lease.context if lease else None
