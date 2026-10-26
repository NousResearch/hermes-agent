"""Which delegation backend a session uses: the public API behind a ``/delegation`` command.

``delegation.backend`` in config.yaml is the global choice. A session may override it (the ``/model``
convention: session-scoped by default, ``--global`` persists); the override decides routing from the
session's next ``delegate_task`` call. The ``delegate_task`` schema (a backend's per-task parameters)
is part of the cached prompt and keeps following config.yaml, so a backend's own task parameters
appear in sessions started after a global switch. After a session switch the model may still send the
previous backend's parameters: each such task is refused (naming the session's backend, the parameter,
and the fix: ``/delegation <backend> --global`` then ``/new``, since ``/new`` starts on the saved backend, or
``/delegation <saved backend>`` back), never run on the new backend's default route.

Session keys are the ``session_key`` a plugin command handler receives: the gateway/TUI route key, or
the session id on the classic in-process CLI. ``delegate_task`` resolves the same key from the turn's
session context, else from the parent agent's session id.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from agent.delegation_registry import DEFAULT_BACKEND, _registry, backend_status, configured_backend
from agent.provider_registry import lower_key

_session_backends: dict[str, str] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class BackendInfo:
    """One selectable delegation backend: ``available`` False carries the ``reason`` it cannot run."""

    name: str
    label: str
    description: str
    available: bool
    reason: str = ""


def _info(name: str) -> BackendInfo:
    provider, reason = backend_status(name)
    if provider is None:
        return BackendInfo(name, name, "", False, reason)
    return BackendInfo(name, provider.display_name, provider.summary, not reason, reason)


def list_backends() -> list[BackendInfo]:
    """``default`` first, then every registered backend (unavailable ones included, with their reason)."""
    names = [DEFAULT_BACKEND, *sorted({p.name for p in _registry.list_providers()} - {DEFAULT_BACKEND})]
    return [_info(name) for name in names]


def get_session_backend(session_key: str) -> Optional[str]:
    """The session's override, ``None`` when it follows config.yaml."""
    with _lock:
        return _session_backends.get(session_key) if session_key else None


def set_session_backend(session_key: str, name: Optional[str]) -> None:
    """Switch one session's backend; ``None``/blank clears the override. Raises ``ValueError`` (with the
    reason) for a backend that cannot run here, so a typo never leaves the session broken."""
    if not session_key:
        raise ValueError("No session to switch: delegation backends are switched per session.")
    if name is None or not name.strip():
        with _lock:
            _session_backends.pop(session_key, None)
        return
    key = lower_key(name)
    info = _info(key)
    if not info.available:
        raise ValueError(info.reason)
    with _lock:
        _session_backends[session_key] = key


def set_global_backend(name: str, *, session_key: str = "") -> None:
    """Persist ``delegation.backend`` to the active profile's config.yaml (raw user config, never merged
    defaults) and drop ``session_key``'s override, which would otherwise shadow it. Raises ``PermissionError``
    on a package-manager-managed install (its config.yaml is not ours to write) and ``ValueError`` for a backend
    that cannot run."""
    from hermes_cli.config import (
        atomic_config_write, format_managed_message, get_config_path, is_managed, read_user_config_raw,
    )
    if is_managed():
        raise PermissionError(format_managed_message("set delegation.backend"))
    key = lower_key(name)
    info = _info(key)
    if not info.available:
        raise ValueError(info.reason)
    path = get_config_path()
    user_config = read_user_config_raw(path)
    section = user_config.get("delegation")
    if not isinstance(section, dict):
        section = user_config["delegation"] = {}
    section["backend"] = key
    atomic_config_write(path, user_config)
    if session_key:
        set_session_backend(session_key, None)


def _current_session_key(parent_agent: Any) -> str:
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", "") or str(getattr(parent_agent, "session_id", "") or "")


def session_delegation_config(config: Mapping[str, Any], parent_agent: Any) -> Mapping[str, Any]:
    """``config`` (the ``delegation`` section) with the calling session's backend override applied."""
    override = get_session_backend(_current_session_key(parent_agent))
    if override is None or override == configured_backend(config):
        return config
    return {**config, "backend": override}


class DelegationBackends:
    """``ctx.delegation_backends``: this API for plugins, as plain data. It is a host remote facade, so a plugin
    running under ``plugins.isolation: host`` lists and switches the owner process's backends and sessions, not
    copies in its own process."""

    def list_backends(self) -> list[dict[str, Any]]:
        return [asdict(info) for info in list_backends()]

    def get_session_backend(self, session_key: str) -> Optional[str]:
        return get_session_backend(session_key)

    def set_session_backend(self, session_key: str, name: Optional[str]) -> None:
        set_session_backend(session_key, name)

    def set_global_backend(self, name: str, session_key: str = "") -> None:
        set_global_backend(name, session_key=session_key)
