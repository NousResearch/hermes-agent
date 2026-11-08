"""Which delegation backend Hermes uses: the public API behind a ``/delegation`` command.

``delegation.backend`` in config.yaml is the saved choice. A session routes on the backend its ``delegate_task``
schema was built from (``agent._delegation_backend``, captured with its tools and pinned with them to the session
row), because that schema (a backend's per-task parameters) is cached prompt: a save, or a config.yaml edit,
changes nothing in a running chat and applies from its next session (``/new``). ``apply_now`` opts into
rebuilding the calling session's tools at its next turn instead, so routing and schema switch together there:
one prompt-cache break, the rule every slash command that changes prompt state follows (``--now``). A task
carrying parameters its session's backend does not declare is still refused (``tools/delegate_tool_routes.py``).

Session keys are the ``session_key`` a plugin command handler receives: the gateway/TUI route key, or the
session id on the classic in-process CLI. A turn resolves the same key from its session context, else from the
agent's session id.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from agent.delegation_registry import DEFAULT_BACKEND, _registry, backend_status, configured_backend
from agent.provider_registry import lower_key

_rebuild_requests: set[str] = set()
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


def saved_backend() -> str:
    """``delegation.backend`` as config.yaml has it now: what a tool build reads."""
    from tools.delegate_tool_config import _load_config
    return configured_backend(_load_config())


def _delegate_def(tools: Any) -> Any:
    return next((t for t in tools or () if isinstance(t, dict)
                 and (t.get("function") or {}).get("name") == "delegate_task"), None)


def adopt_built_backend(agent: Any, defs: Any, backend: str) -> None:
    """Record ``backend`` as the session's when the agent's ``delegate_task`` is the one ``defs`` carries (built
    from ``backend``, or pinned with it); a kept older definition keeps the backend it was built from."""
    if _delegate_def(getattr(agent, "tools", None)) == _delegate_def(defs):
        agent._delegation_backend = backend


def pinned_backend(agent: Any) -> str:
    """The backend ``agent``'s ``delegate_task`` was built from; ``""`` for no agent (or one never built), whose
    tasks route on config.yaml's."""
    pinned = getattr(agent, "_delegation_backend", None)
    return pinned if isinstance(pinned, str) else ""


def set_global_backend(name: str, *, session_key: str = "", apply_now: bool = False) -> None:
    """Persist ``delegation.backend`` to the active profile's config.yaml (raw user config, never merged
    defaults). Running sessions keep routing on the backend their tools were built from; ``apply_now`` rebuilds
    ``session_key``'s tools at that session's next turn, so it routes on and offers the new backend's parameters
    from there (an opt-in prompt-cache break); otherwise sessions pick the switch up from ``/new``. Raises ``PermissionError`` on a package-manager-managed
    install (its config.yaml is not ours to write) and ``ValueError`` (with the reason) for a backend that cannot
    run or an ``apply_now`` without a session, before anything is written."""
    from hermes_cli.config import (
        atomic_config_write, format_managed_message, get_config_path, is_managed, read_user_config_raw,
    )
    if is_managed():
        raise PermissionError(format_managed_message("set delegation.backend"))
    if apply_now and not session_key:
        raise ValueError("No session to apply the switch to: apply_now needs the command's session_key.")
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
    if apply_now:
        with _lock:
            _rebuild_requests.add(session_key)


def _current_session_key(agent: Any) -> str:
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", "") or str(getattr(agent, "session_id", "") or "")


def apply_requested_tool_rebuild(agent: Any) -> bool:
    """Turn start, after the session's tools are restored: rebuild ``agent.tools`` (and with it the backend the
    session routes on) if ``set_global_backend(apply_now=True)`` asked for this session, and pin both to the
    session row so a later fresh agent restores them. True when the tool list changed. Forks (background review, ``/btw``) and
    subagents share the session's key but never take its request."""
    if getattr(agent, "_turn_origin", None) or getattr(agent, "_delegate_depth", 0):
        return False
    key = _current_session_key(agent)
    with _lock:
        if key not in _rebuild_requests:
            return False
        _rebuild_requests.discard(key)
    from hermes_cli.observability.shared_metrics_efficiency import record_cache_break
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    before = getattr(agent, "tools", None)
    refresh_agent_mcp_tools(agent, quiet_mode=True, content_aware=True)  # adopts the saved backend, re-pins a change
    if agent.tools is before:
        return False
    record_cache_break(agent, "toolset_change")
    return True


class DelegationBackends:
    """``ctx.delegation_backends``: this API for plugins, as plain data. It is a host remote facade, so a plugin
    running under ``plugins.isolation: host`` lists and switches the owner process's backend and sessions, not
    copies in its own process."""

    def list_backends(self) -> list[dict[str, Any]]:
        return [asdict(info) for info in list_backends()]

    def set_global_backend(self, name: str, *, session_key: str = "", apply_now: bool = False) -> None:
        set_global_backend(name, session_key=session_key, apply_now=apply_now)
