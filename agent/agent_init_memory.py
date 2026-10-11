"""Memory policy helpers used while constructing and dispatching an agent."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

_GATEWAY_IDENTITY_PARAMS = (
    "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
    "gateway_session_key",
)
_MEMORY_MODES = frozenset({"full", "on_demand", "off"})
_MCP_MEMORY_READ_TOOLS = frozenset({"hermes_mem_search", "hermes_mem_get"})


def _memory_provider_init_kwargs(agent, platform) -> Dict[str, Any]:
    """Scoping kwargs for ``MemoryManager.initialize_all`` (status_callback is CLI-only:
    gateway status travels a different path and the indicator no-ops without it)."""
    kwargs = {
        "session_id": agent.session_id,
        "platform": platform or "cli",
        "hermes_home": str(get_hermes_home()),
        # platform="cron" (scheduler) / "subagent" (delegate_task) → providers skip writes (MemoryProvider.initialize).
        "agent_context": platform if platform in ("cron", "subagent") else "primary",
    }
    if kwargs["platform"] == "cli":
        kwargs["warning_callback"] = agent._emit_warning
        kwargs["status_callback"] = agent._emit_status
    # Session title (e.g. honcho derives chat-scoped session keys from it).
    if agent._session_db:
        with suppress(Exception):
            _st = agent._session_db.get_session_title(agent.session_id)
            if _st:
                kwargs["session_title"] = _st
                _source = agent._session_db.get_session_title_source(agent.session_id)
                if _source:
                    kwargs["session_title_source"] = _source
    # Gateway user/chat identity for per-user scoping (gateway_session_key: stable per-chat
    # Honcho session isolation).
    for _ident in _GATEWAY_IDENTITY_PARAMS:
        _val = getattr(agent, f"_{_ident}")
        if _val:
            kwargs[_ident] = _val
    if agent.session_cwd:
        kwargs["cwd"] = agent.session_cwd
    # Profile identity for per-profile provider scoping
    with suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        kwargs["agent_identity"] = get_active_profile_name()
        kwargs["agent_workspace"] = "hermes"
    return kwargs


def _resolve_memory_mode(memory_mode: Optional[str], skip_memory: bool) -> tuple[str, bool]:
    """Return ``(mode, explicit)`` while preserving the legacy ``skip_memory`` contract.

    ``memory_mode`` is strict when supplied.  When omitted, existing callers keep the
    historical boolean behavior: ``skip_memory=False`` is full memory and
    ``skip_memory=True`` skips the provider while still allowing the special
    ``enabled_toolsets=["memory"]`` built-in-store path used by flush agents.
    """
    if memory_mode is None:
        return ("off" if skip_memory else "full"), False
    normalized = str(memory_mode).strip().lower()
    if normalized not in _MEMORY_MODES:
        raise ValueError(
            f"Invalid memory_mode {memory_mode!r}; expected one of: full, on_demand, off"
        )
    return normalized, True


def _prune_explicit_memory_tools(agent, mode: str, allowed: set[str]) -> None:
    """Apply strict off/on-demand policy after every dynamic tool injection."""
    if mode == "full":
        return

    def _is_memory_tool(name: str) -> bool:
        if name == "memory" or name.startswith("hermes_mem_"):
            return True
        try:
            import model_tools
            return "memory" in (model_tools.get_toolset_for_tool(name) or "").lower()
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError):
            return False

    agent.tools = [
        tool for tool in (agent.tools or [])
        if (
            (name := tool.get("function", {}).get("name")) in allowed
            or not (isinstance(name, str) and _is_memory_tool(name))
        )
    ]
    agent.valid_tool_names = {
        name for name in (agent.valid_tool_names or set())
        if name in allowed or not _is_memory_tool(name)
    }


def memory_tool_call_allowed(agent, tool_name: str) -> bool:
    """Runtime backstop for explicit memory policy after schema/catalog refreshes."""
    if getattr(agent, "_memory_mode_explicit", False) is not True:
        return True
    mode = getattr(agent, "_memory_mode", "off")
    # Full primary agents retain the historical writable memory surface. Delegated
    # children remain bounded by the parent and by DELEGATE_BLOCKED_TOOLS.
    if mode == "full" and getattr(agent, "platform", None) != "subagent":
        return True
    allowed = set(getattr(agent, "_memory_tool_policy_allowlist", ()) or ())
    if tool_name in allowed:
        return True
    lowered = str(tool_name).lower()
    is_memory = (
        lowered == "memory"
        or lowered.startswith("hermes_mem_")
        or "__hermes_mem__" in lowered
    )
    manager = getattr(agent, "_memory_manager", None)
    if manager is not None:
        with suppress(Exception):
            is_memory = is_memory or manager.has_tool(tool_name)
    if not is_memory:
        with suppress(Exception):
            import model_tools
            toolset = str(model_tools.get_toolset_for_tool(tool_name) or "").lower()
            is_memory = "memory" in toolset or "hermes_mem" in toolset
    return not is_memory
