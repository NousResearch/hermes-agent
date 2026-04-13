"""Plan Mode hook plugin for Hermes Agent (Phase B2).

When plan mode is active, only read-only / planning tools are allowed.
Write operations and heavy tools are denied via pre_tool_call.
"""

# ── State ──────────────────────────────────────────────────────────────
_session_states: dict[str, bool] = {}

# Tools always allowed in plan mode, regardless of registry metadata.
_PLAN_OVERRIDE_ALLOW = {
    "plan_mode",
}

# Legacy fallback allowlist for tools that do not expose plan-mode metadata.
_PLAN_ALLOW = {
    "read_file", "search_files", "session_search",
    "skills_list", "skill_view", "tool_search",
}


def _is_allowed_in_plan_mode(tool_name: str) -> bool:
    if tool_name in _PLAN_OVERRIDE_ALLOW:
        return True

    try:
        from tools.registry import registry
    except Exception:
        registry = None

    if registry is not None:
        try:
            metadata = registry.get_metadata(tool_name) or {}
        except Exception:
            metadata = {}

        allowed_in_plan_mode = metadata.get("allowed_in_plan_mode_default")

        if allowed_in_plan_mode is True:
            if any(metadata.get(key) for key in (
                "mutates_local_fs",
                "mutates_agent_state",
                "mutates_browser_session",
                "mutates_external_world",
            )):
                return False
            if metadata.get("risk_level") in ("medium", "high", "critical"):
                return False
            return True
        if allowed_in_plan_mode is False:
            return False

        if any(metadata.get(key) for key in (
            "mutates_local_fs",
            "mutates_agent_state",
            "mutates_browser_session",
            "mutates_external_world",
        )):
            return False
        if metadata.get("risk_level") in ("medium", "high", "critical"):
            return False

    return tool_name in _PLAN_ALLOW


def _get_session_id(kwargs: dict) -> str:
    return kwargs.get("session_id") or "default"


def enter_plan_mode(session_id=None):
    _session_states[session_id or "default"] = True


def exit_plan_mode(session_id=None):
    _session_states.pop(session_id or "default", None)


def is_plan_mode_active(session_id=None) -> bool:
    return _session_states.get(session_id or "default", False)


def is_active(session_id=None) -> bool:
    return is_plan_mode_active(session_id=session_id)


# ── Hook entry point ────────────────────────────────────────────────────
def pre_tool_call(tool_name: str, args: dict, **kwargs) -> dict | None:
    """Called by the plugin system before each tool invocation."""
    session_id = _get_session_id(kwargs)
    if not is_plan_mode_active(session_id=session_id):
        return None

    # memory: read allowed, writes denied
    if tool_name == "memory":
        if args.get("action") == "read":
            return None
        return {
            "action": "deny",
            "reason": "Plan Mode: memory write operations not allowed",
        }

    if _is_allowed_in_plan_mode(tool_name):
        return None

    # Everything else denied in plan mode
    return {
        "action": "deny",
        "reason": f"Plan Mode: {tool_name} not allowed",
    }
