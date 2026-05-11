from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_DEFAULT_TOOL_LIMITS = {
    # Tools with per-session caps
    "web_search": 15,
    "web_extract": 10,
    "browser": 20,
    "image_generate": 3,
    "delegate_task": 2,
    "text_to_speech": 0,   # disabled
    "video_analyze": 2,
    "vision_analyze": 5,
    "cronjob": 0,          # disabled
    "send_message": 0,     # disabled
    "execute_code": 10,
    # Tools unlimited within iteration budget (-1 = unlimited)
    "terminal": -1,
    "read_file": -1,
    "write_file": -1,
    "patch": -1,
    "search_files": -1,
    "memory": -1,
    "session_search": -1,
    "skill_view": -1,
    "skills_list": -1,
    "todo": -1,
    "clarify": -1,
}




@dataclass
class DaimonConfig:
    """Configuration for the Daimon multi-user access control layer."""

    admin_users: list[str] = field(default_factory=list)
    admin_roles: list[str] = field(default_factory=list)
    user_users: list[str] = field(default_factory=list)
    user_roles: list[str] = field(default_factory=list)
    debug_force_tier: str | None = None
    user_model: str = "xiaomi/mimo-v2.5-pro"
    admin_model: str = "anthropic/claude-sonnet-4.6"
    max_iterations: int = 30
    max_threads_per_day: int = 5
    max_turns_per_thread: int = 20
    max_buffer_per_thread: int = 50
    gateway_timeout: int = 600
    max_active_sessions: int = 50
    queue_enabled: bool = True
    per_user_concurrent: bool = True
    tool_limits: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_TOOL_LIMITS))
    responders: list[str] = field(default_factory=lambda: ["creator", "admins"])


def load_daimon_config(raw_config: dict[str, Any]) -> DaimonConfig:
    """Load DaimonConfig from a raw config dict.

    Reads from the ``discord.daimon`` namespace in the config dict.
    User overrides merge on top of defaults. Handles YAML null/None gracefully.
    """
    # Navigate to discord.daimon namespace (guard against None at each level)
    discord = raw_config.get("discord") or {}
    daimon = discord.get("daimon") or {}

    # Build tool_limits: start with defaults, merge user overrides
    tool_limits = dict(_DEFAULT_TOOL_LIMITS)
    user_tool_limits = daimon.get("tool_limits") or {}
    if isinstance(user_tool_limits, dict):
        tool_limits.update(user_tool_limits)

    # Helper to safely get int/bool values (YAML null becomes None in Python)
    def _int(key: str, default: int) -> int:
        val = daimon.get(key)
        return int(val) if val is not None else default

    def _bool(key: str, default: bool) -> bool:
        val = daimon.get(key)
        return bool(val) if val is not None else default

    return DaimonConfig(
        admin_users=[str(u) for u in (daimon.get("admin_users") or [])],
        admin_roles=[str(r) for r in (daimon.get("admin_roles") or [])],
        user_users=[str(u) for u in (daimon.get("user_users") or [])],
        user_roles=[str(r) for r in (daimon.get("user_roles") or [])],
        debug_force_tier=daimon.get("debug_force_tier") or None,
        user_model=daimon.get("user_model") or "xiaomi/mimo-v2.5-pro",
        admin_model=daimon.get("admin_model") or "anthropic/claude-sonnet-4.6",
        max_iterations=_int("max_iterations", 30),
        max_threads_per_day=_int("max_threads_per_day", 5),
        max_turns_per_thread=_int("max_turns_per_thread", 20),
        max_buffer_per_thread=_int("max_buffer_per_thread", 50),
        gateway_timeout=_int("gateway_timeout", 600),
        max_active_sessions=_int("max_active_sessions", 50),
        queue_enabled=_bool("queue_enabled", True),
        per_user_concurrent=_bool("per_user_concurrent", True),
        tool_limits=tool_limits,
        responders=daimon.get("responders") or ["creator", "admins"],
    )
