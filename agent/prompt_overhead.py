"""Resolve model-facing prompt compaction settings.

The resolver is intentionally independent from prompt and tool assembly so
both paths apply the same platform precedence and validation rules.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_SKILL_INDEX_MODES = frozenset({"full", "compact", "minimal"})
_TOOL_SCHEMA_MODES = frozenset({"full", "compact", "minimal"})


@dataclass(frozen=True)
class PromptOverheadModes:
    """Resolved compaction modes for one platform/session."""

    skill_index_mode: str = "full"
    tool_schema_mode: str = "full"
    platform: str = ""


def current_prompt_platform() -> str:
    """Return the active platform without eagerly importing the gateway."""
    platform = os.environ.get("HERMES_PLATFORM") or os.environ.get(
        "HERMES_SESSION_PLATFORM"
    )
    if platform:
        return str(platform).strip().lower()

    session_context = sys.modules.get("gateway.session_context")
    get_session_env = (
        getattr(session_context, "get_session_env", None)
        if session_context is not None
        else None
    )
    if get_session_env is None:
        return ""
    try:
        return str(get_session_env("HERMES_SESSION_PLATFORM") or "").strip().lower()
    except Exception:
        return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _valid_mode(value: Any, allowed: frozenset[str], fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else fallback


def resolve_prompt_overhead_modes(
    config: Optional[Mapping[str, Any]] = None,
    *,
    platform: Optional[str] = None,
) -> PromptOverheadModes:
    """Resolve global modes, then override individual keys for a platform.

    Invalid global values fall back to ``full``. Invalid or missing platform
    values fall back to the corresponding resolved global value.
    """
    root = _mapping(config)
    prompt_cfg = _mapping(root.get("prompt_overhead"))
    resolved_platform = (
        str(current_prompt_platform() if platform is None else platform).strip().lower()
    )

    global_skill = _valid_mode(
        prompt_cfg.get("skill_index_mode"), _SKILL_INDEX_MODES, "full"
    )
    global_tools = _valid_mode(
        prompt_cfg.get("tool_schema_mode"), _TOOL_SCHEMA_MODES, "full"
    )

    platform_cfg: Mapping[str, Any] = {}
    for name, value in _mapping(prompt_cfg.get("platforms")).items():
        if str(name).strip().lower() == resolved_platform:
            platform_cfg = _mapping(value)
            break

    return PromptOverheadModes(
        skill_index_mode=_valid_mode(
            platform_cfg.get("skill_index_mode"), _SKILL_INDEX_MODES, global_skill
        ),
        tool_schema_mode=_valid_mode(
            platform_cfg.get("tool_schema_mode"), _TOOL_SCHEMA_MODES, global_tools
        ),
        platform=resolved_platform,
    )


def _load_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        loaded = load_config() or {}
        return loaded if isinstance(loaded, Mapping) else {}
    except Exception:
        return {}


def get_prompt_overhead_modes(*, platform: Optional[str] = None) -> PromptOverheadModes:
    """Load config and resolve the modes for the active platform."""
    return resolve_prompt_overhead_modes(_load_config(), platform=platform)


__all__ = [
    "PromptOverheadModes",
    "current_prompt_platform",
    "get_prompt_overhead_modes",
    "resolve_prompt_overhead_modes",
]
