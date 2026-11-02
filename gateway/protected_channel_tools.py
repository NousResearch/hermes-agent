"""Opt-in tool selection for an exact gateway channel and receiving bot.

The selected catalog is frozen per turn, and the agent executor checks the same
selection before every dispatch. This is defense in depth for model tool calls;
the target services still own authorization and content access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class ProtectedChannelRejected(RuntimeError):
    """A protected location could not be bound to its configured receiving bot."""


@dataclass(frozen=True)
class ProtectedChannelToolsets:
    platform: str
    chat_id: str
    scope_id: Optional[str]
    bot_profile: str
    toolsets: tuple[str, ...]


def parse_protected_channel_toolsets(raw: Any) -> list[ProtectedChannelToolsets]:
    """Reject invalid entries instead of falling back to the platform catalog."""
    if not isinstance(raw, list):
        raise ValueError("gateway.protected_channel_toolsets must be a list")
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    policies: list[ProtectedChannelToolsets] = []
    seen: set[tuple[str, str, Optional[str], str]] = set()
    fields = {"platform", "chat_id", "scope_id", "bot_profile", "toolsets"}
    for index, entry in enumerate(raw):
        prefix = f"gateway.protected_channel_toolsets[{index}]"
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"{prefix} requires platform, chat_id, scope_id, bot_profile, toolsets")
        platform, chat_id, scope_id = (entry[k] for k in ("platform", "chat_id", "scope_id"))
        if not isinstance(platform, str) or not platform.strip():
            raise ValueError(f"{prefix}.platform is invalid")
        from gateway.config import Platform
        try:
            Platform(platform)
        except ValueError as exc:
            raise ValueError(f"{prefix}.platform is unknown") from exc
        if not isinstance(chat_id, str) or not chat_id.strip():
            raise ValueError(f"{prefix}.chat_id is invalid")
        if scope_id is not None and (not isinstance(scope_id, str) or not scope_id.strip()):
            raise ValueError(f"{prefix}.scope_id is invalid")
        if platform == "slack" and not scope_id:
            raise ValueError(f"{prefix}.scope_id is required for Slack")
        bot = normalize_profile_name(entry["bot_profile"])
        validate_profile_name(bot)
        toolsets = entry["toolsets"]
        if (not isinstance(toolsets, list) or any(not isinstance(t, str) or not t.strip() for t in toolsets)
                or len(toolsets) != len(set(toolsets))):
            raise ValueError(f"{prefix}.toolsets must be a list of distinct names")
        key = (platform, chat_id, scope_id, bot)
        if key in seen:
            raise ValueError(f"{prefix} duplicates a protected channel")
        seen.add(key)
        policies.append(ProtectedChannelToolsets(platform, chat_id, scope_id, bot, tuple(toolsets)))
    return policies


def policy_for_source(config: Any, source: Any, *, transport_profile: str, live_adapter: bool):
    """Resolve a protected location, refusing an unknown workspace or receiving bot.

    Channel ids can be reused by different workspaces or bot installations. Once an
    operator protects a platform/channel pair, a partial match is never ordinary
    traffic: it must not inherit a broader platform catalog.
    """
    policies = getattr(config, "protected_channel_toolsets", ()) or ()
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    chat_id = getattr(source, "chat_id", None)
    candidates = [p for p in policies if p.platform == platform and p.chat_id == chat_id]
    if not candidates:
        return None
    scope_id = getattr(source, "scope_id", None)
    matches = [p for p in candidates if p.scope_id == scope_id and p.bot_profile == transport_profile]
    if not live_adapter or len(matches) != 1:
        raise ProtectedChannelRejected("protected channel has no verified receiving bot and workspace")
    return matches[0]


def effective_toolsets(policy: ProtectedChannelToolsets) -> list[str]:
    """Never let a protected turn gain connected ambient MCP servers later."""
    names = list(policy.toolsets)
    if "no_mcp" not in names:
        names.append("no_mcp")
    return names


def allowed_tool_names(policy: ProtectedChannelToolsets, disabled_toolsets: Optional[list[str]] = None) -> frozenset[str]:
    """Use the actual selected schemas, including Tool Search bridge names."""
    from model_tools import get_tool_definitions

    names: set[str] = set()
    for toolset in policy.toolsets:
        # The sentinel prevents post-connect MCP catalog expansion in the agent;
        # it intentionally has no tool definitions of its own.
        if toolset == "no_mcp":
            continue
        definitions = get_tool_definitions(
            enabled_toolsets=[toolset], disabled_toolsets=disabled_toolsets, quiet_mode=True,
            skip_tool_search_assembly=True)
        selected = {d["function"]["name"] for d in definitions}
        if not selected:
            raise ProtectedChannelRejected(f"protected channel toolset {toolset!r} is unavailable")
        names.update(selected)
    if names:
        definitions = get_tool_definitions(
            enabled_toolsets=effective_toolsets(policy), disabled_toolsets=disabled_toolsets, quiet_mode=True)
        names.update(d["function"]["name"] for d in definitions)
    return frozenset(names)
