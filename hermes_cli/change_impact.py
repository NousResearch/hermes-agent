"""Runtime impact advisories for configuration and tool changes.

This module is intentionally small and side-effect-free: callers that mutate
config/env/tool state can ask what kind of refresh is needed and print the
same concise reminder everywhere instead of each surface guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ImpactScope(str, Enum):
    """How far a change reaches before it becomes visible."""

    IMMEDIATE = "immediate"
    NEW_SESSION = "new_session"
    GATEWAY_RESTART = "gateway_restart"
    PROCESS_RESTART = "process_restart"


@dataclass(frozen=True)
class ChangeAdvisory:
    """User-facing refresh guidance for a just-applied change."""

    scope: ImpactScope
    action: str
    reason: str
    title: str = "Runtime note"


_GATEWAY_ENV_PREFIXES = (
    "DISCORD_",
    "TELEGRAM_",
    "SLACK_",
    "SIGNAL_",
    "SMS_",
    "MATRIX_",
    "MATTERMOST_",
    "DINGTALK_",
    "FEISHU_",
    "WECOM_",
    "WEIXIN_",
    "YUANBAO_",
    "WHATSAPP_",
    "IMESSAGE_",
    "PHOTON_",
    "LINE_",
    "SIMPLEX_",
    "NTFY_",
    "GOOGLE_CHAT_",
    "TEAMS_",
)

_PROCESS_RESTART_KEYS = {
    "security.redact_secrets": ChangeAdvisory(
        scope=ImpactScope.PROCESS_RESTART,
        action="restart Hermes",
        reason="Secret redaction is snapshotted by the running process; restart before expecting this setting to change live behavior.",
    ),
}

_NEW_SESSION_KEYS = {
    "model",
    "toolsets",
    "mcp_servers",
    "custom_providers",
}

_NEW_SESSION_PREFIXES = (
    "model.",
    "fallback_model",
    "auxiliary.",
    "agent.",
    "delegation.",
    "terminal.",
    "display.",
    "compression.",
    "memory.",
    "toolsets",
    "tools.",
    "mcp_servers",
    "custom_providers",
)

_GATEWAY_RESTART_PREFIXES = (
    "gateway.",
    "platforms.",
    "stt.",
    "tts.",
    "cron.",
    "kanban.",
)


def _normalize_key(key: str) -> str:
    return str(key or "").strip()


def advisory_for_config_key(key: str) -> Optional[ChangeAdvisory]:
    """Return refresh guidance for a config/env key, if one is useful.

    The mapping deliberately favors concrete, actionable reminders over broad
    noise. Unknown keys return ``None`` so callers do not nag users for every
    harmless save.
    """

    raw_key = _normalize_key(key)
    if not raw_key:
        return None

    lower = raw_key.lower()
    upper = raw_key.upper()

    if lower in _PROCESS_RESTART_KEYS:
        return _PROCESS_RESTART_KEYS[lower]

    if upper.startswith(_GATEWAY_ENV_PREFIXES):
        return ChangeAdvisory(
            scope=ImpactScope.GATEWAY_RESTART,
            action="hermes gateway restart",
            reason="Gateway credentials/channel settings are read by the running gateway process; restart it before expecting messaging behavior to change.",
        )

    if upper.startswith("TERMINAL_SSH"):
        return ChangeAdvisory(
            scope=ImpactScope.NEW_SESSION,
            action="/new",
            reason="Terminal backend credentials/settings are resolved by new sessions; current terminal environments may keep using the old value.",
        )

    if lower.startswith("platform_toolsets"):
        parts = lower.split(".")
        platform = parts[1] if len(parts) > 1 else ""
        if platform and platform != "cli":
            return ChangeAdvisory(
                scope=ImpactScope.GATEWAY_RESTART,
                action="hermes gateway restart",
                reason="Gateway platform tool schemas are built from platform toolsets when sessions start; restart the gateway to rebuild them cleanly.",
            )
        return ChangeAdvisory(
            scope=ImpactScope.NEW_SESSION,
            action="/new",
            reason="Tool schema changes are snapshotted at current session start; start a new session before expecting these tools to appear or disappear.",
        )

    if lower.startswith(_GATEWAY_RESTART_PREFIXES):
        return ChangeAdvisory(
            scope=ImpactScope.GATEWAY_RESTART,
            action="hermes gateway restart",
            reason="Gateway runtime settings are loaded by the running gateway process; restart it before expecting this change to affect messaging sessions.",
        )

    if lower in _NEW_SESSION_KEYS or lower.startswith(_NEW_SESSION_PREFIXES):
        return ChangeAdvisory(
            scope=ImpactScope.NEW_SESSION,
            action="/new",
            reason="This setting is resolved when new sessions start; current sessions may keep using the old value.",
        )

    return None


def toolset_change_advisory(platform: str = "cli") -> ChangeAdvisory:
    """Return the advisory for a toolset mutation on ``platform``."""

    return advisory_for_config_key(f"platform_toolsets.{platform}") or ChangeAdvisory(
        scope=ImpactScope.NEW_SESSION,
        action="/new",
        reason="Tool schema changes are snapshotted at current session start; start a new session before expecting these tools to appear or disappear.",
    )


def format_advisory(advisory: ChangeAdvisory) -> str:
    """Format a compact, grep/test-friendly advisory line."""

    return f"{advisory.title}: {advisory.reason} Next action: {advisory.action}."


def print_advisory(advisory: Optional[ChangeAdvisory]) -> None:
    """Print ``advisory`` if present.

    Kept as a tiny wrapper so call sites stay consistent and tests can assert a
    stable prefix.
    """

    if advisory is not None:
        print(format_advisory(advisory))
