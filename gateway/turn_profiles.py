"""Hermes gateway turn profile selection.

Runtime-owned profile selection. Brainstack remains evidence input only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Mapping, Sequence


SCHEMA_VERSION = "hermes.turn_profiles.v1"

CONVERSATION_TOOLSETS = ("conversation_tools",)
CONVERSATION_DIRECT_TOOLSETS: tuple[str, ...] = ()
HEAVY_WEB_TOOLSETS = ("heavy_web",)
HEAVY_FILE_TOOLSETS = ("heavy_file",)
HEAVY_CODE_TOOLSETS = ("heavy_code",)
HEAVY_FULL_DEBUG_TOOLSETS = ("heavy_full_debug",)

HEAVY_COMMAND_RE = re.compile(r"^\s*/heavy(?:\s+(?P<bundle>web|file|code|full-debug))?\b", re.I)


@dataclass(frozen=True)
class ResolvedTurnProfile:
    schema: str
    platform: str
    turn_profile: str
    tool_profile: str
    enabled_toolsets: tuple[str, ...]
    reason_code: str
    explicit_heavy: bool
    heavy_bundle: str | None
    url_attachment_candidate_only: bool
    rollback_override_active: bool
    cli_local_unchanged: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["enabled_toolsets"] = list(self.enabled_toolsets)
        return data


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _url_count(prompt: str) -> int:
    return len(re.findall(r"https?://\S+", prompt or ""))


def resolve_turn_profile(
    *,
    platform: str,
    prompt: str,
    current_enabled_toolsets: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> ResolvedTurnProfile:
    """Resolve tool profile from structural signals only.

    No natural-language intent routing. URL presence alone is not fetch.
    """

    env = env or os.environ
    current = tuple(sorted(str(name) for name in current_enabled_toolsets))
    if platform != "discord":
        return ResolvedTurnProfile(
            schema=SCHEMA_VERSION,
            platform=platform,
            turn_profile="heavy_work",
            tool_profile="existing_platform_default",
            enabled_toolsets=current,
            reason_code="NON_DISCORD_UNCHANGED",
            explicit_heavy=False,
            heavy_bundle=None,
            url_attachment_candidate_only=False,
            rollback_override_active=False,
            cli_local_unchanged=platform == "cli",
        )

    rollback = (
        env.get("HERMES_DISCORD_TURN_PROFILE", "").lower() in {"heavy", "heavy_work", "full"}
        or env.get("HERMES_DISCORD_TOOL_PROFILE", "").lower() in {"heavy", "heavy_work", "full"}
    )
    if rollback:
        return ResolvedTurnProfile(
            schema=SCHEMA_VERSION,
            platform=platform,
            turn_profile="heavy_work",
            tool_profile="heavy_work",
            enabled_toolsets=current or HEAVY_FULL_DEBUG_TOOLSETS,
            reason_code="ROLLBACK_OVERRIDE_FORCE_HEAVY",
            explicit_heavy=True,
            heavy_bundle="heavy_work",
            url_attachment_candidate_only=False,
            rollback_override_active=True,
            cli_local_unchanged=False,
        )

    match = HEAVY_COMMAND_RE.match(prompt or "")
    if match:
        bundle = (match.group("bundle") or "work").lower()
        if bundle == "web":
            toolsets = HEAVY_WEB_TOOLSETS
            profile = "heavy_web"
        elif bundle == "file":
            toolsets = HEAVY_FILE_TOOLSETS
            profile = "heavy_file"
        elif bundle == "code":
            toolsets = HEAVY_CODE_TOOLSETS
            profile = "heavy_code"
        elif bundle == "full-debug":
            toolsets = HEAVY_FULL_DEBUG_TOOLSETS
            profile = "heavy_full_debug"
        else:
            toolsets = current or HEAVY_FULL_DEBUG_TOOLSETS
            profile = "heavy_work"
        return ResolvedTurnProfile(
            schema=SCHEMA_VERSION,
            platform=platform,
            turn_profile=profile,
            tool_profile=profile,
            enabled_toolsets=tuple(sorted(toolsets)),
            reason_code="EXPLICIT_HEAVY_COMMAND",
            explicit_heavy=True,
            heavy_bundle=profile,
            url_attachment_candidate_only=False,
            rollback_override_active=False,
            cli_local_unchanged=False,
        )

    if _truthy(env.get("HERMES_DISCORD_CONVERSATION_DIRECT")):
        return ResolvedTurnProfile(
            schema=SCHEMA_VERSION,
            platform=platform,
            turn_profile="conversation_direct",
            tool_profile="conversation_direct",
            enabled_toolsets=CONVERSATION_DIRECT_TOOLSETS,
            reason_code="DISCORD_CONVERSATION_DIRECT_OVERRIDE",
            explicit_heavy=False,
            heavy_bundle=None,
            url_attachment_candidate_only=_url_count(prompt) > 0,
            rollback_override_active=False,
            cli_local_unchanged=False,
        )

    return ResolvedTurnProfile(
        schema=SCHEMA_VERSION,
        platform=platform,
        turn_profile="conversation_tools",
        tool_profile="conversation_tools",
        enabled_toolsets=CONVERSATION_TOOLSETS,
        reason_code="DISCORD_DEFAULT_CONVERSATION",
        explicit_heavy=False,
        heavy_bundle=None,
        url_attachment_candidate_only=_url_count(prompt) > 0,
        rollback_override_active=False,
        cli_local_unchanged=False,
    )
