"""Mechanical validation for model-selected Support Ops targets.

Free-form message text, titles, salutations, names, and verbs are deliberately
opaque here.  Hermes/GPT owns their meaning.  This module validates only an
explicit structured ``target_person`` against the exact registry mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gateway.support_ops_team_registry import (
    SKYVISION_BACKEND_CHANNEL_ID,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    format_resolution_candidates,
    resolve_team_member,
)

EMIL_OWNER_MENTION = "<@1279454038731264061>"
EMIL_OWNER_USER_ID = "1279454038731264061"
KOZHUHAROV_MENTION = "<@1282729392883372174>"
ALEX_MENTION = "<@1282940511962791959>"
IVCHO_MENTION = "<@1283039346295050271>"
FATIH_MENTION = "<@779368140512821268>"
PLAMENA_MENTION = "<@1282940574533423125>"
BACKEND_MENTION = f"{ALEX_MENTION} {IVCHO_MENTION}"
BACKEND_RESOLVER_MENTIONS = frozenset({ALEX_MENTION, IVCHO_MENTION})
SUPPORT_REQUESTER_MENTIONS = frozenset({PLAMENA_MENTION})


@dataclass(frozen=True)
class MentionLintResult:
    ok: bool
    content: str
    blocked_reason: Optional[str] = None


@dataclass(frozen=True)
class DiscordTargetLintResult:
    ok: bool
    blocked_reason: Optional[str] = None
    expected_channel_id: Optional[str] = None
    guidance: Optional[str] = None


def _clarify_target_person_guidance(phrase: str | None = None) -> str:
    if phrase:
        return (
            "Do not create the thread or claim delivery. Tell the requester: "
            f"\"Не изпратих съобщението, защото не съм сигурен кой е '{phrase}'. "
            "Моля уточнете човека/канала и ще го изпратя.\" "
            "After the requester clarifies, record the new alias in durable memory/registry flow and retry."
        )
    return (
        "Do not create the thread or claim delivery. Ask the requester to clarify the target person/channel, "
        "then learn the alias and retry."
    )


def lint_and_resolve_discord_content(content: str) -> MentionLintResult:
    """Return authored content byte-for-byte; text is not routing authority."""
    return MentionLintResult(ok=True, content=str(content or ""))


def lint_discord_target_for_content(
    content: str,
    *,
    chat_id: str,
    thread_id: str | None = None,
    parent_chat_id: str | None = None,
    source_user_id: str | None = None,
) -> DiscordTargetLintResult:
    """Compatibility no-op: free-form content never selects or rejects a target."""
    return DiscordTargetLintResult(ok=True)


def lint_discord_thread_create_target(
    name: str,
    *,
    channel_id: str,
    message_id: str | None = None,
    initial_message: str | None = None,
    target_person: str | None = None,
) -> DiscordTargetLintResult:
    """Validate only a model-selected structured person/channel pair."""
    target_channel_id = str(channel_id or "").strip()

    if target_person:
        resolution = resolve_team_member(target_person)
        if resolution.status == "unknown":
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_unknown_target_person_requires_clarification",
                guidance=_clarify_target_person_guidance(target_person),
            )
        if resolution.status == "ambiguous":
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_ambiguous_target_person_requires_clarification",
                guidance=(
                    _clarify_target_person_guidance(target_person)
                    + f" Candidate registry matches: {format_resolution_candidates(resolution.candidates)}."
                ),
            )
        member = resolution.member
        assert member is not None
        if target_channel_id != member.default_channel_id:
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_target_person_wrong_discord_lane",
                expected_channel_id=member.default_channel_id,
                guidance=(
                    f"Do not create the thread here. target_person={member.key} "
                    f"must use channel_id={member.default_channel_id} ({member.default_channel_name})."
                ),
            )
        if not str(message_id or "").strip() and not str(initial_message or "").strip():
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_target_person_thread_missing_initial_message",
                expected_channel_id=member.default_channel_id,
                guidance="Do not create an empty handoff thread; include initial_message or anchor to an existing message.",
            )
        return DiscordTargetLintResult(ok=True)

    return DiscordTargetLintResult(ok=True)
