"""Support Ops mention lint guards for Discord sends.

This module is intentionally small and deterministic: it does not call Discord,
read secrets, create canonical events, or decide business meaning.  It only
performs pre-send linting on already-authored Support Ops output:

* block unresolved teammate text mentions and placeholders;
* normalize well-known display handles;

It must not infer a route/lane from business words such as PBX/SIP, voucher,
backend, or frontend.  Hermes/LLM reasoning and the explicit channel directory
remain responsible for deciding where a message should go.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from gateway.support_ops_team_registry import (
    SKYVISION_BACKEND_CHANNEL_ID,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    format_resolution_candidates,
    infer_requested_person_phrase,
    infer_salutation_team_member,
    resolve_team_member,
)

UNKNOWN_USER_RE = re.compile(r"(?<![\w-])@unknown-user\b", re.IGNORECASE)
BACKEND_TEXT_MENTION_RE = re.compile(
    r"(?<![\w<@-])@(алекс|ивчо|иво|alex|ivcho|ivo|ivo\s+popov)\b",
    re.IGNORECASE,
)
BACKEND_THREAD_TITLE_RE = re.compile(
    r"(^|[^\w])(?:алекс|алек|ивчо|иво|alex|ivcho|ivo)(?:[^\w]|$)",
    re.IGNORECASE,
)
OWNER_THREAD_TITLE_RE = re.compile(
    r"(^|[^\w])(?:"
    r"емил\s+ломлиев|емо\s+ломлиев|emil\s+lomliev|"
    r"емил|emil|емо(?!\s*(?:к\.?|кожухаров|kozhuharov)\b)"
    r")(?:[^\w]|$)",
    re.IGNORECASE,
)
KOZHUHAROV_TEXT_MENTION_RE = re.compile(
    r"(?<![\w<@-])@(кожухаров|емо\s+к|emo\s+k|kozhuharov)\b",
    re.IGNORECASE,
)
FATIH_TEXT_MENTION_RE = re.compile(r"(?<![\w<@-])@(фатих|fatih)\b", re.IGNORECASE)
RAW_QUOTE_RE = re.compile(r"[\"“”'„‚`].{0,80}(?:пламена|plamena).{0,80}[\"“”'„‚`]", re.IGNORECASE | re.DOTALL)

EMIL_OWNER_MENTION = "<@1279454038731264061>"
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


def _replace_display_handles(text: str) -> tuple[str, Optional[str]]:
    """Normalize canonical display handles only when not an ambiguous raw quote."""
    if RAW_QUOTE_RE.search(text):
        return text, "blocked_plamena_raw_quote_ambiguity"
    text = re.sub(r"\bПламена\b", "Пламенка", text)
    text = re.sub(r"\bPlamena\b", "Пламенка", text, flags=re.IGNORECASE)
    return text, None


def lint_and_resolve_discord_content(content: str) -> MentionLintResult:
    """Fail closed on unresolved teammate mentions without inferring routes."""

    text = str(content or "")
    text, handle_block = _replace_display_handles(text)
    if handle_block:
        return MentionLintResult(ok=False, content=text, blocked_reason=handle_block)

    if UNKNOWN_USER_RE.search(text):
        return MentionLintResult(ok=False, content=text, blocked_reason="blocked_unresolved_unknown_user_placeholder")

    if BACKEND_TEXT_MENTION_RE.search(text) or KOZHUHAROV_TEXT_MENTION_RE.search(text) or FATIH_TEXT_MENTION_RE.search(text):
        return MentionLintResult(ok=False, content=text, blocked_reason="blocked_unresolved_text_teammate_mention")

    return MentionLintResult(ok=True, content=text)


def lint_discord_target_for_content(
    content: str,
    *,
    chat_id: str,
    thread_id: str | None = None,
) -> DiscordTargetLintResult:
    """Validate already-chosen Discord target against explicit teammate mentions.

    This is deliberately not a business classifier. It does not inspect words
    like voucher, backend, PBX, product, reservation, or frontend. It only
    prevents an authored message that already includes exact backend-resolver
    Discord mentions from being delivered to the wrong SkyVision lane.
    """

    text = str(content or "")
    target_chat_id = str(chat_id or "").strip()

    if EMIL_OWNER_MENTION in text and target_chat_id != SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_owner_route_back_mention_wrong_discord_lane",
            expected_channel_id=SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        )

    has_backend_resolver = any(mention in text for mention in BACKEND_RESOLVER_MENTIONS)
    if not has_backend_resolver:
        return DiscordTargetLintResult(ok=True)

    if target_chat_id != SKYVISION_BACKEND_CHANNEL_ID:
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_backend_resolver_mention_wrong_discord_lane",
            expected_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
        )

    has_support_requester = any(mention in text for mention in SUPPORT_REQUESTER_MENTIONS)
    if has_support_requester:
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_mixed_backend_resolver_and_requester_mentions",
            expected_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
        )

    return DiscordTargetLintResult(ok=True)


def lint_discord_thread_create_target(
    name: str,
    *,
    channel_id: str,
    message_id: str | None = None,
    initial_message: str | None = None,
    target_person: str | None = None,
) -> DiscordTargetLintResult:
    """Validate explicit resolver thread titles against the chosen channel.

    This is a safety guard for the Discord thread-creation tool, not a business
    router. It only looks at the already-authored thread title for exact
    teammate labels such as "Алекс" / "Ивчо" or owner labels such as
    "Емо Ломлиев" / "Емил". It does not inspect business
    words like voucher, booking, backend, frontend, PBX, product, or
    reservation, and it does not decide where a case belongs.
    """

    title = str(name or "")
    target_channel_id = str(channel_id or "").strip()
    requested_phrase = infer_requested_person_phrase(f"{name}\n{initial_message or ''}")

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

    salutation_member = infer_salutation_team_member(f"{initial_message or ''}\n{name}")
    if salutation_member is not None and not requested_phrase:
        member = salutation_member
        if target_channel_id != member.default_channel_id:
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_salutation_person_wrong_discord_lane_requires_structured_target_person",
                expected_channel_id=member.default_channel_id,
                guidance=(
                    "Do not create the thread or claim delivery. The starter message appears addressed to "
                    f"{member.display_name}, but no target_person was provided. Retry with "
                    f"target_person='{member.key}' and channel_id={member.default_channel_id} "
                    f"({member.default_channel_name}); otherwise ask the requester to clarify who should receive it."
                ),
            )

    has_owner_title = bool(OWNER_THREAD_TITLE_RE.search(title))
    has_backend_title = bool(BACKEND_THREAD_TITLE_RE.search(title))

    if has_owner_title and target_channel_id != SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_owner_route_back_thread_wrong_discord_lane",
            expected_channel_id=SKYVISION_CONTROL_TOWER_CHANNEL_ID,
            guidance=(
                "Do not create this owner handoff thread in the current channel. "
                "Set target_person='emil_lomliev' and use the control-tower channel."
            ),
        )

    if has_owner_title and not str(message_id or "").strip() and not str(initial_message or "").strip():
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_owner_route_back_thread_missing_initial_message",
            expected_channel_id=SKYVISION_CONTROL_TOWER_CHANNEL_ID,
            guidance="Do not create an empty owner handoff thread; include initial_message or anchor to an existing message.",
        )

    if requested_phrase:
        resolution = resolve_team_member(requested_phrase)
        if resolution.status != "resolved":
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_unresolved_requested_person_requires_clarification",
                guidance=_clarify_target_person_guidance(requested_phrase),
            )
        member = resolution.member
        assert member is not None
        if target_channel_id != member.default_channel_id:
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_requested_person_wrong_discord_lane",
                expected_channel_id=member.default_channel_id,
                guidance=(
                    f"Do not create the thread here. The requested person '{requested_phrase}' resolves to "
                    f"{member.key}; use channel_id={member.default_channel_id} ({member.default_channel_name})."
                ),
            )
        if not str(message_id or "").strip() and not str(initial_message or "").strip():
            return DiscordTargetLintResult(
                ok=False,
                blocked_reason="blocked_requested_person_thread_missing_initial_message",
                expected_channel_id=member.default_channel_id,
                guidance="Do not create an empty handoff thread; include initial_message or anchor to an existing message.",
            )

    if not has_backend_title:
        return DiscordTargetLintResult(ok=True)

    if target_channel_id != SKYVISION_BACKEND_CHANNEL_ID:
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_backend_resolver_thread_wrong_discord_lane",
            expected_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
            guidance=(
                "Do not create this backend handoff thread in the current channel. "
                "Set target_person='alex' or target_person='ivcho' and use the backend channel."
            ),
        )

    if not str(message_id or "").strip() and not str(initial_message or "").strip():
        return DiscordTargetLintResult(
            ok=False,
            blocked_reason="blocked_backend_resolver_thread_missing_initial_message",
            expected_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
            guidance="Do not create an empty backend handoff thread; include initial_message or anchor to an existing message.",
        )

    return DiscordTargetLintResult(ok=True)
