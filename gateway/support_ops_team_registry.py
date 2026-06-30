"""SkyVision Support Ops team/channel registry.

This module is a tiny, deterministic source of truth for Discord people and
lanes that Muncho already knows. It does not decide business meaning and does
not route by operational keywords. Hermes/LLM reasoning should choose a
``target_person``; this registry only validates that the chosen person and lane
are consistent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal


SKYVISION_CONTROL_TOWER_CHANNEL_ID = "1504852355588423801"
SKYVISION_BACKEND_CHANNEL_ID = "1504852408227069993"
SKYVISION_FRONTEND_CHANNEL_ID = "1504852444407140402"
SKYVISION_DEVOPS_CHANNEL_ID = "1504852485083496561"
SKYVISION_BOOKING_OPS_CHANNEL_ID = "1504852553031221391"


@dataclass(frozen=True)
class TeamMember:
    key: str
    display_name: str
    discord_user_id: str
    default_channel_id: str
    default_channel_name: str
    aliases: tuple[str, ...]

    @property
    def mention(self) -> str:
        return f"<@{self.discord_user_id}>"


@dataclass(frozen=True)
class TeamMemberResolution:
    status: Literal["resolved", "unknown", "ambiguous"]
    member: TeamMember | None = None
    candidates: tuple[TeamMember, ...] = ()


TEAM_MEMBERS: tuple[TeamMember, ...] = (
    TeamMember(
        key="emil_lomliev",
        display_name="Emil Lomliev",
        discord_user_id="1279454038731264061",
        default_channel_id=SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        default_channel_name="control-tower",
        aliases=(
            "emil_lomliev",
            "emil lomliev",
            "emil",
            "emo l",
            "емил ломлиев",
            "емо ломлиев",
            "емо л",
            "емо",
            "емил",
        ),
    ),
    TeamMember(
        key="alex",
        display_name="Alex",
        discord_user_id="1282940511962791959",
        default_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
        default_channel_name="backend",
        aliases=("alex", "алекс", "алек"),
    ),
    TeamMember(
        key="ivcho",
        display_name="Ivcho",
        discord_user_id="1283039346295050271",
        default_channel_id=SKYVISION_BACKEND_CHANNEL_ID,
        default_channel_name="backend",
        aliases=("ivcho", "ivo", "ivo popov", "ивчо", "иво", "иво попов"),
    ),
    TeamMember(
        key="fatih",
        display_name="Fatih",
        discord_user_id="779368140512821268",
        default_channel_id=SKYVISION_FRONTEND_CHANNEL_ID,
        default_channel_name="frontend",
        aliases=("fatih", "фатих"),
    ),
    TeamMember(
        key="emil_kozhuharov",
        display_name="Emil Kozhuharov",
        discord_user_id="1282729392883372174",
        default_channel_id=SKYVISION_DEVOPS_CHANNEL_ID,
        default_channel_name="devops",
        aliases=(
            "emil kozhuharov",
            "emo k",
            "kozhuharov",
            "емил кожухаров",
            "емо кожухаров",
            "емо к",
            "емо к.",
            "кожухаров",
        ),
    ),
    TeamMember(
        key="plamenka",
        display_name="Plamenka",
        discord_user_id="1282940574533423125",
        default_channel_id=SKYVISION_BOOKING_OPS_CHANNEL_ID,
        default_channel_name="booking-ops",
        aliases=("plamenka", "plamena", "пламенка", "пламена"),
    ),
)

TEAM_MEMBERS_BY_KEY = {member.key: member for member in TEAM_MEMBERS}
TEAM_MEMBERS_BY_MENTION = {member.mention: member for member in TEAM_MEMBERS}
TEAM_MEMBERS_BY_ID = {member.discord_user_id: member for member in TEAM_MEMBERS}


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"<@!?(\d+)>", r"\1", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w.\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _alias_entries() -> list[tuple[str, TeamMember]]:
    entries: list[tuple[str, TeamMember]] = []
    for member in TEAM_MEMBERS:
        entries.append((_normalize(member.key), member))
        entries.append((_normalize(member.discord_user_id), member))
        entries.append((_normalize(member.mention), member))
        for alias in member.aliases:
            entries.append((_normalize(alias), member))
    return sorted(set(entries), key=lambda item: len(item[0]), reverse=True)


ALIAS_ENTRIES = _alias_entries()


def get_team_member(key: str) -> TeamMember | None:
    return TEAM_MEMBERS_BY_KEY.get(str(key or "").strip())


def _titleish(value: str) -> str:
    return _normalize(value).replace(".", "")


def resolve_team_member(value: str) -> TeamMemberResolution:
    normalized = _normalize(value)
    if not normalized:
        return TeamMemberResolution(status="unknown")

    if normalized in TEAM_MEMBERS_BY_KEY:
        return TeamMemberResolution(status="resolved", member=TEAM_MEMBERS_BY_KEY[normalized])
    if normalized in TEAM_MEMBERS_BY_ID:
        return TeamMemberResolution(status="resolved", member=TEAM_MEMBERS_BY_ID[normalized])
    if f"<@{normalized}>" in TEAM_MEMBERS_BY_MENTION:
        return TeamMemberResolution(status="resolved", member=TEAM_MEMBERS_BY_MENTION[f"<@{normalized}>"])

    matches = {
        member
        for alias, member in ALIAS_ENTRIES
        if normalized == alias
    }
    if len(matches) == 1:
        return TeamMemberResolution(status="resolved", member=next(iter(matches)))
    if len(matches) > 1:
        return TeamMemberResolution(status="ambiguous", candidates=tuple(sorted(matches, key=lambda m: m.key)))
    return TeamMemberResolution(status="unknown")


def infer_requested_person_phrase(text: str) -> str | None:
    """Extract a conversational "write/tell/send to X" phrase if present.

    This is only used to produce a helpful fail-closed clarification. It is not
    a route authority and never creates a target by itself.
    """

    normalized = _titleish(text)
    patterns = (
        r"(?:пиши|прати|изпрати|кажи|докладвай|върни|насочи)"
        r"(?:\s+(?:директно|нов|нова|ново|тред|thread|съобщение))*"
        r"\s+(?:на|до|към)\s+(.+?)(?:$|\s+(?:че|за|относно|в|:))",
        r"(?:write|send|tell|route|report)"
        r"(?:\s+(?:directly|new|thread|message))*"
        r"\s+(?:to\s+)?(.+?)(?:$|\s+(?:that|about|in|:))",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            phrase = match.group(1).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            return phrase or None
    return None


def infer_salutation_team_member(text: str) -> TeamMember | None:
    """Resolve a leading "Name, ..." / "Name: ..." handoff salutation."""

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^,:—–-]{2,40})\s*[,：:—–-]", line, flags=re.UNICODE)
        if not match:
            continue
        resolution = resolve_team_member(match.group(1))
        if resolution.status == "resolved":
            return resolution.member
        return None
    return None


def infer_team_members_from_text(text: str) -> tuple[TeamMember, ...]:
    """Return non-overlapping registry alias matches in already-authored text.

    This is not a business classifier. It only detects known people aliases so a
    caller can fail closed when ``target_person`` is missing or ambiguous.
    """

    normalized = f" {_normalize(text)} "
    occupied: list[range] = []
    matches: list[TeamMember] = []

    for alias, member in ALIAS_ENTRIES:
        if not alias or len(alias) < 3:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.UNICODE)
        for match in pattern.finditer(normalized):
            span = range(match.start(), match.end())
            if any(set(span) & set(existing) for existing in occupied):
                continue
            occupied.append(span)
            if member not in matches:
                matches.append(member)
            break

    return tuple(matches)


def format_resolution_candidates(candidates: Iterable[TeamMember]) -> str:
    return ", ".join(f"{member.key}({member.default_channel_name})" for member in candidates)
