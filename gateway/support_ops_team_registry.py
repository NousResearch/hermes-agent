"""SkyVision Support Ops team/channel registry.

This module is a tiny, deterministic source of truth for Discord people and
lanes that Muncho already knows. It does not decide business meaning and does
not route by operational keywords. Hermes/LLM reasoning should choose a
``target_person``; this registry only validates that the chosen person and lane
are consistent.
"""

from __future__ import annotations

import re
import json
import os
import pathlib
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


def _learned_alias_path() -> pathlib.Path:
    try:
        from hermes_constants import get_hermes_home

        root = pathlib.Path(get_hermes_home())
    except Exception:
        root = pathlib.Path.home() / ".hermes"
    return root / "state" / "team-member-aliases.json"


def _load_learned_aliases(path: pathlib.Path | None = None) -> dict[str, str]:
    target = path or _learned_alias_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    aliases = raw.get("aliases") if isinstance(raw, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {
        _normalize(alias): str(member_key).strip()
        for alias, member_key in aliases.items()
        if _normalize(alias) and str(member_key).strip() in TEAM_MEMBERS_BY_KEY
    }


def learn_team_member_alias(
    alias: str,
    member_key: str,
    *,
    path: pathlib.Path | None = None,
) -> dict[str, str]:
    """Persist one exact model-confirmed alias mapping atomically.

    This function never infers a person from text.  The caller supplies the
    alias and canonical member key after Hermes has clarified the ambiguity.
    """
    normalized = _normalize(alias)
    member_key = str(member_key or "").strip()
    if not normalized:
        raise ValueError("alias is required")
    if member_key not in TEAM_MEMBERS_BY_KEY:
        raise ValueError("unknown member_key")

    existing_resolution = resolve_team_member(normalized)
    if (
        existing_resolution.status == "resolved"
        and existing_resolution.member is not None
        and existing_resolution.member.key != member_key
    ):
        raise ValueError("alias already resolves to a different member")

    target = path or _learned_alias_path()
    aliases = _load_learned_aliases(target)
    aliases[normalized] = member_key
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"schema": "hermes.team_member_aliases.v1", "aliases": aliases}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return {"alias": normalized, "member_key": member_key}


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

    learned_member_key = _load_learned_aliases().get(normalized)
    if learned_member_key:
        return TeamMemberResolution(status="resolved", member=TEAM_MEMBERS_BY_KEY[learned_member_key])

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


def format_resolution_candidates(candidates: Iterable[TeamMember]) -> str:
    return ", ".join(f"{member.key}({member.default_channel_name})" for member in candidates)
