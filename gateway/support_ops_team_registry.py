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


SKYVISION_GUILD_ID = "1282725267068157972"
SKYVISION_CONTROL_TOWER_CHANNEL_ID = "1504852355588423801"
SKYVISION_CHATBOT_WEB_MONITORING_CHANNEL_ID = "1510888721614901358"
SKYVISION_BACKEND_CHANNEL_ID = "1504852408227069993"
SKYVISION_FRONTEND_CHANNEL_ID = "1504852444407140402"
SKYVISION_DEVOPS_CHANNEL_ID = "1504852485083496561"
SKYVISION_BOOKING_OPS_CHANNEL_ID = "1504852553031221391"
SKYVISION_BUSINESS_ACCOUNTING_LEGAL_CHANNEL_ID = "1504852628373373028"
SKYVISION_NASI_AI_OPS_CHANNEL_ID = "1505499746939174993"
SKYVISION_MARKETING_GROWTH_CHANNEL_ID = "1507239177350283274"
SKYVISION_SUPPLIERS_CHANNEL_ID = "1507239385010016308"

# Dedicated public target for the isolated synthetic canary only.  It is not a
# production operational lane and must never be added to the team routing
# registry.  The canary edge independently proves @everyone visibility.
SKYVISION_SYNTHETIC_CANARY_PUBLIC_CHANNEL_ID = "1526858760100909066"

# This lane was created during an abandoned public-only cutover design.  The
# owner has explicitly marked it unused/non-production.  Keep the identifier
# only so audits can prove that it is not accidentally admitted by the
# production registry; do not delete the Discord channel automatically.
SKYVISION_UNUSED_MUNCHO_OPS_PUBLIC_CHANNEL_ID = "1526870121677848636"


@dataclass(frozen=True)
class ApprovedGuildLane:
    """One owner-approved Discord guild lane.

    Visibility is intentionally absent from this registry.  Existing Discord
    channel/role ACLs remain authoritative.  The privileged edge proves the
    current guild, channel type and bot permissions before every operation.
    """

    key: str
    channel_id: str
    channel_name: str
    aliases: tuple[str, ...] = ()
    target_type: Literal["guild_channel", "guild_thread"] = "guild_channel"
    parent_channel_id: str | None = None


@dataclass(frozen=True)
class ApprovedGuildLaneResolution:
    status: Literal["resolved", "unknown", "ambiguous"]
    lane: ApprovedGuildLane | None = None
    candidates: tuple[ApprovedGuildLane, ...] = ()


APPROVED_OPERATIONAL_GUILD_LANES: tuple[ApprovedGuildLane, ...] = (
    ApprovedGuildLane(
        "control_tower",
        SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "control-tower",
        ("control tower", "контролна кула"),
    ),
    ApprovedGuildLane(
        "chatbot_web_monitoring",
        SKYVISION_CHATBOT_WEB_MONITORING_CHANNEL_ID,
        "chatbot-web-monitoring",
        ("chatbot web monitoring",),
    ),
    ApprovedGuildLane(
        "backend",
        SKYVISION_BACKEND_CHANNEL_ID,
        "backend",
    ),
    ApprovedGuildLane(
        "frontend",
        SKYVISION_FRONTEND_CHANNEL_ID,
        "frontend",
    ),
    ApprovedGuildLane(
        "devops",
        SKYVISION_DEVOPS_CHANNEL_ID,
        "devops",
        ("dev ops",),
    ),
    ApprovedGuildLane(
        "booking_ops",
        SKYVISION_BOOKING_OPS_CHANNEL_ID,
        "booking-ops",
        (
            "booking ops",
            "call center",
            "call-center",
            "колцентър",
        ),
    ),
    ApprovedGuildLane(
        "business_accounting_legal",
        SKYVISION_BUSINESS_ACCOUNTING_LEGAL_CHANNEL_ID,
        "business-accounting-legal",
        ("business accounting legal",),
    ),
    ApprovedGuildLane(
        "nasi_ai_ops",
        SKYVISION_NASI_AI_OPS_CHANNEL_ID,
        "nasi-ai-ops",
        ("nasi ai ops",),
    ),
    ApprovedGuildLane(
        "marketing_growth",
        SKYVISION_MARKETING_GROWTH_CHANNEL_ID,
        "marketing-growth",
        ("marketing growth",),
    ),
    ApprovedGuildLane(
        "suppliers",
        SKYVISION_SUPPLIERS_CHANNEL_ID,
        "suppliers",
    ),
)
APPROVED_OPERATIONAL_GUILD_LANES_BY_KEY = {
    lane.key: lane for lane in APPROVED_OPERATIONAL_GUILD_LANES
}
APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID = {
    lane.channel_id: lane for lane in APPROVED_OPERATIONAL_GUILD_LANES
}
SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS = frozenset(
    APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID
)
# Compatibility name used only by the isolated synthetic canary to prove that
# production operational lanes cannot be substituted for its dedicated public
# fixture channel.  These IDs are approved in production; they are not
# globally "locked" or forbidden.
SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS = SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS


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


def normalize_team_member_alias(value: str) -> str:
    """Return the mechanical exact-match representation used by projections."""

    return _normalize(value)


def _approved_lane_alias_entries() -> tuple[tuple[str, ApprovedGuildLane], ...]:
    entries: list[tuple[str, ApprovedGuildLane]] = []
    for lane in APPROVED_OPERATIONAL_GUILD_LANES:
        for value in (
            lane.key,
            lane.channel_id,
            lane.channel_name,
            f"#{lane.channel_name}",
            *lane.aliases,
        ):
            normalized = _normalize(value)
            if normalized:
                entries.append((normalized, lane))
    return tuple(entries)


APPROVED_LANE_ALIAS_ENTRIES = _approved_lane_alias_entries()
STATIC_ALIAS_CHANNEL_IDS = {
    alias: lane.channel_id for alias, lane in APPROVED_LANE_ALIAS_ENTRIES
}


def _canonical_channel_aliases() -> dict[str, dict[str, str]]:
    """Load only the privileged Canonical channel-alias projection."""

    try:
        from gateway.support_ops_alias_projection import (
            AliasProjectionError,
            load_channel_alias_projection,
        )

        return load_channel_alias_projection(
            _canonical_alias_projection_path(),
            normalize_alias=normalize_team_member_alias,
            valid_member_keys=TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=SKYVISION_GUILD_ID,
            static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
        )
    except FileNotFoundError:
        return {}
    except (AliasProjectionError, OSError):
        return {}


def resolve_approved_guild_lane(value: str) -> ApprovedGuildLaneResolution:
    """Resolve one exact model-authored lane key/alias/ID.

    There is deliberately no substring, token, keyword or prose matching.  A
    model-authored structured lane must normalize to an entire registered
    identity; otherwise Hermes asks the requester to clarify.
    """

    normalized = _normalize(value)
    if not normalized:
        return ApprovedGuildLaneResolution("unknown")
    matches = {
        lane.channel_id: lane
        for alias, lane in APPROVED_LANE_ALIAS_ENTRIES
        if alias == normalized
    }
    candidates = tuple(
        sorted(matches.values(), key=lambda lane: (lane.channel_name, lane.channel_id))
    )
    if len(candidates) == 1:
        return ApprovedGuildLaneResolution("resolved", lane=candidates[0])
    if candidates:
        return ApprovedGuildLaneResolution("ambiguous", candidates=candidates)
    learned = _canonical_channel_aliases().get(normalized)
    if learned is not None:
        return ApprovedGuildLaneResolution(
            "resolved",
            lane=ApprovedGuildLane(
                key=f"canonical:{learned['channel_id']}",
                channel_id=learned["channel_id"],
                channel_name=normalized,
                aliases=(normalized,),
                target_type=learned["target_type"],
                parent_channel_id=learned.get("parent_channel_id"),
            ),
        )
    return ApprovedGuildLaneResolution("unknown")


def approved_guild_lane_for_channel_id(
    channel_id: str,
) -> ApprovedGuildLane | None:
    """Return the exact approved root lane for a numeric channel ID."""

    return APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID.get(
        str(channel_id or "").strip()
    )


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
STATIC_ALIAS_MEMBER_KEYS = {
    alias: member.key for alias, member in ALIAS_ENTRIES
}


def _learned_alias_path() -> pathlib.Path:
    try:
        from hermes_constants import get_hermes_home

        root = pathlib.Path(get_hermes_home())
    except Exception:
        root = pathlib.Path.home() / ".hermes"
    return root / "state" / "team-member-aliases.json"


def _canonical_alias_projection_path() -> pathlib.Path:
    from gateway.support_ops_alias_projection import (
        DEFAULT_PUBLIC_ALIAS_PROJECTION_PATH,
    )

    return DEFAULT_PUBLIC_ALIAS_PROJECTION_PATH


def _load_legacy_learned_aliases(path: pathlib.Path) -> dict[str, str]:
    target = pathlib.Path(path)
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


def _load_learned_aliases(path: pathlib.Path | None = None) -> dict[str, str]:
    """Load the Canonical person projection, with a development-only fallback.

    An explicitly supplied path is always the legacy derived cache used by the
    verified writer append path and tests.  Normal resolution first consumes
    the isolated projector's minimal public document.  A present but invalid
    Canonical projection fails closed instead of falling back to stale local
    state.
    """

    if path is not None:
        return _load_legacy_learned_aliases(pathlib.Path(path))
    projection_path = _canonical_alias_projection_path()
    try:
        from gateway.support_ops_alias_projection import (
            AliasProjectionError,
            load_alias_projection,
        )

        return load_alias_projection(
            projection_path,
            normalize_alias=normalize_team_member_alias,
            valid_member_keys=TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=SKYVISION_GUILD_ID,
            static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
        )
    except FileNotFoundError:
        # A production writer boundary makes the Canonical public projection
        # the only normal authority. The mutable legacy file remains available
        # only through an explicitly supplied path for migration/import/tests.
        try:
            from gateway.canonical_writer_boundary import (
                writer_boundary_policy_required,
            )

            if writer_boundary_policy_required():
                return {}
        except Exception:
            return {}
        return _load_legacy_learned_aliases(_learned_alias_path())
    except (AliasProjectionError, OSError):
        return {}


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
