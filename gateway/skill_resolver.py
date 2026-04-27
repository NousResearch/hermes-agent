"""Adapter-agnostic skill event resolver and skill list snapshotter.

Resolves an inbound event payload (button click, reaction add/remove, mention, etc.)
to the list of skills whose ``triggers:`` frontmatter matches the payload. Also
provides :func:`snapshot_skills` — the canonical lazy walker that adapters wrap
to populate :data:`SkillEntry` lists for the resolver.

This module is intentionally adapter-agnostic — it has no dependency on Discord,
Feishu, Matrix, or any other gateway platform. The resolver receives a list of
``(skill_name, frontmatter, triggers)`` tuples plus a payload dict and returns
matching skill names.

The skill list is supplied via callable injection at the adapter layer (see
``gateway.platforms.discord_interactions.DiscordInteractionsHandler``); this
keeps the resolver testable in isolation and avoids fabricating an
``_available_skills`` attribute on adapters that does not exist today.

Backward compatibility helpers:
    - :func:`has_explicit_triggers` lets adapters detect whether the deployed
      skill corpus uses the new ``triggers:`` schema at all. Adapters can use
      this signal to fall through to existing broadcast routing for skills
      without explicit triggers (Feishu BC fork, plan §4 step 1).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Public type alias ─────────────────────────────────────────────────────

# A skill entry is the canonical tuple passed from adapter to resolver:
# (skill_name, frontmatter_dict, triggers_list)
SkillEntry = Tuple[str, Dict[str, Any], List[Dict[str, Any]]]


_KNOWN_EVENT_TYPES = ("mention", "slash", "button", "reaction", "cron")


# ── Public API ────────────────────────────────────────────────────────────


def resolve_event_skills(
    event_type: str,
    payload: Dict[str, Any],
    skills: List[SkillEntry],
) -> List[str]:
    """Return the list of skill names whose triggers match the event payload.

    Args:
        event_type: One of ``mention``, ``slash``, ``button``, ``reaction``, ``cron``.
        payload: Event-specific payload dict. Expected keys per type:
            - mention: ``text`` (str), optional ``channel`` (str)
            - slash: ``name`` (str)
            - button: ``custom_id`` (str), optional ``channel`` (str)
            - reaction: ``emoji`` (str), optional ``channel`` (str), optional
              ``message_age_seconds`` (int)
            - cron: (resolver returns [] — cron is handled by the cron registrar)
        skills: List of ``(skill_name, frontmatter, triggers)`` tuples. Each
            triggers list contains type-keyed dicts (Schema α — see
            ``agent.skill_utils.extract_skill_triggers``).

    Returns:
        Skill names whose triggers match the payload. May be empty.
    """
    if event_type not in _KNOWN_EVENT_TYPES:
        return []

    if event_type == "cron":
        # Cron events are dispatched by the cron registrar with the skill
        # already resolved — the resolver is not called for cron firings.
        return []

    matched: List[str] = []
    for entry in skills:
        if not _is_valid_skill_entry(entry):
            continue
        name, _frontmatter, triggers = entry
        for trigger in triggers:
            if not isinstance(trigger, dict):
                continue
            if trigger.get("type") != event_type:
                continue
            if _trigger_matches(event_type, trigger, payload):
                matched.append(name)
                break  # avoid duplicate match for same skill
    return matched


def has_explicit_triggers(skills: List[SkillEntry]) -> bool:
    """Return True if any skill in the list has at least one explicit trigger.

    Adapters use this to decide whether the deployed corpus uses the new
    schema. If False, BC fallbacks (e.g., Feishu broadcast) should fire when
    no resolver match is found.
    """
    for entry in skills:
        if not _is_valid_skill_entry(entry):
            continue
        _name, _frontmatter, triggers = entry
        if triggers:
            return True
    return False


# ── Internal matchers ─────────────────────────────────────────────────────


def _is_valid_skill_entry(entry: Any) -> bool:
    """Defensive check — adapters may pass malformed lists during reload."""
    return (
        isinstance(entry, tuple)
        and len(entry) == 3
        and isinstance(entry[0], str)
        and isinstance(entry[1], dict)
        and isinstance(entry[2], list)
    )


def _trigger_matches(event_type: str, trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    if event_type == "button":
        return _match_button(trigger, payload)
    if event_type == "reaction":
        return _match_reaction(trigger, payload)
    if event_type == "mention":
        return _match_mention(trigger, payload)
    if event_type == "slash":
        return _match_slash(trigger, payload)
    return False


def _match_button(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    pattern = trigger.get("custom_id_pattern")
    if not isinstance(pattern, str) or not pattern:
        return False
    custom_id = payload.get("custom_id")
    if not isinstance(custom_id, str):
        return False
    if not fnmatch.fnmatchcase(custom_id, pattern):
        return False
    return _channel_filter_passes(trigger, payload)


def _match_reaction(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    expected_emoji = trigger.get("emoji")
    if not isinstance(expected_emoji, str) or not expected_emoji:
        return False
    actual_emoji = payload.get("emoji")
    if actual_emoji != expected_emoji:
        return False
    if not _channel_filter_passes(trigger, payload):
        return False
    return _age_limit_passes(trigger, payload)


def _match_mention(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    pattern = trigger.get("regex")
    if not isinstance(pattern, str) or not pattern:
        return False
    text = payload.get("text")
    if not isinstance(text, str):
        return False
    try:
        if not re.search(pattern, text):
            return False
    except re.error:
        logger.warning("skill_resolver: invalid mention regex %r — skipping", pattern)
        return False
    return _channel_filter_passes(trigger, payload)


def _match_slash(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    expected_name = trigger.get("name")
    if not isinstance(expected_name, str) or not expected_name:
        return False
    actual_name = payload.get("name")
    return actual_name == expected_name


def _channel_filter_passes(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Apply optional channel_filter — accept all if filter absent."""
    channel_filter = trigger.get("channel_filter")
    if channel_filter is None:
        return True
    if not isinstance(channel_filter, list) or not channel_filter:
        return True
    channel = payload.get("channel")
    if channel is None:
        # No channel info on payload → cannot filter; conservative pass
        return True
    return channel in channel_filter


def _age_limit_passes(trigger: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Apply optional age_limit (string like '30d', '14d', '6h') against
    payload['message_age_seconds']. Accept all if either field is missing."""
    raw_limit = trigger.get("age_limit")
    if not raw_limit:
        return True
    age_seconds = payload.get("message_age_seconds")
    if not isinstance(age_seconds, (int, float)):
        return True

    limit_seconds = _parse_age_to_seconds(str(raw_limit))
    if limit_seconds is None:
        return True
    return age_seconds <= limit_seconds


_AGE_UNITS = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 60 * 60 * 24,
    "w": 60 * 60 * 24 * 7,
}


def _parse_age_to_seconds(raw: str) -> int | None:
    """Parse '30d', '14d', '6h', '90m', '120s' into seconds. Return None if invalid."""
    raw = raw.strip().lower()
    if not raw:
        return None
    match = re.fullmatch(r"(\d+)\s*([smhdw])", raw)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    return value * _AGE_UNITS[unit]


# ── Skill snapshotter (lazy walker shared across adapters) ───────────────


def snapshot_skills() -> List[SkillEntry]:
    """Walk all skills directories and return resolver-ready SkillEntry tuples.

    Lazy — called per event by adapter wrappers (typically wrapped in a
    closure passed to :class:`gateway.platforms.discord_interactions.DiscordInteractionsHandler`).
    Returns ``[]`` if skill_utils cannot be imported (e.g., during very early
    bootstrap), so adapters can degrade gracefully.

    Each entry is ``(skill_name, frontmatter_dict, triggers_list)``. The
    triggers list contains both explicit triggers (from
    ``metadata.hermes.triggers``) and implicit triggers (derived from
    ``metadata.hermes.slash_command``) — see
    :func:`agent.skill_utils.get_skill_triggers`.
    """
    try:
        from agent.skill_utils import (
            get_all_skills_dirs,
            get_skill_triggers,
            iter_skill_index_files,
            parse_frontmatter,
        )
    except Exception:
        logger.debug("snapshot_skills: skill_utils import failed", exc_info=True)
        return []

    result: List[SkillEntry] = []
    for skills_dir in get_all_skills_dirs():
        for path in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, _body = parse_frontmatter(content)
            if not isinstance(fm, dict):
                continue
            name = fm.get("name") or path.parent.name
            if not isinstance(name, str):
                continue
            triggers, _is_explicit = get_skill_triggers(fm)
            result.append((name, fm, triggers))
    return result
