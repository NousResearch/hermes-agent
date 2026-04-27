"""Tests for the adapter-agnostic skill event resolver."""

from __future__ import annotations

from gateway.skill_resolver import (
    has_explicit_triggers,
    resolve_event_skills,
)


# Helper to build a SkillEntry tuple.
def _skill(name: str, *triggers: dict) -> tuple:
    return (name, {}, list(triggers))


# ── resolve_event_skills (button) ────────────────────────────────────────


def test_button_match_by_pattern() -> None:
    skills = [
        _skill("approver", {"type": "button", "custom_id_pattern": "approve_*"}),
        _skill("rejecter", {"type": "button", "custom_id_pattern": "reject_*"}),
    ]
    payload = {"custom_id": "approve_42"}
    result = resolve_event_skills("button", payload, skills)
    assert result == ["approver"]


def test_button_no_match_returns_empty() -> None:
    skills = [_skill("approver", {"type": "button", "custom_id_pattern": "approve_*"})]
    payload = {"custom_id": "deploy_5"}
    assert resolve_event_skills("button", payload, skills) == []


def test_button_with_channel_filter_passes() -> None:
    skills = [
        _skill(
            "approver",
            {
                "type": "button",
                "custom_id_pattern": "approve_*",
                "channel_filter": ["회장단-test", "회장단-회의"],
            },
        )
    ]
    payload = {"custom_id": "approve_1", "channel": "회장단-test"}
    assert resolve_event_skills("button", payload, skills) == ["approver"]


def test_button_with_channel_filter_blocks() -> None:
    skills = [
        _skill(
            "approver",
            {
                "type": "button",
                "custom_id_pattern": "approve_*",
                "channel_filter": ["회장단-test"],
            },
        )
    ]
    payload = {"custom_id": "approve_1", "channel": "general"}
    assert resolve_event_skills("button", payload, skills) == []


# ── resolve_event_skills (reaction) ──────────────────────────────────────


def test_reaction_match_exact_emoji() -> None:
    skills = [_skill("completer", {"type": "reaction", "emoji": "✅"})]
    payload = {"emoji": "✅"}
    assert resolve_event_skills("reaction", payload, skills) == ["completer"]


def test_reaction_wrong_emoji_no_match() -> None:
    skills = [_skill("completer", {"type": "reaction", "emoji": "✅"})]
    payload = {"emoji": "👍"}
    assert resolve_event_skills("reaction", payload, skills) == []


def test_reaction_age_limit_within_window() -> None:
    skills = [_skill("completer", {"type": "reaction", "emoji": "✅", "age_limit": "30d"})]
    payload = {"emoji": "✅", "message_age_seconds": 60 * 60 * 24 * 5}  # 5 days
    assert resolve_event_skills("reaction", payload, skills) == ["completer"]


def test_reaction_age_limit_exceeded() -> None:
    skills = [_skill("completer", {"type": "reaction", "emoji": "✅", "age_limit": "14d"})]
    payload = {"emoji": "✅", "message_age_seconds": 60 * 60 * 24 * 30}  # 30 days
    assert resolve_event_skills("reaction", payload, skills) == []


# ── resolve_event_skills (mention) ───────────────────────────────────────


def test_mention_match_by_regex() -> None:
    skills = [_skill("approver", {"type": "mention", "regex": r"approve\s+\d+"})]
    payload = {"text": "@bot approve 42"}
    assert resolve_event_skills("mention", payload, skills) == ["approver"]


def test_mention_invalid_regex_no_crash() -> None:
    """Invalid regex pattern logs warning but does not crash resolver."""
    skills = [_skill("broken", {"type": "mention", "regex": "[unclosed"})]
    payload = {"text": "anything"}
    assert resolve_event_skills("mention", payload, skills) == []


# ── resolve_event_skills (slash) ─────────────────────────────────────────


def test_slash_match_exact_name() -> None:
    skills = [_skill("approver", {"type": "slash", "name": "approve"})]
    payload = {"name": "approve"}
    assert resolve_event_skills("slash", payload, skills) == ["approver"]


def test_slash_wrong_name() -> None:
    skills = [_skill("approver", {"type": "slash", "name": "approve"})]
    payload = {"name": "deploy"}
    assert resolve_event_skills("slash", payload, skills) == []


# ── resolve_event_skills (cron — handled separately by registrar) ────────


def test_cron_returns_empty() -> None:
    """Cron events are handled by the cron registrar, not the resolver."""
    skills = [_skill("daily", {"type": "cron", "schedule": "0 9 * * *"})]
    payload = {"job_id": "daily"}
    assert resolve_event_skills("cron", payload, skills) == []


# ── resolve_event_skills (multiple skills + same trigger type) ───────────


def test_multiple_skills_match_same_button_pattern() -> None:
    skills = [
        _skill("a", {"type": "button", "custom_id_pattern": "approve_*"}),
        _skill("b", {"type": "button", "custom_id_pattern": "approve_*"}),
        _skill("c", {"type": "button", "custom_id_pattern": "reject_*"}),
    ]
    payload = {"custom_id": "approve_1"}
    result = resolve_event_skills("button", payload, skills)
    assert sorted(result) == ["a", "b"]


def test_skill_with_multiple_triggers_only_matched_once() -> None:
    """A skill with two triggers of the same type that both match should
    appear in the result only once (no duplicate names)."""
    skill = (
        "multi",
        {},
        [
            {"type": "button", "custom_id_pattern": "approve_*"},
            {"type": "button", "custom_id_pattern": "approv_*"},
        ],
    )
    payload = {"custom_id": "approve_1"}
    assert resolve_event_skills("button", payload, [skill]) == ["multi"]


# ── unknown event type ──────────────────────────────────────────────────


def test_unknown_event_type_returns_empty() -> None:
    skills = [_skill("foo", {"type": "mention", "regex": "."})]
    payload = {"text": "x"}
    assert resolve_event_skills("websocket", payload, skills) == []


# ── malformed entries ───────────────────────────────────────────────────


def test_malformed_skill_entry_skipped() -> None:
    """Defensive: list with a malformed entry should not crash."""
    skills = [
        ("name_without_other_fields",),  # malformed (1-tuple)
        _skill("good", {"type": "button", "custom_id_pattern": "approve_*"}),
    ]
    payload = {"custom_id": "approve_1"}
    assert resolve_event_skills("button", payload, skills) == ["good"]


# ── has_explicit_triggers ────────────────────────────────────────────────


def test_has_explicit_triggers_true() -> None:
    skills = [
        _skill("a"),  # empty triggers
        _skill("b", {"type": "reaction", "emoji": "✅"}),
    ]
    assert has_explicit_triggers(skills) is True


def test_has_explicit_triggers_false_when_all_empty() -> None:
    skills = [_skill("a"), _skill("b")]
    assert has_explicit_triggers(skills) is False


def test_has_explicit_triggers_false_for_empty_list() -> None:
    assert has_explicit_triggers([]) is False
