"""Tests for the unified trigger framework parsing in agent/skill_utils.py."""

from __future__ import annotations

from agent.skill_utils import (
    derive_implicit_triggers,
    extract_skill_triggers,
    get_skill_triggers,
)


# ── extract_skill_triggers (Schema α — type-keyed dict) ──────────────────


def test_extract_returns_empty_when_no_metadata() -> None:
    assert extract_skill_triggers({}) == []
    assert extract_skill_triggers({"name": "foo"}) == []


def test_extract_returns_empty_when_no_triggers_field() -> None:
    fm = {"metadata": {"hermes": {"tags": ["productivity"]}}}
    assert extract_skill_triggers(fm) == []


def test_extract_mention_trigger_full_shape() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "mention": {
                        "regex": r"approve\s+\d+",
                        "channel_filter": ["bot-commands", "general"],
                    }
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    assert result == [
        {
            "type": "mention",
            "regex": r"approve\s+\d+",
            "channel_filter": ["bot-commands", "general"],
        }
    ]


def test_extract_button_trigger_with_pattern() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "button": {"custom_id_pattern": "approve_*"},
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    assert result == [{"type": "button", "custom_id_pattern": "approve_*"}]


def test_extract_reaction_trigger_with_emoji_and_age_limit() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "reaction": {
                        "emoji": "✅",
                        "channel_filter": ["회장단-test"],
                        "age_limit": "30d",
                    }
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    assert result == [
        {
            "type": "reaction",
            "emoji": "✅",
            "channel_filter": ["회장단-test"],
            "age_limit": "30d",
        }
    ]


def test_extract_cron_trigger() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "cron": {"schedule": "0 9 * * *"},
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    assert result == [{"type": "cron", "schedule": "0 9 * * *"}]


def test_extract_multiple_trigger_types_in_one_skill() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "mention": {"regex": "deploy"},
                    "button": {"custom_id_pattern": "deploy_*"},
                    "reaction": {"emoji": "🚀"},
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    types = sorted(t["type"] for t in result)
    assert types == ["button", "mention", "reaction"]
    assert any(t["type"] == "mention" and t["regex"] == "deploy" for t in result)
    assert any(t["type"] == "button" and t["custom_id_pattern"] == "deploy_*" for t in result)
    assert any(t["type"] == "reaction" and t["emoji"] == "🚀" for t in result)


def test_extract_unknown_trigger_type_skipped() -> None:
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "mention": {"regex": "x"},
                    "websocket": {"url": "ws://..."},  # unknown
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    assert len(result) == 1
    assert result[0]["type"] == "mention"


def test_extract_scalar_shorthand() -> None:
    """Scalar values are mapped to the canonical field for the trigger type."""
    fm = {
        "metadata": {
            "hermes": {
                "triggers": {
                    "slash": "approve",  # scalar → name field
                    "reaction": "✅",  # scalar → emoji field
                    "cron": "0 9 * * *",  # scalar → schedule field
                }
            }
        }
    }
    result = extract_skill_triggers(fm)
    by_type = {t["type"]: t for t in result}
    assert by_type["slash"] == {"type": "slash", "name": "approve"}
    assert by_type["reaction"] == {"type": "reaction", "emoji": "✅"}
    assert by_type["cron"] == {"type": "cron", "schedule": "0 9 * * *"}


def test_extract_handles_non_dict_metadata() -> None:
    """Malformed YAML where metadata is a string should not crash."""
    assert extract_skill_triggers({"metadata": "not a dict"}) == []
    assert extract_skill_triggers({"metadata": {"hermes": "not a dict"}}) == []
    assert extract_skill_triggers({"metadata": {"hermes": {"triggers": "not a dict"}}}) == []


# ── derive_implicit_triggers (rule 1 only — slash from frontmatter) ─────


def test_derive_returns_empty_for_missing_slash_command() -> None:
    assert derive_implicit_triggers({}) == []
    assert derive_implicit_triggers({"metadata": {"hermes": {"tags": []}}}) == []


def test_derive_returns_slash_trigger_when_slash_command_present() -> None:
    fm = {"metadata": {"hermes": {"slash_command": "approve"}}}
    result = derive_implicit_triggers(fm)
    assert result == [{"type": "slash", "name": "approve"}]


def test_derive_strips_whitespace_in_slash_command() -> None:
    fm = {"metadata": {"hermes": {"slash_command": "  deploy  "}}}
    result = derive_implicit_triggers(fm)
    assert result == [{"type": "slash", "name": "deploy"}]


def test_derive_ignores_non_string_slash_command() -> None:
    fm = {"metadata": {"hermes": {"slash_command": 12345}}}
    assert derive_implicit_triggers(fm) == []


def test_derive_ignores_empty_slash_command() -> None:
    fm = {"metadata": {"hermes": {"slash_command": "   "}}}
    assert derive_implicit_triggers(fm) == []


# ── get_skill_triggers (combined accessor) ──────────────────────────────


def test_get_skill_triggers_prefers_explicit() -> None:
    """When a skill has both explicit triggers AND a slash_command field,
    the explicit triggers win and has_explicit_triggers is True."""
    fm = {
        "metadata": {
            "hermes": {
                "slash_command": "old_name",
                "triggers": {"button": {"custom_id_pattern": "do_*"}},
            }
        }
    }
    triggers, is_explicit = get_skill_triggers(fm)
    assert is_explicit is True
    assert triggers == [{"type": "button", "custom_id_pattern": "do_*"}]


def test_get_skill_triggers_falls_back_to_implicit() -> None:
    fm = {"metadata": {"hermes": {"slash_command": "approve"}}}
    triggers, is_explicit = get_skill_triggers(fm)
    assert is_explicit is False
    assert triggers == [{"type": "slash", "name": "approve"}]


def test_get_skill_triggers_returns_empty_for_corpus_skill() -> None:
    """Most skills in the current Hermes corpus have neither explicit triggers
    nor a slash_command field. Result: empty list, is_explicit=False. BC for
    these skills is preserved through the existing prompt-builder injection
    path, NOT via derived triggers."""
    fm = {"metadata": {"hermes": {"tags": ["productivity"]}}}
    triggers, is_explicit = get_skill_triggers(fm)
    assert is_explicit is False
    assert triggers == []
