"""Tests for the cronjob tool schema shape.

Guards the description text that flags ``schedule`` (and ``prompt``) as
REQUIRED for ``action=create`` — the load-bearing fix for description-driven
models (e.g. Grok) that omit schedule when the schema only lists ``action``
in ``required[]``. See issue #32427 / PR #32448.
"""

from __future__ import annotations


def test_cronjob_schema_action_description_flags_create_requirements():
    """`action` description must state schedule + prompt are required for create."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    action_desc = CRONJOB_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert "action=create" in action_desc
    assert "schedule" in action_desc
    assert "REQUIRED" in action_desc


def test_cronjob_schema_directs_non_destructive_actions_to_exact_name_lookup():
    """Run/pause/resume should use deterministic name resolution without listing."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    description = CRONJOB_SCHEMA["description"].lower()

    assert "exact name" in description
    assert "case-insensitive" in description
    assert "run/pause/resume" in description
    assert "do not call action='list' first" in description


def test_cronjob_schema_keeps_list_first_safety_for_remove():
    """Removing a job must still verify its ID before the destructive action."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    description = CRONJOB_SCHEMA["description"].lower()
    remove_guidance = description[description.index("for action='remove'"):]

    assert "action='list'" in remove_guidance
    assert "action='remove'" in remove_guidance
    assert "never guess" in remove_guidance


