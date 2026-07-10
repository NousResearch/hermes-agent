"""Tests for the cronjob tool schema shape.

Guards conditional create requirements and discoverability of the advanced
delivery/chaining/context controls that description-driven models otherwise
omit. See issue #32427 / PR #32448.
"""

from __future__ import annotations


def test_cronjob_schema_action_description_flags_create_requirements():
    """Create requirements must distinguish agent and no-agent jobs."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    action_desc = CRONJOB_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert "action=create" in action_desc
    assert "schedule" in action_desc
    assert "REQUIRED" in action_desc
    assert "prompt or skills" in action_desc
    assert "no_agent=True" in action_desc
    assert "script" in action_desc


def test_cronjob_schema_schedule_description_flags_required_for_create():
    """`schedule` description must explicitly state REQUIRED for action=create."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    schedule_desc = CRONJOB_SCHEMA["parameters"]["properties"]["schedule"]["description"]
    assert "REQUIRED" in schedule_desc
    assert "action=create" in schedule_desc


def test_cronjob_schema_required_array_unchanged():
    """`required[]` stays minimal — `action` only.

    The schema intentionally does NOT promote schedule/prompt into the
    top-level required array because they're only mandatory for
    action=create, not for list/remove/pause/etc. The description text
    carries the conditional requirement instead.
    """
    from tools.cronjob_tools import CRONJOB_SCHEMA

    assert CRONJOB_SCHEMA["parameters"]["required"] == ["action"]


def test_cronjob_schema_keeps_advanced_controls_discoverable():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert "all" in props["deliver"]["description"]
    assert "fire time" in props["deliver"]["description"]
    assert "most recent completed output" in props["context_from"]["description"]
    assert "reduce schema overhead" in props["enabled_toolsets"]["description"]
    assert "reply" in props["attach_to_session"]["description"]
    assert "brief in context" in props["attach_to_session"]["description"]
