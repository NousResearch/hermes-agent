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


def test_cronjob_schema_exposes_artifact_expectations_contract():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    prop = CRONJOB_SCHEMA["parameters"]["properties"]["artifact_expectations"]
    desc = prop["description"]
    assert prop["type"] == "array"
    assert "required_for_delivery" in desc
    assert "warn_only" in desc
    assert "path/glob" in desc
    assert "vault:" in desc
    assert "hermes:" in desc
