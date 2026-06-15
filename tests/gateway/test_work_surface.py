"""Plan 067 Phase 1 — unit tests for the pure /work surface logic."""

from __future__ import annotations

import json

import pytest

from gateway import work_surface
from gateway.work_surface import (
    CUSTOM_ACTION_ID,
    PRESET_ACTION_PREFIX,
    SUBMIT_CALLBACK_ID,
    WorkValidationError,
)


# ---------------------------------------------------------------------------
# Presets load from the vendored data file
# ---------------------------------------------------------------------------


def test_load_presets_returns_the_vendored_specs():
    presets = work_surface.load_presets()
    assert len(presets) == 5
    ids = {p["id"] for p in presets}
    assert {"strata-coverage", "strata-lint", "strata-lift", "drain-run-summary", "backlog-seed-tool"} == ids
    # Every preset carries the five plan_lifecycle dimensions.
    for p in presets:
        assert p["goal"] and p["operator"] and p["repos"]
        assert isinstance(p["success_criteria"], list) and p["success_criteria"]
        assert isinstance(p["constraints"], list) and p["constraints"]
        assert isinstance(p["budget_usd"], (int, float))


def test_find_preset():
    assert work_surface.find_preset("strata-lint")["title"] == "strata lint"
    assert work_surface.find_preset("nope") is None


def test_load_presets_missing_file_is_empty(tmp_path):
    assert work_surface.load_presets(tmp_path / "absent.json") == []


# ---------------------------------------------------------------------------
# Menu blocks
# ---------------------------------------------------------------------------


def test_build_menu_blocks_has_preset_buttons_and_custom():
    presets = work_surface.load_presets()
    blocks = work_surface.build_menu_blocks(presets)

    assert blocks[0]["type"] == "section"
    buttons = [el for b in blocks if b["type"] == "actions" for el in b["elements"]]
    action_ids = {el["action_id"] for el in buttons}

    # One button per preset, with the regex-matched prefix + id in value.
    for p in presets:
        assert f"{PRESET_ACTION_PREFIX}{p['id']}" in action_ids
    assert CUSTOM_ACTION_ID in action_ids
    # Slack caps actions blocks at 5 elements each.
    for b in blocks:
        if b["type"] == "actions":
            assert len(b["elements"]) <= 5


# ---------------------------------------------------------------------------
# Modal view
# ---------------------------------------------------------------------------


def test_build_modal_view_prefilled_from_preset():
    preset = work_surface.find_preset("strata-coverage")
    view = work_surface.build_modal_view(preset, channel_id="C123")

    assert view["type"] == "modal"
    assert view["callback_id"] == SUBMIT_CALLBACK_ID
    meta = json.loads(view["private_metadata"])
    assert meta == {"channel_id": "C123", "preset_id": "strata-coverage"}

    by_id = {b["block_id"]: b for b in view["blocks"] if b.get("type") == "input"}
    assert by_id["goal"]["element"]["initial_value"].startswith("Add a `strata coverage")
    assert by_id["operator"]["element"]["initial_value"] == "blake"
    assert "blakeaber/agentic-hub" in by_id["repos"]["element"]["initial_value"]
    assert by_id["budget"]["element"]["initial_value"] == "25"
    # success/constraints joined one-per-line
    assert "\n" in by_id["success"]["element"]["initial_value"]


def test_build_modal_view_custom_is_empty():
    view = work_surface.build_modal_view(None, channel_id="C9")
    by_id = {b["block_id"]: b for b in view["blocks"] if b.get("type") == "input"}
    assert "initial_value" not in by_id["goal"]["element"]
    # operator still defaults to a sensible value
    assert by_id["operator"]["element"]["initial_value"] == "blake"
    assert json.loads(view["private_metadata"])["preset_id"] == ""


# ---------------------------------------------------------------------------
# view_submission → payload
# ---------------------------------------------------------------------------


def _view(values: dict, *, private_metadata: str = "{}") -> dict:
    state_values = {
        block: {f"{block}_input": {"value": val}} for block, val in values.items()
    }
    return {"state": {"values": state_values}, "private_metadata": private_metadata}


def test_parse_view_submission_full_payload():
    view = _view(
        {
            "goal": "Add a thing",
            "operator": "blake",
            "repos": "blakeaber/agentic-hub, blakeaber/army-of-one",
            "budget": "25",
            "success": "tests pass\nci green",
            "constraints": "Python only\nno terraform",
        }
    )
    payload = work_surface.parse_view_submission(view, plan_id="plan-abc12345")

    assert payload == {
        "goal": "Add a thing",
        "plan_id": "plan-abc12345",
        "repo": "blakeaber/agentic-hub",
        "ref": "main",
        "budget_usd": 25.0,
        "repos": ["blakeaber/agentic-hub", "blakeaber/army-of-one"],
        "success_criteria": ["tests pass", "ci green"],
        "constraints": ["Python only", "no terraform"],
        "operator": "blake",
    }


def test_parse_view_submission_defaults_empty_lists_to_none():
    view = _view(
        {"goal": "g", "operator": "blake", "repos": "r/x", "budget": "5",
         "success": "", "constraints": ""}
    )
    payload = work_surface.parse_view_submission(view, plan_id="p1")
    assert payload["success_criteria"] == ["none"]
    assert payload["constraints"] == ["none"]


def test_parse_view_submission_missing_goal_raises():
    view = _view({"goal": "  ", "operator": "blake", "repos": "r/x", "budget": "5"})
    with pytest.raises(WorkValidationError) as ei:
        work_surface.parse_view_submission(view, plan_id="p1")
    assert ei.value.block_id == "goal"


def test_parse_view_submission_bad_budget_raises():
    view = _view({"goal": "g", "operator": "blake", "repos": "r/x", "budget": "lots"})
    with pytest.raises(WorkValidationError) as ei:
        work_surface.parse_view_submission(view, plan_id="p1")
    assert ei.value.block_id == "budget"


def test_parse_view_submission_negative_budget_raises():
    view = _view({"goal": "g", "operator": "blake", "repos": "r/x", "budget": "-1"})
    with pytest.raises(WorkValidationError) as ei:
        work_surface.parse_view_submission(view, plan_id="p1")
    assert ei.value.block_id == "budget"


def test_parse_view_submission_empty_repos_raises():
    view = _view({"goal": "g", "operator": "blake", "repos": "  ", "budget": "5"})
    with pytest.raises(WorkValidationError) as ei:
        work_surface.parse_view_submission(view, plan_id="p1")
    assert ei.value.block_id == "repos"


def test_read_private_metadata():
    view = {"private_metadata": json.dumps({"channel_id": "C1", "preset_id": "x"})}
    assert work_surface.read_private_metadata(view) == {"channel_id": "C1", "preset_id": "x"}
    assert work_surface.read_private_metadata({"private_metadata": "not json"}) == {}
    assert work_surface.read_private_metadata({}) == {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_dispatch_result():
    text = work_surface.format_dispatch_result(
        "plan-x", {"created_issue_ids": ["AGE-1", "AGE-2"], "phase_ids": ["x-1"], "signaled": True}
    )
    assert "plan-x" in text
    assert "AGE-1, AGE-2" in text
    assert "Drain signaled: True" in text


def test_format_dispatch_result_empty():
    text = work_surface.format_dispatch_result("plan-y", {})
    assert "(none returned)" in text
    assert "Drain signaled: False" in text


def test_build_help_text_lists_presets():
    text = work_surface.build_help_text(work_surface.load_presets())
    assert "/work" in text
    assert "strata coverage" in text
    assert "/resume" in text and "/skip" in text
