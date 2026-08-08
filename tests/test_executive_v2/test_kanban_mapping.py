"""Tests for Phase 4B Kanban Mapping — pure TaskSpec -> create_task kwargs.

Covers the pure ``map_taskspec_to_kanban_payload`` and
``build_kanban_apply_preview`` functions. No DB, no storage, no I/O.
"""

from __future__ import annotations

import json

from agent.executive.kanban_mapping import (
    DEFAULT_CREATED_BY,
    DEFAULT_INITIAL_STATUS,
    DEFAULT_WORKSPACE_KIND,
    IDEMPOTENCY_KEY_PREFIX,
    MAX_PRIORITY,
    build_kanban_apply_preview,
    compute_idempotency_key,
    map_taskspec_to_kanban_payload,
)


def _make_spec(**overrides) -> dict:
    base = {
        "description": "investigate X",
        "assigned_profile": "researcher",
        "inputs": {"criterion_index": 0, "constraints": []},
        "expected_outputs": ["report.md"],
        "dependencies": [],
        "timeout_s": 60,
        "requires_user_input": False,
        "approval_id": None,
        "risk_level": "low",
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────
# 1. map_taskspec_to_kanban_payload tests
# ──────────────────────────────────────────────────────────────────────

def test_map_minimal_task_spec():
    """Empty TaskSpec -> minimal valid kwargs."""
    spec = _make_spec(description="")
    out = map_taskspec_to_kanban_payload(
        spec, spec_index=0, objective_id="obj-1"
    )
    assert out["title"] == "(no description)"  # placeholder
    assert out["body"]  # JSON-serialized metadata
    assert out["parents"] == ()  # resolved at apply time
    assert out["initial_status"] == DEFAULT_INITIAL_STATUS
    assert out["workspace_kind"] == DEFAULT_WORKSPACE_KIND
    assert out["idempotency_key"].startswith(IDEMPOTENCY_KEY_PREFIX)


def test_map_with_assigned_profile():
    """assigned_profile propagates to assignee."""
    spec = _make_spec(assigned_profile="researcher")
    out = map_taskspec_to_kanban_payload(
        spec, spec_index=0, objective_id="obj-1"
    )
    assert out["assignee"] == "researcher"


def test_map_idempotency_key_format():
    """Idempotency key is exec-v2-phase4b:<oid>:<idx>."""
    out = map_taskspec_to_kanban_payload(
        _make_spec(), spec_index=3, objective_id="obj-7"
    )
    assert out["idempotency_key"] == "exec-v2-phase4b:obj-7:3"
    # Direct helper.
    assert compute_idempotency_key("obj-7", 3) == "exec-v2-phase4b:obj-7:3"


def test_map_priority_from_risk_level_high():
    """risk_level=high -> priority=5."""
    spec = _make_spec(risk_level="high")
    out = map_taskspec_to_kanban_payload(
        spec, spec_index=0, objective_id="obj-1"
    )
    assert out["priority"] == 5


def test_map_priority_from_risk_level_medium():
    """risk_level=medium -> priority=2."""
    spec = _make_spec(risk_level="medium")
    out = map_taskspec_to_kanban_payload(
        spec, spec_index=0, objective_id="obj-1"
    )
    assert out["priority"] == 2


def test_map_requires_user_input_boost():
    """requires_user_input=True adds +3 to priority."""
    spec = _make_spec(requires_user_input=True, risk_level="high")
    out = map_taskspec_to_kanban_payload(
        spec, spec_index=0, objective_id="obj-1"
    )
    assert out["priority"] == 5 + 3  # 8


def test_map_session_id_equals_objective_id():
    """session_id is set to objective_id for state_meta linkage."""
    out = map_taskspec_to_kanban_payload(
        _make_spec(), spec_index=0, objective_id="obj-X"
    )
    assert out["session_id"] == "obj-X"


def test_map_initial_status_ready():
    """initial_status is 'ready' — never auto-dispatch (Phase 5+)."""
    out = map_taskspec_to_kanban_payload(
        _make_spec(), spec_index=0, objective_id="obj-1"
    )
    assert out["initial_status"] == "ready"


# ──────────────────────────────────────────────────────────────────────
# 2. build_kanban_apply_preview tests
# ──────────────────────────────────────────────────────────────────────

def test_build_preview_empty_list():
    """Empty task_specs -> empty kwargs list."""
    out = build_kanban_apply_preview([], objective_id="obj-1")
    assert out == []


def test_build_preview_multiple_specs():
    """N specs -> N kwargs, each with parents=() placeholder."""
    specs = [
        _make_spec(description=f"task-{i}", risk_level="low")
        for i in range(3)
    ]
    out = build_kanban_apply_preview(specs, objective_id="obj-1")
    assert len(out) == 3
    for i, kwargs in enumerate(out):
        assert kwargs["parents"] == ()  # placeholder
        assert kwargs["idempotency_key"] == f"exec-v2-phase4b:obj-1:{i}"


def test_build_preview_priority_clamped():
    """Priority is clamped to [0, MAX_PRIORITY]."""
    specs = [
        _make_spec(risk_level="high", requires_user_input=True),  # 5+3=8
        _make_spec(risk_level="high", requires_user_input=True),  # 5+3=8
    ]
    # Even with combined boosts, priority never exceeds MAX_PRIORITY.
    out = build_kanban_apply_preview(specs, objective_id="obj-1")
    for kwargs in out:
        assert 0 <= kwargs["priority"] <= MAX_PRIORITY


def test_build_preview_body_is_valid_json():
    """Body is a JSON-serialized dict with required fields."""
    out = build_kanban_apply_preview(
        [_make_spec(risk_level="medium")], objective_id="obj-1"
    )
    body = json.loads(out[0]["body"])
    assert body["phase"] == "executive_v2_phase4b"
    assert body["risk_level"] == "medium"
    assert "inputs" in body
    assert "expected_outputs" in body