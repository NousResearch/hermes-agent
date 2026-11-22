"""Tests for declarative workflow manifests (dynamic workflow layer)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hermes_cli.workflow_manifest import (
    Workflow,
    WorkflowError,
    WorkflowState,
    is_complete,
    next_steps,
    resume_state_path,
)


def _manifest() -> dict:
    return {
        "name": "review-pr",
        "max_iterations": 100,
        "max_children": 2,
        "steps": [
            {"id": "scan", "goal": "scan the diff"},
            {"id": "test", "goal": "run tests", "depends_on": ["scan"]},
            {"id": "report", "goal": "write report", "depends_on": ["test"]},
        ],
    }


def test_load_and_validate_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wf.json"
        p.write_text(json.dumps(_manifest()), encoding="utf-8")
        wf = Workflow.load(p)
        assert wf.name == "review-pr"
        assert len(wf.steps) == 3
        assert wf.max_iterations == 100
        assert wf.max_children == 2


def test_duplicate_ids_and_unknown_dependencies_are_rejected() -> None:
    bad = _manifest()
    bad["steps"][1]["id"] = "scan"
    try:
        Workflow.from_dict(bad)
    except WorkflowError:
        pass
    else:
        raise AssertionError("duplicate ids must be rejected")

    bad2 = _manifest()
    bad2["steps"][0]["depends_on"] = ["ghost"]
    try:
        Workflow.from_dict(bad2)
    except WorkflowError:
        pass
    else:
        raise AssertionError("unknown dependency must be rejected")


def test_resume_skips_completed_steps_and_respects_dependencies() -> None:
    wf = Workflow.from_dict(_manifest())
    state = WorkflowState()
    assert [s.id for s in next_steps(wf, state)] == ["scan"]

    state.mark("scan")
    assert [s.id for s in next_steps(wf, state)] == ["test"]

    state.mark("test")
    assert [s.id for s in next_steps(wf, state)] == ["report"]

    state.mark("report")
    assert next_steps(wf, state) == []
    assert is_complete(wf, state) is True


def test_state_round_trip_on_disk() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wf.json"
        p.write_text(json.dumps(_manifest()), encoding="utf-8")
        sp = resume_state_path(p)
        assert sp.name == "wf.state.json"

        state = WorkflowState.load(sp)
        state.mark("scan")
        state.save(sp)

        restored = WorkflowState.load(sp)
        assert restored.completed == {"scan"}
