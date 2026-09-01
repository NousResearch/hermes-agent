from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "governance-crossref.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("governance_crossref_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(event_id, event_type, *, source_time, finding_id="f-1", severity="high", owner="octacon", task_id=None, resolution_ref=None):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "occurred_at": source_time,
        "created_at": source_time,
        "payload": {
            "finding_id": finding_id,
            "detector": "test.detector",
            "subject_type": "profile",
            "subject_id": "octacon",
            "severity": severity,
            "source_observed_at": source_time,
            "evidence_refs": ["test:tests/test_governance_findings.py"],
            "owner": owner,
            "task_id": task_id,
            "resolution_ref": resolution_ref,
            "dedupe_key": "test.detector:profile:octacon",
        },
    }


def test_projection_groups_current_findings_and_does_not_use_legacy_files(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "governance" / "logboard").mkdir(parents=True)
    (tmp_path / "governance" / "logboard" / "denji-profile-review-legacy.json").write_text(
        json.dumps({"profiles": {"octacon": {"total_score": 1}}}), encoding="utf-8"
    )

    projection = mod.project_events([_event("opened", "governance.finding.opened", source_time=100)], now=200)

    assert projection["findings"][0]["finding_id"] == "f-1"
    assert projection["findings"][0]["state"] == "open"
    assert projection["groups"]["severity"]["high"] == 1
    assert projection["groups"]["owner"]["octacon"] == 1
    assert projection["findings"][0]["age_days"] == 0


def test_projection_is_byte_stable_and_terminal_findings_are_not_open(tmp_path):
    mod = _load_module()
    events = [
        _event("resolved", "governance.finding.resolved", source_time=200, task_id="task-1", resolution_ref="test:tests/test_resolution.py"),
        _event("opened", "governance.finding.opened", source_time=100),
    ]

    first = mod.render_projection(mod.project_events(events, now=200 + 86400 * 8))
    second = mod.render_projection(mod.project_events(list(reversed(events)), now=200 + 86400 * 8))

    assert first == second
    decoded = json.loads(first)
    assert decoded["findings"][0]["state"] == "resolved"
    assert decoded["findings"][0]["resolution_ref"] == "test:tests/test_resolution.py"


def test_projection_marks_missing_evidence_owner_and_resolution_proof():
    mod = _load_module()
    event = _event("opened", "governance.finding.opened", source_time=100, owner=None)
    event["payload"]["evidence_refs"] = []
    projection = mod.project_events([event], now=100 + 86400 * 8)
    item = projection["findings"][0]

    assert item["missing"] == ["evidence", "owner", "action_task"]
    assert item["overdue"] is True
    assert projection["emit"] is True


def test_projection_emits_only_new_worsened_overdue_or_human_decision_findings():
    mod = _load_module()
    event = _event("opened", "governance.finding.opened", source_time=100, task_id="task-1")
    now = 100 + 86400
    baseline = mod.project_events([event], now=now)
    unchanged = mod.project_events([event], now=now, previous=baseline)

    assert unchanged["emit"] is False
    assert unchanged["attention"] == []

    worsened_event = _event("opened-2", "governance.finding.opened", source_time=100, severity="critical")
    worsened = mod.project_events([worsened_event], now=now, previous=baseline)
    assert worsened["emit"] is True
    assert worsened["attention"][0]["reason"] == "worsened"


def test_cli_writes_projection_and_closure_view_from_canonical_events(tmp_path, monkeypatch):
    mod = _load_module()
    output_dir = tmp_path / "logboard"
    events = [_event("opened", "governance.finding.opened", source_time=100)]

    result = mod.write_projection(output_dir, events, now=100)

    assert (output_dir / "current-findings.json").read_text(encoding="utf-8") == mod.render_projection(result)
    closure = (output_dir / "denji-closure-view.md").read_text(encoding="utf-8")
    assert closure.splitlines()[0] == "Finding | Evidence | Owner | Age | Action task | State | Resolution proof | Next action"
    assert "f-1" in closure
    assert result["metrics"]["human_decisions_waiting"] == 1
    assert "False-positive rate" in closure
