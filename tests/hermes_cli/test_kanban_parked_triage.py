import json

from hermes_cli import kanban_parked_triage as triage


REQUIRED_TYPES = {
    "review-required-with-child",
    "review-required-no-child",
    "returned-awaiting-rework",
    "needs-evidence",
    "needs-human-decision",
    "budget-exhausted-with-artifact",
    "infra-missing",
    "stale-or-contradictory-trace-intent",
    "historical-only-not-current-block",
}

EXPECTED_AUTOMATION_LEVELS = {
    "可自动推进",
    "需窄续跑",
    "需人类决策",
    "需另立基础设施 issue",
    "无需推进",
}

EXPECTED_EVIDENCE_SOURCES = {
    "kanban.task",
    "kanban.child",
    "kanban.run",
    "kanban.comment",
    "kanban.event",
    "github.issue",
    "github.pr",
    "github.comment",
}


def snapshot(**overrides):
    base = {
        "issue": {
            "number": 156,
            "url": "https://github.com/GTZhou/TianGongKaiWu/issues/156",
            "state": "OPEN",
            "title": "test issue",
            "labels": [],
            "body": "",
            "comments": [],
        },
        "prs": [],
        "task": {
            "id": "t_parent",
            "title": "executor task",
            "body": "",
            "status": "blocked",
            "result": None,
        },
        "parents": [],
        "children": [],
        "comments": [],
        "events": [],
        "runs": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value
    return base


def completed_with_block_history():
    return snapshot(
        task={"status": "done", "result": "closed; historical blocked was only a timing sample"},
        issue={"state": "CLOSED", "labels": ["已关闭"]},
        events=[{"kind": "blocked", "payload": {"reason": "needs-human-decision"}}],
        comments=[{"body": "历史 blocked 仅作为耗时样本；当前已经关闭。"}],
    )


CASES = [
    (
        "review-required-with-child",
        snapshot(
            comments=[{"body": "review-required: implementation ready for audit"}],
            children=[{"id": "t_review", "title": "审计 child", "assignee": "yushidafu", "status": "todo"}],
            prs=[{"number": 10, "url": "https://github.com/GTZhou/TianGongKaiWu/pull/10", "state": "OPEN"}],
            runs=[{"summary": "PR ready; tests pass", "outcome": "blocked"}],
        ),
    ),
    (
        "review-required-no-child",
        snapshot(
            comments=[{"body": "review-required: PR ready but no review child exists"}],
            children=[],
            prs=[{"number": 10, "url": "https://github.com/GTZhou/TianGongKaiWu/pull/10", "state": "OPEN"}],
            runs=[{"summary": "head_sha=abc123 tests_run=pytest", "outcome": "blocked"}],
        ),
    ),
    (
        "returned-awaiting-rework",
        snapshot(
            comments=[{"body": "审核退回：blocking_findings=[B1]，等待返工。"}],
            issue={"labels": ["已退回"]},
        ),
    ),
    (
        "needs-evidence",
        snapshot(
            comments=[{"body": "PR URL: 待补；tests: TODO；trace receipt: TBD"}],
            runs=[{"summary": "submitted but evidence is incomplete", "outcome": "blocked"}],
        ),
    ),
    (
        "needs-human-decision",
        snapshot(
            comments=[{"body": "needs-human-decision: 需要少东家拍板是否允许外发。"}],
        ),
    ),
    (
        "budget-exhausted-with-artifact",
        snapshot(
            comments=[{"body": "Iteration budget exhausted after implementation; PR https://github.com/GTZhou/TianGongKaiWu/pull/160 exists"}],
            prs=[{"number": 160, "url": "https://github.com/GTZhou/TianGongKaiWu/pull/160", "state": "OPEN"}],
            runs=[{"outcome": "blocked", "error": "Iteration budget exhausted", "summary": "head_sha=abc tests_run=pytest"}],
        ),
    ),
    (
        "infra-missing",
        snapshot(
            comments=[{"body": "trace-missing: Platform 'telegram' is not configured; kanban-trace helper missing"}],
        ),
    ),
    (
        "stale-or-contradictory-trace-intent",
        snapshot(
            task={"title": "trace intent", "body": "[trace-intent:v1] event=returned source_task=t_review"},
            comments=[{"body": "trace intent asks returned, but source task conclusion is approved and PR merged"}],
            runs=[{"summary": "approved=true merged=true", "outcome": "blocked"}],
        ),
    ),
    ("historical-only-not-current-block", completed_with_block_history()),
]


def test_all_required_parked_types_are_documented_in_runbook():
    assert REQUIRED_TYPES <= set(triage.RUNBOOK)


def test_runbook_schema_enums_align_with_documented_contract():
    assert {item["automation_level"] for item in triage.RUNBOOK.values()} <= EXPECTED_AUTOMATION_LEVELS
    assert set(triage.RESULT_SCHEMA["evidence_source_values"]) == EXPECTED_EVIDENCE_SOURCES


import pytest


@pytest.mark.parametrize(("expected", "snap"), CASES)
def test_classifies_required_parked_types(expected, snap):
    result = triage.diagnose_snapshot(snap)

    assert result["schema"] == "kanban-parked-triage-result:v1"
    assert result["parked_type"] == expected
    assert result["recommendation"]["next_action"]
    assert result["evidence"], result
    assert result["side_effects"] == []


def test_running_task_with_accepted_or_submitted_labels_is_not_current_block():
    snap = snapshot(
        task={"status": "running"},
        issue={"labels": ["enhancement", "已接单", "已提交"]},
        comments=[{"body": "历史 trace 里提到 gateway watcher，但当前任务仍在正常执行。"}],
    )

    result = triage.diagnose_snapshot(snap)

    assert result["is_current_block"] is False
    assert result["parked_type"] == "historical-only-not-current-block"
    assert result["side_effects"] == []



def test_missing_evidence_takes_precedence_over_review_required_placeholder():
    snap = snapshot(
        comments=[{"body": "review-required: PR URL: 待补；tests: TODO；trace receipt: TBD"}],
        children=[],
        prs=[],
        runs=[{"summary": "review requested but evidence missing", "outcome": "blocked"}],
    )

    result = triage.diagnose_snapshot(snap)

    assert result["parked_type"] == "needs-evidence"
    assert result["recommendation"]["automation_level"] == "需窄续跑"



def test_unknown_task_allows_github_current_block_evidence():
    snap = snapshot(
        task={"id": "t_missing", "status": "unknown"},
        issue={"labels": ["已退回"], "comments": [{"body": "审核退回：等待返工。"}]},
        comments=[],
        events=[],
        runs=[],
    )

    result = triage.diagnose_snapshot(snap)

    assert result["is_current_block"] is True
    assert result["parked_type"] == "returned-awaiting-rework"



def test_github_read_failure_returns_nested_issue_fallback(monkeypatch):
    def fake_run_json(command):
        return None, "auth missing"

    monkeypatch.setattr(triage, "_run_json", fake_run_json)

    snapshot_part, warnings = triage.load_github_snapshot("156", "GTZhou/TianGongKaiWu")

    assert warnings == ["github issue read failed: auth missing"]
    assert snapshot_part["issue"]["number"] == 156
    assert snapshot_part["issue"]["url"] == "https://github.com/GTZhou/TianGongKaiWu/issues/156"
    assert snapshot_part["prs"] == []



def test_schema_and_runbook_cli_outputs_are_parseable_json(capsys):
    assert triage.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$id"] == "kanban-parked-triage-result:v1"

    assert triage.main(["runbook", "--format", "json"]) == 0
    runbook = json.loads(capsys.readouterr().out)
    assert runbook["schema"] == "kanban-parked-triage-runbook:v1"
    assert REQUIRED_TYPES <= set(runbook["runbook"])



def test_json_schema_and_markdown_summary_are_stable_and_redacted():
    raw_locator = "telegram:" + "-1001234567890:17585"
    token_marker = "token" + "=synthetic-secret"
    snap = snapshot(
        comments=[{"body": f"needs-human-decision: send to {raw_locator}? {token_marker}"}],
    )
    result = triage.diagnose_snapshot(snap)
    rendered_json = json.loads(triage.render_json(result))
    rendered_md = triage.render_markdown(result)

    assert rendered_json["schema"] == "kanban-parked-triage-result:v1"
    assert rendered_json["parked_type"] == "needs-human-decision"
    assert raw_locator not in rendered_md
    assert token_marker not in rendered_md
    assert "needs-human-decision" in rendered_md
    assert "## Evidence" in rendered_md
    assert "## Recommendation" in rendered_md


def test_cli_diagnose_can_use_fixture_without_external_commands(tmp_path, capsys):
    fixture = tmp_path / "snapshot.json"
    fixture.write_text(json.dumps(CASES[0][1]), encoding="utf-8")

    code = triage.main(["diagnose", "--issue", "156", "--fixture", str(fixture), "--format", "json"])

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["parked_type"] == "review-required-with-child"
    assert output["side_effects"] == []
