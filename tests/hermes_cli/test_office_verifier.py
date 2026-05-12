from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _gate(gate_id: str, gate_type: str, artifacts: list[dict], **extra):
    data = {
        "id": gate_id,
        "title": gate_id.replace("-", " "),
        "requirement_ref": f"REQ-{gate_id}",
        "type": gate_type,
        "required": True,
        "commands": extra.pop("commands", []),
        "expected_artifacts": artifacts,
        "thresholds": extra.pop("thresholds", []),
        "evidence_policy": extra.pop("evidence_policy", "inspect_existing"),
        "allow_blocked": extra.pop("allow_blocked", False),
    }
    data.update(extra)
    return data


def _task_with_gates(conn, workspace: Path, gates: list[dict]) -> tuple[str, int]:
    tid = kb.create_task(
        conn,
        title="gate-bearing task",
        assignee="coder",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    claimed = kb.claim_task(conn, tid, claimer="coder", ttl_seconds=60)
    assert claimed and claimed.id == tid
    run_id = int(kb.active_run(conn, tid).id)
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps({"verification_gates": gates}), run_id),
    )
    conn.commit()
    return tid, run_id


def _verify(conn, tid: str, run_id: int):
    from hermes_cli.office_verifier import verify_task

    return verify_task(conn, tid, run_id=run_id, strict=True)


def test_verifier_writes_auditable_report_with_command_exit_codes_and_events(kanban_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ok.txt").write_text("real evidence\n", encoding="utf-8")
    gate = _gate(
        "unit-smoke",
        "test",
        [{"path": "ok.txt", "must_exist": True, "min_bytes": 5}],
        commands=[{"id": "python-version", "argv": [sys.executable, "-c", "print(123)"], "timeout_seconds": 30}],
        evidence_policy="both",
    )
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = _verify(conn, tid, run_id)
        events = kb.list_events(conn, tid)
        run = kb.get_run(conn, run_id)

    assert report["overall_status"] == "pass"
    assert report["gate_verdicts"][0]["commands_run"][0]["exit_code"] == 0
    assert Path(report["report_path"]).exists()
    assert any(e.kind == "office.verification.started" for e in events)
    assert any(e.kind == "office.verification.completed" for e in events)
    assert run.metadata["verification_report"]["overall_status"] == "pass"


@pytest.mark.parametrize(
    "gate,existing_files,expected_missing",
    [
        (
            _gate(
                "cargo-bench",
                "benchmark",
                [
                    {"path": "target/criterion/**/estimates.json", "must_exist": True, "min_bytes": 20},
                    {"path": "BENCHMARKS.md", "must_exist": True, "min_bytes": 50, "content_regex": "p99|throughput|criterion|benchmark"},
                ],
            ),
            {"BENCHMARKS.md": "# Benchmarks\nworker says cargo bench passed\n"},
            ["target/criterion/**/estimates.json"],
        ),
        (
            _gate(
                "live-pytest",
                "live_service_test",
                [
                    {"path": ".hermes/verification/live-service-ready.log", "must_exist": True, "min_bytes": 5},
                    {"path": "reports/live-pytest.xml", "must_exist": True, "min_bytes": 20, "content_regex": "testsuite"},
                ],
            ),
            {"reports/mock-pytest.xml": "<testsuite tests='1'></testsuite>"},
            [".hermes/verification/live-service-ready.log", "reports/live-pytest.xml"],
        ),
        (
            _gate(
                "helm-install",
                "helm_install",
                [
                    {"path": "reports/helm-install.log", "must_exist": True, "content_regex": "helm (install|upgrade)"},
                    {"path": "reports/helm-status.txt", "must_exist": True, "content_regex": "STATUS: deployed"},
                    {"path": "reports/kubectl-pods.txt", "must_exist": True, "content_regex": "Running|Completed"},
                ],
            ),
            {"reports/helm-lint.log": "lint ok", "reports/helm-template.yaml": "apiVersion: v1"},
            ["reports/helm-install.log", "reports/helm-status.txt", "reports/kubectl-pods.txt"],
        ),
        (
            _gate(
                "k6-p99",
                "k6",
                [{"path": "reports/k6-summary.json", "must_exist": True, "min_bytes": 20}],
                thresholds=[{"name": "p99_under_250ms", "source": "reports/k6-summary.json", "json_path": "metrics.http_req_duration.percentiles.p(99)", "op": "<=", "expected": 250}],
            ),
            {"scripts/load.js": "export default function() {}"},
            ["reports/k6-summary.json"],
        ),
        (
            _gate(
                "pypi-release",
                "release",
                [
                    {"path": "dist/*.whl", "must_exist": True, "min_bytes": 20},
                    {"path": "reports/pypi-url.txt", "must_exist": True, "content_regex": r"https://(test\.)?pypi.org/project/"},
                ],
            ),
            {"scripts/publish.sh": "twine upload dist/*"},
            ["dist/*.whl", "reports/pypi-url.txt"],
        ),
        (
            _gate(
                "benchmark-report",
                "benchmark_artifact",
                [{"path": "BENCHMARKS.md", "must_exist": True, "min_bytes": 50, "content_regex": "p99|throughput|benchmark", "must_be_committed": True}],
            ),
            {"target/criterion/foo/estimates.json": json.dumps({"mean": 1})},
            ["BENCHMARKS.md"],
        ),
    ],
)
def test_false_positive_claims_do_not_pass_without_real_required_artifacts(
    kanban_home, tmp_path, gate, existing_files, expected_missing
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for rel, content in existing_files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = _verify(conn, tid, run_id)

    verdict = report["gate_verdicts"][0]
    assert report["overall_status"] == "fail"
    assert verdict["status"] == "fail"
    for missing in expected_missing:
        assert missing in verdict["missing_artifacts"]


def test_scope_change_request_parser_accepts_exact_block_and_kanban_emits_event(kanban_home):
    from hermes_cli.office_scope import parse_scope_change_requests

    body = """SCOPE_CHANGE_REQUEST
requirement_ref: REQ-HELM-INSTALL
requested_change: accept helm template instead of real cluster install
reason: no local kind/minikube cluster is available after checking docker and kind
attempted_evidence: ran kind get clusters; checked docker info; inspected CI env
impact: deployability would no longer be proven by this task
options:
  - provision kind and rerun helm install
  - approve reducing this gate to template-only for local dev
END_SCOPE_CHANGE_REQUEST"""
    parsed = parse_scope_change_requests(body)
    assert parsed and parsed[0]["requirement_ref"] == "REQ-HELM-INSTALL"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="helm scope", assignee="coder")
        kb.add_comment(conn, tid, author="coder", body=body)
        events = kb.list_events(conn, tid)
    event = next(e for e in events if e.kind == "office.scope_change_requested")
    assert event.payload["requirement_ref"] == "REQ-HELM-INSTALL"
    assert event.payload["source"] == "comment"


def test_non_parseable_scope_caveat_does_not_emit_scope_change_event(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bad scope", assignee="coder")
        kb.add_comment(conn, tid, author="coder", body="Could not really do helm install, so I only ran helm template. Done.")
        events = kb.list_events(conn, tid)
    assert "office.scope_change_requested" not in [e.kind for e in events]


def test_benchmark_gate_with_only_command_and_no_artifact_fails_policy(kanban_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gate = _gate(
        "bench-command-only",
        "benchmark",
        [],
        commands=[{"id": "bench-smoke", "argv": [sys.executable, "-c", "print('bench ran')"], "timeout_seconds": 30}],
    )
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = _verify(conn, tid, run_id)

    verdict = report["gate_verdicts"][0]
    assert report["overall_status"] == "fail"
    assert any(c["name"] == "artifact_policy:benchmark_requires_real_artifact" for c in verdict["threshold_results"])


def test_completion_with_scope_change_request_blocks_instead_of_completing(kanban_home):
    body = """SCOPE_CHANGE_REQUEST
requirement_ref: REQ-PYPI-PUBLISH
requested_change: skip PyPI publish and keep local wheel only
reason: publishing credentials are unavailable after checking configured env
attempted_evidence: checked twine config and release workflow
impact: pip install from PyPI would not be proven
options:
  - provide PyPI token and rerun publish
  - approve local-only package verification
END_SCOPE_CHANGE_REQUEST"""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="publish scope", assignee="coder")
        kb.claim_task(conn, tid, claimer="coder", ttl_seconds=60)
        run_id = kb.active_run(conn, tid).id
        assert kb.complete_task(conn, tid, summary=body, expected_run_id=run_id) is False
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)

    assert task.status == "running"
    assert any(e.kind == "office.scope_change_requested" for e in events)
    assert any(e.kind == "office.completion_blocked_scope_change" for e in events)


def test_gate_bearing_completion_blocks_pending_scope_change_until_approved(kanban_home, tmp_path):
    from hermes_cli.office_verifier import final_completion_ready, verify_task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "BENCHMARKS.md").write_text("# Benchmarks\n\nthroughput: 100 req/s\np99: 10ms\n", encoding="utf-8")
    gate = _gate(
        "bench-report",
        "benchmark",
        [{"path": "BENCHMARKS.md", "must_exist": True, "min_bytes": 20, "content_regex": "p99|throughput|benchmark"}],
    )
    scope_body = """SCOPE_CHANGE_REQUEST
requirement_ref: REQ-BENCH
requested_change: accept smoke benchmark only
reason: load generator unavailable after checking PATH
attempted_evidence: checked k6 and benchmark scripts
impact: p99 target would not be proven
options:
  - install k6 and rerun
END_SCOPE_CHANGE_REQUEST"""
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = verify_task(conn, tid, run_id=run_id, strict=True)
        kb.add_comment(conn, tid, author="coder", body=scope_body)
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, 1)",
            (
                tid,
                run_id,
                "office.review.completed",
                json.dumps({
                    "approved": True,
                    "reviewed_report_path": report["report_path"],
                    "reviewed_report_hash": report["report_hash"],
                    "reviewed_run_id": report["run_id"],
                    "reviewed_diff_ref": "worktree",
                    "findings": [],
                    "gate_report_overall_status": "pass",
                }),
            ),
        )
        conn.commit()
        ok, reason = final_completion_ready(conn, tid)

    assert ok is False
    assert "pending scope change" in reason


def test_gate_bearing_office_completion_requires_verifier_and_independent_review(kanban_home, tmp_path):
    from hermes_cli.office_verifier import final_completion_ready, verify_task, latest_verification_summary

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "BENCHMARKS.md").write_text("# Benchmarks\n\nthroughput: 100 req/s\np99: 10ms\n", encoding="utf-8")
    gate = _gate(
        "bench-report",
        "benchmark",
        [{"path": "BENCHMARKS.md", "must_exist": True, "min_bytes": 20, "content_regex": "p99|throughput|benchmark"}],
    )
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])

        ok, reason = final_completion_ready(conn, tid)
        assert ok is False
        assert "missing verifier report" in reason
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id) is False
        events = kb.list_events(conn, tid)
        assert any(e.kind == "office.completion_blocked_verification" for e in events)

        report = verify_task(conn, tid, run_id=run_id, strict=True)
        assert report["overall_status"] == "pass"
        ok, reason = final_completion_ready(conn, tid)
        assert ok is False
        assert "missing independent reviewer" in reason
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id) is False

        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, 1)",
            (
                tid,
                run_id,
                "office.review.completed",
                json.dumps({
                    "approved": True,
                    "reviewed_report_path": report["report_path"],
                    "reviewed_report_hash": report["report_hash"],
                    "reviewed_run_id": report["run_id"],
                    "reviewed_diff_ref": "worktree",
                    "findings": [],
                    "gate_report_overall_status": "pass",
                }),
            ),
        )
        conn.commit()
        assert kb.complete_task(conn, tid, summary="reviewed and verified", metadata={"changed_files": []}, expected_run_id=run_id) is True
        assert latest_verification_summary(conn, tid)["overall_status"] == "pass"


def test_verifier_rejects_shell_command_strings(kanban_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gate = _gate(
        "shell-string",
        "test",
        [],
        commands=[{"id": "unsafe", "cmd": "echo unsafe", "timeout_seconds": 30}],
    )
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = _verify(conn, tid, run_id)

    assert report["overall_status"] == "fail"
    assert "shell command strings are rejected" in "; ".join(report["gate_errors"])


def test_report_json_outside_workspace_is_rejected(kanban_home, tmp_path):
    from hermes_cli.office_verifier import verify_task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gate = _gate("ok", "test", [])
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        with pytest.raises(ValueError, match="report_json escapes workspace"):
            verify_task(conn, tid, run_id=run_id, report_json=tmp_path / "outside-report.json")


def test_artifact_symlink_escape_fails_policy(kanban_home, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("p99: 1ms\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.txt").symlink_to(outside)
    gate = _gate("artifact-escape", "benchmark", [{"path": "report.txt", "must_exist": True, "content_regex": "p99"}])
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        report = _verify(conn, tid, run_id)

    assert report["overall_status"] == "fail"
    checks = report["gate_verdicts"][0]["threshold_results"]
    assert any(c["name"] == "artifact_workspace:report.txt" and c["status"] == "fail" for c in checks)


def test_pending_scope_change_blocks_non_gate_task_completion(kanban_home):
    scope_body = """SCOPE_CHANGE_REQUEST
requirement_ref: REQ-NO-GATE
requested_change: reduce work
reason: blocker after checking
attempted_evidence: checked tools
impact: acceptance would be weaker
options:
  - approve reduction
END_SCOPE_CHANGE_REQUEST"""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain office task", assignee="coder")
        kb.claim_task(conn, tid, claimer="coder", ttl_seconds=60)
        run_id = kb.active_run(conn, tid).id
        kb.add_comment(conn, tid, author="coder", body=scope_body)
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id) is False
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)

    assert task.status == "running"
    assert any(e.kind == "office.completion_blocked_scope_change" for e in events)


def test_verifier_rerun_requires_fresh_review_for_new_report_hash(kanban_home, tmp_path):
    from hermes_cli.office_verifier import final_completion_ready, verify_task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report_file = workspace / "BENCHMARKS.md"
    report_file.write_text("# Benchmarks\n\nthroughput: 100 req/s\np99: 10ms\n", encoding="utf-8")
    gate = _gate("bench-report", "benchmark", [{"path": "BENCHMARKS.md", "must_exist": True, "min_bytes": 20, "content_regex": "p99|throughput|benchmark"}])
    with kb.connect() as conn:
        tid, run_id = _task_with_gates(conn, workspace, [gate])
        first = verify_task(conn, tid, run_id=run_id, strict=True)
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, 1)",
            (tid, run_id, "office.review.completed", json.dumps({
                "approved": True,
                "reviewed_report_path": first["report_path"],
                "reviewed_report_hash": first["report_hash"],
                "reviewed_run_id": first["run_id"],
                "reviewed_diff_ref": "worktree",
                "findings": [],
                "gate_report_overall_status": "pass",
            })),
        )
        conn.commit()
        assert final_completion_ready(conn, tid)[0] is True
        report_file.write_text("# Benchmarks\n\nthroughput: 200 req/s\np99: 8ms\n", encoding="utf-8")
        second = verify_task(conn, tid, run_id=run_id, strict=True)
        assert second["report_hash"] != first["report_hash"]
        ok, reason = final_completion_ready(conn, tid)

    assert ok is False
    assert "missing independent reviewer" in reason


def test_cli_builtin_registry_includes_office_command():
    import hermes_cli.main as main

    assert "office" in main._BUILTIN_SUBCOMMANDS
