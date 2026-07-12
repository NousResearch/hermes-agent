from __future__ import annotations

import importlib
import json


def _module(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import hermes_delivery

    return importlib.reload(hermes_delivery)


def test_delivery_state_initializes_pipeline(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    snapshot = hermes_delivery.dashboard_snapshot()

    assert snapshot["ok"] is True
    assert snapshot["summary"]["approval_cards"] is False
    assert [stage["id"] for stage in snapshot["stages"]][-2:] == ["deploy", "backwrite"]
    assert "approval cards" in " ".join(snapshot["principles"]).lower()


def test_update_stage_persists_status(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    hermes_delivery.update_stage("deploy", "ready", "policy passed")
    snapshot = hermes_delivery.dashboard_snapshot()
    deploy = next(stage for stage in snapshot["stages"] if stage["id"] == "deploy")

    assert deploy["status"] == "ready"
    assert deploy["notes"] == "policy passed"


def test_deploy_dry_run_writes_ready_run(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(
            command="python3 --version",
            cwd=str(tmp_path),
            execute=False,
        )
    )

    assert run["status"] == "ready"
    assert run["policy"]["allowed"] is True
    assert (tmp_path / "home" / "delivery" / "deploy-runs" / run["run_id"] / "deploy-run.json").exists()


def test_prod_deploy_without_rollback_is_blocked(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(
            environment="prod",
            command="python3 --version",
            cwd=str(tmp_path),
            execute=True,
            force=False,
        )
    )

    assert run["status"] == "blocked"
    assert any(check["name"] == "rollback_declared" and not check["ok"] for check in run["policy"]["checks"])


def test_ssh_deploy_requires_registered_target(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(
            adapter="ssh",
            target="missing-server",
            command="uname -a",
            execute=False,
        )
    )

    assert run["status"] == "blocked"
    assert any(check["name"] == "ssh_target_registered" and not check["ok"] for check in run["policy"]["checks"])


def test_ssh_target_uses_credential_reference_without_leaking_secret(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    target_path = tmp_path / "home" / "deploy-targets.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(
            {
                "schema": "hermes.deploy-targets.v1",
                "targets": {
                    "app-node": {
                        "host": "192.0.2.10",
                        "user": "root",
                        "password_env": "HERMES_DEPLOY_PASSWORD_APP_NODE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_DEPLOY_PASSWORD_APP_NODE", "super-secret-test-value")

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(
            adapter="ssh",
            target="app-node",
            command="hostname",
            execute=False,
        )
    )

    encoded = json.dumps(run, ensure_ascii=False)
    assert run["status"] == "ready"
    assert "env:HERMES_DEPLOY_PASSWORD_APP_NODE" in encoded
    assert "super-secret-test-value" not in encoded


def test_ssh_target_blocks_when_password_reference_is_not_loaded(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    target_path = tmp_path / "home" / "deploy-targets.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(
            {
                "schema": "hermes.deploy-targets.v1",
                "targets": {
                    "app-node": {
                        "host": "192.0.2.10",
                        "user": "root",
                        "password_env": "HERMES_DEPLOY_PASSWORD_APP_NODE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(
            adapter="ssh",
            target="app-node",
            command="hostname",
            execute=False,
        )
    )

    assert run["status"] == "blocked"
    assert any(check["name"] == "ssh_password_loaded" and not check["ok"] for check in run["policy"]["checks"])


def test_deploy_gate_uses_verified_run_artifact_not_state_ledger(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    state = hermes_delivery.load_state()
    state.setdefault("deploy", {}).setdefault("runs", []).append({"run_id": "fake", "status": "done"})
    hermes_delivery.save_state(state)

    failed = hermes_delivery.run_gate("deploy")
    assert failed["status"] == "failed"
    assert any(check["name"] == "deploy_run_exists" and not check["ok"] for check in failed["checks"])

    run = hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(command="python3 --version", cwd=str(tmp_path), execute=True)
    )
    payload_path = tmp_path / "home" / "delivery" / "deploy-runs" / run["run_id"] / "deploy-run.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["status"] = "verified"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    passed = hermes_delivery.run_gate("deploy")
    assert passed["status"] == "passed"
    assert any(check["name"] == "status_verified" and check["ok"] for check in passed["checks"])


def test_capability_manifest_covers_every_chart_stage(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    manifest = hermes_delivery.capability_manifest()
    capability_ids = [item["id"] for item in manifest["capabilities"]]
    stage_ids = [item["id"] for item in hermes_delivery.PIPELINE_STAGES]

    assert capability_ids == stage_ids
    assert manifest["no_approval_cards"] is True
    assert all(item["required_artifacts"] for item in manifest["capabilities"])
    assert all(item["gate_signals"] for item in manifest["capabilities"])


def test_gate_runner_blocks_after_three_failed_attempts(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    first = hermes_delivery.run_gate("spec")
    second = hermes_delivery.run_gate("spec")
    third = hermes_delivery.run_gate("spec")
    spec_stage = next(stage for stage in hermes_delivery.dashboard_snapshot()["stages"] if stage["id"] == "spec")

    assert first["status"] == "failed"
    assert second["gate_state"]["fail_count"] == 2
    assert third["gate_state"]["fail_count"] == 3
    assert spec_stage["status"] == "blocked"


def test_gate_runner_can_check_without_updating_state(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    before = hermes_delivery.dashboard_snapshot()["stages"]

    result = hermes_delivery.run_gate("spec", update_state=False)
    after = hermes_delivery.dashboard_snapshot()["stages"]

    assert result["status"] == "failed"
    assert result["gate_state"] == next(stage for stage in before if stage["id"] == "spec")
    assert after == before


def test_gate_runner_passes_when_stage_artifact_has_required_signals(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / "spec.md").write_text(
        "## Spec\n涉及文件: hermes_delivery.py\n验收测试: pytest\n禁止占位替代真实约束\n",
        encoding="utf-8",
    )

    result = hermes_delivery.run_gate("spec")
    spec_stage = next(stage for stage in hermes_delivery.dashboard_snapshot()["stages"] if stage["id"] == "spec")

    assert result["status"] == "passed"
    assert spec_stage["status"] == "done"


def test_story_gate_passes_with_well_formed_stories(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / "stories.md").write_text(
        "# Stories\n\n"
        "## Story 1\n"
        "- 角色: Hermes operator\n"
        "- 点击路径: Desktop -> Delivery -> GateRunner\n"
        "- 断言: 每个阶段返回 pass/fail 和缺失信号\n\n"
        "## Story 2\n"
        "- Role: release owner\n"
        "- Path: Desktop -> Delivery -> Dispatch -> PR monitor\n"
        "- Assertion: execution job and status artifacts are written\n",
        encoding="utf-8",
    )

    result = hermes_delivery.run_gate("story")

    assert result["status"] == "passed"
    assert any(check["name"] == "story_entries" and check["detail"] == "2" for check in result["checks"])
    assert all(check["ok"] for check in result["checks"] if check["name"].startswith("story:"))


def test_story_gate_fails_when_story_is_missing_role(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / "stories.md").write_text(
        "# Stories\n\n"
        "## Story 1\n"
        "- 点击路径: Desktop -> Delivery -> GateRunner\n"
        "- 断言: 每个阶段返回 pass/fail 和缺失信号\n",
        encoding="utf-8",
    )

    result = hermes_delivery.run_gate("story")

    assert result["status"] == "failed"
    assert any(check["name"] == "story:1:role" and not check["ok"] for check in result["checks"])


def test_story_gate_fails_when_story_is_missing_assertion(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / "stories.md").write_text(
        "# Stories\n\n"
        "## Story 1\n"
        "- 角色: Hermes operator\n"
        "- 点击路径: Desktop -> Delivery -> GateRunner\n",
        encoding="utf-8",
    )

    result = hermes_delivery.run_gate("story")

    assert result["status"] == "failed"
    assert any(check["name"] == "story:1:assertion" and not check["ok"] for check in result["checks"])


def test_story_gate_fails_when_no_story_sections_exist(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / "stories.md").write_text("# Stories\n\n角色 path assertion only in metadata\n", encoding="utf-8")

    result = hermes_delivery.run_gate("story")

    assert result["status"] == "failed"
    assert any(check["name"] == "story_entries" and not check["ok"] for check in result["checks"])


def test_execution_job_lifecycle_writes_dispatch_artifact(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    job = hermes_delivery.create_execution_job(
        title="Implement capability",
        owner="Hermes",
        executor="codex",
        branch="feat/hermes-loop",
        command="python3 --version",
        repo=str(tmp_path),
    )
    planned = hermes_delivery.run_execution_job(job["job_id"], execute=False)

    assert planned["status"] == "ready"
    assert planned["attempt"] == 1
    assert hermes_delivery.list_jobs()[0]["job_id"] == job["job_id"]
    assert (tmp_path / "home" / "delivery" / "jobs" / job["job_id"] / "execution-job.json").exists()
    assert (tmp_path / "home" / "delivery" / "jobs" / job["job_id"] / "job.log").exists()
    gate = hermes_delivery.run_gate("dispatch")
    assert gate["status"] == "passed"
    assert {check["name"] for check in gate["checks"]} >= {"dispatch_recorded", "job_log_exists", "has_command"}


def test_dispatch_gate_fails_for_unexecuted_job_without_log(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    job_dir = delivery / "jobs" / "job-stale"
    job_dir.mkdir(parents=True)
    (job_dir / "execution-job.json").write_text(
        json.dumps(
            {
                "schema": "hermes.execution-job.v1",
                "job_id": "job-stale",
                "owner": "Hermes",
                "executor": "codex",
                "branch": "feat/stale",
                "command": "python3 --version",
                "status": "queued",
                "attempt": 0,
            }
        ),
        encoding="utf-8",
    )

    gate = hermes_delivery.run_gate("dispatch")

    assert gate["status"] == "failed"
    assert any(check["name"] == "dispatch_recorded" and not check["ok"] for check in gate["checks"])
    assert any(check["name"] == "job_log_exists" and not check["ok"] for check in gate["checks"])


def test_acceptance_delivery_and_backwrite_artifacts_feed_gates(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    acceptance = hermes_delivery.write_acceptance_report(
        machine_evidence=["机器 pytest passed"],
        ai_review_evidence=["AI review Codex approved"],
        human_evidence="人工 browser 核心路径 3分钟 passed",
    )
    summary = hermes_delivery.write_delivery_summary("notification sent")
    backwrite = hermes_delivery.write_backwrite_blueprint("baseline updated")

    assert acceptance["gate"]["status"] == "passed"
    assert summary["gate"]["status"] == "passed"
    assert backwrite["gate"]["status"] == "passed"


def test_pr_and_ci_monitors_write_gate_artifacts(tmp_path, monkeypatch):
    hermes_delivery = _module(tmp_path, monkeypatch)

    def fake_run_json_command(args, cwd=None):
        if args[:3] == ["gh", "pr", "view"]:
            return 0, {
                "number": 12,
                "url": "https://github.example/pr/12",
                "state": "OPEN",
                "title": "feat: capability",
                "body": "Implements STO-1",
                "headRefName": "feat/hermes-loop",
                "closingIssuesReferences": [{"number": 1}],
            }, ""
        if args[:3] == ["gh", "pr", "checks"]:
            return 0, [
                {"name": "test", "state": "pass", "link": "https://github.example/checks/1"},
                {"name": "build", "state": "success", "link": "https://github.example/checks/2"},
            ], ""
        raise AssertionError(args)

    monkeypatch.setattr(hermes_delivery, "_run_json_command", fake_run_json_command)

    pr = hermes_delivery.refresh_pr_status(repo="owner/repo", pr="12", issue="STO-1")
    ci = hermes_delivery.refresh_ci_status(repo="owner/repo", ref="12")

    assert pr["state"] == "OPEN"
    assert pr["linked_issue"] is True
    assert ci["checks"][0]["conclusion"] == "success"
    assert hermes_delivery.run_gate("pr_monitor")["status"] == "passed"
    assert hermes_delivery.run_gate("ci_monitor")["status"] == "passed"
