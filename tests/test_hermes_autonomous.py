from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time


def _modules(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import hermes_autonomous
    import hermes_delivery
    from hermes_cli import kanban_db

    return (
        importlib.reload(hermes_delivery),
        importlib.reload(hermes_autonomous),
        importlib.reload(kanban_db),
    )


def test_anti_shell_blocks_empty_completion_claims(tmp_path, monkeypatch):
    _, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)

    report = hermes_autonomous.run_anti_shell_check(str(tmp_path))

    assert report["status"] == "failed"
    assert any(check["name"] == "spec_artifact_exists" and not check["ok"] for check in report["checks"])
    assert (tmp_path / "home" / "delivery" / "anti-shell" / "report.json").exists()


def test_anti_shell_passes_when_real_evidence_exists(tmp_path, monkeypatch):
    hermes_delivery, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)
    delivery = tmp_path / "home" / "delivery"
    (delivery / "acceptance").mkdir(parents=True)
    (delivery / "ci").mkdir(parents=True)
    (delivery / "jobs" / "job-1").mkdir(parents=True)
    (delivery / "spec.md").write_text("files + tests", encoding="utf-8")
    (delivery / "tasks.md").write_text("issue pr files verify", encoding="utf-8")
    (delivery / "pr-status.json").write_text(json.dumps({"url": "https://example/pr/1"}), encoding="utf-8")
    (delivery / "ci" / "latest.json").write_text(json.dumps({"checks": [{"conclusion": "success"}]}), encoding="utf-8")
    (delivery / "acceptance" / "report.md").write_text("machine AI human browser", encoding="utf-8")
    (delivery / "jobs" / "job-1" / "execution-job.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "real-change.txt").write_text("real evidence\n", encoding="utf-8")
    hermes_delivery.run_deploy(
        hermes_delivery.DeployRequest(command="python3 --version", cwd=str(tmp_path), execute=True)
    )

    report = hermes_autonomous.run_anti_shell_check(str(tmp_path))

    assert report["status"] == "passed"


def test_autonomous_tick_creates_goal_mode_task_for_failed_gate(tmp_path, monkeypatch):
    _, hermes_autonomous, kanban_db = _modules(tmp_path, monkeypatch)

    result = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            create_tasks=True,
            board="default",
            assignee="default",
            goal_max_turns=17,
            workspace_path=str(tmp_path),
            repo=str(tmp_path),
        )
    )

    assert result["ok"] is True
    created = result["tick"]["created_tasks"]
    assert created
    with kanban_db.connect(board="default") as conn:
        row = conn.execute(
            "SELECT goal_mode, goal_max_turns, idempotency_key FROM tasks WHERE id = ?",
            (created[0]["id"],),
        ).fetchone()
    assert row["goal_mode"] == 1
    assert row["goal_max_turns"] == 17
    assert str(row["idempotency_key"]).startswith("hermes-autonomous:")
    assert result["tick"]["no_idle_enforced"] is True
    assert result["tick"]["heartbeat"]["no_idle_enforced"] is True
    assert (tmp_path / "home" / "delivery" / "autonomous-heartbeat.json").exists()


def test_autonomous_tick_retries_when_prior_goal_task_finished_but_gate_still_fails(tmp_path, monkeypatch):
    _, hermes_autonomous, kanban_db = _modules(tmp_path, monkeypatch)

    first = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            create_tasks=True,
            board="default",
            assignee="default",
            workspace_path=str(tmp_path),
            repo=str(tmp_path),
        )
    )
    first_task = first["tick"]["created_tasks"][0]["id"]
    with kanban_db.connect(board="default") as conn:
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), first_task),
        )
        conn.commit()

    second = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            create_tasks=True,
            board="default",
            assignee="default",
            workspace_path=str(tmp_path),
            repo=str(tmp_path),
        )
    )
    second_task = second["tick"]["created_tasks"][0]

    assert second_task["created"] is True
    assert second_task["id"] != first_task
    assert second_task["previous_task"]["id"] == first_task
    assert ":retry:" in second_task["idempotency_key"]


def test_autonomous_tick_treats_failed_deploy_gate_as_active_work(tmp_path, monkeypatch):
    _, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)
    monkeypatch.setattr(
        hermes_autonomous,
        "run_capability_audit",
        lambda **kwargs: {
            "passed": 12,
            "failed": 1,
            "total": 13,
            "results": [{"stage_id": "deploy", "status": "failed"}],
        },
    )
    monkeypatch.setattr(
        hermes_autonomous,
        "run_anti_shell_check",
        lambda repo=None: {"status": "passed", "passed": 10, "failed": 0},
    )

    result = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            create_tasks=True,
            board="default",
            assignee="default",
            workspace_path=str(tmp_path),
            repo=str(tmp_path),
        )
    )

    assert result["tick"]["next_action"] == "deploy"
    assert result["tick"]["reason"] == "gate_failed"
    assert result["tick"]["created_tasks"][0]["stage"] == "deploy"
    assert result["tick"]["heartbeat"]["mode"] == "active"
    assert result["tick"]["heartbeat"]["wake_agent"] is True


def test_autonomous_tick_recovers_blocked_task_even_when_gates_pass(tmp_path, monkeypatch):
    _, hermes_autonomous, kanban_db = _modules(tmp_path, monkeypatch)
    monkeypatch.setattr(
        hermes_autonomous,
        "run_capability_audit",
        lambda **kwargs: {"passed": 13, "failed": 0, "total": 13, "results": []},
    )
    monkeypatch.setattr(
        hermes_autonomous,
        "run_anti_shell_check",
        lambda repo=None: {"status": "passed", "passed": 10, "failed": 0},
    )
    with kanban_db.connect(board="default") as conn:
        blocked_id = kanban_db.create_task(
            conn,
            title="Hermes autonomous: fix deploy gate",
            body="blocked",
            assignee="default",
            created_by="test",
            idempotency_key="hermes-autonomous:deploy",
            initial_status="blocked",
            board="default",
        )

    result = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            create_tasks=True,
            board="default",
            assignee="default",
            workspace_path=str(tmp_path),
            repo=str(tmp_path),
        )
    )

    heartbeat = result["tick"]["heartbeat"]
    assert result["tick"]["reason"] == "all_gates_passed"
    assert heartbeat["mode"] == "blocked_task_needs_retry"
    assert heartbeat["wake_agent"] is True
    assert heartbeat["blocked_tasks"][0]["id"] == blocked_id
    assert result["tick"]["created_tasks"][0]["created"] is True
    assert result["tick"]["created_tasks"][0]["previous_task"]["id"] == blocked_id


def test_autonomous_tick_dispatch_uses_kanban_connection(tmp_path, monkeypatch):
    _, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)

    result = hermes_autonomous.autonomous_tick(
        hermes_autonomous.TickOptions(
            dispatch=True,
            board="default",
            repo=str(tmp_path),
            workspace_path=str(tmp_path),
        )
    )

    assert result["ok"] is True
    assert isinstance(result["tick"]["dispatch"], dict)
    assert "spawned" in result["tick"]["dispatch"]


def test_install_watchdog_creates_single_no_agent_cron_job(tmp_path, monkeypatch):
    _, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)

    first = hermes_autonomous.install_watchdog(
        schedule="every 2m",
        repo=str(tmp_path),
        workspace_path=str(tmp_path),
        assignee="default",
        goal_max_turns=19,
    )
    second = hermes_autonomous.install_watchdog(
        schedule="every 3m",
        repo=str(tmp_path),
        workspace_path=str(tmp_path),
        assignee="default",
        goal_max_turns=21,
    )
    status = hermes_autonomous.watchdog_status()

    assert first["job"]["no_agent"] is True
    assert first["job"]["script"] == "autonomous_watchdog.py"
    assert second["removed"] == [first["job"]["id"]]
    assert status["installed"] is True
    assert len(status["jobs"]) == 1
    assert status["jobs"][0]["schedule_display"] == "every 3m"
    script = (tmp_path / "home" / "scripts" / "autonomous_watchdog.py").read_text(encoding="utf-8")
    assert "len(spawned_raw) if isinstance(spawned_raw, list)" in script


def test_generated_watchdog_imports_runtime_from_repo(tmp_path, monkeypatch):
    _, hermes_autonomous, _ = _modules(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hermes_autonomous.py").write_text(
        "class TickOptions:\n"
        "    def __init__(self, **kwargs): pass\n"
        "def autonomous_tick(options):\n"
        "    return {'tick': {'created_tasks': [], 'heartbeat': {'wake_agent': False}, 'dispatch': {}}}\n",
        encoding="utf-8",
    )
    script = tmp_path / "watchdog.py"
    script.write_text(
        hermes_autonomous._watchdog_script_content(
            repo=str(repo),
            workspace_path=str(repo),
            board="default",
            assignee="default",
            goal_max_turns=10,
        ),
        encoding="utf-8",
    )

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert json.loads(result.stdout) == {"wakeAgent": False}
