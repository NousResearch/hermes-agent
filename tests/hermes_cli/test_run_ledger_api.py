import json
from pathlib import Path

from fastapi.testclient import TestClient

from hermes_cli import web_server


def _client() -> TestClient:
    return TestClient(web_server.app)


def _auth_headers() -> dict[str, str]:
    return {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}


def _write_ledger(home: Path, project: str, rows: list[dict]) -> Path:
    ledger = home / ".claude" / "teams" / project / "runs" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return ledger


def test_runs_api_requires_session_token() -> None:
    response = _client().get("/api/runs?project=staam")

    assert response.status_code == 401


def test_runs_api_merges_filters_and_sorts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_started",
                "run_id": "run_1",
                "task_id": "task_a",
                "agent_id": "claude",
                "run_type": "dispatch",
                "started_at": "2026-05-30T01:00:00+00:00",
                "command": "claude -p work",
            },
            {"not valid json": True},
            {
                "event": "run_finished",
                "run_id": "run_1",
                "task_id": "task_a",
                "agent_id": "claude",
                "run_type": "dispatch",
                "started_at": "2026-05-30T01:00:00+00:00",
                "finished_at": "2026-05-30T01:00:03+00:00",
                "duration_seconds": 3.25,
                "exit_code": 0,
                "classification": "ok",
                "stdout_tail": "done",
            },
            {
                "event": "run_finished",
                "run_id": "run_2",
                "task_id": "task_b",
                "agent_id": "deepseek",
                "run_type": "review",
                "started_at": "2026-05-30T02:00:00+00:00",
                "duration_seconds": 9,
                "exit_code": 1,
                "classification": "process_error",
                "stderr_tail": "failed",
            },
        ],
    )

    response = _client().get(
        "/api/runs?project=staam&classification=ok&limit=10",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["runs"][0]["run_id"] == "run_1"
    assert body["runs"][0]["command"] == "claude -p work"
    assert body["runs"][0]["classification"] == "ok"


def test_runs_api_filters_by_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_claude",
                "task_id": "task_a",
                "agent_id": "claude-code",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
            {
                "event": "run_finished",
                "run_id": "run_deepseek",
                "task_id": "task_b",
                "agent_id": "deepseek-tui",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().get("/api/runs?project=staam&agent_id=deepseek-tui", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["runs"][0]["run_id"] == "run_deepseek"


def test_runs_api_redacts_and_truncates_sensitive_details(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    long_stdout = "prefix-" + ("x" * 2100) + " sk-testsecret1234567890"
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_secret",
                "task_id": "task_secret",
                "agent_id": "claude",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "process_error",
                "command": "ANTHROPIC_API_KEY=secret-value claude -p work --token abcdefghijklmnop",
                "stdout_tail": long_stdout,
                "stderr_tail": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            },
        ],
    )

    response = _client().get("/api/runs?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    run = response.json()["runs"][0]
    assert "secret-value" not in run["command"]
    assert "abcdefghijklmnop" not in run["command"]
    assert "abcdefghijklmnopqrstuvwxyz" not in run["stderr_tail"]
    assert "sk-testsecret1234567890" not in run["stdout_tail"]
    assert "[REDACTED]" in run["command"]
    assert run["stdout_tail"].startswith("[truncated ")
    assert len(run["stdout_tail"]) < len(long_stdout)


def test_runs_summary_counts_and_duration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_1",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
                "duration_seconds": 2,
            },
            {
                "event": "run_finished",
                "run_id": "run_2",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
                "duration_seconds": 4,
            },
        ],
    )

    response = _client().get("/api/runs/summary?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["classification_counts"] == {"ok": 1, "timeout": 1}
    assert body["avg_duration_seconds"] == 3
    assert [run["run_id"] for run in body["recent_runs"]] == ["run_2", "run_1"]


def test_runs_summary_filters_by_agent_and_reports_agent_counts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_claude",
                "agent_id": "claude-code",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
                "duration_seconds": 2,
            },
            {
                "event": "run_finished",
                "run_id": "run_deepseek",
                "agent_id": "deepseek-tui",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
                "duration_seconds": 4,
            },
        ],
    )

    response = _client().get("/api/runs/summary?project=staam&agent_id=deepseek-tui", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["classification_counts"] == {"timeout": 1}
    assert body["agent_counts"] == {"deepseek-tui": 1}
    assert body["avg_duration_seconds"] == 4
    assert body["recent_runs"][0]["run_id"] == "run_deepseek"


def test_runs_api_rejects_path_traversal_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))

    response = _client().get("/api/runs?project=../staam", headers=_auth_headers())

    assert response.status_code == 400


def test_run_projects_discovers_ledgers_sorted_by_latest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "older",
        [
            {
                "event": "run_finished",
                "run_id": "run_old",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
        ],
    )
    _write_ledger(
        tmp_path,
        "newer",
        [
            {
                "event": "run_finished",
                "run_id": "run_new",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().get("/api/runs/projects", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["default_project"] == "newer"
    assert body["projects"] == [
        {"name": "newer", "total_runs": 1, "latest_started_at": "2026-05-30T02:00:00+00:00"},
        {"name": "older", "total_runs": 1, "latest_started_at": "2026-05-30T01:00:00+00:00"},
    ]


def test_run_projects_falls_back_to_staam_without_ledgers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))

    response = _client().get("/api/runs/projects", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json() == {"projects": [], "default_project": "staam"}


def test_run_tasks_groups_runs_by_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_ok",
                "task_id": "task_ok",
                "agent_id": "claude",
                "run_type": "review",
                "started_at": "2026-05-30T01:00:00+00:00",
                "duration_seconds": 2,
                "classification": "ok",
            },
            {
                "event": "run_finished",
                "run_id": "run_fail",
                "task_id": "task_fail",
                "agent_id": "deepseek",
                "run_type": "review",
                "started_at": "2026-05-30T02:00:00+00:00",
                "duration_seconds": 4,
                "classification": "process_error",
                "stderr_tail": "API_KEY=secret-value failed",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["tasks"][0]["task_id"] == "task_fail"
    assert body["tasks"][0]["status"] == "failed"
    assert body["tasks"][0]["agents"] == ["deepseek"]
    assert body["tasks"][0]["classifications"] == {"process_error": 1}
    assert "secret-value" not in body["tasks"][0]["last_error_excerpt"]
    assert body["tasks"][1]["task_id"] == "task_ok"
    assert body["tasks"][1]["status"] == "ok"


def test_run_tasks_filters_by_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_claude",
                "task_id": "task_shared",
                "agent_id": "claude-code",
                "run_type": "review",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
            {
                "event": "run_finished",
                "run_id": "run_deepseek",
                "task_id": "task_shared",
                "agent_id": "deepseek-tui",
                "run_type": "review",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam&agent_id=deepseek-tui", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["tasks"][0]["task_id"] == "task_shared"
    assert body["tasks"][0]["agents"] == ["deepseek-tui"]
    assert body["tasks"][0]["run_count"] == 1
    assert body["tasks"][0]["status"] == "timeout"


def test_run_tasks_marks_started_runs_stale(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server.time, "time", lambda: 1_800.0)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_started",
                "run_id": "run_stale",
                "task_id": "task_stale",
                "agent_id": "claude",
                "started_at": "1970-01-01T00:00:00+00:00",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["status"] == "stale"
    assert task["classifications"] == {"stale": 1}


def test_run_tasks_keeps_recent_started_runs_running(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server.time, "time", lambda: 1_800.0)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_started",
                "run_id": "run_running",
                "task_id": "task_running",
                "agent_id": "claude",
                "started_at": "1970-01-01T00:20:00+00:00",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["status"] == "running"
    assert task["classifications"] == {"running": 1}


def test_run_tasks_filters_by_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_ok",
                "task_id": "task_ok",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam&status=timeout", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["tasks"][0]["task_id"] == "task_timeout"


def test_run_tasks_includes_lifecycle_without_counting_as_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "lifecycle_event",
                "event_id": "life_compiled",
                "run_id": "life_compiled",
                "task_id": "task_life",
                "agent_id": "codex",
                "run_type": "compile",
                "phase": "compiled",
                "status": "completed",
                "started_at": "2026-05-30T00:00:00+00:00",
                "finished_at": "2026-05-30T00:00:00+00:00",
                "message": "Task card compiled",
            },
            {
                "event": "run_finished",
                "run_id": "run_dispatch",
                "task_id": "task_life",
                "agent_id": "claude-code",
                "run_type": "dispatch",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
            {
                "event": "lifecycle_event",
                "event_id": "life_gate",
                "run_id": "life_gate",
                "task_id": "task_life",
                "agent_id": "codex",
                "run_type": "gate_checked",
                "phase": "gate_checked",
                "status": "approved",
                "decision": "approved",
                "related_run_id": "run_dispatch",
                "started_at": "2026-05-30T02:00:00+00:00",
                "finished_at": "2026-05-30T02:00:00+00:00",
            },
        ],
    )

    response = _client().get("/api/runs/tasks?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["status"] == "ok"
    assert task["run_count"] == 1
    assert task["current_phase"] == "gate_checked"
    assert [item["phase"] for item in task["lifecycle"]] == ["compiled", "gate_checked"]
    assert task["lifecycle"][1]["decision"] == "approved"
    assert task["lifecycle"][1]["related_run_id"] == "run_dispatch"
    assert task["agents"] == ["claude-code", "codex"]


def test_runs_watchdog_counts_attention_tasks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server.time, "time", lambda: 1_800.0)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_ok",
                "task_id": "task_ok",
                "agent_id": "codex",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "ok",
            },
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "claude-code",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
                "stderr_tail": "token=secret-value timed out",
            },
            {
                "event": "run_started",
                "run_id": "run_stale",
                "task_id": "task_stale",
                "agent_id": "deepseek-tui",
                "started_at": "1970-01-01T00:00:00+00:00",
            },
            {
                "event": "run_finished",
                "run_id": "run_failed",
                "task_id": "task_failed",
                "agent_id": "claude-code",
                "started_at": "2026-05-30T03:00:00+00:00",
                "classification": "process_error",
            },
        ],
    )

    response = _client().get("/api/runs/watchdog?project=staam", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["project"] == "staam"
    assert body["total_tasks"] == 4
    assert body["attention_count"] == 3
    assert body["status_counts"] == {
        "ok": 1,
        "running": 0,
        "stale": 1,
        "timeout": 1,
        "failed": 1,
        "unknown": 0,
    }
    assert body["agent_counts"]["claude-code"]["timeout"] == 1
    assert body["agent_counts"]["claude-code"]["failed"] == 1
    attention_ids = {task["task_id"] for task in body["attention_tasks"]}
    assert attention_ids == {"task_timeout", "task_stale", "task_failed"}
    timeout_task = next(task for task in body["attention_tasks"] if task["task_id"] == "task_timeout")
    assert timeout_task["reason"] == "Agent execution timed out"
    assert "smaller scope" in timeout_task["suggested_action"]
    assert "secret-value" not in timeout_task["last_error_excerpt"]


def test_runs_watchdog_filters_by_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_claude",
                "task_id": "task_shared",
                "agent_id": "claude-code",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "timeout",
            },
            {
                "event": "run_finished",
                "run_id": "run_deepseek",
                "task_id": "task_shared",
                "agent_id": "deepseek-tui",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "ok",
            },
        ],
    )

    response = _client().get(
        "/api/runs/watchdog?project=staam&agent_id=deepseek-tui",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "deepseek-tui"
    assert body["total_tasks"] == 1
    assert body["attention_count"] == 0
    assert body["status_counts"]["ok"] == 1
    assert body["attention_tasks"] == []


def test_agent_effectiveness_scores_runs_and_handoffs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_ok",
                "task_id": "task_ok",
                "agent_id": "deepseek-tui",
                "started_at": "2026-05-30T01:00:00+00:00",
                "finished_at": "2026-05-30T01:00:02+00:00",
                "classification": "ok",
                "duration_seconds": 2,
            },
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "started_at": "2026-05-30T02:00:00+00:00",
                "finished_at": "2026-05-30T02:00:04+00:00",
                "classification": "timeout",
                "duration_seconds": 4,
            },
            {
                "event": "lifecycle_event",
                "run_id": "life_revision",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "phase": "execution_policy_applied",
                "status": "revision_needed",
                "policy_action": "revision_needed",
                "started_at": "2026-05-30T02:00:05+00:00",
            },
            {
                "event": "run_finished",
                "run_id": "run_claude",
                "task_id": "task_claude",
                "agent_id": "claude",
                "started_at": "2026-05-30T03:00:00+00:00",
                "finished_at": "2026-05-30T03:00:01+00:00",
                "classification": "ok",
                "duration_seconds": 1,
            },
        ],
    )
    (tmp_path / "agent-runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "handoff_1",
                        "agent_id": "deepseek-tui",
                        "status": "completed",
                        "duration_seconds": 3,
                        "updated_at": "2026-05-30T02:00:06Z",
                    },
                    {
                        "run_id": "handoff_2",
                        "agent_id": "claude",
                        "status": "failed",
                        "duration_seconds": 5,
                        "updated_at": "2026-05-30T03:00:06Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    response = _client().get("/api/agents/effectiveness?project=staam", headers=_auth_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["totals"]["agents"] == 2
    assert body["totals"]["run_count"] == 3
    deepseek = next(item for item in body["agents"] if item["agent_id"] == "deepseek-tui")
    assert deepseek["run_count"] == 2
    assert deepseek["ok_runs"] == 1
    assert deepseek["timeout_runs"] == 1
    assert deepseek["success_rate"] == 50.0
    assert deepseek["timeout_rate"] == 50.0
    assert deepseek["handoff_count"] == 1
    assert deepseek["handoff_success_rate"] == 100.0
    assert deepseek["revision_needed_count"] == 1
    assert deepseek["avg_duration_seconds"] == 3.0

    filtered = _client().get(
        "/api/agents/effectiveness?project=staam&agent_id=claude",
        headers=_auth_headers(),
    )
    assert filtered.status_code == 200
    assert [item["agent_id"] for item in filtered.json()["agents"]] == ["claude"]


def test_execution_policy_uses_effectiveness_to_switch_away_from_unstable_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": f"run_timeout_{i}",
                "task_id": f"task_noise_{i}",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": f"2026-05-30T01:00:0{i}+00:00",
                "classification": "timeout",
                "duration_seconds": 4,
            }
            for i in range(4)
        ]
        + [
            {
                "event": "run_finished",
                "run_id": "run_claude_ok",
                "task_id": "task_claude_ok",
                "agent_id": "claude",
                "model_ref": "claude_opus",
                "started_at": "2026-05-30T01:01:00+00:00",
                "classification": "ok",
                "duration_seconds": 1,
            },
            {
                "event": "run_finished",
                "run_id": "run_target",
                "task_id": "task_target",
                "agent_id": "opencode",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T01:02:00+00:00",
                "classification": "timeout",
                "duration_seconds": 2,
            },
        ],
    )

    response = _client().get(
        "/api/runs/tasks/task_target/execution-policy?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    policy = response.json()
    assert policy["action"] == "switch_agent"
    assert policy["next_agent_id"] == "claude"
    assert policy["next_model_ref"] == "claude_opus"


def _write_policy_agents_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
version: "test"
agents:
  - agent_id: claude
    name: Claude
    role: lead_implementer
    model_ref: claude_opus
    runtime: claude_code_cli
    tools: [file, terminal]
    permission: ask
    capabilities: [code_edit, test_run]
    risk_allowed: [R1, R2, R3]
  - agent_id: deepseek-tui
    name: DeepSeek
    role: fast_worker
    model_ref: deepseek_pro
    runtime: deepseek_tui_cli
    model_strategy:
      mode: fallback
      primary: opencode_go_deepseek_flash
      chain: [opencode_go_deepseek_flash, deepseek_flash, deepseek_pro]
      fallback_on: [timeout, rate_limited]
    tools: [file, terminal]
    permission: ask
    capabilities: [small_fix, test_generation, bug_reproduction]
    risk_allowed: [R0, R1, R2]
  - agent_id: codex
    name: Codex
    role: principal_engineer
    model_ref: codex_cli
    runtime: codex_cli
    tools: [file]
    permission: read_only
    capabilities: [code_review]
    risk_allowed: [R0, R1, R2, R3]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_policy_models_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
models:
  claude_opus:
    provider: anthropic
    model: claude-opus
    role: primary_claude_code
    status: active
  opencode_go_deepseek_flash:
    provider: opencode-go
    model: deepseek-flash
    role: experimental_cheap_task
    status: experimental
    tokens_per_million: 0.05
  deepseek_flash:
    provider: deepseek
    model: deepseek-flash
    role: primary_hermes
    status: active
    tokens_per_million: 0.1
  deepseek_pro:
    provider: deepseek
    model: deepseek-pro
    role: complex_reasoning
    status: active
    tokens_per_million: 0.5
  codex_cli:
    provider: external
    model: codex-cli
    role: principal_engineer
    status: active
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _patch_policy_config(monkeypatch, tmp_path: Path) -> None:
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    _write_policy_agents_yaml(agents_path)
    _write_policy_models_yaml(models_path)
    monkeypatch.setattr(web_server, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(web_server, "_MODELS_CONFIG_PATH", models_path)


def test_run_task_execution_policy_switches_agent_after_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().get(
        "/api/runs/tasks/task_timeout/execution-policy?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    policy = response.json()
    assert policy["action"] == "switch_agent"
    assert policy["should_execute"] is True
    assert policy["next_agent_id"] == "claude"
    assert policy["next_model_ref"] == "claude_opus"
    assert policy["latest_run_id"] == "run_timeout"


def test_run_task_execution_policy_switches_model_after_rate_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_rate",
                "task_id": "task_rate",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "rate_limited",
            },
        ],
    )

    response = _client().get(
        "/api/runs/tasks/task_rate/execution-policy?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    policy = response.json()
    assert policy["action"] == "switch_model"
    assert policy["should_execute"] is True
    assert policy["next_agent_id"] == "deepseek-tui"
    assert policy["next_model_ref"] == "deepseek_flash"


def test_run_task_execution_policy_uses_latest_run_when_order_is_oldest_first(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    task = {
        "task_id": "task_retry",
        "status": "timeout",
        "run_count": 2,
        "latest_run_id": "run_new",
        "runs": [
            {
                "event": "run_finished",
                "run_id": "run_old",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T01:00:00+00:00",
                "classification": "rate_limited",
            },
            {
                "event": "run_finished",
                "run_id": "run_new",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    }
    from agent.managed_agents.execution_policy import decide_execution_policy
    from agent.managed_agents.registry import load_agent_registry

    policy = decide_execution_policy(
        load_agent_registry(web_server._AGENTS_CONFIG_PATH),
        web_server._load_models_config(),
        task,
        task_type="tests",
        risk_level="R1",
    ).to_dict()

    assert policy["latest_run_id"] == "run_new"
    assert policy["latest_classification"] == "timeout"
    assert policy["action"] == "switch_agent"


def test_run_tasks_can_include_execution_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_rate",
                "task_id": "task_rate",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "rate_limited",
            },
        ],
    )

    response = _client().get(
        "/api/runs/tasks?project=staam&include_policy=true&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    task = response.json()["tasks"][0]
    assert task["execution_policy"]["action"] == "switch_model"
    assert task["execution_policy"]["next_model_ref"] == "deepseek_flash"


def test_run_tasks_policy_tolerates_unregistered_ledger_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_opencode",
                "task_id": "task_opencode",
                "agent_id": "opencode",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "ok",
            },
        ],
    )

    response = _client().get(
        "/api/runs/tasks?project=staam&include_policy=true&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    task = response.json()["tasks"][0]
    assert task["task_id"] == "task_opencode"
    assert task["execution_policy"]["action"] == "complete"


def test_record_run_task_execution_policy_appends_lifecycle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    ledger_path = _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/record?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["execution_policy"]["action"] == "switch_agent"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    lifecycle = rows[-1]
    assert lifecycle["event"] == "lifecycle_event"
    assert lifecycle["phase"] == "execution_policy_decided"
    assert lifecycle["policy_action"] == "switch_agent"
    assert lifecycle["execution_policy"]["next_agent_id"] == "claude"


def test_apply_run_task_execution_policy_queues_handoff(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    ledger_path = _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )

    response = _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["applied"] is True
    assert body["agent_run"]["status"] == "queued"
    assert body["agent_run"]["agent_id"] == "claude"
    assert body["agent_run"]["source"] == "execution_policy"
    stored_runs = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"]
    assert stored_runs[0]["run_id"] == body["agent_run"]["run_id"]
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    lifecycle = rows[-1]
    assert lifecycle["phase"] == "execution_policy_applied"
    assert lifecycle["status"] == "queued"
    assert lifecycle["handoff_run_id"] == body["agent_run"]["run_id"]


def test_apply_run_task_execution_policy_noops_on_complete(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_ok",
                "task_id": "task_ok",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "ok",
            },
        ],
    )

    response = _client().post(
        "/api/runs/tasks/task_ok/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applied"] is False
    assert body["reason"] == "policy_action_does_not_require_execution"
    assert body["agent_run"] is None
    assert not (tmp_path / "agent-runs.json").exists()


def test_apply_run_task_execution_policy_manual_review_does_not_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_auth",
                "task_id": "task_auth",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "auth_error",
            },
        ],
    )

    response = _client().post(
        "/api/runs/tasks/task_auth/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applied"] is False
    assert body["reason"] == "manual_review_required"
    assert body["agent_run"] is None
    assert not (tmp_path / "agent-runs.json").exists()


def test_execute_policy_handoff_run_completes_queued_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    ledger_path = _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    queued = _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    ).json()["agent_run"]

    def fake_turn(run: dict) -> dict:
        assert run["run_id"] == queued["run_id"]
        return {
            "summary": "executor completed",
            "status": "completed",
            "duration_seconds": 0.1,
            "api_calls": 1,
            "usage": {"input_tokens": 1},
            "model": "fake-model",
            "model_ref": run["model_ref"],
        }

    monkeypatch.setattr(web_server, "_run_policy_handoff_turn", fake_turn)

    response = _client().post(
        f"/api/agents/runs/{queued['run_id']}/execute?wait=true",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["result_summary"] == "executor completed"
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"][0]
    assert stored["status"] == "completed"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-2]["phase"] == "execution_handoff_started"
    assert rows[-1]["phase"] == "execution_handoff_finished"
    assert rows[-1]["status"] == "completed"


def test_executor_tick_runs_next_queued_policy_handoff(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )
    monkeypatch.setattr(
        web_server,
        "_run_policy_handoff_turn",
        lambda run: {"summary": "tick completed", "status": "completed", "model_ref": run["model_ref"]},
    )

    response = _client().post("/api/agents/runs/executor/tick?limit=1&wait=true", headers=_auth_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["executed"][0]["status"] == "completed"
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"][0]
    assert stored["result_summary"] == "tick completed"


def test_executor_tick_limit_zero_does_not_execute_policy_handoff(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    def fail_if_called(run: dict) -> dict:
        raise AssertionError(f"executor should not run for limit=0: {run.get('run_id')}")

    monkeypatch.setattr(web_server, "_run_policy_handoff_turn", fail_if_called)

    response = _client().post("/api/agents/runs/executor/tick?limit=0", headers=_auth_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"ok": True, "executed": [], "count": 0}
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"][0]
    assert stored["status"] == "queued"


def test_executor_tick_rejects_negative_limit() -> None:
    response = _client().post("/api/agents/runs/executor/tick?limit=-1", headers=_auth_headers())

    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_executor_tick_default_claims_handoff_without_inline_execution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    )

    def fake_start(run_id: str, *, timeout_seconds: float | None = None) -> dict:
        run = web_server._claim_agent_run_for_execution(run_id, worker_mode="process")
        run["executor"] = {**run.get("executor", {}), "pid": 12345, "timeout_seconds": timeout_seconds}
        return run

    monkeypatch.setattr(web_server, "_start_agent_run_worker", fake_start)

    response = _client().post("/api/agents/runs/executor/tick?limit=1", headers=_auth_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["executed"][0]["status"] == "running"
    assert body["executed"][0]["executor"]["mode"] == "process"
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"][0]
    assert stored["status"] == "running"
    assert stored["result_summary"] == "Queued by Apply Policy. Awaiting executor pickup."


def test_cancel_agent_run_only_allows_queued_or_running(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    queued = {
        "run_id": "run_queued",
        "agent_id": "claude",
        "status": "queued",
        "source": "execution_policy",
        "created_at": "2026-05-30T01:00:00Z",
    }
    completed = {
        "run_id": "run_completed",
        "agent_id": "claude",
        "status": "completed",
        "source": "execution_policy",
        "created_at": "2026-05-30T01:01:00Z",
    }
    (tmp_path / "agent-runs.json").write_text(
        json.dumps({"runs": [queued, completed]}),
        encoding="utf-8",
    )

    response = _client().post("/api/agents/runs/run_queued/cancel", headers=_auth_headers())

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"]
    assert stored[0]["status"] == "cancelled"

    response = _client().post("/api/agents/runs/run_completed/cancel", headers=_auth_headers())

    assert response.status_code == 400
    assert "Cannot cancel" in response.json()["detail"]


def test_executor_scheduler_status_and_controls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    web_server._stop_executor_scheduler()

    response = _client().get("/api/agents/runs/executor/scheduler", headers=_auth_headers())

    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False

    response = _client().post(
        "/api/agents/runs/executor/scheduler/start?interval_seconds=1&timeout_seconds=5",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is True
    assert body["interval_seconds"] == 1.0
    assert body["timeout_seconds"] == 5.0

    response = _client().post("/api/agents/runs/executor/scheduler/stop", headers=_auth_headers())

    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False


def test_executor_finish_does_not_overwrite_cancelled_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    queued = _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    ).json()["agent_run"]
    running = web_server._claim_agent_run_for_execution(queued["run_id"], worker_mode="process")
    runs = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"]
    runs[0]["status"] = "cancelled"
    (tmp_path / "agent-runs.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")

    result = web_server._finish_agent_run_execution(running, result={"summary": "late success"})

    assert result["status"] == "cancelled"
    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"][0]
    assert stored["status"] == "cancelled"
    assert stored["result_summary"] == "Queued by Apply Policy. Awaiting executor pickup."


def test_execute_policy_handoff_run_records_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    ledger_path = _write_ledger(
        tmp_path,
        "staam",
        [
            {
                "event": "run_finished",
                "run_id": "run_timeout",
                "task_id": "task_timeout",
                "agent_id": "deepseek-tui",
                "model_ref": "opencode_go_deepseek_flash",
                "started_at": "2026-05-30T02:00:00+00:00",
                "classification": "timeout",
            },
        ],
    )
    queued = _client().post(
        "/api/runs/tasks/task_timeout/execution-policy/apply?project=staam&task_type=tests&risk_level=R1",
        headers=_auth_headers(),
    ).json()["agent_run"]
    monkeypatch.setattr(
        web_server,
        "_run_policy_handoff_turn",
        lambda _run: (_ for _ in ()).throw(RuntimeError("executor failed")),
    )

    response = _client().post(
        f"/api/agents/runs/{queued['run_id']}/execute?wait=true",
        headers=_auth_headers(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == "executor failed"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "execution_handoff_finished"
    assert rows[-1]["status"] == "failed"


def test_policy_handoff_turn_allows_external_runtime_agents(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    _patch_policy_config(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "_valid_agent_ids", None)
    monkeypatch.setattr(web_server, "_external_runtime_agent_ids", None)

    captured: dict[str, object] = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "status": "completed",
                        "summary": "external handoff completed",
                        "duration_seconds": 0.2,
                        "api_calls": 0,
                        "model_ref": "codex_cli",
                    }
                ]
            }
        )

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate_task)

    result = web_server._run_policy_handoff_turn(
        {
            "run_id": "arun-external",
            "agent_id": "codex",
            "workspace": str(tmp_path),
            "prompt": "Review this handoff",
            "model_ref": "codex_cli",
        }
    )

    assert result["status"] == "completed"
    assert result["summary"] == "external handoff completed"
    assert captured["agent_id"] == "codex"
    parent = captured["parent_agent"]
    assert parent.terminal_cwd == str(tmp_path)


def test_kernelization_smoke_runs_full_executor_loop_without_external_cli(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "_valid_agent_ids", None)
    monkeypatch.setattr(web_server, "_external_runtime_agent_ids", None)

    response = _client().post(
        "/api/agents/runs/kernelization-smoke",
        headers=_auth_headers(),
        json={"project": "staam", "task_type": "tests", "risk_level": "R1"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["execution_policy"]["action"] == "switch_agent"
    assert body["agent_run"]["status"] == "completed"
    assert body["agent_run"]["agent_id"] == "claude"
    assert body["scheduler"]["queued_count"] == 0
    assert body["steps"] == [
        "seed_run_finished",
        "execution_policy_applied",
        "agent_run_queued",
        "executor_claimed",
        "agent_run_completed",
        "run_ledger_lifecycle_recorded",
    ]

    stored_runs = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"]
    assert stored_runs[0]["status"] == "completed"
    rows = [
        json.loads(line)
        for line in (tmp_path / ".claude" / "teams" / "staam" / "runs" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["run_type"] == "kernelization_smoke_seed"
    phases = [row.get("phase") for row in rows if row.get("event") == "lifecycle_event"]
    assert phases == [
        "execution_policy_applied",
        "execution_handoff_started",
        "execution_handoff_finished",
    ]


def test_external_agent_eval_records_cli_results_without_blocking_dashboard(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    _patch_policy_config(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "_valid_agent_ids", None)
    monkeypatch.setattr(web_server, "_external_runtime_agent_ids", None)

    class FakeChild:
        timeout_seconds = None

        def run_conversation(self, _prompt: str, task_id: str | None = None) -> dict:
            return {
                "completed": True,
                "final_response": f"HERMES_EVAL_OK {task_id}",
                "error": "",
                "exit_reason": "completed",
                "duration_seconds": 0.01,
            }

    monkeypatch.setattr(
        "tools.delegate_tool._build_external_cli_child",
        lambda **_kwargs: FakeChild(),
    )

    response = _client().post(
        "/api/agents/eval",
        headers=_auth_headers(),
        json={"project": "staam", "agent_ids": ["claude"], "timeout_seconds": 2},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["results"][0]["agent_id"] == "claude"
    assert body["results"][0]["classification"] == "ok"
    stored_runs = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))["runs"]
    assert stored_runs[0]["source"] == "external_agent_eval"
    assert stored_runs[0]["status"] == "completed"
    rows = [
        json.loads(line)
        for line in (tmp_path / ".claude" / "teams" / "staam" / "runs" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in rows] == ["run_started", "run_finished"]
    assert rows[-1]["classification"] == "ok"


def test_external_agent_eval_skips_non_external_agents(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_server.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(web_server, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    agents_path.write_text(
        """
version: test
agents:
  - agent_id: internal
    name: Internal
    role: managed_worker
    model_ref: deepseek_pro
    tools: [file]
    permission: ask
    capabilities: [test_generation]
    risk_allowed: [R0, R1]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    models_path.write_text("models: {deepseek_pro: {provider: deepseek, model: deepseek-pro}}\n", encoding="utf-8")
    monkeypatch.setattr(web_server, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(web_server, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(web_server, "_valid_agent_ids", None)
    monkeypatch.setattr(web_server, "_external_runtime_agent_ids", None)

    response = _client().post(
        "/api/agents/eval",
        headers=_auth_headers(),
        json={"project": "staam", "agent_ids": ["internal"], "timeout_seconds": 2},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"] == [
        {
            "agent_id": "internal",
            "status": "skipped",
            "classification": "not_external_runtime",
            "error": "Only external CLI runtime agents are eligible for this eval.",
        }
    ]


# ---------------------------------------------------------------------------
# Task Taxonomy tests
# ---------------------------------------------------------------------------

def test_task_type_enum_members():
    from agent.managed_agents.execution_policy import TaskType
    expected = {"implementation", "bugfix", "refactor", "test", "review",
                "architecture", "documentation", "smoke", "migration", "investigation"}
    members = set(TaskType.__members__.keys())
    assert members == expected, f"Expected {sorted(expected)}, got {sorted(members)}"


def test_task_type_from_valid_string():
    from agent.managed_agents.execution_policy import TaskType
    assert TaskType.from_task_type("bugfix") == TaskType.bugfix
    assert TaskType.from_task_type("review") == TaskType.review
    assert TaskType.from_task_type("smoke") == TaskType.smoke


def test_task_type_from_missing_falls_back_to_investigation():
    from agent.managed_agents.execution_policy import TaskType
    assert TaskType.from_task_type(None) == TaskType.investigation
    assert TaskType.from_task_type("") == TaskType.investigation


def test_task_type_from_invalid_falls_back_to_investigation():
    from agent.managed_agents.execution_policy import TaskType
    assert TaskType.from_task_type("not_a_real_type") == TaskType.investigation
    assert TaskType.from_task_type("BOGUS") == TaskType.investigation


def test_task_type_unknown_confidence_flag():
    from agent.managed_agents.execution_policy import TaskType
    result = TaskType.from_task_type("unknown_input")
    assert result == TaskType.investigation


def test_task_taxonomy_round_trips_in_enum():
    from agent.managed_agents.execution_policy import TaskType
    for member in TaskType:
        resolved = TaskType.from_task_type(member.value)
        assert resolved == member, f"{member.value} did not round-trip"


def test_kernelization_smoke_returns_taxonomy():
    from hermes_cli import web_server as ws
    from fastapi.testclient import TestClient
    client = TestClient(ws.app)
    headers = {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}
    resp = client.post("/api/agents/runs/kernelization-smoke",
                       json={"project": "staam", "task_type": "review"},
                       headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "task_taxonomy" in data
    assert data["task_taxonomy"]["task_type"] == "review"
    # review is a valid TaskType so confidence is high
    assert data["task_taxonomy"]["taxonomy_confidence"] == "high"


def test_kernelization_smoke_missing_task_type_falls_back():
    from hermes_cli import web_server as ws
    from fastapi.testclient import TestClient
    client = TestClient(ws.app)
    headers = {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}
    resp = client.post("/api/agents/runs/kernelization-smoke",
                       json={"project": "staam"},
                       headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "task_taxonomy" in data
    # default task_type in AgentRunSmokeCreate is "tests" which maps to TaskType.test
    assert data["task_taxonomy"]["task_type"] == "test"
    assert data["task_taxonomy"]["taxonomy_confidence"] == "high"


def test_old_tasks_without_task_type_are_compatible():
    """Simulate an old task with no task_type field and verify it works."""
    from agent.managed_agents.execution_policy import TaskType
    # Old tasks without task_type should resolve to investigation
    task_type = TaskType.from_task_type(None)
    assert task_type == TaskType.investigation
    # The default task_type field in models should not break old API consumers
    assert task_type.value == "investigation"
