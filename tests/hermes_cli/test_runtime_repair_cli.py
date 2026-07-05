import json
import os
import subprocess
import sys
from types import SimpleNamespace

from hermes_cli import active_sessions


def _write_registry(home, entries):
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(runtime / "active_sessions.json", entries)


def _isolate_goal_home(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import goals

    home = tmp_path / ".hermes-goals"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    goals._DB_CACHE.clear()
    return home


def test_runtime_active_sessions_status_shows_safe_owner_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    from hermes_cli import runtime_cli

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)
    _write_registry(
        home,
        [
            {
                "lease_id": "lease-1",
                "session_id": "session-1",
                "session_key": "session-1",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": 12345,
                "process_start_time": 1.0,
                "created_at": 10.0,
                "updated_at": 12.0,
                "metadata": {
                    "current_tool": "terminal",
                    "current_tool_args": "do not expose this command",
                },
            }
        ],
    )

    assert runtime_cli._cmd_active_sessions_status(SimpleNamespace()) == 0

    out = capsys.readouterr().out
    assert "checked=1" in out
    assert "live=1" in out
    assert "session-1" in out
    assert "owner_kind=cli" in out
    assert "current_tool_args" not in out
    assert "do not expose" not in out


def test_runtime_active_sessions_repair_stale_only_removes_only_dead_leases(
    tmp_path,
    monkeypatch,
    capsys,
):
    from hermes_cli import runtime_cli

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        active_sessions,
        "_pid_alive",
        lambda pid, *_args: int(pid) == 111,
    )
    _write_registry(
        home,
        [
            {
                "lease_id": "live",
                "session_id": "live-session",
                "session_key": "live-session",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": 111,
            },
            {
                "lease_id": "dead",
                "session_id": "dead-session",
                "session_key": "dead-session",
                "surface": "gateway:telegram",
                "owner_kind": "gateway",
                "pid": 222,
            },
        ],
    )

    rc = runtime_cli._cmd_active_sessions_repair(
        SimpleNamespace(stale_only=True, session_id=None)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "checked=2" in out
    assert "stale=1" in out
    assert "repaired=1" in out
    assert "skipped_live=1" in out
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "live-session"
    ]


def test_runtime_active_sessions_repair_stale_only_refuses_targeted_live_owner(
    tmp_path,
    monkeypatch,
    capsys,
):
    from hermes_cli import runtime_cli

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)
    _write_registry(
        home,
        [
            {
                "lease_id": "live",
                "session_id": "live-session",
                "session_key": "live-session",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": 111,
            }
        ],
    )

    rc = runtime_cli._cmd_active_sessions_repair(
        SimpleNamespace(stale_only=True, session_id="live-session")
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "checked=1" in out
    assert "repaired=0" in out
    assert "skipped_live=1" in out
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "live-session"
    ]


def test_runtime_active_sessions_status_registered_in_main_cli(tmp_path):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / ".hermes")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "runtime",
            "active-sessions",
            "status",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "active sessions:" in result.stdout


def test_runtime_active_sessions_diagnose_no_lock_reports_degraded_owner(
    tmp_path,
    monkeypatch,
    capsys,
):
    from hermes_cli import runtime_cli

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)
    _write_registry(
        home,
        [
            {
                "lease_id": "lease-1",
                "session_id": "session-1",
                "session_key": "session-1",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": 12345,
                "process_start_time": 1.0,
            }
        ],
    )
    lock_dir = home / "runtime" / "active_sessions.lock.d"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": 67890,
                "session_id": "metadata-session",
                "surface": "tui",
                "owner_kind": "metadata_update",
                "cwd": "C:/Users/Admin/private project",
            }
        ),
        encoding="utf-8",
    )

    rc = runtime_cli._cmd_active_sessions_diagnose(
        SimpleNamespace(json=False, no_lock=True)
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "active sessions:" in out
    assert "lock_status=degraded" in out
    assert "owner_kind=metadata_update" in out
    assert "session_id=metadata-session" in out
    assert "checked=1" in out
    assert "private project" not in out


def test_runtime_active_sessions_diagnose_registered_in_main_cli(tmp_path):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / ".hermes")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "runtime",
            "active-sessions",
            "diagnose",
            "--no-lock",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "active sessions:" in result.stdout


def test_runtime_recovery_prompt_prints_bounded_watchdog_action(tmp_path, monkeypatch, capsys):
    from hermes_cli import runtime_cli

    home = tmp_path / ".hermes"
    recovery_dir = home / "request_watchdog"
    recovery_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    packet = {
        "api_call_count": 4,
        "estimated_context_tokens": 250_000,
        "kind": "high_context_request_watchdog_recovery",
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "queued_steer_count": 2,
        "raw_log": "do not expose raw log",
        "request_id_fingerprint": "abcd1234abcd1234",
        "session_id": "session-recover",
        "status": {"status": "terminal_recovery_needed"},
    }
    (recovery_dir / "001-session-recover.json").write_text(json.dumps(packet), encoding="utf-8")

    rc = runtime_cli._cmd_recovery_prompt(
        SimpleNamespace(session="session-recover", json=False)
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "recommended_action=bounded_recovery_prompt" in out
    assert "inspect current DB/log/repo state first" in out
    assert "Do not replay broad history" in out
    assert "terminal-only write/test mode" in out
    assert "compact final/blocked answer mode" in out
    assert "Keep gpt-5.5" in out
    assert "queued steer cannot land until the active request exits or reaches a tool boundary" in out
    assert "mass-close" not in out.lower()
    assert "raw log" not in out


def test_runtime_recovery_prompt_registered_in_main_cli(tmp_path):
    home = tmp_path / ".hermes"
    recovery_dir = home / "request_watchdog"
    recovery_dir.mkdir(parents=True)
    packet = {
        "kind": "high_context_request_watchdog_recovery",
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "queued_steer_count": 1,
        "request_id_fingerprint": "abcd1234abcd1234",
        "session_id": "session-recover",
        "status": {"status": "terminal_recovery_needed"},
    }
    (recovery_dir / "001-session-recover.json").write_text(json.dumps(packet), encoding="utf-8")
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "runtime",
            "recovery",
            "prompt",
            "--session",
            "session-recover",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "bounded_recovery_prompt" in result.stdout
    assert "Keep gpt-5.5" in result.stdout


def test_goal_cli_status_pause_resume_complete_records_reasons(
    tmp_path,
    monkeypatch,
    capsys,
):
    _isolate_goal_home(tmp_path, monkeypatch)
    from hermes_cli import goal_cli
    from hermes_cli.goals import GoalManager

    GoalManager("goal-session").set("finish the thing")

    assert goal_cli._cmd_goal_status(SimpleNamespace(session_id="goal-session")) == 0
    assert "finish the thing" in capsys.readouterr().out

    assert goal_cli._cmd_goal_pause(
        SimpleNamespace(session_id="goal-session", reason="operator pause")
    ) == 0
    assert GoalManager("goal-session").state.paused_reason == "operator pause"

    assert goal_cli._cmd_goal_resume(
        SimpleNamespace(session_id="goal-session", reason="operator resume")
    ) == 0
    resumed = GoalManager("goal-session").state
    assert resumed.status == "active"
    assert resumed.last_reason == "operator resume"

    assert goal_cli._cmd_goal_complete(
        SimpleNamespace(session_id="goal-session", reason="verified manually")
    ) == 0
    completed = GoalManager("goal-session").state
    assert completed.status == "done"
    assert completed.last_reason == "verified manually"


def test_goal_cli_clear_records_audit_reason(tmp_path, monkeypatch):
    _isolate_goal_home(tmp_path, monkeypatch)
    from hermes_cli import goal_cli
    from hermes_cli.goals import GoalManager, load_goal

    GoalManager("goal-clear-session").set("clear me")

    assert goal_cli._cmd_goal_clear(
        SimpleNamespace(session_id="goal-clear-session", reason="operator clear")
    ) == 0

    stored = load_goal("goal-clear-session")
    assert stored.status == "cleared"
    assert stored.last_reason == "operator clear"


def test_goal_cli_complete_without_active_goal_exits_nonzero(tmp_path, monkeypatch, capsys):
    _isolate_goal_home(tmp_path, monkeypatch)
    from hermes_cli import goal_cli

    rc = goal_cli._cmd_goal_complete(
        SimpleNamespace(session_id="missing-goal", reason="not actually done")
    )

    assert rc == 1
    assert "No active goal" in capsys.readouterr().out


def test_goal_cli_pause_resume_refuse_done_goal(tmp_path, monkeypatch, capsys):
    _isolate_goal_home(tmp_path, monkeypatch)
    from hermes_cli import goal_cli
    from hermes_cli.goals import GoalManager

    mgr = GoalManager("done-goal-session")
    mgr.set("already done")
    mgr.mark_done("verified")

    pause_rc = goal_cli._cmd_goal_pause(
        SimpleNamespace(session_id="done-goal-session", reason="bad pause")
    )
    resume_rc = goal_cli._cmd_goal_resume(
        SimpleNamespace(session_id="done-goal-session", reason="bad resume")
    )

    assert pause_rc == 1
    assert resume_rc == 1
    assert GoalManager("done-goal-session").state.status == "done"
    out = capsys.readouterr().out
    assert "cannot pause" in out
    assert "cannot resume" in out


def test_goal_status_registered_in_main_cli(tmp_path):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / ".hermes")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "goal",
            "status",
            "--session-id",
            "missing-goal",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "No active goal" in result.stdout
