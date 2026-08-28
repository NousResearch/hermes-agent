import json
from types import SimpleNamespace
from unittest.mock import patch

from gateway.vault_reports import append_vault_context, write_terminal_report


CONFIG = {
    "kanban": {
        "vault_reports": {
            "enabled": True,
            "command": ["/fixed/python", "/fixed/vault.py"],
            "vault_path": "/fixed/vault",
            "projects": {"exampleproject-burndown": "exampleproject-v18"},
            "max_context_chars": 600,
            "stale_after_hours": 72,
            "timeout_seconds": 5,
        }
    }
}


def test_terminal_report_uses_fixed_command_json_stdin_and_no_shell():
    completed = SimpleNamespace(returncode=0, stdout='{"status":"ok"}', stderr="")
    with patch("gateway.vault_reports.subprocess.run", return_value=completed) as run:
        assert (
            write_terminal_report(
                CONFIG,
                board="exampleproject-burndown",
                root_task_id="t_root",
                event_kind="completed",
                title="Burndown",
                outcome="Audit complete",
                branch="codex/test",
                commit="abcdef1",
            )
            is True
        )
    args, kwargs = run.call_args
    assert args[0] == [
        "/fixed/python",
        "/fixed/vault.py",
        "report",
        "write",
        "--vault",
        "/fixed/vault",
    ]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 5
    assert json.loads(kwargs["input"])["workstream_id"] == "exampleproject-burndown--t_root"


def test_disabled_or_failed_bridge_is_fail_open():
    assert (
        write_terminal_report(
            {},
            board="x",
            root_task_id="t_x",
            event_kind="completed",
            title="x",
            outcome="x",
        )
        is False
    )
    with patch("gateway.vault_reports.subprocess.run", side_effect=TimeoutError):
        assert (
            write_terminal_report(
                CONFIG,
                board="exampleproject-burndown",
                root_task_id="t_x",
                event_kind="completed",
                title="x",
                outcome="x",
            )
            is False
        )


def test_progress_context_is_exact_and_labeled():
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "status": "ok",
            "context": "Vault context (narrative-only): audit complete",
        }),
        stderr="",
    )
    with patch("gateway.vault_reports.subprocess.run", return_value=completed) as run:
        response = append_vault_context(
            CONFIG,
            "Live Kanban status.",
            board="exampleproject-burndown",
            root_task_ids=["t_root"],
        )
    assert response.startswith("Live Kanban status.")
    assert "narrative-only" in response
    command = run.call_args.args[0]
    assert command[command.index("--root-task-id") + 1] == "t_root"
