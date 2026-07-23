from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli as cli_mod
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


class _FakeConn:
    def close(self):
        return None


class _FakeCli:
    def __init__(self):
        self.session_id = "worker-session"
        self.agent = SimpleNamespace(_interrupt_requested=False)
        self._last_chat_result = None

    def _print_exit_summary(self):
        return None


def test_default_spawn_remains_hermes_worker(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen = {}

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace(pid=999)

    monkeypatch.setattr("hermes_cli.kanban_db.subprocess.Popen", fake_popen)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _profile: str(tmp_path / ".hermes" / "profiles" / "claude-coder"))

    task = kb.Task(
        id="task-1",
        title="Title",
        body="Body",
        assignee="claude-coder",
        status="running",
        priority=0,
        created_by=None,
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        branch_name=None,
        current_run_id=None,
        goal_mode=False,
        goal_max_turns=None,
        skills=None,
        max_runtime_seconds=None,
    )
    pid = kb._default_spawn(task, str(workspace))
    assert pid == 999
    assert Path(seen["argv"][0]).name == "hermes"
    chat_index = seen["argv"].index("chat")
    assert seen["argv"][chat_index + 1] == "-q"
    assert "claude" not in seen["argv"]
    assert "codex" not in seen["argv"]


def test_external_cli_worker_helper_skips_internal_backend(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    handled, response = cli_mod._run_external_cli_worker_once(_FakeCli())
    assert handled is False
    assert response is None


def test_external_cli_worker_helper_completes_task_inside_formal_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "claude",
                    "authentication_mode": "cli_managed_subscription",
                    "output_mode": "structured",
                },
            }
        },
    )
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_task",
        lambda _conn, _task_id: SimpleNamespace(
            id="task-1",
            title="Ship it",
            body="Run tests",
            assignee="claude-coder",
            max_runtime_seconds=30,
        ),
    )

    completed = {}
    monkeypatch.setattr(
        "hermes_cli.kanban_db.complete_task",
        lambda _conn, task_id, result=None, summary=None, metadata=None: completed.update(
            task_id=task_id,
            result=result,
            summary=summary,
            metadata=metadata,
        )
        or True,
    )
    monkeypatch.setattr("hermes_cli.kanban_db.block_task", lambda *_a, **_k: pytest.fail("adapter path should not block here"))

    class _Adapter:
        def run(self, cfg, req):
            assert cfg.execution_backend == "external_cli"
            assert req.task_id == "task-1"
            assert req.workspace_path == str(tmp_path)
            assert "Ship it" in req.prompt
            return SimpleNamespace(
                status="COMPLETED",
                structured_payload=SimpleNamespace(
                    action="complete",
                    summary="done",
                    metadata={"changed_files": ["cli.py"]},
                    reason=None,
                    block_kind=None,
                ),
                as_metadata=lambda: {
                    "status": "COMPLETED",
                    "stdout_artifact_path": str(tmp_path / "stdout.log"),
                    "stderr_artifact_path": str(tmp_path / "stderr.log"),
                    "stdout_summary": "done",
                },
            )

    monkeypatch.setattr("hermes_cli.external_cli_adapter.ExternalCliAgentAdapter", _Adapter)
    cli = _FakeCli()
    handled, response = cli_mod._run_external_cli_worker_once(cli)
    assert handled is True
    assert response == "done"
    assert completed["task_id"] == "task-1"
    assert completed["metadata"] == {"changed_files": ["cli.py"]}
    assert cli._last_chat_result["failed"] is False
    assert "stdout_text" not in cli._last_chat_result["external_cli"]
    assert cli._last_chat_result["external_cli"]["status"] == "COMPLETED"


def test_external_cli_worker_helper_maps_quota_to_existing_exit_75(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "codex",
                    "authentication_mode": "cli_managed_subscription",
                },
            }
        },
    )
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_task",
        lambda _conn, _task_id: SimpleNamespace(
            id="task-1",
            title="Ship it",
            body="Run tests",
            assignee="codex-coder",
            max_runtime_seconds=30,
        ),
    )
    monkeypatch.setattr("hermes_cli.kanban_db.complete_task", lambda *_a, **_k: pytest.fail("quota should not complete task"))
    monkeypatch.setattr("hermes_cli.kanban_db.block_task", lambda *_a, **_k: pytest.fail("quota should not block task directly"))

    class _Adapter:
        def run(self, cfg, req):
            return SimpleNamespace(
                status="BLOCKED_QUOTA",
                structured_payload=None,
                as_metadata=lambda: {"status": "BLOCKED_QUOTA", "stderr_signature": "quota exhausted"},
            )

    monkeypatch.setattr("hermes_cli.external_cli_adapter.ExternalCliAgentAdapter", _Adapter)
    cli = _FakeCli()
    handled, response = cli_mod._run_external_cli_worker_once(cli)
    assert handled is True
    assert "quota exhaustion" in response
    assert cli._last_chat_result["failed"] is True
    assert cli._last_chat_result["failure_reason"] == "billing"
    assert cli_mod.kanban_worker_exit_code(cli._last_chat_result) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_external_cli_terminal_failure_blocks_task_instead_of_leaving_running(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "codex",
                    "authentication_mode": "cli_managed_subscription",
                },
            }
        },
    )
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_task",
        lambda _conn, _task_id: SimpleNamespace(
            id="task-1",
            title="Ship it",
            body="Run tests",
            assignee="codex-coder",
            max_runtime_seconds=30,
        ),
    )
    blocked = {}
    monkeypatch.setattr(
        "hermes_cli.kanban_db.block_task",
        lambda _conn, task_id, reason=None, kind=None: blocked.update(
            task_id=task_id, reason=reason, kind=kind
        )
        or True,
    )

    class _Adapter:
        def run(self, cfg, req):
            return SimpleNamespace(
                status="FAILED_INVALID_INVOCATION",
                structured_payload=None,
                as_metadata=lambda: {
                    "status": "FAILED_INVALID_INVOCATION",
                    "structured_output_status": "terminal_failure",
                },
            )

    monkeypatch.setattr(
        "hermes_cli.external_cli_adapter.ExternalCliAgentAdapter", _Adapter
    )
    cli = _FakeCli()
    handled, response = cli_mod._run_external_cli_worker_once(cli)

    assert handled is True
    assert response == "Error: codex rejected the worker invocation."
    assert blocked == {
        "task_id": "task-1",
        "reason": "codex rejected the worker invocation.",
        "kind": "capability",
    }
    assert cli._last_chat_result["failed"] is True


def test_external_cli_worker_helper_blocks_auth_without_provider_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "claude",
                    "authentication_mode": "cli_managed_subscription",
                },
            }
        },
    )
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_task",
        lambda _conn, _task_id: SimpleNamespace(
            id="task-1",
            title="Ship it",
            body="Run tests",
            assignee="claude-coder",
            max_runtime_seconds=30,
        ),
    )
    monkeypatch.setattr("hermes_cli.kanban_db.complete_task", lambda *_a, **_k: pytest.fail("auth block should not complete task"))
    blocked = {}
    monkeypatch.setattr(
        "hermes_cli.kanban_db.block_task",
        lambda _conn, task_id, reason=None, kind=None: blocked.update(task_id=task_id, reason=reason, kind=kind) or True,
    )

    calls = []

    class _Adapter:
        def run(self, cfg, req):
            calls.append(cfg.executable)
            return SimpleNamespace(
                status="BLOCKED_AUTH",
                structured_payload=None,
                as_metadata=lambda: {"status": "BLOCKED_AUTH"},
            )

    monkeypatch.setattr("hermes_cli.external_cli_adapter.ExternalCliAgentAdapter", _Adapter)
    cli = _FakeCli()
    handled, response = cli_mod._run_external_cli_worker_once(cli)
    assert handled is True
    assert "login session" in response
    assert blocked == {
        "task_id": "task-1",
        "reason": "claude requires a CLI-managed login session before this task can run.",
        "kind": "capability",
    }
    assert calls == ["claude"]
    assert cli._last_chat_result["failed"] is False


@pytest.mark.parametrize("status", ["BLOCKED_AUTH", "BLOCKED_PERMISSION"])
def test_external_cli_worker_block_transition_failure_is_internal_failure(monkeypatch, tmp_path, status):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "claude",
                    "authentication_mode": "cli_managed_subscription",
                },
            }
        },
    )
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_task",
        lambda _conn, _task_id: SimpleNamespace(
            id="task-1",
            title="Ship it",
            body="Run tests",
            assignee="claude-coder",
            max_runtime_seconds=30,
        ),
    )
    monkeypatch.setattr("hermes_cli.kanban_db.complete_task", lambda *_a, **_k: pytest.fail("blocked result must not complete"))
    monkeypatch.setattr("hermes_cli.kanban_db.block_task", lambda *_a, **_k: False)

    class _Adapter:
        def run(self, cfg, req):
            return SimpleNamespace(
                status=status,
                structured_payload=None,
                as_metadata=lambda: {"status": status},
            )

    monkeypatch.setattr("hermes_cli.external_cli_adapter.ExternalCliAgentAdapter", _Adapter)
    cli = _FakeCli()
    handled, response = cli_mod._run_external_cli_worker_once(cli)
    assert handled is True
    assert response.startswith("Error:")
    assert cli._last_chat_result["failed"] is True
    assert cli._last_chat_result["external_cli"]["status"] == "FAILED_INTERNAL"
    assert cli_mod.kanban_worker_exit_code(cli._last_chat_result) == 1


def test_external_cli_evidence_directory_sanitizes_task_and_is_collision_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    first = cli_mod._external_cli_worker_evidence_dir("../../unsafe task")
    second = cli_mod._external_cli_worker_evidence_dir("../../unsafe task")
    root = tmp_path / ".hermes" / "cache" / "external_cli_adapter"
    assert first.is_relative_to(root)
    assert second.is_relative_to(root)
    assert ".." not in first.parts
    assert "unsafe-task" in first.parts
    assert first != second
