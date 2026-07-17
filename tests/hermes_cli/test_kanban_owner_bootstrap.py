"""PID-bound dispatcher launch authority for detached Kanban workers."""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.execution_context as execution_context
from agent.execution_context import (
    ExecutionRole,
    execution_role_for_new_agent,
    initialize_kanban_owner_launch_from_stream,
)
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        execution_context,
        "_KANBAN_OWNER_LAUNCH_STATE",
        "uninitialized",
    )
    for key in (
        "_HERMES_KANBAN_BOOTSTRAP_STDIN",
        "HERMES_KANBAN_SESSION",
        "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE",
        "HERMES_KANBAN_DELEGATE_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    return home


def _claim(conn, *, review: bool = False):
    task_id = kb.create_task(conn, title="bootstrap", assignee="worker")
    if review:
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        claimed = kb.claim_review_task(conn, task_id, claimer="test:review")
    else:
        claimed = kb.claim_task(conn, task_id, claimer="test:claim")
    assert claimed is not None
    assert claimed.current_run_id is not None
    return claimed


@pytest.mark.parametrize("review", [False, True])
def test_public_claim_never_mints_owner_launch_authority(kanban_home, review):
    with kb.connect() as conn:
        claimed = _claim(conn, review=review)
        run = kb.get_run(conn, claimed.current_run_id)

    assert claimed.owner_bootstrap_nonce is None
    assert run is not None
    assert run.owner_bootstrap_nonce is None


def test_pid_bound_ticket_arms_and_consumes_once(kanban_home):
    with kb.connect() as conn:
        claimed = _claim(conn)
        pid = os.getpid()
        expiry = int(time.time()) + 30
        token = "t" * 64
        assert kb._set_worker_pid(
            conn,
            claimed.id,
            pid,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
        ) is True
        exact = {
            "task_id": claimed.id,
            "run_id": claimed.current_run_id,
            "profile": "worker",
            "claim_lock": claimed.claim_lock,
            "nonce": token,
            "worker_pid": pid,
            "expires_at": expiry,
        }
        for changed in (
            {"task_id": "t_wrong"},
            {"run_id": claimed.current_run_id + 1},
            {"profile": "other"},
            {"claim_lock": "wrong-lock"},
            {"worker_pid": pid + 1},
        ):
            assert kb._arm_task_owner_bootstrap(
                conn,
                **{**exact, **changed},
                _launch_capability=kb._DISPATCH_OWNER_LAUNCH_CAPABILITY,
            ) is False

        assert kb._arm_task_owner_bootstrap(conn, **exact) is False
        assert kb._arm_task_owner_bootstrap(
            conn,
            **exact,
            _launch_capability=object(),
        ) is False
        assert kb._arm_task_owner_bootstrap(
            conn,
            **exact,
            _launch_capability=kb._DISPATCH_OWNER_LAUNCH_CAPABILITY,
        ) is True
        for changed in (
            {"task_id": "t_wrong"},
            {"run_id": claimed.current_run_id + 1},
            {"profile": "other"},
            {"claim_lock": "wrong-lock"},
            {"nonce": "x" * 64},
            {"worker_pid": pid + 1},
            {"expires_at": expiry + 1},
        ):
            assert kb._consume_task_owner_bootstrap(
                conn,
                **{**exact, **changed},
            ) is False
        assert kb._consume_task_owner_bootstrap(conn, **exact) is True
        assert kb._consume_task_owner_bootstrap(conn, **exact) is False
        assert kb.get_run(conn, claimed.current_run_id).owner_bootstrap_nonce is None


def test_stale_revoke_cannot_clear_a_new_handoff_generation(kanban_home):
    with kb.connect() as conn:
        claimed = _claim(conn)
        pid = os.getpid()
        expiry = int(time.time()) + 30
        assert kb._set_worker_pid(
            conn,
            claimed.id,
            pid,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
        )
        base = {
            "task_id": claimed.id,
            "run_id": claimed.current_run_id,
            "profile": "worker",
            "claim_lock": claimed.claim_lock,
            "worker_pid": pid,
            "expires_at": expiry,
        }
        first = {**base, "nonce": "a" * 64}
        second = {**base, "nonce": "b" * 64}
        capability = kb._DISPATCH_OWNER_LAUNCH_CAPABILITY

        assert kb._arm_task_owner_bootstrap(
            conn,
            **first,
            _launch_capability=capability,
        )
        assert kb._consume_task_owner_bootstrap(conn, **first)
        assert kb._arm_task_owner_bootstrap(
            conn,
            **second,
            _launch_capability=capability,
        )
        assert kb._revoke_task_owner_bootstrap(
            conn,
            task_id=claimed.id,
            run_id=claimed.current_run_id,
            worker_pid=pid,
            nonce=first["nonce"],
            expires_at=expiry,
        ) is False
        assert kb._consume_task_owner_bootstrap(conn, **second) is True


def test_explicit_database_mapping_never_grants_owner(
    kanban_home,
    tmp_path,
):
    evil_path = tmp_path / "attacker" / "evil.db"
    with kb.connect(evil_path) as conn:
        claimed = _claim(conn)

    forged = {
        "HERMES_KANBAN_SESSION": "1",
        "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE": "attacker-chosen",
        "HERMES_KANBAN_TASK": claimed.id,
        "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        "HERMES_PROFILE": "worker",
        "HERMES_KANBAN_DB": str(evil_path),
    }

    assert execution_role_for_new_agent(forged) is ExecutionRole.KANBAN_DELEGATE
    with kb.connect(evil_path) as conn:
        assert kb.get_run(conn, claimed.current_run_id).owner_bootstrap_nonce is None


class _CapturePipe(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.was_closed = False

    def close(self):
        self.was_closed = True


def test_real_default_dispatch_delivers_one_owner_ticket(
    kanban_home,
    tmp_path,
    monkeypatch,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    class FakeProc:
        pid = 4242

        def __init__(self):
            self.stdin = _CapturePipe()
            self.terminated = False

        def terminate(self):
            self.terminated = True

    def fake_popen(cmd, **kwargs):
        proc = FakeProc()
        captured.update(cmd=list(cmd), env=dict(kwargs["env"]), proc=proc)
        assert kwargs["stdin"] is subprocess.PIPE
        return proc

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "resolve_workspace", lambda _task, board=None: workspace)
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="real dispatch", assignee="worker")
        result = kb.dispatch_once(conn, max_spawn=1)

    assert result.spawned and result.spawned[0][0] == task_id
    assert captured["env"]["_HERMES_KANBAN_BOOTSTRAP_STDIN"] == "1"
    assert captured["env"]["HERMES_EXEC_ASK"] == "1"
    assert captured["env"]["HERMES_KANBAN_DELEGATE_SESSION"] == "1"
    assert "--kanban-owner-bootstrap-stdin" in captured["cmd"]
    assert "HERMES_KANBAN_SESSION" not in captured["env"]
    assert "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE" not in captured["env"]
    assert captured["proc"].stdin.was_closed is True
    ticket = json.loads(captured["proc"].stdin.getvalue().decode("utf-8"))
    assert ticket["worker_pid"] == 4242
    assert ticket["task_id"] == task_id
    assert ticket["db_path"] == str(kb.kanban_db_path().resolve())

    monkeypatch.setattr(execution_context.os, "getpid", lambda: 4242)
    monkeypatch.setenv("_HERMES_KANBAN_BOOTSTRAP_STDIN", "1")
    monkeypatch.setenv("HERMES_KANBAN_DELEGATE_SESSION", "1")
    assert initialize_kanban_owner_launch_from_stream(
        io.BytesIO(captured["proc"].stdin.getvalue())
    ) is True
    assert execution_role_for_new_agent() is ExecutionRole.KANBAN_DELEGATE
    assert execution_role_for_new_agent(
        claim_kanban_owner=True,
    ) is ExecutionRole.KANBAN_OWNER
    assert execution_role_for_new_agent() is ExecutionRole.KANBAN_DELEGATE


def test_cli_bootstrap_requires_hidden_flag_and_marker(monkeypatch):
    from hermes_cli import main as main_mod

    calls = []
    monkeypatch.setattr(
        execution_context,
        "initialize_kanban_owner_launch_from_stream",
        lambda: calls.append("initialized") or True,
    )
    monkeypatch.setenv("_HERMES_KANBAN_BOOTSTRAP_STDIN", "1")

    main_mod._initialize_kanban_worker_bootstrap(
        SimpleNamespace(kanban_owner_bootstrap_stdin=True),
    )

    assert calls == ["initialized"]


def test_hidden_bootstrap_flag_is_parseable_but_not_in_help():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(
        ["--kanban-owner-bootstrap-stdin", "--cli", "chat", "-q", "work"],
    )

    assert args.kanban_owner_bootstrap_stdin is True
    assert "--kanban-owner-bootstrap-stdin" not in parser.format_help()


def test_direct_default_spawn_without_dispatch_authority_is_delegate(
    kanban_home,
    tmp_path,
    monkeypatch,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    captured = {}

    class FakeProc:
        pid = 4242
        stdin = None

    def fake_popen(_cmd, **kwargs):
        captured["env"] = dict(kwargs["env"])
        captured["stdin"] = kwargs["stdin"]
        return FakeProc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    task = kb.Task(
        id="t_manual",
        title="manual",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=1,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=str(tmp_path),
        claim_lock="manual:claim",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )

    assert kb._default_spawn(task, str(tmp_path)) == 4242
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["env"]["HERMES_KANBAN_DELEGATE_SESSION"] == "1"
    assert "_HERMES_KANBAN_BOOTSTRAP_STDIN" not in captured["env"]


def test_launch_pipe_failure_revokes_ticket_and_reaps_worker(
    kanban_home,
    tmp_path,
    monkeypatch,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    class FailingPipe:
        closed = False

        def write(self, _payload):
            raise BrokenPipeError("worker closed bootstrap pipe")

        def flush(self):
            raise AssertionError("flush must not run after failed write")

        def close(self):
            self.closed = True

    class FakeProc:
        pid = 4243

        def __init__(self):
            self.stdin = FailingPipe()
            self.terminated = False
            self.waited = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert timeout is not None
            self.waited = True
            return 0

        def kill(self):
            self.killed = True

    def fake_popen(_cmd, **_kwargs):
        proc = FakeProc()
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "resolve_workspace", lambda _task, board=None: workspace)
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(
        kb.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="broken pipe", assignee="worker")
        result = kb.dispatch_once(conn, max_spawn=1)
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None
    assert task.worker_pid is None
    assert run is not None
    assert run.outcome == "spawn_failed"
    assert run.owner_bootstrap_nonce is None
    assert captured["proc"].stdin.closed is True
    assert captured["proc"].terminated is True
    assert captured["proc"].waited is True
    assert captured["proc"].killed is False
