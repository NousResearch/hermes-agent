"""End-to-end contract for the gateway Kanban pre-spawn command."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from hermes_cli import config as config_mod
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_spawn_supervisor as supervisor


def _write_consumer(path: Path) -> None:
    path.write_text(
        """
import json
import os
import pathlib
import sys
import time

request = json.load(sys.stdin)
record = pathlib.Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "allow"
with record.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "request": request,
        "saw_secret": "OPENAI_API_KEY" in os.environ,
    }) + "\\n")
if mode == "uncertain":
    time.sleep(60)
sys.exit(0 if mode == "allow" else 7)
""".strip(),
        encoding="utf-8",
    )


def _write_worker(path: Path) -> None:
    path.write_text(
        """
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open("a", encoding="utf-8") as handle:
    handle.write("started\\n")
""".strip(),
        encoding="utf-8",
    )


async def _run_one_gateway_tick(monkeypatch, runner: GatewayRunner) -> None:
    """Drive one dispatcher tick.

    Uses a real asyncio.to_thread for the dispatch call so spawned
    subprocesses can open a new DB connection without racing the gateway's
    own hold.  Sleep is still stubbed out so the loop terminates quickly.
    """
    calls = 0
    original_to_thread = asyncio.to_thread

    async def controlled_to_thread(fn, *args):
        nonlocal calls
        calls += 1
        result = await original_to_thread(fn, *args)
        if calls >= 3:  # reaper, dispatch, ready probe
            runner._running = False
        return result

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers._to_thread_process_service", controlled_to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    await asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10)


import os
import signal


def _wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_for_pid_exit(pid: int, timeout: float = 20.0) -> None:
    """Poll until the process exits or we timeout; POSIX only."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return  # process exists but we can't signal it — wait a moment
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} did not exit within {timeout}s")


_MALFORMED_RUN_IDS = [
    pytest.param(None, id="null"),
    pytest.param(False, id="false"),
    pytest.param(True, id="true"),
    pytest.param(1.0, id="float"),
    pytest.param("1", id="string"),
    pytest.param("01", id="noncanonical-string"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(2**63, id="sqlite-integer-overflow"),
    pytest.param([], id="list"),
    pytest.param({}, id="object"),
]


def _task_run_state(conn, task_id: str) -> tuple[tuple, tuple[tuple, ...], tuple[tuple, ...]]:
    task = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id, "
        "block_kind, block_recurrences FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    runs = conn.execute(
        "SELECT id, status, outcome, summary, error, ended_at, claim_lock, "
        "claim_expires, worker_pid FROM task_runs WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    events = conn.execute(
        "SELECT id, run_id, kind, payload FROM task_events "
        "WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return tuple(task), tuple(map(tuple, runs)), tuple(map(tuple, events))


@pytest.mark.parametrize(
    "run_fields",
    [pytest.param({}, id="missing")]
    + [pytest.param({"run_id": run_id}, id=mark.id) for mark in _MALFORMED_RUN_IDS for run_id in mark.values],
)
def test_supervisor_main_rejects_malformed_run_id_before_any_side_effect(
    tmp_path, monkeypatch, run_fields
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="malformed request", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="current-worker")
        assert claimed is not None
        before = _task_run_state(conn, task_id)

    gate_calls = []
    exec_calls = []
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: gate_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        supervisor.os,
        "execvpe",
        lambda *_args, **_kwargs: exec_calls.append((_args, _kwargs)),
    )
    request = {
        "task_id": task_id,
        "assignee": "worker",
        "board": kb.DEFAULT_BOARD,
        "workspace": str(tmp_path),
        **run_fields,
    }

    result = supervisor.main(
        [json.dumps(["gate"]), "5", json.dumps(request), json.dumps(["worker"])]
    )

    assert result == 64
    assert gate_calls == []
    assert exec_calls == []
    with kb.connect() as conn:
        assert _task_run_state(conn, task_id) == before


@pytest.mark.parametrize("run_id", _MALFORMED_RUN_IDS)
def test_block_launch_defensively_rejects_malformed_run_id_without_block_task(
    monkeypatch, run_id
):
    block_calls = []
    monkeypatch.setattr(kb, "block_task", lambda *_args, **_kwargs: block_calls.append((_args, _kwargs)))

    result = supervisor._block_launch(
        {"task_id": "t_current", "run_id": run_id, "board": kb.DEFAULT_BOARD},
        "launch denied",
    )

    assert result is False
    assert block_calls == []


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_block_launch_retries_until_current_run_is_durably_blocked(
    tmp_path, monkeypatch, failure
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="durable denial", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="supervisor")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    original_block_task = kb.block_task
    attempts = 0

    def transient_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "exception":
                raise RuntimeError("transient write failure")
            return False
        return original_block_task(*args, **kwargs)

    monkeypatch.setattr(kb, "block_task", transient_failure)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _delay: None)

    assert supervisor._block_launch(
        {"task_id": task_id, "run_id": run_id, "board": kb.DEFAULT_BOARD},
        "launch denied",
    )
    assert attempts == 2
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"


@pytest.mark.parametrize(
    "timeout",
    ["nan", "inf", float("nan"), float("inf"), 10**10000],
    ids=["nan-string", "inf-string", "nan-float", "inf-float", "overflow"],
)
def test_configured_pre_spawn_fn_rejects_invalid_timeout(timeout):
    with pytest.raises(
        ValueError,
        match="kanban.pre_spawn_timeout_seconds must be a positive finite number",
    ):
        kbd.configured_pre_spawn_fn(
            {
                "pre_spawn_command": ["gate"],
                "pre_spawn_timeout_seconds": timeout,
            }
        )


def test_gateway_pre_spawn_command_allows_exactly_one_worker_without_secret_or_task_content(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "top-secret")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    gate_record = tmp_path / "gate.json"
    worker_record = tmp_path / "worker.txt"
    gate = tmp_path / "gate.py"
    worker = tmp_path / "worker.py"
    _write_consumer(gate)
    _write_worker(worker)

    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "pre_spawn_command": [sys.executable, str(gate), str(gate_record)],
            "pre_spawn_timeout_seconds": 2,
        }
    }
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(kb, "list_boards", lambda **_kw: [{"slug": kb.DEFAULT_BOARD}])
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: [sys.executable, str(worker), str(worker_record)])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="private title",
            body="private body",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    asyncio.run(_run_one_gateway_tick(monkeypatch, runner))

    _wait_for(worker_record)
    gate_payload = json.loads(gate_record.read_text(encoding="utf-8").splitlines()[0])
    assert worker_record.read_text(encoding="utf-8").splitlines() == ["started"]
    assert gate_payload["saw_secret"] is False
    assert gate_payload["request"] == {
        "task_id": task_id,
        "run_id": 1,
        "assignee": "worker",
        "board": "default",
        "workspace": str(workspace),
    }


def test_gateway_pre_spawn_command_failure_blocks_without_starting_or_retrying(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    gate_record = tmp_path / "gate.jsonl"
    worker_record = tmp_path / "worker.txt"
    gate = tmp_path / "gate.py"
    worker = tmp_path / "worker.py"
    _write_consumer(gate)
    _write_worker(worker)
    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "pre_spawn_command": [
                sys.executable, str(gate), str(gate_record), "deny",
            ],
            "pre_spawn_timeout_seconds": 2,
        }
    }
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(kb, "list_boards", lambda **_kw: [{"slug": kb.DEFAULT_BOARD}])
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(
        kbd,
        "_resolve_hermes_argv",
        lambda: [sys.executable, str(worker), str(worker_record)],
    )
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="must be gated",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    asyncio.run(_run_one_gateway_tick(monkeypatch, runner))
    _wait_for(gate_record)

    # Poll for the supervisor to call block_task. The supervisor exits quickly
    # but stays as a zombie until its parent (the test process) reaps it —
    # polling the PID with os.kill would spin forever on the zombie.
    # Poll the task status directly instead.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with kb.connect() as conn:
            if kb.get_task(conn, task_id).status == "blocked":
                break
        time.sleep(0.02)
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "blocked"

    runner._running = True
    asyncio.run(_run_one_gateway_tick(monkeypatch, runner))
    assert not worker_record.exists()
    assert len(gate_record.read_text(encoding="utf-8").splitlines()) == 1


def test_delayed_pre_spawn_rejection_cannot_block_successor_run(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="successor must survive stale gate",
            assignee="worker",
        )
        first = kb.claim_task(conn, task_id, claimer="stale-supervisor")
        assert first is not None
        stale_run_id = first.current_run_id
        assert stale_run_id is not None

    gate_started = threading.Event()
    release_gate = threading.Event()

    def delayed_rejection(*_args, **_kwargs):
        gate_started.set()
        assert release_gate.wait(timeout=5)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(supervisor.subprocess, "run", delayed_rejection)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            supervisor.supervise(
                ["gate"],
                5,
                {
                    "task_id": task_id,
                    "run_id": stale_run_id,
                    "assignee": "worker",
                    "board": kb.DEFAULT_BOARD,
                    "workspace": str(tmp_path),
                },
                ["worker"],
            )
        )
    )
    thread.start()
    assert gate_started.wait(timeout=5)
    with kb.connect() as conn:
        assert kb.reclaim_task(conn, task_id, reason="test rollover")
        successor = kb.claim_task(conn, task_id, claimer="successor")
        assert successor is not None
        assert successor.current_run_id != stale_run_id

    release_gate.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == [78]

    with kb.connect() as conn:
        live = kb.get_task(conn, task_id)
        assert live is not None
        assert live.status == "running"
        assert live.current_run_id == successor.current_run_id


def test_delayed_pre_spawn_approval_cannot_launch_for_superseded_run(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="successor must survive stale approval",
            assignee="worker",
        )
        first = kb.claim_task(conn, task_id, claimer="stale-supervisor")
        assert first is not None
        stale_run_id = first.current_run_id
        assert stale_run_id is not None
        kbd._set_worker_pid(conn, task_id, os.getpid())

    gate_started = threading.Event()
    release_gate = threading.Event()

    def delayed_approval(*_args, **_kwargs):
        gate_started.set()
        assert release_gate.wait(timeout=5)
        return SimpleNamespace(returncode=0)

    exec_calls = []
    monkeypatch.setattr(supervisor.subprocess, "run", delayed_approval)
    monkeypatch.setattr(
        supervisor.os,
        "execvpe",
        lambda *_args, **_kwargs: exec_calls.append((_args, _kwargs)),
    )
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            supervisor.supervise(
                ["gate"],
                5,
                {
                    "task_id": task_id,
                    "run_id": stale_run_id,
                    "assignee": "worker",
                    "board": kb.DEFAULT_BOARD,
                    "workspace": str(tmp_path),
                },
                ["worker"],
            )
        )
    )
    thread.start()
    assert gate_started.wait(timeout=5)
    with kb.connect() as conn:
        assert kb.reclaim_task(conn, task_id, reason="test rollover")
        successor = kb.claim_task(conn, task_id, claimer="successor")
        assert successor is not None
        assert successor.current_run_id != stale_run_id

    release_gate.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == [78]
    assert exec_calls == []

    with kb.connect() as conn:
        live = kb.get_task(conn, task_id)
        assert live is not None
        assert live.status == "running"
        assert live.current_run_id == successor.current_run_id


def test_ownership_handshake_waits_for_dispatcher_to_persist_supervisor_pid(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="pid handshake", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="supervisor")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    def persist_pid():
        time.sleep(0.05)
        with kb.connect() as conn:
            kbd._set_worker_pid(conn, task_id, os.getpid())

    writer = threading.Thread(target=persist_pid)
    writer.start()
    assert supervisor._still_owns_run(
        {"task_id": task_id, "run_id": run_id, "board": kb.DEFAULT_BOARD}
    )
    writer.join(timeout=5)
    assert not writer.is_alive()


def test_missing_supervisor_pid_after_approval_durably_blocks_current_run(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="missing supervisor pid", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="supervisor")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(supervisor, "_OWNERSHIP_HANDSHAKE_SECONDS", 0)
    exec_calls = []
    monkeypatch.setattr(
        supervisor.os,
        "execvpe",
        lambda *_args, **_kwargs: exec_calls.append((_args, _kwargs)),
    )

    result = supervisor.supervise(
        ["gate"],
        5,
        {
            "task_id": task_id,
            "run_id": run_id,
            "assignee": "worker",
            "board": kb.DEFAULT_BOARD,
            "workspace": str(tmp_path),
        },
        ["worker"],
    )

    assert result == 78
    assert exec_calls == []
    with kb.connect() as conn:
        live = kb.get_task(conn, task_id)
        assert live is not None
        assert live.status == "blocked"
        assert live.current_run_id is None
