import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hermes_cli import active_sessions


def test_resolve_max_concurrent_sessions_values(caplog):
    assert active_sessions.resolve_max_concurrent_sessions({}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": None}) is None
    caplog.set_level(logging.WARNING)
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": {}}) is None
    assert (
        active_sessions.resolve_max_concurrent_sessions(
            {"gateway": {"max_concurrent_sessions": {}}}
        )
        is None
    )
    assert caplog.records == []
    caplog.clear()
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": 0}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": -1}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": "3"}) == 3
    assert (
        active_sessions.resolve_max_concurrent_sessions(
            {"gateway": {"max_concurrent_sessions": 4}}
        )
        == 4
    )
    assert (
        active_sessions.resolve_max_concurrent_sessions(
            {"max_concurrent_sessions": 2, "gateway": {"max_concurrent_sessions": 4}}
        )
        == 2
    )

    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": "many"}) is None
    assert any(
        "Ignoring invalid max_concurrent_sessions='many'" in record.message
        for record in caplog.records
    )


def test_active_session_lease_blocks_until_release(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = {"max_concurrent_sessions": 1}

    lease, message = active_sessions.try_acquire_active_session(
        session_id="session-1",
        surface="cli",
        config=cfg,
    )

    assert message is None
    assert lease is not None

    blocked_lease, blocked_message = active_sessions.try_acquire_active_session(
        session_id="session-2",
        surface="tui",
        config=cfg,
    )

    assert blocked_lease is None
    assert blocked_message == (
        "Hermes is at the active session limit (1/1). "
        "Try again when another session finishes."
    )

    lease.release()

    next_lease, next_message = active_sessions.try_acquire_active_session(
        session_id="session-3",
        surface="gateway:telegram",
        config=cfg,
    )

    assert next_message is None
    assert next_lease is not None
    next_lease.release()
    assert active_sessions.active_session_registry_snapshot() == []


def test_active_session_tracks_lease_even_when_limit_disabled(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    lease, message = active_sessions.try_acquire_active_session(
        session_id="tracked-without-cap",
        surface="cli",
        config={},
    )

    assert message is None
    assert lease is not None
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "tracked-without-cap"
    ]

    lease.release()
    assert active_sessions.active_session_registry_snapshot() == []


def test_release_active_sessions_for_current_process_skips_other_pid(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)

    current_start = active_sessions._process_start_time(os.getpid())
    now = time.time()
    active_sessions._write_entries(
        active_sessions._state_path(),
        [
            {
                "lease_id": "ours",
                "session_id": "worker-session",
                "surface": "kanban-worker",
                "pid": os.getpid(),
                "process_start_time": current_start,
                "updated_at": now,
            },
            {
                "lease_id": "other-pid",
                "session_id": "worker-session",
                "surface": "kanban-worker",
                "pid": os.getpid() + 100_000,
                "process_start_time": current_start,
                "updated_at": now,
            },
            {
                "lease_id": "other-session",
                "session_id": "other-session",
                "surface": "kanban-worker",
                "pid": os.getpid(),
                "process_start_time": current_start,
                "updated_at": now,
            },
        ],
    )

    released = active_sessions.release_active_sessions_for_current_process(
        session_id="worker-session",
        surface="kanban-worker",
    )

    assert released == 1
    assert [
        entry["lease_id"]
        for entry in active_sessions.active_session_registry_snapshot()
    ] == ["other-pid", "other-session"]


def test_active_session_lease_metadata_has_value_free_owner_summary(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    workspace = tmp_path / "secret workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(workspace)

    lease, message = active_sessions.try_acquire_active_session(
        session_id="agent:main:telegram:dm:secret-chat",
        surface="gateway:telegram",
        config={},
        metadata={
            "chat_id": "secret-chat",
            "user_id": "private-user",
            "current_tool": "terminal",
        },
    )

    assert message is None
    assert lease is not None

    [entry] = active_sessions.active_session_registry_snapshot()
    assert entry["session_id"] == "agent:main:telegram:dm:secret-chat"
    assert entry["session_key"] == "agent:main:telegram:dm:secret-chat"
    assert entry["owner_kind"] == "gateway"
    assert entry["created_at"] == entry["started_at"]
    assert entry["updated_at"] == entry["started_at"]
    assert isinstance(entry["cwd_fingerprint"], str)
    assert isinstance(entry["command_line_fingerprint"], str)
    assert len(entry["cwd_fingerprint"]) == 16
    assert len(entry["command_line_fingerprint"]) == 16
    assert entry["owner_summary"] == {
        "pid": os.getpid(),
        "session_id_fingerprint": entry["owner_summary"]["session_id_fingerprint"],
        "session_key_fingerprint": entry["owner_summary"]["session_key_fingerprint"],
        "surface": "gateway:telegram",
        "owner_kind": "gateway",
        "cwd_fingerprint": entry["cwd_fingerprint"],
        "command_line_fingerprint": entry["command_line_fingerprint"],
    }
    assert len(entry["owner_summary"]["session_id_fingerprint"]) == 16
    assert len(entry["owner_summary"]["session_key_fingerprint"]) == 16
    assert "secret workspace" not in str(entry["owner_summary"])
    assert "secret-chat" not in str(entry["owner_summary"])
    assert "private-user" not in str(entry["owner_summary"])

    lease.release()


def test_update_active_session_metadata_keeps_runtime_status_value_free(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    lease, message = active_sessions.try_acquire_active_session(
        session_id="runtime-status",
        surface="cli",
        config={},
    )

    assert message is None
    assert lease is not None

    updated = active_sessions.update_active_session_metadata(
        session_id="runtime-status",
        metadata={
            "current_tool": "terminal",
            "activity_kind": "tool_running",
            "last_activity_ts": 123.0,
            "unsafe_prompt": "do not persist this prompt text",
            "current_tool_args": "do not persist command args",
        },
    )

    assert updated == 1
    [entry] = active_sessions.active_session_registry_snapshot()
    assert entry["session_id"] == "runtime-status"
    assert entry["metadata"] == {
        "activity_kind": "tool_running",
        "current_tool": "terminal",
        "last_activity_ts": 123.0,
    }
    assert "unsafe_prompt" not in str(entry)
    assert "current_tool_args" not in str(entry)

    active_sessions.update_active_session_metadata(
        session_id="runtime-status",
        metadata={"current_tool": None, "activity_kind": "generic"},
    )

    [entry] = active_sessions.active_session_registry_snapshot()
    assert entry["metadata"]["activity_kind"] == "generic"
    assert "current_tool" not in entry["metadata"]

    lease.release()
    assert active_sessions.active_session_registry_snapshot() == []


def test_update_active_session_metadata_keeps_model_policy_status_value_free(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    lease, message = active_sessions.try_acquire_active_session(
        session_id="policy-status",
        surface="cli",
        config={},
    )

    assert message is None
    assert lease is not None

    updated = active_sessions.update_active_session_metadata(
        session_id="policy-status",
        metadata={
            "model_policy_violation": True,
            "required_model": "gpt-5.5",
            "model_policy_recommended_action": "interrupt_and_restore_fixed_model",
            "provider_payload": "do not persist provider payload",
            "raw_prompt": "do not persist prompt text",
        },
    )

    assert updated == 1
    [entry] = active_sessions.active_session_registry_snapshot()
    assert entry["metadata"] == {
        "model_policy_violation": True,
        "required_model": "gpt-5.5",
        "model_policy_recommended_action": "interrupt_and_restore_fixed_model",
    }
    assert "provider_payload" not in str(entry)
    assert "prompt text" not in str(entry)

    lease.release()


def test_active_session_registry_prunes_dead_pids(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "gateway.status._pid_exists",
        lambda pid: int(pid) != 99999999,
    )
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "stale",
                "session_id": "stale-session",
                "surface": "cli",
                "pid": 99999999,
                "started_at": 1,
                "updated_at": 1,
            }
        ],
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="session-1",
        surface="cli",
        config={"max_concurrent_sessions": 1},
    )

    assert message is None
    assert lease is not None
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "session-1"
    ]
    lease.release()

def test_transfer_active_session_reanchors_existing_lease(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    lease, message = active_sessions.try_acquire_active_session(
        session_id="session-old",
        surface="tui",
        config={"max_concurrent_sessions": 1},
        metadata={"live_session_id": "ui-1"},
    )

    assert message is None
    assert lease is not None
    [before] = active_sessions.active_session_registry_snapshot()
    previous_owner_summary = before["owner_summary"]

    assert active_sessions.transfer_active_session(
        lease,
        session_id="session-new",
        metadata={"live_session_id": "ui-1"},
    )

    snapshot = active_sessions.active_session_registry_snapshot()
    assert lease.session_id == "session-new"
    assert len(snapshot) == 1
    assert snapshot[0]["session_id"] == "session-new"
    assert snapshot[0]["session_key"] == "session-new"
    assert snapshot[0]["metadata"] == {"live_session_id": "ui-1"}
    assert snapshot[0]["previous_owner_summary"] == previous_owner_summary
    assert snapshot[0]["owner_summary"]["session_id"] == "session-new"
    assert snapshot[0]["owner_summary"]["session_key"] == "session-new"
    lease.release()


def test_prune_dead_active_session_leases_finalizes_db_sessions(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "stale",
                "session_id": "stale-session",
                "surface": "cli",
                "pid": 99999999,
                "started_at": 1,
                "updated_at": 1,
            },
            {
                "lease_id": "live",
                "session_id": "live-session",
                "surface": "cli",
                "pid": 12345,
                "started_at": 2,
                "updated_at": 2,
            },
        ],
    )
    monkeypatch.setattr(
        active_sessions,
        "_pid_alive",
        lambda pid, process_start_time=None: int(pid) == 12345,
    )
    calls = []

    class FakeDB:
        def end_session(self, session_id, reason):
            calls.append((session_id, reason))

    assert active_sessions.prune_dead_active_session_leases(session_db=FakeDB()) == 1
    assert calls == [("stale-session", "stale_active_session")]
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "live-session"
    ]


def test_prune_dead_active_session_leases_skips_blank_session_ids(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "blank",
                "session_id": "  ",
                "surface": "cli",
                "pid": 99999999,
                "started_at": 1,
                "updated_at": 1,
            },
        ],
    )
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: False)
    calls = []

    class FakeDB:
        def end_session(self, session_id, reason):
            calls.append((session_id, reason))

    assert active_sessions.prune_dead_active_session_leases(session_db=FakeDB()) == 0
    assert calls == []
    assert active_sessions.active_session_registry_snapshot() == []


def test_prune_dead_active_session_leases_keeps_db_open_when_live_lease_remains(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "dead",
                "session_id": "same-session",
                "surface": "cli",
                "pid": 111,
                "started_at": 1,
                "updated_at": 1,
            },
            {
                "lease_id": "live",
                "session_id": "same-session",
                "surface": "cli",
                "pid": 222,
                "started_at": 2,
                "updated_at": 2,
            },
        ],
    )
    monkeypatch.setattr(
        active_sessions,
        "_pid_alive",
        lambda pid, process_start_time=None: int(pid) == 222,
    )
    calls = []

    class FakeDB:
        def end_session(self, session_id, reason):
            calls.append((session_id, reason))

    assert active_sessions.prune_dead_active_session_leases(session_db=FakeDB()) == 0
    assert calls == []
    assert [entry["lease_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "live"
    ]


def test_prune_dead_active_session_leases_skips_runtime_activity_and_queued_steer(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "fresh",
                "session_id": "fresh-session",
                "surface": "cli",
                "pid": 111,
                "started_at": 1,
                "updated_at": 1,
                "metadata": {"last_activity_age_seconds": 1},
            },
            {
                "lease_id": "steer",
                "session_id": "steer-session",
                "surface": "cli",
                "pid": 222,
                "started_at": 2,
                "updated_at": 2,
                "metadata": {"pending_steer_count": 1},
            },
        ],
    )
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: False)
    calls = []

    class FakeDB:
        def end_session(self, session_id, reason):
            calls.append((session_id, reason))

    assert active_sessions.prune_dead_active_session_leases(session_db=FakeDB()) == 0
    assert calls == []
    assert active_sessions.active_session_registry_snapshot() == []


def test_prune_dead_active_session_leases_only_uses_registry_evidence(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "dead",
                "session_id": "registry-session",
                "surface": "cli",
                "pid": 99999999,
                "started_at": 1,
                "updated_at": 1,
            },
        ],
    )
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: False)
    calls = []

    class FakeDB:
        def end_session(self, session_id, reason):
            calls.append((session_id, reason))

        def list_sessions_rich(self):
            raise AssertionError("DB-wide session scan must not run")

        def maybe_auto_prune_and_vacuum(self, **_kwargs):
            raise AssertionError("DB prune must not run")

    assert active_sessions.prune_dead_active_session_leases(session_db=FakeDB()) == 1
    assert calls == [("registry-session", "stale_active_session")]


def test_pid_alive_uses_safe_pid_exists_without_signalling(monkeypatch):
    checked: list[int] = []

    monkeypatch.setattr(
        active_sessions.os,
        "kill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("os.kill used")),
    )
    monkeypatch.setattr(
        "gateway.status._pid_exists",
        lambda pid: checked.append(int(pid)) or True,
    )

    assert active_sessions._pid_alive(12345) is True
    assert checked == [12345]


def test_active_session_hard_exit_is_reclaimed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo_root)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "from hermes_cli.active_sessions import try_acquire_active_session\n"
                "lease, message = try_acquire_active_session("
                "session_id='crash-session', surface='cli', "
                "config={'max_concurrent_sessions': 1})\n"
                "assert message is None, message\n"
                "print(os.getpid(), flush=True)\n"
                "os._exit(0)\n"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    child_pid = int(child.stdout.strip())

    lease, message = active_sessions.try_acquire_active_session(
        session_id="next-session",
        surface="cli",
        config={"max_concurrent_sessions": 1},
    )

    assert child_pid > 0
    assert message is None
    assert lease is not None
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "next-session"
    ]
    lease.release()


def test_concurrent_acquire_claims_only_one_last_slot(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = {"max_concurrent_sessions": 1}

    def _claim(index: int):
        return active_sessions.try_acquire_active_session(
            session_id=f"session-{index}",
            surface="cli",
            config=cfg,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_claim, range(8)))

    leases = [lease for lease, message in results if lease is not None and message is None]
    blocked = [message for lease, message in results if lease is None and message]

    try:
        assert len(leases) == 1
        assert len(blocked) == 7
        assert active_sessions.active_session_registry_snapshot()[0]["session_id"].startswith("session-")
    finally:
        for lease in leases:
            lease.release()


def test_cross_process_acquire_claims_only_one_last_slot(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo_root = Path(__file__).resolve().parents[2]
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    go_file = tmp_path / "go"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo_root)
    script = (
        "import os, time\n"
        "from pathlib import Path\n"
        "from hermes_cli.active_sessions import try_acquire_active_session\n"
        "idx = os.environ['WORKER_INDEX']\n"
        "worker_count = int(os.environ['WORKER_COUNT'])\n"
        "delayed_worker = os.environ.get('DELAYED_WORKER_INDEX')\n"
        "ready_dir = Path(os.environ['READY_DIR'])\n"
        "results_dir = Path(os.environ['RESULTS_DIR'])\n"
        "go_file = Path(os.environ['GO_FILE'])\n"
        "(ready_dir / idx).write_text('ready', encoding='utf-8')\n"
        "deadline = time.time() + 10\n"
        "while not go_file.exists():\n"
        "    if time.time() > deadline:\n"
        "        raise RuntimeError('timed out waiting for go file')\n"
        "    time.sleep(0.01)\n"
        "if idx == delayed_worker:\n"
        "    time.sleep(2.5)\n"
        "lease, message = try_acquire_active_session(\n"
        "    session_id=f'process-{idx}',\n"
        "    surface='cli',\n"
        "    config={'max_concurrent_sessions': 1},\n"
        ")\n"
        "if lease is None:\n"
        "    (results_dir / idx).write_text('BLOCK', encoding='utf-8')\n"
        "    print('BLOCK', flush=True)\n"
        "else:\n"
        "    (results_dir / idx).write_text('OK', encoding='utf-8')\n"
        "    print('OK', flush=True)\n"
        "    deadline = time.time() + 10\n"
        "    while len(list(results_dir.iterdir())) < worker_count:\n"
        "        if time.time() > deadline:\n"
        "            raise RuntimeError('timed out waiting for all workers to attempt acquire')\n"
        "        time.sleep(0.01)\n"
        "    lease.release()\n"
    )
    workers: list[subprocess.Popen[str]] = []
    try:
        for index in range(6):
            worker_env = env.copy()
            worker_env["WORKER_INDEX"] = str(index)
            worker_env["WORKER_COUNT"] = "6"
            worker_env["DELAYED_WORKER_INDEX"] = "5"
            worker_env["READY_DIR"] = str(ready_dir)
            worker_env["RESULTS_DIR"] = str(results_dir)
            worker_env["GO_FILE"] = str(go_file)
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    env=worker_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        deadline = time.time() + 10
        while len(list(ready_dir.iterdir())) < len(workers):
            if time.time() > deadline:
                raise AssertionError("workers did not become ready")
            time.sleep(0.01)
        go_file.write_text("go", encoding="utf-8")

        outputs = []
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=10)
            assert worker.returncode == 0, stderr
            outputs.append(stdout.strip())
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()

    assert outputs.count("OK") == 1
    assert outputs.count("BLOCK") == len(workers) - 1
    assert active_sessions.active_session_registry_snapshot() == []


def test_pid_start_time_mismatch_prunes_reused_pid(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr(active_sessions, "_process_start_time", lambda _pid: 200.0)
    runtime = home / "runtime"
    runtime.mkdir(parents=True)
    active_sessions._write_entries(
        runtime / "active_sessions.json",
        [
            {
                "lease_id": "stale-reused-pid",
                "session_id": "stale-session",
                "surface": "cli",
                "pid": os.getpid(),
                "process_start_time": 100.0,
                "started_at": 1,
                "updated_at": 1,
            }
        ],
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="new-session",
        surface="cli",
        config={"max_concurrent_sessions": 1},
    )

    assert message is None
    assert lease is not None
    assert [entry["session_id"] for entry in active_sessions.active_session_registry_snapshot()] == [
        "new-session"
    ]
    lease.release()


def test_windows_directory_lock_reclaims_stale_owner_and_writes_metadata(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions.os, "name", "nt", raising=False)
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: False)

    lock_path = active_sessions._lock_path()
    lock_dir = lock_path.with_name(f"{lock_path.name}.d")
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": 99999999,
                "process_start_time": 1.0,
                "session_id": "stale-session",
                "created_at": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with active_sessions._FileLock(
        lock_path,
        owner_metadata={
            "session_id": "new-session",
            "surface": "cli",
            "owner_kind": "test",
        },
    ):
        owner = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert owner["session_id"] == "new-session"
        assert owner["surface"] == "cli"
        assert owner["owner_kind"] == "test"
        assert isinstance(owner["created_at"], float)
        assert "process_start_time" in owner

    assert not lock_dir.exists()


def test_windows_directory_lock_release_retries_transient_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions.os, "name", "nt", raising=False)

    lock_path = active_sessions._lock_path()
    lock_dir = lock_path.with_name(f"{lock_path.name}.d")
    original_rmdir = os.rmdir
    attempts = {"remaining": 3}

    def flaky_rmdir(path, *args, **kwargs):
        if Path(path) == lock_dir and attempts["remaining"] > 0:
            attempts["remaining"] -= 1
            raise PermissionError("simulated transient Windows cleanup busy")
        return original_rmdir(path, *args, **kwargs)

    sleeps = []
    monkeypatch.setattr(active_sessions.time, "sleep", lambda seconds: sleeps.append(seconds))

    with active_sessions._FileLock(
        lock_path,
        owner_metadata={"owner_kind": "test"},
    ):
        assert lock_dir.exists()
        monkeypatch.setattr(active_sessions.os, "rmdir", flaky_rmdir)

    assert attempts["remaining"] == 0
    assert sleeps == [active_sessions._WINDOWS_LOCK_RELEASE_RETRY_INTERVAL_SECONDS] * 3
    assert not lock_dir.exists()


def test_windows_directory_lock_reclaims_only_stale_ownerless_lock_dir(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions.os, "name", "nt", raising=False)

    lock_path = active_sessions._lock_path()
    lock_dir = lock_path.with_name(f"{lock_path.name}.d")
    lock_dir.mkdir(parents=True)
    lock = active_sessions._FileLock(lock_path)

    assert lock._try_reclaim_stale_lock_dir(lock_dir) is False
    assert lock_dir.exists()

    old_time = (
        time.time()
        - active_sessions._WINDOWS_OWNERLESS_LOCK_RECLAIM_GRACE_SECONDS
        - 1.0
    )
    os.utime(lock_dir, (old_time, old_time))

    assert lock._try_reclaim_stale_lock_dir(lock_dir) is True
    assert not lock_dir.exists()


def test_live_windows_directory_lock_owner_returns_safe_busy_message(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(active_sessions.os, "name", "nt", raising=False)
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)
    monkeypatch.setattr(active_sessions.time, "sleep", lambda _seconds: None)

    time_values = iter([1.0, 100.0, 131.0])
    monkeypatch.setattr(
        active_sessions.time,
        "time",
        lambda: next(time_values, 131.0),
    )

    lock_path = active_sessions._lock_path()
    lock_dir = lock_path.with_name(f"{lock_path.name}.d")
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": 12345,
                "process_start_time": 456.0,
                "session_id": "live-session",
                "surface": "tui",
                "owner_kind": "try_acquire",
                "cwd": "C:/Users/Admin/secret folder",
                "command_line_fingerprint": "cmd:fingerprint",
            }
        ),
        encoding="utf-8",
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="new-session",
        surface="cli",
        config={"max_concurrent_sessions": 5},
    )

    assert lease is None
    assert message == (
        "Hermes active session registry is busy; owner "
        "pid=12345 session_id=live-session surface=tui owner_kind=try_acquire "
        "command_line_fingerprint=cmd:fingerprint. "
        "Try again shortly or run `hermes runtime active-sessions status`."
    )
    assert not active_sessions._state_path().exists()
    assert "secret folder" not in message


def test_repeated_active_session_lock_busy_logs_one_deduped_alert(
    tmp_path,
    monkeypatch,
    caplog,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        active_sessions,
        "_LOCK_BUSY_ALERT_STATE",
        {"window_start": 0.0, "count": 0, "alerted": False},
        raising=False,
    )

    owner = {
        "pid": 12345,
        "session_id": "live-session",
        "surface": "tui",
        "owner_kind": "try_acquire",
        "cwd": "C:/Users/Admin/secret folder",
    }

    def raise_busy(_self):
        raise active_sessions.ActiveSessionLockBusyError(owner)

    monkeypatch.setattr(active_sessions._FileLock, "__enter__", raise_busy)
    caplog.set_level(logging.WARNING, logger="hermes_cli.active_sessions")

    messages = []
    for _attempt in range(4):
        lease, message = active_sessions.try_acquire_active_session(
            session_id="new-session",
            surface="cli",
            config={"max_concurrent_sessions": 5},
        )
        assert lease is None
        messages.append(message)

    assert all("live-session" in message for message in messages)
    alerts = [
        record.message
        for record in caplog.records
        if "Repeated active session registry lock timeouts" in record.message
    ]
    assert len(alerts) == 1
    assert "count=3" in alerts[0]
    assert "live-session" in alerts[0]
    assert "secret folder" not in alerts[0]


def test_active_session_registry_status_degrades_when_metadata_update_lock_is_busy(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    active_sessions._write_entries(
        active_sessions._state_path(),
        [
            {
                "lease_id": "live",
                "session_id": "live-session",
                "session_key": "live-session",
                "surface": "tui",
                "owner_kind": "tui",
                "pid": 12345,
                "process_start_time": 456.0,
                "metadata": {"current_tool": "terminal"},
            }
        ],
    )
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)

    owner = {
        "pid": 67890,
        "session_id": "metadata-session",
        "surface": "tui",
        "owner_kind": "metadata_update",
        "cwd": "C:/Users/Admin/private project",
        "command_line_fingerprint": "abcdef1234567890",
    }

    def raise_busy(_self):
        raise active_sessions.ActiveSessionLockBusyError(owner)

    monkeypatch.setattr(active_sessions._FileLock, "__enter__", raise_busy)

    report = active_sessions.active_session_registry_status()

    assert report["checked"] == 1
    assert report["live"] == 1
    assert report["stale"] == 0
    assert report["lock_status"] == "degraded"
    assert report["lock_owner_summary"]["owner_kind"] == "metadata_update"
    assert report["lock_owner_summary"]["session_id"] == "metadata-session"
    assert report["entries"][0]["session_id"] == "live-session"
    assert report["entries"][0]["runtime_status"] == "live"
    assert "private project" not in json.dumps(report)


def test_update_active_session_metadata_fails_open_when_registry_lock_is_busy(
    tmp_path,
    monkeypatch,
    caplog,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    active_sessions._write_entries(
        active_sessions._state_path(),
        [
            {
                "lease_id": "ours",
                "session_id": "metadata-target",
                "session_key": "metadata-target",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": os.getpid(),
                "process_start_time": active_sessions._process_start_time(os.getpid()),
            }
        ],
    )

    owner = {
        "pid": 67890,
        "session_id": "metadata-target",
        "owner_kind": "metadata_update",
    }

    def raise_busy(_self):
        raise active_sessions.ActiveSessionLockBusyError(owner)

    monkeypatch.setattr(active_sessions._FileLock, "__enter__", raise_busy)
    caplog.set_level(logging.WARNING, logger="hermes_cli.active_sessions")

    assert active_sessions.update_active_session_metadata(
        session_id="metadata-target",
        metadata={"current_tool": "terminal"},
    ) == 0
    assert active_sessions._read_entries(active_sessions._state_path())[0].get("metadata") is None
    assert any(
        "Skipped active session metadata update because registry lock is busy" in record.message
        for record in caplog.records
    )


def test_update_active_session_metadata_dedupes_lock_busy_warning(
    tmp_path,
    monkeypatch,
    caplog,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        active_sessions,
        "_METADATA_UPDATE_BUSY_LOG_STATE",
        {"window_start": 0.0, "messages": {}},
        raising=False,
    )
    active_sessions._write_entries(
        active_sessions._state_path(),
        [
            {
                "lease_id": "ours",
                "session_id": "metadata-target",
                "session_key": "metadata-target",
                "surface": "cli",
                "owner_kind": "cli",
                "pid": os.getpid(),
                "process_start_time": active_sessions._process_start_time(os.getpid()),
            }
        ],
    )

    owner = {
        "pid": 67890,
        "session_id": "metadata-target",
        "owner_kind": "metadata_update",
    }

    def raise_busy(_self):
        raise active_sessions.ActiveSessionLockBusyError(owner)

    monkeypatch.setattr(active_sessions._FileLock, "__enter__", raise_busy)
    caplog.set_level(logging.WARNING, logger="hermes_cli.active_sessions")

    for _attempt in range(5):
        assert active_sessions.update_active_session_metadata(
            session_id="metadata-target",
            metadata={"current_tool": "terminal"},
        ) == 0

    warnings = [
        record.message
        for record in caplog.records
        if "Skipped active session metadata update because registry lock is busy" in record.message
    ]
    assert len(warnings) == 1
