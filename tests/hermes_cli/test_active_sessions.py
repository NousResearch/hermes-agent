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

    caplog.set_level(logging.WARNING)
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
        "HT AI Agent is at the active session limit (1/1). "
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
    assert active_sessions.transfer_active_session(
        lease,
        session_id="session-new",
        metadata={"live_session_id": "ui-1"},
    )

    snapshot = active_sessions.active_session_registry_snapshot()
    assert lease.session_id == "session-new"
    assert len(snapshot) == 1
    assert snapshot[0]["session_id"] == "session-new"
    assert snapshot[0]["metadata"] == {"live_session_id": "ui-1"}
    lease.release()


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
