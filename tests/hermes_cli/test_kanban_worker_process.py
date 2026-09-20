"""Lifecycle tests for Kanban worker process ownership and tree cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_worker_process as worker_process
from hermes_cli.kanban_worker_process import (
    cleanup_worker_tree,
    get_worker_receipt,
    spawn_worker_process,
)


_CHILD_CODE = r'''
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
role = sys.argv[2]
(root / f"{role}.pid").write_text(str(os.getpid()), encoding="ascii")
if role == "root":
    subprocess.Popen([sys.executable, "-c", __import__("base64").b64decode(sys.argv[3]).decode(), str(root), "python-child", sys.argv[3]], close_fds=True)
elif role == "python-child":
    subprocess.Popen([sys.executable, "-c", __import__("base64").b64decode(sys.argv[3]).decode(), str(root), "node-like-child", sys.argv[3]], close_fds=True)
time.sleep(60)
'''

_EXITING_CODE = r'''
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "root.pid").write_text(str(os.getpid()), encoding="ascii")
subprocess.Popen([
    sys.executable, "-c", __import__("base64").b64decode(sys.argv[2]).decode(),
    str(root),
], close_fds=True)
os._exit(0)
'''

_SURVIVING_CHILD_CODE = r'''
import os
import time
from pathlib import Path

root = Path(__import__("sys").argv[1])
(root / "surviving-child.pid").write_text(str(os.getpid()), encoding="ascii")
time.sleep(60)
'''


def _wait_pid_gone(pid: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            proc = psutil.Process(pid)
            if proc.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def _wait_record_closed(pid: int, start_time: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker_process._record_for_pid(pid, start_time=start_time) is None:
            return True
        time.sleep(0.05)
    return worker_process._record_for_pid(pid, start_time=start_time) is None


def _spawn_descendant_tree(root: Path, envelope_path: Path | None = None):
    import base64

    encoded = base64.b64encode(_CHILD_CODE.encode()).decode()
    return spawn_worker_process(
        [sys.executable, "-c", _CHILD_CODE, str(root), "root", encoded],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        envelope_path=str(envelope_path) if envelope_path else None,
    )


def _spawn_exiting_tree(root: Path):
    import base64

    encoded = base64.b64encode(_SURVIVING_CHILD_CODE.encode()).decode()
    return spawn_worker_process(
        [sys.executable, "-c", _EXITING_CODE, str(root), encoded],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        monitor=False,
    )


def test_timeout_reaps_nested_tree_and_preserves_unrelated_sibling(tmp_path):
    """A timeout owns and reaps nested descendants, not just the wrapper PID."""
    receipt_root = tmp_path / "owned"
    receipt_root.mkdir()
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    worker = _spawn_descendant_tree(receipt_root)
    try:
        assert worker.pid > 0
        assert get_worker_receipt(worker.pid)["start_time"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all((receipt_root / f"{role}.pid").exists() for role in ("root", "python-child", "node-like-child")):
                break
            time.sleep(0.05)
        pids = [int((receipt_root / f"{role}.pid").read_text()) for role in ("root", "python-child", "node-like-child")]
        result = cleanup_worker_tree(worker.pid, reason="timeout")
        assert result["identity_verified"] is True
        assert result["forced_cleanup"] is True
        assert result["tree_cleanup"] is True
        assert result["cleanup_reason"] == "timeout"
        assert all(_wait_pid_gone(pid) for pid in pids)
        assert unrelated.poll() is None, "cleanup killed an unrelated sibling"
    finally:
        if unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)


def test_cleanup_is_idempotent_and_pid_reuse_guard_never_signals_replacement(tmp_path, monkeypatch):
    """Repeated cleanup is harmless and a mismatched start time is never killed."""
    root = tmp_path / "one"
    root.mkdir()
    worker = spawn_worker_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        receipt = get_worker_receipt(worker.pid)
        mismatch = dict(receipt)
        mismatch["start_time"] = int(receipt["start_time"]) + 1
        seen = []
        monkeypatch.setattr(
            "hermes_cli.kanban_worker_process._send_graceful_signal",
            lambda pid, **kwargs: seen.append(pid),
        )
        result = cleanup_worker_tree(
            worker.pid,
            start_time=mismatch["start_time"],
            owner_kind=receipt["owner_kind"],
            owner_id=receipt["owner_id"],
            reason="pid-reuse",
        )
        assert result["identity_verified"] is False
        assert result["pid_reused"] is True
        assert seen == []
        assert worker.poll() is None
        result = cleanup_worker_tree(worker.pid, reason="cancellation")
        assert result["tree_cleanup"] is True
        again = cleanup_worker_tree(worker.pid, reason="cancellation")
        assert again["already_absent"] is True
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)


def test_pid_reuse_with_surviving_owned_tree_is_unsafe_to_reclaim(monkeypatch):
    """A recycled root PID must not hide a still-live owned tree."""
    monkeypatch.setattr(
        worker_process,
        "_identity_state",
        lambda pid, start_time: (True, False),
    )
    monkeypatch.setattr(
        worker_process,
        "_tree_alive",
        lambda record, owner_kind, owner_id, pid, **kwargs: True,
    )
    monkeypatch.setattr(
        worker_process,
        "_send_graceful_signal",
        lambda *args, **kwargs: pytest.fail("recycled PID must never be signalled"),
    )

    result = cleanup_worker_tree(
        123,
        start_time=456,
        owner_kind="process_group",
        owner_id="123",
        reason="pid-reuse",
    )

    assert result["identity_verified"] is False
    assert result["pid_reused"] is True
    assert result["survived_cleanup"] is True
    assert result["unsafe_to_reclaim"] is True
    assert result["tree_cleanup"] is False


def test_cleanup_without_creation_identity_never_signals_owned_tree(monkeypatch):
    """An owner handle without a root fingerprint must fail closed."""
    monkeypatch.setattr(
        worker_process,
        "_send_graceful_signal",
        lambda *args, **kwargs: pytest.fail("unfingerprinted tree must not be signalled"),
    )
    monkeypatch.setattr(
        worker_process,
        "_send_forced_signal",
        lambda *args, **kwargs: pytest.fail("unfingerprinted tree must not be signalled"),
    )

    result = cleanup_worker_tree(
        123,
        owner_kind="process_group",
        owner_id="123",
        reason="missing-fingerprint",
    )

    assert result["identity_verified"] is False
    assert result["tree_cleanup"] is False
    assert result["survived_cleanup"] is True
    assert result["unsafe_to_reclaim"] is True


def test_cleanup_does_not_bind_recycled_pid_to_new_inprocess_record(monkeypatch):
    """Persisted old ownership must not use a newer record with the same PID."""
    new_record = type(
        "Record",
        (),
        {
            "receipt": type(
                "Receipt",
                (),
                {
                    "start_time": 999,
                    "owner_kind": "process_group",
                    "owner_id": "new-group",
                    "envelope_path": None,
                },
            )(),
            "owner": None,
        },
    )()
    observed_records = []
    monkeypatch.setitem(worker_process._RECORDS, 123, new_record)
    monkeypatch.setattr(
        worker_process,
        "_identity_state",
        lambda pid, start_time: (True, False),
    )
    monkeypatch.setattr(
        worker_process,
        "_tree_alive",
        lambda record, owner_kind, owner_id, pid, **kwargs: (
            observed_records.append(record) or False
        ),
    )

    result = cleanup_worker_tree(
        123,
        start_time=456,
        owner_kind="process_group",
        owner_id="old-group",
        reason="pid-reuse",
    )

    assert observed_records == [None]
    assert result["pid_reused"] is True
    assert result["tree_cleanup"] is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_unavailable_windows_job_handle_fails_closed(monkeypatch):
    """An unopenable persisted Job Object cannot prove descendants are gone."""
    monkeypatch.setattr(
        worker_process._WindowsJob,
        "open",
        classmethod(lambda cls, name: None),
    )
    monkeypatch.setattr(worker_process, "_pid_alive", lambda pid: False)

    assert worker_process._tree_alive(
        None,
        "job_object",
        "Local\\HermesKanbanWorker_missing",
        123,
    ) is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_unavailable_windows_job_handle_fails_closed_for_live_tree(tmp_path, monkeypatch):
    """An unavailable persisted Job Object cannot override a live root identity."""
    worker = spawn_worker_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        monitor=False,
    )
    receipt = worker.receipt.as_dict()
    monkeypatch.setattr(
        worker_process._WindowsJob,
        "open",
        classmethod(lambda cls, name: None),
    )
    try:
        assert worker_process._tree_alive(
            None,
            receipt["owner_kind"],
            receipt["owner_id"],
            receipt["pid"],
            start_time=receipt["start_time"],
        ) is True
    finally:
        if worker.poll() is None:
            result = cleanup_worker_tree(
                worker.pid,
                start_time=receipt["start_time"],
                owner_kind=receipt["owner_kind"],
                owner_id=receipt["owner_id"],
            )
            assert result["tree_cleanup"] is True
            worker.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_unavailable_windows_job_handle_fails_closed_for_ambiguous_identity(
    monkeypatch,
):
    """An identity probe failure is ambiguous, not proof that the tree is gone."""
    monkeypatch.setattr(
        worker_process._WindowsJob,
        "open",
        classmethod(lambda cls, name: None),
    )
    monkeypatch.setattr(
        worker_process.psutil,
        "Process",
        lambda pid: (_ for _ in ()).throw(worker_process.psutil.AccessDenied(pid)),
    )

    assert worker_process._tree_alive(
        None,
        "job_object",
        "Local\\HermesKanbanWorker_ambiguous",
        123,
        start_time=456,
    ) is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_persisted_fast_exit_proves_absent_after_job_handle_close(tmp_path, monkeypatch):
    """A closed Job Object plus an absent exact root is safe to retry."""
    envelope = tmp_path / "fast-exit.json"
    worker = spawn_worker_process(
        [sys.executable, "-c", "import os; os._exit(0)"],
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        envelope_path=str(envelope),
    )
    receipt = worker.receipt.as_dict()
    try:
        assert worker.wait(timeout=5) == 0
        assert _wait_record_closed(worker.pid, receipt["start_time"])
        assert envelope.exists()
        envelope.unlink()
        monkeypatch.setattr(
            worker_process._WindowsJob,
            "open",
            classmethod(lambda cls, name: None),
        )

        result = cleanup_worker_tree(
            receipt["pid"],
            start_time=receipt["start_time"],
            owner_kind=receipt["owner_kind"],
            owner_id=receipt["owner_id"],
            reason="persisted-retry",
        )
        assert result["identity_verified"] is True
        assert result["already_absent"] is True
        assert result["tree_cleanup"] is True
        assert result["survived_cleanup"] is False
        again = cleanup_worker_tree(
            receipt["pid"],
            start_time=receipt["start_time"],
            owner_kind=receipt["owner_kind"],
            owner_id=receipt["owner_id"],
            reason="persisted-retry-again",
        )
        assert again["already_absent"] is True
        assert again["tree_cleanup"] is True
        assert again["survived_cleanup"] is False
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_no_descendant_crash_terminalizes_after_persisted_job_close(tmp_path, monkeypatch):
    """A root-only crash is reclaimed after its monitor closes the Job Object."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    envelope = tmp_path / "crash.json"
    worker = spawn_worker_process(
        [sys.executable, "-c", "import os; os._exit(7)"],
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        envelope_path=str(envelope),
    )
    try:
        task_id = kb.create_task(conn, title="root-only crash", assignee="default")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        assert worker.wait(timeout=5) == 7
        assert _wait_record_closed(worker.pid, worker.receipt.start_time)
        monkeypatch.setattr(
            worker_process._WindowsJob,
            "open",
            classmethod(lambda cls, name: None),
        )

        assert kb.detect_crashed_workers(conn) == [task_id]
        row = conn.execute(
            "SELECT status, worker_pid, worker_start_time, worker_owner_id "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["worker_pid"] is None
        assert row["worker_start_time"] is None
        assert row["worker_owner_id"] is None
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == "worker_exit_unknown"
        payload = json.loads(event["payload"])
        assert payload["failure_class"] == "worker_exit_unknown"
        assert payload["tree_cleanup"] is True
        assert payload["already_absent"] is True
        assert payload["survived_cleanup"] is False
        assert payload["forced_action"] is None
    finally:
        conn.close()
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_reclaim_waits_for_recycled_pid_tree_to_disappear(tmp_path, monkeypatch):
    """Retry admission stays held while the old owned tree is unverified."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="recycled root", assignee="default")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_start_time=?, "
            "worker_owner_kind=?, worker_owner_id=? WHERE id=?",
            (123, 456, "process_group", "123", task_id),
        )
        conn.commit()
        monkeypatch.setattr(
            kb,
            "cleanup_worker_tree",
            lambda *args, **kwargs: {
                "identity_verified": False,
                "pid_reused": True,
                "graceful_cleanup": False,
                "forced_cleanup": False,
                "tree_cleanup": False,
                "already_absent": False,
                "survived_cleanup": True,
                "graceful_action": None,
                "forced_action": None,
            },
        )

        assert kb.reclaim_task(conn, task_id, reason="recycled root") is False
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] == claimed.claim_lock
        assert row["worker_pid"] == 123
    finally:
        conn.close()


def test_windows_terminate_process_fallback_is_not_graceful(monkeypatch):
    """A hard Windows terminate fallback must not be labelled graceful."""
    called = []

    class _Process:
        def terminate(self):
            called.append(True)

    class _Record:
        process = _Process()
        receipt = type("Receipt", (), {"owner_kind": "job_object"})()

    monkeypatch.setattr(worker_process.signal, "CTRL_BREAK_EVENT", None, raising=False)
    result = worker_process._send_graceful_signal(123, record=_Record())

    assert result == "unavailable"
    assert called == []


def test_normal_completion_writes_truthful_exit_envelope(tmp_path):
    """Normal exit records rc=0 without claiming forced cleanup."""
    envelope = tmp_path / "exit.json"
    worker = spawn_worker_process(
        [sys.executable, "-c", "print('done', flush=True)"],
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        envelope_path=str(envelope),
    )
    assert worker.wait(timeout=5) == 0
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not envelope.exists():
        time.sleep(0.05)
    payload = json.loads(envelope.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0
    assert payload["forced_cleanup"] is False
    assert payload["tree_cleanup"] is True
    assert payload["pid"] == worker.pid


def test_monitor_does_not_overwrite_authoritative_wrapper_envelope(tmp_path):
    """The parent monitor preserves the wrapper's typed terminal receipt."""
    envelope = tmp_path / "authoritative-exit.json"
    child = (
        "import json, os; "
        "json.dump({'schema_version': 1, 'task_id': 't_race', 'run_id': 7, "
        "'failure_class': 'success', 'exit_code': 0}, "
        "open(os.environ['HERMES_KANBAN_EXIT_ENVELOPE'], 'w', encoding='utf-8'))"
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_KANBAN_EXIT_ENVELOPE": str(envelope),
            "HERMES_KANBAN_TASK_ID": "t_race",
            "HERMES_KANBAN_RUN_ID": "7",
        }
    )
    worker = spawn_worker_process(
        [sys.executable, "-c", child],
        cwd=str(tmp_path),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        envelope_path=str(envelope),
    )
    assert worker.wait(timeout=5) == 0
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            payload = json.loads(envelope.read_text(encoding="utf-8"))
            if payload.get("failure_class") == "success":
                break
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    payload = json.loads(envelope.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["task_id"] == "t_race"
    assert payload["failure_class"] == "success"
    assert "cleanup_reason" not in payload


def test_max_runtime_reclaims_owned_tree_and_persists_identity_evidence(tmp_path, monkeypatch):
    """The dispatcher timeout path kills descendants and records ownership evidence."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    tree_root = tmp_path / "timeout-tree"
    tree_root.mkdir()
    envelope = tree_root / "worker.exit.json"
    worker = _spawn_descendant_tree(tree_root, envelope)
    try:
        task_id = kb.create_task(conn, title="owned timeout", assignee="default")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        conn.execute(
            "UPDATE tasks SET started_at=?, max_runtime_seconds=? WHERE id=?",
            (int(time.time()) - 120, 1, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at=? WHERE id=?",
            (int(time.time()) - 120, claimed.current_run_id),
        )
        conn.commit()
        assert conn.execute(
            "SELECT worker_start_time, worker_owner_kind, worker_owner_id "
            "FROM tasks WHERE id=?", (task_id,)
        ).fetchone()["worker_owner_kind"]
        assert conn.execute(
            "SELECT worker_exit_envelope FROM tasks WHERE id=?", (task_id,)
        ).fetchone()["worker_exit_envelope"] == str(envelope)

        result = kb.enforce_max_runtime(conn)
        assert result == [task_id]
        row = conn.execute(
            "SELECT status, worker_pid FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        assert row["status"] == "ready"
        assert row["worker_pid"] is None
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1", (task_id,)
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload["forced_cleanup"] is True
        assert payload["tree_cleanup"] is True
        assert payload["identity_verified"] is True
        pids = [
            int((tree_root / f"{role}.pid").read_text())
            for role in ("root", "python-child", "node-like-child")
        ]
        assert all(_wait_pid_gone(pid) for pid in pids)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_manual_cancellation_reaps_owned_tree_and_is_not_a_duplicate_spawn(tmp_path, monkeypatch):
    """The explicit cancellation entrypoint uses the same ownership proof."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    root = tmp_path / "cancel-tree"
    root.mkdir()
    worker = _spawn_descendant_tree(root)
    try:
        task_id = kb.create_task(conn, title="owned cancellation", assignee="default")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        assert kb.reclaim_task(conn, task_id, reason="user cancellation") is True
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (task_id,)
        ).fetchone()["status"] == "ready"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1", (task_id,)
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload["forced_cleanup"] is True
        assert payload["tree_cleanup"] is True
        assert all(
            _wait_pid_gone(int((root / f"{role}.pid").read_text()))
            for role in ("root", "python-child", "node-like-child")
        )
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_crash_reclaims_descendant_after_wrapper_exits(tmp_path, monkeypatch):
    """Crash recovery reaps a child that outlives an exited wrapper."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    root = tmp_path / "crash-tree"
    root.mkdir()
    worker = _spawn_exiting_tree(root)
    task_id = None
    try:
        task_id = kb.create_task(conn, title="wrapper crash", assignee="default")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        assert worker.wait(timeout=5) == 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (root / "surviving-child.pid").exists():
            time.sleep(0.05)
        assert (root / "surviving-child.pid").exists()
        crashed = kb.detect_crashed_workers(conn)
        assert crashed == [task_id]
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1", (task_id,)
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload["forced_cleanup"] is True
        assert payload["tree_cleanup"] is True
        assert _wait_pid_gone(int((root / "surviving-child.pid").read_text()))
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "win32", reason="covers Windows job-object lifetime")
def test_detached_worker_survives_one_shot_dispatcher_exit(tmp_path):
    """A one-shot CLI dispatcher must not kill its worker on interpreter exit."""
    marker = tmp_path / "worker-survived.marker"
    child_code = (
        "import pathlib,sys,time; "
        "time.sleep(0.5); pathlib.Path(sys.argv[1]).write_text('ok', encoding='ascii')"
    )
    helper_code = r'''
import subprocess
import sys
from hermes_cli.kanban_worker_process import spawn_worker_process

marker, child_code = sys.argv[1:]
spawn_worker_process(
    [sys.executable, "-c", child_code, marker],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    monitor=False,
    kill_on_parent_exit=False,
)
'''
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", helper_code, str(marker), child_code],
        cwd=str(tmp_path),
        env=env,
        check=True,
        timeout=20,
    )
    assert result.returncode == 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text(encoding="ascii") == "ok"
