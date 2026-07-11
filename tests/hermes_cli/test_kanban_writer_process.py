from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_writer as writer


pytestmark = pytest.mark.real_kanban_writer


def _named_board_path(tmp_path: Path) -> Path:
    # Keep the Unix-domain socket path below Linux's 108-byte sun_path limit.
    return tmp_path / "kanban.db"


def _init_board(path: Path) -> None:
    with writer.privileged_maintenance(path):
        kb.init_db(path)


def test_named_board_path_wins_over_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "boards" / "alpha" / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")
    assert writer._infer_board(db_path) == "alpha"


def test_unactivated_board_preserves_official_bootstrap_and_direct_mutation(
    tmp_path: Path,
) -> None:
    db_path = _named_board_path(tmp_path)
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn, title="pre-activation", created_by="default"
        )
        assert kb.get_task(conn, task_id) is not None


def test_activation_marker_is_durable_and_independent_from_auth_token(
    tmp_path: Path,
) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    token_path = writer.writer_token_path(db_path)
    activation_path = writer.writer_activation_path(db_path)

    writer._write_private_token(token_path, "temporary-auth-token")
    assert not writer.is_writer_required(db_path)

    writer.activate_writer_requirement(db_path)
    assert activation_path.is_file()
    assert writer.is_writer_required(db_path)

    token_path.unlink()
    assert writer.is_writer_required(db_path)
    with kb.connect(db_path) as conn:
        with pytest.raises(writer.WriterUnavailableError):
            kb.create_task(
                conn, title="must not fall back", created_by="default"
            )


def test_request_envelope_preserves_board_actor_source_and_writer_pid(
    tmp_path: Path,
) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    service = writer.KanbanWriterService(db_path, board="tour-platform")
    service.start()
    try:
        client = writer.KanbanWriterClient(
            db_path,
            board="tour-platform",
            actor_profile="ava",
            source="gateway",
        )
        task_id = client.mutate(
            "create_task",
            {"title": "attributed", "created_by": "ava"},
            request_id="attributed-create",
        )
        with kb.connect(db_path) as conn:
            task = kb.get_task(conn, task_id)
            request = conn.execute(
                "SELECT board, actor_profile, source, writer_pid "
                "FROM kanban_writer_requests WHERE request_id = ?",
                ("attributed-create",),
            ).fetchone()
        assert task is not None and task.created_by == "ava"
        assert tuple(request) == ("tour-platform", "ava", "gateway", os.getpid())
    finally:
        service.stop()


def test_seven_profiles_mutate_through_one_writer_pid(tmp_path: Path) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    service = writer.KanbanWriterService(db_path, board="tour-platform")
    service.start()
    profiles = [
        "default",
        "ava",
        "tp-planner",
        "tp-builder-api",
        "tp-builder-ui",
        "tp-builder-fix",
        "tp-reviewer",
    ]
    try:
        def create(profile: str) -> str:
            return writer.KanbanWriterClient(
                db_path,
                board="tour-platform",
                actor_profile=profile,
                source="worker",
            ).mutate(
                "create_task",
                {"title": f"from-{profile}", "created_by": profile},
                request_id=f"profile-{profile}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as pool:
            task_ids = list(pool.map(create, profiles))

        assert len(set(task_ids)) == len(profiles)
        with kb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT actor_profile, writer_pid FROM kanban_writer_requests "
                "WHERE request_id LIKE 'profile-%' ORDER BY actor_profile"
            ).fetchall()
        assert {row["actor_profile"] for row in rows} == set(profiles)
        assert {row["writer_pid"] for row in rows} == {os.getpid()}
    finally:
        service.stop()


def test_mutation_source_context_is_scoped_to_connection(tmp_path: Path) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)

    with writer.mutation_source("gateway"):
        with kb.connect(db_path) as conn:
            assert conn._kanban_source == "gateway"

    with kb.connect(db_path) as conn:
        assert conn._kanban_source == "worker"


def test_connection_attribution_reaches_writer_ledger(tmp_path: Path) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    service = writer.KanbanWriterService(db_path, board="tour-platform")
    service.start()
    try:
        with kb.connect(
            db_path,
            board="tour-platform",
            source="dashboard",
            actor_profile="rita",
        ) as conn:
            task_id = kb.create_task(conn, title="attributed", created_by="rita")
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT actor_profile, source FROM kanban_writer_requests "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        assert task_id
        assert row == ("rita", "dashboard")
    finally:
        service.stop()


def test_stop_drains_inflight_mutation_and_checkpoints_wal(tmp_path: Path) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    service = writer.KanbanWriterService(db_path, board="tour-platform")
    service.start()
    client = writer.KanbanWriterClient(
        db_path,
        board="tour-platform",
        actor_profile="ava",
        source="gateway",
        timeout_seconds=3,
    )

    service._mutation_lock.acquire()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            mutation = pool.submit(
                client.mutate,
                "create_task",
                {"title": "drain me", "created_by": "ava"},
                request_id="drain-me",
            )
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and service._peer_slots._value == 64:
                time.sleep(0.01)
            assert service._peer_slots._value < 64, "request never reached writer"

            stopping = pool.submit(service.stop)
            time.sleep(0.1)
            assert not stopping.done(), "stop returned before the in-flight mutation drained"
            service._mutation_lock.release()

            task_id = mutation.result(timeout=3)
            stopping.result(timeout=3)
    finally:
        if service._mutation_lock.locked():
            service._mutation_lock.release()
        service.stop()

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        assert kb.get_task(conn, task_id) is not None
    wal_path = Path(str(db_path) + "-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0


def test_stop_timeout_defers_cleanup_until_last_peer_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    service = writer.KanbanWriterService(db_path, board="tour-platform")
    service.start()
    monkeypatch.setattr(writer, "DEFAULT_TIMEOUT_SECONDS", 0.05)
    entered = threading.Event()
    release = threading.Event()
    original_handle = service._handle

    def blocked_handle(request):
        entered.set()
        assert release.wait(2)
        return original_handle(request)

    monkeypatch.setattr(service, "_handle", blocked_handle)
    client = writer.KanbanWriterClient(
        db_path,
        board="tour-platform",
        actor_profile="ava",
        source="gateway",
        timeout_seconds=3,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        mutation = pool.submit(
            client.mutate,
            "create_task",
            {"title": "finish after stop timeout", "created_by": "ava"},
            request_id="deferred-stop",
        )
        assert entered.wait(1)
        with pytest.raises(writer.WriterUnavailableError, match="timed out draining"):
            service.stop()
        assert service._lock_handle is not None
        assert service.socket_path.exists()
        release.set()
        assert mutation.result(timeout=3)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if service._lock_handle is None and not service.socket_path.exists():
            break
        time.sleep(0.01)
    assert service._lock_handle is None
    assert not service.socket_path.exists()

    replacement = writer.KanbanWriterService(db_path, board="tour-platform")
    replacement.start()
    replacement.stop()


def test_module_entrypoint_runs_standalone_writer_process(tmp_path: Path) -> None:
    db_path = _named_board_path(tmp_path)
    _init_board(db_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_cli.kanban_writer",
            "serve",
            "--db",
            str(db_path),
            "--board",
            "tour-platform",
        ],
        cwd=Path(__file__).parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        socket_path = writer.writer_socket_path(db_path)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not socket_path.exists():
            if process.poll() is not None:
                break
            time.sleep(0.02)
        assert process.poll() is None, process.stderr.read()
        assert socket_path.exists()

        task_id = writer.KanbanWriterClient(
            db_path,
            board="tour-platform",
            actor_profile="default",
            source="cli",
        ).mutate(
            "create_task",
            {"title": "standalone", "created_by": "default"},
            request_id="standalone-create",
        )
        with kb.connect(db_path) as conn:
            request = conn.execute(
                "SELECT writer_pid FROM kanban_writer_requests WHERE request_id = ?",
                ("standalone-create",),
            ).fetchone()
        assert request["writer_pid"] == process.pid
        assert task_id
    finally:
        if process.poll() is None:
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
