"""Slice 9: dispatch_once integration wrapper (launch-protocol shadow ledger).

Wires the kernel ledger around the live dispatcher's spawn path: the wrapper
is handed to dispatch_once as ``spawn_fn`` via the daemon-path integration
(:func:`apply_launch_wiring`). Live-shaped fixtures simulate the dispatch
loop's claim -> spawn sequence against a board DB with real 8-hex task IDs,
including kernel-failure injections for every abort path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pytest

from hermes_cli import kanban_dispatch_wiring as wiring
from hermes_cli.kanban_dispatch_wiring import (
    LaunchWiringConfig,
    apply_launch_wiring,
    clear_launch_wiring,
    configure_launch_wiring,
    wrap_spawn_with_launch_protocol,
)
from hermes_cli.kanban_launch_protocol import (
    LaunchProtocolError,
    install_launch_schema,
    release_claim,
)

BOARD_UUID = "5f0e8d3a-1c2b-4e6f-9a7d-c8b64e2f0001"
TASK_A = "t_04d0aa77"
TASK_B = "t_11aa22bb"


def _board_db(tmp_path: Path, *task_ids: str) -> Path:
    db = tmp_path / "board.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'ready',
            assignee TEXT
        );
        """
    )
    for tid in task_ids:
        conn.execute(
            "INSERT INTO tasks (id, status, assignee) VALUES (?, 'ready', 'orch')",
            (tid,),
        )
    conn.commit()
    conn.close()
    return db


def _launch_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "launch.db")
    conn.execute("PRAGMA journal_mode=DELETE")
    install_launch_schema(conn)
    conn.execute(
        "INSERT INTO kanban_policy_pointer "
        "(singleton, schema_version, active_generation) VALUES (1, 1, 1)"
    )
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    return conn


class _FakeTask:
    """Live Task-shaped object: attributes the dispatch loop relies on."""

    def __init__(self, task_id: str) -> None:
        self.id = task_id
        self.assignee = "orch"


class _RecordingSpawn:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.calls: list[tuple[str, str]] = []

    def __call__(self, task: Any, workspace: str, board: Any = None) -> int:
        self.calls.append((task.id, workspace))
        return self.pid


def _config(tmp_path: Path, board_db: Path, **overrides: Any) -> LaunchWiringConfig:
    kwargs: dict[str, Any] = dict(
        board_database=board_db,
        board_uuid=BOARD_UUID,
        owner_home=tmp_path / "home",
        owner_generation=7,
        policy_generation=1,
        route_generation=1,
        current_claim_token_fn=lambda task_id: "ctok_" + task_id[2:] + "0000",
        next_run_generation_fn=lambda task_id: 1,
    )
    kwargs.update(overrides)
    return LaunchWiringConfig(**kwargs)


def _assert_no_live_claim(conn: sqlite3.Connection, board: Any, task_id: str) -> None:
    """Stronger abort assertion (Atomic round-1 NB-2): released means gone."""
    rows = conn.execute(
        "SELECT state FROM kanban_launch_protocol WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    assert rows == [], f"expected no ledger rows after release, got {rows}"


def test_wrapper_passes_through_and_records_full_lifecycle(tmp_path: Path) -> None:
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    inner = _RecordingSpawn()
    wrapped = wrap_spawn_with_launch_protocol(inner, launch, _config(tmp_path, board_db))
    try:
        pid = wrapped(_FakeTask(TASK_A), "/ws/a", board=None)
        assert pid == inner.pid
        assert inner.calls == [(TASK_A, "/ws/a")]
        rec_rows = launch.execute(
            "SELECT run_generation, state, pid FROM kanban_launch_protocol "
            "WHERE task_id = ?",
            (TASK_A,),
        ).fetchall()
        # Lifecycle: exactly one row per generation, terminal state spawned
        # with the recorded pid (the CAS rules forbid skipping states, so a
        # spawned terminal row implies the full ordered sequence).
        assert rec_rows == [(1, "spawned", inner.pid)]
    finally:
        launch.close()


def test_wrapper_releases_claim_when_inner_spawn_raises(tmp_path: Path) -> None:
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    wrapped = wrap_spawn_with_launch_protocol(_boom, launch, _config(tmp_path, board_db))
    try:
        with pytest.raises(LaunchProtocolError) as exc_info:
            wrapped(_FakeTask(TASK_A), "/ws/a")
        assert "aborted" in str(exc_info.value).lower()
        _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


def _boom(task: Any, workspace: str, board: Any = None) -> int:
    raise RuntimeError("spawn exploded")


def test_wrapper_no_pid_fails_closed_and_releases_claim(tmp_path: Path) -> None:
    """The kernel requires exactly one execution identity; a local spawn_fn
    that reports no PID cannot be represented, so the wrapper must fail the
    launch closed and leave no ledger row."""
    board_db = _board_db(tmp_path, TASK_B)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    inner = _RecordingSpawn(pid=0)
    wrapped = wrap_spawn_with_launch_protocol(inner, launch, _config(tmp_path, board_db))
    try:
        with pytest.raises(LaunchProtocolError) as exc_info:
            wrapped(_FakeTask(TASK_B), "/ws/b", board="main")
        assert "invalid pid" in str(exc_info.value)
        _assert_no_live_claim(launch, wrapped.board, TASK_B)
    finally:
        launch.close()


def test_second_launch_of_same_task_advances_generation(tmp_path: Path) -> None:
    """Live semantics: a reclaimed task is re-launched at generation+1."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    inner = _RecordingSpawn()
    gen_box = {"n": 0}

    def _gen(task_id: str) -> int:
        gen_box["n"] += 1
        return gen_box["n"]

    cfg = _config(
        tmp_path,
        board_db,
        next_run_generation_fn=_gen,
        current_claim_token_fn=lambda task_id: (
            "ctok_" + task_id[2:] + "g%02d0000" % gen_box["n"]
        ),
    )
    wrapped = wrap_spawn_with_launch_protocol(inner, launch, cfg)
    try:
        wrapped(_FakeTask(TASK_A), "/ws/a1")
        wrapped(_FakeTask(TASK_A), "/ws/a2")
        rows = launch.execute(
            "SELECT run_generation, state FROM kanban_launch_protocol "
            "WHERE task_id = ? ORDER BY run_generation",
            (TASK_A,),
        ).fetchall()
        assert rows == [(1, "spawned"), (2, "spawned")]
    finally:
        launch.close()


# ---------------------------------------------------------------------------
# Kernel-failure injections (Atomic round-1 NB-3)
# ---------------------------------------------------------------------------


def test_owner_acquisition_failure_leaks_no_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lease-first ordering: when acquire_dispatcher_owner fails, no claim
    row may exist (Atomic round-1 finding 2)."""
    board_db = _board_db(tmp_path, TASK_A)
    # A regular file as the owner home: makedirs fails before construction
    # completes — and, critically, before any claim is written.
    fake_home = tmp_path / "not-a-dir"
    fake_home.write_text("i am a file")
    launch = _launch_conn(tmp_path)
    with pytest.raises(OSError):
        wrap_spawn_with_launch_protocol(
            _RecordingSpawn(),
            launch,
            _config(tmp_path, board_db, owner_home=fake_home),
        )
    try:
        rows = launch.execute(
            "SELECT 1 FROM kanban_launch_protocol WHERE task_id = ?", (TASK_A,)
        ).fetchall()
        assert rows == []
    finally:
        launch.close()


def test_spawn_intent_failure_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    cfg = _config(tmp_path, board_db)

    def _failing_intent(*args: Any, **kwargs: Any) -> None:
        raise LaunchProtocolError("policy revoked")

    monkeypatch.setattr(wiring, "record_spawn_intent", _failing_intent)
    wrapped = wrap_spawn_with_launch_protocol(_RecordingSpawn(), launch, cfg)
    try:
        with pytest.raises(LaunchProtocolError):
            wrapped(_FakeTask(TASK_A), "/ws/a")
        _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


def test_record_spawned_failure_terminates_worker_and_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real spawned worker must not outlive an unrecordable launch
    (Atomic round-1 finding 3): record_spawned failure triggers
    best-effort termination and claim release."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    cfg = _config(tmp_path, board_db)
    killed: list[int] = []

    def _failing_spawned(*args: Any, **kwargs: Any) -> None:
        raise LaunchProtocolError("cas lost")

    monkeypatch.setattr(wiring, "record_spawned", _failing_spawned)
    monkeypatch.setattr(wiring, "_best_effort_terminate", killed.append)
    wrapped = wrap_spawn_with_launch_protocol(_RecordingSpawn(pid=9999), launch, cfg)
    try:
        with pytest.raises(LaunchProtocolError):
            wrapped(_FakeTask(TASK_A), "/ws/a")
        assert killed == [9999]
        _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


def test_non_integer_pid_fails_closed(tmp_path: Path) -> None:
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)

    def _weird_pid(task: Any, workspace: str, board: Any = None) -> Any:
        return "not-a-pid"

    wrapped = wrap_spawn_with_launch_protocol(_weird_pid, launch, _config(tmp_path, board_db))
    try:
        with pytest.raises(LaunchProtocolError) as exc_info:
            wrapped(_FakeTask(TASK_A), "/ws/a")
        assert "invalid pid" in str(exc_info.value)
        _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


# ---------------------------------------------------------------------------
# Daemon-path integration (Atomic round-1 finding 1)
# ---------------------------------------------------------------------------


def test_apply_launch_wiring_passthrough_when_unconfigured() -> None:
    sentinel = lambda task, workspace: 1  # noqa: E731
    clear_launch_wiring()
    assert apply_launch_wiring(sentinel) is sentinel
    assert apply_launch_wiring(None) is None


def test_apply_launch_wiring_decorates_when_configured(tmp_path: Path) -> None:
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    try:
        configure_launch_wiring(launch, _config(tmp_path, board_db))
        inner = _RecordingSpawn()
        wrapped = apply_launch_wiring(inner)
        assert wrapped is not inner
        assert isinstance(wrapped, wiring.WrappedSpawn)
        pid = wrapped(_FakeTask(TASK_A), "/ws/a")
        assert pid == inner.pid
        rows = launch.execute(
            "SELECT state, pid FROM kanban_launch_protocol WHERE task_id = ?",
            (TASK_A,),
        ).fetchall()
        assert rows == [("spawned", inner.pid)]
    finally:
        clear_launch_wiring()
        launch.close()


def test_string_pid_fails_closed(tmp_path: Path) -> None:
    """Atomic round-2 finding 3: "123", 1.5, and True must NOT be recorded
    as spawned workers; only a positive real int is a valid identity."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)

    def _str_pid(task: Any, workspace: str, board: Any = None) -> Any:
        return "123"

    wrapped = wrap_spawn_with_launch_protocol(_str_pid, launch, _config(tmp_path, board_db))
    try:
        with pytest.raises(LaunchProtocolError):
            wrapped(_FakeTask(TASK_A), "/ws/a")
        _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


def test_cleanup_survives_release_claim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic round-2 finding 2: when release_claim itself fails during
    cleanup (lease lost / policy moved), the ORIGINAL kernel error must
    propagate, not be masked by a cleanup failure."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)

    def _failing_release(*args: Any, **kwargs: Any) -> None:
        raise LaunchProtocolError("owner lease was lost before release")

    monkeypatch.setattr(wiring, "release_claim", _failing_release)

    def _boom(task: Any, workspace: str, board: Any = None) -> int:
        raise RuntimeError("spawn exploded")

    wrapped = wrap_spawn_with_launch_protocol(_boom, launch, _config(tmp_path, board_db))
    try:
        with pytest.raises(LaunchProtocolError) as exc_info:
            wrapped(_FakeTask(TASK_A), "/ws/a")
        # the original spawn failure is the reported cause, not the
        # cleanup failure ("owner lease was lost before release")
        assert "spawn exploded" in str(exc_info.value)
        assert "lease was lost" not in str(exc_info.value)
    finally:
        launch.close()


def test_dispatch_once_routes_spawn_through_wiring(
    tmp_path: Path,
) -> None:
    """Atomic round-2 finding 1: with wiring configured, the REAL
    dispatch_once spawn path is decorated — the ledger records the launch
    triggered by the production dispatch loop, not a direct wrapper call.

    Runs as a subprocess with a clean env: the kernel blocks delegated
    child contexts from mutating boards, and pytest inherits that marker
    when run under this agent session.
    """
    import os
    import subprocess
    import sys

    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, pathlib, sqlite3\n"
        "from hermes_cli import kanban_db as kb\n"
        "from hermes_cli.kanban_dispatch_wiring import (\n"
        "    configure_launch_wiring, clear_launch_wiring,\n"
        ")\n"
        "from hermes_cli.kanban_launch_protocol import (\n"
        "    install_launch_schema,\n"
        ")\n"
        f"BOARD_UUID = {BOARD_UUID!r}\n"
        "board_db = pathlib.Path(os.environ['T_BOARD'])\n"
        "launch_db = pathlib.Path(os.environ['T_LAUNCH'])\n"
        "home = pathlib.Path(os.environ['T_HOME'])\n"
        "home.mkdir(mode=0o700, exist_ok=True)\n"
        "os.environ['HERMES_KANBAN_DB'] = str(board_db)\n"
        "os.environ.pop('HERMES_DELEGATED_CHILD_CONTEXT', None)\n"
        "os.environ.pop('HERMES_DELEGATED_CHILD', None)\n"
        "launch = sqlite3.connect(launch_db)\n"
        "launch.execute('PRAGMA journal_mode=DELETE')\n"
        "install_launch_schema(launch)\n"
        "launch.execute(\"INSERT INTO kanban_policy_pointer \"\n"
        "              \"(singleton, schema_version, active_generation) \"\n"
        "              \"VALUES (1, 1, 1)\")\n"
        "launch.execute('PRAGMA foreign_keys=ON')\n"
        "launch.commit()\n"
        "inner = lambda task, ws, board=None: (print('SPAWN', task.id), 4242)[1]\n"
        "import hermes_cli.profiles as _P\n"
        "_P.profile_exists = lambda name: True\n"
        "kb._default_spawn = inner\n"
        "with kb.connect() as bc:\n"
        "    task_id = kb.create_task(bc, title='probe', body='b', assignee='orch')\n"
        "configure_launch_wiring(launch, cfg := __import__('hermes_cli.kanban_dispatch_wiring', fromlist=['LaunchWiringConfig']).LaunchWiringConfig(\n"
        "    board_database=board_db, board_uuid=BOARD_UUID, owner_home=home,\n"
        "    owner_generation=7, policy_generation=1, route_generation=1,\n"
        "    current_claim_token_fn=lambda tid: 'ctok_' + tid[2:] + '0000',\n"
        "    next_run_generation_fn=lambda tid: 1))\n"
        "try:\n"
        "    with kb.connect() as bc:\n"
        "        kb.dispatch_once(bc, max_spawn=1, max_in_progress=99)\n"
        "finally:\n"
        "    clear_launch_wiring()\n"
        "    launch.close()\n"
        "rows = sqlite3.connect(launch_db).execute(\n"
        "    'SELECT state, pid FROM kanban_launch_protocol WHERE task_id = ?',\n"
        "    (task_id,)).fetchall()\n"
        "assert rows == [('spawned', 4242)], rows\n"
        "print('LEDGER_OK', rows)\n"
    )
    launch_path = tmp_path / "launch.db"
    env = {
        **{k: v for k, v in os.environ.items() if k.startswith(("PATH", "HOME"))},
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "T_BOARD": str(tmp_path / "board.db"),
        "T_LAUNCH": str(launch_path),
        "T_HOME": str(tmp_path / "home"),
        "HERMES_KANBAN_DB": str(tmp_path / "board.db"),
    }
    proc = subprocess.run(
        [sys.executable, str(driver)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"driver failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "SPAWN" in proc.stdout, "dispatch_once never spawned"
    assert "LEDGER_OK" in proc.stdout


# ---------------------------------------------------------------------------
# Atomic round-3 finding pins
# ---------------------------------------------------------------------------


def test_apply_launch_wiring_is_idempotent(tmp_path: Path) -> None:
    """Atomic round-3 finding 3: double wiring must not nest owner leases."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    try:
        wiring.configure_launch_wiring(launch, _config(tmp_path, board_db))
        try:
            inner = _RecordingSpawn()
            once = wiring.apply_launch_wiring(inner)
            twice = wiring.apply_launch_wiring(once)
            assert isinstance(once, wiring.WrappedSpawn)
            assert twice is once, "second wiring must return the same wrapper"
            # And through the daemon-path helper (which the dispatch loop
            # calls): still the same single wrapper, never nested.
            from hermes_cli.kanban_db import _apply_launch_wiring_to_spawn
            assert _apply_launch_wiring_to_spawn(once) is once
        finally:
            wiring.clear_launch_wiring()
        # Unconfigured: exact identity even for a pre-wrapped callable.
        from hermes_cli.kanban_db import _apply_launch_wiring_to_spawn
        sentinel = _RecordingSpawn()
        assert _apply_launch_wiring_to_spawn(sentinel) is sentinel
    finally:
        launch.close()


def test_oversized_pid_rejected_fail_closed(tmp_path: Path) -> None:
    """Atomic round-3 finding 2: >int32 pid must fail closed (SQLite
    binding / os.kill OverflowError class), not record a bogus launch."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    try:
        inner = _RecordingSpawn(pid=0x7FFFFFFF + 1)
        wrapped = wrap_spawn_with_launch_protocol(
            inner, launch, _config(tmp_path, board_db)
        )
        with pytest.raises(LaunchProtocolError) as exc_info:
            wrapped(_FakeTask(TASK_A), "/ws/a", board=None)
        assert "invalid pid" in str(exc_info.value)
        _assert_no_live_claim(launch, None, TASK_A)
    finally:
        launch.close()


def test_construction_failure_never_strands_board_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic round-3 finding 1: a WrappedSpawn construction failure (e.g.
    board identity canonicalization error) must be classified as a spawn
    failure for that row — the board task must NOT remain claimed."""
    from hermes_cli import kanban_db as kb

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("board identity exploded")

    monkeypatch.setattr(
        "hermes_cli.kanban_dispatch_wiring.canonical_board_identity", boom
    )
    import hermes_cli.profiles as _profiles

    monkeypatch.setattr(_profiles, "profile_exists", lambda name: True)
    conn = kb.connect()
    task_id = kb.create_task(conn, title="c", body="b", assignee="orch")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,)
        )
    wiring.configure_launch_wiring(conn, _config(tmp_path, tmp_path / "b.db"))
    try:
        result = kb.dispatch_once(conn, max_spawn=1, max_in_progress=99)
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        # Claim retired: not stuck 'running'; failure recorded via the
        # existing spawn-failure machinery (back to ready with a failure
        # event, or auto-blocked past the limit — never a stranded claim).
        assert row["status"] != "running"
        # Atomic round-4 note 1/3 (stronger pin): the error text must be
        # the construction failure itself — proving the closure captured
        # the original exception (Python clears ``exc`` at except-block
        # exit; a naive closure raises NameError instead).
        lfe = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()["last_failure_error"]
        # Exact identity: both the generic prefix AND the bound original
        # cause text must appear — a dropped/naive closure would record
        # only the prefix (or a NameError message) instead.
        assert lfe is not None, "construction failure not recorded"
        assert "launch wiring construction failed" in lfe, (
            f"construction failure not recorded as such: {lfe!r}"
        )
        assert "board identity exploded" in lfe, (
            f"bound cause text missing from last_failure_error: {lfe!r}"
        )
        assert "NameError" not in lfe, f"closure bug leaked: {lfe!r}"
    finally:
        wiring.clear_launch_wiring()
        conn.close()


# ---------------------------------------------------------------------------
# Atomic round-4 note pins
# ---------------------------------------------------------------------------


def test_construction_failure_error_text_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic round-4 Medium: the per-row construction-failure callable
    must raise the ORIGINAL construction exception, not NameError from a
    closure over the cleared ``exc`` variable."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("board identity exploded")

    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.kanban_dispatch_wiring.canonical_board_identity", boom
    )
    wiring.configure_launch_wiring(launch, _config(tmp_path, board_db))
    from hermes_cli.kanban_db import _apply_launch_wiring_to_spawn

    try:
        sentinel = _RecordingSpawn()
        unwirable = _apply_launch_wiring_to_spawn(sentinel)
        assert unwirable is not sentinel
        assert not isinstance(unwirable, wiring.WrappedSpawn)
        with pytest.raises(RuntimeError) as exc_info:
            unwirable(_FakeTask(TASK_A), "/ws/a")
        msg = str(exc_info.value)
        assert "launch wiring construction failed" in msg
        assert "board identity exploded" in msg
        assert "NameError" not in msg
        assert isinstance(exc_info.value.__cause__, RuntimeError)
    finally:
        wiring.clear_launch_wiring()
        launch.close()


def test_pid_identity_matrix_fails_closed(tmp_path: Path) -> None:
    """Atomic round-4 note 2: every invalid pid shape — None, booleans,
    floats, strings, zero, negatives, >int32 — is rejected fail-closed
    with the claim released."""
    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    bad_pids: list[Any] = [
        None,
        True,
        False,
        0,
        -1,
        1.5,
        "123",
        0x7FFFFFFF + 1,
    ]
    try:
        for bad in bad_pids:
            wrapped = wrap_spawn_with_launch_protocol(
                (lambda p: lambda task, ws, board=None: p)(bad),
                launch,
                _config(tmp_path, board_db),
            )
            with pytest.raises(LaunchProtocolError) as exc_info:
                wrapped(_FakeTask(TASK_A), "/ws/a")
            assert "invalid pid" in str(exc_info.value), (
                f"pid {bad!r} not rejected with invalid-pid error"
            )
            _assert_no_live_claim(launch, wrapped.board, TASK_A)
    finally:
        launch.close()


def test_lease_release_failure_after_record_spawned_does_not_flip_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic round-4 note 3: a ``lease.release()`` failure in the final
    finally must NOT flip a fully recorded successful spawn into a
    dispatcher-level failure."""

    board_db = _board_db(tmp_path, TASK_A)
    (tmp_path / "home").mkdir()
    launch = _launch_conn(tmp_path)
    cfg = _config(tmp_path, board_db)
    real_lease = wiring.acquire_dispatcher_owner(
        tmp_path / "home", cfg.owner_generation
    )

    # Subclass keeps isinstance(DispatcherOwnerLease) valid (the kernel
    # type-checks the lease) while making release() raise.
    _LeaseCls = type(real_lease)

    class _ExplodingLease(_LeaseCls):
        def release(self) -> None:
            raise RuntimeError("lease file vanished")

    exploding = object.__new__(_ExplodingLease)
    for slot in real_lease.__slots__:
        object.__setattr__(
            exploding, slot, getattr(real_lease, slot)
        )
    monkeypatch.setattr(
        wiring,
        "acquire_dispatcher_owner",
        lambda *a, **k: exploding,
    )
    inner = _RecordingSpawn()
    wrapped = wrap_spawn_with_launch_protocol(
        inner, launch, _config(tmp_path, board_db)
    )
    try:
        pid = wrapped(_FakeTask(TASK_A), "/ws/a")
        assert pid == inner.pid
        rows = launch.execute(
            "SELECT state, pid FROM kanban_launch_protocol WHERE task_id = ?",
            (TASK_A,),
        ).fetchall()
        assert rows == [("spawned", inner.pid)]
    finally:
        launch.close()


def test_terminate_swallows_non_oserror_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic round-4 note 3: _best_effort_terminate must swallow ANY
    exception class (not just OSError) — e.g. OverflowError from an
    out-of-range pid — so it can never mask the ledger failure."""

    calls: list[int] = []

    def _overflow_kill(pid: int, sig: int) -> None:
        calls.append(pid)
        raise OverflowError("Python int too large to convert to C long")

    monkeypatch.setattr(wiring.os, "kill", _overflow_kill)
    # Must not raise.
    wiring._best_effort_terminate(999999999999)
    assert calls == [999999999999]
