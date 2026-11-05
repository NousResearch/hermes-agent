"""Resource, patience and cross-caller contracts of the ownership fence (#149132)."""
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
import hermes_state_lineage as lineage
from hermes_cli.sessions_repair_profiles import Store, _MoveBatch, _Session


@contextmanager
def _tracked_descriptors(monkeypatch):
    opened = set()
    original_open, original_close = os.open, os.close

    def open_file(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.add(fd)
        return fd

    def close_file(fd):
        try:
            return original_close(fd)
        finally:
            opened.discard(fd)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "close", close_file)
    try:
        yield opened
    finally:
        for fd in list(opened):
            original_close(fd)
            opened.discard(fd)


@pytest.mark.parametrize("failure", ["timeout", "fstat"])
def test_failed_lock_entry_closes_every_open_descriptor(tmp_path, monkeypatch, failure):
    path = tmp_path / "guard.lock"
    with _tracked_descriptors(monkeypatch) as opened:
        if failure == "timeout":
            with lineage._advisory_lock_file(path, 0):
                held = set(opened)
                for _ in range(5):
                    with pytest.raises(lineage.SessionLineageInventoryError):
                        with lineage._advisory_lock_file(path, 0):
                            pytest.fail("contended lock was acquired")
                    assert opened == held
        else:
            def unreadable(_fd):
                raise OSError("injected fstat failure")
            monkeypatch.setattr(os, "fstat", unreadable)
            with pytest.raises(lineage.SessionLineageInventoryError):
                with lineage._advisory_lock_file(path, 0):
                    pytest.fail("unverified lock was acquired")
            assert not opened
    assert not opened


@pytest.mark.platforms("posix")
def test_unlock_failure_still_closes_the_descriptor(tmp_path, monkeypatch):
    import fcntl

    original = fcntl.flock

    def broken_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("injected unlock failure")
        return original(fd, operation)

    monkeypatch.setattr(fcntl, "flock", broken_unlock)
    with _tracked_descriptors(monkeypatch) as opened:
        with pytest.raises(lineage.SessionLineageInventoryError):
            with lineage._advisory_lock_file(tmp_path / "guard.lock", 0):
                pass
        assert not opened


def _hold_session_lock(db_path, session_id, channel):
    with lineage.session_lineage_guard(db_path, session_id):
        channel.send("held")
        if not channel.poll(10):
            raise RuntimeError("parent did not release its fixture lock")
        channel.recv()
    channel.send("released")
    channel.close()


@pytest.mark.parametrize("operation", ["append", "batch", "create", "usage", "receipt"])
def test_transcript_writers_keep_their_declared_wait_budget(tmp_path, monkeypatch, operation):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    if operation != "create":
        db.create_session("waiting", "cli")
    db._TRANSCRIPT_WRITE_PATIENCE_S = 2.0
    from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution
    scope = dict(epoch=begin_runtime_epoch(db, instance_id="wait-budget"),
                 execution_id="writer", session_id="waiting", generation=0)
    if operation == "receipt":
        register_worker_execution(db, **scope, kind="compute", adoption_secret="fixture-only")
    context = multiprocessing.get_context("spawn")
    channel, other = context.Pipe()
    holder = context.Process(target=_hold_session_lock, args=(db.db_path, "waiting", other))
    holder.start()
    try:
        assert channel.poll(5) and channel.recv() == "held"
        defaults = dict(lineage.session_lineage_guard.__wrapped__.__kwdefaults__)
        monkeypatch.setattr(lineage.session_lineage_guard.__wrapped__, "__kwdefaults__", {**defaults, "timeout_s": 0.01})
        clock = [0.0]
        polls = []

        def advance(_delay):
            polls.append(True)
            clock[0] += 0.1
            if len(polls) == 2:
                channel.send("release")
                assert channel.poll(5) and channel.recv() == "released"

        monkeypatch.setattr(lineage, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance))
        original_lock, budgets = lineage._lock_file, []

        @contextmanager
        def observe_budget(path, timeout_s):
            budgets.append(timeout_s)
            with original_lock(path, timeout_s) as fd:
                yield fd

        monkeypatch.setattr(lineage, "_lock_file", observe_budget)
        calls = {
            "append": lambda: db.append_message("waiting", "user", "retained"),
            "batch": lambda: db.append_messages_batch("waiting", [{"role": "user", "content": "retained"}]),
            "create": lambda: db.create_session("waiting", "cli"),
            "usage": lambda: db.record_auxiliary_usage("waiting", "vision", input_tokens=1),
            "receipt": lambda: mutate_worker_execution(db, **scope, sequence=1, operation="usage.main", payload={"input_tokens": 1}),
        }
        calls[operation]()
        observed = {
            "append": lambda: len(db.get_messages("waiting")),
            "batch": lambda: len(db.get_messages("waiting")),
            "create": lambda: int(db.get_session("waiting") is not None),
            "usage": lambda: db.auxiliary_usage_by_task("waiting")["vision"]["input_tokens"],
            "receipt": lambda: db.get_session("waiting")["input_tokens"],
        }
        assert len(polls) == 2 and observed[operation]() == 1
        assert 0 < budgets[0] <= db._TRANSCRIPT_WRITE_PATIENCE_S
    finally:
        if holder.is_alive():
            channel.send("release")
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        channel.close()
        other.close()
        db.close()


def _batch_with_limited_descriptors(root, mode, size, channel):
    import resource

    os.environ["HERMES_HOME"] = str(root)
    stores = Store("default", Path(root)), Store("science", Path(root) / "profiles" / "science")
    stores[1].home.mkdir(parents=True)
    session = _Session(read_only=False)
    source, target = (session.db(store) for store in stores)
    ids = [f"batch-{index:03}" for index in range(size)]
    if mode == "move":
        for sid in ids:
            source.create_session(sid, "cli")
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    available = len(os.listdir("/proc/self/fd")) + 65
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], available), limits[1]))
    try:
        if mode == "import":
            result = source.import_sessions([{"id": sid, "source": "cli", "messages": []} for sid in ids])
            assert result["imported"] == len(ids)
        else:
            transfer = _MoveBatch(*stores)
            for sid in ids:
                transfer.fix_for(sid)
            result = transfer.run(session)
            assert all(result[sid].get("moved") == 1 for sid in ids), result
            assert all(source.get_session(sid) is None and target.get_session(sid) is not None for sid in ids)
        channel.send({"ok": True, "count": len(ids)})
    except Exception as exc:
        channel.send({"ok": False, "stage": "batch", "type": type(exc).__name__, "error": str(exc)})
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
        session.close()
        channel.close()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("mode", ["import", "move"])
@pytest.mark.parametrize("size", [128, 500])
def test_supported_batch_works_with_bounded_descriptor_headroom(tmp_path, mode, size):
    context = multiprocessing.get_context("spawn")
    channel, other = context.Pipe()
    worker = context.Process(target=_batch_with_limited_descriptors, args=(tmp_path / "hermes", mode, size, other))
    worker.start()
    try:
        assert channel.poll(30), "bounded batch did not finish"
        result = channel.recv()
        worker.join(timeout=5)
        assert worker.exitcode == 0 and result["ok"], result
        assert result["count"] == size
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        channel.close()
        other.close()


def test_repair_and_multi_id_guard_use_the_same_kernel_lock_order(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    stores = Store("default", root), Store("science", root / "profiles" / "science")
    stores[1].home.mkdir(parents=True)
    session = _Session(read_only=False)
    source = session.db(stores[0])
    session.db(stores[1])
    ids = [f"order-{index}" for index in range(8)]
    for sid in ids:
        source.create_session(sid, "cli")
    original = lineage._advisory_lock_file
    acquired = {"direct": [], "repair": []}
    operation = ["direct"]

    @contextmanager
    def observed(path, timeout_s):
        acquired[operation[0]].append(path.name)
        with original(path, timeout_s) as fd:
            yield fd

    monkeypatch.setattr(lineage, "_advisory_lock_file", observed)
    try:
        with lineage.session_lineage_guard(source.db_path, ids):
            pass
        operation[0] = "repair"
        transfer = _MoveBatch(*stores)
        for sid in reversed(ids):
            transfer.fix_for(sid)
        assert all(result.get("moved") == 1 for result in transfer.run(session).values())
        assert acquired["direct"] == acquired["repair"]
    finally:
        session.close()


def test_multi_id_acquisition_spends_one_shared_timeout(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    clock = [0.0]

    @contextmanager
    def contended(path, timeout_s):
        if timeout_s <= 0.25:
            raise lineage.SessionLineageInventoryError("shared acquisition budget exhausted")
        clock[0] += 0.25
        yield -1

    monkeypatch.setattr(lineage, "_lock_file", contended)
    monkeypatch.setattr(lineage, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    try:
        with pytest.raises(lineage.SessionLineageInventoryError, match="shared acquisition budget"):
            with lineage.session_lineage_guard(db.db_path, [f"deadline-{i}" for i in range(16)], timeout_s=0.5):
                pass
    finally:
        db.close()


def test_process_and_advisory_phases_share_the_same_timeout(tmp_path, monkeypatch):
    clock, advisory_budget, released = [0.0], [], []

    class ProcessLock:
        def acquire(self, *, timeout):
            clock[0] += 0.4
            return True

        def release(self):
            released.append(True)

    @contextmanager
    def advisory(path, timeout_s):
        advisory_budget.append(timeout_s)
        yield -1

    monkeypatch.setattr(lineage, "_ProcessLockEntry", lambda: SimpleNamespace(lock=ProcessLock(), users=0))
    monkeypatch.setattr(lineage, "_advisory_lock_file", advisory)
    monkeypatch.setattr(lineage, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with lineage._lock_file(tmp_path / "phase.lock", 0.5):
        pass
    assert advisory_budget == [pytest.approx(0.1)]
    assert released == [True]


def test_nested_guards_reuse_complete_sets_and_refuse_order_inversion(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    ordered = sorted((lineage._lineage_lock_paths(root, [f"nested-{i}"])[0], f"nested-{i}") for i in range(16))
    low, high = ordered[0][1], ordered[-1][1]
    try:
        with lineage.session_lineage_guard(db.db_path, [low, high]):
            with lineage.session_lineage_guard(db.db_path, high):
                with lineage.session_lineage_guard(db.db_path, low):
                    pass
        with lineage.session_lineage_guard(db.db_path, high):
            with pytest.raises(lineage.SessionLineageInventoryError, match="order"):
                with lineage.session_lineage_guard(db.db_path, low):
                    pytest.fail("nested acquisition inverted the global order")
        assert not lineage._PROCESS_LOCKS
    finally:
        db.close()


def test_nonoverlapping_stripes_can_make_progress_concurrently(tmp_path, monkeypatch):
    import threading

    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    ordered = sorted((lineage._lineage_lock_paths(root, [f"parallel-{i}"])[0], f"parallel-{i}") for i in range(16))
    low, high = ordered[0][1], ordered[-1][1]
    finished, outcome = threading.Event(), []

    def writer():
        try:
            with lineage.session_lineage_guard(db.db_path, high, timeout_s=0):
                outcome.append("entered")
        except Exception as exc:
            outcome.append(type(exc).__name__)
        finally:
            finished.set()

    worker = threading.Thread(target=writer, daemon=True)
    try:
        with lineage.session_lineage_guard(db.db_path, low):
            worker.start()
            assert finished.wait(5) and outcome == ["entered"]
        worker.join(timeout=5)
        assert not worker.is_alive()
    finally:
        db.close()


def test_inherited_process_entries_do_not_change_current_lock_order(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    # The cache key includes the PID because a child must not reuse a parent's fd.
    # Apply the same process boundary when checking nested acquisition order.
    inherited = {f"999999999:{root / '.session-lineage-locks' / 'stripe-99.lock'}": -1}
    monkeypatch.setattr(lineage._THREAD_LOCKS, "active", inherited, raising=False)
    try:
        with lineage.session_lineage_guard(db.db_path, "current-process", timeout_s=0):
            pass
        assert inherited == {f"999999999:{root / '.session-lineage-locks' / 'stripe-99.lock'}": -1}
    finally:
        db.close()


def test_interrupted_process_lock_entry_releases_its_registry_reference(tmp_path, monkeypatch):
    class InterruptedLock:
        def acquire(self, *, timeout):
            raise KeyboardInterrupt("injected interrupted acquisition")

        def release(self):
            pytest.fail("an unacquired process lock must not be released")

    before = dict(lineage._PROCESS_LOCKS)
    monkeypatch.setattr(lineage, "_ProcessLockEntry", lambda: SimpleNamespace(lock=InterruptedLock(), users=0))
    try:
        with pytest.raises(KeyboardInterrupt):
            with lineage._lock_file(tmp_path / "interrupted.lock", 0):
                pytest.fail("interrupted acquisition entered its body")
        assert lineage._PROCESS_LOCKS == before
    finally:
        for key in set(lineage._PROCESS_LOCKS) - set(before):
            lineage._PROCESS_LOCKS.pop(key)


def test_shared_stripe_never_substitutes_for_session_ownership(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    buckets = {}
    for index in range(100):
        sid = f"collision-{index}"
        bucket = buckets.setdefault(lineage._lineage_lock_paths(root, [sid])[0], [])
        bucket.append(sid)
        if len(bucket) == 3:
            owned, foreign, unused = bucket
            break
    else:
        pytest.fail("fixture did not find three IDs sharing a coordination stripe")
    local, sibling = SessionDB(root / "state.db"), SessionDB(profile / "state.db")
    try:
        local.create_session(owned, "cli")
        sibling.create_session(foreign, "cli")
        with lineage.session_lineage_guard(local.db_path, owned):
            with pytest.raises(lineage.SessionIdOwnedElsewhereError):
                local.create_session(foreign, "cli")
            local.create_session(unused, "cli")
        assert local.get_session(foreign) is None
        assert sibling.get_session(foreign)["ended_at"] is None
        assert local.get_session(unused) is not None
    finally:
        sibling.close()
        local.close()
