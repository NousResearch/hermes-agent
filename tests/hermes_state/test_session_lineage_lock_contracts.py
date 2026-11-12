"""Resource, patience and cross-caller contracts of the ownership fence (#149132)."""
from contextlib import contextmanager
from itertools import pairwise
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


def test_lineage_lock_namespace_is_shared_private_and_not_a_profile(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.profiles import _iter_named_profile_dirs

    root = tmp_path / "hermes"
    profiles_root = root / "profiles"
    profile = profiles_root / "science"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("")
    if os.name != "nt":
        profiles_root.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(str(root))
    try:
        with SessionDB(root / "state.db") as default_db, SessionDB(profile / "state.db") as named_db:
            if os.name != "nt":
                root.chmod(0o755)
            with lineage.session_lineage_guard(default_db.db_path, "shared-lock-namespace"):
                pass
            with lineage.session_lineage_guard(named_db.db_path, "shared-lock-namespace"):
                pass

            lock_root = profiles_root / ".session-lineage-locks"
            default_lock = lineage._lineage_lock_paths(root, ["shared-lock-namespace"])[0]
            named_lock = lineage._lineage_lock_paths(root, ["shared-lock-namespace"])[0]
            assert default_lock == named_lock == lock_root / default_lock.name
            assert default_lock.is_file()
            if os.name != "nt":
                assert root.stat().st_mode & 0o777 == 0o755
                assert profiles_root.stat().st_mode & 0o777 == 0o755
                assert lock_root.stat().st_mode & 0o077 == 0
                assert default_lock.stat().st_mode & 0o077 == 0

            candidates = lineage.physical_store_candidates(root)
            assert {name for _, name in candidates} == {"default", "science"}
            assert all(path.parent != lock_root for path, _ in candidates)
            assert [entry.name for entry in _iter_named_profile_dirs()] == ["science"]
            # Reserved metadata is not a profile, but a stray physical store is
            # evidence requiring inspection, never silently excluded or deleted.
            stray = lock_root / "state.db"
            stray.write_bytes(b"stray metadata")
            with pytest.raises(lineage.SessionLineageInventoryError, match="reserved"):
                lineage.physical_store_candidates(root)
            assert stray.read_bytes() == b"stray metadata"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("redirect", ["profiles", "lock-directory"])
def test_lineage_lock_namespace_refuses_directory_redirection(tmp_path, monkeypatch, redirect):
    root = tmp_path / "hermes"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    profiles_root = root / "profiles"
    if redirect == "profiles":
        profiles_root.symlink_to(outside, target_is_directory=True)
    else:
        profiles_root.mkdir()
        (profiles_root / ".session-lineage-locks").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    with SessionDB(root / "state.db") as db:
        with pytest.raises(lineage.SessionLineageInventoryError):
            with lineage.session_lineage_guard(db.db_path, "redirected-lock-namespace"):
                pytest.fail("redirected lineage lock directory was accepted")
    assert not list(outside.iterdir())


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
    inherited = {f"999999999:{root / 'profiles' / '.session-lineage-locks' / 'stripe-99.lock'}": -1}
    monkeypatch.setattr(lineage._THREAD_LOCKS, "active", inherited, raising=False)
    try:
        with lineage.session_lineage_guard(db.db_path, "current-process", timeout_s=0):
            pass
        assert inherited == {f"999999999:{root / 'profiles' / '.session-lineage-locks' / 'stripe-99.lock'}": -1}
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


def test_named_profile_pending_accounting_leaves_default_tree_untouched(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.conformance.test_profile_write_tripwire import ProfileTripwire

    root = tmp_path / 'hermes'
    profile = root / 'profiles' / 'science'
    profile.mkdir(parents=True)
    (root / 'config.yaml').write_text('model: {}\n')
    monkeypatch.setenv('HERMES_HOME', str(profile))
    token = set_hermes_home_override(str(profile))
    tripwire = ProfileTripwire(root, profile)
    try:
        with SessionDB(profile / 'state.db') as db:
            db.create_session('usage', 'cli')
            monkeypatch.setattr(db, '_token_writer_loop', lambda: None)
            db.queue_token_counts('usage', input_tokens=3)
            db._token_writer_thread.join(5)
            assert db.has_pending_token_counts()
            tripwire.assert_default_untouched()
            assert db.flush_token_counts()
            assert db.get_session('usage')['input_tokens'] == 3
        tripwire.assert_default_untouched()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize('held', [False, True])
def test_legacy_accounting_marker_blocks_without_mutation(tmp_path, monkeypatch, held):
    from contextlib import nullcontext
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / 'hermes'
    profile = root / 'profiles' / 'science'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(profile))
    token = set_hermes_home_override(str(profile))
    try:
        with SessionDB(profile / 'state.db') as db:
            db.create_session('legacy-usage', 'cli')
            _, prefix = db._token_marker_prefix()
            legacy = root / '.session-lineage-locks' / 'accounting'
            legacy.mkdir(parents=True)
            marker = legacy / (prefix + 'legacy.lock')
            marker.write_bytes(b'legacy receipt')
            if os.name != 'nt':
                marker.chmod(0o600)
            before = marker.stat().st_mtime_ns, marker.read_bytes()
            guard = lineage._advisory_lock_file(marker, 0, create=False) if held else nullcontext()
            with guard:
                assert db.has_pending_token_counts()
                with pytest.raises(lineage.SessionLineageInventoryError, match='legacy'):
                    db.queue_token_counts('legacy-usage', input_tokens=3)
            assert (marker.stat().st_mtime_ns, marker.read_bytes()) == before
            assert not db._token_queue and not db._token_writer_uncertain
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms('posix')
def test_legacy_accounting_directory_symlink_fails_closed_without_writing(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / 'hermes'
    profile = root / 'profiles' / 'science'
    profile.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / '.session-lineage-locks').symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv('HERMES_HOME', str(profile))
    token = set_hermes_home_override(str(profile))
    try:
        with SessionDB(profile / 'state.db') as db:
            db.create_session('legacy-symlink', 'cli')
            assert db.has_pending_token_counts()
            with pytest.raises(lineage.SessionLineageInventoryError, match='legacy'):
                db.queue_token_counts('legacy-symlink', input_tokens=3)
            assert not list(outside.iterdir())
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('parent,mode', [
    ('root', 0o775), ('root', 0o757), ('profiles', 0o775), ('profiles', 0o757),
])
def test_lineage_namespace_refuses_writable_parent_without_chmod(tmp_path, monkeypatch, parent, mode):
    accessible_parent = tmp_path / 'accessible'
    accessible_parent.mkdir()
    accessible_parent.chmod(0o755)
    root = accessible_parent / 'hermes'
    root.mkdir(mode=0o755)
    profiles = root / 'profiles'
    profiles.mkdir(mode=0o755)
    selected = root if parent == 'root' else profiles
    monkeypatch.setenv('HERMES_HOME', str(root))
    with SessionDB(root / 'state.db') as db:
        selected.chmod(mode)
        if parent == 'profiles':
            root.chmod(0o755)
        with pytest.raises(lineage.SessionLineageInventoryError, match='permissions'):
            with lineage.session_lineage_guard(db.db_path, 'unsafe-parent'):
                pytest.fail('writable namespace parent was accepted')
    assert selected.stat().st_mode & 0o777 == mode
    assert not (profiles / '.session-lineage-locks').exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["create", "append"])
def test_private_hermes_home_accepts_group_writable_profiles_for_create_and_append(
    tmp_path, monkeypatch, operation,
):
    home = tmp_path / ".hermes"
    profiles = home / "profiles"
    profiles.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        if operation == "append":
            db.create_session("existing", "cli")

    home.chmod(0o700)
    profiles.chmod(0o775)
    with SessionDB(home / "state.db") as db:
        if operation == "create":
            db.create_session("new-session", "cli")
            assert db.get_session("new-session") is not None
        else:
            assert db.append_message("existing", "user", "append allowed") is not None

    assert home.stat().st_mode & 0o777 == 0o700
    assert profiles.stat().st_mode & 0o777 == 0o775
    lock_root = profiles / ".session-lineage-locks"
    assert lock_root.stat().st_mode & 0o077 == 0


@pytest.mark.platforms("posix")
def test_sibling_exclusive_inventory_wait_uses_callers_remaining_budget(tmp_path, monkeypatch):
    import sqlite3
    import threading
    import time

    home = tmp_path / "hermes"
    sibling_home = home / "profiles" / "science"
    sibling_home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        """database:
  journal_mode: delete
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    default = SessionDB(home / "state.db")
    sibling = SessionDB(sibling_home / "state.db")
    assert not default._wal_active and not sibling._wal_active
    default.create_session("default-session", "cli")
    default._WRITE_PATIENCE_S = 10.0
    default._TRANSCRIPT_WRITE_PATIENCE_S = 10.0

    ready, release = threading.Event(), threading.Event()
    held_at, holder_errors = [], []

    def hold_sibling_exclusive():
        conn = sqlite3.connect(sibling.db_path, timeout=0, isolation_level=None)
        try:
            conn.execute("BEGIN EXCLUSIVE")
            held_at.append(time.monotonic())
            ready.set()
            if not release.wait(6):
                raise RuntimeError("exclusive lock release timed out")
            conn.execute("ROLLBACK")
        except BaseException as exc:
            holder_errors.append(f"{type(exc).__name__}: {exc}")
            ready.set()
        finally:
            conn.close()

    worker = threading.Thread(target=hold_sibling_exclusive)
    worker.start()
    timer = None
    try:
        assert ready.wait(5), f"sibling lock was not acquired: {holder_errors}"
        timer = threading.Timer(3.1, release.set)
        timer.start()
        started = time.monotonic()
        outcome = None
        try:
            default.append_message("default-session", "user", "wait within caller budget")
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
    finally:
        release.set() if timer is None else None
        if timer is not None:
            timer.join(5)
        worker.join(5)
        default.close()
        sibling.close()

    assert not worker.is_alive()
    assert not holder_errors
    assert outcome is None, f"append failed after {elapsed:.3f}s: {outcome}"
    assert 3.0 <= elapsed < 10.0


def test_inventory_busy_timeout_shrinks_across_sql_statements(tmp_path, monkeypatch):
    import sqlite3

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        db.create_session("inventory-target", "cli")
        db_path = db.db_path

    busy_timeouts = []
    original_connect = sqlite3.connect

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            if sql.startswith("PRAGMA busy_timeout = "):
                busy_timeouts.append(int(sql.rsplit("=", 1)[1].strip()))
            return super().execute(sql, parameters)

    def observed_connect(*args, **kwargs):
        kwargs["factory"] = ObservedConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(lineage.sqlite3, "connect", observed_connect)
    ticks = iter((0.0, 0.1, 0.3, 0.6, 0.8))
    monkeypatch.setattr(lineage, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    rows = lineage._rows_for_ids(db_path, "default", ("inventory-target",), timeout_s=1.0)

    assert set(rows) == {"inventory-target"}
    assert len(busy_timeouts) == 4
    assert all(0 < timeout <= 1000 for timeout in busy_timeouts)
    assert all(left > right for left, right in pairwise(busy_timeouts))
    assert busy_timeouts[-1] >= 100


@pytest.mark.platforms("posix")
def test_group_writable_home_is_safe_behind_private_parent(tmp_path, monkeypatch):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    home = parent / "hermes"
    home.mkdir(mode=0o775)
    home.chmod(0o775)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        db.create_session("private-parent", "cli")
        assert db.get_session("private-parent") is not None
    assert home.stat().st_mode & 0o777 == 0o700


@pytest.mark.platforms('posix')
def test_lineage_parent_trust_accepts_root_owner_and_refuses_other_owner():
    import stat
    from types import SimpleNamespace

    root_owned = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
    lineage._validate_lineage_parent(Path('/synthetic'), root_owned)
    other = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=os.geteuid() + 1)
    with pytest.raises(lineage.SessionLineageInventoryError, match='owner'):
        lineage._validate_lineage_parent(Path('/synthetic'), other)


@pytest.fixture
def legacy_barrier_store(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    profiles = root / 'profiles'
    profiles.mkdir(mode=0o700)
    profiles.chmod(0o700)
    profile = profiles / 'science'
    profile.mkdir(mode=0o700)
    monkeypatch.setenv('HERMES_HOME', str(root))
    db = SessionDB(profile / 'state.db')
    db.create_session('legacy-barrier', 'cli')
    yield db, root
    if db._conn is not None:
        db.close()


def _legacy_marker(db, root):
    _, prefix = db._token_marker_prefix()
    lock_root = root / '.session-lineage-locks'
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_root.chmod(0o700)
    directory = lock_root / 'accounting'
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    marker = directory / (prefix + 'fixture.lock')
    marker.write_bytes(b'legacy pending')
    return marker


@pytest.mark.parametrize('writer_state', ['closed', 'stopped', 'failed_worker'])
def test_closed_queue_refuses_matching_legacy_marker(legacy_barrier_store, writer_state):
    import sqlite3
    db, root = legacy_barrier_store
    _, prefix = db._token_marker_prefix()
    if writer_state == 'closed':
        db.close()
    elif writer_state == 'stopped':
        db._token_writer_stop = True
    else:
        db._token_writer_thread = SimpleNamespace(is_alive=lambda: False)
    legacy_root = root / '.session-lineage-locks'
    legacy_root.mkdir(mode=0o700, exist_ok=True)
    legacy_root.chmod(0o700)
    directory = legacy_root / 'accounting'
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    marker = directory / (prefix + 'fixture.lock')
    marker.write_bytes(b'legacy pending')
    before = marker.stat().st_mtime_ns
    with pytest.raises(lineage.SessionLineageInventoryError, match='legacy'):
        db.queue_token_counts('legacy-barrier', input_tokens=3)
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 0
    assert marker.read_bytes() == b'legacy pending' and marker.stat().st_mtime_ns == before


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('component,mode', [('root', 0o775), ('root', 0o757), ('accounting', 0o775), ('accounting', 0o757)])
def test_untrusted_empty_legacy_directory_blocks_queue(legacy_barrier_store, monkeypatch, component, mode):
    db, root = legacy_barrier_store
    legacy_root = root / '.session-lineage-locks'
    directory = legacy_root / 'accounting'
    directory.mkdir(parents=True)
    selected = legacy_root if component == 'root' else directory
    selected.chmod(mode)
    monkeypatch.setattr(db, '_token_writer_loop', lambda: None)
    with pytest.raises(lineage.SessionLineageInventoryError):
        db.queue_token_counts('legacy-barrier', input_tokens=3)
    assert db.has_pending_token_counts()
    assert not db._token_queue and not db._token_writer_uncertain
    assert selected.stat().st_mode & 0o777 == mode
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 0


@pytest.mark.parametrize('operation', ['update', 'auxiliary'])
def test_direct_usage_refuses_legacy_marker(legacy_barrier_store, operation):
    db, root = legacy_barrier_store
    marker = _legacy_marker(db, root)
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    with pytest.raises(lineage.SessionLineageInventoryError, match='legacy'):
        if operation == 'update':
            db.update_token_counts('legacy-barrier', input_tokens=3)
        else:
            db.record_auxiliary_usage('legacy-barrier', 'vision', input_tokens=3)
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 0
    assert db._conn.execute("SELECT COUNT(*) FROM session_model_usage").fetchone()[0] == 0
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before


def test_unreadable_legacy_inventory_reports_pending_and_refuses(legacy_barrier_store, monkeypatch):
    db, root = legacy_barrier_store
    directory = root / '.session-lineage-locks' / 'accounting'
    directory.mkdir(parents=True)
    original = os.scandir
    def deny(path):
        if Path(path) == directory:
            raise PermissionError('injected legacy inventory failure')
        return original(path)
    monkeypatch.setattr(os, 'scandir', deny)
    with pytest.raises(lineage.SessionLineageInventoryError):
        db.queue_token_counts('legacy-barrier', input_tokens=3)
    assert db.has_pending_token_counts()
    assert not db._token_queue and not db._token_writer_uncertain
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 0


@pytest.mark.platforms('posix')
def test_trusted_empty_legacy_directories_allow_usage_without_changing_modes(legacy_barrier_store, monkeypatch):
    db, root = legacy_barrier_store
    legacy_root = root / '.session-lineage-locks'
    directory = legacy_root / 'accounting'
    directory.mkdir(parents=True)
    legacy_root.chmod(0o755)
    directory.chmod(0o755)
    monkeypatch.setattr(db, '_token_writer_loop', lambda: None)
    db.queue_token_counts('legacy-barrier', input_tokens=3)
    db._token_writer_thread.join(5)
    assert db.flush_token_counts()
    db.update_token_counts('legacy-barrier', input_tokens=2)
    assert db.get_session('legacy-barrier')['input_tokens'] == 5
    assert legacy_root.stat().st_mode & 0o777 == 0o755
    assert directory.stat().st_mode & 0o777 == 0o755
    assert not list(directory.iterdir())


def test_previously_accepted_batch_is_not_replayed_or_dropped_when_legacy_marker_appears(legacy_barrier_store, monkeypatch):
    db, root = legacy_barrier_store
    monkeypatch.setattr(db, '_token_writer_loop', lambda: None)
    db.queue_token_counts('legacy-barrier', input_tokens=3)
    db._token_writer_thread.join(5)
    marker = _legacy_marker(db, root)
    assert db.flush_token_counts()
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 3
    assert db.has_pending_token_counts()
    with pytest.raises(lineage.SessionLineageInventoryError, match='legacy'):
        db.queue_token_counts('legacy-barrier', input_tokens=7)
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 3
    assert marker.read_bytes() == b'legacy pending'


def test_closed_queue_noop_without_legacy_inventory_remains_supported(legacy_barrier_store):
    import sqlite3
    db, root = legacy_barrier_store
    db.close()
    db.queue_token_counts('legacy-barrier', input_tokens=0)
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT input_tokens FROM sessions WHERE id='legacy-barrier'").fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM session_model_usage').fetchone()[0] == 0
    assert not (root / '.session-lineage-locks').exists()


@pytest.mark.parametrize('component', ['root', 'accounting'])
def test_malformed_legacy_directory_fails_closed_without_mutation(legacy_barrier_store, component):
    db, root = legacy_barrier_store
    legacy_root = root / '.session-lineage-locks'
    if component == 'root':
        path = legacy_root
    else:
        legacy_root.mkdir()
        path = legacy_root / 'accounting'
    path.write_bytes(b'not a directory')
    before = path.stat().st_mtime_ns
    with pytest.raises(lineage.SessionLineageInventoryError):
        db.queue_token_counts('legacy-barrier', input_tokens=3)
    assert db.has_pending_token_counts()
    assert path.read_bytes() == b'not a directory' and path.stat().st_mtime_ns == before
    assert not db._token_queue


def test_auxiliary_noop_stays_noop_with_legacy_marker(legacy_barrier_store):
    db, root = legacy_barrier_store
    marker = _legacy_marker(db, root)
    assert db.record_auxiliary_usage('legacy-barrier', '', input_tokens=3) is None
    assert marker.read_bytes() == b'legacy pending'
    assert db._conn.execute('SELECT count(*) FROM session_model_usage').fetchone()[0] == 0


def test_standalone_synchronous_usage_does_not_initialize_marker_identity(tmp_path, monkeypatch):
    import hermes_state_usage as usage

    managed_root = tmp_path / 'managed'
    managed_root.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(managed_root))
    outside = tmp_path / 'outside'
    outside.mkdir()
    with SessionDB(outside / 'state.db') as db:
        db.create_session('standalone', 'cli')
        def unexpected_identity(_db):
            raise AssertionError('standalone direct usage must not inspect marker identity')
        monkeypatch.setattr(usage, '_accounting_marker_identity', unexpected_identity)
        db.update_token_counts('standalone', input_tokens=3)
        db.record_auxiliary_usage('standalone', 'vision', input_tokens=2)
        assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='standalone'").fetchone()[0] == 3
        assert db._conn.execute("SELECT input_tokens FROM session_model_usage WHERE task='vision'").fetchone()[0] == 2
        assert not (outside / '.session-lineage-locks').exists()
        assert not (managed_root / '.session-lineage-locks').exists()
