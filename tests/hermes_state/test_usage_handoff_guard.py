"""Accounting ownership stays fenced through commit and rejected batches remain pending."""
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    a, b = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    yield a, b
    try:
        a.close()
    except TimeoutError:
        pass  # The rejected queue remains attached to this disposable handle.
    b.close()


def test_update_and_usage_row_commit_together(stores):
    a, _ = stores
    a.update_token_counts("s", source="cli", model="m", billing_provider="p", input_tokens=7, api_call_count=1)
    assert a.get_session("s")["input_tokens"] == 7
    assert tuple(a._read_one("SELECT model,billing_provider,input_tokens FROM session_model_usage WHERE session_id='s'")) == ("m", "p", 7)


def test_adoption_waits_for_accounting_transaction(stores, monkeypatch):
    a, b = stores
    a.create_session("s", "cli")
    entered, release, adopted = threading.Event(), threading.Event(), threading.Event()
    errors, outcome = [], []
    original = a._update_token_counts_in_transaction
    def paused(conn, sid, **values):
        entered.set()
        assert release.wait(10)
        return original(conn, sid, **values)
    monkeypatch.setattr(a, "_update_token_counts_in_transaction", paused)
    def update():
        try:
            a.update_token_counts("s", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
        except Exception as exc:
            errors.append(exc)
    def transfer():
        try:
            outcome.append(b.adopt_session_lineage_from(a, "s"))
        except Exception as exc:
            errors.append(exc)
        finally:
            adopted.set()
    writer = threading.Thread(target=update)
    mover = threading.Thread(target=transfer)
    writer.start()
    assert entered.wait(5)
    mover.start()
    assert not adopted.wait(0.2)
    release.set()
    writer.join(10)
    mover.join(10)
    assert not writer.is_alive() and not mover.is_alive()
    assert not errors, errors
    assert outcome[0]["donor_retired"], outcome
    assert b.get_session("s")["input_tokens"] == 7


def test_queued_foreign_owner_refusal_keeps_pending_and_close_open(stores):
    a, b = stores
    a.create_session("s", "cli")
    a._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
    a.queue_token_counts("s", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
    a._token_writer_thread = None
    # A legacy/external split owner may exist after acceptance; never drop its queued delta.
    a._conn.execute("UPDATE sessions SET ended_at=2,end_reason='adopted_by_profile' WHERE id='s'")
    b._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES ('s','cli',1,'science')")
    a._conn.commit()
    b._conn.commit()
    before = tuple(a._read_one("SELECT input_tokens,model,billing_provider FROM sessions WHERE id='s'"))
    assert a.flush_token_counts() is False
    assert len(a._token_queue) == 1
    assert tuple(a._read_one("SELECT input_tokens,model,billing_provider FROM sessions WHERE id='s'")) == before
    with pytest.raises(TimeoutError):
        a.close()
    assert a._conn is not None and len(a._token_queue) == 1


def test_queued_retry_does_not_replay_committed_prefix(stores, monkeypatch):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    a._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
    a.queue_token_counts("one", input_tokens=3, api_call_count=1)
    a.queue_token_counts("two", input_tokens=7, api_call_count=1)
    a._token_writer_thread = None
    original = a.update_token_counts
    def fail_second(sid, **kwargs):
        if sid == "two":
            raise SessionIdOwnedElsewhereError("fixture refusal")
        return original(sid, **kwargs)
    monkeypatch.setattr(a, "update_token_counts", fail_second)
    assert a.flush_token_counts() is False
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='one'")[0] == 3
    assert len(a._token_queue) == 1
    monkeypatch.setattr(a, "update_token_counts", original)
    assert a.flush_token_counts()
    assert a.get_session("one")["input_tokens"] == 3
    assert a.get_session("two")["input_tokens"] == 7


def test_new_enqueue_after_real_adoption_refuses_without_accepting(stores):
    a, b = stores
    a.create_session("s", "cli")
    assert b.adopt_session_lineage_from(a, "s")["donor_retired"]
    with pytest.raises(SessionIdOwnedElsewhereError):
        a.queue_token_counts("s", input_tokens=7)
    assert not a.has_pending_token_counts()


def test_accounting_failure_rolls_back_placeholder_and_usage(stores, monkeypatch):
    a, _ = stores
    def fail(*args, **kwargs):
        raise RuntimeError("injected usage write failure")
    monkeypatch.setattr(a, "_record_model_usage", fail)
    with pytest.raises(RuntimeError, match="injected"):
        a.update_token_counts("new", source="cli", model="m", billing_provider="p",
                              input_tokens=7, api_call_count=1)
    assert a._read_one("SELECT id FROM sessions WHERE id='new'") is None
    assert a._read_one("SELECT session_id FROM session_model_usage WHERE session_id='new'") is None


def test_enqueue_waits_for_actual_adoption_then_refuses(stores, monkeypatch):
    a, b = stores
    a.create_session("s", "cli")
    entered, release, enqueue_done = threading.Event(), threading.Event(), threading.Event()
    original = b._adopt_lineage_segments_locked
    outcomes, errors = [], []
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(b, "_adopt_lineage_segments_locked", paused)
    def move():
        outcomes.append(b.adopt_session_lineage_from(a, "s"))
    def enqueue():
        try:
            a.queue_token_counts("s", input_tokens=7, api_call_count=1)
        except Exception as exc:
            errors.append(exc)
        finally:
            enqueue_done.set()
    mover = threading.Thread(target=move)
    writer = threading.Thread(target=enqueue)
    mover.start()
    assert entered.wait(5)
    writer.start()
    assert not enqueue_done.wait(0.2)
    release.set()
    mover.join(10)
    writer.join(10)
    assert not mover.is_alive() and not writer.is_alive()
    assert outcomes[0]["donor_retired"], outcomes
    assert len(errors) == 1 and isinstance(errors[0], SessionIdOwnedElsewhereError)
    assert not a.has_pending_token_counts()


def test_accounting_uses_transcript_write_patience(stores, monkeypatch):
    a, _ = stores
    calls = []
    original = a._execute_write
    def observed(fn, **kwargs):
        calls.append(kwargs.get("patience_s"))
        return original(fn, **kwargs)
    monkeypatch.setattr(a, "_execute_write", observed)
    a.update_token_counts("s", source="cli", input_tokens=1)
    assert calls == [a._TRANSCRIPT_WRITE_PATIENCE_S]


def test_enqueue_after_failed_close_stays_behind_retained_batch(stores, monkeypatch):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    a._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
    a.queue_token_counts("one", input_tokens=3)
    a._token_writer_thread = None
    original = a.update_token_counts
    def refuse_one(sid, **kwargs):
        if sid == "one":
            raise SessionIdOwnedElsewhereError("fixture refusal")
        return original(sid, **kwargs)
    monkeypatch.setattr(a, "update_token_counts", refuse_one)
    with pytest.raises(TimeoutError):
        a.close()
    a.queue_token_counts("two", input_tokens=7)
    assert [sid for sid, _ in a._token_queue] == ["one", "two"]
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='two'")[0] == 0
    monkeypatch.setattr(a, "update_token_counts", original)
    assert a.flush_token_counts()
    assert (a.get_session("one")["input_tokens"], a.get_session("two")["input_tokens"]) == (3, 7)


def test_other_handle_pending_refuses_adoption_until_committed(stores):
    a, b = stores
    a.create_session("s", "cli")
    with SessionDB(a.db_path) as other:
        other._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
        other.queue_token_counts("s", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
        other._token_writer_thread = None
        assert a.has_pending_token_counts()
        refused = b.adopt_session_lineage_from(a, "s")
        assert not refused.get("adopted") and not refused.get("donor_retired"), refused
        assert other.flush_token_counts()
        assert not a.has_pending_token_counts()
        done = b.adopt_session_lineage_from(a, "s")
        assert done["donor_retired"], done
        assert b.get_session("s")["input_tokens"] == 7


def test_failed_batch_keeps_cross_handle_marker_until_retry(stores, monkeypatch):
    a, _ = stores
    a.create_session("s", "cli")
    with SessionDB(a.db_path) as other:
        other._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
        other.queue_token_counts("s", input_tokens=7)
        other._token_writer_thread = None
        original = other.update_token_counts
        def refuse(*args, **kwargs):
            raise SessionIdOwnedElsewhereError("fixture")
        monkeypatch.setattr(other, "update_token_counts", refuse)
        assert not other.flush_token_counts()
        assert a.has_pending_token_counts()
        with pytest.raises(TimeoutError):
            other.close()
        assert a.has_pending_token_counts()
        monkeypatch.setattr(other, "update_token_counts", original)
        assert other.flush_token_counts()
        assert not a.has_pending_token_counts()
        assert a.get_session("s")["input_tokens"] == 7


def test_separate_process_pending_refuses_adoption_until_drained(stores):
    import os
    import subprocess
    import sys

    a, b = stores
    a.create_session("s", "cli")
    program = """
from types import SimpleNamespace
import sys
from hermes_state import SessionDB
from pathlib import Path
db = SessionDB(Path(sys.argv[1]))
db._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
db.queue_token_counts("s", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
db._token_writer_thread = None
print("READY", flush=True)
sys.stdin.readline()
assert db.flush_token_counts()
db.close()
print("DONE", flush=True)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", program, str(a.db_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=os.environ.copy())
    try:
        ready, lines = threading.Event(), []
        def read_ready():
            lines.append(child.stdout.readline().strip())
            ready.set()
        threading.Thread(target=read_ready, daemon=True).start()
        assert ready.wait(10), "child did not report readiness"
        line = lines[0]
        if line != "READY":
            _out, err = child.communicate(timeout=5)
            pytest.fail(f"child setup failed: {err}")
        refused = b.adopt_session_lineage_from(a, "s")
        assert not refused.get("adopted") and not refused.get("donor_retired"), refused
        child.stdin.write("drain\n")
        child.stdin.flush()
        out, err = child.communicate(timeout=15)
        assert child.returncode == 0 and "DONE" in out, err
        done = b.adopt_session_lineage_from(a, "s")
        assert done["donor_retired"], done
        assert b.get_session("s")["input_tokens"] == 7
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def test_marker_identity_failure_refuses_before_acceptance(stores, monkeypatch):
    from hermes_state_lineage import SessionLineageInventoryError

    a, _ = stores
    a.create_session("s", "cli")
    def fail():
        raise SessionLineageInventoryError("fixture marker inventory failure")
    monkeypatch.setattr(a, "_token_marker_prefix", fail)
    with pytest.raises(SessionLineageInventoryError):
        a.queue_token_counts("s", input_tokens=7)
    assert not a._token_queue
    assert a.has_pending_token_counts()


def test_hardlinked_store_fails_closed(stores, tmp_path):
    import os
    from hermes_state_lineage import SessionLineageInventoryError

    a, _ = stores
    a.create_session("s", "cli")
    os.link(a.db_path, tmp_path / "alias.db")
    with pytest.raises(SessionLineageInventoryError):
        a.queue_token_counts("s", input_tokens=1)
    assert not a._token_queue
    assert a.has_pending_token_counts()



def test_unsafe_marker_inventory_fails_closed(stores):
    a, _ = stores
    directory, prefix = a._token_marker_prefix()
    (directory / (prefix + "unsafe.lock")).symlink_to(a.db_path)
    assert a.has_pending_token_counts()


def test_marker_inventory_bound_fails_closed(stores, monkeypatch):
    import hermes_state_usage
    a, _ = stores
    _directory, prefix = a._token_marker_prefix()
    class Scan:
        def __enter__(self):
            return iter(SimpleNamespace(name=f"{prefix}{n}.lock") for n in range(8193))
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(hermes_state_usage.os, "scandir", lambda *_: Scan())
    assert a.has_pending_token_counts()


def test_store_replacement_refuses_new_enqueue(stores, monkeypatch):
    from hermes_state_lineage import SessionLineageInventoryError
    a, _ = stores
    a.create_session("s", "cli")
    monkeypatch.setattr(a, "_db_file_was_replaced", lambda: True)
    with pytest.raises(SessionLineageInventoryError):
        a.queue_token_counts("s", input_tokens=1)
    assert not a._token_queue


def test_marker_tracks_same_path_replacement_and_same_inode_rename(tmp_path, monkeypatch):
    import os
    from types import MethodType
    from hermes_state_usage import SessionUsageMixin

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = tmp_path / "state.db"
    original.write_bytes(b"original")
    def fake(path):
        st = os.stat(path)
        obj = SimpleNamespace(db_path=path, _db_file_identity=(st.st_dev, st.st_ino),
                              _db_file_was_replaced=lambda: False, _token_pending_marker=None,
                              _token_pending_marker_path=None, _token_queue=[], _token_writer_busy=False,
                              _token_writer_uncertain=[], _token_queue_cond=threading.Condition(threading.Lock()))
        obj._token_marker_prefix = MethodType(SessionUsageMixin._token_marker_prefix, obj)
        return obj
    old = fake(original)
    SessionUsageMixin._hold_token_marker(old)
    try:
        renamed = tmp_path / "renamed.db"
        original.rename(renamed)
        original.write_bytes(b"replacement")
        assert SessionUsageMixin.has_pending_token_counts(fake(original))
        assert SessionUsageMixin.has_pending_token_counts(fake(renamed))
    finally:
        old._token_pending_marker.__exit__(None, None, None)


def test_ordinary_lineage_lock_files_do_not_count_toward_marker_bound(stores, tmp_path):
    a, _ = stores
    lock_root = tmp_path / ".session-lineage-locks"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_root.chmod(0o700)
    for n in range(8193):
        (lock_root / f"ordinary-{n}.lock").touch()
    assert not a.has_pending_token_counts()


def test_repeated_drains_remove_only_own_idle_marker(stores):
    a, _ = stores
    a.create_session("s", "cli")
    directory, _ = a._token_marker_prefix()
    for _ in range(20):
        a.queue_token_counts("s", input_tokens=1)
        assert a.flush_token_counts()
        assert not list(directory.iterdir())
    assert a.get_session("s")["input_tokens"] == 20


def test_marker_unlink_failure_keeps_pending_guard(stores, monkeypatch):
    from pathlib import Path

    a, b = stores
    a.create_session("s", "cli")
    a._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
    a.queue_token_counts("s", input_tokens=1)
    a._token_writer_thread = None
    original = Path.unlink
    marker_path = a._token_pending_marker_path
    def fail_marker(path, *args, **kwargs):
        if path == marker_path:
            raise PermissionError("fixture sharing violation")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_marker)
    assert not a.flush_token_counts()
    assert a.has_pending_token_counts()
    assert b.adopt_session_lineage_from(a, "s")["donor_retired"] is False
    with pytest.raises(TimeoutError):
        a.close()
    monkeypatch.setattr(Path, "unlink", original)
    assert a.flush_token_counts()
    assert not a.has_pending_token_counts()


def test_raw_advisory_default_creates_and_no_create_refuses_missing(tmp_path):
    from hermes_state_lineage import _advisory_lock_file, SessionLineageInventoryError

    created = tmp_path / "created.lock"
    with _advisory_lock_file(created, 0):
        assert created.is_file()
    missing = tmp_path / "missing.lock"
    with pytest.raises(SessionLineageInventoryError):
        with _advisory_lock_file(missing, 0, create=False):
            pass
    assert not missing.exists()


def test_marker_disappearing_during_scan_does_not_reappear(stores, monkeypatch):
    from contextlib import contextmanager
    import hermes_state_lineage as lineage

    a, _ = stores
    directory, prefix = a._token_marker_prefix()
    marker = directory / (prefix + "fixture.lock")
    marker.touch()
    original = lineage._advisory_lock_file
    @contextmanager
    def disappear(path, timeout, **kwargs):
        path.unlink()
        with original(path, timeout, **kwargs) as fd:
            yield fd
    monkeypatch.setattr(lineage, "_advisory_lock_file", disappear)
    assert a.has_pending_token_counts()
    assert not marker.exists()


def test_marker_directory_identity_churn_fails_closed(stores, monkeypatch):
    import hermes_state_usage

    a, _ = stores
    with monkeypatch.context() as scoped:
        scoped.setattr(hermes_state_usage.os.path, "samestat", lambda *_: False)
        assert a.has_pending_token_counts()


@pytest.mark.parametrize("replacement", [False, True])
def test_held_marker_missing_or_replaced_refuses_new_enqueue(stores, replacement):
    from hermes_state_lineage import SessionLineageInventoryError

    a, _ = stores
    a.create_session("s", "cli")
    a._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
    a.queue_token_counts("s", input_tokens=3)
    a._token_writer_thread = None
    marker = a._token_pending_marker_path
    marker.unlink()  # simulate external loss; never a normal marker owner action
    if replacement:
        marker.touch()
    with pytest.raises(SessionLineageInventoryError):
        a.queue_token_counts("s", input_tokens=7)
    assert len(a._token_queue) == 1
    if replacement:
        # The accepted first delta commits, but the foreign replacement is not unlinked.
        assert not a.flush_token_counts()
        assert marker.exists()
        marker.unlink()  # fixture owns the replacement; allow the known-idle handle to close
    assert a.flush_token_counts()
    assert a.get_session("s")["input_tokens"] == 3


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after_commit", [False, True])
def test_public_flush_interrupt_retains_uncertain_item_without_replaying_prefix(stores, monkeypatch, interrupt, after_commit):
    a, _ = stores
    for sid in ("one", "two", "three"):
        a.create_session(sid, "cli")
    # Public queue acceptance, with a stopped writer so the caller performs the drain.
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    for sid, tokens in (("one", 3), ("two", 7), ("three", 11)):
        a.queue_token_counts(sid, input_tokens=tokens, api_call_count=1)
        a._token_writer_thread.join(5)
    original = a.update_token_counts
    def interrupted(sid, **kwargs):
        if sid == "two":
            if after_commit:
                original(sid, **kwargs)
            raise interrupt()
        return original(sid, **kwargs)
    monkeypatch.setattr(a, "update_token_counts", interrupted)
    with pytest.raises(interrupt):
        a.flush_token_counts()
    assert not a._token_writer_busy
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='one'")[0] == 3
    assert [sid for sid, _ in a._token_writer_uncertain] == ["two"]
    assert [sid for sid, _ in a._token_queue] == ["three"]
    assert a.has_pending_token_counts()
    with SessionDB(a.db_path) as observer:
        assert observer.has_pending_token_counts()  # another handle sees the held uncertainty marker
    monkeypatch.setattr(a, "update_token_counts", original)
    assert not a.flush_token_counts()  # no automatic replay of unknown-effect item or its suffix
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='one'")[0] == 3
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='two'")[0] == (7 if after_commit else 0)
    with pytest.raises(TimeoutError):
        a.close()


def test_exception_after_real_commit_is_retained_unknown_not_replayed(stores, monkeypatch):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    for sid, tokens in (("one", 3), ("two", 7)):
        a.queue_token_counts(sid, input_tokens=tokens, api_call_count=1)
        a._token_writer_thread.join(5)
    original = a.update_token_counts
    def commit_then_error(sid, **kwargs):
        original(sid, **kwargs)
        if sid == "two":
            raise OSError("fixture unknown acknowledgement")
    monkeypatch.setattr(a, "update_token_counts", commit_then_error)
    assert not a.flush_token_counts()
    assert not a._token_writer_busy
    assert [sid for sid, _ in a._token_writer_uncertain] == ["two"]
    assert not a._token_queue
    monkeypatch.setattr(a, "update_token_counts", original)
    assert not a.flush_token_counts()
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='one'")[0] == 3
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='two'")[0] == 7
    with pytest.raises(TimeoutError):
        a.close()


def test_waiting_flush_observes_background_unknown_and_does_not_drain_suffix(stores, monkeypatch):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    entered, release = threading.Event(), threading.Event()
    original = a.update_token_counts
    def commit_then_wait_and_error(sid, **kwargs):
        original(sid, **kwargs)
        if sid == "one":
            entered.set()
            assert release.wait(10)
            raise OSError("fixture acknowledgement lost")
    monkeypatch.setattr(a, "update_token_counts", commit_then_wait_and_error)
    a.queue_token_counts("one", input_tokens=3)
    assert entered.wait(5)
    a.queue_token_counts("two", input_tokens=7)
    result = []
    waiter = threading.Thread(target=lambda: result.append(a.flush_token_counts(timeout=10)))
    waiter.start()
    release.set()
    waiter.join(10)
    assert not waiter.is_alive() and result == [False]
    assert [sid for sid, _ in a._token_writer_uncertain] == ["one"]
    assert [sid for sid, _ in a._token_queue] == ["two"]
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='one'")[0] == 3
    assert a._read_one("SELECT input_tokens FROM sessions WHERE id='two'")[0] == 0
    with pytest.raises(TimeoutError):
        a.close()


def test_interrupt_at_claimed_batch_entry_keeps_work_uncertain(stores, monkeypatch):
    a, _ = stores
    a.create_session("s", "cli")
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    a.queue_token_counts("s", input_tokens=7)
    a._token_writer_thread.join(5)
    def interrupted(_batch):
        raise KeyboardInterrupt()
    monkeypatch.setattr(a, "_apply_claimed_batch", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            a.flush_token_counts()
        assert not a._token_writer_busy
        assert [sid for sid, _ in a._token_writer_uncertain] == ["s"]
        assert not a.flush_token_counts()
    finally:
        # Avoid hanging the disposable pre-fix RED fixture.
        with a._token_queue_cond:
            if a._token_writer_busy:
                a._token_writer_busy = False


def test_interrupt_immediately_after_claim_is_retained_unknown(stores, monkeypatch):
    a, _ = stores
    a.create_session("s", "cli")
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    a.queue_token_counts("s", input_tokens=7)
    a._token_writer_thread.join(5)
    original = a._claim_token_batch_locked
    def interrupted():
        original()
        raise KeyboardInterrupt()
    monkeypatch.setattr(a, "_claim_token_batch_locked", interrupted)
    with pytest.raises(KeyboardInterrupt):
        a.flush_token_counts()
    assert not a._token_writer_busy
    assert [sid for sid, _ in a._token_writer_uncertain] == ["s"]
    assert not a._token_queue and not a.flush_token_counts()


def test_interrupt_during_close_drain_retains_unknown_and_handle(stores, monkeypatch):
    a, _ = stores
    a.create_session("s", "cli")
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    a.queue_token_counts("s", input_tokens=7)
    a._token_writer_thread.join(5)
    def interrupted(_batch):
        raise KeyboardInterrupt()
    monkeypatch.setattr(a, "_apply_claimed_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        a.close()
    assert not a._token_writer_busy
    assert [sid for sid, _ in a._token_writer_uncertain] == ["s"]
    assert a._conn is not None
    with pytest.raises(TimeoutError):
        a.close()


@pytest.mark.parametrize("operation", ["flush", "close"])
def test_public_drain_does_not_report_success_with_new_accepted_tail(stores, monkeypatch, operation):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    monkeypatch.setattr(a, "_token_writer_loop", lambda: None)
    a.queue_token_counts("one", input_tokens=3)
    a._token_writer_thread.join(5)
    entered, release = threading.Event(), threading.Event()
    original = a.update_token_counts
    def paused(sid, **kwargs):
        original(sid, **kwargs)
        if sid == "one":
            entered.set()
            assert release.wait(10)
    monkeypatch.setattr(a, "update_token_counts", paused)
    outcomes = []
    def drain():
        try:
            outcomes.append(a.flush_token_counts() if operation == "flush" else a.close())
        except TimeoutError:
            outcomes.append(False)
    worker = threading.Thread(target=drain)
    worker.start()
    assert entered.wait(5)
    a.queue_token_counts("two", input_tokens=7)
    release.set()
    worker.join(10)
    assert not worker.is_alive() and len(outcomes) == 1
    # Success is allowed only when the accepted successor is durably drained.
    if outcomes[0] is not False:
        assert not a._token_queue and not a._token_writer_busy
    else:
        assert a._conn is not None and a.has_pending_token_counts()


def test_background_worker_continues_after_accepted_successor(stores, monkeypatch):
    a, _ = stores
    a.create_session("one", "cli")
    a.create_session("two", "cli")
    entered, release = threading.Event(), threading.Event()
    original = a.update_token_counts
    def paused(sid, **kwargs):
        original(sid, **kwargs)
        if sid == "one":
            entered.set()
            assert release.wait(10)
    monkeypatch.setattr(a, "update_token_counts", paused)
    a.queue_token_counts("one", input_tokens=3)
    assert entered.wait(5)
    a.queue_token_counts("two", input_tokens=7)
    release.set()
    assert a.flush_token_counts(timeout=10)
    assert a.get_session("one")["input_tokens"] == 3
    assert a.get_session("two")["input_tokens"] == 7
