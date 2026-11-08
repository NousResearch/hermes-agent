"""The local backend's identity, claim preconditions and the operations beyond the core set.

- The store's database file and owning profile are frozen when the ports open: a file link into
  another profile is refused there, a relative path never follows a later working directory, and
  a directory alias of the same profile is still the same profile.
- ``claim_next`` takes typed preconditions the backend checks inside the claim transaction (the
  validated head, the owner's admission gate), never a callback that is handed a connection.
- Each further operation is today's call behind the pin: refused for another agent's scope,
  the same rows as today for the store's own.
"""
import contextvars
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_scope import Group, Principal, Scope, push_state
from hermes_state import SessionDB
from hermes_state_port import ANY_HEAD, AdmissionGate, ClaimPreconditions, StoragePinRefused
from hermes_state_port_local import LocalSessionStorageProvider
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


def _open(db):
    return LocalSessionStorageProvider().open(db)


def test_a_database_file_linked_into_another_profile_is_refused_at_open(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    with closing(SessionDB(b / "state.db")) as victim:
        victim.create_session("secret", "cli")
        victim.append_messages_batch("secret", [{"role": "user", "content": "PROFILE_B_SECRET"}])
        (a / "state.db").symlink_to(b / "state.db")
        with closing(SessionDB(a / "state.db")) as db:
            with pytest.raises(StoragePinRefused, match="link"):
                _open(db)
        assert victim.get_session_title("secret") is None


def test_a_handle_opened_by_a_relative_path_is_refused(tmp_path, monkeypatch):
    """Which file a relative handle opened depends on a working directory that may since have
    moved, so the ports refuse it, whether wrapped before or after the move; the handle itself
    keeps working exactly as it always has."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.chdir(a)
    with closing(SessionDB(Path("state.db"))) as db:
        db.create_session("secret-a", "cli")
        with pytest.raises(StoragePinRefused, match="which file"):
            _open(db)
        monkeypatch.chdir(b)
        with pytest.raises(StoragePinRefused, match="which file"):
            _open(db)
        monkeypatch.chdir(a)
        assert db.get_session("secret-a")["id"] == "secret-a"


def test_a_handle_that_does_not_record_its_opened_file_is_refused():
    class Handle:
        db_path = Path("state.db")
    with pytest.raises(StoragePinRefused, match="which file"):
        _open(Handle())


def test_a_relative_scope_names_the_directory_it_resolves_to_now(tmp_path, monkeypatch):
    """A relative scope agent accepted in A names B's directory after the working directory
    moves to B, and is refused there, however it was resolved before."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "home").mkdir(parents=True), (b / "home").mkdir(parents=True)
    with closing(SessionDB(a / "home" / "state.db")) as db:
        storage = _open(db)
        db.create_session("secret-a", "cli")
        monkeypatch.chdir(a)
        with push_state(scope=Scope("local", "home", Principal("owner"))):
            assert storage.sessions.get("secret-a")["id"] == "secret-a"
        monkeypatch.chdir(b)
        with push_state(scope=Scope("local", "home", Principal("owner"))):
            with pytest.raises(StoragePinRefused):
                storage.sessions.get("secret-a")


def test_a_directory_alias_of_the_profile_is_the_same_profile(tmp_path):
    home = tmp_path / "real"
    home.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    with closing(SessionDB(alias / "state.db")) as db:
        storage = _open(db)
        with bind_profile_scope(home, hydrate_secrets=False):
            storage.sessions.create("s1", "cli")
        with bind_profile_scope(alias, hydrate_secrets=False):
            assert storage.sessions.get("s1")["id"] == "s1"


class _Gate(AdmissionGate):
    def __init__(self):
        self.closed = False
        self.calls = []

    def require_open(self, *args):
        self.calls.append(args)
        if self.closed:
            raise RuntimeStoreError("runtime_draining")


@pytest.fixture
def queued(tmp_path):
    """A session with two admissions queued under a fresh epoch: ``(db, storage, epoch, rows)``."""
    with closing(SessionDB(tmp_path / "state.db")) as db:
        storage = _open(db)
        with bind_profile_scope(tmp_path, hydrate_secrets=False):
            storage.sessions.create("s1", "cli")
            epoch = storage.admission.begin_epoch(instance_id="owner")
            rows = [storage.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id=r,
                                            payload={"text": r}) for r in ("one", "two")]
            yield db, storage, epoch, rows


def _state(db, storage):
    return [row["status"] for row in storage.admission.list("s1", pending_only=False)], \
        db.get_session("s1")["runtime_generation"]


def test_a_claim_whose_validated_head_moved_claims_nothing(queued):
    db, storage, epoch, rows = queued
    before = _state(db, storage)
    assert storage.admission.claim_next(epoch=epoch, session_id="s1",
                                        preconditions=ClaimPreconditions(head=rows[1]["admission_id"])) is False
    assert storage.admission.claim_next(epoch=epoch, session_id="s1",
                                        preconditions=ClaimPreconditions(head=None)) is False
    assert _state(db, storage) == before
    claimed = storage.admission.claim_next(epoch=epoch, session_id="s1",
                                           preconditions=ClaimPreconditions(head=rows[0]["admission_id"]))
    assert claimed["admission_id"] == rows[0]["admission_id"] and claimed["status"] == "started"


def test_the_gate_is_asked_with_no_connection_and_an_open_one_claims(queued):
    _db, storage, epoch, rows = queued
    gate = _Gate()
    claimed = storage.admission.claim_next(epoch=epoch, session_id="s1",
                                           preconditions=ClaimPreconditions(head=ANY_HEAD, gate=gate))
    assert claimed["admission_id"] == rows[0]["admission_id"]
    assert gate.calls and all(args == () for args in gate.calls)


def test_a_gate_that_closes_while_the_claim_waits_for_the_writer_refuses_it(queued, tmp_path):
    """The owner starts draining after its claim began waiting for the write lock: the claim is
    refused inside its transaction, the head stays queued and the generation does not move."""
    db, storage, epoch, rows = queued
    before = _state(db, storage)
    gate = _Gate()
    holder = sqlite3.connect(tmp_path / "state.db", timeout=0, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")  # another writer holds the lock; the claim waits behind it
    outcome = {}
    waiting = threading.Event()
    real_sleep = db._sleep_before_write_retry

    def sleep(*args):
        waiting.set()
        return real_sleep(*args)

    db._sleep_before_write_retry = sleep

    def claim():
        try:
            outcome["row"] = storage.admission.claim_next(
                epoch=epoch, session_id="s1",
                preconditions=ClaimPreconditions(head=rows[0]["admission_id"], gate=gate))
        except RuntimeStoreError as exc:
            outcome["refused"] = exc
    worker = threading.Thread(target=contextvars.copy_context().run, args=(claim,))
    worker.start()
    try:
        assert waiting.wait(10), "the claim never waited for the writer"
        gate.closed = True  # the drain begins while the claim waits
    finally:
        holder.rollback()
        holder.close()
    worker.join(10)
    assert "row" not in outcome and str(outcome["refused"].reason) == "runtime_draining"
    assert _state(db, storage) == before


def _foreign(home):
    return bind_state(Scope("acme", str(home), Principal("u"), owner=Group("g"), role="member"))


def test_every_further_operation_is_pinned(queued, tmp_path):
    db, storage, epoch, rows = queued
    s, a = storage.sessions, storage.admission
    calls = [
        lambda: s.replace_messages("s1", []), lambda: s.rewind_to_message("s1", 1),
        lambda: s.rewind_user_turn("s1", -1), lambda: s.set_user_message_content("s1", 1, "x"),
        lambda: s.deactivate_message("s1", 1), lambda: s.set_message_reaction("s1", 1, "x"),
        lambda: s.publish_compression_child(parent_session_id="s1", child_session_id="c", source="cli",
                                            messages=[]),
        lambda: s.archive_and_compact("s1", []), lambda: s.record_usage("s1", input_tokens=1),
        lambda: s.record_auxiliary_usage("s1", "title", input_tokens=1), lambda: s.flush_usage(),
        lambda: s.read_auxiliary_usage("s1"),
        lambda: a.steer(epoch=epoch, admission_id=rows[1]["admission_id"], generation=1),
        lambda: a.revert_steer(epoch=epoch, admission_id=rows[1]["admission_id"], generation=1),
        lambda: a.mutate_session(epoch=epoch, principal_id="u", session_id="s1", request_id="m",
                                 expected_revision=0, operation="rename", payload={"title": "t"}),
        lambda: a.register_worker(epoch=epoch, execution_id="w", session_id="s1", generation=0,
                                  kind="compute", adoption_secret="x"),
        lambda: a.adopt_worker(epoch=epoch, execution_id="w", session_id="s1", generation=0,
                               adoption_secret="x"),
        lambda: a.apply_worker_op(epoch=epoch, execution_id="w", session_id="s1", generation=0, sequence=1,
                                  operation="usage.main", payload={}),
    ]
    before = (_state(db, storage), db.get_messages("s1"), db.get_session("s1"))
    with _foreign(tmp_path):
        for call in calls:
            with pytest.raises(StoragePinRefused):
                call()
    assert (_state(db, storage), db.get_messages("s1"), db.get_session("s1")) == before


def test_transcript_usage_and_compression_operations_are_todays_writes(queued):
    db, storage, _epoch, _rows = queued
    s = storage.sessions
    s.append("s1", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}])
    first = db.get_messages("s1")[0]["id"]
    assert s.set_user_message_content("s1", first, "edited") == 1
    assert s.set_message_reaction("s1", first, "👍")
    s.record_usage("s1", queued=False, input_tokens=5, output_tokens=2)
    s.record_usage("s1", input_tokens=1)
    assert s.flush_usage()
    s.record_auxiliary_usage("s1", "title", model="m", input_tokens=3)
    assert s.read_auxiliary_usage("s1") == db.auxiliary_usage_by_task("s1")
    assert s.read_auxiliary_usage("s1")["title"]["input_tokens"] == 3
    assert db.get_session("s1")["input_tokens"] == 6
    assert s.rewind_to_message("s1", first)["rewound_count"] == 2
    assert [m["content"] for m in s.read_conversation("s1")] == []
    s.replace_messages("s1", [{"role": "user", "content": "again"}])
    assert s.archive_and_compact("s1", [{"role": "user", "content": "summary"}]) >= 1
    assert [m["content"] for m in s.read_conversation("s1")] == ["summary"]
    s.publish_compression_child(parent_session_id="s1", child_session_id="s2", source="cli",
                                messages=[{"role": "user", "content": "child"}],
                                require_compression_lease=False)
    assert db.get_session("s2")["parent_session_id"] == "s1"


def test_steering_settles_with_its_transfers(queued):
    _db, storage, epoch, rows = queued
    a = storage.admission
    claimed = a.claim_next(epoch=epoch, session_id="s1")
    gen = claimed["generation"]
    _row, made = a.steer(epoch=epoch, admission_id=rows[1]["admission_id"], generation=gen)
    assert made and a.steer(epoch=epoch, admission_id=rows[1]["admission_id"], generation=gen)[1] is False
    a.settle(epoch=epoch, admission_id=claimed["admission_id"], generation=gen, outcome="completed",
             transfers={rows[1]["admission_id"]: True})
    assert a.get(rows[1]["admission_id"])["outcome"] == "steered"


def test_runtime_mutations_and_worker_receipts_are_todays_ledger(queued):
    db, storage, epoch, rows = queued
    a = storage.admission
    for row in rows:
        a.cancel(epoch=epoch, admission_id=row["admission_id"])
    renamed = a.mutate_session(epoch=epoch, principal_id="human", session_id="s1", request_id="m",
                               expected_revision=db.get_session("s1")["runtime_revision"],
                               operation="rename", payload={"title": "named"})
    assert db.get_session("s1")["title"] == "named"
    assert a.mutate_session(epoch=epoch, principal_id="human", session_id="s1", request_id="m",
                            expected_revision=renamed["revision"] - 1, operation="rename",
                            payload={"title": "named"}) == renamed  # an exact retry replays
    gen = a.generation("s1")
    a.register_worker(epoch=epoch, execution_id="w", session_id="s1", generation=gen, kind="compute",
                      adoption_secret="secret")
    a.apply_worker_op(epoch=epoch, execution_id="w", session_id="s1", generation=gen, sequence=1,
                      operation="transcript.append", payload={"messages": [{"role": "user", "content": "w"}]})
    assert [m["content"] for m in db.get_messages("s1")] == ["w"]
    later = begin_runtime_epoch(db, instance_id="next")
    adopted = a.adopt_worker(epoch=later, execution_id="w", session_id="s1", generation=gen,
                             adoption_secret="secret")
    assert adopted["owner_epoch"] == later


def test_a_directory_alias_moved_before_wrapping_is_refused(tmp_path):
    """The handle opened A's file through an alias; the alias now names B. Wrapping must not label
    A's open connection as B's profile."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    with closing(SessionDB(b / "state.db")) as other:
        other.create_session("b-own", "cli")
    alias = tmp_path / "alias"
    alias.symlink_to(a, target_is_directory=True)
    with closing(SessionDB(alias / "state.db")) as db:
        db.create_session("secret-a", "cli")
        db.append_message("secret-a", "user", "A only")
        alias.unlink()
        alias.symlink_to(b, target_is_directory=True)
        with pytest.raises(StoragePinRefused, match="no longer names"):
            _open(db)
