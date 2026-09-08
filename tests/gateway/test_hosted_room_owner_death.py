"""Storage-level owner-death recovery: the descriptor column and the transactional proof guard.

Pure `gateway.hosted_room_driver` tests over a real temporary SQLite store. No runtime, no
sessions, no processes: every "death" here is a proof value handed to a transition, and what is
under test is which transitions accept it and which refuse.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms


ROOM_ID = "death-room"
PROFILE = "ops"
GATEWAY = "death-gateway"


def _local_domain() -> dict:
    from gateway import hosted_room_owner_probe

    domain = hosted_room_owner_probe.capture_local_domain()
    if domain is None:
        pytest.skip("this host cannot supply the Linux descriptor domain")
    return domain


@pytest.fixture
def store(tmp_path):
    """One admitted, descriptor-bearing attempt owned by the current lease."""
    db = tmp_path / "state.db"
    clock_value = [1_000.0]

    def clock() -> float:
        return clock_value[0]

    identity = state.TaskIdentity(ROOM_ID, "death-task", "thread-1", "turn-1")
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Death", authority_gateway_id=GATEWAY,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=clock())
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "offline", "source_event_seq": 1},
        clock=clock)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=GATEWAY, authority_epoch=1,
        process_generation="owner-generation", ttl_seconds=10_000, clock=clock)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    # The domain half must be this host's real one -- a descriptor from another boot or pid
    # namespace is correctly refused at capture, so it cannot be hand-written. Only the target
    # half (home, profile, the two session identifiers) is synthetic here.
    domain = _local_domain()
    native = {
        **domain, "home": "/tmp/home", "profile": PROFILE,
        "runtime_session_id": "runtime-1", "stored_session_key": "stored-1"}
    state.fence_task_admission(db, attempt, clock=clock, owner_descriptor=native)
    return {
        "db": db, "identity": identity, "lease": lease, "attempt": attempt, "clock": clock,
        "clock_value": clock_value, "native": native,
        # The complete descriptor exactly as admission stored it, for the negatives to damage.
        "native_full": state.get_task(db, identity)["owner_descriptor"]}


def _set_descriptor(db, identity, raw) -> None:
    """Write the raw column directly: only a stored value can be malformed."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_driver_tasks SET owner_descriptor=? WHERE room_id=? AND task_id=?",
            (raw, identity.room_id, identity.task_id))


def _make_indeterminate(store) -> dict:
    """Reach `indeterminate` through real transitions under the ORIGINAL owner's lease."""
    db, identity, lease, clock = store["db"], store["identity"], store["lease"], store["clock"]
    task = state.get_task(db, identity)
    state.begin_task_cancel(
        db, identity, cancel_id="stop-1",
        expected_cancel_generation=int(task["cancel_generation"]), clock=clock)
    task = state.get_task(db, identity)
    state.mark_stop_uncertain(
        db, identity, lease, expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]), clock=clock)
    return state.get_task(db, identity)


def _requeue(store, task, **extra):
    return state.requeue_indeterminate_task(
        store["db"], store["identity"], store["lease"],
        expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]), clock=store["clock"], **extra)


def _proof(store, task, **overrides) -> state.OwnerDeathProof:
    built = state.owner_death_proof(task, "ended")
    assert built is not None, "the fixture attempt must be able to produce a proof"
    return state.OwnerDeathProof(**{**vars(built), **overrides})


# --- the column ---------------------------------------------------------------

def test_admission_stamps_the_descriptor_and_the_stamp_together(store):
    task = state.get_task(store["db"], store["identity"])

    assert task["admitted_at"] is not None
    descriptor = task["owner_descriptor"]
    assert descriptor["cancel_generation_at_admission"] == store["attempt"].cancel_generation
    assert descriptor["execution_generation"] == store["attempt"].execution_generation
    assert descriptor["run_gateway_id"] == GATEWAY
    assert descriptor["run_process_generation"] == "owner-generation"
    assert descriptor["task_id"] == store["identity"].task_id


@pytest.mark.linux_only
@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "current"])
@pytest.mark.parametrize("drift", [-3600.0, 3600.0])
def test_live_owner_clock_drift_cannot_authorize_recovery(store, monkeypatch, legacy, drift):
    """Real test PID and SQLite evidence; only psutil's wall-clock metadata is shifted."""
    import os
    import psutil
    from psutil import _pslinux
    from gateway import hosted_room_owner_probe as probe
    from hermes_state_common import _proc_start_ticks

    task = _make_indeterminate(store)
    descriptor = dict(task["owner_descriptor"])
    if legacy:
        descriptor["descriptor_version"] = 1
        descriptor.pop("process_start_ticks", None)
        _set_descriptor(store["db"], store["identity"], json.dumps(descriptor))
        task = state.get_task(store["db"], store["identity"])
    assert descriptor["pid"] == os.getpid()
    ticks = _proc_start_ticks(os.getpid())
    assert ticks is not None
    assert probe.probe_owner_incarnation(descriptor) == probe.ALIVE
    epoch = psutil.Process(os.getpid()).create_time()
    boot_time = _pslinux.boot_time
    monkeypatch.setattr(_pslinux, "boot_time", lambda: boot_time() + drift)
    assert psutil.Process(os.getpid()).create_time() == pytest.approx(epoch + drift, rel=0, abs=0.001)
    assert _proc_start_ticks(os.getpid()) == ticks

    verdict = probe.probe_owner_incarnation(descriptor)
    proof = state.owner_death_proof(task, verdict)
    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=proof)
    assert verdict == (probe.UNKNOWN if legacy else probe.ALIVE)
    assert proof is None
    with pytest.raises(state.DriverValidationError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=task["execution_generation"],
            expected_cancel_generation=task["cancel_generation"], cancel_id="stop-1",
            clock=store["clock"], death_authorized=True, owner_death_proof=proof)
    with pytest.raises(state.DriverValidationError):
        state.resolve_indeterminate_task(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=task["execution_generation"],
            expected_cancel_generation=task["cancel_generation"], settlement_id="clock-drift",
            status="failed", result={"error": "unproven death"}, clock=store["clock"],
            death_authorized=True, owner_death_proof=proof)
    assert state.get_task(store["db"], store["identity"]) == task


def test_an_incomplete_native_half_stores_no_descriptor_at_all(tmp_path):
    """Never a partial map: the attempt is admitted, it simply gains no recovery eligibility."""
    db = tmp_path / "partial.db"
    identity = state.TaskIdentity(ROOM_ID, "partial-task", "thread-1", "turn-1")
    clock = lambda: 500.0
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Partial", authority_gateway_id=GATEWAY,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=clock())
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "offline", "source_event_seq": 1},
        clock=clock)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=GATEWAY, authority_epoch=1,
        process_generation="p", ttl_seconds=10_000, clock=clock)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)

    state.fence_task_admission(
        db, attempt, clock=clock, owner_descriptor={"pid": 1, "boot_id": "boot-1"})

    task = state.get_task(db, identity)
    assert task["admitted_at"] is not None, "capture failure must not refuse the admission"
    assert task["owner_descriptor"] is None


def test_start_and_requeue_both_clear_the_descriptor(store):
    task = _make_indeterminate(store)

    requeued = _requeue(store, task, owner_death_proof=_proof(store, task))

    assert requeued["status"] == "queued"
    assert requeued["owner_descriptor"] is None and requeued["admitted_at"] is None
    # A new generation starts clean even if a descriptor were somehow present.
    _set_descriptor(store["db"], store["identity"], json.dumps(store["native"]))
    state.start_task(
        store["db"], store["identity"], store["lease"],
        expected_cancel_generation=int(requeued["cancel_generation"]), clock=store["clock"])
    assert state.get_task(store["db"], store["identity"])["owner_descriptor"] is None


def test_an_older_store_migrates_with_null_descriptors(tmp_path):
    """The existing rebuild migration keeps rows, stored results AND every fence.

    The row is carried through real transitions first, so the migration is checked against durable
    work with a settled result, an advanced execution generation, an advanced cancellation
    generation and a recorded cancel id -- not against a bare queued prompt.
    """
    db = tmp_path / "legacy.db"
    identity = state.TaskIdentity(ROOM_ID, "legacy-task", "thread-1", "turn-1")
    clock = lambda: 700.0
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Legacy", authority_gateway_id=GATEWAY,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=clock())
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "offline", "source_event_seq": 1},
        clock=clock)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=GATEWAY, authority_epoch=1,
        process_generation="legacy-generation", ttl_seconds=10_000, clock=clock)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    state.fence_task_admission(db, attempt, clock=clock)
    state.begin_task_cancel(
        db, identity, cancel_id="legacy-stop", expected_cancel_generation=0, clock=clock)
    stopping = state.get_task(db, identity)
    state.settle_stopping_task(
        db, identity, lease,
        expected_execution_generation=int(stopping["execution_generation"]),
        expected_cancel_generation=int(stopping["cancel_generation"]),
        settlement_id="legacy-settlement", status="settled",
        result={"text": "durable work that predates the column"}, clock=clock)
    before = state.get_task(db, identity)
    assert before["status"] == "settled" and before["result"]["text"]
    # A faithful pre-column table: the real DDL with only `owner_descriptor` removed, so every
    # constraint and the hosted_rooms foreign key survive and the schema validator still applies.
    with sqlite3.connect(db) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='hosted_room_driver_tasks'").fetchone()[0]
        older = "\n".join(
            line for line in ddl.splitlines()
            if "owner_descriptor" not in line and not line.strip().startswith("--"))
        columns = ", ".join(
            column for column in state._TASK_COLUMN_ORDER if column != "owner_descriptor")
        conn.execute(older.replace("hosted_room_driver_tasks", "older_tasks", 1))
        conn.execute(f"INSERT INTO older_tasks ({columns}) "
                     f"SELECT {columns} FROM hosted_room_driver_tasks")
        conn.execute("DROP TABLE hosted_room_driver_tasks")
        conn.execute("ALTER TABLE older_tasks RENAME TO hosted_room_driver_tasks")
        # An older store has its status index too; without it the store reads as uninitialized
        # and the rebuild migration would never be the path under test.
        conn.execute(state._TASK_INDEX_SQL.format(if_not_exists=""))
    with sqlite3.connect(db) as conn:
        assert "owner_descriptor" not in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='hosted_room_driver_tasks'").fetchone()[0]

    task = state.get_task(db, identity)  # opening the store runs the rebuild migration

    assert task["owner_descriptor"] is None, "a migrated row must not invent a descriptor"
    # The durable work and every fence survive the rebuild unchanged.
    assert task["status"] == "settled"
    assert task["result"] == {"text": "durable work that predates the column"}
    assert task["settlement_id"] == "legacy-settlement"
    assert task["execution_generation"] == before["execution_generation"] == 1
    assert task["cancel_generation"] == before["cancel_generation"] == 1
    assert task["cancel_id"] == "legacy-stop"
    assert task["run_gateway_id"] == GATEWAY
    assert task["run_process_generation"] == "legacy-generation"
    assert task["admitted_at"] == before["admitted_at"] is not None
    assert task["payload"]["prompt"] == "offline"
    with sqlite3.connect(db) as conn:
        assert "owner_descriptor" in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='hosted_room_driver_tasks'").fetchone()[0]


# --- the requeue branches -----------------------------------------------------

def test_the_original_owner_may_still_requeue_a_legacy_row(store):
    """Unchanged behaviour for rows with no descriptor: no new privilege, no new refusal."""
    _set_descriptor(store["db"], store["identity"], None)
    task = _make_indeterminate(store)

    assert _requeue(store, task)["status"] == "queued"


def test_the_original_owner_cannot_requeue_a_descriptor_row_without_proof(store):
    """Process-generation equality is not cessation once a descriptor exists.

    This is the same lease and the same process generation that admitted the attempt — the exact
    case the old short-circuit allowed.
    """
    task = _make_indeterminate(store)

    with pytest.raises(state.InvalidTaskTransitionError):
        _requeue(store, task)

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate"


@pytest.mark.parametrize(
    "raw", ["not valid json", "null", "[1, 2]", '"a string"', "123", "", "{"])
def test_a_malformed_non_null_descriptor_refuses_the_same_generation_requeue(store, raw):
    """Decided on the RAW column: a corrupt value must not fall into the legacy branch.

    Same original process generation and a valid current lease, so nothing but the descriptor
    branch can produce the refusal.
    """
    task = _make_indeterminate(store)
    _set_descriptor(store["db"], store["identity"], raw)

    with pytest.raises(state.InvalidTaskTransitionError):
        _requeue(store, task)

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate"


@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "v2"])
def test_a_complete_proof_lets_the_attempt_be_requeued_once(store, legacy):
    import psutil
    from gateway import hosted_room_owner_probe as probe

    task = _make_indeterminate(store)
    descriptor = dict(task["owner_descriptor"])
    descriptor["pid"] = next(
        pid for pid in range(4_000_000, 4_100_000) if not psutil.pid_exists(pid))
    if legacy:
        descriptor["descriptor_version"] = 1
        descriptor.pop("process_start_ticks")
    _set_descriptor(store["db"], store["identity"], json.dumps(descriptor))
    task = state.get_task(store["db"], store["identity"])
    proof = state.owner_death_proof(task, probe.probe_owner_incarnation(descriptor))
    assert proof is not None

    requeued = _requeue(store, task, owner_death_proof=proof)

    assert requeued["status"] == "queued"
    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=proof)


# --- the proof guard ----------------------------------------------------------

def test_a_proof_is_only_built_from_a_complete_ended_row(store):
    task = state.get_task(store["db"], store["identity"])

    assert state.owner_death_proof(task, "alive") is None
    assert state.owner_death_proof(task, "unknown") is None
    assert state.owner_death_proof({**task, "owner_descriptor": None}, "ended") is None
    assert state.owner_death_proof({**task, "admitted_at": None}, "ended") is None
    assert state.owner_death_proof({**task, "run_gateway_id": None}, "ended") is None


@pytest.mark.parametrize(
    "override, why",
    [
        ({"verdict": "alive"}, "only an ended verdict authorizes anything"),
        ({"execution_generation": 99}, "proof reuse on a later attempt of the same task"),
        ({"admitted_at": 1.0}, "a stale admission stamp"),
        ({"run": ("other-gateway", "owner-generation", 1)}, "a foreign original run owner"),
        ({"run": ("death-gateway", "other-generation", 1)}, "a stale run process generation"),
        ({"descriptor": {"pid": 1}}, "a forged descriptor"),
        ({"identity": state.TaskIdentity(ROOM_ID, "other-task", "thread-1", "turn-1")},
         "cross-task proof reuse"),
    ])
def test_a_proof_that_does_not_bind_this_attempt_is_refused(store, override, why):
    task = _make_indeterminate(store)

    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=_proof(store, task, **override))

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate", why


# --- death-authorized terminal routes ----------------------------------------

def test_a_death_authorized_route_cannot_fall_through_without_its_proof(store):
    """A caller whose only evidence is death must carry the proof, or nothing commits."""
    task = _make_indeterminate(store)

    with pytest.raises(state.DriverValidationError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            cancel_id="stop-1", clock=store["clock"], death_authorized=True)
    # And the inverse: a proof handed to a route that did not declare itself is refused, so it
    # can never be silently ignored.
    with pytest.raises(state.DriverValidationError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            cancel_id="stop-1", clock=store["clock"],
            owner_death_proof=_proof(store, task))

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate"


def test_death_cancellation_requires_this_attempt_s_own_stop_intent(store):
    """A never-stopped dead attempt stays retryable; it is not cancelled by its owner's death."""
    task = state.get_task(store["db"], store["identity"])
    state.mark_stop_uncertain  # the row below never entered `stopping`
    with sqlite3.connect(store["db"]) as conn:
        conn.execute(
            "UPDATE hosted_room_driver_tasks SET status='indeterminate' "
            "WHERE room_id=? AND task_id=?", (ROOM_ID, store["identity"].task_id))
    task = state.get_task(store["db"], store["identity"])
    assert not task["cancel_id"] and int(task["cancel_generation"]) == 0

    with pytest.raises(state.StaleTaskError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            cancel_id="owner-death", clock=store["clock"], death_authorized=True,
            owner_death_proof=_proof(store, task))

    # ... and the same attempt IS requeueable, which is what "keep it retryable" means.
    assert _requeue(store, task, owner_death_proof=_proof(store, task))["status"] == "queued"


def test_a_stopped_attempt_with_a_dead_owner_resolves_as_its_recorded_stop(store):
    task = _make_indeterminate(store)
    assert task["cancel_id"] == "stop-1"

    resolved = state.resolve_indeterminate_cancellation(
        store["db"], store["identity"], store["lease"],
        expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]),
        cancel_id="stop-1", clock=store["clock"], death_authorized=True,
        owner_death_proof=_proof(store, task))

    assert resolved["status"] == "cancelled"
    assert resolved["cancel_id"] == "stop-1", "the recorded stop identity was replaced"


# --- F1: a parseable object is not a descriptor ------------------------------

_MISSING = object()


@pytest.mark.parametrize(
    "field, value, why",
    [
        # Native target half: the process probe validates the DOMAIN, so losing any of these
        # leaves something that still answers `ended` while proving nothing about the attempt.
        ("home", _MISSING, "no execution home"),
        ("profile", _MISSING, "no profile"),
        ("runtime_session_id", _MISSING, "no runtime handle"),
        ("stored_session_key", _MISSING, "no durable key"),
        ("home", "", "an empty execution home"),
        ("profile", 7, "a non-string profile"),
        ("process_start_ticks", _MISSING, "v2 requires a boot-relative stamp"),
        ("process_start_ticks", None, "unreadable boot-relative stamp"),
        ("process_start_ticks", True, "a bool is not a tick stamp"),
        ("process_start_ticks", 1.0, "a float is not a tick stamp"),
        ("process_start_ticks", "1", "a string is not a tick stamp"),
        ("process_start_ticks", 0, "a zero tick stamp"),
        ("process_start_ticks", -1, "a negative tick stamp"),
        ("descriptor_version", 2.0, "a float is not a version"),
        # Admission half.
        ("cancel_generation_at_admission", _MISSING, "no admission-cancel coordinate"),
        ("execution_generation", _MISSING, "no execution generation"),
        ("run_lease_generation", _MISSING, "no run lease generation"),
        ("room_id", _MISSING, "no room id"),
        ("task_id", _MISSING, "no task id"),
        ("thread_id", _MISSING, "no thread id"),
        ("turn_id", _MISSING, "no turn id"),
        ("run_gateway_id", _MISSING, "no original run gateway"),
        ("run_process_generation", _MISSING, "no original run process generation"),
        # Impersonation and coercion.
        ("execution_generation", True, "a bool is not a generation even though True == 1"),
        ("cancel_generation_at_admission", True, "a bool admission-cancel coordinate"),
        ("run_lease_generation", 1.0, "a float is not a generation even though 1.0 == 1"),
        ("execution_generation", "1", "a numeric string is not a generation"),
        ("run_lease_generation", "1", "a numeric string run lease generation"),
        ("task_id", 5, "a numeric task id"),
        # Ranges, not only types.
        ("execution_generation", 0, "a zero execution generation"),
        ("execution_generation", -1, "a negative execution generation"),
        ("run_lease_generation", 0, "a zero run lease generation"),
        ("run_lease_generation", -2, "a negative run lease generation"),
        ("cancel_generation_at_admission", -1, "a negative admission-cancel coordinate"),
        # Internal contradiction with the row it is stored on.
        ("task_id", "a-different-task", "a nested task id contradicting the row"),
        ("room_id", "a-different-room", "a nested room id contradicting the row"),
        ("execution_generation", 99, "a nested execution generation contradicting the row"),
        ("run_lease_generation", 99, "a nested run lease generation contradicting the row"),
        ("run_gateway_id", "other-gateway", "a nested run owner contradicting the row"),
        # A coordinate claiming a stop the row never reached.
        ("cancel_generation_at_admission", 99, "an admission-cancel above the row's current one"),
    ])
def test_an_incomplete_or_inconsistent_descriptor_authorizes_nothing(store, field, value, why):
    """F1: neither the ordinary Retry path nor a hand-built proof may use a deficient descriptor.

    The stored value stays a parseable non-NULL JSON object throughout, so the raw-NULL legacy
    branch is never what produces the refusal.
    """
    task = _make_indeterminate(store)
    broken = {k: v for k, v in store["native_full"].items() if not (k == field and value is _MISSING)}
    if value is not _MISSING:
        broken[field] = value
    _set_descriptor(store["db"], store["identity"], json.dumps(broken))
    damaged = state.get_task(store["db"], store["identity"])

    # The builder tells the truth about deficient evidence...
    built = state.owner_death_proof(damaged, "ended")
    # ... and the transaction is authoritative even for a proof built from the intact row.
    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=built or _proof(store, task))

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate", why
    assert state.get_task(store["db"], store["identity"])["admitted_at"] is not None


@pytest.mark.parametrize(
    "field, value, why",
    [
        ("pid", True, "a bool pid inside an otherwise equal map"),
        ("process_start_ticks", True, "a bool tick stamp"),
        ("process_start_ticks", 1.0, "a float tick stamp"),
        ("process_start_ticks", 1, "a substituted valid tick stamp must bind the stored one"),
        ("execution_generation", True, "a bool generation that dict equality accepts"),
        ("run_lease_generation", 1.0, "a float generation that dict equality accepts"),
        ("cancel_generation_at_admission", True, "a bool admission coordinate"),
    ])
def test_a_supplied_proof_descriptor_is_validated_not_merely_compared(store, field, value, why):
    """`{"pid": 1} == {"pid": True}` is True in Python, so equality alone is not a check."""
    task = _make_indeterminate(store)
    forged = {**store["native_full"], field: value}

    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=_proof(store, task, descriptor=forged))

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate", why


@pytest.mark.parametrize(
    "override, why",
    [
        ({"cancel_generation_at_admission": 99}, "an admission-cancel coordinate the row denies"),
        ({"cancel_generation_at_admission": True}, "a bool admission-cancel coordinate"),
        ({"execution_generation": True}, "a bool execution generation"),
        ({"execution_generation": 1.0}, "a float execution generation"),
        ({"admitted_at": float("inf")}, "an infinite admission stamp"),
        ({"admitted_at": 12}, "an int admission stamp is not the stored float"),
        ({"run": ("death-gateway", "owner-generation", True)}, "a bool run lease generation"),
        ({"run": ("death-gateway", "owner-generation")}, "a truncated run triple"),
        ({"verdict": "unknown"}, "an unknown verdict"),
    ])
def test_an_explicit_proof_coordinate_that_the_row_denies_is_refused(store, override, why):
    """F1: every coordinate the proof carries explicitly is bound, not just carried along.

    The stored descriptor is intact here, so only the proof's own field is wrong -- which a
    stored-vs-supplied comparison alone would not necessarily catch.
    """
    task = _make_indeterminate(store)

    with pytest.raises(state.DriverStateError):
        _requeue(store, task, owner_death_proof=_proof(store, task, **override))

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate", why


# --- F1 applies to every death-terminal sibling, not only requeue ------------

def _deficient_proof(store, task):
    """A proof whose stored descriptor lost a required native target field."""
    broken = {k: v for k, v in store["native_full"].items() if k != "home"}
    _set_descriptor(store["db"], store["identity"], json.dumps(broken))
    return state.OwnerDeathProof(**{**vars(_proof(store, task)), "descriptor": broken})


def test_a_deficient_descriptor_refuses_the_death_settle_sibling(store):
    """The deadline-failed death route uses the same guard as requeue."""
    task = _make_indeterminate(store)
    proof = _deficient_proof(store, task)

    with pytest.raises(state.DriverStateError):
        state.resolve_indeterminate_task(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            settlement_id="deadline:1", status="failed", result={"error": "timeout"},
            clock=store["clock"], death_authorized=True, owner_death_proof=proof)

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate"


def test_a_deficient_descriptor_refuses_the_death_cancel_sibling(store):
    task = _make_indeterminate(store)
    proof = _deficient_proof(store, task)

    with pytest.raises(state.DriverStateError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            cancel_id="stop-1", clock=store["clock"], death_authorized=True,
            owner_death_proof=proof)

    assert state.get_task(store["db"], store["identity"])["status"] == "indeterminate"


def test_a_deficient_descriptor_refuses_the_stopping_death_siblings(store):
    """`complete_task_cancel` and `settle_stopping_task` share the guard too."""
    task = state.get_task(store["db"], store["identity"])
    state.begin_task_cancel(
        store["db"], store["identity"], cancel_id="user-stop",
        expected_cancel_generation=int(task["cancel_generation"]), clock=store["clock"])
    stopping = state.get_task(store["db"], store["identity"])
    proof = _deficient_proof(store, stopping)

    with pytest.raises(state.DriverStateError):
        state.complete_task_cancel(
            store["db"], store["identity"], cancel_id="user-stop",
            expected_cancel_generation=int(stopping["cancel_generation"]), clock=store["clock"],
            death_authorized=True, owner_death_proof=proof)
    with pytest.raises(state.DriverStateError):
        state.settle_stopping_task(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(stopping["execution_generation"]),
            expected_cancel_generation=int(stopping["cancel_generation"]),
            settlement_id="deadline:1", status="failed", result={"error": "timeout"},
            clock=store["clock"], death_authorized=True, owner_death_proof=proof)

    assert state.get_task(store["db"], store["identity"])["status"] == "stopping"


def test_the_death_terminal_siblings_still_commit_under_a_complete_proof(store):
    """Positive control for the siblings: the guard binds, it does not block the route."""
    task = _make_indeterminate(store)

    resolved = state.resolve_indeterminate_task(
        store["db"], store["identity"], store["lease"],
        expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]),
        settlement_id=f"deadline:{int(task['execution_generation'])}", status="failed",
        result={"error": "timeout", "reason_code": "turn_deadline_exceeded"},
        clock=store["clock"], death_authorized=True, owner_death_proof=_proof(store, task))

    assert resolved["status"] == "failed"
    assert resolved["result"]["reason_code"] == "turn_deadline_exceeded"


# --- F2: the recorded Stop identity is not the caller's to replace -----------

@pytest.mark.parametrize("from_status", ["indeterminate", "deferred"])
def test_death_cancellation_cannot_replace_the_recorded_stop_id(store, from_status):
    """F2: under a valid proof and current lease, a different cancel id must still refuse."""
    task = _make_indeterminate(store)
    if from_status == "deferred":
        task = state.defer_indeterminate_task(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            reason="offline-fixture", clock=store["clock"])
    assert task["status"] == from_status and task["cancel_id"] == "stop-1"

    with pytest.raises(state.StaleTaskError):
        state.resolve_indeterminate_cancellation(
            store["db"], store["identity"], store["lease"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            cancel_id="replacement-stop", clock=store["clock"], death_authorized=True,
            owner_death_proof=_proof(store, task), from_status=from_status)

    unchanged = state.get_task(store["db"], store["identity"])
    assert unchanged["status"] == from_status
    assert unchanged["cancel_id"] == "stop-1", "the recorded stop identity was replaced"

    # The recorded id still resolves, so this is a binding check and not a broken route.
    resolved = state.resolve_indeterminate_cancellation(
        store["db"], store["identity"], store["lease"],
        expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]),
        cancel_id="stop-1", clock=store["clock"], death_authorized=True,
        owner_death_proof=_proof(store, task), from_status=from_status)
    assert resolved["status"] == "cancelled" and resolved["cancel_id"] == "stop-1"


def test_the_ordinary_acknowledged_cancellation_stays_valid_with_a_descriptor(store):
    """Native acknowledgement is unaffected by the descriptor: the owner's backend is alive."""
    task = state.get_task(store["db"], store["identity"])
    state.begin_task_cancel(
        store["db"], store["identity"], cancel_id="user-stop",
        expected_cancel_generation=int(task["cancel_generation"]), clock=store["clock"])
    stopping = state.get_task(store["db"], store["identity"])

    completed = state.complete_task_cancel(
        store["db"], store["identity"], cancel_id="user-stop",
        expected_cancel_generation=int(stopping["cancel_generation"]), clock=store["clock"])

    assert completed["status"] == "cancelled"
    assert completed["owner_descriptor"] is not None, "an ordinary Stop clears no evidence"
