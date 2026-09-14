"""Stage-3 acceptance harness scenarios (B/F/P/L/R/PR/E per design doc).

Hermetic: temporary SQLite DB, stub spawn_fn, real kernel APIs, no live DB,
no systemd, no real subprocess, no network. RED/GREEN: these tests run
against the reference wiring in hermes_cli.kanban_wired_dispatcher; the live
dispatcher is not touched.
"""

from __future__ import annotations

import random
import sqlite3
import uuid
from pathlib import Path

import pytest

from hermes_cli import kanban_wired_dispatcher as wiring
from hermes_cli.kanban_admission import AdmissionPolicy
from hermes_cli.kanban_launch_protocol import (
    acquire_dispatcher_owner,
    canonical_board_identity,
)
from hermes_cli.kanban_profiles import LANE_PROFILE_NAMES
from hermes_cli.kanban_route_contract import RouteContract
from tests.hermes_cli.kanban_harness_invariants import assert_holds, check_all

BOARD_UUID = "11111111-1111-4111-8111-111111111111"
OWNER_GENERATION = 7
POLICY_GENERATION = 12
LANES = tuple(LANE_PROFILE_NAMES)
_EXECUTION = {
    "planning": "read_only",
    "research": "read_only",
    "coding": "remote_isolated",
    "review": "remote_isolated",
    "security": "remote_isolated",
    "ops": "remote_approval_gated",
}


def _task_id(n: int) -> str:
    return f"t_{n:016x}"


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.db_path = tmp_path / "board.db"
        self.db_path.touch()
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.board = wiring.HarnessBoard(self.conn, POLICY_GENERATION)
        self.identity = canonical_board_identity(self.db_path, BOARD_UUID)
        self.home = tmp_path / "home"
        run = self.home / "run"
        run.mkdir(parents=True)
        run.chmod(0o700)
        self.home.chmod(0o755)
        self.lease = acquire_dispatcher_owner(self.home, OWNER_GENERATION)
        self.spawn_log: list[dict] = []
        self.evidence_log: list[dict] = []
        self.lifecycle_log: list[dict] = []  # shared across dispatchers
        # Immutable spawn-invocation audit log: unlike spawn_log (the liveness
        # source, cleared to simulate worker death) this is never cleared.
        self.invocations: list[dict] = []
        self.next_pid = 1000
        self._route_hash: str | None = None

    def contract(self) -> RouteContract:
        if self._route_hash is None:
            self._route_hash = self._build_contract().hash
        return self._build_contract()

    def _build_contract(self) -> RouteContract:
        raw = {
            "routing_contract_version": 1,
            "default_assignee": "",
            "orchestrator_profile": "kanban-orchestrator",
            "lanes": {
                lane: {
                    "profile": LANE_PROFILE_NAMES[lane],
                    "execution": _EXECUTION[lane],
                }
                for lane in LANES
            },
        }
        return RouteContract.from_mapping(raw)

    def policy(self, **overrides: object) -> AdmissionPolicy:
        values: dict[str, object] = {
            "schema_version": 1,
            "generation": POLICY_GENERATION,
            "mode": "open",
            "paused": False,
            "admissions": [],
        }
        values.update(overrides)
        return AdmissionPolicy.from_mapping(values)

    def spawn_ok(self, call) -> int:
        self.next_pid += 1
        entry = {
            "task_id": call.task_id,
            "pid": self.next_pid,
            "profile": call.profile,
            "run_generation": call.run_generation,
        }
        self.spawn_log.append(entry)  # mutable liveness source (cleared on death)
        self.invocations.append(dict(entry))  # immutable audit log
        return self.next_pid

    def spawn_raises(self, _call) -> int:
        raise RuntimeError("simulated spawn failure")

    def dispatcher(self, **kwargs: object) -> wiring.WiredDispatcher:
        defaults: dict[str, object] = {
            "board": self.board,
            "board_identity": self.identity,
            "owner_lease": self.lease,
            "owner_generation": OWNER_GENERATION,
            "policy": self.policy(),
            "route_contract": self.contract(),
            "spawn_fn": self.spawn_ok,
            "spawn_log": self.spawn_log,
            "expected_route_hash": self._route_hash,
        }
        defaults.update(kwargs)
        d = wiring.WiredDispatcher(**defaults)
        if "lifecycle_log" not in kwargs:
            # Bind the dispatcher's lifecycle history to the shared harness
            # log so multiple dispatcher instances see one history.
            d.lifecycle_log = self.lifecycle_log
        return d

    def holds(self, quiesced: bool = True) -> None:
        assert_holds(
            self.conn,
            self.invocations,
            self.evidence_log,
            quiesced=quiesced,
            lifecycle_log=self.lifecycle_log,
        )

    def note(self, task_id: str, **kv: object) -> None:
        entry = {"task_id": task_id}
        entry.update(kv)
        self.evidence_log.append(entry)


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


# ---------------- B: basic lifecycle ----------------

def test_b1_happy_path_reaches_spawned(harness: Harness) -> None:
    harness.board.add_task(_task_id(1), lane="coding")
    d = harness.dispatcher()
    report = d.tick()
    assert report.spawned == [_task_id(1)]
    row = harness.conn.execute(
        "SELECT state, pid FROM kanban_launch_protocol"
    ).fetchone()
    assert row["state"] == "spawned"
    assert row["pid"] == 1001
    d.worker_outcome(_task_id(1), 0, "ok")
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_b2_spawn_failure_releases_claim(harness: Harness) -> None:
    harness.board.add_task(_task_id(2), lane="research")
    report = harness.dispatcher(spawn_fn=harness.spawn_raises).tick()
    assert report.spawned == []
    assert report.released == [_task_id(2)]
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    assert harness.board.tasks()[0]["status"] == "ready"
    harness.holds()


def test_b3_worker_completes(harness: Harness) -> None:
    harness.board.add_task(_task_id(3), lane="planning")
    d = harness.dispatcher()
    d.tick()
    d.worker_outcome(_task_id(3), 0, "complete")
    assert harness.board.tasks()[0]["status"] == "done"
    harness.holds()


def test_b4_worker_blocks(harness: Harness) -> None:
    harness.board.add_task(_task_id(4), lane="ops")
    d = harness.dispatcher()
    d.tick()
    d.worker_outcome(_task_id(4), 3, "worker blocked")
    assert harness.board.tasks()[0]["status"] == "blocked"
    harness.holds()


# ---------------- F: freeze ----------------

def test_f1_freeze_before_tick_spawns_nothing(harness: Harness) -> None:
    harness.board.add_task(_task_id(5), lane="coding")
    d = harness.dispatcher(policy=harness.policy(mode="frozen"))
    report = d.tick()
    assert report.spawned == []
    assert report.aborted == [_task_id(5)]
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_f2_freeze_inside_window_aborts_at_intent(harness: Harness) -> None:
    from hermes_cli.kanban_launch_protocol import (
        freeze_ack_blockers,
        release_claim,
    )

    harness.board.add_task(_task_id(6), lane="coding")

    def flip() -> None:
        harness.board.set_policy_generation(POLICY_GENERATION + 1)

    d = harness.dispatcher(mid_spawn_hook=flip)
    report = d.tick()
    assert report.spawned == []
    assert harness.spawn_log == []  # abort BEFORE spawn: no process ever
    assert _task_id(6) in report.aborted
    # The kernel refuses to release a claim against a different active
    # generation: the row persists as freeze-ack evidence, no process was
    # created (fail closed), and the blocker is visible to the controller.
    assert _task_id(6) in report.unresolved
    assert len(freeze_ack_blockers(harness.conn)) == 1
    # Controller recovery: restore the generation, then the claim releases.
    harness.board.set_policy_generation(POLICY_GENERATION)
    blocker = freeze_ack_blockers(harness.conn)[0]
    release_claim(
        harness.conn,
        board=harness.identity,
        task_id=blocker.task_id,
        run_generation=blocker.run_generation,
        dispatcher_owner_generation=blocker.dispatcher_owner_generation,
        policy_generation=blocker.policy_generation,
        route_generation=blocker.route_generation,
        claim_token=blocker.claim_token,
        owner_lease=harness.lease,
    )
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_f3_freeze_while_running_leaves_workers_alone(harness: Harness) -> None:
    harness.board.add_task(_task_id(7), lane="coding")
    d = harness.dispatcher()
    d.tick()
    assert harness.spawn_log  # running
    frozen = harness.dispatcher(policy=harness.policy(mode="frozen"))
    report = frozen.tick()
    assert report.spawned == []
    row = harness.conn.execute("SELECT state FROM kanban_launch_protocol").fetchone()
    assert row["state"] == "spawned"  # untouched
    d.worker_outcome(_task_id(7), 0)
    harness.holds()


def test_f4_unfreeze_resumes(harness: Harness) -> None:
    harness.board.add_task(_task_id(8), lane="coding")
    frozen = harness.dispatcher(policy=harness.policy(mode="frozen"))
    assert frozen.tick().spawned == []
    d = harness.dispatcher()
    frozen = harness.dispatcher(policy=harness.policy(mode="frozen"))
    assert frozen.tick().spawned == []
    report = d.tick()
    assert report.spawned == [_task_id(8)]
    d.worker_outcome(_task_id(8), 0)
    harness.holds()


# ---------------- P: pause ----------------

def test_p1_pause_empty_tick_no_reclaim(harness: Harness) -> None:
    harness.board.add_task(_task_id(9), lane="coding")
    paused = harness.dispatcher(policy=harness.policy(paused=True))
    report = paused.tick()
    assert report.spawned == [] and report.reclaimed == []
    assert harness.board.tasks()[0]["status"] == "ready"
    harness.holds()


def test_p2_pause_mid_window_aborts(harness: Harness) -> None:
    harness.board.add_task(_task_id(10), lane="coding")

    def flip() -> None:
        harness.board.set_policy_generation(POLICY_GENERATION + 1)

    report = harness.dispatcher(mid_spawn_hook=flip).tick()
    assert report.spawned == []
    assert harness.invocations == []  # no process created (abort before spawn)
    assert _task_id(10) in report.aborted
    harness.board.set_policy_generation(POLICY_GENERATION)
    from hermes_cli.kanban_launch_protocol import freeze_ack_blockers, release_claim

    blocker = freeze_ack_blockers(harness.conn)[0]
    release_claim(
        harness.conn,
        board=harness.identity,
        task_id=blocker.task_id,
        run_generation=blocker.run_generation,
        dispatcher_owner_generation=blocker.dispatcher_owner_generation,
        policy_generation=blocker.policy_generation,
        route_generation=blocker.route_generation,
        claim_token=blocker.claim_token,
        owner_lease=harness.lease,
    )
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_p3_pause_stops_reclaim_frozen_does_not(harness: Harness) -> None:
    harness.board.add_task(_task_id(60), lane="coding")
    d = harness.dispatcher()
    d.tick()
    harness.spawn_log.clear()  # worker dead
    # Frozen: reclaim still runs (only spawns stop).
    frozen = harness.dispatcher(policy=harness.policy(mode="frozen"))
    frozen_report = frozen.tick()
    assert _task_id(60) in frozen_report.reclaimed
    # Reset: respawn, kill again, then pause must NOT reclaim.
    harness.dispatcher().tick()
    harness.spawn_log.clear()
    paused = harness.dispatcher(policy=harness.policy(paused=True))
    paused_report = paused.tick()
    assert paused_report.reclaimed == []
    assert paused_report.spawned == []
    # Pause leaves the dead worker's ledger row unresolved BY DESIGN (no
    # reclaim): it stays as controller-visible evidence until unpaused.
    harness.holds(quiesced=False)
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 1
    # Unpausing reclaims it and the lifecycle closes.
    resumed = harness.dispatcher()
    assert resumed.tick().reclaimed == [_task_id(60)]
    resumed.worker_outcome(_task_id(60), 0) if harness.board.tasks()[0]["status"] == "running" else None
    harness.holds()



def test_f5b_flip_after_spawned_cas(harness: Harness) -> None:
    # Disturbance between the spawned CAS commit and the final guard: the
    # ledger reaches 'spawned' under the old generation, then the pointer
    # flips. The guard must flag it: aborted+unresolved with terminal
    # lifecycle evidence, process removed from liveness, row left as a
    # freeze_ack blocker for the controller (kernel refuses to mutate it).
    from hermes_cli.kanban_launch_protocol import freeze_ack_blockers

    harness.board.add_task(_task_id(71), lane="coding")
    d = harness.dispatcher()

    real_spawned = wiring.record_spawned

    def spawn_cas_then_flip(conn, **kw):
        result = real_spawned(conn, **kw)
        harness.board.set_policy_generation(POLICY_GENERATION + 1)
        return result

    wiring.record_spawned = spawn_cas_then_flip
    try:
        report = d.tick()
    finally:
        wiring.record_spawned = real_spawned
    assert report.spawned == []
    assert _task_id(71) in report.aborted
    assert _task_id(71) in report.unresolved
    # The kernel's freeze_ack_blockers lists only PRE-spawn rows; a spawned
    # row under a revoked generation is not silently droppable — it persists
    # visibly in the ledger until the controller drains it after freeze
    # acknowledgment (generation-scoped APIs refuse to mutate it).
    assert freeze_ack_blockers(harness.conn) == []
    row = harness.conn.execute("SELECT state FROM kanban_launch_protocol").fetchone()
    assert row is not None and row["state"] == "spawned"
    # Lifecycle: created, then aborted with terminal evidence.
    events = [e["event"] for e in harness.lifecycle_log if e["task_id"] == _task_id(71)]
    assert events == ["spawned", "aborted"]
    aborted_ev = [e for e in harness.lifecycle_log if e["task_id"] == _task_id(71) and e["event"] == "aborted"][0]
    assert aborted_ev["exit_status"] == 1
    assert "policy flipped" in aborted_ev["detail"]
    # Process removed from liveness: a later tick never counts it alive.
    assert harness.spawn_log == []
    # Bounded evidence holds for the aborted event.
    residual = [
        v
        for v in check_all(
            harness.conn,
            harness.invocations,
            harness.evidence_log,
            quiesced=False,
            lifecycle_log=harness.lifecycle_log,
        )
        if not v.startswith("fail_closed")
    ]
    assert residual == []


def test_f5_spawn_stub_flips_policy_after_creation(harness: Harness) -> None:
    # Post-creation disturbance: the stub records the process, THEN revokes
    # the active policy generation. The ledger must never reach 'spawned'
    # under the revoked generation, and the lifecycle must show spawned
    # followed by aborted (fail-closed evidence of the torn-down launch).
    harness.board.add_task(_task_id(70), lane="coding")

    def spawn_then_flip(call) -> int:
        harness.next_pid += 1
        entry = {
            "task_id": call.task_id,
            "pid": harness.next_pid,
            "profile": call.profile,
            "run_generation": call.run_generation,
        }
        harness.spawn_log.append(entry)
        harness.invocations.append(dict(entry))
        harness.board.set_policy_generation(POLICY_GENERATION + 1)
        return harness.next_pid

    d = harness.dispatcher(spawn_fn=spawn_then_flip)
    report = d.tick()
    assert report.spawned == []
    assert _task_id(70) in report.aborted
    # Ledger row remains at spawn_intent (a freeze_ack blocker), never
    # 'spawned'.
    state = harness.conn.execute(
        "SELECT state FROM kanban_launch_protocol"
    ).fetchone()["state"]
    assert state == "spawn_intent"
    # Lifecycle shows the real order: process created, then aborted.
    events = [
        e["event"]
        for e in harness.lifecycle_log
        if e["task_id"] == _task_id(70)
    ]
    assert events == ["spawned", "aborted"]
    violations = check_all(
        harness.conn,
        harness.invocations,
        harness.evidence_log,
        quiesced=False,
        lifecycle_log=harness.lifecycle_log,
    )
    assert any(v.startswith("fail_closed") for v in violations)
    # Controller recovery: the created (stub) process is terminated —
    # remove it from the liveness source — and the lifecycle is closed.
    harness.board.set_policy_generation(POLICY_GENERATION)
    harness.spawn_log.clear()  # stub process terminated
    from hermes_cli.kanban_launch_protocol import freeze_ack_blockers, release_claim

    blocker = freeze_ack_blockers(harness.conn)[0]
    release_claim(
        harness.conn,
        board=harness.identity,
        task_id=blocker.task_id,
        run_generation=blocker.run_generation,
        dispatcher_owner_generation=blocker.dispatcher_owner_generation,
        policy_generation=blocker.policy_generation,
        route_generation=blocker.route_generation,
        claim_token=blocker.claim_token,
        owner_lease=harness.lease,
    )
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    # The pathological path legitimately leaves a permanent fail_closed
    # violation in the history (that is its purpose); verify the remaining
    # invariants still hold with fail_closed excluded.
    residual = [
        v
        for v in check_all(
            harness.conn,
            harness.invocations,
            harness.evidence_log,
            quiesced=False,
            lifecycle_log=harness.lifecycle_log,
        )
        if not v.startswith("fail_closed")
    ]
    assert residual == []


def test_l1_lease_loss_rejects_old_owner(harness: Harness) -> None:
    from hermes_cli.kanban_launch_protocol import (
        LaunchProtocolError,
        claim_not_spawned,
        record_spawn_intent,
    )

    harness.board.add_task(_task_id(11), lane="coding")
    claim_not_spawned(
        harness.conn,
        board=harness.identity,
        task_id=_task_id(11),
        run_generation=1,
        dispatcher_owner_generation=OWNER_GENERATION,
        policy_generation=POLICY_GENERATION,
        route_generation=1,
        claim_token="claim_token_0123456789abcdef",
    )
    old = harness.dispatcher()
    harness.lease.release()  # old owner loses the lock
    acquire_dispatcher_owner(harness.home, OWNER_GENERATION + 1)
    # Stale lease is rejected by the kernel lease validation itself.
    with pytest.raises(LaunchProtocolError):
        record_spawn_intent(
            harness.conn,
            board=harness.identity,
            task_id=_task_id(11),
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=1,
            claim_token="claim_token_0123456789abcdef",
            owner_lease=old.lease,
        )
    # The old dispatcher's own tick cannot create any process either.
    report = old.tick()
    assert report.spawned == []
    assert harness.invocations == []
    # Unresolved claim persists as controller-visible evidence.
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 1
    harness.holds(quiesced=False)


def test_l2_ttl_expiry_heartbeat_wins(harness: Harness) -> None:
    harness.board.add_task(_task_id(65), lane="coding")
    d = harness.dispatcher()
    d.tick()
    # Worker alive (pid present in spawn log): no reclaim even after many
    # ticks — the harness liveness source is the spawn log.
    for _ in range(3):
        report = d.tick()
        assert report.reclaimed == []
    assert harness.board.tasks()[0]["status"] == "running"
    harness.holds(quiesced=False)


def test_l3_dead_worker_reclaimed_once(harness: Harness) -> None:
    harness.board.add_task(_task_id(12), lane="coding")
    d = harness.dispatcher()
    d.tick()
    harness.spawn_log.clear()  # simulate worker death
    report = d.tick()
    assert report.reclaimed == [_task_id(12)]
    # Reclaim bumped run_generation: the respawn is a FRESH lifecycle
    # identity, so the no-double-spawn invariant holds over the full
    # lifecycle history even though the task executed twice in total.
    row = harness.board.tasks()[0]
    assert row["run_generation"] == 2
    assert row["status"] == "running"
    generations = {
        e["run_generation"]
        for e in d.lifecycle_log
        if e["task_id"] == _task_id(12) and e["event"] == "spawned"
    }
    assert generations == {1, 2}
    d.worker_outcome(_task_id(12), 0)
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_r1_valid_lane_routes_to_profile(harness: Harness) -> None:
    harness.board.add_task(_task_id(13), lane="security")
    d = harness.dispatcher()
    report = d.tick()
    assert report.spawned == [_task_id(13)]
    assert harness.spawn_log[0]["profile"] == LANE_PROFILE_NAMES["security"]
    d.worker_outcome(_task_id(13), 0)
    harness.holds()


def test_r1b_explicit_assignee_is_authoritative(harness: Harness) -> None:
    harness.board.conn.execute(
        "INSERT INTO harness_tasks (task_id, status, lane, assignee) "
        "VALUES (?, 'ready', 'coding', 'kanban-research')",
        (_task_id(19),),
    )
    d = harness.dispatcher()
    report = d.tick()
    assert report.spawned == [_task_id(19)]
    assert harness.spawn_log[0]["profile"] == "kanban-research"
    d.worker_outcome(_task_id(19), 0)
    harness.holds()


def test_r2_unknown_lane_triages(harness: Harness) -> None:
    harness.board.add_task(_task_id(14), lane="nonexistent")
    report = harness.dispatcher().tick()
    assert report.spawned == []
    assert report.triaged == [_task_id(14)]
    assert harness.board.tasks()[0]["status"] == "triage"
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_r3_contract_hash_mismatch_refuses(harness: Harness) -> None:
    harness.board.add_task(_task_id(15), lane="coding")
    bad = RouteContract.invalid()
    report = harness.dispatcher(route_contract=bad).tick()
    assert report.spawned == []
    assert report.refused_profiles == [_task_id(15)]
    assert harness.board.tasks()[0]["status"] == "ready"  # untouched
    harness.holds()
    # Hash integrity: identical raw contract parses to the same hash; any
    # content change produces a different hash (tamper detection basis).
    good = harness.contract()
    same = RouteContract.from_mapping(
        {
            "routing_contract_version": 1,
            "default_assignee": "",
            "orchestrator_profile": "kanban-orchestrator",
            "lanes": {
                lane: {
                    "profile": LANE_PROFILE_NAMES[lane],
                    "execution": _EXECUTION[lane],
                }
                for lane in LANES
            },
        }
    )
    assert good.hash == same.hash and len(good.hash) == 64
    tampered_raw = {
        "routing_contract_version": 1,
        "default_assignee": "",
        "orchestrator_profile": "kanban-orchestrator",
        "lanes": {
            lane: {
                "profile": LANE_PROFILE_NAMES[lane],
                "execution": _EXECUTION[lane],
            }
            for lane in LANES
        },
    }
    tampered_raw["lanes"]["planning"] = {
        "profile": "kanban-planning",
        "execution": "remote_isolated",
    }
    tampered = RouteContract.from_mapping(tampered_raw)
    assert tampered.valid and tampered.hash != good.hash
    # Pinned-hash refusal: a VALID but re-hashed contract must be refused
    # because its hash no longer matches the dispatcher's pinned hash.
    pinned = harness.dispatcher(expected_route_hash=good.hash)
    harness.board.add_task(_task_id(21), lane="coding")
    mismatch_report = pinned_with(pinned, tampered).tick()
    assert mismatch_report.spawned == []
    assert _task_id(21) in mismatch_report.refused_profiles


def test_pr1_registry_valid_spawn_proceeds(harness: Harness) -> None:
    harness.board.add_task(_task_id(16), lane="review")
    d = harness.dispatcher()
    assert d.tick().spawned == [_task_id(16)]
    assert harness.spawn_log[0]["profile"] == "kanban-review"
    d.worker_outcome(_task_id(16), 0)
    harness.holds()


def test_pr2_registry_tampering_refuses_spawn(harness: Harness, monkeypatch) -> None:
    harness.board.add_task(_task_id(17), lane="review")
    from hermes_cli import kanban_profiles as kp

    original = dict(kp.PROFILES["kanban-review"].toolsets)
    object.__setattr__(
        kp.PROFILES["kanban-review"],
        "toolsets",
        {t: None for t in original if t != "terminal"},
    )
    try:
        report = harness.dispatcher().tick()
        assert report.spawned == []
        assert report.refused_profiles == [_task_id(17)]
    finally:
        from types import MappingProxyType

        object.__setattr__(
            kp.PROFILES["kanban-review"],
            "toolsets",
            MappingProxyType(original),
        )
        kp.validate_registry()  # prove restoration
    harness.holds()


def test_pr3_unknown_profile_blocked(harness: Harness) -> None:
    harness.board.add_task(_task_id(18), lane="coding")
    contract = harness.contract()
    assert contract.valid
    stripped = {
        name: profile
        for name, profile in wiring.PROFILES.items()
        if name != LANE_PROFILE_NAMES["coding"]
    }
    original_profiles = wiring.PROFILES
    wiring.PROFILES = stripped
    try:
        report = harness.dispatcher().tick()
        assert report.spawned == []
    finally:
        wiring.PROFILES = original_profiles
    harness.holds()
    assert report.refused_profiles == [_task_id(18)]
    assert harness.board.tasks()[0]["status"] == "blocked"
    harness.holds()


def pinned_with(d, contract):
    d.route_contract = contract
    return d


# ---------------- E: end-to-end ----------------

def test_e1_full_loop_three_tasks(harness: Harness) -> None:
    for i, lane in enumerate(("coding", "research", "ops")):
        harness.board.add_task(_task_id(20 + i), lane=lane)
    d = harness.dispatcher()
    report = d.tick()
    assert len(report.spawned) == 3
    for i in range(3):
        d.worker_outcome(_task_id(20 + i), 0)
    assert all(t["status"] == "done" for t in harness.board.tasks())
    assert harness.conn.execute("SELECT COUNT(*) c FROM kanban_launch_protocol").fetchone()["c"] == 0
    harness.holds()


def test_e2_mid_run_freeze_bounds_spawns(harness: Harness) -> None:
    for i in range(3):
        harness.board.add_task(_task_id(30 + i), lane="coding")
    d = harness.dispatcher()
    d.tick()
    assert len(harness.spawn_log) == 3
    frozen = harness.dispatcher(policy=harness.policy(mode="frozen"))
    assert frozen.tick().spawned == []
    assert len(harness.spawn_log) == 3  # no duplicates
    for i in range(3):
        d.worker_outcome(_task_id(30 + i), 0)
    harness.holds()


def test_e3_crash_and_reclaim(harness: Harness) -> None:
    harness.board.add_task(_task_id(40), lane="coding")
    d = harness.dispatcher()
    d.tick()
    harness.spawn_log.clear()  # simulate worker death
    report = d.tick()
    assert report.reclaimed == [_task_id(40)]
    assert harness.board.tasks()[0]["run_generation"] == 2  # fresh lifecycle
    generations = {
        e["run_generation"]
        for e in harness.lifecycle_log
        if e["task_id"] == _task_id(40) and e["event"] == "spawned"
    }
    assert generations == {1, 2}
    d.worker_outcome(_task_id(40), 0)
    assert harness.board.tasks()[0]["status"] == "done"
    harness.holds()


def test_e4_soak_lite_seeded(harness: Harness) -> None:
    rng = random.Random(20260912)
    tasks = [_task_id(50 + i) for i in range(6)]
    for i, t in enumerate(tasks):
        harness.board.add_task(t, lane=LANES[i % len(LANES)])
    d = harness.dispatcher()
    completed = 0
    for tick in range(200):
        mode = "open"
        if rng.random() < 0.15:
            mode = "frozen"
        policy = harness.policy(mode=mode)
        d.policy = policy
        report = d.tick()
        for task_id in report.spawned:
            harness.note(task_id)
        # workers finish stochastically
        running = harness.board.tasks("running")
        for row in running:
            if rng.random() < 0.3:
                d.worker_outcome(row["task_id"], 0)
                completed += 1
        violations = __import__(
            "tests.hermes_cli.kanban_harness_invariants",
            fromlist=["check_all"],
        ).check_all(harness.conn, harness.invocations, harness.evidence_log, quiesced=False, lifecycle_log=harness.lifecycle_log)
        assert violations == [], f"tick {tick}: {violations}"
        if completed == len(tasks):
            break
    assert completed == len(tasks)
    harness.holds()
