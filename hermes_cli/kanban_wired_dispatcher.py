"""Reference wiring of the Kanban admission kernel into a dispatcher tick.

Stage-3 harness support module. This is NOT the live dispatcher: it is the
paper design from ``kanban-stage1-wiring-inventory-2026-09-12.md`` expressed
as executable code so the acceptance scenarios in
``test_kanban_stage3_harness.py`` can prove the kernel primitives compose
fail-closed under every disturbance class.

Purity constraints (same as the kernel):
- no live DB, no systemd, no real subprocess, no network;
- every mutation goes through the kernel launch-protocol API;
- the admission gate is evaluated twice per spawn path (pre-tick and
  inside the claim->intent window) exactly as the inventory requires.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from hermes_cli.kanban_admission import (
    AdmissionPolicy,
    AdmissionRequest,
)
from hermes_cli.kanban_launch_protocol import (
    DispatcherOwnerLease,
    LaunchProtocolError,
    claim_not_spawned,
    install_launch_schema,
    record_run_outcome,
    record_spawn_intent,
    record_spawned,
    release_claim,
)
from hermes_cli.kanban_profiles import (
    LANE_PROFILE_NAMES,
    PROFILES,
    validate_registry,
)
from hermes_cli.kanban_route_contract import RouteContract, resolve_route

READY_STATUSES = ("ready",)
TERMINAL_STATUSES = ("done", "blocked", "triage")


@dataclass
class SpawnCall:
    """One stub spawn invocation record."""

    task_id: str
    profile: str
    env: dict
    pid: int
    run_generation: int = 1


@dataclass
class TickReport:
    """Bounded per-tick evidence for invariant checking."""

    spawned: list[str] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    reclaimed: list[str] = field(default_factory=list)
    triaged: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    refused_profiles: list[str] = field(default_factory=list)
    spawn_errors: list[str] = field(default_factory=list)


class HarnessBoard:
    """Hermetic fixture: launch schema + policy pointer + minimal task table."""

    def __init__(self, conn: sqlite3.Connection, policy_generation: int) -> None:
        self.conn = conn
        install_launch_schema(conn)
        pointer = conn.execute(
            "SELECT singleton FROM kanban_policy_pointer WHERE singleton = 1"
        ).fetchone()
        if pointer is None:
            conn.execute(
                "INSERT INTO kanban_policy_pointer "
                "(singleton, schema_version, active_generation) VALUES (1, 1, ?)",
                (policy_generation,),
            )
        else:
            self.set_policy_generation(policy_generation)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS harness_tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                lane TEXT,
                assignee TEXT,
                run_generation INTEGER NOT NULL DEFAULT 1
            ) STRICT"""
        )

    def set_policy_generation(self, generation: int) -> None:
        self.conn.execute(
            "UPDATE kanban_policy_pointer SET active_generation = ? "
            "WHERE singleton = 1",
            (generation,),
        )

    def add_task(self, task_id: str, lane: str, status: str = "ready") -> None:
        self.conn.execute(
            "INSERT INTO harness_tasks (task_id, status, lane) VALUES (?, ?, ?)",
            (task_id, status, lane),
        )

    def tasks(self, status: Optional[str] = None) -> list[sqlite3.Row]:
        if status is None:
            return self.conn.execute(
                "SELECT * FROM harness_tasks"
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM harness_tasks WHERE status = ?", (status,)
        ).fetchall()

    def bump_run_generation(self, task_id: str) -> None:
        self.conn.execute(
            "UPDATE harness_tasks SET run_generation = run_generation + 1 "
            "WHERE task_id = ?",
            (task_id,),
        )

    def set_status(self, task_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE harness_tasks SET status = ? WHERE task_id = ?", (status, task_id)
        )


def _claim_token(task_id: str, run_generation: int) -> str:
    raw = uuid.uuid5(uuid.NAMESPACE_URL, f"{task_id}:{run_generation}").hex
    return f"claim_token_{raw[:20]}"


class WiredDispatcher:
    """One wired dispatcher lifetime bound to a lease and a policy snapshot."""

    def __init__(
        self,
        board: HarnessBoard,
        board_identity,
        owner_lease: DispatcherOwnerLease,
        owner_generation: int,
        policy: AdmissionPolicy,
        route_contract: RouteContract,
        spawn_fn: Callable[[SpawnCall], int],
        *,
        mid_spawn_hook: Optional[Callable[[], None]] = None,
        check_registry: bool = True,
        spawn_log: Optional[list] = None,
        expected_route_hash: Optional[str] = None,
    ) -> None:
        self.board = board
        self.identity = board_identity
        self.lease = owner_lease
        self.owner_generation = owner_generation
        self.policy = policy
        self.route_contract = route_contract
        self.spawn_fn = spawn_fn
        self.mid_spawn_hook = mid_spawn_hook
        self.check_registry = check_registry
        self.next_pid = 1000
        # Shared with the harness so worker death can be simulated by
        # removing the pid from the spawn log (the stub liveness source).
        self.spawn_log_cache: list[dict] = [] if spawn_log is None else spawn_log
        # Full lifecycle history (survives spawn_log clears used to simulate
        # worker death) so invariants can see every lifecycle event.
        self.lifecycle_log: list[dict] = []
        self.expected_route_hash = expected_route_hash

    def _policy_generation(self) -> int:
        row = self.board.conn.execute(
            "SELECT active_generation FROM kanban_policy_pointer WHERE singleton = 1"
        ).fetchone()
        return int(row[0])

    def tick(self) -> TickReport:
        report = TickReport()

        # Startup fail-closed checks (wiring inventory §2.4 / R3 / PR2).
        if self.check_registry:
            try:
                validate_registry()
            except ValueError:
                for row in self.board.tasks():
                    report.refused_profiles.append(row["task_id"])
                return report
        # A dispatcher holding a policy snapshot whose generation no longer
        # matches the protected pointer must not spawn anything (stale
        # snapshot fail-closed): the pointer's policy was never evaluated.
        if self._policy_generation() != self.policy.generation:
            for row in self.board.tasks():
                if row["status"] in READY_STATUSES:
                    report.aborted.append(row["task_id"])
                    self.lifecycle_log.append(
                        {
                            "task_id": row["task_id"],
                            "run_generation": row["run_generation"],
                            "event": "aborted",
                            "detail": "stale policy snapshot",
                        }
                    )
            return report
        if not self.route_contract.valid or (
            self.expected_route_hash is not None
            and self.route_contract.hash != self.expected_route_hash
        ):
            for row in self.board.tasks():
                report.refused_profiles.append(row["task_id"])
            return report

        # Gate evaluation #1: pre-tick. Pause denies everything including
        # reclaim; frozen still reclaims but spawns nothing unlisted.
        request = AdmissionRequest(
            action="spawn",
            policy_generation=self.policy.generation,
            board_uuid=self.identity.board_uuid,
        )
        decision = self.policy.decide(request)
        if not decision.allowed and self.policy.paused:
            return report  # P1: no reclaim, no spawn, empty tick

        if self.policy.paused:
            return report

        for row in self.board.tasks("running"):
            self._reclaim_if_dead(row, report)

        if not decision.allowed and self.policy.mode == "frozen":
            # F1/F3: reclaims ran, no spawns; record per-task aborts.
            for row in self.board.tasks():
                if row["status"] in READY_STATUSES:
                    report.aborted.append(row["task_id"])
                    self.lifecycle_log.append(
                        {
                            "task_id": row["task_id"],
                            "run_generation": row["run_generation"],
                            "event": "aborted",
                            "detail": "frozen: spawn denied",
                        }
                    )
            return report

        for row in self.board.tasks():
            if row["status"] not in READY_STATUSES:
                continue
            task_id = row["task_id"]
            # Per-task admission (allowlist identity granularity).
            task_decision = self.policy.decide(
                AdmissionRequest(
                    action="spawn",
                    policy_generation=self.policy.generation,
                    board_uuid=self.identity.board_uuid,
                    task_id=task_id,
                    run_generation=row["run_generation"],
                )
            )
            if not task_decision.allowed:
                report.aborted.append(task_id)
                self.lifecycle_log.append(
                    {
                        "task_id": task_id,
                        "run_generation": row["run_generation"],
                        "event": "aborted",
                        "detail": f"admission denied: {task_decision.reason}",
                    }
                )
                continue
            assignee = row["assignee"] if "assignee" in row.keys() else None
            routing = resolve_route(
                self.route_contract,
                task_id,
                explicit_assignee=assignee,
                lane=row["lane"],
            )
            if routing.status != "routed" or routing.profile is None:
                self.board.set_status(task_id, "triage")
                report.triaged.append(task_id)
                continue
            profile = routing.profile
            if profile not in PROFILES:
                self.board.set_status(task_id, "blocked")
                report.refused_profiles.append(task_id)
                continue
            if not self._launch(row, task_id, profile, report):
                continue
            self.board.set_status(task_id, "running")
        return report

    def _launch(self, row, task_id: str, profile: str, report: TickReport) -> bool:
        conn = self.board.conn
        run_generation = row["run_generation"]
        token = _claim_token(task_id, run_generation)
        route_generation = self.route_contract.routing_contract_version
        common = dict(
            board=self.identity,
            task_id=task_id,
            run_generation=run_generation,
            dispatcher_owner_generation=self.owner_generation,
            policy_generation=self.policy.generation,
            route_generation=route_generation,
            claim_token=token,
        )
        try:
            claim_not_spawned(conn, **common)
            record_spawn_intent(
                conn, owner_lease=self.lease, **common
            )
            if self.mid_spawn_hook is not None:
                # Disturbance point: policy flip / lease loss between the
                # committed intent and process creation.
                self.mid_spawn_hook()
            # Unconditional launch-window revalidation: the active policy
            # pointer is re-read before ANY process creation; a flip or a
            # lost owner lease aborts here (fail closed — the committed
            # intent row is left as freeze-ack evidence for the controller).
            if self._policy_generation() != common["policy_generation"]:
                raise LaunchProtocolError(
                    "policy generation changed inside launch window"
                )
            if not self.lease.validate():
                raise LaunchProtocolError("owner lease lost inside launch window")
            # Second admission evaluation (documented gate): the policy is
            # re-decided for this exact task after the intent commits.
            recheck = self.policy.decide(
                AdmissionRequest(
                    action="spawn",
                    policy_generation=self.policy.generation,
                    board_uuid=self.identity.board_uuid,
                    task_id=task_id,
                    run_generation=run_generation,
                )
            )
            if not recheck.allowed:
                raise LaunchProtocolError(
                    "admission re-evaluation denied the launch"
                )
            call = SpawnCall(
                task_id=task_id,
                profile=profile,
                env={"HERMES_KANBAN_TASK": task_id},
                pid=0,
                run_generation=run_generation,
            )
            pid = self.spawn_fn(call)
            call.pid = pid
            # The process NOW exists. Record the spawned lifecycle event
            # immediately so any subsequent abort-after-creation is visible
            # to the fail-closed invariant (abort after spawned event).
            self.lifecycle_log.append(
                {
                    "task_id": task_id,
                    "run_generation": run_generation,
                    "event": "spawned",
                    "pid": pid,
                }
            )
            # Post-creation revalidation: spawn_fn may itself have flipped
            # the policy pointer or torn down the lease. Re-check both before
            # the spawned CAS; on mismatch the ledger row stays at
            # spawn_intent (freeze-ack evidence), the launch reports
            # aborted, and the invariant sees spawned-then-aborted.
            if self._policy_generation() != common["policy_generation"]:
                raise LaunchProtocolError(
                    "policy generation changed during process creation"
                )
            if not self.lease.validate():
                raise LaunchProtocolError("owner lease lost during creation")
            record_spawned(
                conn,
                pid=pid,
                owner_lease=self.lease,
                **common,
            )
            # Final guard: the pointer may have changed between the last
            # re-check and this CAS committing. If so the row reached
            # 'spawned' under a now-revoked generation: record terminal
            # evidence, kill the worker in the liveness source, and leave
            # the ledger row visibly spawned for the controller's
            # freeze-acknowledgment drain (the kernel's generation-scoped
            # APIs refuse to mutate it).
            if self._policy_generation() != common["policy_generation"]:
                # The row briefly reached 'spawned' under a now-revoked
                # generation. The kernel outcome/release APIs correctly
                # refuse to mutate claims against a different active
                # generation, so the row becomes a freeze_ack blocker the
                # controller must acknowledge and drain; the wiring records
                # terminal evidence in the lifecycle log and removes the
                # process from the liveness source (the stub worker is dead
                # to the dispatcher — never re-spawned, never counted alive).
                self.lifecycle_log.append(
                    {
                        "task_id": task_id,
                        "run_generation": run_generation,
                        "event": "aborted",
                        "exit_status": 1,
                        "detail": "policy flipped across spawned CAS",
                    }
                )
                report.aborted.append(task_id)
                report.unresolved.append(task_id)
                self._kill_spawn(task_id, run_generation)
                return False
            report.spawned.append(task_id)
            return True
        except Exception as exc:  # noqa: BLE001 - harness records every abort
            report.spawn_errors.append(f"{task_id}:{type(exc).__name__}")
            released = False
            try:
                release_claim(
                    conn,
                    owner_lease=self.lease,
                    **common,
                )
                report.released.append(task_id)
                released = True
            except Exception:  # noqa: BLE001
                # Policy flipped mid-window: the kernel deliberately refuses
                # to release a claim against a different active generation.
                # The row stays as freeze-ack evidence; no process exists.
                pass
            report.aborted.append(task_id)
            if not released:
                report.unresolved.append(task_id)
            self.lifecycle_log.append(
                {
                    "task_id": task_id,
                    "run_generation": run_generation,
                    "event": "aborted",
                    "detail": f"launch failed: {type(exc).__name__}",
                }
            )
            return False

    def _reclaim_if_dead(self, row, report: TickReport) -> None:
        # TTL/crash reclaim goes through the ledger (single authority):
        # a spawned row whose worker is dead is retired via
        # record_run_outcome, never deleted directly.
        conn = self.board.conn
        record_row = conn.execute(
            "SELECT claim_token, dispatcher_owner_generation, "
            "policy_generation, route_generation, pid "
            "FROM kanban_launch_protocol WHERE task_id = ? AND state = 'spawned'",
            (row["task_id"],),
        ).fetchone()
        if record_row is None:
            return
        # Liveness: a pid is alive iff it appears in the harness spawn log
        # (the stub equivalent of os.kill(pid, 0)). F3: running workers are
        # never touched by a frozen tick.
        live_pids = {call["pid"] for call in self.spawn_log_cache}
        if record_row["pid"] in live_pids:
            return
        # Skip tasks spawned by another dispatcher generation (lease loss
        # scenario: the new owner must not retire the old owner's rows).
        if record_row["dispatcher_owner_generation"] != self.owner_generation:
            return
        token = record_row["claim_token"]
        outcome_holder: dict = {}
        try:
            outcome_holder["outcome"] = record_run_outcome(
                conn,
                board=self.identity,
                task_id=row["task_id"],
                run_generation=row["run_generation"],
                dispatcher_owner_generation=record_row[
                    "dispatcher_owner_generation"
                ],
                policy_generation=record_row["policy_generation"],
                route_generation=record_row["route_generation"],
                claim_token=token,
                owner_lease=self.lease,
                exit_status=1,
                detail="reclaim: worker dead",
            )
            self.board.set_status(row["task_id"], "ready")
            self.board.bump_run_generation(row["task_id"])
            report.reclaimed.append(row["task_id"])
            outcome = outcome_holder["outcome"]
            self.lifecycle_log.append(
                {
                    "task_id": outcome.task_id,
                    "run_generation": outcome.run_generation,
                    "event": "outcome_recorded",
                    "exit_status": outcome.exit_status,
                    "detail": outcome.detail,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    def _kill_spawn(self, task_id: str, run_generation: int) -> None:
        """Remove a spawned process from the harness liveness source."""
        self.spawn_log_cache[:] = [
            call
            for call in self.spawn_log_cache
            if not (
                call.get("task_id") == task_id
                and call.get("run_generation", 1) == run_generation
            )
        ]

    def worker_outcome(
        self,
        task_id: str,
        exit_status: int,
        detail: str = "",
    ) -> None:
        """Worker-side terminal transition (complete/block/crash evidence)."""
        conn = self.board.conn
        row = conn.execute(
            "SELECT claim_token, run_generation, dispatcher_owner_generation, "
            "policy_generation, route_generation FROM kanban_launch_protocol "
            "WHERE task_id = ? AND state = 'spawned'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise AssertionError(f"no spawned row for {task_id}")
        outcome = record_run_outcome(
            conn,
            board=self.identity,
            task_id=task_id,
            run_generation=row["run_generation"],
            dispatcher_owner_generation=row["dispatcher_owner_generation"],
            policy_generation=row["policy_generation"],
            route_generation=row["route_generation"],
            claim_token=row["claim_token"],
            owner_lease=self.lease,
            exit_status=exit_status,
            detail=detail,
        )
        self.board.set_status(
            task_id, "done" if exit_status == 0 else "blocked"
        )
        # Evidence taken from the kernel-returned RunOutcome, not synthesized.
        self.lifecycle_log.append(
            {
                "task_id": outcome.task_id,
                "run_generation": outcome.run_generation,
                "event": "outcome_recorded",
                "exit_status": outcome.exit_status,
                "detail": outcome.detail,
            }
        )


def lane_profile(lane: str) -> str:
    return LANE_PROFILE_NAMES[lane]


@dataclasses.dataclass(frozen=True)
class HarnessIdentity:
    """Small helper mirroring BoardIdentity construction inputs."""

    board_uuid: str
    device: int
    inode: int
