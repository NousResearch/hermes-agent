"""Contract cases for the turn lease, the hygiene streak and the stale-running handoff reclaim of ``CoordinationRepository``.

Each case asserts a race or state outcome through acquire, refresh, release and ``turn_lease_expires_at`` results: one winner among contenders; a holder's re-acquire confirms without extending; refresh after expiry revives until a contender reclaims; release is holder-qualified and idempotent; a dead ``pid=`` holder stamped with this PID namespace is reclaimed before expiry (an unstamped one only where namespaces do not exist); the key is the conversation. :func:`acquire_turn_lease_waiting` is pinned with a scripted repository and an injected clock.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import pytest

from persistence import PermanentStoreError, SessionId, StoreContention, TransientStoreError
from persistence.authority.coordination import HandoffStatus, acquire_turn_lease_waiting, turn_lease_holder_pid
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind
from tests.contracts.conftest import require_capability
from tests.contracts._concurrency import run_concurrently
from tests.contracts._scopes import profile_scope
from tests.contracts._waiting import wait_past

ROOT = SessionId("lease-root")
CHILD = SessionId("lease-child")
BRANCH = SessionId("lease-branch")
OTHER = SessionId("lease-other")
TTL = 2.0
HALF = 1.0


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc.wait(timeout=60.0) == 0
    return proc.pid


def pidns_token() -> str:
    """The ``:pidns=<id>`` fragment this process stamps on its holders, ``""`` where there is none."""
    if sys.platform != "linux":
        return ""
    try:
        link = os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""
    return f":pidns={link.split('[', 1)[1].rstrip(']')}" if "[" in link else ""


def dead_holder(suffix: str = "turn=gone") -> str:
    """A holder naming a dead local process, stamped with this PID namespace, so the reclaim may probe it."""
    return f"pid={dead_pid()}{pidns_token()}:{suffix}"


def race_on_threads(attempts: list[Callable[[], Any]]) -> list[Any]:
    """Run each call on its own OS thread, released together: the physical two-contender race for a backend whose CAS lives under a process-wide write lock."""
    barrier = threading.Barrier(len(attempts))

    def run(make: Callable[[], Any]) -> Any:
        barrier.wait(timeout=30.0)
        return make()

    with ThreadPoolExecutor(max_workers=len(attempts)) as pool:
        return list(pool.map(run, attempts))


# ── try_acquire / refresh / release ───────────────────────────────────────────


def test_turn_lease_two_contenders_one_winner_loser_cannot_refresh(backend, scope_a):
    coord = backend.open(scope_a).coordination
    require_capability(backend, "transactional_claims")
    holders = [f"contender-{i}" for i in range(6)]
    outcomes = run_concurrently(*[lambda h=h: coord.try_acquire_turn_lease(session_id=ROOT, holder=h, ttl_seconds=300.0) for h in holders])
    assert outcomes.count(True) == 1 and outcomes.count(False) == len(holders) - 1
    winner = holders[outcomes.index(True)]
    loser = next(h for h in holders if h != winner)
    assert coord.refresh_turn_lease(session_id=ROOT, holder=winner, ttl_seconds=300.0) is True
    assert coord.refresh_turn_lease(session_id=ROOT, holder=loser, ttl_seconds=300.0) is False


def test_turn_lease_two_contenders_on_two_threads_one_winner(backend, scope_a):
    coord = backend.open(scope_a).coordination
    require_capability(backend, "transactional_claims")
    outcomes = race_on_threads([
        lambda: coord.try_acquire_turn_lease(session_id=OTHER, holder="thread-a", ttl_seconds=300.0),
        lambda: coord.try_acquire_turn_lease(session_id=OTHER, holder="thread-b", ttl_seconds=300.0),
    ])
    assert sorted(outcomes) == [False, True]
    winner, loser = ("thread-a", "thread-b") if outcomes[0] else ("thread-b", "thread-a")
    coord.release_turn_lease(session_id=OTHER, holder=loser)
    assert coord.refresh_turn_lease(session_id=OTHER, holder=winner, ttl_seconds=300.0) is True


def test_turn_lease_holder_reacquire_confirms_without_extending(backend, scope_a):
    coord = backend.open(scope_a).coordination
    start = time.time()
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=TTL) is True
    expires = coord.turn_lease_expires_at(session_id=ROOT, holder="holder-a")
    assert expires is not None and start + TTL <= expires <= time.time() + TTL
    wait_past(start + HALF)
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    assert coord.turn_lease_expires_at(session_id=ROOT, holder="holder-a") == expires
    wait_past(expires)
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-b", ttl_seconds=300.0) is True


def test_turn_lease_expires_at_reads_only_the_holders_row(backend, scope_a):
    # CONTRACT: the committed expiry of the holder's row on the conversation; None for another holder and for empty arguments; the TTL is clamped to 0.1 s.
    coord = backend.open(scope_a).coordination
    before = time.time()
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=0.0) is True
    assert coord.turn_lease_expires_at(session_id=ROOT, holder="holder-a") >= before + 0.1
    assert coord.turn_lease_expires_at(session_id=ROOT, holder="holder-b") is None
    assert coord.turn_lease_expires_at(session_id=SessionId(""), holder="holder-a") is None
    assert coord.turn_lease_expires_at(session_id=ROOT, holder="") is None


def test_turn_lease_refresh_extends_the_expiry(backend, scope_a):
    coord = backend.open(scope_a).coordination
    start = time.time()
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=TTL) is True
    expires = time.time() + TTL
    wait_past(start + HALF)
    assert coord.refresh_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    wait_past(expires)
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-b", ttl_seconds=300.0) is False


def test_turn_lease_refresh_after_expiry_revives_until_reclaimed_then_fails(backend, scope_a):
    coord = backend.open(scope_a).coordination
    # Arm 1: nobody contended; the slow turn's late refresh revives the lease.
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    assert coord.refresh_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="contender", ttl_seconds=300.0) is False
    # Arm 2: a contender reclaimed the expired row; the old holder's refresh fails.
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder="slow-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder="contender", ttl_seconds=300.0) is True
    assert coord.refresh_turn_lease(session_id=OTHER, holder="slow-turn", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=OTHER, holder="contender", ttl_seconds=300.0) is True


def test_turn_lease_release_is_holder_qualified_and_idempotent(backend, scope_a):
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    coord.release_turn_lease(session_id=ROOT, holder="holder-b")
    assert coord.refresh_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-b", ttl_seconds=300.0) is False
    coord.release_turn_lease(session_id=ROOT, holder="holder-a")
    assert coord.refresh_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is False
    coord.release_turn_lease(session_id=ROOT, holder="holder-a")
    coord.release_turn_lease(session_id=OTHER, holder="nobody")
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-b", ttl_seconds=300.0) is True


def test_turn_lease_expired_but_unreclaimed_row_is_still_releasable_by_its_holder(backend, scope_a):
    """The holder's release of its own expired row removes it (matching refresh's holder-identity rule): the row is no longer revivable by a late refresh."""
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    coord.release_turn_lease(session_id=ROOT, holder="slow-turn")
    assert coord.refresh_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=300.0) is False


def test_turn_lease_dead_pid_holder_is_reclaimed_before_expiry(backend, scope_a):
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder=dead_holder("turn=t1:platform=cli"), ttl_seconds=300.0) is True
    live = f"pid={os.getpid()}{pidns_token()}:turn=t2:platform=cli"
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder=live, ttl_seconds=300.0) is True
    # The current process is alive: not reclaimed before expiry.
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder=dead_holder("turn=t3:platform=cli"), ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=ROOT, holder=live, ttl_seconds=300.0) is True
    # An unstructured holder is never reclaimed before expiry.
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder="unstructured", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder=dead_holder("turn=t4:platform=cli"), ttl_seconds=300.0) is False


def test_turn_lease_unstamped_or_foreign_namespace_holder_defers_to_its_expiry(backend, scope_a):
    # CONTRACT: where PID namespaces exist, a holder without this namespace's stamp, or with another's, is never probed; it keeps its row until expiry.
    if not pidns_token():
        pytest.skip("this platform has no PID namespace identity: every pid holder is probed")
    coord = backend.open(scope_a).coordination
    for sid, holder in ((ROOT, f"pid={dead_pid()}:turn=unstamped"), (OTHER, f"pid={dead_pid()}:pidns=1:turn=foreign")):
        assert coord.try_acquire_turn_lease(session_id=sid, holder=holder, ttl_seconds=300.0) is True
        assert coord.try_acquire_turn_lease(session_id=sid, holder="contender", ttl_seconds=300.0) is False


def test_turn_lease_rows_expired_a_day_ago_are_swept(backend, scope_a, monkeypatch):
    # CONTRACT: every acquisition first deletes rows expired more than a day ago, so a late refresh of such a row fails; a recently expired row survives for its holder's refresh.
    coord = backend.open(scope_a).coordination
    now = [1_800_000_000.0]
    monkeypatch.setattr("time.time", lambda: now[0])
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder="ancient", ttl_seconds=1.0) is True
    assert coord.try_acquire_turn_lease(session_id=CHILD, holder="recent", ttl_seconds=86400.0) is True
    now[0] += 86400.0 + 2.0
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="sweeper", ttl_seconds=300.0) is True
    assert coord.refresh_turn_lease(session_id=OTHER, holder="ancient", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=CHILD, holder="recent", ttl_seconds=300.0) is True


def test_turn_lease_key_is_the_conversation_not_the_segment(backend, scope_a):
    coord, sessions = backend.open(scope_a).coordination, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    sessions.create(NewSession(
        session_id=BRANCH, source="cli", parent_session_id=ROOT, lineage_kind=SessionLineageKind.BRANCH,
        model_config={"_branched_from": str(ROOT)}))
    assert coord.try_acquire_turn_lease(session_id=CHILD, holder="segment-holder", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="root-holder", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=ROOT, holder="segment-holder", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=BRANCH, holder="branch-holder", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(
        session_id=SessionId("never-created"), holder="ghost", ttl_seconds=300.0) is True
    coord.release_turn_lease(session_id=ROOT, holder="segment-holder")
    assert coord.try_acquire_turn_lease(session_id=CHILD, holder="root-holder", ttl_seconds=300.0) is True


def test_turn_lease_key_walks_the_parent_as_it_stands_now_and_tool_children_stand_alone(backend, scope_a):
    """The walk reads the PARENT's current end reason, not a creation-time flag on the child: a child created before its parent was ended by compression still contends for the root once it is. A tool child never climbs."""
    coord, sessions = backend.open(scope_a).coordination, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    tool = SessionId("lease-tool-child")
    sessions.create(NewSession(session_id=tool, source="tool", parent_session_id=ROOT))
    # Parent still live: the child is its own conversation.
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="root-holder", ttl_seconds=300.0) is True
    assert coord.try_acquire_turn_lease(session_id=CHILD, holder="early-child", ttl_seconds=300.0) is True
    coord.release_turn_lease(session_id=CHILD, holder="early-child")
    # Parent ended by compression afterwards: the same child now contends for the root's lease.
    sessions.end(ROOT, END_REASON_COMPRESSION)
    assert coord.try_acquire_turn_lease(session_id=CHILD, holder="late-child", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=CHILD, holder="root-holder", ttl_seconds=300.0) is True
    # A tool child of the compression-ended root stands alone.
    assert coord.try_acquire_turn_lease(session_id=tool, holder="tool-holder", ttl_seconds=300.0) is True


def test_turn_lease_empty_arguments_refused_and_scopes_isolated(backend, scope_a, scope_b):
    store_b = backend.open(scope_b)
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=SessionId(""), holder="h", ttl_seconds=300.0) is False
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=ROOT, holder="", ttl_seconds=300.0) is False
    coord.release_turn_lease(session_id=SessionId(""), holder="h")
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    # Same session id, other profile: an independent lease.
    assert store_b.coordination.try_acquire_turn_lease(session_id=ROOT, holder="holder-b", ttl_seconds=300.0) is True
    assert store_b.coordination.refresh_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is False


def test_turn_lease_is_shared_by_every_handle_on_the_profile(backend, scope_a):
    """A lease taken through one handle is the lease every other handle on the same profile sees, as two ``SessionDB`` on one file share it."""
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="seam-holder", ttl_seconds=300.0) is True
    other = backend.open(profile_scope(scope_a.name)).coordination
    assert other.refresh_turn_lease(session_id=ROOT, holder="seam-holder", ttl_seconds=300.0) is True
    assert other.try_acquire_turn_lease(session_id=ROOT, holder="other-process", ttl_seconds=300.0) is False


def test_turn_lease_attempt_under_a_held_writer_gives_up_within_the_attempt_patience(backend, backend_name, scope_a):
    """SQLite arm: one attempt given the waiting loop's per-attempt patience (``patience_s=0.5``) against a writer held by another connection gives up with ``StoreContention`` within that patience, not the store's routine 20 s write patience, so ``acquire_turn_lease_waiting`` keeps its deadline."""
    store_a = backend.open(scope_a)
    if backend_name != "sqlite":
        return
    import sqlite3

    store_a.ready()
    blocker = sqlite3.connect(str(store_a.store_path), timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        with pytest.raises(StoreContention):
            store_a.coordination.try_acquire_turn_lease(session_id=ROOT, holder="waiter", ttl_seconds=30.0, patience_s=0.5)
    finally:
        blocker.rollback()
        blocker.close()
    assert time.monotonic() - started < 5.0


# ── hygiene failure streak ────────────────────────────────────────────────────


def test_hygiene_streak_increments_resets_and_ignores_empty_keys(backend, scope_a, scope_b):
    store_b = backend.open(scope_b)
    coord = backend.open(scope_a).coordination
    assert [coord.increment_hygiene_failure_streak(session_key="agent:main:k1") for _ in range(3)] == [1, 2, 3]
    assert coord.increment_hygiene_failure_streak(session_key="agent:main:k2") == 1
    assert store_b.coordination.increment_hygiene_failure_streak(session_key="agent:main:k1") == 1
    coord.reset_hygiene_failure_streak(session_key="agent:main:k1")
    coord.reset_hygiene_failure_streak(session_key="agent:main:k1")
    assert coord.increment_hygiene_failure_streak(session_key="agent:main:k1") == 1
    assert coord.increment_hygiene_failure_streak(session_key="agent:main:k2") == 2
    assert coord.increment_hygiene_failure_streak(session_key="") == 1
    assert coord.increment_hygiene_failure_streak(session_key="") == 1
    coord.reset_hygiene_failure_streak(session_key="")
    coord.reset_hygiene_failure_streak(session_key="agent:main:never")


def test_hygiene_streak_two_incrementers_observe_distinct_consecutive_values(backend, scope_a):
    coord = backend.open(scope_a).coordination
    require_capability(backend, "transactional_claims")
    seen = race_on_threads([
        lambda: coord.increment_hygiene_failure_streak(session_key="agent:main:race"),
        lambda: coord.increment_hygiene_failure_streak(session_key="agent:main:race"),
    ])
    assert sorted(seen) == [1, 2]
    assert coord.increment_hygiene_failure_streak(session_key="agent:main:race") == 3


# ── reclaim_stale_running_handoffs ────────────────────────────────────────────


def test_reclaim_stale_running_handoffs_fails_running_rows_once_and_rearms(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    coord = backend.open(scope_a).coordination
    store_a.sessions.create(NewSession(ROOT, "cli"))
    store_a.sessions.create(NewSession(OTHER, "cli"))
    store_b.sessions.create(NewSession(ROOT, "cli"))
    assert coord.request_handoff(session_id=ROOT, platform="telegram") is True
    assert coord.claim_handoff(session_id=ROOT) is True
    assert coord.request_handoff(session_id=OTHER, platform="discord") is True
    assert store_b.coordination.request_handoff(session_id=ROOT, platform="telegram") is True
    assert store_b.coordination.claim_handoff(session_id=ROOT) is True

    assert coord.reclaim_stale_running_handoffs(error="gateway died mid-handoff") == (ROOT,)
    reclaimed = coord.get_handoff(session_id=ROOT)
    assert reclaimed is not None and reclaimed.state is HandoffStatus.FAILED
    assert reclaimed.error == "gateway died mid-handoff"
    pending = coord.get_handoff(session_id=OTHER)
    assert pending is not None and pending.state is HandoffStatus.PENDING
    # Other profile's running row is untouched.
    foreign = store_b.coordination.get_handoff(session_id=ROOT)
    assert foreign is not None and foreign.state is HandoffStatus.RUNNING
    # Second reclaimer: nothing left; the first reclaimer's error stands.
    assert coord.reclaim_stale_running_handoffs(error="again") == ()
    again = coord.get_handoff(session_id=ROOT)
    assert again is not None and again.error == "gateway died mid-handoff"
    # Failed re-arms.
    assert coord.request_handoff(session_id=ROOT, platform="telegram") is True


def test_reclaim_stale_running_handoffs_never_raises_on_a_torn_down_store(backend, scope_a):
    """The startup reclaim is an operator trace, never a crash: after teardown it reads as nothing reclaimed."""
    coord = backend.open(scope_a).coordination
    backend.close()
    assert coord.reclaim_stale_running_handoffs(error="after teardown") == ()


def test_reclaim_stale_running_handoffs_two_reclaimers_one_gets_each_row(backend, scope_a):
    store_a = backend.open(scope_a)
    coord = backend.open(scope_a).coordination
    require_capability(backend, "transactional_claims")
    for sid in (ROOT, OTHER):
        store_a.sessions.create(NewSession(sid, "cli"))
        assert coord.request_handoff(session_id=sid, platform="telegram") is True
        assert coord.claim_handoff(session_id=sid) is True
    outcomes = race_on_threads([
        lambda: coord.reclaim_stale_running_handoffs(error="reclaimer-a"),
        lambda: coord.reclaim_stale_running_handoffs(error="reclaimer-b"),
    ])
    assert sorted(sid for ids in outcomes for sid in ids) == sorted([ROOT, OTHER])
    assert not (set(outcomes[0]) & set(outcomes[1]))


# ── acquire_turn_lease_waiting: the wait loop ─────────────────────────────────


class Script:
    """A scripted coordination repository plus a fake clock: each ``try_acquire_turn_lease`` pops the next outcome (a bool, or an exception to raise) and records its ``patience_s``; every sleep advances the clock."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.patience: list[Optional[float]] = []
        self.now = 1000.0
        self.sleeps: list[float] = []

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0, patience_s: Optional[float] = None) -> bool:
        self.attempts += 1
        self.patience.append(patience_s)
        outcome = self.outcomes.pop(0) if self.outcomes else False
        if isinstance(outcome, BaseException):
            raise outcome
        return bool(outcome)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _wait(script: Script, **options: Any) -> bool:
    return acquire_turn_lease_waiting(script, session_id=ROOT, holder="h", clock=script.clock, sleep=script.sleep, **options)  # type: ignore[arg-type]


def test_wait_loop_returns_the_first_win_and_never_notices():
    script = Script([True])
    notices: list[float] = []
    assert _wait(script, wait_seconds=10.0, on_wait=notices.append) is True
    assert script.attempts == 1 and notices == [] and script.sleeps == [] and script.patience == [0.5]


def test_wait_loop_should_abort_wins_before_the_first_attempt():
    script = Script([True])
    assert _wait(script, wait_seconds=10.0, should_abort=lambda: True) is False
    assert script.attempts == 0


def test_wait_loop_polls_notices_and_clamps_the_last_sleep_to_the_deadline():
    """Misses poll at the interval; on_wait fires on the first miss (elapsed 0) and then every notice interval; the final sleep is clamped to the remaining time and the deadline returns False."""
    script = Script([False] * 10)
    notices: list[float] = []
    assert _wait(script, wait_seconds=2.5, poll_interval_seconds=1.0, on_wait=notices.append, wait_notice_interval_seconds=2.0) is False
    assert script.sleeps == [1.0, 1.0, 0.5]
    assert script.attempts == 4
    assert notices == [0.0, 2.0]


def test_wait_loop_notice_interval_zero_fires_on_every_miss():
    script = Script([False, False, True])
    notices: list[float] = []
    assert _wait(script, wait_seconds=10.0, poll_interval_seconds=0.5, on_wait=notices.append, wait_notice_interval_seconds=0.0) is True
    assert notices == [0.0, 0.5] and script.attempts == 3


def test_wait_loop_zero_wait_makes_exactly_one_attempt():
    script = Script([False, True])
    assert _wait(script, wait_seconds=0.0) is False
    assert script.attempts == 1 and script.sleeps == []


def test_wait_loop_contention_is_not_a_holder_and_others_propagate():
    # CONTRACT: the loop polls only on the "locked" class: StoreContention calls on_contended, never on_wait, retries after one short jitter with the poll interval moved into the write patience; any other error, a plain TransientStoreError included, propagates at once.
    script = Script([StoreContention("database is locked"), True])
    contended: list[bool] = []
    notices: list[float] = []
    assert _wait(script, wait_seconds=10.0, poll_interval_seconds=0.1, on_contended=lambda: contended.append(True), on_wait=notices.append) is True
    assert script.attempts == 2 and contended == [True] and notices == []
    assert script.patience == [0.5, pytest.approx(0.6)]
    assert len(script.sleeps) == 1 and 0.02 <= script.sleeps[0] <= 0.15
    failing = Script([PermanentStoreError("disk I/O error")])
    with pytest.raises(PermanentStoreError):
        _wait(failing, wait_seconds=10.0)
    transient = Script([TransientStoreError("disk I/O error"), True])
    with pytest.raises(TransientStoreError):
        _wait(transient, wait_seconds=10.0)
    assert transient.attempts == 1 and transient.sleeps == []


def test_wait_loop_callback_errors_are_swallowed():
    calls = {"n": 0}

    def boom(*_: Any) -> Any:
        calls["n"] += 1
        raise RuntimeError("callback failed")

    script = Script([False, True])
    assert _wait(script, wait_seconds=10.0, poll_interval_seconds=0.1, on_wait=boom, should_abort=boom) is True
    # should_abort before each of the two attempts, on_wait once after the miss: all three raised, all swallowed.
    assert calls["n"] == 3 and script.attempts == 2 and script.sleeps == [0.1]


def test_wait_loop_over_the_backend_waits_for_the_holder(backend, scope_a):
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="holder-a", ttl_seconds=300.0) is True
    assert acquire_turn_lease_waiting(coord, session_id=ROOT, holder="holder-b", wait_seconds=0.0) is False
    coord.release_turn_lease(session_id=ROOT, holder="holder-a")
    assert acquire_turn_lease_waiting(coord, session_id=ROOT, holder="holder-b", wait_seconds=0.0) is True


def test_turn_lease_holder_pid_parses_only_the_structured_segment():
    assert turn_lease_holder_pid("pid=4242:turn=abc:platform=cli") == 4242
    assert turn_lease_holder_pid("turn=abc:pid=7") == 7
    assert turn_lease_holder_pid("pid=0:turn=abc") is None
    assert turn_lease_holder_pid("prefix-pid=12") is None
    assert turn_lease_holder_pid("unstructured") is None
    assert turn_lease_holder_pid("") is None
