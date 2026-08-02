"""Regression: prevent transcript fork when two paths compress the same session_id.

Damien's incident (Discord, 2026-05-28): a long Hermes session in a Discord
gateway hit the compression threshold at the end of a turn.  The parent agent
finished delivering the response and ``conversation_loop.py`` fired
``_spawn_background_review(...)`` — which builds a forked ``AIAgent`` that
inherits ``agent.session_id`` (see ``agent/background_review.py``::
``review_agent.session_id = agent.session_id``).  Roughly two seconds later
a synthetic ``Background process proc_… completed`` event arrived and
started a fresh turn on the same parent ``session_id`` (still cached in the
gateway's ``SessionEntry``).  Both paths hit preflight compression on the
same parent transcript and called ``_compress_context`` concurrently.  Each
ended the parent and created its own CHILD session in ``state.db``, both
parented to the same old id.  The gateway's ``SessionEntry`` only caught one
rotation; the other child became an orphan that silently accumulated writes.

Repro shape on Damien's machine:

  parent 20260527_234659_e65f0e  ended_at=set  end_reason='compression'
  child  20260528_113619_fc80e1  parent=20260527_234659_e65f0e  (in SessionEntry)
  child  <orphan>                parent=20260527_234659_e65f0e  (silent writes)

This regression simulates the two concurrent ``compress_context`` calls
against a shared ``state.db`` and asserts that the per-session compression
lock added in this PR prevents the orphan child.  Without the lock the
fixture deterministically produces 2 children; with the lock, exactly 1.
"""

from __future__ import annotations

import copy
import inspect
import os
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(
    db: SessionDB,
    session_id: str,
    *,
    stub_compressor: bool = True,
):
    """Build an AIAgent that's wired to ``db`` and pinned to ``session_id``."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    # Stub the compressor so it returns deterministic output and DOESN'T make
    # an LLM call.  Sleep inside compress() so the two threads' rotations
    # actually overlap — without that the OS could happen to serialize them
    # and hide the bug.
    if not stub_compressor:
        return agent

    compressor = MagicMock()

    def _compress_with_overlap(*_a, **_kw):
        time.sleep(0.15)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress_with_overlap
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # These tests cover the ROTATION fallback path (forking, child sessions,
    # lock contention) — pin in_place=False so they keep exercising it
    # regardless of the global default (which flipped to True in #38763).
    agent.compression_in_place = False
    return agent


def _count_children(db: SessionDB, parent_sid: str) -> int:
    """Count rows in state.db whose parent_session_id == parent_sid."""
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    return len(rows)


def _live_child_id(db: SessionDB, parent_sid: str) -> str | None:
    """The single child id of ``parent_sid``, or None when there is none.

    Fails loudly on more than one child: callers use this to prove the agents
    converged on the winner's session, so a multi-child state is a fork and
    must not be silently reduced to 'the first row'.
    """
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    assert len(rows) <= 1, f"expected at most one child of {parent_sid}, got {rows!r}"
    return rows[0][0] if rows else None


def _wait_for_touch(touch_calls: list[str], value: str, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value in touch_calls:
            return
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for touch activity {value!r}; calls={touch_calls!r}")













def test_concurrent_compression_does_not_fork_session(tmp_path: Path) -> None:
    """Two AIAgents that share a session_id MUST NOT both rotate it.

    Without the per-session compression lock this fixture deterministically
    produces 2 child sessions (transcript fork). With the lock at most one
    path rotates: normally exactly 1 canonical child, or — under heavy DB
    write contention that makes the winner's child create_session exhaust its
    retries — 0, because _compress_context safely rolls back to the parent
    instead of orphaning a child. The forbidden outcome is 2+ (the fork).
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "PARENT_TEST_SESSION"
    db.create_session(parent_sid, source="discord")

    # Two agents on the same session_id, both wired to the same db —
    # mirrors the parent-turn agent + the background-review fork right
    # after a turn ends.
    agent_a = _build_agent_with_db(db, parent_sid)
    agent_b = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def run(agent):
        try:
            agent._compress_context(messages, "sys", approx_tokens=120_000)
        except Exception:
            # Surface to the test if either raises — should not happen.
            raise

    t_a = threading.Thread(target=run, args=(agent_a,), name="main_turn")
    t_b = threading.Thread(target=run, args=(agent_b,), name="review_fork")
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # The invariant Damien's incident is about: the parent must NEVER end up
    # with two (or more) children — that is the transcript fork. The lock
    # guarantees only one path rotates.
    #
    # Zero children is also a valid, non-forking outcome: under heavy DB write
    # contention the winner's child ``create_session`` can exhaust its retry
    # budget, and ``_compress_context`` deliberately rolls the live id back to
    # the (still-indexed) parent rather than stranding an orphan child — see
    # the create-failure rollback in agent/conversation_compression.py. That
    # safe rollback leaves 0 children and is correct. So the contract is
    # ``children <= 1``; only ``>= 2`` is the bug. Asserting an exact ``== 1``
    # made this test flaky under the concurrent CI load that triggers the
    # contention rollback (#54465 churn surfaced it).
    n_children = _count_children(db, parent_sid)
    assert n_children <= 1, (
        f"Compression lock failed: parent session has {n_children} children in "
        "state.db (transcript fork). This is Damien's incident shape — see the "
        "test docstring. Two or more children means the lock did not serialize "
        "the concurrent rotations."
    )

    # Every agent that moved off the parent must have landed on the SAME id.
    # Counting movers is the wrong contract: the loser can legitimately end up
    # on the child too, without rotating anything itself — it takes the lock
    # after the winner released it, sees the parent was already rotated, and
    # _adopt_live_compression_child() points it at the winner's single child
    # (the "compression recovery: stale session=... adopted live child=..."
    # path). That convergence is the fix working, not a fork; the fork is two
    # DIFFERENT live ids. Asserting ``movers <= 1`` failed on that healthy
    # outcome under concurrent load.
    moved = {a.session_id for a in (agent_a, agent_b) if a.session_id != parent_sid}
    assert len(moved) <= 1, (
        f"Expected at most one post-compression session id, got {sorted(moved)}. "
        "Two distinct ids means the lock didn't serialize them (transcript fork)."
    )
    assert len(moved) == n_children, (
        f"Inconsistent state: agents live on {sorted(moved)} but {n_children} "
        "child session(s) exist — rotation and child creation diverged."
    )
    if moved:
        child = _live_child_id(db, parent_sid)
        assert moved == {child}, (
            f"Agents live on {sorted(moved)} but the parent's only child is "
            f"{child} — an agent is writing to a session outside the lineage."
        )

    # The lock must be released after both paths finished, regardless of
    # whether the winner committed a child or rolled back.
    assert db.get_compression_lock_holder(parent_sid) is None, (
        "Compression lock leaked: still held after both paths completed."
    )


def test_durable_message_committed_before_lease_is_adopted(
    tmp_path: Path,
) -> None:
    """A durable row absent from the caller snapshot must still be compressed.

    Previously this path aborted and returned the stale snapshot unchanged,
    which permanently wedged busy sessions: every compress attempt saw the
    DB ahead of the in-memory list, logged "changed before lease
    acquisition", and never called the compressor. Adopting the durable
    transcript keeps the late-committed turn and lets compression proceed.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PRE_LEASE_DURABLE_RACE"
    db.create_session(parent_sid, source="webui")
    db.append_message(parent_sid, "user", "old durable")

    # Frontend takes its snapshot, then another producer commits before this
    # compressor acquires the lease.
    stale_snapshot = [{"role": "user", "content": "old durable"}]
    db.append_message(parent_sid, "assistant", "late committed before lease")
    agent = _build_agent_with_db(db, parent_sid)

    returned, _system_prompt = agent._compress_context(
        stale_snapshot, "sys", approx_tokens=120_000
    )

    agent.context_compressor.compress.assert_called_once()
    compressed_arg = agent.context_compressor.compress.call_args.args[0]
    assert [m["content"] for m in compressed_arg] == [
        "old durable",
        "late committed before lease",
    ]
    # Must not echo the stale snapshot — compression proceeded on the
    # adopted durable transcript (rotation publishes a child session).
    assert returned is not stale_snapshot
    assert returned[0]["content"] == "[CONTEXT COMPACTION] summary"
    assert agent.session_id != parent_sid
    child_id = _live_child_id(db, parent_sid)
    assert child_id is not None
    assert child_id == agent.session_id





def test_fence_cancelled_compression_leaves_lock_reacquirable(tmp_path: Path) -> None:
    """A fence-cancelled attempt must not poison the per-session lock.

    Lock-release verification for the hygiene-timeout path: after the gateway
    times out and cancels a hygiene compression at the commit fence, the very
    next attempt on the same session (e.g. the user running ``/compress``)
    must acquire the compression lock and commit normally. A leaked lock here
    would silently block every future compaction for the session until TTL
    expiry.
    """
    from agent.conversation_compression import CompressionCommitFence

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HYGIENE_LOCK_REACQUIRE"
    db.create_session(session_id, source="telegram")

    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    summary_started = threading.Event()
    release_summary = threading.Event()

    def _slow_summary(*_args, **_kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=5)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_summary
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    fence = CompressionCommitFence()
    result = {}

    def _run_compression() -> None:
        result["value"] = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            commit_fence=fence,
        )

    worker = threading.Thread(target=_run_compression, name="fenced-hygiene")
    worker.start()
    assert summary_started.wait(timeout=2)
    assert fence.cancel_before_commit() is True
    release_summary.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    # Cancelled attempt: no mutation, and — the invariant under test — the
    # per-session compression lock is fully released.
    assert result["value"][0] is messages
    assert db.get_compression_lock_holder(session_id) is None

    # The NEXT attempt (no fence — a manual /compress retry) must be able to
    # acquire the lock and commit an in-place compaction normally.
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert retried is not messages
    assert len(retried) < len(messages)
    assert agent.session_id == session_id  # in-place: same session id
    assert agent._last_compaction_in_place is True
    assert db.get_compression_lock_holder(session_id) is None


def test_commit_fence_waits_for_an_active_commit() -> None:
    """A timeout that loses the fence race cannot overlap the live turn."""
    from agent.conversation_compression import CompressionCommitFence

    fence = CompressionCommitFence()
    assert fence.begin_commit() is True
    assert fence.try_cancel_before_commit() is None
    cancel_started = threading.Event()
    cancel_finished = threading.Event()
    result = {}

    def _cancel() -> None:
        cancel_started.set()
        result["cancelled"] = fence.cancel_before_commit()
        cancel_finished.set()

    waiter = threading.Thread(target=_cancel, name="hygiene-timeout-fence")
    waiter.start()
    try:
        assert cancel_started.wait(timeout=2)
        assert not cancel_finished.is_set()
    finally:
        fence.finish_commit()
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result["cancelled"] is False


def test_delayed_contender_adopts_unique_rotated_child(tmp_path: Path) -> None:
    """A stale agent must continue on the winner's compacted child transcript."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "STALE_PARENT"
    child_sid = "CANONICAL_CHILD"
    db.create_session(parent_sid, source="webui")
    db.end_session(parent_sid, "compression")
    db.create_session(child_sid, source="webui", parent_session_id=parent_sid)
    compacted = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "compacted tail"},
    ]
    db.replace_messages(child_sid, compacted)

    agent = _build_agent_with_db(db, parent_sid)
    stale_messages = [
        {"role": "user", "content": "stale"},
        {"role": "assistant", "content": "x" * 1000},
    ]
    recovered, _system_prompt = agent._compress_context(
        stale_messages, "sys", approx_tokens=120_000
    )

    assert agent.session_id == child_sid
    assert [(m["role"], m["content"]) for m in recovered] == [
        ("user", "[CONTEXT COMPACTION] summary"),
        ("assistant", "compacted tail"),
    ]
    assert agent._session_db_created is True
    assert agent._flushed_db_message_session_id == child_sid
    assert agent._last_flushed_db_idx == len(recovered)
    agent.context_compressor.compress.assert_not_called()
    lifecycle_args, lifecycle_kwargs = agent.context_compressor.on_session_start.call_args
    assert lifecycle_args == (child_sid,)
    assert lifecycle_kwargs["boundary_reason"] == "compression"
    assert lifecycle_kwargs["old_session_id"] == parent_sid
    assert lifecycle_kwargs["session_db"] is db








def _no_consecutive_user_roles(messages: list) -> bool:
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return all(
        not (roles[i] == roles[i + 1] == "user") for i in range(len(roles) - 1)
    )


def test_restored_anchor_never_creates_consecutive_user_roles() -> None:
    """Anchor restoration must preserve strict role alternation (#55677).

    The original insertion helper could land the human anchor directly next
    to user-role scaffolding (index-0 insert before a leading synthetic user
    turn, or a bare scaffolding-only transcript), producing user/user
    adjacency that strict chat templates reject.
    """
    from agent.conversation_compression import _insert_real_user_anchor

    anchor = {"role": "user", "content": "REAL HUMAN ASK"}

    # Leading synthetic user turn before the assistant summary.
    compressed = [
        {
            "role": "user",
            "content": "[System: Your previous response was truncated ...]",
            "_empty_recovery_synthetic": True,
        },
        {"role": "assistant", "content": "summary"},
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert any(m.get("content", "").startswith("REAL HUMAN ASK") for m in compressed)

    # Scaffolding-only transcript: the anchor is merged, not inserted
    # adjacent, and the merged turn leads with the human ask.
    compressed = [
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert len(compressed) == 1
    assert compressed[0]["content"].startswith("REAL HUMAN ASK")
    assert not compressed[0].get("_todo_snapshot_synthetic")




def test_compression_persists_child_handoff_immediately(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "HEADLESS_PREFLIGHT_PARENT"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)
    child_sid = agent.session_id

    assert child_sid != parent_sid
    assert db.get_session(parent_sid)["end_reason"] == "compression"
    assert len(db.get_messages(child_sid)) == len(compressed)

    agent._flush_messages_to_session_db(compressed, None)
    assert len(db.get_messages(child_sid)) == len(compressed)




@pytest.mark.parametrize("in_place", [False, True])
def test_equal_copy_compression_result_does_not_rewrite_session(
    tmp_path: Path,
    in_place: bool,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = f"EQUAL_COPY_NOOP_{in_place}"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    setattr(agent, "compression_in_place", in_place)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    compressor = getattr(agent, "context_compressor")
    compressor.compress.side_effect = lambda incoming, **_kw: list(incoming)

    with patch.object(
        db,
        "archive_and_compact",
        wraps=db.archive_and_compact,
    ) as archive_and_compact:
        returned, _sp = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
        )

    assert returned is messages
    assert getattr(agent, "session_id") == parent_sid
    assert _count_children(db, parent_sid) == 0
    parent = db.get_session(parent_sid)
    assert parent is not None
    assert parent["end_reason"] is None
    assert db.get_compression_lock_holder(parent_sid) is None
    archive_and_compact.assert_not_called()




def test_post_compress_exception_stops_lock_refresher(tmp_path: Path, monkeypatch) -> None:
    """A warning-path exception after compress() returns must still release the lock."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=0.15)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 0.15
    agent._compression_lock_refresh_interval = 0.05
    agent.context_compressor._last_summary_error = "summary failed"
    agent._emit_warning = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("warn boom"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="warn boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    time.sleep(0.25)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True








def test_signature_introspection_exception_releases_lock_and_refresher(
    tmp_path: Path, monkeypatch
) -> None:
    """Capability inspection failures must not leak the acquired lock lease."""
    from agent.conversation_compression import (
        _CompressionLockLeaseRefresher as RealLeaseRefresher,
    )

    refreshers = []

    class RecordingLeaseRefresher(RealLeaseRefresher):
        def start(self):
            refreshers.append(self)
            return super().start()

    monkeypatch.setattr(
        "agent.conversation_compression._CompressionLockLeaseRefresher",
        RecordingLeaseRefresher,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "SIGNATURE_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_refresh_interval = 0.1

    class SignatureBomb:
        calls = 0

        @property
        def __signature__(self):
            raise RuntimeError("signature boom")

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("engine must not run after signature failure")

    bomb = SignatureBomb()
    agent.context_compressor.compress = bomb
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="signature boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert bomb.calls == 0
    assert db.get_compression_lock_holder(parent_sid) is None
    assert len(refreshers) == 1
    assert not refreshers[0]._thread.is_alive()








def _make_legacy_session_db_class() -> type:
    """Model the class retained in ``sys.modules`` before the lock API existed.

    During the real version-skew incident, a re-imported compression module
    imports the same still-loaded ``hermes_state`` module, whose ``SessionDB``
    class is old. The test replaces that module attribute with this lockless
    class and forwards all persistence operations to a current real database.
    """
    source_path = inspect.getfile(SessionDB)
    namespace = {"__name__": "hermes_state"}
    source = '''
class SessionDB:
    def __init__(self, real_db):
        self._real = real_db

    def __getattribute__(self, name):
        if name in {"_real", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)
'''
    exec(compile(source, source_path, "exec"), namespace)
    return namespace["SessionDB"]


class _NominalSessionDBImpostor:
    """A proxy that spoofs names but lacks the real SessionDB source contract."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    def create_session(self, *args, **kwargs):
        return self._real.create_session(*args, **kwargs)

    def __getattr__(self, name):
        if name == "try_acquire_compression_lock":
            raise AttributeError(name)
        return getattr(self._real, name)


_NominalSessionDBImpostor.__module__ = "hermes_state"
_NominalSessionDBImpostor.__name__ = "SessionDB"


class _BrokenLockLookupDB:
    """A present lock API whose instance lookup fails unexpectedly."""

    def __init__(self, real_db: SessionDB, error: Exception) -> None:
        self._real = real_db
        self._error = error

    def try_acquire_compression_lock(self, *_args, **_kwargs):
        raise AssertionError("the broken lookup must not resolve to a callable")

    def __getattribute__(self, name):
        if name == "try_acquire_compression_lock":
            raise object.__getattribute__(self, "_error")
        if name in {"_real", "_error", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)


class _NonCallableLockAPI:
    """A present lock API descriptor that resolves to a non-callable value."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    try_acquire_compression_lock = None

    def __getattr__(self, name):
        return getattr(self._real, name)










@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("simulated lock-table corruption"),
        AttributeError("simulated internal lock attribute error"),
        TypeError("simulated internal lock type error"),
    ],
)
def test_real_lock_api_internal_errors_fail_closed_skips_compression(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    """Errors after a real lock API resolves must preserve session lineage.

    ``AttributeError`` only means version skew while resolving the method. This
    test injects failures beneath the real ``SessionDB.try_acquire...`` body,
    proving that an internal AttributeError or TypeError cannot take the
    structural-absence compatibility path.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ERRORING_LOCK_TEST"
    db.create_session(parent_sid, source="discord")

    def _fail_lock_write(_fn):
        raise error

    monkeypatch.setattr(db, "_execute_write", _fail_lock_write)
    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Skipped: messages returned verbatim, no rotation, compressor never ran.
    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()




def test_review_fork_disables_compression_to_prevent_stale_parent_fork(tmp_path: Path) -> None:
    """The background-review fork must set ``compression_enabled = False``
    so it can never compress the parent it shares a session_id with
    (issue #38727).

    The per-session compression lock only serialises a SAME-WINDOW concurrent
    race. It does NOT stop a stale parent from being compressed again in a
    LATER turn: if ``review_agent`` had won the race, its new child session is
    never adopted by the gateway (the fork is single-lifecycle and dies right
    after one ``run_conversation``), so the foreground path would start the
    next turn from the stale parent and compress it AGAIN — leaving the same
    parent with two sibling children.

    The fix makes the review fork never trigger compression at all. Both
    compression trigger sites in ``agent/conversation_loop.py`` gate on
    ``agent.compression_enabled`` BEFORE calling ``_compress_context``:
      • preflight (``if agent.compression_enabled and len(messages) > ...``)
      • mid-loop  (``if agent.compression_enabled and _compressor.should_compress(...)``)
    so a fork with the flag cleared never reaches the rotation path.

    This test pins the contract at the source: ``_run_review_in_thread``
    must set ``review_agent.compression_enabled = False`` on the fork it
    builds. It calls the real worker synchronously with
    ``AIAgent.run_conversation`` patched (so no LLM call happens) and
    captures the constructed review agent to assert the flag.
    """
    import agent.background_review as br

    captured = {}

    def _fake_run_conversation(self, *_a, **_k):
        captured["compression_enabled"] = self.compression_enabled
        captured["session_id"] = self.session_id
        return {"final_response": "", "messages": []}

    parent_sid = "REVIEW_FORK_FLAG_TEST"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    parent = _build_agent_with_db(db, parent_sid)

    # The worker does a local ``from run_agent import AIAgent``; patching
    # the class method covers that import path.
    from run_agent import AIAgent

    with patch.object(AIAgent, "run_conversation", _fake_run_conversation):
        br._run_review_in_thread(
            parent,
            [{"role": "user", "content": "hi"}],
            "review this conversation",
        )

    assert captured, (
        "_run_review_in_thread never reached run_conversation — the spawn path "
        "changed; update this test to capture the review AIAgent."
    )
    assert captured["session_id"] == parent_sid, (
        "Review fork should inherit the parent's session_id (shared id is the "
        "whole reason compression must be disabled)."
    )
    assert captured["compression_enabled"] is False, (
        "FIX REGRESSION: background-review fork did NOT disable compression. "
        "It shares the parent's session_id, so an enabled fork can rotate the "
        "parent into an orphan child (issue #38727). The trigger gates in "
        "conversation_loop.py only short-circuit when compression_enabled is "
        "False — this flag MUST be cleared on the review fork."
    )
    db.close()


# ── Lease-refresher bounded-failure tolerance (salvage follow-up, #54465) ────
# A single falsy refresh (transient DB blip) must NOT permanently kill the
# lease — only a *persistent* failure (genuine lost-ownership) should stop the
# refresher after a bounded number of consecutive failures. Without this, one
# escaped lock-contention error silently reintroduces the TTL-expiry wedge the
# PR set out to fix.


class _FlakyRefreshDB:
    """A db whose refresh_compression_lock returns a scripted sequence."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return True  # steady-state success after the scripted prefix


def _no_sleep(refresher) -> None:
    """Make the refresher loop iterate without real wall-clock sleeps.

    ``_stop.wait(interval)`` returns False (keep looping) instantly instead of
    blocking for the (clamped) interval, so count-based tests stay fast and
    deterministic — the loop's termination is driven by the failure cap / the
    scripted db, not by timing.
    """
    refresher._stop.wait = lambda _interval: False  # type: ignore[assignment]








def test_lease_refresher_failure_window_is_bounded_by_ttl() -> None:
    """Persistent failure stops within one lease's worth of time, not forever.

    The contract (not a magic count): the give-up window
    ``cap * refresh_interval`` must be <= the TTL, so a stuck refresher can
    never hold the lock past its TTL. We assert that relationship directly
    rather than freezing a literal cap (behavior contract over snapshot).
    """
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    ttl, interval = 10.0, 2.0  # cap should be int(10/2) = 5
    db = _FlakyRefreshDB([False] * 50)  # never recovers (lost ownership)
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=ttl, refresh_interval_seconds=interval
    )
    _no_sleep(refresher)
    refresher._run()

    cap = refresher._max_consecutive_failures
    assert cap == int(ttl / interval), "cap must derive from ttl/interval"
    # Stops at the cap — not on the first failure, not forever.
    assert db.calls == cap
    # The invariant that makes the cap honest: total tolerance <= one TTL.
    assert cap * interval <= ttl, (
        f"give-up window {cap * interval}s must not exceed the lease TTL {ttl}s"
    )


def test_hard_interrupt_aborts_compression_and_unblocks_session_writes(tmp_path: Path) -> None:
    """Ctrl+C must abort an interrupt-protected summary without leaving the
    session write-blocked behind its compression lease."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    original_messages = copy.deepcopy(messages)

    def _cancelled_compress(*_args, **_kwargs):
        agent._hard_interrupt_requested.set()
        assert aux._aux_interrupt_cancel_requested() is True
        messages[0]["content"] = "must be rolled back"
        raise aux.AuxiliaryExplicitCancellation()

    agent.context_compressor.compress.side_effect = _cancelled_compress

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == original_messages
    assert messages == original_messages
    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_late_hard_interrupt_restores_full_compressor_attempt_state_and_retry(
    tmp_path: Path,
) -> None:
    """A stop after provider success but before compress() returns is a true no-op."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "LATE_HARD_INTERRUPT_STATE_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    provider_returned = threading.Event()
    allow_compress_return = threading.Event()
    shared_telemetry = {"shared": [1, 2, 3]}
    state_fields = {
        "_previous_summary": "old-summary",
        "_summary_has_user_turn": False,
        "compression_count": 4,
        "_last_compression_savings_pct": 37.5,
        "_ineffective_compression_count": 1,
        "_anti_thrash_recovery_deadline": 123.0,
        "_fallback_compression_streak": 1,
        "_verify_compaction_cleared_threshold": False,
        "_last_compression_made_progress": False,
        "_summary_failure_cooldown_until": 456.0,
        "_cooldown_persist_failed": True,
        "_last_summary_error": "old-error",
        "_consecutive_timeout_failures": 2,
        "_last_summary_dropped_count": 3,
        "_last_summary_fallback_used": True,
        "_last_compress_aborted": False,
        "_last_summary_auth_failure": True,
        "_last_summary_network_failure": True,
        "_last_aux_model_failure_error": "old-aux-error",
        "_last_aux_model_failure_model": "old-aux-model",
        "_summary_model_fallen_back": True,
        "summary_model": "old-summary-model",
        "_last_compression_telemetry": shared_telemetry,
        "_active_compression_telemetry": shared_telemetry,
        "_compression_telemetry_seed": {"seed": [3]},
    }
    for name, value in state_fields.items():
        setattr(agent.context_compressor, name, copy.deepcopy(value))
    restored_shared_telemetry = copy.deepcopy(shared_telemetry)
    agent.context_compressor._last_compression_telemetry = restored_shared_telemetry
    agent.context_compressor._active_compression_telemetry = restored_shared_telemetry

    def _provider_succeeded_then_waits(*_args, **_kwargs):
        for name in state_fields:
            setattr(agent.context_compressor, name, f"mutated-{name}")
        provider_returned.set()
        assert allow_compress_return.wait(timeout=5)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] cancelled summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _provider_succeeded_then_waits
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert provider_returned.wait(timeout=2)
    agent.hard_interrupt("cancel after provider return")
    allow_compress_return.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    assert {
        name: copy.deepcopy(getattr(agent.context_compressor, name))
        for name in state_fields
    } == state_fields
    assert (
        agent.context_compressor._active_compression_telemetry
        is agent.context_compressor._last_compression_telemetry
    )
    assert db.get_compression_lock_holder(session_id) is None

    agent.clear_interrupt()
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )
    assert retried is not messages
    assert retried[0]["content"] == "[CONTEXT COMPACTION] retry summary"


def test_force_cancel_restores_newer_durable_cooldown_captured_under_lease(
    tmp_path: Path,
) -> None:
    """A stale forced attempt rolls back to the lease-protected durable row."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "FORCE_CANCEL_DURABLE_COOLDOWN"
    db.create_session(session_id, source="cli")

    # B binds first and therefore has no local cooldown. A then persists a
    # newer cooldown for the same durable session before B acquires its lease.
    stale_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    writer_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    stale = stale_agent.context_compressor
    writer = writer_agent.context_compressor
    assert isinstance(stale, ContextCompressor)
    assert isinstance(writer, ContextCompressor)
    assert stale._summary_failure_cooldown_until == 0.0

    writer._record_compression_failure_cooldown(120.0, "newer durable failure")
    durable_before = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_before[0] is not None

    stale_seed = {"seed": ["truly-pre-attempt"]}
    stale._compression_telemetry_seed = copy.deepcopy(stale_seed)
    stale._previous_summary = "pre-attempt-summary"
    stale_agent._compression_feasibility_checked = True
    stale_agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _clear_then_hard_cancel() -> None:
        real_clear(stale)
        stale_agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    # Exercise the built-in force=True mutation point deterministically: force
    # clears the durable cooldown, then the frozen host cancellation unwinds it.
    stale._clear_compression_failure_cooldown = _clear_then_hard_cancel

    compressed, _prompt = stale_agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    durable_after = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_after == durable_before
    assert stale._summary_failure_cooldown_until > time.monotonic()
    assert stale._last_summary_error == "newer durable failure"
    assert stale._cooldown_persist_failed is False
    assert stale._compression_telemetry_seed == stale_seed
    assert stale._previous_summary == "pre-attempt-summary"
    assert db.get_compression_lock_holder(session_id) is None

    # A future compressor refresh must still observe the exact row rather than
    # the cancelled force attempt having permanently cleared it.
    future_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    future = future_agent.context_compressor.get_active_compression_failure_cooldown(
        refresh=True
    )
    assert future is not None
    assert future["error"] == "newer durable failure"


def test_unrelated_interrupted_error_propagates_and_releases_compression_lease(
    tmp_path: Path,
) -> None:
    """A plugin/OS InterruptedError is a failure, not an explicit transaction abort."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "UNRELATED_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _provider_interrupted(*_args, **_kwargs):
        messages[0]["content"] = "must be rolled back"
        raise InterruptedError("provider syscall interrupted")

    agent.context_compressor.compress.side_effect = _provider_interrupted

    with pytest.raises(InterruptedError, match="provider syscall interrupted"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_redirect_interrupt_remains_protected_during_compression(tmp_path: Path) -> None:
    """Redirects use interrupt_requested=True/message=None; only the atomic
    hard-cancel event may override summary protection."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "REDIRECT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._interrupt_requested = True
    agent._interrupt_message = None
    agent._pending_redirect = "new correction"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _protected_noop(current, **_kwargs):
        assert aux._aux_interrupt_cancel_requested() is False
        return copy.deepcopy(current)

    agent.context_compressor.compress.side_effect = _protected_noop

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == messages
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_cancel_between_compress_return_and_commit_begin_wins_atomically(
    tmp_path: Path,
) -> None:
    """The hard-stop admission and commit admission share one fence lock."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_COMMIT_RACE"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    before_commit = threading.Event()
    allow_commit_check = threading.Event()

    class _CommitBarrierList(list):
        def __eq__(self, other):
            before_commit.set()
            assert allow_commit_check.wait(timeout=5)
            return super().__eq__(other)

    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: _CommitBarrierList(
        [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
    )
    archive_spy = MagicMock(wraps=db.archive_and_compact)
    db.archive_and_compact = archive_spy
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert before_commit.wait(timeout=2)

    agent.hard_interrupt("cancel before commit admission")
    allow_commit_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    archive_spy.assert_not_called()
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_stop_waits_for_commit_already_admitted(tmp_path: Path) -> None:
    """A surfaced stop never races an untracked post-return transcript commit."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_AFTER_COMMIT_ADMISSION"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    commit_started = threading.Event()
    allow_commit = threading.Event()
    stop_returned = threading.Event()
    original_archive = db.archive_and_compact

    def _blocked_archive(*args, **kwargs):
        commit_started.set()
        assert allow_commit.wait(timeout=5)
        return original_archive(*args, **kwargs)

    db.archive_and_compact = _blocked_archive
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compression_result: dict[str, tuple] = {}
    compression = threading.Thread(
        target=lambda: compression_result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    compression.start()
    assert commit_started.wait(timeout=2)

    stop = threading.Thread(
        target=lambda: (
            agent.hard_interrupt("stop after commit admission"),
            stop_returned.set(),
        ),
        daemon=True,
    )
    stop.start()
    assert not stop_returned.wait(timeout=0.1)
    allow_commit.set()
    compression.join(timeout=5)
    stop.join(timeout=5)

    assert not compression.is_alive()
    assert not stop.is_alive()
    assert stop_returned.is_set()
    assert compression_result["value"][0][0]["content"] == (
        "[CONTEXT COMPACTION] summary"
    )
    assert agent._hard_interrupt_requested.is_set()
    assert db.get_compression_lock_holder(session_id) is None


@pytest.mark.parametrize("deadline_offset", [-10.0, 0.05, None])
def test_force_cancel_restores_exact_expired_or_expiring_cooldown_row(
    tmp_path: Path,
    deadline_offset: float | None,
) -> None:
    """Cancellation preserves raw cooldown columns even after their deadline."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = f"RAW_COOLDOWN_{deadline_offset}"
    db.create_session(session_id, source="cli")
    deadline = time.time() + deadline_offset if deadline_offset is not None else None
    db.restore_compression_failure_cooldown_row(
        session_id,
        {
            "session_exists": True,
            "cooldown_until": deadline,
            "error": "expired-but-exact",
        },
    )
    before = db.get_compression_failure_cooldown_row(session_id)

    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    # A stale local persistence-failure marker must not suppress restoration
    # once the raw durable row was captured authoritatively under the lease.
    compressor._cooldown_persist_failed = True
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        if deadline_offset is not None and deadline_offset > 0:
            assert deadline is not None
            while time.time() <= deadline:
                time.sleep(0.005)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    compressed, _prompt = agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    assert db.get_compression_failure_cooldown_row(session_id) == before
    assert db.get_compression_lock_holder(session_id) is None


def test_cooldown_rollback_failure_surfaces_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compensating write cannot masquerade as a mutation-free cancel."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_WRITE_FAILURE"
    db.create_session(session_id, source="cli")
    db.record_compression_failure_cooldown(
        session_id,
        time.time() + 120.0,
        "must-restore",
    )
    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    def _rollback_write_fails(_self, _session_id, _snapshot) -> None:
        raise sqlite3.OperationalError("forced rollback write failure")

    monkeypatch.setattr(
        SessionDB,
        "restore_compression_failure_cooldown_row",
        _rollback_write_fails,
    )

    with pytest.raises(sqlite3.OperationalError, match="forced rollback write failure"):
        agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            force=True,
        )

    assert db.get_compression_lock_holder(session_id) is None


def test_exact_cooldown_restore_api_propagates_sqlite_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "RAW_COOLDOWN_WRITE_FAILURE"
    db.create_session(session_id, source="test")

    def _write_fails(_callback) -> None:
        raise sqlite3.OperationalError("forced low-level write failure")

    monkeypatch.setattr(db, "_execute_write", _write_fails)

    with pytest.raises(sqlite3.OperationalError, match="forced low-level write failure"):
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": time.time() + 10.0,
                "error": "must propagate",
            },
        )
