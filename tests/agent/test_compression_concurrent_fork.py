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

import inspect
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
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
    compressor = MagicMock()

    def _compress_with_overlap(*_a, **_kw):
        time.sleep(0.25)
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

    # The number of agents that rotated their session_id must match the number
    # of children created — and must never exceed one. (Both rotating would be
    # the fork; the winner rolling back to parent under contention yields zero,
    # which agrees with zero children.)
    rotated = sum(
        1 for a in (agent_a, agent_b) if a.session_id != parent_sid
    )
    assert rotated <= 1, (
        f"Expected at most one agent to rotate session_id, got {rotated}. "
        "More than one rotating means the lock didn't serialize them."
    )
    assert rotated == n_children, (
        f"Inconsistent state: {rotated} agent(s) rotated but {n_children} "
        "child session(s) exist — rotation and child creation diverged."
    )

    # The lock must be released after both paths finished, regardless of
    # whether the winner committed a child or rolled back.
    assert db.get_compression_lock_holder(parent_sid) is None, (
        "Compression lock leaked: still held after both paths completed."
    )


def test_skipped_compression_returns_messages_unchanged(tmp_path: Path) -> None:
    """The loser of the lock race must return its input messages verbatim.

    Callers (preflight compression in ``conversation_loop.py``) detect the
    no-op via ``len(returned) == len(input)`` and stop the auto-compress
    retry loop.  If the skipped path returned the compressed view, that
    detection would break and the caller would mutate the conversation
    without going through state.db rotation.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "LOSER_TEST"
    db.create_session(parent_sid, source="discord")

    # Pre-acquire the lock so the agent's compress_context sees it held.
    held = db.try_acquire_compression_lock(parent_sid, "external_holder")
    assert held is True

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": "m1"}, {"role": "user", "content": "m2"}]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Skipped: messages returned verbatim, no rotation
    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    # Compressor was never called (the skip happens before .compress())
    agent.context_compressor.compress.assert_not_called()


def test_compression_restores_user_turn_when_compressor_drops_all_users(tmp_path: Path) -> None:
    """Provider chat templates need at least one user message after compaction.

    A plugin or future compressor can legally return a compacted context made
    only of assistant/tool summary rows.  Before the guard in
    ``compress_context``, that transcript went straight into the next API call;
    LM Studio / llama.cpp Jinja templates then failed with "No user query found
    in messages."  Preserve the last real user turn from the pre-compression
    transcript instead of inventing a new active request.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "NO_USER_AFTER_COMPRESS"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION] earlier work was summarized",
        }
    ]
    messages = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "please continue from here"},
        {"role": "assistant", "content": "working"},
    ]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    user_messages = [msg for msg in compressed if msg.get("role") == "user"]
    assert user_messages == [{"role": "user", "content": "please continue from here"}]


def test_lock_refresh_keeps_owner_live_past_initial_ttl(tmp_path: Path, monkeypatch) -> None:
    """The owning compression call must keep its lease alive while it runs."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=1.0)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "REFRESH_TEST"
    db.create_session(parent_sid, source="discord")

    agent_a = _build_agent_with_db(db, parent_sid)
    agent_a._compression_lock_ttl_seconds = 1.0
    agent_a._compression_lock_refresh_interval = 0.25
    compression_started = threading.Event()
    release_compression = threading.Event()

    def _slow_compress(*_a, **_kw):
        compression_started.set()
        assert release_compression.wait(timeout=10)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent_a.context_compressor.compress.side_effect = _slow_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def run(agent):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    t_a = threading.Thread(target=run, args=(agent_a,), name="refresh_owner")
    t_a.start()
    try:
        assert compression_started.wait(timeout=10), "compression never acquired its lock"
        assert db.get_compression_lock_holder(parent_sid) is not None
        time.sleep(1.2)
        assert db.try_acquire_compression_lock(
            parent_sid, "refresh_probe", ttl_seconds=1.0
        ) is False, "live owner lease expired and was reclaimable before compression finished"
    finally:
        release_compression.set()
        t_a.join(timeout=10)

    assert not t_a.is_alive()
    assert _count_children(db, parent_sid) == 1
    assert db.get_compression_lock_holder(parent_sid) is None


def test_post_compress_exception_stops_lock_refresher(tmp_path: Path, monkeypatch) -> None:
    """A warning-path exception after compress() returns must still release the lock."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=1.0)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 1.0
    agent._compression_lock_refresh_interval = 0.1
    agent.context_compressor._last_summary_error = "summary failed"
    agent._emit_warning = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("warn boom"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="warn boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    time.sleep(1.3)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True


def test_abort_warning_exception_stops_lock_refresher(tmp_path: Path, monkeypatch) -> None:
    """An abort-path warning exception must still release the refreshed lock."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=1.0)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_ABORT_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 1.0
    agent._compression_lock_refresh_interval = 0.1

    def _aborting_compress(*_a, **_kw):
        agent.context_compressor._last_compress_aborted = True
        agent.context_compressor._last_summary_error = "summary failed"
        return [{"role": "user", "content": "tail"}]

    agent.context_compressor.compress.side_effect = _aborting_compress
    agent._emit_warning = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("abort boom"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="abort boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    time.sleep(1.3)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True


def test_internal_typeerror_stops_lock_refresher_without_retry(tmp_path: Path, monkeypatch) -> None:
    """An engine TypeError must release the refreshed lock without a second call."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=1.0)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_TYPEERROR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 1.0
    agent._compression_lock_refresh_interval = 0.1

    calls = []

    def _internal_typeerror(*_a, **_kw):
        calls.append(_kw)
        raise TypeError("engine implementation bug")

    agent.context_compressor.compress.side_effect = _internal_typeerror

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(TypeError, match="engine implementation bug"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert len(calls) == 1
    time.sleep(1.3)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True


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


def test_missing_lock_subsystem_fails_open_not_infinite_loop(tmp_path: Path, monkeypatch) -> None:
    """A truly old in-memory SessionDB class must still make progress.

    A module reload can update ``conversation_compression`` while the cached
    ``hermes_state.SessionDB`` class remains pre-lock. The compatibility path is
    only valid for that exact class identity, not a proxy that merely uses the
    same name.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "SKEW_TEST_SESSION"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    legacy_type = _make_legacy_session_db_class()
    import hermes_state

    real_session_db_type = hermes_state.SessionDB
    monkeypatch.setattr(hermes_state, "SessionDB", legacy_type)
    try:
        # The same module now exposes its genuinely old SessionDB class; its
        # instance forwards persistence/rotation operations to a real database.
        agent._session_db = legacy_type(db)
        monkeypatch.setattr(
            "agent.conversation_compression._CompressionLockLeaseRefresher",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("lock refresher should not start on fail-open lock skew")
            ),
        )
        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)
    finally:
        monkeypatch.setattr(hermes_state, "SessionDB", real_session_db_type)

    assert agent.context_compressor.compress.call_count == 1
    assert len(compressed) < len(messages), (
        "Compression made no progress despite failing open — loop would still spin."
    )
    assert agent.session_id != parent_sid


def test_nominal_sessiondb_impostor_fails_closed(tmp_path: Path) -> None:
    """A name/module-spoofing proxy is not the legacy SessionDB compatibility case."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "NOMINAL_SESSIONDB_IMPOSTOR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._session_db = _NominalSessionDBImpostor(db)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()


def test_noncallable_lock_api_fails_closed(tmp_path: Path) -> None:
    """A present but non-callable lock API is not legacy version skew."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "NONCALLABLE_LOCK_API_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._session_db = _NonCallableLockAPI(db)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("simulated lock lookup failure"),
        AttributeError("simulated lock lookup attribute error"),
        TypeError("simulated lock lookup type error"),
    ],
)
def test_nonmissing_lock_lookup_errors_fail_closed(
    tmp_path: Path, error: Exception
) -> None:
    """Only AttributeError for an absent API may use the compatibility path."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "BROKEN_LOCK_LOOKUP_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._session_db = _BrokenLockLookupDB(db, error)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()


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


def test_post_acquire_error_releases_owned_lock(tmp_path: Path, monkeypatch) -> None:
    """A failure after acquisition commits must not strand the holder lease."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "POST_ACQUIRE_ERROR_TEST"
    db.create_session(parent_sid, source="discord")

    original_acquire = db.try_acquire_compression_lock

    def _acquire_then_raise(session_id, holder, ttl_seconds=300.0):
        assert original_acquire(session_id, holder, ttl_seconds=ttl_seconds) is True
        raise RuntimeError("simulated post-acquire failure")

    monkeypatch.setattr(db, "try_acquire_compression_lock", _acquire_then_raise)
    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    assert db.get_compression_lock_holder(parent_sid) is None
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


def test_lease_refresher_survives_single_transient_failure() -> None:
    """One False (transient blip) followed by success must NOT stop the loop.

    Regression for the W1/W2 finding: the original ``if not refreshed: break``
    treated a one-off failure identically to genuine lost-ownership, killing
    the lease on the first hiccup.
    """
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    # Script: success, FAILURE (blip), success, then stop the loop externally.
    db = _FlakyRefreshDB([True, False, True])
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=10.0, refresh_interval_seconds=0.001
    )
    # Stop after exactly 4 ticks (3 scripted + 1 steady success), no real sleep.
    refresher._stop.wait = lambda _i: db.calls >= 4  # type: ignore[assignment]
    refresher._run()

    # The single False at call 2 must NOT have ended the loop — we keep going
    # past it (calls reach >= 4), proving the blip was tolerated.
    assert db.calls >= 4, (
        "Lease refresher stopped after a single transient failure — the "
        "bounded-tolerance fix regressed (one blip must not kill the lease)."
    )


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


def test_lease_refresher_failure_cap_has_floor_of_one() -> None:
    """A degenerate interval >= ttl still tolerates exactly one blip (floor 1)."""
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    db = _FlakyRefreshDB([False] * 10)
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=1.0, refresh_interval_seconds=5.0
    )
    _no_sleep(refresher)
    refresher._run()
    assert refresher._max_consecutive_failures == 1
    assert db.calls == 1


def test_lease_refresher_recovers_after_raise() -> None:
    """A raise treated as a failure tick must RESET on a later success — the
    exception arm gets the same blip-tolerance as a falsy return, not just a
    'doesn't crash' guarantee."""
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    class _RaiseThenOKDB:
        """Raise once, then succeed forever — the transient-blip analog."""

        def __init__(self):
            self.calls = 0

        def refresh_compression_lock(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated DB hiccup")
            return True

    db = _RaiseThenOKDB()
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=10.0, refresh_interval_seconds=2.0
    )
    # Run a handful of ticks past the raise, then stop.
    refresher._stop.wait = lambda _i: db.calls >= 4  # type: ignore[assignment]
    refresher._run()  # must not propagate the RuntimeError
    # Survived the raise and kept refreshing — the counter reset on recovery.
    assert db.calls >= 4


def test_lease_refresher_stops_on_persistent_raise() -> None:
    """A refresh that raises every tick is bounded by the same TTL-derived cap,
    never propagates, and never loops forever."""
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    class _AlwaysRaiseDB:
        def __init__(self):
            self.calls = 0

        def refresh_compression_lock(self, *a, **k):
            self.calls += 1
            raise RuntimeError("simulated DB hiccup")

    db = _AlwaysRaiseDB()
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=10.0, refresh_interval_seconds=2.0
    )
    _no_sleep(refresher)
    refresher._run()  # must not propagate
    assert db.calls == refresher._max_consecutive_failures
