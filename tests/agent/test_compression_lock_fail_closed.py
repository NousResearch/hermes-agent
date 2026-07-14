"""Regression tests for #63129 — compression lock must fail CLOSED on
transient errors, only failing OPEN on structural absence (version skew).

Background: ``agent/conversation_compression.py`` wraps
``_lock_db.try_acquire_compression_lock(...)`` in a single broad
``except Exception`` handler that ALWAYS sets ``_lock_acquired = True``
and proceeds with compression. The intent was to tolerate one specific
case — a version-skewed ``SessionDB`` that structurally lacks the lock
method — but as written the handler also fails open on *transient*
errors (e.g. ``sqlite3.OperationalError: database is locked``,
``RuntimeError`` from a flaky db connection). In that case two
compactors can run concurrently on the same session and fork it.

These tests assert the new split-handler behavior:
  - ``AttributeError`` / ``TypeError`` → fail OPEN (the version-skew
    case the original #34475 rationale was about — AttributeError when
    the method is missing, TypeError when the signature changed).
  - ANY OTHER ``Exception`` → fail CLOSED (skip compression this cycle;
    caller will retry next round).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Build an AIAgent wired to ``db`` and pinned to ``session_id``.

    Same shape as test_compression_concurrent_fork.py — see that file
    for context on why we patch context_compressor and pin
    compression_in_place=False.
    """
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

    compressor = MagicMock()

    def _compress_ok(*_a, **_kw):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress_ok
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # These tests cover the rotation path (transient-error fallback) —
    # pin in_place=False so they exercise that branch regardless of the
    # global default.
    agent.compression_in_place = False
    return agent


class _TransientErrorDB:
    """Simulates a SessionDB whose lock method raises a transient error.

    A ``sqlite3.OperationalError`` is the canonical transient error —
    it can happen if the underlying db is briefly locked by another
    writer, or if the connection has been interrupted. The compression
    code's lock must treat this as fail-CLOSED (skip compression this
    cycle), not fail-OPEN (proceed with rotation and risk fork).
    """

    def __init__(self, real_db: SessionDB, exc: BaseException) -> None:
        self._real = real_db
        self._exc = exc
        self.acquire_call_count = 0

    def try_acquire_compression_lock(self, *_a, **_k):
        self.acquire_call_count += 1
        raise self._exc

    def get_compression_lock_holder(self, *_a, **_k):
        raise self._exc

    def release_compression_lock(self, *_a, **_k):
        raise self._exc

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_sqlite_transient_error_fails_closed(tmp_path: Path) -> None:
    """#63129: ``sqlite3.OperationalError`` on try_acquire must NOT cause
    the code to proceed with compression. Lock is reported as
    NOT acquired, and the caller's ``len(returned) == len(input)`` check
    detects the skip.

    The unfixed code catches ``Exception`` and sets ``_lock_acquired = True``,
    so it proceeds with rotation despite the error — risking a fork.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "TRANSIENT_LOCK_ERROR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    transient_db = _TransientErrorDB(
        db, sqlite3.OperationalError("database is locked")
    )
    agent._session_db = transient_db

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    # Call _compress_context — should NOT raise (the broad except still
    # catches), but it should also NOT rotate the session (transient
    # error → fail closed → skip).
    # Returns (messages, system_prompt) tuple.
    result_messages, _system_prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    # Caller-side detection: input length == output length means "no
    # compression happened this cycle; retry next time."
    assert len(result_messages) == len(messages), (
        "transient lock error must skip compression this cycle "
        "(return messages unchanged) instead of proceeding to rotate"
    )

    # Critical: session_id must NOT have rotated. If it rotated, a
    # concurrent compactor could also have rotated and forked the
    # session.
    assert agent.session_id == parent_sid, (
        f"session_id must not rotate on transient lock error "
        f"(was {parent_sid!r}, now {agent.session_id!r})"
    )

    # The exception was raised at least once into the except block.
    assert transient_db.acquire_call_count >= 1


def test_runtime_error_fails_closed(tmp_path: Path) -> None:
    """Generalize: any Exception other than ``AttributeError`` /
    ``TypeError`` must fail closed. ``RuntimeError`` is a common
    transient shape (e.g. lock subsystem raised a generic runtime
    error)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "RUNTIME_ERROR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._session_db = _TransientErrorDB(db, RuntimeError("lock subsystem glitch"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    result_messages, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert len(result_messages) == len(messages), (
        "RuntimeError on try_acquire must skip compression this cycle"
    )
    assert agent.session_id == parent_sid


def test_attribute_error_still_fails_open(tmp_path: Path) -> None:
    """Regression guard: the version-skew path (lock method missing →
    AttributeError) must STILL fail OPEN. The split-handler fix must
    preserve the original #34475 rationale: AttributeError (and the
    sibling TypeError for stale signatures) means the lock subsystem
    is structurally absent, so skip locking and proceed with
    compression rather than spinning the outer loop forever."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ATTRIBUTE_ERROR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)

    class _MissingLockDB:
        def try_acquire_compression_lock(self, *_a, **_k):
            raise AttributeError(
                "'SessionDB' object has no attribute 'try_acquire_compression_lock'"
            )

        def get_compression_lock_holder(self, *_a, **_k):
            raise AttributeError(
                "'SessionDB' object has no attribute 'get_compression_lock_holder'"
            )

        def release_compression_lock(self, *_a, **_k):
            raise AttributeError(
                "'SessionDB' object has no attribute 'release_compression_lock'"
            )

        def __getattr__(self, name):
            return getattr(db, name)

    agent._session_db = _MissingLockDB()

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    # Should NOT raise — must fail OPEN and proceed.
    result_messages, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    # With fail-OPEN, compression proceeds: result length < input length
    # (the stub compressor returns 2 messages).
    assert len(result_messages) < len(messages), (
        "AttributeError on try_acquire must fail OPEN (proceed with "
        "compression) — version skew path, original #34475 rationale"
    )


def test_type_error_fails_open_stale_signature(tmp_path: Path) -> None:
    """TypeError means the lock method exists but its signature is stale
    (e.g. before ``ttl_seconds=`` was added). Same fail-OPEN rationale
    as AttributeError."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "TYPE_ERROR_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)

    class _StaleSignatureDB:
        def try_acquire_compression_lock(self, *_a, **_k):
            raise TypeError(
                "try_acquire_compression_lock() got an unexpected keyword argument 'ttl_seconds'"
            )

        def get_compression_lock_holder(self, *_a, **_k):
            return None  # works

        def release_compression_lock(self, *_a, **_k):
            pass

        def __getattr__(self, name):
            return getattr(db, name)

    agent._session_db = _StaleSignatureDB()

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    # Should NOT raise — must fail OPEN.
    result_messages, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert len(result_messages) < len(messages), (
        "TypeError on try_acquire must fail OPEN (proceed with "
        "compression) — stale signature is structural absence"
    )


def test_two_concurrent_compressors_with_transient_error_dont_fork(tmp_path: Path) -> None:
    """Integration check: when the lock subsystem raises a transient
    error, TWO concurrent compression calls must BOTH skip
    (len(result) == len(input)) rather than BOTH rotate and fork.

    With the unfixed broad-except, both would fail open and rotate →
    two children, exactly the bug-shape. With the fix, both skip →
    no rotation, no fork.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "CONCURRENT_TRANSIENT_TEST"
    db.create_session(parent_sid, source="discord")

    agent_a = _build_agent_with_db(db, parent_sid)
    agent_b = _build_agent_with_db(db, parent_sid)

    transient_db_a = _TransientErrorDB(
        db, sqlite3.OperationalError("database is locked")
    )
    transient_db_b = _TransientErrorDB(
        db, sqlite3.OperationalError("database is locked")
    )
    agent_a._session_db = transient_db_a
    agent_b._session_db = transient_db_b

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    errors: list[Exception] = []

    def run(agent):
        try:
            agent._compress_context(messages, "sys", approx_tokens=120_000)
        except Exception as exc:
            errors.append(exc)

    t_a = threading.Thread(target=run, args=(agent_a,))
    t_b = threading.Thread(target=run, args=(agent_b,))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # Neither compressor should raise — the broad except still catches
    # the transient error, the fix just changes the recovery path.
    assert not errors, f"_compress_context raised unexpectedly: {errors}"

    # Neither session_id should have rotated.
    assert agent_a.session_id == parent_sid, (
        f"agent_a session rotated on transient error: {agent_a.session_id!r}"
    )
    assert agent_b.session_id == parent_sid, (
        f"agent_b session rotated on transient error: {agent_b.session_id!r}"
    )

    # No children in state.db.
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?", (parent_sid,)
    ).fetchall()
    assert not rows, (
        f"transient-error path forked session: {len(rows)} child rows "
        f"in state.db (must be 0)"
    )
