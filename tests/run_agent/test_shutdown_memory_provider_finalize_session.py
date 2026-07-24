"""Regression tests for AIAgent.shutdown_memory_provider's finalize_session param.

Gateway soft cache eviction of a finalizable session runs two steps back to
back: commit_memory_session() (fires on_session_end() to extract memory
without a transport teardown), then shutdown_memory_provider() (tears down
the transport). Before finalize_session existed, the second call ALSO fired
on_session_end() — double-finalizing the same transcript, which double-ingests
it into any external memory backend that treats session-end as a one-shot
event (e.g. full-session ingestion). finalize_session=False lets the second
call skip on_session_end() and only tear down the transport.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_minimal_agent(memory_manager, context_compressor, session_id="abc"):
    """Build an object with just enough surface for shutdown_memory_provider
    (and, where used, commit_memory_session) to run — AIAgent.__init__ is too
    heavy for a focused unit test."""
    from run_agent import AIAgent

    obj = SimpleNamespace(
        _memory_manager=memory_manager,
        context_compressor=context_compressor,
        session_id=session_id,
    )
    obj.shutdown_memory_provider = AIAgent.shutdown_memory_provider.__get__(obj)
    obj.commit_memory_session = AIAgent.commit_memory_session.__get__(obj)
    return obj


def test_default_finalize_session_true_preserves_prior_behavior():
    """No finalize_session kwarg == full finalization, unchanged from before
    the parameter existed (CLI exit, /reset, gateway session expiry, etc.)."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-1")

    msgs = [{"role": "user", "content": "hi"}]
    agent.shutdown_memory_provider(msgs)

    mm.on_session_end.assert_called_once_with(msgs)
    mm.shutdown_all.assert_called_once_with()
    ctx.on_session_end.assert_called_once_with("sess-1", msgs)


def test_finalize_session_false_skips_on_session_end_but_tears_down_transport():
    """finalize_session=False: on_session_end() is skipped on both the memory
    manager and the context engine, but shutdown_all() (transport teardown)
    still fires — a resumable soft eviction still frees the client/keepalive
    resources, it just doesn't re-finalize the session."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-2")

    msgs = [{"role": "user", "content": "hi"}]
    agent.shutdown_memory_provider(msgs, finalize_session=False)

    mm.on_session_end.assert_not_called()
    mm.shutdown_all.assert_called_once_with()
    ctx.on_session_end.assert_not_called()


def test_ordinary_resumable_eviction_calls_on_session_end_zero_times():
    """A resumable soft eviction (no prior commit_memory_session) must call
    on_session_end() zero times — cal88's suggested case #1. It is not a real
    session boundary, so nothing should be finalized at all."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-3")

    # Soft eviction path: only the transport-only call runs, exactly as
    # gateway/run.py's _release_evicted_agent_soft does for every eviction.
    agent.shutdown_memory_provider(finalize_session=False)

    mm.on_session_end.assert_not_called()
    ctx.on_session_end.assert_not_called()
    mm.shutdown_all.assert_called_once_with()


def test_finalizable_eviction_calls_on_session_end_exactly_once():
    """A finalizable LRU-cap eviction runs commit_memory_session() (fires
    on_session_end) THEN shutdown_memory_provider(finalize_session=False)
    (transport-only) — cal88's suggested case #2. on_session_end() must fire
    exactly once total across both calls, not twice."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-4")

    msgs = [{"role": "user", "content": "hi"}]
    # Mirrors gateway/run.py's _commit_then_release_soft: commit first,
    # then the transport-only release.
    agent.commit_memory_session(msgs)
    agent.shutdown_memory_provider(msgs, finalize_session=False)

    assert mm.on_session_end.call_count == 1
    assert ctx.on_session_end.call_count == 1
    mm.shutdown_all.assert_called_once_with()


def test_repeated_transport_only_shutdown_is_idempotent_re_finalization():
    """Calling the transport-only path twice (e.g. a retried eviction) must
    never fire on_session_end() — cal88's suggested case #4. shutdown_all()
    firing twice is the provider's own responsibility to tolerate; this
    only asserts finalization itself stays at zero."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-5")

    agent.shutdown_memory_provider(finalize_session=False)
    agent.shutdown_memory_provider(finalize_session=False)

    mm.on_session_end.assert_not_called()
    ctx.on_session_end.assert_not_called()
    assert mm.shutdown_all.call_count == 2


def test_finalize_session_false_tolerates_missing_memory_manager():
    """No external memory provider configured — finalize_session=False must
    not raise, and the context engine hook is still skipped."""
    ctx = MagicMock()
    agent = _make_minimal_agent(None, ctx, session_id="sess-6")

    agent.shutdown_memory_provider(finalize_session=False)

    ctx.on_session_end.assert_not_called()
