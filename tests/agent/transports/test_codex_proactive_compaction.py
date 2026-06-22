"""Tests for proactive context-pressure retire + handoff on the Codex
app-server runtime path (issue #36801).

The codex subprocess owns its own conversation context, so Hermes cannot
compress it in place — the guard's job is to detect high context occupancy
*after* a turn, summarize Hermes' projection, retire the thread, and stash the
summary so the next turn reseeds a fresh thread instead of cold-resetting.
"""

import types

import pytest

from agent.codex_runtime import _maybe_retire_codex_for_context
from agent.context_compressor import ContextCompressor


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_agent(*, prompt_tokens, window, threshold=0.85, enabled=True,
                summary="GOAL: ship\nFILES: a.py"):
    session = _FakeSession()
    compressor = types.SimpleNamespace(
        build_handoff_summary=lambda messages: summary,
    )
    agent = types.SimpleNamespace(
        compression_enabled=enabled,
        codex_retire_threshold=threshold,
        context_compressor=compressor,
        _codex_session=session,
        _codex_pending_handoff=None,
    )
    turn = types.SimpleNamespace(model_context_window=window)
    usage_result = {"prompt_tokens": prompt_tokens}
    return agent, turn, usage_result, session


def test_retires_and_stashes_summary_when_over_threshold():
    agent, turn, usage, session = _make_agent(prompt_tokens=90_000, window=100_000)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is True
    assert agent._codex_session is None
    assert agent._codex_pending_handoff == "GOAL: ship\nFILES: a.py"


def test_no_retire_when_under_threshold():
    agent, turn, usage, session = _make_agent(prompt_tokens=50_000, window=100_000)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is False
    assert agent._codex_session is session
    assert agent._codex_pending_handoff is None


def test_retires_clean_when_summary_unavailable():
    agent, turn, usage, session = _make_agent(prompt_tokens=95_000, window=100_000, summary=None)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is True
    assert agent._codex_session is None
    # No seed stashed → next turn starts clean (documented fallback).
    assert agent._codex_pending_handoff is None


def test_noop_when_compression_disabled():
    agent, turn, usage, session = _make_agent(prompt_tokens=99_000, window=100_000, enabled=False)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is False
    assert agent._codex_session is session


def test_noop_when_threshold_zero_disables():
    agent, turn, usage, session = _make_agent(prompt_tokens=99_000, window=100_000, threshold=0)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is False


def test_noop_when_window_unknown():
    agent, turn, usage, session = _make_agent(prompt_tokens=99_000, window=None)
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert session.closed is False


def test_noop_when_session_already_retired():
    agent, turn, usage, _session = _make_agent(prompt_tokens=99_000, window=100_000)
    agent._codex_session = None  # e.g. crash/should_retire path already closed it
    # Must not raise even though there is no session to close.
    _maybe_retire_codex_for_context(agent, turn, usage, messages=[{"role": "user", "content": "hi"}])
    assert agent._codex_pending_handoff is None


def test_build_handoff_summary_empty_returns_none():
    compressor = object.__new__(ContextCompressor)  # bypass heavy __init__
    assert compressor.build_handoff_summary([]) is None


def test_build_handoff_summary_delegates_to_generate_summary():
    compressor = object.__new__(ContextCompressor)
    compressor._generate_summary = lambda msgs: "SUMMARY"  # type: ignore[method-assign]
    out = compressor.build_handoff_summary([{"role": "user", "content": "hello"}])
    assert out == "SUMMARY"


def test_build_handoff_summary_swallows_generator_errors():
    compressor = object.__new__(ContextCompressor)

    def _boom(_msgs):
        raise RuntimeError("aux model down")

    compressor._generate_summary = _boom  # type: ignore[method-assign]
    assert compressor.build_handoff_summary([{"role": "user", "content": "x"}]) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
