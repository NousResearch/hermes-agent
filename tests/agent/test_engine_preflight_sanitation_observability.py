"""Engine-preflight sanitation observability: warn when no claim validates.

When a context engine's ``should_compress_preflight()`` returns True, the
turn-phase wire (``agent/turn_context_compaction.py``) flags
``agent._engine_preflight_requested`` and hands off to ``_compress_context``.
The engine is then supposed to hand a sanitation claim back through
``prepare_compression_operation()``. When it does not — stale handoff,
attempt-generation mismatch, replay divergence — compression previously fell
back to a full generic (summarizing) attempt with zero observability, silently
paying the summary cost and cache break the sanitation path exists to avoid.

Contracts pinned here:

* The wire sets ``_engine_preflight_requested`` around the engine-driven
  ``_compress_context`` pass and clears it in a ``finally``.
* A generic attempt that consumes the flag (prepared_operation is None, no
  retry candidate) logs the mismatch warning ONCE with session + attempt
  generation, and still completes generically.
* A claimed (pure-sanitation) attempt logs nothing.
* The flag never leaks into a later unrelated attempt.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import TurnContext
from tests.agent.test_engine_preflight_wire import (
    _build,
    _history,
    _make_agent,
    _stub_compressor,
)
from tests.agent.test_sanitation_commit import _make_harness


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


_COMPRESSOR_MINIMUM_STATE = (
    # The commit path reads these even on bare SimpleNamespace engines; missing
    # members break the attempt flow.
    "compression_count",
    "_last_compress_aborted",
    "_last_summary_error",
    "_last_compression_made_progress",
    "_last_summary_fallback_used",
    "_last_feasibility_skip",
)


def _preflight_compressor(compress=None):
    """Engine-shaped stub: preflight accepts, prepare never yields a claim."""
    comp = _stub_compressor(preflight=lambda _messages: True)
    for name in _COMPRESSOR_MINIMUM_STATE:
        if not hasattr(comp, name):
            setattr(comp, name, 0 if name == "compression_count" else False)
    comp._last_summary_error = None
    comp.prepare_compression_operation = lambda *args, **kwargs: None
    comp.compress = compress or (lambda messages, **kwargs: messages)
    return comp


def test_engine_preflight_flag_set_and_cleared_around_engine_pass():
    compressor = _preflight_compressor()
    agent = _make_agent(compressor)
    assert getattr(agent, "_engine_preflight_requested", False) is False

    ctx = _build(agent, conversation_history=_history())

    assert isinstance(ctx, TurnContext)
    assert agent._engine_preflight_requested is False
    assert agent._compress_context.called


def test_no_validated_claim_warns_once_and_compression_completes_generically(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Engine accepted preflight but handed back no claim: the generic attempt
    warns ONCE with session + attempt generation, and the attempt still runs
    its generic summary boundary end-to-end."""
    import agent.conversation_compression as compression
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    (tmp_path / "hermes-home").mkdir()
    db = SessionDB(tmp_path / "state.db")
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id="preflight-unclaimed",
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent._ensure_db_session()
    agent._flush_messages_to_session_db(_history(), [])

    generic_calls: list[int] = []

    def rewriting_compress(messages, current_tokens=None, focus_topic=None,
                           force=False, bypass_cooldown=False, operation_claim=None):
        generic_calls.append(1)
        return [
            {"role": "user", "content": "[summary] earlier state"},
            {"role": "assistant", "content": "retained tail"},
        ]

    compressor = _preflight_compressor(compress=rewriting_compress)
    agent.context_compressor = compressor
    # The turn's engine-driven preflight wire flags the request just before
    # _compress_context() forwards here.
    agent._engine_preflight_requested = True

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        returned, returned_prompt = compression.compress_context(
            agent,
            _history(),
            "system prompt",
            approx_tokens=1000,
        )

    warnings = [
        record
        for record in caplog.records
        if "Engine preflight requested sanitation but no claim validated"
        in record.getMessage()
    ]
    assert len(warnings) == 1
    assert f"session={agent.session_id}" in warnings[0].getMessage()
    assert "attempt_generation=" in warnings[0].getMessage()
    assert "compression ran generic" in warnings[0].getMessage()
    # The generic attempt actually ran and completed its summary boundary.
    assert len(generic_calls) == 1
    assert len(returned) == 2
    assert agent._last_compression_attempt_in_place is True
    assert agent._engine_preflight_requested is False


def test_claimed_sanitation_attempt_does_not_warn(tmp_path, monkeypatch, caplog):
    """A validated claim converts the attempt to pure sanitation: no generic
    warning, and the claimed result commits."""
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1, initial_status=None)
    engine = harness.agent.context_compressor
    real_prepare = engine.prepare_compression_operation
    prepared_claims: list[object] = []

    def prepare_with_claim(messages, **kwargs):
        prepared = real_prepare(messages, **kwargs)
        if prepared is not None:
            prepared_claims.append(prepared[1])
        return prepared

    engine.prepare_compression_operation = prepare_with_claim
    harness.agent._engine_preflight_requested = True

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        returned, _ = compression.compress_context(
            harness.agent,
            harness.messages,
            "system",
            approx_tokens=100_000,
        )

    assert not [
        record
        for record in caplog.records
        if "Engine preflight requested sanitation but no claim validated"
        in record.getMessage()
    ]
    assert len(prepared_claims) == 1
    from tests.agent.test_sanitation_commit import _without_persistence_markers

    assert _without_persistence_markers(returned) == _without_persistence_markers(
        harness.candidate
    )
    assert harness.agent._engine_preflight_requested is False


def test_flag_does_not_leak_into_a_later_attempt(tmp_path, monkeypatch, caplog):
    """The flag is per-attempt: the first (generic) attempt warns ONCE and the
    attempt-consumed signal is cleared in the attempt's finally, so a later
    unrelated generic attempt — with no engine preflight — stays silent."""
    import agent.conversation_compression as compression

    harness = _make_harness(
        tmp_path,
        rounds=1,
        status=None,
        initial_status=None,
        current_operation=None,
    )

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        # Attempt 1: the wire flagged a preflight request that never validated.
        harness.agent._engine_preflight_requested = True
        compression.compress_context(
            harness.agent,
            harness.messages,
            "system",
            approx_tokens=100_000,
        )
        first_warnings = [
            record
            for record in caplog.records
            if "Engine preflight requested sanitation but no claim validated"
            in record.getMessage()
        ]
        assert len(first_warnings) == 1
        assert harness.agent._engine_preflight_requested is False

        # Attempt 2: no new preflight, no leaked flag — no warning.
        caplog.clear()
        compression.compress_context(
            harness.agent,
            harness.messages,
            "system",
            approx_tokens=100_000,
        )
        assert not [
            record
            for record in caplog.records
            if "Engine preflight requested sanitation but no claim validated"
            in record.getMessage()
        ]
        assert harness.agent._engine_preflight_requested is False
