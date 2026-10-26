"""The proactive tool-result prune under ``compression.checkpoint_required``.

The prune rewrites the durable transcript between compactions, so no pre-compress
checkpoint precedes it: the armed gate withholds it (suppression, not refusal) and
reports only a prune that would actually have run.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import conversation_compression_checkpoint as gate
from agent.conversation_compression_checkpoint import _prune_would_have_run
from agent.turn_preflight import _proactive_tool_result_prune


class _CountingPruneCompressor:
    """Built-in-shaped double: counts instead of asserting, because the call site swallows
    exceptions. Publishes ``would_proactively_prune`` with a configured trigger by default."""

    def __init__(self, proactive_prune_tokens: int = 48_000):
        self.calls = 0
        self.proactive_prune_tokens = proactive_prune_tokens

    def would_proactively_prune(self, current_tokens=None):
        return self.proactive_prune_tokens > 0 and (
            current_tokens is None or current_tokens >= self.proactive_prune_tokens
        )

    def prune_tool_results_only(self, messages, current_tokens=None):
        self.calls += 1
        return list(messages) + [{"role": "system", "content": "pruned"}], 3


def _engine_stub():
    """Built against the currently imported ABC (sibling suites purge ``agent.*``)."""
    from agent.context_engine import ContextEngine

    class _CompressOnlyEngine(ContextEngine):
        @property
        def name(self) -> str:
            return "compress-only"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
            return messages

    return _CompressOnlyEngine


def _agent(checkpoint_required: bool):
    return SimpleNamespace(compression_checkpoint_required=checkpoint_required)


def _prune_warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "tool-result prune" in r.getMessage()]


@pytest.fixture(autouse=True)
def _fresh_once_log(monkeypatch):
    monkeypatch.setattr(gate, "_checkpoint_gate_warned", set())


def test_proactive_prune_is_suppressed_when_checkpoint_required():
    """The gate reaches the prune and the transcript survives intact."""
    compressor = _CountingPruneCompressor()
    messages = [{"role": "user", "content": "scan"}]
    agent = _agent(True)

    assert _proactive_tool_result_prune(agent, compressor, messages, 400_000) is messages
    assert compressor.calls == 0
    assert agent._checkpoint_gate_suppression_count == 1


def test_proactive_prune_still_runs_when_gate_is_off():
    """Sabotage control: gate off, the harness really reaches the prune."""
    compressor = _CountingPruneCompressor()
    messages = [{"role": "user", "content": "scan"}]

    with patch("agent.compaction_events.publish_prune"):
        result = _proactive_tool_result_prune(_agent(False), compressor, messages, 400_000)

    assert compressor.calls == 1
    assert result is not messages
    assert result[-1]["content"] == "pruned"


def test_engine_override_of_prune_is_suppressed_too():
    """One call site covers the built-in and every engine overriding the hook."""

    class _PruningEngine(_engine_stub()):
        calls = 0

        def prune_tool_results_only(self, messages, current_tokens=None):
            type(self).calls += 1
            return list(messages), 3

    engine = _PruningEngine()
    messages = [{"role": "user", "content": "scan"}]
    agent = _agent(True)

    assert _proactive_tool_result_prune(agent, engine, messages, 400_000) is messages
    assert _PruningEngine.calls == 0
    # An overriding engine owns its trigger policy: the host would have dispatched it.
    assert agent._checkpoint_gate_suppression_count == 1

    with patch("agent.compaction_events.publish_prune"):
        _proactive_tool_result_prune(_agent(False), engine, messages, 400_000)
    assert _PruningEngine.calls == 1


def test_prune_suppression_logs_once_and_names_the_availability_consequence(caplog):
    compressor = _CountingPruneCompressor()
    agent = _agent(True)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        for _ in range(2):
            _proactive_tool_result_prune(agent, compressor, [], 400_000)

    warnings = _prune_warnings(caplog)
    assert len(warnings) == 1
    assert "checkpoint_required" in warnings[0].getMessage()
    assert "halt" in warnings[0].getMessage()
    # Suppression is re-evaluated every call; only the log is deduplicated.
    assert agent._checkpoint_gate_suppression_count == 2


def test_disabled_prune_is_not_reported_as_a_suppressed_authority(caplog):
    """The shipping default (``proactive_prune_tokens: 0``) withholds nothing."""
    compressor = _CountingPruneCompressor(proactive_prune_tokens=0)
    messages = [{"role": "user", "content": "scan"}]
    agent = _agent(True)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        assert _proactive_tool_result_prune(agent, compressor, messages, 400_000) is messages

    assert compressor.calls == 0
    assert getattr(agent, "_checkpoint_gate_suppression_count", 0) == 0
    assert _prune_warnings(caplog) == []


def test_prune_below_its_trigger_is_not_reported_as_a_suppressed_authority(caplog):
    compressor = _CountingPruneCompressor(proactive_prune_tokens=400_000)
    agent = _agent(True)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _proactive_tool_result_prune(agent, compressor, [], 399_999)
    assert getattr(agent, "_checkpoint_gate_suppression_count", 0) == 0
    assert _prune_warnings(caplog) == []

    # One token more and the same compressor IS a withheld authority.
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _proactive_tool_result_prune(agent, compressor, [], 400_000)
    assert agent._checkpoint_gate_suppression_count == 1
    assert len(_prune_warnings(caplog)) == 1


def test_would_proactively_prune_is_the_compressors_own_precondition():
    """Answered by the real ``ContextCompressor``, so the report cannot drift from its trigger."""
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        default = ContextCompressor(model="test", quiet_mode=True)
        configured = ContextCompressor(model="test", quiet_mode=True, proactive_prune_tokens=48_000)

    assert default.proactive_prune_tokens == 0
    assert default.would_proactively_prune(400_000) is False
    assert _prune_would_have_run(default, 400_000) is False
    assert configured.would_proactively_prune(47_999) is False
    assert configured.would_proactively_prune(48_000) is True
    assert _prune_would_have_run(configured, 47_999) is False
    assert _prune_would_have_run(configured, 48_000) is True
    # An unknown token count cannot rule the prune out; the prune proceeds on None.
    assert configured.would_proactively_prune(None) is True


def test_prune_hookless_and_broken_predicate_shapes_are_handled():
    """No hook = no authority; a raising predicate reads as "would have run"."""
    assert _prune_would_have_run(SimpleNamespace(), 400_000) is False

    class _BrokenPredicate:
        def would_proactively_prune(self, current_tokens=None):
            raise RuntimeError("engine blew up")

        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages, 0

    assert _prune_would_have_run(_BrokenPredicate(), 400_000) is True


def test_engine_that_never_overrode_the_prune_hook_is_not_a_suppressed_authority():
    """``__func__`` identity tells occupancy of the hook from inheriting the ABC no-op."""
    compress_only = _engine_stub()

    class _InheritsTheDefault(compress_only):
        pass

    class _OccupiesTheHook(compress_only):
        def prune_tool_results_only(self, messages, current_tokens=None):
            return list(messages), 1

    assert _prune_would_have_run(_InheritsTheDefault(), 400_000) is False
    assert _prune_would_have_run(_OccupiesTheHook(), 400_000) is True
