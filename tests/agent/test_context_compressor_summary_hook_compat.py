"""Compatibility coverage for third-party summary hook overrides."""

import pytest

from agent.context_compressor import ContextCompressor


class _LegacySummaryCompressor(ContextCompressor):
    """Model an engine released before ``bypass_cooldown`` was added."""

    def __init__(self) -> None:
        super().__init__(
            model="test-model", protect_first_n=2, protect_last_n=2,
            quiet_mode=True, config_context_length=40_960,
        )
        self.summary_calls = []

    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context=""):
        self.summary_calls.append((focus_topic, memory_context, {}))
        return "## Goal\nPreserve compatibility with legacy summary hooks."


class _KwargsSummaryCompressor(_LegacySummaryCompressor):
    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", **kwargs):
        self.summary_calls.append((focus_topic, memory_context, kwargs))
        return "## Goal\nPreserve bypass semantics for extensible hooks."


class _ExplicitSummaryCompressor(_KwargsSummaryCompressor):
    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", *, bypass_cooldown=True):
        return super()._generate_summary(
            turns_to_summarize, focus_topic, memory_context, bypass_cooldown=bypass_cooldown,
        )


class _PositionalSummaryCompressor(_LegacySummaryCompressor):
    def _generate_summary(self, turns_to_summarize, bypass_cooldown=False, /, focus_topic=None, memory_context=""):
        return super()._generate_summary(turns_to_summarize, focus_topic, memory_context)


def _messages():
    return [{"role": "system", "content": "system"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "context " * 1_000}
        for i in range(14)
    ]


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("engine", [
    _LegacySummaryCompressor, _KwargsSummaryCompressor,
    _ExplicitSummaryCompressor, _PositionalSummaryCompressor,
])
def test_compress_calls_supported_summary_signature_once(engine, bypass):
    compressor = engine()
    messages = _messages()
    compressed = compressor.compress(
        messages, current_tokens=30_000, focus_topic="compatibility",
        memory_context="plugin memory", bypass_cooldown=bypass,
    )
    expected = {"bypass_cooldown": bypass} if isinstance(compressor, _KwargsSummaryCompressor) else {}
    assert len(compressed) < len(messages)
    assert compressor.summary_calls == [("compatibility", "plugin memory", expected)]


@pytest.mark.parametrize("bypass", [False, True])
def test_summary_hook_type_error_is_not_retried(bypass):
    class FailingCompressor(_KwargsSummaryCompressor):
        def _generate_summary(self, *args, **kwargs):
            super()._generate_summary(*args, **kwargs)
            raise TypeError("inside stateful hook")

    compressor = FailingCompressor()
    with pytest.raises(TypeError, match="inside stateful hook"):
        compressor.compress(_messages(), current_tokens=30_000, bypass_cooldown=bypass)
    assert len(compressor.summary_calls) == 1
