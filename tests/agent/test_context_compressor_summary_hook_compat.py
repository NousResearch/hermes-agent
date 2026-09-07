"""Compatibility coverage for third-party summary hook overrides."""

from agent.context_compressor import ContextCompressor


class _LegacySummaryCompressor(ContextCompressor):
    """Model an engine released before ``bypass_cooldown`` was added."""

    def __init__(self) -> None:
        super().__init__(
            model="test-model",
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            config_context_length=40_960,
        )
        self.summary_calls: list[tuple[list[dict], str | None, str]] = []

    def _generate_summary(
        self,
        turns_to_summarize: list[dict],
        focus_topic: str | None = None,
        memory_context: str = "",
    ) -> str:
        self.summary_calls.append((turns_to_summarize, focus_topic, memory_context))
        return "## Goal\nPreserve compatibility with legacy summary hooks."


class _KwargsSummaryCompressor(_LegacySummaryCompressor):
    def __init__(self) -> None:
        super().__init__()
        self.summary_kwargs: list[dict[str, object]] = []

    def _generate_summary(
        self,
        turns_to_summarize: list[dict],
        focus_topic: str | None = None,
        memory_context: str = "",
        **kwargs: object,
    ) -> str:
        self.summary_kwargs.append(kwargs)
        return "## Goal\nPreserve bypass semantics for extensible hooks."


def _messages() -> list[dict]:
    messages = [{"role": "system", "content": "system"}]
    messages.extend(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index} " + "context " * 1_000,
        }
        for index in range(14)
    )
    return messages


def test_compress_omits_bypass_cooldown_for_legacy_summary_override() -> None:
    compressor = _LegacySummaryCompressor()
    messages = _messages()

    compressed = compressor.compress(
        messages,
        current_tokens=30_000,
        memory_context="plugin memory",
        bypass_cooldown=True,
    )

    assert len(compressed) < len(messages)
    assert len(compressor.summary_calls) == 1
    assert compressor.summary_calls[0][2] == "plugin memory"


def test_compress_passes_bypass_cooldown_to_kwargs_summary_override() -> None:
    compressor = _KwargsSummaryCompressor()

    compressor.compress(_messages(), current_tokens=30_000, bypass_cooldown=True)

    assert compressor.summary_kwargs == [{"bypass_cooldown": True}]
