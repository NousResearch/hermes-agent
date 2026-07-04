from types import SimpleNamespace

from agent.conversation_compression import (
    CONTEXT_LIMIT_STATUS_MARKER,
    format_context_limit_status,
    maybe_emit_context_limit_status,
)


def _agent(tokens=0, threshold=100_000, context_length=200_000):
    events = []

    def emit_status(message: str) -> None:
        events.append(message)

    agent = SimpleNamespace(
        compression_enabled=True,
        session_id="session-1",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=tokens,
            threshold_tokens=threshold,
            context_length=context_length,
        ),
        _emit_status=emit_status,
    )
    return agent, events


def test_context_limit_status_format_includes_marker_and_pressure():
    text = format_context_limit_status(90_000, 100_000, 200_000)

    assert CONTEXT_LIMIT_STATUS_MARKER in text
    assert "~90,000 tokens" in text
    assert "90% of the ~100,000 compression threshold" in text
    assert "45% of model context" in text
    assert "About 10,000 tokens before compaction" in text


def test_context_limit_status_emits_once_per_pressure_band():
    agent, events = _agent()

    maybe_emit_context_limit_status(agent, 89_999)
    assert events == []

    maybe_emit_context_limit_status(agent, 90_000)
    assert len(events) == 1
    assert CONTEXT_LIMIT_STATUS_MARKER in events[0]

    maybe_emit_context_limit_status(agent, 91_000)
    assert len(events) == 1

    maybe_emit_context_limit_status(agent, 95_000)
    assert len(events) == 2


def test_context_limit_status_skips_when_compaction_is_due():
    agent, events = _agent()

    maybe_emit_context_limit_status(agent, 100_000)

    assert events == []
