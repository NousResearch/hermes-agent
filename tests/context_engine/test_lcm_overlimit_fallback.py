import pytest

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine, LCMFailOpenRecoveryError
from plugins.context_engine.lcm.tokens import count_messages_tokens


class RecordingFallbackCompressor:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        self.calls.append(
            {
                "messages": messages,
                "current_tokens": current_tokens,
                "focus_topic": focus_topic,
                "force": force,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.result


def _engine(tmp_path, *, context_length):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        context_threshold=0.01,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.update_model("unit-test-model", context_length, provider="unit-test")
    engine.on_session_start(
        "session-1",
        hermes_home=str(tmp_path),
        model="unit-test-model",
        provider="unit-test",
        context_length=context_length,
        platform="pytest",
    )
    return engine


def _over_limit_messages():
    blob = "overflow payload " * 400
    return [
        {"role": "system", "content": "You are testing LCM."},
        {"role": "user", "content": blob},
        {"role": "assistant", "content": blob},
        {"role": "user", "content": blob},
    ]


def test_over_limit_failopen_attempts_builtin_compressor_fallback(tmp_path, monkeypatch):
    messages = _over_limit_messages()
    raw_tokens = count_messages_tokens(messages)
    engine = _engine(tmp_path, context_length=max(1, raw_tokens - 1))
    fallback_result = [
        messages[0],
        {"role": "assistant", "content": "fallback compressed context"},
        messages[-1],
    ]
    fallback = RecordingFallbackCompressor(result=fallback_result)

    monkeypatch.setattr(engine, "_build_fail_open_fallback_compressor", lambda: fallback)

    def _raise_db(*_args, **_kwargs):
        raise RuntimeError("primary LCM store unavailable")

    monkeypatch.setattr(engine, "_ingest_messages", _raise_db)

    returned = engine.compress(messages, current_tokens=raw_tokens, focus_topic="overflow focus")

    assert returned == fallback_result
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["messages"] is messages
    assert fallback.calls[0]["current_tokens"] == raw_tokens
    assert fallback.calls[0]["focus_topic"] == "overflow focus"
    assert fallback.calls[0]["force"] is True

    status = engine.get_status()
    assert status["degraded"] is True
    assert status["last_fail_open_over_context_limit"] is True
    assert status["last_fail_open_fallback_attempted"] is True
    assert status["last_fail_open_fallback_status"] == "succeeded"
    assert status["fail_open_total_count"] == 1
    assert status["fail_open_consecutive_count"] == 1


def test_over_limit_fallback_failure_raises_loud_recoverable_error(tmp_path, monkeypatch):
    messages = _over_limit_messages()
    raw_tokens = count_messages_tokens(messages)
    engine = _engine(tmp_path, context_length=max(1, raw_tokens - 1))
    fallback = RecordingFallbackCompressor(exc=RuntimeError("fallback compressor failed"))

    monkeypatch.setattr(engine, "_build_fail_open_fallback_compressor", lambda: fallback)

    def _raise_db(*_args, **_kwargs):
        raise RuntimeError("primary LCM store unavailable")

    monkeypatch.setattr(engine, "_ingest_messages", _raise_db)

    with pytest.raises(LCMFailOpenRecoveryError) as exc_info:
        engine.compress(messages, current_tokens=raw_tokens)

    assert "recoverable" in str(exc_info.value).lower()
    assert "fallback compressor failed" in str(exc_info.value)
    assert len(fallback.calls) == 1

    status = engine.get_status()
    assert status["degraded"] is True
    assert status["last_fail_open_over_context_limit"] is True
    assert status["last_fail_open_fallback_attempted"] is True
    assert status["last_fail_open_fallback_status"] == "failed"
    assert status["fail_open_total_count"] == 1
    assert status["fail_open_consecutive_count"] == 1
