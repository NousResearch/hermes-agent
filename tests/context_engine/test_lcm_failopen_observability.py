import logging

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm.tokens import count_messages_tokens


LOGGER_NAME = "plugins.context_engine.lcm.engine"


def _engine(tmp_path, *, context_length=100_000):
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


def _messages():
    return [
        {"role": "system", "content": "You are testing LCM."},
        {"role": "user", "content": "Important historical fact A."},
        {"role": "assistant", "content": "Important historical answer B."},
        {"role": "user", "content": "Fresh user request."},
    ]


def test_summary_exception_below_limit_fails_open_unchanged_and_marks_degraded(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path, context_length=100_000)
    messages = _messages()

    def _raise_summary(*_args, **_kwargs):
        raise RuntimeError("summary exploded")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _raise_summary)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        returned = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert returned is messages
    assert returned == messages

    status = engine.get_status()
    assert status["degraded"] is True
    assert status["context_engine_degraded"] is True
    assert status["degraded_reason"] == "fail_open"
    assert status["last_fail_open_error_type"] == "RuntimeError"
    assert "summary exploded" in status["last_fail_open_error"]
    assert status["fail_open_total_count"] == 1
    assert status["fail_open_consecutive_count"] == 1
    assert status["last_fail_open_fallback_attempted"] is False
    assert status["last_fail_open_fallback_status"] == "not_needed_below_limit"
    assert "event=lcm_fail_open_degraded" in caplog.text
    assert "fallback_attempted=False" in caplog.text


def test_three_consecutive_db_failopens_are_visible_in_status_and_logs(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path, context_length=100_000)
    messages = _messages()

    def _raise_db(*_args, **_kwargs):
        raise OSError("sqlite write failed")

    monkeypatch.setattr(engine, "_ingest_messages", _raise_db)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        for _ in range(3):
            returned = engine.compress(messages, current_tokens=count_messages_tokens(messages))
            assert returned is messages

    status = engine.get_status()
    assert status["degraded"] is True
    assert status["degraded_reason"] == "fail_open"
    assert status["last_fail_open_error_type"] == "OSError"
    assert status["fail_open_total_count"] == 3
    assert status["fail_open_consecutive_count"] == 3
    assert status["fail_open_window_seconds"] > 0
    assert status["fail_open_window_count"] >= 3
    assert status["fail_open_rate_per_minute"] > 0
    assert caplog.text.count("event=lcm_fail_open_degraded") == 3
    assert "consecutive=3" in caplog.text
