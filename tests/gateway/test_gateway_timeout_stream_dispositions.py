import logging

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SessionSource
from gateway.run import _log_gateway_disposition


def _make_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c1",
        user_id="u1",
        user_name="tester",
        chat_type="dm",
    )


def test_logs_timeout_disposition(caplog):
    source = _make_source()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_gateway_disposition(
            "timeout",
            source=source,
            session_key="agent:main:telegram:dm:c1",
            reason="agent_timeout:300s",
        )
    assert any("gateway_disposition" in r.message and "disposition=timeout" in r.message for r in caplog.records)


def test_logs_streamed_already_sent_disposition(caplog):
    source = _make_source()
    event = MessageEvent(source=source, text="hello", message_id="m1")
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_gateway_disposition(
            "streamed_already_sent",
            source=source,
            session_key="agent:main:telegram:dm:c1",
            event=event,
            reason="stream_consumer_delivered_response",
        )
    assert any("gateway_disposition" in r.message and "disposition=streamed_already_sent" in r.message for r in caplog.records)
