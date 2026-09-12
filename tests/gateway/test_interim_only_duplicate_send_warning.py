"""#107619 — interim-only GatewayStreamConsumer must not log a false-positive
``possible duplicate send`` warning.

streaming.enabled=false + interim assistant messages still constructs a
consumer. After commentary, segment leftovers are empty and finish() does
not adopt the payload. Flags stay false. The normal final-send must proceed
(already_sent unset) without the WeCom ack-timeout diagnostic.

CONTROL: leftover stream text + flags still false MUST still warn.
"""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


WARNING_NEEDLE = "possible duplicate send"
LOGGER_NAME = "gateway.run"
FINAL_TEXT = "The user still needs this 508-character-class reply delivered."


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _response(**overrides):
    payload = {
        "final_response": FINAL_TEXT,
        "failed": False,
        "response_transformed": False,
        "response_previewed": False,
    }
    payload.update(overrides)
    return payload


def _consumer(*, accumulated="", message_id=None, last_sent_text=""):
    """Never-streamed interim consumer after commentary reset, unless leftovers set."""
    return SimpleNamespace(
        final_response_sent=False,
        final_content_delivered=False,
        _accumulated=accumulated,
        _message_id=message_id,
        _last_sent_text=last_sent_text,
        message_id=message_id,
        adapter=None,
        _turn_split_delivery=False,
    )


def _ctx(consumer):
    return TurnContext(
        source=SimpleNamespace(chat_id="c1"),
        session_key="agent:main:telegram:dm:c1",
        stream_consumer_holder=[consumer],
    )


async def _mark(response, consumer):
    await GatewayRunner._run_agent_mark_streamed_delivery(
        _runner(), response, _ctx(consumer)
    )


@pytest.mark.asyncio
async def test_detection_interim_only_never_streamed_does_not_warn(caplog):
    """Detection: no on_delta leftovers, flags all false, non-empty final.

    Must NOT log the WeCom duplicate-send warning; already_sent stays unset
    so the normal final-send still happens.
    """
    response = _response()
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        await _mark(response, _consumer())

    assert WARNING_NEEDLE not in caplog.text
    assert "already_sent" not in response


@pytest.mark.asyncio
async def test_control_streamed_leftover_flags_false_still_warns(caplog):
    """CONTROL: on_delta leftover + flags false → WeCom warning MUST fire.

    already_sent must stay unset (do not silence a real stream attempt).
    """
    response = _response()
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        await _mark(response, _consumer(accumulated="partial streamed leftover"))

    assert WARNING_NEEDLE in caplog.text
    assert "already_sent" not in response


@pytest.mark.asyncio
async def test_fail_open_no_consumer_no_warning(caplog):
    """Fail-open: _sc is None → no warning, already_sent unset."""
    response = _response()
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        await _mark(response, None)

    assert WARNING_NEEDLE not in caplog.text
    assert "already_sent" not in response
