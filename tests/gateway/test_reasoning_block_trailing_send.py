"""The reasoning block must survive a streamed turn.

``display.show_reasoning`` renders the last reasoning block above the reply. When streaming already
delivered the body, the composed response is never sent — so a block prepended onto it was dropped
silently and the user saw only the streamed body. The block is now composed on its own and sent as
a trailing message, mirroring the runtime footer.
"""

import importlib

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner

# Imported at module scope on purpose: ``gateway.run`` performs install-manifest I/O during import,
# which the harness's home-I/O guard refuses once a test is running. The sibling gateway tests
# import it this way for the same reason.
from gateway.run import GatewayRunner

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


def _runner(adapter):
    """A real ``GatewayRunner`` over a capturing adapter — the seam under test is its method."""
    runner = _make_runner(adapter)
    assert isinstance(runner, GatewayRunner)
    return runner


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585"
    )


def _event(source):
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")


class _Entry:
    session_id = "s1"


async def _deliver(runner, event, source, agent_result, response, reasoning_block, footer_line=""):
    """The completion seam that decides whether the caller sends the body again."""
    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None, agent_result, [], response,
        reasoning_block, footer_line, False,
    )


@pytest.mark.asyncio
async def test_streamed_turn_sends_the_block_and_not_the_body():
    """``already_sent``: the stream owns the body, the held-back block still reaches the user."""
    adapter = ProgressCaptureAdapter()
    runner = _runner(adapter)
    source = _source()
    event = _event(source)

    delivered = await _deliver(
        runner, event, source, {"already_sent": True, "media_already_delivered": True},
        "the answer", "THINKING",
    )

    assert delivered is None, "the streamed body must not be sent a second time"
    assert [c["content"] for c in adapter.sent] == ["THINKING"]


@pytest.mark.asyncio
async def test_trailing_block_precedes_the_trailing_footer():
    """The block reads above the reply, so it must be sent before the footer."""
    adapter = ProgressCaptureAdapter()
    runner = _runner(adapter)
    source = _source()
    event = _event(source)

    await _deliver(
        runner, event, source, {"already_sent": True, "media_already_delivered": True},
        "the answer", "THINKING", "FOOTER",
    )

    assert [c["content"] for c in adapter.sent] == ["THINKING", "FOOTER"]


@pytest.mark.asyncio
async def test_non_streamed_turn_leaves_the_block_to_the_caller():
    """Without ``already_sent`` the caller still owns the body and prepends there — the seam must
    not also send the block, or every normal reply would carry it twice."""
    adapter = ProgressCaptureAdapter()
    runner = _runner(adapter)
    source = _source()
    event = _event(source)

    delivered = await _deliver(
        runner, event, source, {"media_already_delivered": True}, "the answer", "THINKING",
    )

    assert delivered == "the answer"
    assert adapter.sent == []


def test_reasoning_block_excludes_the_response_body(monkeypatch):
    """The block is composed independently: rendering it with the body inlined would either leak
    the answer into the trailing message or duplicate it in the prepended one."""
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_display_bool", lambda *a, **k: True)
    monkeypatch.setattr(
        importlib.import_module("gateway.display_config"), "resolve_display_setting", lambda *a, **k: "code"
    )

    runner = _runner(ProgressCaptureAdapter())
    block = runner._hmwa_reasoning_block(
        {"last_reasoning": "THINKING"}, "the answer", _source(), False,
    )

    assert "THINKING" in block
    assert "the answer" not in block
