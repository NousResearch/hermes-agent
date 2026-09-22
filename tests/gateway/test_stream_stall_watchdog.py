"""E2E: the stream stall watchdog delivers the partial when the upstream API dies.

When the upstream dies mid-stream no ``_DONE`` ever reaches the consumer, so the run
loop would sit on its queue forever: the user stares at a live "typing" reply that
never resolves, and no finalize is ever attempted (#25010 is the related-but-distinct
finalize bug - there the stream is still alive).

The watchdog arms on the first text delta, is disarmed by any control sentinel (a tool
run is *expected* silence), and delivers the accumulated partial with an
incomplete-response notice once the window expires.

These tests drive a real ``GatewayStreamConsumer`` against a fake adapter, following
``test_stream_final_contract.py``: the transport is faked, the watchdog is not.  Each
test picks the stall window it needs - ``FIRE_WINDOW`` for the cases that must trip,
``SAFE_WINDOW`` for the cases that must not (wide enough that the run loop's 50 ms
yield can never race the boundary handling that disarms the clock).

The death-path test runs over both progressive transports (``edit`` and native
``draft``): the watchdog must never regress into "editMessageText only".
"""

import asyncio
import contextlib

import pytest

from gateway import stream_consumer as sc
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

FIRE_WINDOW = 0.2
SAFE_WINDOW = 0.5


def _make_adapter(*, draft=False, stream_is_message=False):
    """A BasePlatformAdapter double recording every visible delivery."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    A = type("WatchdogAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    A.__abstractmethods__ = frozenset()
    adapter = A.__new__(A)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.draft_stream_is_message = stream_is_message
    adapter.draft_calls = []
    adapter.send_calls = []
    adapter.edit_calls = []

    if draft:
        def _supports(chat_id=None, chat_type=None, metadata=None):
            return True

        adapter.supports_draft_streaming = _supports

    async def _send_draft(*, chat_id, draft_id, content, metadata=None):
        adapter.draft_calls.append(content)
        return SendResult(success=True, message_id=None)

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        adapter.send_calls.append(content)
        return SendResult(success=True, message_id="msg-1")

    async def _edit(chat_id, message_id, content, **kw):
        adapter.edit_calls.append(content)
        return SendResult(success=True, message_id=message_id)

    adapter.send_draft = _send_draft
    adapter.send = _send
    adapter.edit_message = _edit
    return adapter


def _consumer(adapter, *, transport="edit"):
    cfg = StreamConsumerConfig(transport=transport, chat_type="dm",
                               edit_interval=0.01, buffer_threshold=1, cursor="")
    return GatewayStreamConsumer(adapter, "chat-1", cfg)


def _delivered(adapter):
    """Everything the user could have seen (best effort, order not guaranteed)."""
    return list(adapter.send_calls) + list(adapter.edit_calls) + list(adapter.draft_calls)


async def _drop(task):
    task.cancel()
    with contextlib.suppress(BaseException):
        await task


@pytest.fixture(autouse=True)
def _no_configured_timeout(monkeypatch):
    """The environment must not leak an operator-configured timeout into these runs."""
    monkeypatch.delenv("HERMES_STREAM_STALL_TIMEOUT", raising=False)


@pytest.mark.parametrize("transport", ["edit", "draft"])
@pytest.mark.asyncio
async def test_stall_delivers_partial(monkeypatch, transport):
    """Upstream dies mid-answer (no ``finish()``): the partial ships with the notice.

    Both progressive transports are policed, so the watchdog cannot silently regress
    into "editMessageText only": ``edit`` and native ``draft``.  Neither is cumulative -
    a draft reply surfaces as a fresh message, see ``test_cumulative_transport_untouched``
    for the transports that stay out of scope.
    """
    monkeypatch.setattr(sc, "STREAM_STALL_DEFAULT_TIMEOUT_S", FIRE_WINDOW)
    adapter = _make_adapter(draft=transport == "draft")
    consumer = _consumer(adapter, transport=transport)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Here is the first half of the answer")
    await asyncio.sleep(0.05)

    assert consumer._cumulative_transport() is False

    # Nothing else ever arrives - exactly the upstream-death shape.  The consumer has to
    # resolve by itself instead of waiting for a _DONE that cannot come.
    await asyncio.wait_for(task, timeout=FIRE_WINDOW + 2.0)

    final = [text for text in _delivered(adapter)
             if sc.STREAM_STALL_NOTICE in text]
    assert final, "the partial must reach the adapter with the notice"
    assert final[-1].startswith("Here is the first half of the answer")
    assert consumer._final_response_sent is True


@pytest.mark.asyncio
async def test_tool_run_silence_does_not_stall(monkeypatch):
    """A tool run is expected silence - it must never be read as a dead stream."""
    monkeypatch.setattr(sc, "STREAM_STALL_DEFAULT_TIMEOUT_S", SAFE_WINDOW)
    adapter = _make_adapter()
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Let me look that up")
    await asyncio.sleep(0.05)

    consumer.on_delta(None)  # tool boundary: the stream legitimately goes quiet
    await asyncio.sleep(3 * SAFE_WINDOW)

    assert not task.done(), "the watchdog must not fire across a tool run"
    assert not any(sc.STREAM_STALL_NOTICE in text for text in _delivered(adapter))

    consumer.on_delta("Found it - here is the answer")
    await asyncio.sleep(0.05)
    consumer.finish("Found it - here is the answer")
    await asyncio.wait_for(task, timeout=2.0)

    assert not any(sc.STREAM_STALL_NOTICE in text for text in _delivered(adapter))
    assert consumer._final_response_sent is True


@pytest.mark.asyncio
async def test_cumulative_transport_untouched(monkeypatch):
    """Stream-is-the-message drafts keep their own liveness semantics (no watchdog)."""
    monkeypatch.setattr(sc, "STREAM_STALL_DEFAULT_TIMEOUT_S", SAFE_WINDOW)
    adapter = _make_adapter(draft=True, stream_is_message=True)
    consumer = _consumer(adapter, transport="draft")
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("cumulative stream body")
    await asyncio.sleep(0.05)

    assert consumer._cumulative_transport() is True
    await asyncio.sleep(3 * SAFE_WINDOW)

    assert not task.done(), "cumulative transports are out of scope"
    assert not any(sc.STREAM_STALL_NOTICE in text for text in _delivered(adapter))
    await _drop(task)


@pytest.mark.asyncio
async def test_configured_timeout_wins_over_the_default(monkeypatch):
    """``agent.stream_stall_timeout`` (bridged to the env var) drives the window."""
    monkeypatch.setattr(sc, "STREAM_STALL_DEFAULT_TIMEOUT_S", 30.0)  # default never fires
    monkeypatch.setenv("HERMES_STREAM_STALL_TIMEOUT", "0.2")
    adapter = _make_adapter()
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("partial body")
    await asyncio.sleep(0.05)

    await asyncio.wait_for(task, timeout=3.0)

    assert consumer._stream_stall_timeout() == 0.2
    assert sc.STREAM_STALL_NOTICE in _delivered(adapter)[-1]


@pytest.mark.asyncio
async def test_zero_disables_the_watchdog(monkeypatch):
    """``stream_stall_timeout: 0`` disables the watchdog - 0 is not "fire immediately"."""
    monkeypatch.setattr(sc, "STREAM_STALL_DEFAULT_TIMEOUT_S", SAFE_WINDOW)
    monkeypatch.setenv("HERMES_STREAM_STALL_TIMEOUT", "0")
    adapter = _make_adapter()
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("partial body")
    await asyncio.sleep(0.05)

    await asyncio.sleep(3 * SAFE_WINDOW)  # the window has long elapsed by now

    assert not task.done(), "a disabled watchdog must never resolve the stream"
    assert not any(sc.STREAM_STALL_NOTICE in text for text in _delivered(adapter))

    consumer.finish("partial body")
    await asyncio.wait_for(task, timeout=2.0)
    assert consumer._final_response_sent is True
