"""A2A replies are atomic: the caller must receive the turn's final reply byte-for-byte.

Regression for #95753.

``A2AAdapter.send()`` resolves the caller's task only on the final (``notify``) send and drops
every other send while still reporting success.  With gateway streaming on, the stream consumer
treated those dropped previews as visible text, so the caller got the reply with its head cut
off, only the first chunk of a long reply, a tool-turn preamble instead of the answer, or ``""``
when the suppressed final was stashed on a restored copy of the event.

The turn below runs the real A2A HTTP server and client extractor, the real turn wiring
(``TurnRunner._setup_stream_consumer``, ``GatewayStreamConsumer``,
``_run_agent_mark_streamed_delivery``) and the base adapter's real final send.  The agent is a
scripted (synthetic) sequence of the callbacks it fires; no model is called.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import urllib.request
from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, StreamingConfig
from gateway.display_config import resolve_display_setting
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter

_LONG = "\n".join(
    f"Line {i:04d}: the quick brown fox jumps over the lazy dog; checksum {i * 7919 % 10007}."
    for i in range(130))  # ~9 KB: more than one 4096-char chat chunk
_PREAMBLE = "\n\nRunning the three checks before I answer."

# case -> (agent callbacks, final reply).  ("delta", text) streams text, ("tick",) lets the stream
# consumer run one pass (a preview edit), ("interim", text) is a completed assistant message
# before a tool call (or the final itself, as the codex app-server bridge delivers it).
CASES = {
    "short_single_delta": ([("delta", "PONG peer-b"), ("tick",)], "PONG peer-b"),
    "hyphen_head": ([("delta", "REF-"), ("tick",), ("delta", "0001-ALPHA model-x provider-y")],
                    "REF-0001-ALPHA model-x provider-y"),
    "midword_head": ([("delta", "ACK Q"), ("tick",), ("delta", "ZX-42-ABC peer-b/host-b")],
                     "ACK QZX-42-ABC peer-b/host-b"),
    "long_multichunk": ([("delta", _LONG[i:i + 61]) for i in range(0, len(_LONG), 61)] + [("tick",)], _LONG),
    "final_as_commentary": ([("interim", "READY commentary-lane final")], "READY commentary-lane final"),
    # The preamble's deltas and its tool boundary land in the same consumer pass.
    "tool_turn_preamble_then_final": ([("delta", _PREAMBLE), ("interim", _PREAMBLE.strip()),
                                       ("delta", "RESULT: all three checks passed."), ("tick",)],
                                      "RESULT: all three checks passed."),
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _consumer_tick(consumer) -> None:
    """Wait for one real consumer pass over what is queued now, not a fixed sleep."""
    reached = asyncio.Event()
    original = consumer._should_edit

    def observed(tick):
        reached.set()
        return original(tick)

    consumer._should_edit = observed
    try:
        await asyncio.wait_for(reached.wait(), 5)
        await asyncio.sleep(0)
    finally:
        consumer._should_edit = original


def _handler(runner, case, *, restored):
    steps, final = CASES[case]

    async def handle(event):
        ctx = TurnContext(source=event.source, user_config={}, resolve_display_setting=resolve_display_setting,
                          interim_assistant_messages_enabled=True, event_message_id=event.message_id,
                          session_key="agent:main:a2a:dm:ctx", _run_still_current=lambda: True)
        consumer, delta_cb, interim_cb, _ = TurnRunner(runner, ctx)._setup_stream_consumer("a2a")
        task = asyncio.create_task(consumer.run()) if consumer else None
        streamed = ""
        for kind, *text in steps:
            if kind == "delta" and delta_cb is not None:  # the agent streams only into a sink
                delta_cb(text[0])
                streamed += text[0]
            elif kind == "interim":
                interim_cb(text[0], already_streamed=text[0] in streamed)
            if kind != "delta" and consumer is not None:
                await _consumer_tick(consumer)
        if consumer is not None:
            consumer.finish(final)
            await asyncio.wait_for(task, 10)
        response = {"final_response": final}
        await runner._run_agent_mark_streamed_delivery(response, ctx)
        if response.get("already_sent"):
            # run_turn_persistence: the body is not sent twice; the delivered text is stashed on
            # the event the turn executed on (the session authority restores its own copy).
            (replace(event) if restored else event)._streamed_final_response = final
            return None
        return final

    return handle


def _send_message(url: str, text: str) -> str:
    """One blocking SendMessage, as ``a2a_call`` sends it; returns the caller-visible reply."""
    body = {"jsonrpc": "2.0", "id": "req-1", "method": "SendMessage",
            "params": {"message": protocol.text_message(protocol.ROLE_USER, text,
                                                        context_id=protocol.new_context_id())}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode("utf-8"))
    assert "error" not in resp, resp
    return tools._reply_text_from_result(resp.get("result", {}))


def _receiver(tmp_path):
    """A real A2A adapter owned by a real gateway runner with streaming enabled."""
    from gateway.run import GatewayRunner
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions",
                                         streaming=StreamingConfig(enabled=True)))
    runner.adapters = {Platform("a2a"): adapter}
    adapter.gateway_runner = runner
    return adapter, runner


def _exchange(monkeypatch, tmp_path, case, restored) -> str:
    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter, runner = _receiver(tmp_path)
    adapter.set_message_handler(_handler(runner, case, restored=restored))
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(adapter.connect())
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(10)
    try:
        return _send_message(f"http://127.0.0.1:{port}", "probe")
    finally:
        asyncio.run_coroutine_threadsafe(adapter.disconnect(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


@pytest.mark.parametrize("restored", [False, True], ids=["in_process_event", "restored_event_copy"])
@pytest.mark.parametrize("case", list(CASES))
def test_caller_receives_final_reply_byte_for_byte(monkeypatch, tmp_path, case, restored):
    final = CASES[case][1]
    received = _exchange(monkeypatch, tmp_path, case, restored)
    assert received.encode("utf-8") == final.encode("utf-8"), (
        f"caller got {len(received)} of {len(final)} chars: {received[:40]!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["hyphen_head", "midword_head", "long_multichunk"])
async def test_proxy_stream_resolves_task_with_one_complete_final(monkeypatch, tmp_path, case):
    """The proxy path keeps a stream consumer and feeds it every delta, even for A2A."""
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    adapter, runner = _receiver(tmp_path)
    source = adapter.build_source(chat_id="ctx", chat_type="dm", user_id="peer-b")
    consumer = runner._proxy_stream_consumer(source, "task", None, lambda: True)
    sends = []
    original = adapter.send

    async def send(*args, **kwargs):
        sends.append(kwargs.get("metadata"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(adapter, "send", send)
    future = adapter._add_pending("task", "ctx")
    task = asyncio.create_task(consumer.run())
    steps, final = CASES[case]
    try:
        for kind, *text in steps:
            if kind == "delta":
                consumer.on_delta(text[0])
            else:
                await _consumer_tick(consumer)
        consumer.finish(final)
        await asyncio.wait_for(task, 10)
        assert future.result() == (protocol.STATE_COMPLETED, final)
        assert len(sends) == 1 and sends[0]["notify"] is True
    finally:
        adapter._pop_pending("task")
