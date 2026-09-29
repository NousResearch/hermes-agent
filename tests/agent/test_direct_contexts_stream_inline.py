"""Delegated children and cron turns stream on the wire (#90202, #100260).

``should_use_direct_api_call`` contexts (gateway cron turns, delegate_task
children) must not spawn the interrupt worker — it wedges inside their nested
thread pools (#62151, #60203). The original fix short-circuited them onto the
NON-streaming wire, which silently dropped every liveness property streaming
provides: edge proxies killed the silent POST (z.ai HTTP 524, #90202), and the
non-stream stale watchdog could not tell a reasoning model's thinking phase
from a hung provider (#100260 — children died at exactly ``stale_timeout``).

These tests pin the replacement contract: those contexts stay on the streaming
path, issue ``stream=True`` on the calling thread (no worker), and keep the
stale detector + cross-thread interrupt abort working from the monitor thread.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers
from agent.chat_completion_helpers import (
    interruptible_streaming_api_call,
    should_use_direct_api_call,
)


# ---------------------------------------------------------------------------
# Real OpenAI-wire SSE server: records the wire ``stream`` flag per request.
# ---------------------------------------------------------------------------


class _Wire:
    def __init__(self, *, stall_after_first_chunk: bool = False):
        self.requests: list[dict] = []
        self.stall = stall_after_first_chunk
        self.hits = threading.Semaphore(0)
        wire = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    # Local-endpoint capability probes (/api/show etc.) —
                    # answer fast so agent construction never waits on the
                    # stalling stream below.
                    self.send_response(404)
                    self.end_headers()
                    return
                wire.requests.append(body)
                wire.hits.release()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                first = {
                    "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
                                 "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
                self.wfile.flush()
                if wire.stall:
                    try:
                        for _ in range(400):
                            time.sleep(0.05)
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    except Exception:
                        pass
                    return
                second = {
                    "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
                }
                fin = {
                    "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
                for c in (second, fin):
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def wire():
    w = _Wire()
    yield w
    w.close()


@pytest.fixture
def stalling_wire():
    w = _Wire(stall_after_first_chunk=True)
    yield w
    w.close()


def _make_agent(base_url: str, *, platform: str):
    return run_agent.AIAgent(
        api_key="test-key",
        base_url=base_url,
        model="m",
        provider="custom",
        platform=platform,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        max_iterations=1,
    )


_KW = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("platform", ["subagent", "cron"])
def test_direct_contexts_stream_on_the_wire_and_on_the_calling_thread(wire, platform):
    agent = _make_agent(wire.base_url, platform=platform)
    assert should_use_direct_api_call(agent) is True

    issued_on = {}
    real_create = agent._create_request_openai_client

    def spy(*a, **k):
        issued_on["tid"] = threading.get_ident()
        return real_create(*a, **k)

    agent._create_request_openai_client = spy

    response = interruptible_streaming_api_call(agent, dict(_KW))

    completions = [r for r in wire.requests if "messages" in r]
    assert completions, "no chat completion reached the wire"
    assert completions[-1].get("stream") is True, (
        f"{platform} turn went out non-streaming: stream={completions[-1].get('stream')!r}"
    )
    # No interrupt worker: the request was dispatched from the caller's thread
    # (the #62151 / #60203 deadlock class needs the request on a spawned worker).
    assert issued_on["tid"] == threading.get_ident()
    assert response.choices[0].message.content == "hello world"
    assert response.choices[0].finish_reason == "stop"


def test_interactive_platform_still_uses_the_worker_thread(wire):
    """Regression guard for the refactor: non-direct contexts keep the
    interrupt worker (interactive /stop responsiveness relies on it)."""
    agent = _make_agent(wire.base_url, platform="cli")
    assert should_use_direct_api_call(agent) is False

    issued_on = {}
    real_create = agent._create_request_openai_client

    def spy(*a, **k):
        issued_on["tid"] = threading.get_ident()
        return real_create(*a, **k)

    agent._create_request_openai_client = spy
    response = interruptible_streaming_api_call(agent, dict(_KW))

    assert issued_on["tid"] != threading.get_ident()
    assert response.choices[0].message.content == "hello world"


def test_inline_stream_stale_detector_still_fires_from_monitor_thread(
    stalling_wire, monkeypatch
):
    """The stale-stream detector moved onto a monitor thread for inline
    mode; a stream that sends one chunk then only keep-alives must still be
    killed at the stale budget instead of hanging until the socket dies."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "1.0")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    agent = _make_agent(stalling_wire.base_url, platform="subagent")

    started = time.time()
    response = interruptible_streaming_api_call(agent, dict(_KW))
    elapsed = time.time() - started

    assert elapsed < 6.0, f"inline stream was not bounded by the stale detector ({elapsed:.1f}s)"
    # A partial delta was delivered → the loop gets the length-truncated
    # partial-stream stub (same contract as the worker path).
    assert getattr(response, "id", None) == helpers.PARTIAL_STREAM_STUB_ID
    assert response.choices[0].finish_reason == helpers.FINISH_REASON_LENGTH


def test_inline_stream_cross_thread_interrupt_aborts_promptly(stalling_wire, monkeypatch):
    """``AIAgent.interrupt()`` from another thread (cron watchdog, delegation
    stall monitor) must abort the inline stream and surface InterruptedError
    — the property the direct_api_call path guaranteed via
    ``_active_request_abort``."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "60")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    agent = _make_agent(stalling_wire.base_url, platform="cron")
    box: dict = {}

    def _run():
        t0 = time.time()
        try:
            interruptible_streaming_api_call(agent, dict(_KW))
            box["outcome"] = "returned"
        except BaseException as exc:  # noqa: BLE001 — record whatever surfaces
            box["outcome"] = type(exc).__name__
        box["elapsed"] = time.time() - t0

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert stalling_wire.hits.acquire(timeout=5.0), "request never reached the wire"
    time.sleep(0.3)  # let the first chunk land
    agent.interrupt("test interrupt")
    worker.join(timeout=10.0)

    assert not worker.is_alive(), "inline stream did not unwind after interrupt"
    assert box["outcome"] == "InterruptedError"
    assert box["elapsed"] < 5.0


def test_should_use_direct_api_call_gate_is_unchanged():
    """The routing predicate itself is untouched — only what it routes to."""
    def mk(platform, api_mode="chat_completions", provider="openrouter"):
        return SimpleNamespace(platform=platform, api_mode=api_mode, provider=provider)

    assert should_use_direct_api_call(mk("cron")) is True
    assert should_use_direct_api_call(mk("subagent")) is True
    assert should_use_direct_api_call(mk("cli")) is False
    assert should_use_direct_api_call(mk("cron", api_mode="anthropic_messages")) is False
    assert should_use_direct_api_call(mk("cron", provider="moa")) is False
