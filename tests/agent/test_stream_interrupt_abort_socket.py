"""An interrupt abort must reach the in-flight stream's socket (#98974).

Report #98974: ``/stop`` / ``/reset`` logged ``OpenAI client aborted
(stream_interrupt_abort, ..., tcp_force_closed=0, deferred_close=stranger_thread)
— no sockets found`` and the self-hosted serve kept generating for minutes.
Two shapes miss the socket: (a) the pool sweep skips the connection checked
out for the in-flight body read — the stale kill already shuts down the
attempt's own socket, the interrupt path did not; (b) the abort fires during
``create()``'s connect/TLS window, before any socket exists, so nothing stops
the request once headers arrive. Both stay shutdown-only (never a cross-thread
``close()``, #30858). Shape (b) is driven through the production entry
(``interruptible_streaming_api_call`` -> monitor ``_abort_for_interrupt`` ->
``on_stream_created`` wiring) on both streaming wires.
"""
import socket as _socket
import time
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers


def _agent():
    return run_agent.AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


def _call(agent):
    call = helpers._StreamingCall(agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, None)
    call._stream_stale_timeout = 5.0
    return call


def _response_over(reader):
    """Live httpx 0.28 wrapper shape down to the socket; ``close`` must never run."""
    def _no_close():
        raise AssertionError("stranger thread must never close the response")
    h11_conn = SimpleNamespace(_network_stream=SimpleNamespace(_sock=reader),
                               _stream=None, _connection=None, _httpcore_stream=None)
    pool_stream = SimpleNamespace(_stream=SimpleNamespace(_connection=h11_conn), _connection=None)
    return SimpleNamespace(close=_no_close,
                           stream=SimpleNamespace(_stream=SimpleNamespace(_httpcore_stream=pool_stream)))


def _assert_shut_down(reader, writer):
    writer.settimeout(5)
    assert writer.recv(1) == b"", "the in-flight stream's socket was not shut down"


def test_interrupt_abort_shuts_down_the_attempts_own_socket():
    reader, writer = _socket.socketpair()
    try:
        call = _call(_agent())
        call._attempt_stream_response = _response_over(reader)
        call.worker = None
        call._monitor_interrupted = {"yes": False}  # set by the poll loop in production
        call._abort_for_interrupt(stale_elapsed=1.0)
        assert call._request_cancelled["value"] is True
        _assert_shut_down(reader, writer)
    finally:
        reader.close()
        writer.close()


class _LateStream:
    """A ``create()`` whose headers arrive only AFTER ``/stop`` was aborted during the connect
    window: the socket did not exist when the monitor swept, so only the ``on_stream_created``
    wiring can shut it down. Iterating raises like a reader on a shut-down socket."""

    def __init__(self, call, reader):
        self.response = _response_over(reader)
        call.agent._interrupt_requested = True  # /stop lands while create() is connecting
        deadline = time.time() + 5
        while not call._request_cancelled["value"] and time.time() < deadline:
            time.sleep(0.02)
        assert call._request_cancelled["value"], "monitor never ran _abort_for_interrupt"

    def __iter__(self):
        import httpx
        raise httpx.ReadError("socket shut down")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_interrupted_during_connect(monkeypatch, agent, reader, api_mode):
    agent.api_mode = api_mode
    if api_mode == "anthropic_messages":
        holder = {}
        _orig_wire = helpers._StreamingCall._call_wire

        def _wire(self, stream_attempt_id):
            holder["call"] = self
            return _orig_wire(self, stream_attempt_id)

        def _fake_anthropic_client(**_kw):
            return SimpleNamespace(messages=SimpleNamespace(stream=lambda **_k: _LateStream(holder["call"], reader)),
                                   close=lambda: None)
        monkeypatch.setattr(helpers._StreamingCall, "_call_wire", _wire)
        monkeypatch.setattr(agent, "_create_request_anthropic_client", _fake_anthropic_client)
    else:
        monkeypatch.setattr(helpers._StreamingCall, "_open_chat_stream",
                            lambda self, stream_kwargs: _LateStream(self, reader))
    with pytest.raises(InterruptedError):
        helpers.interruptible_streaming_api_call(
            agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_interrupt_during_connect_window_shuts_down_the_late_socket(monkeypatch, api_mode):
    reader, writer = _socket.socketpair()
    try:
        _run_interrupted_during_connect(monkeypatch, _agent(), reader, api_mode)
        _assert_shut_down(reader, writer)
    finally:
        reader.close()
        writer.close()
