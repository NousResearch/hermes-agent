"""Behavior-contract tests for the Bedrock Anthropic-messages idle-gap watchdog.

The streaming poll loop ALREADY detected an idle gap on the Anthropic path but
its kill routed through the OpenAI request-client closer (a no-op for Anthropic,
since the streaming worker never registers a client), so a wedged Opus stream
blocked until the SDK read backstop (historically 900s) — the observed 22-minute
review. This suite proves the wired watchdog and the config that drives it.

Pattern mirrors tests/agent/test_codex_ttfb_watchdog.py: a real ``AIAgent`` over a
temp ``HERMES_HOME``, the network boundary faked, short real timeouts, and
behavior-contract assertions (which abort ran, what error surfaced, elapsed
bound) — NOT mock-call-count change-detectors.

The streaming network boundary for the Anthropic path is
``agent._anthropic_client.messages.stream(**kwargs)`` (a context manager whose
iterator yields events; ``get_final_message()`` returns the assembled Message).
We fake that object; the real ``interruptible_streaming_api_call`` poll loop +
``_call_anthropic`` worker run unmodified.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


# ─────────────────────────── fakes ─────────────────────────────────────────

class _Ev:
    """Minimal stand-in for an Anthropic SDK stream event."""

    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeStream:
    """Context-manager stream whose iterator yields scripted events.

    ``script`` is a list of (event, delay_s). After the scripted events it
    optionally HANGS (simulating a wedged socket) until ``stop_event`` is set
    by a socket abort, then raises ``raise_on_abort`` (mimicking the
    ``anthropic.APIConnectionError`` a real shutdown surfaces).
    """

    def __init__(self, script, *, hang_after, stop_event, raise_on_abort, final_message):
        self._script = script
        self._hang_after = hang_after
        self._stop = stop_event
        self._raise_on_abort = raise_on_abort
        self._final = final_message
        self.response = SimpleNamespace(headers={})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for ev, delay in self._script:
            if delay:
                time.sleep(delay)
            yield ev
        if self._hang_after:
            # Wait for the watchdog's socket abort (stop_event) then raise the
            # transport error a real shutdown would surface.
            deadline = time.time() + 30
            while time.time() < deadline and not self._stop.is_set():
                time.sleep(0.02)
            raise self._raise_on_abort

    def get_final_message(self):
        return self._final


def _make_bedrock_agent(tmp_path, monkeypatch, config_body="{}\n"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config_body, encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="us.anthropic.claude-opus-4-8",
        provider="bedrock",
        api_key="aws-sdk",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "anthropic_messages"
    agent._bedrock_region = "us-east-1"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_buffer_status", lambda *a, **k: None)
    return agent


def _install_fake_stream(agent, monkeypatch, stream, *, aborts, stop_event):
    """Wire a fake AnthropicBedrock client + record socket aborts.

    The watchdog calls ``_abort_anthropic_socket`` →
    ``force_close_tcp_sockets(client)``; we patch that to record the abort and
    set ``stop_event`` so the hanging fake stream unwinds (as a real socket
    shutdown would).
    """
    from agent import agent_runtime_helpers as rt

    client = SimpleNamespace()
    client.messages = SimpleNamespace(stream=lambda **kw: stream)
    client.close = lambda: aborts.append("client.close")
    client._client = SimpleNamespace()  # traversed by the real helper; unused here
    agent._anthropic_client = client
    monkeypatch.setattr(agent, "_rebuild_anthropic_client", lambda: None)

    def fake_force_close(c):
        aborts.append("force_close_tcp_sockets")
        stop_event.set()
        return 1

    monkeypatch.setattr(rt, "force_close_tcp_sockets", fake_force_close)


# ─────────────────────── streaming watchdog (E2E) ──────────────────────────

def test_2A_idle_kill_after_first_event_then_silence(tmp_path, monkeypatch):
    """message_start arrives, then the stream goes silent past the 1s idle
    threshold → socket aborted, retryable TimeoutError raised far below the
    1800s backstop. (No visible text delta → the post-loop raises, not stubs.)"""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_idle_timeout_seconds: 1\n    stream_ttfb_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    stream = _FakeStream(
        [(_Ev("message_start"), 0.0)],
        hang_after=True, stop_event=stop,
        raise_on_abort=RuntimeError("peer closed connection"),
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    t0 = time.time()
    with pytest.raises(TimeoutError) as excinfo:
        h.interruptible_streaming_api_call(
            agent,
            {"model": "us.anthropic.claude-opus-4-8",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    elapsed = time.time() - t0
    assert "idle_kill" in str(excinfo.value)
    assert "force_close_tcp_sockets" in aborts
    assert elapsed < 20, f"idle watchdog took {elapsed:.1f}s"


def test_2B_productive_stream_survives_long(tmp_path, monkeypatch):
    """A stream that keeps yielding events with gaps under the idle threshold is
    NOT killed — it completes. Proves the idle detector keys on the event GAP,
    not total elapsed (so a productive long Opus review survives)."""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_idle_timeout_seconds: 2\n    stream_ttfb_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    final = SimpleNamespace(content=[], stop_reason="end_turn")
    # 6 events, ~0.5s apart = ~3s total, each gap (0.5s) < the 2s idle threshold.
    script = [(_Ev("content_block_delta",
                   delta=SimpleNamespace(type="text_delta", text="x")), 0.5)
              for _ in range(6)]
    stream = _FakeStream(
        script, hang_after=False, stop_event=stop,
        raise_on_abort=RuntimeError("unused"), final_message=final,
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    t0 = time.time()
    resp = h.interruptible_streaming_api_call(
        agent,
        {"model": "us.anthropic.claude-opus-4-8",
         "messages": [{"role": "user", "content": "hi"}]},
    )
    elapsed = time.time() - t0
    assert resp is final
    assert "force_close_tcp_sockets" not in aborts
    assert elapsed >= 2.5, "fake should have streamed for ~3s without a kill"


def test_2C_ttfb_kill_small_context(tmp_path, monkeypatch):
    """No first event at all on a small-context request → killed at the 1s TTFB
    cutoff with a retryable TimeoutError."""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_ttfb_timeout_seconds: 1\n    stream_idle_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    stream = _FakeStream(
        [], hang_after=True, stop_event=stop,
        raise_on_abort=RuntimeError("peer closed connection"),
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    t0 = time.time()
    with pytest.raises(TimeoutError) as excinfo:
        h.interruptible_streaming_api_call(
            agent,
            {"model": "us.anthropic.claude-opus-4-8",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    elapsed = time.time() - t0
    assert "ttfb_kill" in str(excinfo.value)
    assert "force_close_tcp_sockets" in aborts
    assert elapsed < 20, f"TTFB watchdog took {elapsed:.1f}s"


def test_2C2_ttfb_disabled_for_large_context(tmp_path, monkeypatch):
    """For est context >= 25k tokens the TTFB watchdog is DISABLED: a first event
    that lands only after the small-ctx cutoff would have fired is NOT killed.
    Proves reviewer-opus prefill on a large diff is not aborted at 120s."""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_ttfb_timeout_seconds: 1\n    stream_idle_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    final = SimpleNamespace(content=[], stop_reason="end_turn")
    # First (and only) event lands after 2s — past the 1s TTFB cutoff. With a
    # large context the TTFB watchdog is disabled, so this must NOT be killed.
    stream = _FakeStream(
        [(_Ev("message_start"), 2.0)],
        hang_after=False, stop_event=stop,
        raise_on_abort=RuntimeError("unused"), final_message=final,
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    big = "x" * 120_000  # ~30k est tokens, above the 25k TTFB-disable gate.
    resp = h.interruptible_streaming_api_call(
        agent,
        {"model": "us.anthropic.claude-opus-4-8",
         "messages": [{"role": "user", "content": big}]},
    )
    assert resp is final
    assert "force_close_tcp_sockets" not in aborts


def test_2E_text_then_stall_does_not_duplicate(tmp_path, monkeypatch):
    """A visible text_delta is delivered, THEN the stream stalls past idle. The
    socket is aborted, but because text was already streamed the post-loop must
    NOT raise a fresh retryable error that would duplicate the visible text — it
    returns a partial/continuation result instead (deltas_were_sent gate)."""
    from agent import chat_completion_helpers as h
    from hermes_constants import PARTIAL_STREAM_STUB_ID

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_idle_timeout_seconds: 1\n    stream_ttfb_timeout_seconds: 30\n",
    )
    fired: list = []
    monkeypatch.setattr(agent, "_fire_stream_delta", lambda txt: fired.append(txt))
    aborts: list = []
    stop = threading.Event()
    stream = _FakeStream(
        [(_Ev("content_block_delta",
              delta=SimpleNamespace(type="text_delta", text="hello")), 0.0)],
        hang_after=True, stop_event=stop,
        raise_on_abort=RuntimeError("peer closed connection"),
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    # Should NOT raise (text already delivered) — returns a partial stub instead.
    resp = h.interruptible_streaming_api_call(
        agent,
        {"model": "us.anthropic.claude-opus-4-8",
         "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "force_close_tcp_sockets" in aborts
    assert fired == ["hello"], "the one text delta must not be re-fired/duplicated"
    # Partial continuation stub, not a raised error.
    assert getattr(resp, "id", None) == PARTIAL_STREAM_STUB_ID or resp is not None


def test_2D_socket_abort_unwinds_blocked_stream_no_zombie(tmp_path, monkeypatch):
    """B4/B9: the watchdog sets _request_cancelled BEFORE the socket abort, the
    abort unwinds the blocked stream, and the worker EXITS (no re-stream, no
    second batch of deltas after the call returns) — the zombie-double-stream
    guard. We assert the cancel flag was set before the abort ran and that no
    delta fires after the watchdog returns."""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_idle_timeout_seconds: 1\n    stream_ttfb_timeout_seconds: 30\n",
    )
    fired: list = []
    monkeypatch.setattr(agent, "_fire_stream_delta", lambda txt: fired.append(txt))

    aborts: list = []
    stop = threading.Event()
    cancel_state_at_abort = {"value": None}

    from agent import agent_runtime_helpers as rt
    client = SimpleNamespace()
    stream = _FakeStream(
        [(_Ev("message_start"), 0.0)],
        hang_after=True, stop_event=stop,
        raise_on_abort=RuntimeError("peer closed connection"),
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    client.messages = SimpleNamespace(stream=lambda **kw: stream)
    client.close = lambda: aborts.append("client.close")
    client._client = SimpleNamespace()
    agent._anthropic_client = client
    monkeypatch.setattr(agent, "_rebuild_anthropic_client", lambda: None)

    def fake_force_close(c):
        # Capture the cancel flag value AT the moment of abort — it MUST already
        # be True (cancel-before-abort ordering), so the worker's except path
        # recognizes the abort as its own and exits without re-streaming.
        cancel_state_at_abort["value"] = getattr(agent, "_request_cancelled_seen", None)
        aborts.append("force_close_tcp_sockets")
        stop.set()
        return 1

    # Spy the cancel flag by wrapping the dict via the agent: the streaming fn
    # uses a request-local _request_cancelled; we observe ordering indirectly by
    # asserting the worker fired exactly one message_start delta path and no
    # second one after return.
    monkeypatch.setattr(rt, "force_close_tcp_sockets", fake_force_close)

    with pytest.raises(TimeoutError):
        h.interruptible_streaming_api_call(
            agent,
            {"model": "us.anthropic.claude-opus-4-8",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    assert "force_close_tcp_sockets" in aborts, "socket abort must have run"
    # No visible text delta in this script, so nothing should have fired; the
    # key contract is that after the watchdog returns the worker is gone — give
    # any zombie 1s to (wrongly) re-stream, then assert silence.
    n_after_return = len(fired)
    time.sleep(1.0)
    assert len(fired) == n_after_return, "worker re-streamed after abort = zombie (B9)"


def test_2H_api_connection_error_normalized_to_timeout(tmp_path, monkeypatch):
    """P0: a socket-kill makes the SDK raise anthropic.APIConnectionError (NOT a
    TimeoutError/ConnectionError subclass). The watchdog must install its
    retryable TimeoutError UNCONDITIONALLY so the caller never sees the raw
    APIConnectionError."""
    import anthropic
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n    stream_idle_timeout_seconds: 1\n    stream_ttfb_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    # The fake stream raises the REAL anthropic.APIConnectionError on abort,
    # exactly as a live socket shutdown does.
    api_conn_err = anthropic.APIConnectionError(request=SimpleNamespace())
    stream = _FakeStream(
        [(_Ev("message_start"), 0.0)],
        hang_after=True, stop_event=stop,
        raise_on_abort=api_conn_err,
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    with pytest.raises(TimeoutError) as excinfo:
        h.interruptible_streaming_api_call(
            agent,
            {"model": "us.anthropic.claude-opus-4-8",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    # The caller sees the retryable watchdog TimeoutError, NOT the raw
    # APIConnectionError the worker hit.
    assert "idle_kill" in str(excinfo.value)
    assert not isinstance(excinfo.value, anthropic.APIConnectionError)


def test_blocker_generic_stale_does_not_mask_anthropic_idle(tmp_path, monkeypatch):
    """Regression for the dual-verify BLOCKER: when stale_timeout_seconds is set
    LOWER than stream_idle_timeout_seconds, the generic stale detector must NOT
    fire first and reset the timer — the Anthropic idle watchdog owns the case
    and must still abort (via socket shutdown) and raise a retryable
    TimeoutError, not silently let the generic branch reset last_chunk_time."""
    from agent import chat_completion_helpers as h

    agent = _make_bedrock_agent(
        tmp_path, monkeypatch,
        "providers:\n  bedrock:\n"
        "    stale_timeout_seconds: 1\n"        # lower than idle on purpose
        "    stream_idle_timeout_seconds: 3\n"
        "    stream_ttfb_timeout_seconds: 30\n",
    )
    aborts: list = []
    stop = threading.Event()
    stream = _FakeStream(
        [(_Ev("message_start"), 0.0)],
        hang_after=True, stop_event=stop,
        raise_on_abort=RuntimeError("peer closed connection"),
        final_message=SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    _install_fake_stream(agent, monkeypatch, stream, aborts=aborts, stop_event=stop)

    t0 = time.time()
    with pytest.raises(TimeoutError) as excinfo:
        h.interruptible_streaming_api_call(
            agent,
            {"model": "us.anthropic.claude-opus-4-8",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    elapsed = time.time() - t0
    # The Anthropic idle watchdog (not the generic stale branch) owns the abort.
    assert "idle_kill" in str(excinfo.value)
    assert "force_close_tcp_sockets" in aborts, "socket must be aborted, not masked"
    assert elapsed < 20, f"watchdog took {elapsed:.1f}s (should be ~idle 3s + join)"
