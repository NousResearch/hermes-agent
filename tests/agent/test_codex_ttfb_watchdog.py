"""Regression tests for the Codex time-to-first-byte (TTFB) watchdog.

The chatgpt.com/backend-api/codex endpoint has an intermittent failure mode
where it accepts the connection but never emits a single stream event. The
watchdog in ``interruptible_api_call`` kills such a connection at a short TTFB
cutoff (instead of waiting out the much longer wall-clock stale timeout) so the
retry loop can reconnect promptly. Once any stream event arrives, the TTFB
watchdog is satisfied and a separate idle watchdog handles streams that stop
emitting SSE events.

The "bytes flowing" signal is ``agent._codex_stream_last_event_ts``, set on
*any* event by ``codex_runtime.run_codex_stream`` — so reasoning-only or
tool-call-only turns (which emit no output-text deltas) are not mistaken for a
stall.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # The watchdog is gated on the codex_responses api_mode; assert/force it so
    # the test is robust to detection-logic changes elsewhere.
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    # Keep the wall-clock stale timeout high so any early kill is unambiguously
    # the TTFB path, not the stale-call path.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0
    )
    return agent


def test_ttfb_kills_when_no_stream_event(tmp_path, monkeypatch):
    """Backend accepts the connection but emits no event -> killed at the TTFB
    cutoff, well before the 60s wall-clock stale timeout, with a retryable
    TimeoutError and a ``codex_ttfb_kill`` close reason."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        # Never set _codex_stream_last_event_ts: simulate zero events arriving.
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        elapsed = time.time() - t0
        assert "TTFB" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
        # ~1s cutoff + 2s join grace; must be far under the 60s stale timeout.
        assert elapsed < 15, f"TTFB watchdog took {elapsed:.1f}s"
    finally:
        stop["flag"] = True


def test_ttfb_default_tolerates_slow_first_event(tmp_path, monkeypatch):
    """With no env var set, the no-byte TTFB default is generous (120s), so a
    request whose first stream event is merely slow (~2s of backend admission /
    prefill) is NOT killed. This is the subscription-backed Codex case the tight
    12s default used to abort mid-prefill."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    # Default behavior: no explicit TTFB override.
    monkeypatch.delenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_CODEX_TTFB_MAX_SECONDS", raising=False)

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_slow_first_event(api_kwargs, client=None, on_first_delta=None):
        # Backend is alive but slow to admit: first event lands after ~2s,
        # well under the 120s default cutoff. Mark the first byte so the
        # no-byte detector sees activity, then return the response.
        time.sleep(2.0)
        agent._codex_stream_last_event_ts = time.time()
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_slow_first_event)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_ttfb_includes_silent_hang_hint_for_gpt_5_5(tmp_path, monkeypatch):
    """The no-first-byte watchdog should surface the same actionable hint as the
    stale-call timeout path when the model matches the silent-hang heuristic."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    statuses: list[str] = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_buffer_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(agent, "_emit_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        message = str(excinfo.value)
        assert "gpt-5.4" in message
        assert "gpt-5.3-codex" in message
        assert "gpt-5.4-codex" in message
        assert "codex_ttfb_kill" in closes
        assert statuses, "expected a user-facing watchdog status"
        assert any("gpt-5.4" in s and "gpt-5.3-codex" in s for s in statuses)
    finally:
        stop["flag"] = True


def test_ttfb_high_env_is_capped_for_openai_codex(tmp_path, monkeypatch):
    """A stale local env value like 90s must not make openai-codex wait 90s
    before reconnecting when the backend emits no SSE frames."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("HERMES_CODEX_TTFB_MAX_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.4", "input": "hi"})
        elapsed = time.time() - t0
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
        assert elapsed < 15, f"TTFB watchdog ignored cap and took {elapsed:.1f}s"
    finally:
        stop["flag"] = True


def test_ttfb_does_not_kill_when_events_flow(tmp_path, monkeypatch):
    """Once a stream event has arrived, a generation that runs past the TTFB
    cutoff is NOT killed by the watchdog — it completes normally."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # Bytes flowing: mark stream activity right away, then keep generating
        # past the 1s TTFB cutoff before returning a real response.
        agent._codex_stream_last_event_ts = time.time()
        if on_first_delta:
            on_first_delta()
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_event_idle_kills_after_first_event_then_silence(tmp_path, monkeypatch):
    """If Codex emits an opening SSE event and then goes silent, kill it via
    the stream-idle watchdog instead of waiting for the long non-stream stale
    timeout."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        agent._codex_stream_last_event_ts = time.time()
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        assert "after first byte" in str(excinfo.value)
        assert "codex_stream_idle_kill" in closes
        assert "codex_ttfb_kill" not in closes
    finally:
        stop["flag"] = True


def test_ttfb_disabled_via_env_zero(tmp_path, monkeypatch):
    """Setting HERMES_CODEX_TTFB_TIMEOUT_SECONDS=0 disables the TTFB watchdog;
    a no-event stall then falls through to the (here, 60s) stale timeout, so a
    short hang is NOT killed by TTFB."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # No event marker, but only briefly — well under the 60s stale timeout.
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_large_codex_request_waits_instead_of_ttfb_reconnect(tmp_path, monkeypatch):
    """Large Codex inputs can legitimately take longer than the small-request
    first-byte cutoff before the first SSE frame. Scale the TTFB timeout up
    for those requests instead of killing/retrying at the small-request cutoff."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason)
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason)
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # No event marker for 2s: this would trip the 1s TTFB watchdog on a
        # small request, but should be allowed for a large request.
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    large_input = "x" * 44_000  # ~11k estimated tokens, above the 10k gate.
    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_large_codex_request_can_still_ttfb_reconnect_when_capped(tmp_path, monkeypatch):
    """Large Codex requests should keep a finite TTFB watchdog instead of
    disabling it entirely. A low max cap should still force an early reconnect."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("HERMES_CODEX_TTFB_MAX_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason)
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason)
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000  # ~11k estimated tokens, above the large-request gate.
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
    finally:
        stop["flag"] = True


def test_large_codex_request_strict_ttfb_env_still_reconnects(tmp_path, monkeypatch):
    """Operators can force the old early-reconnect behavior for large inputs
    with HERMES_CODEX_TTFB_STRICT=1."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("HERMES_CODEX_TTFB_STRICT", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason)
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason)
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
    finally:
        stop["flag"] = True


def test_large_codex_request_hard_ceiling_reclaims_silent_stall(tmp_path, monkeypatch):
    """#64507 regression: a large Codex request (TTFB watchdog disabled by the
    size gate, stale floor *raised*) that never emits a single byte must still
    be reclaimed at a finite hard ceiling — not hang for 13+ minutes while the
    worker stays idle and the session shows as active.

    Uses the real default TTFB threshold (120s) and asserts the request dies at
    the hard ceiling regardless of the size-based TTFB disable.
    """
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    # Real default TTFB threshold (no HERMES_CODEX_TTFB_* override) → for a
    # >10k-token request the no-byte TTFB watchdog is auto-disabled.
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "3")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        # No event marker AND no event ever: the exact issue-64507 stall.
        deadline = time.time() + 120
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000  # ~11k estimated tokens → TTFB disabled, stale raised
    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        elapsed = time.time() - t0
        # Must die at the hard ceiling (3s), nowhere near the raised stale floor.
        assert elapsed < 30, f"hard ceiling took {elapsed:.1f}s — stall not reclaimed"
        assert "stale_call_kill" in closes, f"stale kill expected, got {closes}"
        assert "timed out after" in str(excinfo.value)
        assert "with no response" in str(excinfo.value)
    finally:
        stop["flag"] = True


def test_large_codex_request_hard_ceiling_disabled_restores_legacy(tmp_path, monkeypatch):
    """Setting HERMES_CODEX_HARD_TIMEOUT_SECONDS=0 disables the ceiling entirely,
    restoring the pre-#64507 behavior (request waits out the raised stale floor
    instead of being capped). Keeps the knob for operators who must.
    """
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # No event, but only briefly — well under the (here 60s) stale timeout.
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    large_input = "x" * 44_000
    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes
    assert "stale_call_kill" not in closes


def test_large_codex_request_hard_ceiling_caps_raised_stale_floor(tmp_path, monkeypatch):
    """The hard ceiling must cap the raised stale floor (openai-codex can push
    the stale timeout to 1200s at >100k tokens). A large silent stall must die
    at the ceiling, proving the min() wins over the floor.
    """
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "4")
    # Force the >100k-token tier so openai_codex_stale_timeout_floor returns 1200s.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 1200.0
    )

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 200
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    huge_input = "x" * 500_000  # ~125k tokens → stale floor 1200s
    t0 = time.time()
    try:
        with pytest.raises(TimeoutError):
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": huge_input})
        elapsed = time.time() - t0
        assert elapsed < 40, f"hard ceiling lost to stale floor: {elapsed:.1f}s"
        assert "stale_call_kill" in closes, f"stale kill expected, got {closes}"
    finally:
        stop["flag"] = True
