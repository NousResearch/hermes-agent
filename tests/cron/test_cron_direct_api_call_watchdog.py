"""Regression guard for #80759 — the inline non-streaming call must be bounded.

Cron turns (and delegated children) are routed by ``should_use_direct_api_call``
onto ``direct_api_call``, which ran the request inline with no stale detector:
the abort plumbing was registered but nothing ever invoked it. A provider that
accepted the request and then went silent — connection held open, zero bytes,
no error — hung the cron run until an external actor killed it (which also
orphaned the execution row). The only other bound was the 1800s-default httpx
read timeout, and the job-level inactivity monitor was observed not to fire.

These tests pin the watchdog contract: it aborts the in-flight sockets through
the already-registered abort hook, surfaces a retryable ``TimeoutError`` (never
``InterruptedError``), feeds the cross-turn stale circuit breaker, and stays
out of the way of a healthy call. They also pin #85252: the keepalive httpx
client uses ``read=None``, so a stranger-thread abort that finds no sockets
must not leave the call unbounded — ``direct_api_call`` injects a per-call
read timeout matching the stale budget as a hard backstop.
"""

import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent

from agent.chat_completion_helpers import direct_api_call


def _make_agent(*, stale_timeout, platform="cron"):
    agent = MagicMock()
    agent.platform = platform
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._touch_activity = MagicMock()
    agent._buffer_status = MagicMock()
    agent._create_request_openai_client = MagicMock()
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    agent._compute_non_stream_stale_timeout = lambda api_payload: stale_timeout
    return agent


def _stalling_client(agent, *, aborted, release_after=5.0):
    """A client whose request blocks until the watchdog aborts its sockets."""
    fake_client = MagicMock()
    release_after_abort = threading.Event()

    def _abort(client, reason):
        aborted.append(reason)
        release_after_abort.set()

    def _stalled_request(**_kwargs):
        # The provider accepted the request and went silent. The socket
        # shutdown is what unblocks it, exactly as in production.
        if not release_after_abort.wait(timeout=release_after):
            raise AssertionError("watchdog never aborted the stalled request")
        raise ConnectionError("socket shut down")

    fake_client.chat.completions.create.side_effect = _stalled_request
    agent._abort_request_openai_client.side_effect = _abort
    agent._create_request_openai_client.return_value = fake_client
    return fake_client


def test_stalled_inline_call_is_aborted_and_raises_retryable_timeout():
    agent = _make_agent(stale_timeout=0.2)
    aborted: list[str] = []
    _stalling_client(agent, aborted=aborted)

    started = time.time()
    with pytest.raises(TimeoutError) as excinfo:
        direct_api_call(agent, {"model": "m", "messages": []})
    elapsed = time.time() - started

    assert aborted == ["stale_call_kill"]
    assert "no response" in str(excinfo.value)
    assert elapsed < 4.0, "watchdog did not bound the call"


def test_watchdog_abort_never_surfaces_as_interrupted_error():
    """InterruptedError means "the user wants to stop" — the outer loop does
    not retry it. A watchdog abort must stay retryable."""
    agent = _make_agent(stale_timeout=0.2)
    _stalling_client(agent, aborted=[])

    with pytest.raises(TimeoutError):
        direct_api_call(agent, {"model": "m", "messages": []})


def test_watchdog_kill_feeds_the_cross_turn_stale_circuit_breaker():
    """Without a bump the #58962 breaker can never trip for cron sessions."""
    agent = _make_agent(stale_timeout=0.2)
    _stalling_client(agent, aborted=[])

    with pytest.raises(TimeoutError):
        direct_api_call(agent, {"model": "m", "messages": []})

    assert agent._consecutive_stale_streams == 1


def test_retry_after_a_watchdog_kill_gets_a_fresh_pool_and_succeeds():
    agent = _make_agent(stale_timeout=0.2)
    _stalling_client(agent, aborted=[])

    with pytest.raises(TimeoutError):
        direct_api_call(agent, {"model": "m", "messages": []})

    # The kill must really close the wire client so the retry rebuilds it.
    assert agent._close_request_openai_client.call_args.kwargs["reason"] == (
        "request_error_cleanup"
    )

    healthy_client = MagicMock()
    healthy_client.chat.completions.create.return_value = SimpleNamespace(id="ok")
    agent._create_request_openai_client.side_effect = None
    agent._create_request_openai_client.return_value = healthy_client

    assert direct_api_call(agent, {"model": "m", "messages": []}).id == "ok"
    assert agent._consecutive_stale_streams == 0


def test_healthy_call_is_untouched_by_the_watchdog():
    agent = _make_agent(stale_timeout=30.0)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(id="fast")
    agent._create_request_openai_client.return_value = fake_client

    assert direct_api_call(agent, {"model": "m", "messages": []}).id == "fast"
    agent._abort_request_openai_client.assert_not_called()
    assert agent._close_request_openai_client.call_args.kwargs["reason"] == (
        "request_complete"
    )


def test_local_endpoint_infinite_budget_leaves_the_watchdog_disarmed():
    """``_compute_non_stream_stale_timeout`` returns inf for a local endpoint
    on the implicit default — that opt-out must survive on this path too."""
    agent = _make_agent(stale_timeout=float("inf"))
    fake_client = MagicMock()
    started = threading.Event()
    release = threading.Event()

    def _slow(**_kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return SimpleNamespace(id="slow-but-healthy")

    fake_client.chat.completions.create.side_effect = _slow
    agent._create_request_openai_client.return_value = fake_client

    box = {}

    def _run():
        box["response"] = direct_api_call(agent, {"model": "m", "messages": []})

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert started.wait(timeout=2.0)
    time.sleep(0.3)
    release.set()
    worker.join(timeout=3.0)

    assert box["response"].id == "slow-but-healthy"
    agent._abort_request_openai_client.assert_not_called()


def test_watchdog_uses_the_same_budget_as_the_interrupt_worker_path():
    """The budget comes from ``_compute_non_stream_stale_timeout`` — the same
    resolver the worker path's stale detector uses — with the live request
    payload, so provider config and context scaling both apply."""
    seen: list[dict] = []
    agent = _make_agent(stale_timeout=30.0)
    agent._compute_non_stream_stale_timeout = lambda payload: (
        seen.append(payload) or 30.0
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(id="ok")
    agent._create_request_openai_client.return_value = fake_client

    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    direct_api_call(agent, payload)

    assert seen == [payload]


# ---------------------------------------------------------------------------
# End-to-end: a real AIAgent on the real cron routing + client lifecycle.
# ---------------------------------------------------------------------------


class _StallingWireClient:
    """An OpenAI-shaped client whose request stalls until its sockets die."""

    def __init__(self):
        self._client = SimpleNamespace(is_closed=False)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.responses = SimpleNamespace()
        self.close_calls = 0
        self.sockets_shut_down = threading.Event()

    def _create(self, **_kwargs):
        if not self.sockets_shut_down.wait(timeout=5.0):
            raise AssertionError("watchdog never shut the stalled request down")
        raise ConnectionError("socket shut down")

    def close(self):
        self.close_calls += 1
        self._client.is_closed = True


def _build_cron_agent(monkeypatch):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = "cron"
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent._base_url = agent.base_url
    agent.model = "some/model"
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._client_lock = threading.RLock()
    agent._client_kwargs = {"api_key": "***", "base_url": agent.base_url}
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.reasoning_callback = None
    agent.status_callback = None
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "0.3")
    return agent


def test_e2e_cron_turn_is_bounded_through_the_real_agent_routing(monkeypatch):
    """The real chain: cron platform → ``should_use_direct_api_call`` →
    ``direct_api_call`` → the agent's own stale-timeout resolver → the real
    cross-thread abort → a retryable ``TimeoutError`` on the caller."""
    wire = _StallingWireClient()
    agent = _build_cron_agent(monkeypatch)
    agent.client = wire
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kwargs: wire)
    monkeypatch.setattr(
        run_agent.AIAgent,
        "_force_close_tcp_sockets",
        lambda self, client: (client.sockets_shut_down.set(), 1)[1],
    )

    started = time.time()
    with pytest.raises(TimeoutError):
        agent._interruptible_api_call({"model": agent.model, "messages": []})
    elapsed = time.time() - started

    assert elapsed < 4.0, "cron turn was not bounded by the watchdog"
    # The aborted pool is poisoned, so the killed client is really closed
    # instead of being cached for the retry.
    assert wire.close_calls == 1


# ---------------------------------------------------------------------------
# Salvage follow-up (#75301 state discipline): locked lifecycle transitions.
# ---------------------------------------------------------------------------


def test_interrupt_abort_is_not_misclassified_as_provider_staleness():
    """A user/monitor interrupt that kills the call must not advance the
    cross-turn stale circuit breaker, even if the stale timer fires right
    after — the ``cancelled`` flag owns the outcome (#75301 design)."""
    agent = _make_agent(stale_timeout=30.0)
    release_after_abort = threading.Event()
    fake_client = MagicMock()

    def _stalled_request(**_kwargs):
        assert release_after_abort.wait(timeout=5.0)
        raise ConnectionError("socket shut down")

    fake_client.chat.completions.create.side_effect = _stalled_request
    agent._create_request_openai_client.return_value = fake_client

    def _abort(client, reason):
        release_after_abort.set()

    agent._abort_request_openai_client.side_effect = _abort

    interrupt_fired = threading.Event()

    def _interrupt_soon():
        # Wait until the abort hook is registered, then interrupt like
        # run_agent.interrupt() does (registered under the same name).
        deadline = time.time() + 2.0
        while time.time() < deadline:
            hook = agent._active_request_abort
            if callable(hook) and not isinstance(hook, MagicMock):
                agent._interrupt_requested = True
                # interrupt owns the outcome...
                assert hook("interrupt_abort") is False
                # ...so a stale timer racing in afterwards is inert:
                assert hook("stale_call_kill") is False
                interrupt_fired.set()
                return
            time.sleep(0.005)

    worker = threading.Thread(target=_interrupt_soon, daemon=True)
    worker.start()

    with pytest.raises(InterruptedError):
        direct_api_call(agent, {"model": "m", "messages": []})

    assert interrupt_fired.wait(timeout=1.0)
    assert agent._consecutive_stale_streams == 0, (
        "interrupt was misclassified as provider staleness"
    )


def test_late_stale_timer_after_completion_is_inert():
    """A timer callback that loses the race to a completed request must not
    bump the streak: ``done`` is set under the lock before the unwind."""
    agent = _make_agent(stale_timeout=30.0)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(id="ok")
    agent._create_request_openai_client.return_value = fake_client

    captured = {}
    real_timer = threading.Timer

    class CapturingTimer(real_timer):
        def __init__(self, interval, function, *a, **k):
            captured["fn"] = function
            super().__init__(interval, function, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(threading, "Timer", CapturingTimer)
        assert direct_api_call(agent, {"model": "m", "messages": []}).id == "ok"

    # Fire the (already-cancelled) timer callback manually, simulating a
    # timer thread that had already dequeued before cancel().
    captured["fn"]()
    assert agent._consecutive_stale_streams == 0
    agent._abort_request_openai_client.assert_not_called()


def test_timer_firing_before_client_registration_fails_the_dispatch():
    """Registration race: if the budget expires while the client is still
    being constructed, the freshly-registered client is aborted and the call
    fails with a retryable TimeoutError instead of opening a new socket
    after the only watchdog was spent."""
    agent = _make_agent(stale_timeout=0.05)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(id="late")

    def _slow_create(*, reason, api_kwargs):
        # The timer (50ms budget) fires while construction is in flight.
        time.sleep(0.4)
        return fake_client

    agent._create_request_openai_client.side_effect = _slow_create

    with pytest.raises(TimeoutError):
        direct_api_call(agent, {"model": "m", "messages": []})

    fake_client.chat.completions.create.assert_not_called()
    agent._abort_request_openai_client.assert_called_once_with(
        fake_client, reason="stale_call_kill"
    )


def test_resolver_exception_propagates_instead_of_disarming_the_watchdog():
    """A raising stale-timeout resolver must propagate (fail closed), not be
    swallowed into an infinite budget that silently reinstates the hang."""
    agent = _make_agent(stale_timeout=0.2)

    def _broken_resolver(api_payload):
        raise RuntimeError("resolver regression")

    agent._compute_non_stream_stale_timeout = _broken_resolver

    with pytest.raises(RuntimeError, match="resolver regression"):
        direct_api_call(agent, {"model": "m", "messages": []})


# ---------------------------------------------------------------------------
# #85252: hard socket bound when stranger-thread abort cannot kill the recv.
# ---------------------------------------------------------------------------


def test_inline_hard_timeout_matches_stale_budget():
    """Keepalive httpx uses read=None. The injected timeout's read budget
    must equal the stale watchdog so a no-op abort cannot hang for hours."""
    from agent.chat_completion_helpers import _inline_nonstream_hard_timeout

    timeout = _inline_nonstream_hard_timeout(600.0)
    assert timeout is not None
    assert timeout.read == 600.0
    assert timeout.connect == 60.0
    assert timeout.write == 60.0
    assert timeout.pool == 60.0


def test_inline_hard_timeout_disarmed_when_watchdog_is_disarmed():
    from agent.chat_completion_helpers import _inline_nonstream_hard_timeout

    assert _inline_nonstream_hard_timeout(float("inf")) is None
    assert _inline_nonstream_hard_timeout(0) is None
    assert _inline_nonstream_hard_timeout(-1) is None


def test_inline_call_passes_hard_read_timeout_to_the_sdk():
    """The bound has to actually reach chat.completions.create — a helper
    that is never wired in would leave cron on read=None (#85252)."""
    agent = _make_agent(stale_timeout=0.5)
    fake_client = MagicMock()
    captured = {}

    def _create(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(id="ok")

    fake_client.chat.completions.create.side_effect = _create
    agent._create_request_openai_client.return_value = fake_client

    assert direct_api_call(agent, {"model": "m", "messages": []}).id == "ok"
    timeout = captured["timeout"]
    assert timeout is not None
    assert timeout.read == 0.5


def test_inline_call_does_not_override_explicit_timeout():
    """A transport/provider that already set timeout= must keep it."""
    agent = _make_agent(stale_timeout=30.0)
    fake_client = MagicMock()
    captured = {}

    def _create(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(id="ok")

    fake_client.chat.completions.create.side_effect = _create
    agent._create_request_openai_client.return_value = fake_client

    assert direct_api_call(
        agent, {"model": "m", "messages": [], "timeout": 12.0}
    ).id == "ok"
    assert captured["timeout"] == 12.0


def test_infinite_budget_does_not_inject_a_hard_timeout():
    agent = _make_agent(stale_timeout=float("inf"))
    fake_client = MagicMock()
    captured = {}

    def _create(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(id="ok")

    fake_client.chat.completions.create.side_effect = _create
    agent._create_request_openai_client.return_value = fake_client

    assert direct_api_call(agent, {"model": "m", "messages": []}).id == "ok"
    assert "timeout" not in captured or captured["timeout"] is None
