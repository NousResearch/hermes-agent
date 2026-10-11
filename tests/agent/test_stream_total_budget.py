"""A streaming request needs a total wall-clock budget, not only an idle detector (#5450).

Every existing bound on the main loop's streaming path is an IDLE bound: the socket
read timeout (per read), the stale-stream detector (per chunk gap, and it re-arms
from ``last_chunk_time`` on every chunk), and the turn-liveness watchdog (which
``_touch_activity`` re-arms on chunk arrival too). A provider trickling one token
per interval therefore never crosses any of them, and the agent loop is blocked
indefinitely -- the general case the stale detector, by its own docstring, does
not cover.

The budget is deliberately WALL-CLOCK: the relation under test is that a request
older than its budget is ended even while chunks are still arriving, while the
same request inside its budget is left alone. ``last_chunk_time`` is held fresh
throughout so a passing test cannot be explained by idle detection.

Expiry ABORTS the request into the existing transient-error ladder -- the same
cancellation the stale kill already performs, so already-delivered text survives
through the shipped partial-delivery stub. It must NOT count toward the
cross-turn stale breaker: a chunk-trickling provider is demonstrably alive, and
triping the breaker there would stop a healthy provider from streaming.
"""
import threading
import time
from pathlib import Path

from agent import chat_completion_helpers as helpers

# Bounded, event-based waits only. No sleep, no elapsed-time assertion: these
# tests must not depend on how busy the runner is.
_POLL_WAIT_S = 5.0


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _agent(provider="custom", model="m", **overrides):
    import run_agent

    kwargs = dict(
        model=model,
        provider=provider,
        api_key="sk-dummy",
        base_url="http://127.0.0.1:1/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        max_iterations=1,
    )
    kwargs.update(overrides)
    return run_agent.AIAgent(**kwargs)


def _call(agent, *, total_timeout=None, age_s=0.0, chunks_flowing=True):
    """One streaming request with the budget resolved and the clock injected.

    ``age_s`` back-dates the request start (the injected wall clock) and
    ``chunks_flowing`` keeps ``last_chunk_time`` fresh, so the only thing that
    can trip the budget is elapsed wall time since the request began.
    """
    call = helpers._StreamingCall(agent, {"model": agent.model, "messages": []}, None)
    call._call_done = threading.Event()
    call._monitor_interrupted = {"yes": False}
    call._stream_stale_timeout = 3600.0  # idle detector must not be what fires
    call._resolve_total_timeout()
    if total_timeout is not None:
        call._stream_total_timeout = total_timeout
    call._request_started_at = time.monotonic() - age_s
    # Production stamps the attempt clock at request start (then per attempt).
    call._attempt_started_at = call._request_started_at
    if chunks_flowing:
        call.last_chunk_time["t"] = time.time()
    return call


def _run_monitor(call):
    """Drive the real monitor loop until the request is done; report whether the
    expiry action was taken."""
    expired = threading.Event()
    call._kill_over_budget_stream = lambda elapsed: expired.set()

    def _target():
        call._resolve_stale_timeout = lambda: None
        call._monitor_loop()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    try:
        return expired.wait(timeout=_POLL_WAIT_S)
    finally:
        call._call_done.set()
        worker.join(timeout=_POLL_WAIT_S)


def test_budget_ends_a_stream_that_is_still_progressing(tmp_path, monkeypatch):
    """The invariant: total elapsed wall clock governs the request, so a stream
    that keeps delivering chunks is ended once it is past its budget -- and the
    same request, inside its budget, is left running.

    Configured through ``config.yaml`` (not an env var, not a hardcoded default)
    and resolved by the production resolver, so the whole chain is exercised.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, """\
providers:
  custom:
    stream_total_timeout_seconds: 60
    models:
      m:
        stream_total_timeout_seconds: 30
""")
    import importlib

    from hermes_cli import timeouts as to_mod

    importlib.reload(to_mod)

    agent = _agent()
    assert to_mod.get_provider_stream_total_timeout("custom", "m") == 30.0, (
        "per-model override must win over the provider-wide budget"
    )

    # Past the budget while chunks keep arriving -> ended.
    past = _call(agent, age_s=600.0)
    assert past._stream_total_timeout == 30.0, "the configured budget did not reach the request"
    assert _run_monitor(past) is True, (
        "a request past its wall-clock budget kept running while chunks were still "
        "arriving; no total budget governs a progressing stream"
    )

    # Inside the budget, same stream shape -> untouched.
    inside = _call(agent, age_s=5.0)
    assert _run_monitor(inside) is False, (
        "a request inside its budget was ended early"
    )

    # Unset budget: nothing governs the request, so a long-progressing stream runs.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, "providers:\n  custom: {}\n")
    importlib.reload(to_mod)
    unbounded = _call(agent)
    assert unbounded._stream_total_timeout is None, "an unset budget must stay unset (opt-in)"
    assert _run_monitor(unbounded) is False, (
        "a stream was ended with no budget configured"
    )


def test_expiry_does_not_poison_the_stale_breaker(tmp_path, monkeypatch):
    """Safety relation: the kill cancels the attempt, but must not count toward
    ``_consecutive_stale_streams``.

    That streak drives ``_check_stale_giveup``, which stops streaming entirely
    after repeated unresponsive attempts. A provider trickling tokens IS
    responsive; charging its budget expiry against the breaker would take a
    healthy provider offline.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, """\
providers:
  custom:
    stream_total_timeout_seconds: 60
""")
    import importlib

    from hermes_cli import timeouts as to_mod

    importlib.reload(to_mod)

    agent = _agent()
    agent._consecutive_stale_streams = 2
    cancelled = []
    call = _call(agent, age_s=600.0)
    monkeypatch.setattr(
        call, "_cancel_current_stream_attempt",
        lambda reason: cancelled.append(reason), raising=False)
    monkeypatch.setattr(call, "clients", types_namespace(diag={}), raising=False)

    call._kill_over_budget_stream(600.0)

    assert cancelled, "expiry must cancel the in-flight attempt, or the reader never unwinds"
    assert agent._consecutive_stale_streams == 2, (
        "budget expiry was charged to the stale breaker; a provider still streaming "
        "tokens is responsive and must stay streamable"
    )


def types_namespace(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


def test_a_retry_after_a_budget_kill_is_still_bounded(tmp_path, monkeypatch):
    """The deadline must SPAN the ladder's retry, not be latched off after one kill.

    Regression for a one-shot flag that made ``_total_budget_spent`` return ``None``
    for the rest of the request once the budget had fired: the transient-error ladder's
    retry then had NO wall-clock ceiling at all, so a retry that trickled re-entered the
    exact indefinite-block case this budget exists to close. The absolute deadline now
    governs the retry too; the only concession is a bounded per-attempt runway, so the
    0.3s monitor poll cannot re-kill the retry the instant it reopens (a tight loop).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, """\
providers:
  custom:
    stream_total_timeout_seconds: 30
""")
    import importlib

    from hermes_cli import timeouts as to_mod

    importlib.reload(to_mod)

    agent = _agent()
    call = _call(agent, age_s=600.0)
    # The REAL kill (the monitor's own action), so the retry starts from exactly the
    # state that kill leaves behind rather than from a stubbed one.
    monkeypatch.setattr(call, "_cancel_current_stream_attempt", lambda reason: None, raising=False)
    monkeypatch.setattr(call, "clients", types_namespace(diag={}), raising=False)
    call._kill_over_budget_stream(600.0)

    # The ladder's retry: a NEW attempt on a request whose deadline is already spent.
    # Chunks keep flowing, so only the total budget can end it.
    call._call_done = threading.Event()
    call._start_stream_attempt()
    call._request_started_at = time.monotonic() - 600.0
    call.last_chunk_time["t"] = time.time()
    # Just reopened: the deadline is still spent, but the fresh attempt runway must
    # keep the retry from being killed on its first poll (a tight immediate loop).
    assert call._total_budget_spent(time.monotonic()) is None, (
        "the retry was killed on its first poll after reopening -- a tight immediate "
        "retry loop instead of a bounded runway"
    )

    # Runway spent: the SAME absolute deadline must end the retry too, so a retry that
    # trickles is bounded instead of blocking the turn indefinitely.
    call._attempt_started_at -= 600.0
    assert call._total_budget_spent(time.monotonic()) is not None, (
        "the retry after a budget kill had no wall-clock ceiling; a retry that "
        "trickles blocks the turn indefinitely"
    )
    call._call_done = threading.Event()
    assert _run_monitor(call) is True, (
        "the retry's spent deadline never reached the monitor, so the ladder's retry "
        "is not actually bounded by the configured budget"
    )
