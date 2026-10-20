"""Compression falls back after an aborted (stalled) summary — #78981.

A summariser that keeps the connection open but never emits a real token
produces no fence progress, so the host's progress-aware timeout aborts the
worker and returns "continue without compression". Nothing raises out of the
auxiliary client on that path, so its configured ``fallback_chain`` — the
user's declared answer to "this route is unhealthy" — was never consulted for
the one failure mode that most needs it.

These tests pin the contract:

* an aborted stall re-attempts compression once with the summary route pinned
  to the configured ``auxiliary.compression.fallback_chain``;
* the pinned route reaches the summary ``call_llm`` (provider/model/base_url/
  api_key/timeout), and is single-use so the compressor's own main-model retry
  does not re-issue the same failed route;
* the historical "continue without compression" degrade survives when no chain
  is configured or the fallback attempt also stalls.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    pin_summary_route,
    take_pinned_summary_route,
)
from agent.conversation_compression import (
    CompressionCommitFence,
    _run_pinned_compression_retry,
    resolve_compression_fallback_route,
    run_compress_context_with_progress_timeout,
)

CHAIN_ENTRY = {
    "provider": "custom",
    "model": "backup-summarizer",
    "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback",
    "timeout": 45,
}


def _patch_chain(chain):
    """Pin auxiliary.compression config without touching the real config.yaml."""
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"fallback_chain": chain},
    )


class _StalledSummaryWorker:
    """A compression worker whose first attempt streams nothing at all.

    Mirrors the reported shape: the provider holds the connection open, so the
    worker never calls ``fence.touch_progress()`` and the host's idle budget
    lapses. ``stall_attempts`` controls how many attempts hang; any later
    attempt commits a real summary.
    """

    def __init__(self, compressed, *, stall_attempts=1):
        self.compressed = compressed
        self.stall_attempts = stall_attempts
        self.routes = []
        self.fences = []
        self._lock = threading.Lock()
        self.release = threading.Event()

    @property
    def attempts(self):
        return len(self.routes)

    def __call__(self, fence: CompressionCommitFence):
        with self._lock:
            self.routes.append(take_pinned_summary_route())
            self.fences.append(fence)
            attempt = len(self.routes)
        if attempt <= self.stall_attempts:
            # Connection open, zero tokens, zero fence progress.
            self.release.wait(timeout=10)
            return ([{"role": "assistant", "content": "late"}], "late-prompt")
        if not fence.begin_commit():
            return ([{"role": "assistant", "content": "cancelled"}], "cancelled")
        try:
            return (self.compressed, "summarized-prompt")
        finally:
            fence.finish_commit()


def _run(worker, *, chain, timeouts, messages, idle=2.0, ceiling=5.0):
    with _patch_chain(chain):
        return run_compress_context_with_progress_timeout(
            worker=worker,
            messages=messages,
            system_prompt_fallback="degraded-prompt",
            idle_timeout_seconds=idle,
            total_ceiling_seconds=ceiling,
            on_timeout=lambda *args: timeouts.append(args),
        )


# ---------------------------------------------------------------------------
# Fence-level contract: an aborted stall consults the configured chain
# ---------------------------------------------------------------------------


def test_stalled_summary_attempts_configured_fallback_chain():
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    worker = _StalledSummaryWorker(compressed)
    timeouts = []

    try:
        msgs, prompt = _run(
            worker, chain=[CHAIN_ENTRY], timeouts=timeouts, messages=original
        )
    finally:
        worker.release.set()

    assert worker.attempts == 2, "the aborted stall must be retried once"
    assert worker.routes[0] is None, "the primary attempt is never pinned"
    pinned = worker.routes[1]
    assert pinned is not None, "the retry must carry the configured fallback route"
    assert pinned["provider"] == "custom"
    assert pinned["model"] == "backup-summarizer"
    assert msgs == compressed, "the fallback attempt's compression must be published"
    assert prompt == "summarized-prompt"
    assert not timeouts, "no continue-without-compression degrade after a recovery"


def test_retry_runs_on_a_host_published_fence():
    """The aborted fence vetoes every future commit, so the retry needs a new
    one — minted through the host so ``/stop`` admits against the attempt that
    is actually running."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary"}]
    worker = _StalledSummaryWorker(compressed)
    minted = []

    def _new_fence():
        fence = CompressionCommitFence()
        minted.append(fence)
        return fence

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, _prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=2.0,
                total_ceiling_seconds=5.0,
                new_fence=_new_fence,
            )
    finally:
        worker.release.set()

    assert msgs == compressed
    assert len(minted) == 1, "exactly one fence is minted for the one retry"
    assert worker.fences[1] is minted[0]
    assert worker.fences[1] is not worker.fences[0]
    assert worker.fences[0].is_cancelled, "the aborted attempt stays cancelled"


def test_hard_interrupt_suppresses_the_fallback_attempt():
    """An explicit stop is not an unhealthy route — don't start another
    summary on the user's behalf after they asked for the turn to end."""
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    stopped = threading.Event()
    stopped.set()
    agent = SimpleNamespace(_hard_interrupt_requested=stopped)
    timeouts = []

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=2.0,
                total_ceiling_seconds=5.0,
                on_timeout=lambda *args: timeouts.append(args),
                telemetry_agent=agent,
            )
    finally:
        worker.release.set()

    assert worker.attempts == 1
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_no_fallback_chain_configured_degrades_without_retry():
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    timeouts = []

    try:
        msgs, prompt = _run(worker, chain=[], timeouts=timeouts, messages=original)
    finally:
        worker.release.set()

    assert worker.attempts == 1, "nothing to fall back to — do not burn a retry"
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_fallback_that_also_stalls_degrades_after_one_attempt():
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker(
        [{"role": "user", "content": "unused"}], stall_attempts=2
    )
    timeouts = []
    entry = dict(CHAIN_ENTRY, timeout=2.0)

    try:
        msgs, prompt = _run(worker, chain=[entry], timeouts=timeouts, messages=original)
    finally:
        worker.release.set()

    assert worker.attempts == 2, "the fallback is attempted once, not in a loop"
    # The primary leaves the fallback the entry's 2.0s timeout; the fallback ends by the pass's own deadline.
    assert abs(worker.fences[1].deadline_monotonic - (worker.fences[0].deadline_monotonic + 2.0)) < 1e-3, (
        "the fallback must not mint a second total ceiling"
    )
    assert msgs is original, "no messages may be dropped when both routes stall"
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1, "the degrade must be reported exactly once"


def test_fallback_runs_inside_the_pass_ceiling():
    """A primary that kept streaming used the whole ceiling and its fallback then got another, so one
    pass could hold the turn for two ceilings. The fallback's window is reserved inside the one ceiling."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    deadlines = []
    stop = threading.Event()

    def worker(fence):
        deadlines.append(fence.deadline_monotonic)
        if take_pinned_summary_route() is None:
            # A slow summary that streams until the host stops waiting for it, then takes 0.2s to unwind.
            while not fence.deadline_exceeded and not stop.wait(0.01):
                fence.touch_progress()
            stop.wait(0.2)
            return original, ""
        assert fence.begin_commit()
        fence.finish_commit()
        return compressed, "summarized-prompt"

    started = time.monotonic()
    try:
        msgs, _prompt = _run(worker, chain=[dict(CHAIN_ENTRY, timeout=0.5)], timeouts=[], messages=original,
                             idle=1.0, ceiling=2.0)
    finally:
        stop.set()

    assert msgs == compressed
    primary_deadline, fallback_deadline = deadlines
    assert primary_deadline - started <= 1.5 + 0.1, "the primary leaves the fallback its 0.5s"
    # The primary's unwind came out of the fallback's 0.5s, not on top of it.
    assert fallback_deadline - primary_deadline <= 0.5 + 0.02, "the fallback ends by the pass ceiling"


def test_deterministic_summary_runs_inside_the_pass_ceiling():
    """After the primary and the configured fallback, the deterministic summary's pre-commit work (callbacks,
    pruning) also ran on a fresh window past the pass. It now shares what is left of the one ceiling."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "deterministic summary"}]
    deadlines = []
    stop = threading.Event()

    def worker(fence):
        deadlines.append(fence.deadline_monotonic)
        route = take_pinned_summary_route()
        if route is None:
            while not fence.deadline_exceeded and not stop.wait(0.01):
                fence.touch_progress()
            return original, ""
        if not route.get("deterministic"):
            stop.wait(0.2)
            return original, ""  # the fallback model spent 0.2s and produced nothing
        stop.wait(0.1)  # pre-commit work before the commit is admitted
        assert fence.begin_commit()
        fence.finish_commit()
        return compressed, "summarized-prompt"

    started = time.monotonic()
    try:
        with _patch_chain([dict(CHAIN_ENTRY, timeout=0.5)]):
            msgs, _prompt = run_compress_context_with_progress_timeout(
                worker=worker, messages=original, system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=1.0, total_ceiling_seconds=5.0, request_exceeds_window=True,
            )
    finally:
        stop.set()

    assert msgs == compressed
    assert len(deadlines) == 3
    # An over-window pass is one idle window (1.0s) end to end: primary, fallback and deterministic summary.
    assert max(deadlines) - started <= 1.0 + 0.05
    assert time.monotonic() - started <= 1.0 + 0.15


def test_fallback_with_no_time_left_in_the_pass_is_skipped(caplog):
    """A primary whose teardown outlasted the reserved window leaves the fallback nothing: skip it plainly,
    without minting a fence or starting a summary it has no time to finish."""
    minted, ran = [], []
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        recovered = _run_pinned_compression_retry(
            dict(CHAIN_ENTRY, label="fallback_chain[0](custom)"), worker=ran.append, messages=[],
            system_prompt_fallback="degraded-prompt", idle_timeout_seconds=1.0, total_ceiling_seconds=-0.2,
            on_commit_overrun=None, on_timeout_cause=None, telemetry_agent=None, new_fence=lambda: minted.append(1),
        )

    assert recovered is None
    assert not minted and not ran
    assert "no time left for fallback_chain[0](custom)" in caplog.text


class _ReadThenStallWorker(_StalledSummaryWorker):
    """The primary attempt's server reads the prompt (pushing the fence deadline back) past the pass's own
    deadline, then stalls."""

    def __call__(self, fence: CompressionCommitFence):
        if not self.routes:
            assert fence.extend_deadline(10.0)
            reading_until = time.monotonic() + 2.5
            while time.monotonic() < reading_until:
                fence.touch_progress()
                self.release.wait(0.05)
        return super().__call__(fence)


def test_stall_after_prompt_reading_still_funds_the_fallback():
    """Prompt reading moves the fence deadline; the fallback must be budgeted from the moved deadline,
    not the primary attempt's original one, or a long read followed by a stall skips the fallback."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary"}]
    worker = _ReadThenStallWorker(compressed)
    timeouts = []

    try:
        msgs, prompt = _run(
            # Without the extension the 2.5s read runs past the pass's 2s and leaves the fallback nothing.
            worker, chain=[dict(CHAIN_ENTRY, timeout=2.0)], timeouts=timeouts, messages=original,
            idle=2.0, ceiling=2.0,
        )
    finally:
        worker.release.set()

    assert worker.attempts == 2, "the extended budget must fund the fallback attempt"
    assert msgs == compressed
    assert prompt == "summarized-prompt"
    assert not timeouts


def test_same_turn_fallback_retry_is_not_gated_by_the_primary_stall_backoff(tmp_path, monkeypatch):
    """The cancelled primary worker persists ``stall_interrupted`` while the fallback retry is already
    running; that cooldown must not no-op the retry (#112387). Real AIAgent, real ``compress_context``
    and facade timeout wrap; only the summary route (``compress``) is stubbed: the primary route stalls
    silently until the host cancels its fence, the pinned fallback route returns a real compressed list."""
    import os
    import time
    from pathlib import Path

    import agent.conversation_compression as cc
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from hermes_state import SessionDB

    session_id = "STALL_FALLBACK_SAME_TURN"
    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id=session_id, skip_context_files=True, skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    compressor = agent.context_compressor
    compressor.threshold_tokens = 1_000
    original = [{"role": "user", "content": f"m{i} " + "x" * 400} for i in range(20)]
    live = list(original)
    routes = []

    def fake_compress(messages, **kwargs):
        route = take_pinned_summary_route()
        routes.append((route["label"] if route else None, kwargs.get("bypass_cooldown", False)))
        if route is None:
            cancelled = getattr(compressor, "_compression_cancelled_check", None)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (callable(cancelled) and cancelled()):
                time.sleep(0.0005)
            raise AuxiliaryExplicitCancellation()
        return [{"role": "user", "content": "summary of earlier turns"}, original[-1]]

    compressor.compress = fake_compress
    real_route = cc.resolve_compression_fallback_route

    def route_after_primary_unwound():
        # Pin the race outcome from the report: the primary persisted its backoff before the retry ran.
        # The lookup before the primary started (its fallback reservation) has nothing to wait for.
        deadline = time.monotonic() + 3
        while routes and time.monotonic() < deadline and compressor._summary_failure_cooldown_until <= time.monotonic():
            time.sleep(0.001)
        return real_route()

    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda compression_cfg=None: (0.4, 4.0))
    monkeypatch.setattr(cc, "resolve_compression_fallback_route", route_after_primary_unwound)
    with _patch_chain([CHAIN_ENTRY]):
        out_msgs, _prompt = agent._compress_context(live, "sys", approx_tokens=50_000)

    assert [r[0] for r in routes] == [None, "fallback_chain[0](custom)"]
    assert routes[1][1] is True, "the same-turn retry must bypass the cooldown the primary just armed"
    assert len(out_msgs) == 2 and out_msgs is not live
    assert getattr(agent, "_last_compression_timed_out", None) is not True


# ---------------------------------------------------------------------------
# Route resolution: a chain entry becomes an explicit summary route
# ---------------------------------------------------------------------------


def test_resolved_route_carries_entry_credentials_and_timeout():
    with _patch_chain([CHAIN_ENTRY]):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["provider"] == "custom"
    assert route["model"] == "backup-summarizer"
    assert route["base_url"] == "https://fallback.invalid/v1"
    assert route["api_key"] == "sk-fallback"
    # Per-entry timeouts already govern aux-client fallback candidates
    # (#62452); the stall retry honours the same declaration.
    assert route["timeout"] == 45.0


def test_incomplete_chain_entries_are_skipped():
    chain = [
        "not-a-mapping",
        {"model": "orphan-model"},          # no provider
        {"provider": "custom"},             # no model
        CHAIN_ENTRY,
    ]
    with _patch_chain(chain):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["model"] == "backup-summarizer"


def test_no_chain_resolves_to_no_route():
    with _patch_chain([]):
        assert resolve_compression_fallback_route() is None


# ---------------------------------------------------------------------------
# Injection point: the pinned route reaches the summary call
# ---------------------------------------------------------------------------


def _make_compressor(summary_model="aux-summarizer"):
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(
            model="main-model",
            quiet_mode=True,
            summary_model_override=summary_model,
        )


def _msgs():
    return [
        {"role": "user", "content": "u1 " + "x" * 200},
        {"role": "assistant", "content": "a1 " + "y" * 200},
        {"role": "user", "content": "u2 " + "z" * 200},
    ]


def _ok_response(content="SUMMARY BODY"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_pinned_route_overrides_the_summary_call_route():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 1
    call = calls[0]
    assert call["task"] == "compression"
    assert call["provider"] == "custom"
    assert call["model"] == "backup-summarizer"
    assert call["base_url"] == "https://fallback.invalid/v1"
    assert call["api_key"] == "sk-fallback"
    assert call["timeout"] == 45


def test_pinned_route_is_not_reissued_by_the_main_model_retry():
    """The compressor's own main-model retry must not re-run the failed route."""
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("Request timed out.")
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 2
    assert calls[0]["provider"] == "custom"
    assert "provider" not in calls[1], (
        "the retry must route normally, not repeat the stalled fallback route"
    )


def test_unpinned_summary_call_keeps_task_routing():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        summary = compressor._generate_summary(_msgs())

    assert summary
    assert calls and "provider" not in calls[0]
    assert calls[0]["model"] == "aux-summarizer"
