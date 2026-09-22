"""A repeated summary stall escalates to the deterministic fallback summary — #112420.

After one stall the host records a stall-class backoff (``_consecutive_timeout_failures`` >= 1). When the
next attempt stalls again, "continuing without compression" would re-enter the same silent route every
turn, so the stall retry ladder ends with a deterministic rung: the worker is re-run with the summary LLM
skipped and compress() commits its static fallback summary through the ordinary lease/fence pipeline.
A first stall keeps today's behaviour (backoff, LLM retry after it lapses) unless the request is already
above the model's context window — then nothing can be sent unchanged and the rung runs at once (#114594). A pinned fallback route whose
summary call fails still commits under the default ``abort_on_summary_failure=false`` — that must not be
logged as a recovery (#112387 review caveat).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.conversation_compression as cc
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import SUMMARY_PREFIX, pin_summary_route
from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout
from hermes_state import SessionDB

CHAIN_ENTRY = {
    "provider": "custom", "model": "backup-summarizer", "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback", "timeout": 0.4,
}


def _make_agent(tmp_path, tag):
    db = SessionDB(db_path=Path(tmp_path) / f"state-{tag}.db")
    session_id = f"STALL_DETERMINISTIC_{tag}"
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
    agent.context_compressor.threshold_tokens = 1_000
    return agent


def _transcript():
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + ("lorem ipsum " * 300)}
        for i in range(40)
    ]


def _stalling_call_llm(compressor, calls, *, fail_when_pinned=False):
    """Summary call that never streams: hangs until the host cancels the fence, like a held-open socket."""
    def _call(**kwargs):
        calls.append(kwargs.get("provider") or "primary")
        if fail_when_pinned and "provider" in kwargs:
            raise RuntimeError("fallback route exploded")
        cancelled = getattr(compressor, "_compression_cancelled_check", None)
        # The host cancels on its idle watchdog (a fraction of a second here). This deadline is only a
        # backstop so a bug cannot hang the suite; polling coarsely keeps the spin off a loaded box.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not (callable(cancelled) and cancelled()):
            time.sleep(0.01)
        raise AuxiliaryExplicitCancellation()

    return _call


def _summary_rows(messages):
    return [m for m in messages if isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX)]


@pytest.fixture
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda compression_cfg=None: (0.4, 4.0))


def test_second_consecutive_stall_commits_the_deterministic_fallback_summary(tmp_path, fast_timeouts):
    """Real AIAgent + real ``compress_context`` + facade timeout wrap; only ``call_llm`` is stubbed.
    Stall 1: backoff recorded, transcript untouched. Stall 2 (after the backoff lapsed): the deterministic
    rung commits a fallback summary instead of continuing without compression."""
    agent = _make_agent(tmp_path, "A")
    compressor = agent.context_compressor
    calls = []
    live = _transcript()
    with patch("agent.context_compressor.call_llm", side_effect=_stalling_call_llm(compressor, calls)), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
        first, _ = agent._compress_context(live, "sys", approx_tokens=50_000)
        assert first is live, "a first stall keeps the transcript and only arms the backoff"
        assert compressor._consecutive_timeout_failures >= 1
        assert compressor._summary_failure_cooldown_until > time.monotonic()
        assert calls == ["primary"], "no deterministic rung on the FIRST stall: the LLM route gets its backoff retry"

        # The backoff lapses; the still-oversized context re-triggers compression (the reporter's next turn).
        # The host arms this cooldown synchronously, but the fence-cancelled worker keeps unwinding after
        # ``_compress_context`` has returned and may re-arm it from its own thread — a legitimate late write,
        # not a result under test. Clear and retry until an attempt actually runs rather than racing it, so
        # the assertions below depend only on the synchronous arm.
        second = live
        for _ in range(5):
            # Re-assert the state this phase tests — one stall already on the record (line above) — and
            # clear the lapsed backoff. Pinning the counter also stops a retry from ACCUMULATING stall
            # history, which would let a regression that makes escalation harder (e.g. a threshold of 3)
            # satisfy itself on a later iteration and pass. A refused, or fleetingly degraded, attempt
            # stays retryable; a regression cannot buy itself green.
            compressor._consecutive_timeout_failures = 1
            compressor._summary_failure_cooldown_until = 0.0
            compressor._session_db.clear_compression_failure_cooldown(compressor._session_id)
            calls.clear()
            second, _ = agent._compress_context(live, "sys", approx_tokens=50_000)
            if second is not live:
                break
            time.sleep(0.05)

    assert second is not live and len(second) < len(live)
    assert len(_summary_rows(second)) == 1, "the deterministic fallback summary is committed as the handoff"
    assert calls == ["primary"], "the deterministic rung makes no summary LLM call"
    assert getattr(agent, "_last_compression_timed_out", None) is not True
    # The stalled LLM route stays in its backoff even though the deterministic rung committed. This arm
    # comes from the cancelled PRIMARY worker's `stall_interrupted` record (the deterministic retry path
    # never reaches `on_timeout`), which normally lands while the retry above runs — hence the read last,
    # after that work, rather than immediately after the clear.
    assert compressor._summary_failure_cooldown_until > time.monotonic(), (
        "the stalled LLM route keeps its stall backoff after the deterministic commit"
    )


def test_failing_pinned_fallback_route_is_not_logged_as_recovered(tmp_path, fast_timeouts, caplog):
    """Primary stalls, the fallback_chain route raises: compress() still commits its static fallback
    summary (abort_on_summary_failure=false), and the host log must say so instead of 'recovered'."""
    agent = _make_agent(tmp_path, "B")
    compressor = agent.context_compressor
    calls = []
    live = _transcript()
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")
    with patch(
        "agent.context_compressor.call_llm", side_effect=_stalling_call_llm(compressor, calls, fail_when_pinned=True),
    ), patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [CHAIN_ENTRY]}):
        out, _ = agent._compress_context(live, "sys", approx_tokens=50_000)

    assert calls == ["primary", "custom"]
    assert out is not live and len(_summary_rows(out)) == 1
    records = [r for r in caplog.records if r.name == "agent.conversation_compression"]
    assert not any("recovered on fallback_chain[0]" in r.getMessage() for r in records)
    assert any(
        "committed a deterministic fallback summary on fallback_chain[0]" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in records
    )


def test_deterministic_pin_is_consumed_and_a_real_route_is_left_alone():
    from agent.context_compressor import (
        DETERMINISTIC_SUMMARY_ROUTE, take_deterministic_summary_pin, take_pinned_summary_route,
    )

    with pin_summary_route(dict(CHAIN_ENTRY)):
        assert take_deterministic_summary_pin() is False
        assert take_pinned_summary_route()["model"] == "backup-summarizer"
    with pin_summary_route(dict(DETERMINISTIC_SUMMARY_ROUTE)):
        assert take_deterministic_summary_pin() is True
        assert take_deterministic_summary_pin() is False, "single use"
    assert take_deterministic_summary_pin() is False


def test_fence_level_retry_ladder_is_unchanged_without_a_prior_timeout():
    """No agent-level timeout history (fence-level callers, fresh compressors): a stall with no chain still
    degrades in one attempt — the deterministic rung is escalation, not the default."""
    attempts = []
    release = threading.Event()

    def worker(fence: CompressionCommitFence):
        attempts.append(fence)
        # Block until the test releases it. The fence must give up on its idle window while the worker is
        # demonstrably still running; a fixed sleep instead races the idle timer under load.
        release.wait(10.0)
        return ([{"role": "assistant", "content": "late"}], "late-prompt")

    original = [{"role": "user", "content": "keep-me"}]
    try:
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker, messages=original, system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05, total_ceiling_seconds=2.0, on_timeout=lambda *a: None,
            )
        assert msgs is original and prompt == "degraded-prompt"
        assert len(attempts) == 1
    finally:
        release.set()


def test_over_window_request_commits_the_deterministic_fallback_on_the_first_stall(tmp_path, fast_timeouts):
    """A request above the model's context window cannot be sent unchanged, so waiting for a SECOND stall
    (which on the messaging gateway never comes — the first one auto-reset the session) is a dead end:
    the deterministic rung runs on the first stall and the transcript is committed (#114594)."""
    agent = _make_agent(tmp_path, "C")
    compressor = agent.context_compressor
    compressor.context_length = 64_000
    calls = []
    live = _transcript()
    with patch("agent.context_compressor.call_llm", side_effect=_stalling_call_llm(compressor, calls)), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
        out, _ = agent._compress_context(live, "sys", approx_tokens=70_000)

    assert calls == ["primary"], "the deterministic rung makes no second summary LLM call"
    assert out is not live and len(out) < len(live)
    assert len(_summary_rows(out)) == 1, "the over-window request got a committed deterministic handoff"
    assert getattr(agent, "_last_compression_timed_out", None) is not True
