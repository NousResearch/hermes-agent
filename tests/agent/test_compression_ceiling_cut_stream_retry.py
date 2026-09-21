"""A summary whose total ceiling expired while it was still producing output gets one reasoning-off retry — #107516.

A thinking summariser on a slow route (one local model serving both the conversation and the summary) can still
be producing reasoning tokens when ``compression.context_total_ceiling_seconds`` expires. Before #107516 the
attempt was discarded ("continuing without compression") and the next automatic attempt re-ran the same summary
on the same route; the deterministic fallback only engaged after the 2nd or 3rd ceiling. Now, when
``auxiliary.compression`` sets no reasoning control of its own, the same attempt's ladder is: the configured
``fallback_chain`` entry, then the same route once with reasoning switched off (one inactivity budget), then the
deterministic rung. An explicit reasoning setting, the over-window path and an idle stall keep the old ladder.

Synchronisation: the stubbed summary call keeps "streaming" (ticking the fence progress hook) until the HOST has
explicitly cancelled its fence — which it only does once it has taken the ceiling path — so no assertion depends
on how fast a loaded box schedules the waiting thread.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.conversation_compression as cc
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import SUMMARY_PREFIX, take_pinned_summary_route
from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout
from hermes_state import SessionDB

REASONING_OFF = {"enabled": False}
CHAIN_ENTRY = {
    "provider": "custom", "model": "backup-summarizer", "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback",
}
# Backstop only: a correct run leaves every stubbed stream within a few seconds.
_BACKSTOP_SECONDS = 20.0


def _make_agent(tmp_path, tag):
    db = SessionDB(db_path=Path(tmp_path) / f"state-{tag}.db")
    session_id = f"CEILING_CUT_{tag}"
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


def _summary_rows(messages):
    return [m for m in messages if isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX)]


def _ok_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _stream_until_host_cancels(fence, tick) -> bool:
    """Keep producing output until the host explicitly cancels ``fence`` (it only does after taking its
    timeout path). ``False`` when the backstop expired instead — callers turn that into a failure."""
    backstop = time.monotonic() + _BACKSTOP_SECONDS
    while not fence._cancelled and time.monotonic() < backstop:
        tick()
        time.sleep(0.01)
    return bool(fence._cancelled)


def _summary_call(agent, calls, backstops, *, reasoning_off_answers=True, chain_answers=False):
    """Stubbed ``call_llm``: the primary route streams until the ceiling cuts it; the reasoning-off rung answers
    at once unless told otherwise; a pinned chain entry answers only when ``chain_answers``."""
    from agent import auxiliary_client as aux

    def _call(**kwargs):
        if "provider" in kwargs:
            calls.append("chain")
            if chain_answers:
                return _ok_response("CHAIN SUMMARY")
        elif kwargs.get("reasoning_config") == REASONING_OFF:
            calls.append("reasoning-off")
            if reasoning_off_answers:
                return _ok_response("REASONING-OFF SUMMARY")
        else:
            calls.append("primary")
        hook = getattr(aux._aux_progress, "hook", None)
        if not _stream_until_host_cancels(
            agent._active_compression_commit_fence, hook if callable(hook) else (lambda: None),
        ):
            backstops.append(calls[-1])
        # What the aux stream consumer raises once the host has stopped waiting.
        raise AuxiliaryExplicitCancellation()

    return _call


@pytest.fixture
def fast_timeouts(monkeypatch):
    # (idle, ceiling): the ceiling cuts the streaming primary; the idle window bounds the reasoning-off rung. The
    # stubs tick every 10 ms, so the idle watchdog can only pre-empt the ceiling if a thread is starved for longer
    # than the idle window — which the ladder check in ``_compress`` turns into a failure, never a vacuous pass.
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda compression_cfg=None: (2.0, 3.0))


def _compress(agent, task_config, **stub_kwargs):
    """Run one real compaction; returns ``(live, out, calls)``. Also proves which host branch ran: the ladder was
    entered exactly once, from the ceiling-with-output branch (``retry_without_reasoning=True`` is offered BEFORE
    the configuration gate inside ``_stall_retry_routes``), and no stubbed stream outlived its host."""
    live = _transcript()
    calls, backstops, offered = [], [], []
    real_ladder = cc._retry_compression_on_fallback_chain

    def _ladder_spy(**kwargs):
        offered.append(kwargs.get("retry_without_reasoning"))
        return real_ladder(**kwargs)

    with patch("agent.context_compressor.call_llm", side_effect=_summary_call(agent, calls, backstops, **stub_kwargs)), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config), \
            patch.object(cc, "_retry_compression_on_fallback_chain", side_effect=_ladder_spy):
        out, _ = agent._compress_context(live, "sys", approx_tokens=50_000)
    assert backstops == [], f"stubbed stream(s) {backstops} hit the backstop: the host never cancelled them"
    assert offered == [True], (
        f"expected the host's ceiling-with-output branch exactly once, got ladder calls {offered}: this run "
        "exercised another path (e.g. an idle timeout after thread starvation) and proves nothing"
    )
    return live, out, calls


def _log_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "agent.conversation_compression"]


def test_reasoning_off_retry_keeps_an_llm_summary_when_the_ceiling_cuts_a_producing_summary(
    tmp_path, fast_timeouts, caplog,
):
    """Real AIAgent + real ``compress_context`` + facade timeout wrap; only ``call_llm`` is stubbed. Before
    #107516 this attempt handed the transcript back unchanged and the next turn re-ran the same summary."""
    agent = _make_agent(tmp_path, "A")
    assert agent.context_compressor._consecutive_timeout_failures == 0, "precondition: no prior stall"
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")
    live, out, calls = _compress(agent, {"fallback_chain": []})

    assert calls == ["primary", "reasoning-off"]
    assert out is not live and len(out) < len(live), "the attempt converged instead of deferring to the next turn"
    rows = _summary_rows(out)
    assert len(rows) == 1 and "REASONING-OFF SUMMARY" in rows[0]["content"], "the handoff is the LLM summary"
    assert agent.context_compressor._fallback_compression_streak == 0, "an LLM summary is not a fallback commit"
    assert getattr(agent, "_last_compression_timed_out", None) is not True
    messages = _log_messages(caplog)
    assert any("retrying once on the same summary route without reasoning" in m for m in messages)
    assert any("recovered on same summary route without reasoning" in m for m in messages)


def test_deterministic_rung_follows_when_the_reasoning_off_retry_is_cut_too(tmp_path, fast_timeouts, caplog):
    """A route that ignores the disable is cut after the rung's inactivity window; the same attempt then commits
    the deterministic fallback summary instead of continuing uncompressed."""
    agent = _make_agent(tmp_path, "B")
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")
    live, out, calls = _compress(agent, {"fallback_chain": []}, reasoning_off_answers=False)

    assert calls == ["primary", "reasoning-off"], "the deterministic rung makes no summary LLM call"
    assert out is not live and len(out) < len(live)
    rows = _summary_rows(out)
    assert len(rows) == 1, "exactly one handoff row"
    assert "REASONING-OFF SUMMARY" not in rows[0]["content"], "the handoff is the deterministic one"
    assert agent.context_compressor._fallback_compression_streak == 1
    assert getattr(agent, "_last_compression_timed_out", None) is not True
    assert any(
        "stalled on every summary route — committing the deterministic fallback summary" in m
        for m in _log_messages(caplog)
    )


def test_configured_fallback_chain_runs_before_the_reasoning_off_retry(tmp_path, fast_timeouts):
    """The operator's ``fallback_chain`` keeps its priority: a healthy chain entry recovers the attempt and the
    reasoning-off retry never runs (a failing reasoning-off call must not pre-empt it with a static summary)."""
    agent = _make_agent(tmp_path, "C")
    live, out, calls = _compress(agent, {"fallback_chain": [CHAIN_ENTRY]}, chain_answers=True)

    assert calls == ["primary", "chain"]
    assert out is not live
    rows = _summary_rows(out)
    assert len(rows) == 1 and "CHAIN SUMMARY" in rows[0]["content"]


@pytest.mark.parametrize(
    "task_config",
    [
        {"reasoning_effort": "none", "fallback_chain": []},
        {"reasoning_effort": "high", "fallback_chain": []},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "fallback_chain": []},
    ],
    ids=["reasoning_effort-none", "reasoning_effort-high", "vendor-extra_body"],
)
def test_explicit_compression_reasoning_setting_keeps_the_previous_ladder(tmp_path, fast_timeouts, task_config):
    """An operator's own reasoning control for compression is never overridden: no reasoning-off retry and no new
    escalation — the first ceiling still continues uncompressed and records its backoff, as before #107516.
    ``_compress`` proves the host offered the retry, so it is the configuration gate that withheld it."""
    agent = _make_agent(tmp_path, "D")
    live, out, calls = _compress(agent, task_config)

    assert calls == ["primary"]
    assert out is live, "unchanged behaviour: the first ceiling continues without compression"
    assert agent.context_compressor._consecutive_timeout_failures >= 1


def test_fence_level_ladder_order_and_budgets_after_a_producing_ceiling_cut():
    """Primary (full ceiling) → same route without reasoning (one inactivity budget, not a second ceiling) →
    deterministic rung, in one host call and without ``on_timeout``. Budgets are read from the fences the host
    arms, not from wall-clock timing."""
    original = [{"role": "user", "content": "keep-me"}]
    marker = ([{"role": "assistant", "content": "deterministic"}], "deterministic-prompt")
    kinds = []
    budgets = []
    arm = CompressionCommitFence.set_total_ceiling_seconds

    def _recording_arm(self, seconds):
        budgets.append(float(seconds))
        return arm(self, seconds)

    def _kind(pin):
        if pin is None:
            return "primary"
        if pin.get("deterministic") is True:
            return "deterministic"
        if pin.get("reasoning_config") == REASONING_OFF and not pin.get("model"):
            return "reasoning-off"
        return "other"

    backstops = []

    def worker(fence):
        kind = _kind(take_pinned_summary_route())
        kinds.append(kind)
        if kind == "deterministic":
            return marker
        if not _stream_until_host_cancels(fence, fence.touch_progress):
            backstops.append(kind)
        return original, "aborted"

    timeouts = []
    with patch.object(CompressionCommitFence, "set_total_ceiling_seconds", _recording_arm), \
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}):
        result = run_compress_context_with_progress_timeout(
            worker=worker, messages=original, system_prompt_fallback="degraded-prompt",
            idle_timeout_seconds=1.5, total_ceiling_seconds=3.0, on_timeout=lambda *a: timeouts.append(a),
        )

    assert backstops == [], f"worker(s) {backstops} hit the backstop: the host never cancelled them"
    assert kinds == ["primary", "reasoning-off", "deterministic"]
    assert budgets[:2] == [3.0, 1.5], "primary keeps the ceiling; the reasoning-off rung gets one inactivity budget"
    assert result == marker
    assert timeouts == [], "a ladder that committed never reports a timeout"
