"""Tests for the optional wall-clock run budget (agent.run_budget_seconds / --run-budget).

Covers:

1. Stale-timeout deadline scaling in
   ``run_agent.py:AIAgent._compute_non_stream_stale_timeout``:
    - an active run budget caps request watchdogs, including local and explicit
      stale timeouts, at the remaining wall-clock deadline;
   - the cap never RAISES the timeout above what it would otherwise be;
   - explicit user configuration (provider ``stale_timeout_seconds`` or the
     ``HERMES_API_CALL_STALE_TIMEOUT`` env var) always wins untouched;
   - no budget => completely unchanged behavior.

2. One-time-ness of the 80% wrap-up notice injection in
   ``agent.conversation_loop._maybe_inject_run_budget_wrapup``:
   - fires once (latched), not repeatedly;
   - never fires when no budget is set or before the 80% threshold;
   - appended to the newest tool message (cache-safe /steer channel), no
     synthetic user message.

3. Normalization of the config/CLI value
   (``agent.agent_init._normalize_run_budget_seconds``): dormant on
   null/invalid/non-positive input.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")

def _make_agent(tmp_path, monkeypatch, config_body: str = "", **overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STREAM_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, config_body)

    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai",
        api_key="sk-dummy",
        base_url="https://api.openai.com/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)

# ── normalization ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (None, None),
    (0, None),
    (-5, None),
    ("abc", None),
    (True, None),   # YAML `true` must not become a 1-second budget
    (False, None),
    (float("nan"), None),
    (900, 900.0),
    ("850", 850.0),
    (0.5, 0.5),
])
def test_normalize_run_budget_seconds(raw, expected):
    from agent.agent_init import _normalize_run_budget_seconds
    assert _normalize_run_budget_seconds(raw) == expected

# ── constructor / config plumbing ─────────────────────────────────────────

def test_config_key_sets_budget(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path, monkeypatch,
        config_body="agent:\n  run_budget_seconds: 750\n",
    )
    assert agent.run_budget_seconds == 750.0

def test_constructor_arg_wins_over_config(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path, monkeypatch,
        config_body="agent:\n  run_budget_seconds: 750\n",
        run_budget_seconds=900,
    )
    assert agent.run_budget_seconds == 900.0

# ── stale-timeout deadline scaling ─────────────────────────────────────────

def test_no_budget_stale_timeout_unchanged(monkeypatch, tmp_path):
    """Without a run budget the implicit 90s default is untouched."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch)
    base, _implicit = agent._resolved_api_call_stale_timeout_base()
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == base

def test_active_budget_caps_implicit_reasoning_floor(monkeypatch, tmp_path):
    """deepseek-v4-pro's 600s implicit floor yields to a tighter deadline cap.

    900s budget, 800s elapsed -> remaining 100s, so the watchdog cannot outlive it.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) <= 100.0

def test_active_budget_cap_never_exceeds_remaining(monkeypatch, tmp_path):
    """The stale detector cannot be configured beyond the remaining run deadline."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = time.time() - 100  # remaining ~800
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert timeout <= 800.0

def test_active_budget_never_raises_timeout(monkeypatch, tmp_path):
    """The deadline cap NEVER loosens an already-tighter implicit timeout.

    90s default with lots of remaining budget (cap would be ~445s) -> stays 90s.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    base, _implicit = agent._resolved_api_call_stale_timeout_base()
    agent._run_budget_started_at = time.time() - 10
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == base

def test_run_budget_caps_explicit_provider_timeout(monkeypatch, tmp_path):
    """The outer run deadline also bounds a longer profile stale timeout."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 1800.0)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) <= 100.0

def test_run_budget_caps_explicit_env_timeout(monkeypatch, tmp_path):
    """The outer run deadline also bounds an explicit API stale timeout."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "1200")
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) <= 100.0

def test_budget_without_started_clock_is_inert(monkeypatch, tmp_path):
    """A configured budget with no running turn clock changes nothing."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = None
    base, _implicit = agent._resolved_api_call_stale_timeout_base()
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == base


@pytest.mark.parametrize(
    "config_body,env_value,expected",
    [
        ("", None, 100.0),
        ("providers:\n  openai:\n    stale_timeout_seconds: 1200\n", None, 100.0),
        ("", "900", 100.0),
    ],
)
def test_cloud_stream_budget_preserves_explicit_deadlines(
    monkeypatch, tmp_path, config_body, env_value, expected
):
    """Both stream owners cap implicit patience without changing explicit precedence."""
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    if env_value is not None:
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", env_value)
    monkeypatch.setattr(helpers.time, "time", lambda: 1000.0)
    agent._run_budget_started_at = 1000.0 - 800
    payload = {"model": agent.model, "messages": [{"role": "user", "content": "hi"}]}
    call = helpers._StreamingCall(agent, payload, None)
    call._resolve_stale_timeout()
    assert call._stream_stale_timeout == expected


def test_local_stream_stale_timeout_obeys_run_budget(monkeypatch, tmp_path):
    """The local model prefill grace cannot outlive the run deadline."""
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(
        tmp_path,
        monkeypatch,
        base_url="http://127.0.0.1:11434/v1",
        run_budget_seconds=900,
    )
    call = helpers._StreamingCall(agent, {"model": agent.model}, None)
    call._resolve_stale_timeout()
    without_clock = call._stream_stale_timeout
    agent._run_budget_started_at = time.time() - 800
    call._resolve_stale_timeout()
    assert without_clock == 900.0
    assert call._stream_stale_timeout <= 100.0


def test_interruptible_backoff_stops_at_run_deadline(monkeypatch):
    """Provider backoff is cut at the same deadline while keeping the normal interrupt path."""
    from types import SimpleNamespace
    from agent import turn_recovery

    class Clock:
        now = 100.0

        def time(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(turn_recovery.time, "time", clock.time)
    monkeypatch.setattr(turn_recovery.time, "sleep", clock.sleep)
    agent = SimpleNamespace(
        run_budget_seconds=3.0,
        _run_budget_started_at=100.0,
        _interrupt_requested=False,
        _adopt_nous_key_before_expiry=lambda: None,
    )
    result = turn_recovery.interruptible_backoff_sleep(
        agent, 120.0, None, messages=[], conversation_history=[], api_call_count=1,
        abort_message="", interrupt_text="", activity_label="test",
    )
    assert result is None
    assert 102.99 <= clock.now <= 103.000001


def test_expired_budget_prevents_another_provider_attempt(monkeypatch):
    from types import SimpleNamespace
    from agent import conversation_loop, run_budget

    monkeypatch.setattr(run_budget, "run_budget_expired", lambda _agent: True)
    agent = SimpleNamespace(run_budget_seconds=3.0)
    loop_state = SimpleNamespace(retry_count=0, max_retries=3, run_budget_exhausted=False)

    assert conversation_loop._run_api_retry_loop(agent, loop_state) is None
    assert loop_state.run_budget_exhausted is True


def test_stream_monitor_cuts_off_at_absolute_run_deadline(monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    from agent import chat_completion_stream_monitor, run_budget
    from agent.chat_completion_helpers import _StreamingCall

    call = object.__new__(_StreamingCall)
    call._call_done = Event()
    call.agent = SimpleNamespace(base_url=None)
    call.api_kwargs = {"model": "test"}
    call.last_chunk_time = {"t": time.time()}  # activity is fresh; stale logic must not fire
    call._stream_stale_timeout = 300.0
    killed = []
    call._kill_for_run_budget = lambda: killed.append(True)
    monkeypatch.setattr(run_budget, "run_budget_expired", lambda _agent: True)

    chat_completion_stream_monitor.StreamingWaitMonitor._monitor_loop(call)

    assert killed == [True]


def test_expired_run_budget_skips_iteration_summary(monkeypatch):
    from types import SimpleNamespace
    from agent import run_budget, turn_finalizer

    monkeypatch.setattr(run_budget, "run_budget_expired", lambda _agent: True)
    agent = SimpleNamespace(
        max_iterations=2,
        iteration_budget=SimpleNamespace(remaining=0),
        run_budget_seconds=3.0,
        quiet_mode=True,
    )
    agent._emit_diagnostic_status = lambda *_args: None
    agent._handle_max_iterations = lambda *_args: pytest.fail("summary request exceeded the run budget")

    final, reason, _preserved, interrupted = turn_finalizer._resolve_budget_fallback(
        agent, final_response=None, api_call_count=2, interrupted=False, failed=False,
        messages=[], _turn_exit_reason="unknown", _pending_verification_response=None,
        _pending_verification_response_previewed=False, logger=None,
    )
    assert reason == "run_budget_exhausted"
    assert "budget" in final.lower()
    assert interrupted is False


# ── wrap-up injection one-time-ness ────────────────────────────────────────

class _StubAgent:
    def __init__(self, budget=None, started=None):
        self.run_budget_seconds = budget
        self._run_budget_started_at = started
        self._run_budget_wrapup_injected = False

def _tool_messages():
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result one"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "result two"},
    ]

def test_wrapup_not_injected_when_unset():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=None, started=time.time() - 10_000)
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert messages == _tool_messages()

def test_wrapup_not_injected_before_threshold():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=900, started=time.time() - 100)  # 11% elapsed
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert agent._run_budget_wrapup_injected is False

def test_wrapup_injected_once_after_threshold():
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)  # 89% elapsed
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    assert agent._run_budget_wrapup_injected is True
    # Appended to the NEWEST tool message; earlier messages untouched.
    assert RUN_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]
    assert messages[-1]["content"].startswith("result two")
    assert messages[2]["content"] == "result one"
    # No synthetic user message was inserted.
    assert [m["role"] for m in messages] == [
        "user", "assistant", "tool", "assistant", "tool",
    ]

    # Second call: latched, no re-injection.
    snapshot = [dict(m) for m in messages]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert messages == snapshot
    assert messages[-1]["content"].count(RUN_BUDGET_WRAPUP_NOTICE) == 1

def test_wrapup_retries_when_no_tool_message_yet():
    """First iteration (no tool results) can't inject; the latch stays open
    so the next iteration with a tool result delivers the notice."""
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)
    messages = [{"role": "user", "content": "do the task"}]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert agent._run_budget_wrapup_injected is False

    messages += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    assert RUN_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]

def test_wrapup_not_injected_without_turn_clock():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=900, started=None)
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False

def test_wrapup_skips_already_persisted_tool_row():
    """An already-flushed tool row is append-only; mutating it would diverge the
    replayed transcript from the live wire bytes and break the prompt cache (same
    contract as _maybe_inject_iteration_budget_warning)."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup

    agent = _StubAgent(budget=900, started=time.time() - 800)
    messages = _tool_messages()
    messages[-1][_DB_PERSISTED_MARKER] = True
    snapshot = [dict(m) for m in messages]

    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert agent._run_budget_wrapup_injected is False
    assert messages == snapshot

def test_wrapup_multimodal_tool_content():
    """Content-blocks tool results get a text block appended, not clobbered."""
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1",
         "content": [{"type": "text", "text": "block"}]},
    ]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    blocks = messages[-1]["content"]
    assert blocks[0] == {"type": "text", "text": "block"}
    assert blocks[-1] == {"type": "text", "text": RUN_BUDGET_WRAPUP_NOTICE}

# ── pre-flush wiring (the notice must reach durable bytes) ─────────────────

def test_wrapup_lands_in_the_persisted_row_via_pre_flush_hook(monkeypatch, tmp_path):
    """The wrap-up notice must be injected BEFORE the tool row is flushed, or it
    never reaches SQLite: prepare_iteration's own call always runs on an
    already-persisted row (the previous iteration's flush already ran)."""
    from hermes_state import SessionDB
    from agent.tool_executor import _flush_session_db_after_tool_progress
    from agent.conversation_loop import RUN_BUDGET_WRAPUP_NOTICE

    session_db = SessionDB(db_path=tmp_path / "proof.db")
    try:
        agent = _make_agent(tmp_path, monkeypatch, session_db=session_db, run_budget_seconds=900)
        agent._run_budget_started_at = time.time() - 800  # 89% elapsed
        messages = _tool_messages()

        assert _flush_session_db_after_tool_progress(agent, messages, stage="checkpoint")

        persisted = session_db.get_messages(agent.session_id)
        assert any(RUN_BUDGET_WRAPUP_NOTICE in str(row["content"]) for row in persisted)
        assert agent._run_budget_wrapup_injected is True
    finally:
        session_db.close()

# ── turn clock stamping ────────────────────────────────────────────────────

def test_gemini_long_decode_scales_implicit_stale_timeout_by_hops(monkeypatch, tmp_path):
    """One non-streaming stale timer spans every unary long-decode hop: an xhigh budget (262,144 tokens,
    8 hops of 32,768) gets 8x the 600s gemini-4 floor, not 600s for the whole answer. Explicit config
    still wins untouched, and a request without a long-decode budget is unchanged."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, model="gemini-4-argon", provider="gemini", api_key="AQ.dummy",
                        base_url="https://generativelanguage.googleapis.com/v1beta")
    payload = {"messages": [{"role": "user", "content": "hi"}], "model": "gemini-4-argon"}
    assert agent._compute_non_stream_stale_timeout(payload) == 600.0
    long_decode = {**payload, "extra_body": {"max_output_tokens": 262_144}}
    assert agent._compute_non_stream_stale_timeout(long_decode) == 600.0 * 8
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "120")
    assert agent._compute_non_stream_stale_timeout(long_decode) == 120.0
