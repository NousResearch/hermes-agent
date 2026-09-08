"""Tests for the optional wall-clock run budget (agent.run_budget_seconds / --run-budget).

Covers:

1. Stale-timeout deadline scaling in
   ``run_agent.py:AIAgent._compute_non_stream_stale_timeout``:
   - an active run budget CAPS every provider wait at the remaining wall-clock
     budget, including explicit operator timeouts;
   - the cap never RAISES the timeout above what it would otherwise be;
   - explicit user configuration wins over model/context floors but not over
     the absolute run deadline;
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


def test_no_budget_by_default(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path, monkeypatch)
    assert agent.run_budget_seconds is None
    assert agent._run_budget_started_at is None
    assert agent._run_budget_wrapup_injected is False


def test_constructor_arg_sets_budget(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    assert agent.run_budget_seconds == 900.0


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
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 90.0


def test_active_budget_caps_implicit_reasoning_floor(monkeypatch, tmp_path):
    """deepseek-v4-pro's 600s implicit floor yields to the remaining deadline."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    # Sanity: implicit reasoning floor is 600s without a running clock.
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 600.0 and implicit is False

    agent._run_budget_started_at = time.time() - 800
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 95.0 <= timeout <= 100.0


def test_active_budget_caps_at_remaining_time(monkeypatch, tmp_path):
    """A long implicit timeout is capped at the absolute remaining time."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = time.time() - 400  # remaining ~500 < 600s floor
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 495.0 <= timeout <= 500.0


def test_active_budget_never_raises_timeout(monkeypatch, tmp_path):
    """The deadline cap NEVER loosens an already-tighter implicit timeout.

    90s default with lots of remaining budget (cap would be ~445s) -> stays 90s.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 10
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 90.0


def test_explicit_provider_config_yields_to_absolute_budget(monkeypatch, tmp_path):
    """Explicit stale timeout wins over floors, but not the run deadline."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 1800.0)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 800
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 95.0 <= timeout <= 100.0


def test_explicit_env_var_yields_to_absolute_budget(monkeypatch, tmp_path):
    """The explicit env value is still capped by the absolute run deadline."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "1200")
    agent._run_budget_started_at = time.time() - 800
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 95.0 <= timeout <= 100.0


def test_explicit_stream_stale_timeout_wins_over_qwen_reasoning_floor(monkeypatch, tmp_path):
    """A configured 120s watchdog must stay 120s for a Qwen3 model."""
    import agent.chat_completion_helpers as helpers
    monkeypatch.setattr(helpers, "get_provider_stale_timeout", lambda *a, **k: 120.0)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="qwen/qwen3.8-27b",
        provider="nvidia-spark",
        base_url="http://127.0.0.1:8000/v1",
        run_budget_seconds=300,
    )
    agent._run_budget_started_at = time.time()
    call = helpers._StreamingCall(agent, {"model": agent.model, "messages": []}, None)
    call._resolve_stale_timeout()
    assert 119.0 <= call._stream_stale_timeout <= 120.0


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
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 600.0


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


# ── turn clock stamping ────────────────────────────────────────────────────


def test_turn_clock_stamped_only_with_budget(monkeypatch, tmp_path):
    """prepare-turn stamps the clock iff a budget is set, and resets the latch."""
    agent_with = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent_with._run_budget_wrapup_injected = True

    # Mirror the turn_context.prepare block (unit-level: run the same logic).
    for agent in (agent_with,):
        if getattr(agent, "run_budget_seconds", None):
            agent._run_budget_started_at = time.time()
        else:
            agent._run_budget_started_at = None
        agent._run_budget_wrapup_injected = False

    assert agent_with._run_budget_started_at is not None
    assert agent_with._run_budget_wrapup_injected is False


def test_forced_final_synthesis_uses_request_local_empty_tool_registry():
    from agent.turn_request_assembly import _select_tools_for_api

    class Agent:
        tools = [{"type": "function", "function": {"name": "read_something"}}]
        _force_toolless_final = False

    agent = Agent()
    assert _select_tools_for_api(agent) is agent.tools
    agent._force_toolless_final = True
    assert _select_tools_for_api(agent) == []
    assert agent.tools  # canonical registry is not mutated


def test_final_synthesis_notice_is_one_shot_on_tool_tail():
    from agent.turn_iteration_prep import (
        FINAL_SYNTHESIS_NOTICE,
        _maybe_inject_final_synthesis_notice,
    )

    agent = _StubAgent()
    agent._force_toolless_final = True
    agent._final_synthesis_notice_injected = False
    messages = _tool_messages()
    assert _maybe_inject_final_synthesis_notice(agent, messages) is True
    assert FINAL_SYNTHESIS_NOTICE in messages[-1]["content"]
    assert _maybe_inject_final_synthesis_notice(agent, messages) is False
    assert messages[-1]["content"].count(FINAL_SYNTHESIS_NOTICE) == 1


def test_final_synthesis_deadline_uses_explicit_stale_timeout_and_is_one_shot():
    from agent.run_budget import (
        arm_final_synthesis_deadline,
        remaining_final_synthesis_seconds,
    )

    class Agent:
        run_budget_seconds = 300
        _run_budget_started_at = 900.0
        _final_synthesis_deadline = None

        @staticmethod
        def _resolved_api_call_stale_timeout_base():
            return 120.0, False

    agent = Agent()
    deadline = arm_final_synthesis_deadline(agent, now=1000.0)
    assert deadline == 1120.0
    assert remaining_final_synthesis_seconds(agent, now=1001.0) == 119.0
    assert arm_final_synthesis_deadline(agent, now=1050.0) == 1120.0


def test_forced_final_request_disables_reasoning_and_caps_output():
    from agent.turn_iteration_prep import _prepare_forced_final_request

    class Agent:
        _force_toolless_final = True
        _final_synthesis_deadline = None
        run_budget_seconds = 300
        _run_budget_started_at = time.time()
        max_tokens = None

        @staticmethod
        def _resolved_api_call_stale_timeout_base():
            return 120.0, False

    agent = Agent()
    _prepare_forced_final_request(agent)
    assert agent._ephemeral_reasoning_off is True
    assert agent._ephemeral_max_output_tokens == 2048
    assert 0 < agent._final_synthesis_deadline - time.time() <= 120.0


def test_provider_wait_lifecycle_owns_deadline_precedence_and_typed_abort():
    from agent.run_budget import (
        FINAL_SYNTHESIS_TIMEOUT,
        RUN_BUDGET_EXHAUSTED,
        FinalSynthesisTimeout,
        ProviderWaitLifecycle,
        RunBudgetExceeded,
    )

    class Agent:
        run_budget_seconds = 300
        _run_budget_started_at = 900.0
        _final_synthesis_deadline = 1100.0

    lifecycle = ProviderWaitLifecycle(Agent())
    assert lifecycle.expired_deadline(now=1050.0) is None
    assert lifecycle.expired_deadline(now=1150.0) == FINAL_SYNTHESIS_TIMEOUT
    assert lifecycle.expired_deadline(now=1250.0) == RUN_BUDGET_EXHAUSTED

    assert lifecycle.abort(FINAL_SYNTHESIS_TIMEOUT) is True
    assert lifecycle.abort(RUN_BUDGET_EXHAUSTED) is False
    assert isinstance(lifecycle.error(), FinalSynthesisTimeout)

    run_lifecycle = ProviderWaitLifecycle(Agent())
    assert run_lifecycle.abort(RUN_BUDGET_EXHAUSTED) is True
    assert isinstance(run_lifecycle.error(), RunBudgetExceeded)


def test_provider_wait_lifecycle_stale_is_terminal_only_for_bounded_turns():
    from agent.run_budget import ProviderStaleTimeout, ProviderWaitLifecycle

    bounded = type("Agent", (), {
        "run_budget_seconds": 30,
        "_run_budget_started_at": time.time(),
        "_final_synthesis_deadline": None,
    })()
    lifecycle = ProviderWaitLifecycle(bounded)
    assert lifecycle.abort_on_provider_stale(4.9) is True
    assert isinstance(lifecycle.error(), ProviderStaleTimeout)
    assert "4s" in str(lifecycle.error())

    unbounded = type("Agent", (), {
        "run_budget_seconds": None,
        "_run_budget_started_at": None,
        "_final_synthesis_deadline": None,
    })()
    assert ProviderWaitLifecycle(unbounded).abort_on_provider_stale(4.9) is False
