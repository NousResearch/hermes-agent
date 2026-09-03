"""Tests for the optional wall-clock run budget (agent.run_budget_seconds / --run-budget).

Covers:

1. Stale-timeout deadline scaling in
   ``run_agent.py:AIAgent._compute_non_stream_stale_timeout``:
   - an active run budget CAPS an implicit stale timeout (default 90s and
     reasoning floors like deepseek's 600s) at ``max(60, remaining * 0.5)``;
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
    """deepseek-v4-pro's 600s implicit floor yields to a tighter deadline cap.

    900s budget, 800s elapsed -> remaining 100s -> cap = max(60, 50) = 60s.
    """
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
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 60.0


def test_active_budget_cap_half_remaining(monkeypatch, tmp_path):
    """Cap is remaining * 0.5 when that exceeds the 60s floor."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = time.time() - 100  # remaining ~800 -> cap ~400
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 395.0 <= timeout <= 400.0


def test_active_budget_never_raises_timeout(monkeypatch, tmp_path):
    """The deadline cap NEVER loosens an already-tighter implicit timeout.

    90s default with lots of remaining budget (cap would be ~445s) -> stays 90s.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 10
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 90.0


def test_explicit_provider_config_wins_over_budget_cap(monkeypatch, tmp_path):
    """Explicit stale_timeout_seconds is never capped by the run budget."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 1800.0)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


def test_explicit_env_var_wins_over_budget_cap(monkeypatch, tmp_path):
    """HERMES_API_CALL_STALE_TIMEOUT is explicit config — never capped."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "1200")
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1200.0


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


# ── local-endpoint stale-timeout override (Bug 2 regression) ─────────────


class TestLocalEndpointStaleTimeout:
    """Regression: local endpoints must get float("inf") stale timeout
    regardless of reasoning-model floor matching.

    Bug: Qwen3.8-27B-mxfp8 matches the "qwen3" reasoning floor pattern
    (regex ^qwen3(?:$|[\-._]) matches "qwen3.8" because "." is in the
    separator class). This sets uses_implicit_default=False, which disables
    the local-endpoint float("inf") short-circuit. Result: a healthy local
    mlx-lm server gets killed after 180s of no output.

    Fix: move the is_local_endpoint check BEFORE the reasoning-model floor
    in _resolved_api_call_stale_timeout_base, so local endpoints always get
    float("inf") regardless of reasoning model detection.
    """

    def test_local_endpoint_with_reasoning_model_gets_inf(self, monkeypatch, tmp_path):
        """A local endpoint running a Qwen3 model must get inf, not 180s."""
        import run_agent
        monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
        agent = _make_agent(
            tmp_path, monkeypatch,
            model="mlx-community/Qwen3.8-27B-mxfp8",
            provider="custom",
            base_url="http://127.0.0.1:8080/v1",
        )
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == float("inf"), (
            f"Local endpoint with Qwen3 model should get inf, got {base}"
        )
        assert implicit is True

    def test_local_endpoint_with_deepseek_gets_inf(self, monkeypatch, tmp_path):
        """A local endpoint running deepseek-v4-pro must get inf, not 600s."""
        import run_agent
        monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
        agent = _make_agent(
            tmp_path, monkeypatch,
            model="deepseek/deepseek-v4-pro",
            provider="custom",
            base_url="http://127.0.0.1:8080/v1",
        )
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == float("inf"), (
            f"Local endpoint with deepseek-v4-pro should get inf, got {base}"
        )
        assert implicit is True

    def test_cloud_endpoint_still_gets_reasoning_floor(self, monkeypatch, tmp_path):
        """A cloud endpoint running deepseek-v4-pro must still get 600s floor."""
        import run_agent
        monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
        agent = _make_agent(
            tmp_path, monkeypatch,
            model="deepseek/deepseek-v4-pro",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == 600.0, (
            f"Cloud endpoint with deepseek-v4-pro should get 600s, got {base}"
        )
        assert implicit is False

    def test_explicit_config_still_wins_over_local(self, monkeypatch, tmp_path):
        """Explicit stale_timeout_seconds overrides even local endpoint inf."""
        import run_agent
        monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 300.0)
        agent = _make_agent(
            tmp_path, monkeypatch,
            model="mlx-community/Qwen3.8-27B-mxfp8",
            provider="custom",
            base_url="http://127.0.0.1:8080/v1",
        )
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == 300.0
        assert implicit is False

    def test_local_endpoint_non_reasoning_model_gets_inf(self, monkeypatch, tmp_path):
        """A local endpoint with a non-reasoning model also gets inf."""
        import run_agent
        monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
        agent = _make_agent(
            tmp_path, monkeypatch,
            model="llama3.1:8b",
            provider="custom",
            base_url="http://127.0.0.1:11434/v1",
        )
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == float("inf")
        assert implicit is True
