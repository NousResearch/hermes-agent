"""Regression test for the small-prompt + high-effort watchdog kill.

Before this fix: on a small prompt (< 10K tokens) with reasoning_effort=high
or xhigh, the Codex stream idle watchdog killed the connection at the 12s
small-bucket default — well before GPT-5.5+ finished server-side thinking
(30-90s typical for xhigh). This surfaced as:

    [Errno 32] Broken pipe

After this fix: when reasoning_config['effort'] is high or xhigh, the idle
and TTFB watchdog budgets are multiplied by HERMES_CODEX_HIGH_EFFORT_MULTIPLIER
(default 5x), giving the server time to emit its first SSE event.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch, reasoning_effort=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="copilot",
        api_key="sk-dummy",
        base_url="https://api.githubcopilot.com",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    if reasoning_effort:
        agent.reasoning_config = {"effort": reasoning_effort}
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _capture_idle_and_ttfb(agent, monkeypatch, api_kwargs):
    """Drive interruptible_api_call partway and capture the computed
    _codex_idle_timeout and _ttfb_timeout values WITHOUT actually firing
    a network request. The simplest way is to monkeypatch
    _create_request_openai_client to raise immediately and then introspect
    the env-derived defaults via direct calculation.

    Since the watchdog values are computed inline (not factored), we test
    by calling the helper with a tiny api_kwargs (under 10K tokens) AND
    mocking the inner call to assert via captured kwargs. The cleanest
    proof is to monkeypatch _env_float to capture every call and inspect
    what the effort branch did to the defaults.
    """
    captured = {}
    from agent import chat_completion_helpers as h

    orig_env_float = h._env_float

    def _spy_env_float(key, default):
        v = orig_env_float(key, default)
        captured[key] = v
        return v

    monkeypatch.setattr(h, "_env_float", _spy_env_float)
    # Force the call to bail early so we don't need to mock the SDK
    monkeypatch.setattr(
        agent, "_create_request_openai_client",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("test-bail")),
    )
    try:
        h.interruptible_api_call(agent, api_kwargs)
    except Exception:
        pass
    return captured


def test_high_effort_small_prompt_extends_idle_timeout(tmp_path, monkeypatch):
    """With effort=high on a tiny prompt (< 10K tokens), the idle watchdog
    must NOT use the 12s default — the multiplier should kick in."""
    # Use the public default multiplier (5x)
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)

    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort="high")
    # Tiny api_kwargs that estimate to well under 10K tokens
    tiny_kwargs = {"model": "gpt-5.5", "input": "hi"}

    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    # The HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS lookup uses
    # _codex_idle_timeout_default as its fallback. For a tiny prompt that
    # default was 12.0; with effort=high (5x multiplier), it should be 60.0.
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None, f"watchdog timeout was never computed: {captured}"
    assert idle >= 60.0, (
        f"expected idle timeout >= 60s for effort=high on small prompt, "
        f"got {idle}s. The 5x effort multiplier did not apply."
    )


def test_xhigh_effort_small_prompt_extends_idle_timeout(tmp_path, monkeypatch):
    """Same for effort=xhigh — Hermes normalizes xhigh→high for Copilot
    but the multiplier branch checks both values to be safe."""
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort="xhigh")
    tiny_kwargs = {"model": "gpt-5.5", "input": "hi"}
    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None
    assert idle >= 60.0, (
        f"expected idle timeout >= 60s for effort=xhigh on small prompt, got {idle}s"
    )


def test_default_effort_small_prompt_keeps_12s(tmp_path, monkeypatch):
    """No effort set → no multiplier → keep the legacy 12s small-prompt
    default. This pins that we don't accidentally inflate timeouts for
    default-effort calls."""
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort=None)
    tiny_kwargs = {"model": "gpt-5.5", "input": "hi"}
    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None
    assert idle == 12.0, (
        f"expected legacy 12s idle for default-effort small prompt, got {idle}s"
    )


def test_payload_high_effort_extends_idle_timeout_without_agent_attr(tmp_path, monkeypatch):
    """The wire payload is the source of truth for stream effort.

    Some call paths populate api_kwargs['reasoning'] without setting
    agent.reasoning_config. Those still need the high-effort watchdog budget.
    """
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort=None)
    tiny_kwargs = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "high"},
    }

    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None
    assert idle >= 60.0, (
        f"expected payload effort=high to extend idle timeout, got {idle}s"
    )


def test_payload_medium_effort_extends_idle_timeout_without_agent_attr(tmp_path, monkeypatch):
    """Medium effort is still non-low and needs more than the 12s bucket."""
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort=None)
    tiny_kwargs = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "medium"},
    }

    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None
    assert idle >= 42.0, (
        f"expected payload effort=medium to extend idle timeout, got {idle}s"
    )


def test_medium_effort_multiplier_one_does_not_shrink_stream_timeout(tmp_path, monkeypatch):
    """Disabling scaling must not make the stream watchdog more aggressive."""
    monkeypatch.setenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", "1.0")
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort=None)
    tiny_kwargs = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "medium"},
    }

    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle == 12.0


def test_payload_effort_wins_over_agent_attr_for_stream_timeout(tmp_path, monkeypatch):
    """Do not inflate stream watchdogs from stale agent-level reasoning_config.

    api_kwargs is what will actually be sent over the wire, so a low-effort
    payload must keep the small-prompt 12s budget even if agent state says high.
    """
    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort="high")
    tiny_kwargs = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "low"},
    }

    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle is not None
    assert idle == 12.0, (
        f"expected payload effort=low to keep legacy idle timeout, got {idle}s"
    )


def test_multiplier_env_var_can_override(tmp_path, monkeypatch):
    """HERMES_CODEX_HIGH_EFFORT_MULTIPLIER=10 → 12s × 10 = 120s on tiny+xhigh."""
    monkeypatch.setenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", "10.0")
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort="xhigh")
    tiny_kwargs = {"model": "gpt-5.5", "input": "hi"}
    captured = _capture_idle_and_ttfb(agent, monkeypatch, tiny_kwargs)
    idle = captured.get("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert idle == 120.0, f"expected 120s with 10x multiplier, got {idle}s"


def test_ttfb_also_extended_by_multiplier(tmp_path, monkeypatch):
    """The TTFB watchdog (separate from idle) must also be multiplied —
    the first SSE event on xhigh can take >120s. With 5x default → 600s."""
    from agent import chat_completion_helpers as h

    monkeypatch.delenv("HERMES_CODEX_HIGH_EFFORT_MULTIPLIER", raising=False)
    monkeypatch.delenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", "0")  # never disable
    agent = _make_codex_agent(tmp_path, monkeypatch, reasoning_effort="xhigh")

    scaled_calls = []
    original = h._reasoning_effort_scaled_timeout

    def _spy_scaled_timeout(timeout, api_payload, scaled_agent, env_name):
        scaled = original(timeout, api_payload, scaled_agent, env_name)
        scaled_calls.append((timeout, scaled, env_name))
        return scaled

    monkeypatch.setattr(h, "_reasoning_effort_scaled_timeout", _spy_scaled_timeout)
    _capture_idle_and_ttfb(agent, monkeypatch, {"model": "gpt-5.5", "input": "hi"})

    assert (120.0, 600.0, "HERMES_CODEX_HIGH_EFFORT_MULTIPLIER") in scaled_calls
