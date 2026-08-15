"""Regression tests for the reasoning-model stale-timeout floor (issue #52217).

Reasoning models (Nemotron 3 Ultra, OpenAI o1/o3, Anthropic Opus 4.x
thinking, DeepSeek R1, Qwen QwQ, xAI Grok reasoning) routinely exceed
the 180s / 90s chat-model stale-timeout defaults during their
thinking phase.  Hermes's default cloud-stream stale detector
(``HERMES_STREAM_STALE_TIMEOUT`` = 180s) and non-stream detector
(``HERMES_API_CALL_STALE_TIMEOUT`` = 90s) both fire before the
upstream proxy's idle timeout on a healthy reasoning stream.  Result:
the user sees ``API call failed after 3 retries: [Errno 32] Broken
pipe`` for every Nemotron 3 Ultra turn.

These tests pin the floor's behavior:

1. ``get_reasoning_stale_timeout_floor`` returns the right floor for
   every key in the allowlist, ``None`` for every negative case
   (gpt-4o, olmo-1, etc.), and longest-substring-first wins on
   shared prefixes (``o3-mini-`` > ``o3-``).
2. The non-stream resolver at
   ``run_agent.py:AIAgent._resolved_api_call_stale_timeout_base``
   consults the floor at priority 4 (after explicit user config,
   provider config, and env var; before the 90s default), and
   keeps it implicit so the local-endpoint short-circuit in
   ``_compute_non_stream_stale_timeout`` remains active. Explicitly configured
   local timeouts and cloud reasoning-model floors remain finite.
3. The stream stale-timeout resolution (mirrored here as in
   ``test_stream_read_timeout_floor.py`` because the real builder
   lives inside a worker thread) consults the floor after the
   context-size scaling block, raising the timeout for reasoning
   models without lowering it for non-reasoning models.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── pure-function resolver ────────────────────────────────────────────────


@pytest.mark.parametrize("model,expected", [
    # NVIDIA Nemotron reasoning family (longest keys first).
    ("nvidia/nemotron-3.5-lightning-30b-a3b", 300.0),
    ("nvidia/nemotron-3-ultra-550b-a55b", 600.0),
    ("nvidia/nemotron-3-super-120b-a12b", 600.0),
    ("nvidia/nemotron-3-nano-30b-a3b", 300.0),
    # DeepSeek R1 + DeepSeek reasoner + V4 reasoning series.
    # V4 emits reasoning_content in a separate delta before final
    # content (same shape as R1), so it needs the same 600s floor.
    ("deepseek/deepseek-r1", 600.0),
    ("deepseek/deepseek-r1-distill-llama-70b", 600.0),
    ("deepseek/deepseek-reasoner", 600.0),
    ("deepseek/deepseek-v4-flash", 600.0),
    ("deepseek/deepseek-v4-pro", 600.0),
    ("deepseek-v4-flash-free", 600.0),   # catalog -free variant inherits via separator anchor
    # Qwen QwQ + Qwen3 thinking variants (qwen3 family entry matches all).
    ("qwen/qwq-32b-preview", 300.0),
    ("qwen/qwen3-235b-a22b-thinking", 180.0),
    ("qwen/qwen3-32b", 180.0),
    # OpenAI o-series — each variant enumerated explicitly.
    # Longest match wins (o3-mini beats o3 on shared prefix).
    ("openai/o1", 600.0),
    ("openai/o1-mini", 600.0),
    ("openai/o1-pro", 600.0),
    ("openai/o1-preview", 600.0),
    ("openai/o3", 600.0),
    ("openai/o3-pro", 600.0),
    ("openai/o3-mini", 300.0),
    ("openai/o4-mini", 300.0),
    # Anthropic Claude 4.x thinking variants.
    ("anthropic/claude-opus-4-6", 240.0),
    ("anthropic/claude-opus-4-20250514", 240.0),
    ("anthropic/claude-sonnet-4.5", 180.0),
    ("anthropic/claude-sonnet-4.6", 180.0),
    # Anthropic Mythos-class named reasoning models — deep-reasoning tier.
    ("anthropic/claude-fable-5", 600.0),
    ("claude-fable-5", 600.0),
    ("claude-fable", 600.0),
    # xAI Grok reasoning variants — explicit, not bare `grok`.
    ("x-ai/grok-4-fast-reasoning", 300.0),
    ("x-ai/grok-4.20-reasoning", 300.0),
    ("x-ai/grok-4.5", 300.0),
    ("x-ai/grok-4.6", 300.0),
    ("x-ai/grok-4-fast-non-reasoning", 180.0),
])
def test_reasoning_stale_timeout_floor_positive_cases(model, expected):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    assert get_reasoning_stale_timeout_floor(model) == expected, (
        f"get_reasoning_stale_timeout_floor({model!r}) should return "
        f"{expected}; bare substrings and shared prefixes must not "
        f"over-match community derivatives."
    )







# ── integration: _resolved_api_call_stale_timeout_base ─────────────────────


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)










def test_non_reasoning_model_keeps_default(monkeypatch, tmp_path):
    """GPT-5 (non-reasoning) without env var / config -> 90s default, implicit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    # No provider config, no env var, no floor match -> 90s implicit default.
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",
    )
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True


def test_automatic_reasoning_floor_preserves_local_nonstream_opt_out(
    monkeypatch, tmp_path
):
    """An automatic model floor must not look like an explicit user timeout."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="openai-api",
        base_url="http://localhost:5001/v1",
        model="Qwen3.8-27B",
    )

    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 180.0
    assert implicit is True
    assert agent._compute_non_stream_stale_timeout(
        {"messages": [{"role": "user", "content": "analyze this program"}]}
    ) == float("inf")


def test_automatic_reasoning_floor_remains_finite_for_cloud_endpoint(
    monkeypatch, tmp_path
):
    """The local opt-out must not disable the same floor for hosted models."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="openai-api",
        base_url="https://models.example.com/v1",
        model="Qwen3.8-27B",
    )

    assert agent._compute_non_stream_stale_timeout({"messages": []}) == 180.0


def test_explicit_local_reasoning_timeout_remains_finite(monkeypatch, tmp_path):
    """A user-configured local timeout must continue to override the opt-out."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 420.0)

    agent = _make_agent(
        tmp_path,
        provider="openai-api",
        base_url="http://localhost:5001/v1",
        model="Qwen3.8-27B",
    )

    assert agent._compute_non_stream_stale_timeout({"messages": []}) == 420.0


# ── stream-side mirror (the real builder lives in a worker thread) ────────


def _resolve_stream_stale_timeout(
    model: str | None,
    base_url: str,
    est_tokens: int,
    stale_base: float = 180.0,
) -> float:
    """Mirror of the stale-stream resolution in agent/chat_completion_helpers.py.

    Kept in lockstep with the production code at lines 2539-2575 of
    agent/chat_completion_helpers.py.  When that block changes, this
    mirror must change too — the failing-test signal is the divergence.
    """
    from agent.model_metadata import is_local_endpoint
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    # Provider-configured stale timeout wins (mirrors get_provider_stale_timeout).
    if stale_base != 180.0:
        pass  # In production this is sourced from config; here we parameterize.

    if stale_base == 180.0 and base_url and is_local_endpoint(base_url):
        return float("inf")

    if est_tokens > 100_000:
        timeout = max(stale_base, 300.0)
    elif est_tokens > 50_000:
        timeout = max(stale_base, 240.0)
    else:
        timeout = stale_base

    # Reasoning-model floor (the new branch this PR adds).
    floor = get_reasoning_stale_timeout_floor(model)
    if floor is not None:
        timeout = max(timeout, floor)
    return timeout


def test_stream_stale_timeout_floor_for_nemotron_3_ultra():
    """Small-context Nemotron 3 Ultra without explicit config -> 600s floor.

    Without the floor, this would be 180s (the default), which is shorter
    than NVIDIA NIM's ~120s upstream idle kill — guaranteeing broken pipe.
    """
    timeout = _resolve_stream_stale_timeout(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        est_tokens=10_000,
    )
    assert timeout == 600.0
