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
   returns ``uses_implicit_default=False`` so the local-endpoint
   short-circuit in ``_compute_non_stream_stale_timeout`` does not
   disable stale detection for a reasoning model running on a local
   NIM endpoint.
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
    # xAI Grok reasoning variants — explicit, not bare `grok`.
    ("x-ai/grok-4-fast-reasoning", 300.0),
    ("x-ai/grok-4.20-reasoning", 300.0),
    ("x-ai/grok-4-fast-non-reasoning", 180.0),
    # MiniMax M2.x reasoning models (#62353).
    ("minimax/minimax-m2.7", 300.0),
    ("minimax/minimax-m2.5", 300.0),
    ("minimax-m2.7", 300.0),
    ("minimax-m2.5", 300.0),
])
def test_reasoning_stale_timeout_floor_positive_cases(model, expected):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    assert get_reasoning_stale_timeout_floor(model) == expected, (
        f"get_reasoning_stale_timeout_floor({model!r}) should return "
        f"{expected}; bare substrings and shared prefixes must not "
        f"over-match community derivatives."
    )


@pytest.mark.parametrize("model", [
    # Non-reasoning chat models — no floor.
    "gpt-4o",
    "gpt-5",
    "claude-3-5-sonnet-20240620",
    "llama-3.3-70b-instruct",
    "gemini-2.5-pro",
    # Start-of-slug anchor traps — the slug must be at the START of
    # the bare model name (after aggregator-prefix strip).  Bare
    # substring matching would over-match these.
    "olmo-1",
    "olmo-13b",
    "llama-4-70b-o1-preview",     # embedded `o1-preview`, NOT start of slug
    "some-model-o3-mini-fork",    # embedded `o3-mini`, NOT start of slug
    # Bare "grok" must not over-match non-reasoning Grok SKUs.
    "x-ai/grok-3",
    "x-ai/grok-4",
    "x-ai/grok-4-0709",
    "x-ai/grok-code-fast-1",
    # Qwen2 must not match Qwen3 (different family).
    "qwen2-72b-instruct",
    # Non-reasoning DeepSeek chat must not match the v4 reasoning entries.
    "deepseek-chat",
    "deepseek/deepseek-chat",
    "some-deepseek-v4-flash",     # embedded v4 slug, NOT start of slug
    # Empty / None / non-string inputs — must return None, not raise.
    "",
    None,
    12345,
    [],
])
def test_reasoning_stale_timeout_floor_negative_cases(model):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    assert get_reasoning_stale_timeout_floor(model) is None, (
        f"get_reasoning_stale_timeout_floor({model!r}) must return None "
        f"for non-reasoning models and start-of-slug-anchor traps."
    )


def test_longest_substring_wins_on_shared_prefix():
    """`o3-mini` must beat `o3` so the smaller floor applies."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    # o3-mini (7 chars) wins over o3 (2 chars) on shared prefix.
    assert get_reasoning_stale_timeout_floor("openai/o3-mini") == 300.0
    assert get_reasoning_stale_timeout_floor("openai/o3") == 600.0
    # Even with deep aggregator prefix chains the model name resolves
    # correctly (start-of-slug anchor + rsplit('/') strip).
    assert get_reasoning_stale_timeout_floor("openrouter/openai/o3-mini") == 300.0
    assert get_reasoning_stale_timeout_floor("openrouter/anthropic/claude-opus-4-6") == 240.0



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


def test_reasoning_floor_applies_to_nemotron_3_ultra(monkeypatch, tmp_path):
    """Nemotron 3 Ultra without explicit config gets the 600s floor."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    # Isolate the floor path from leaked provider config: with no per-model /
    # per-provider stale_timeout_seconds, the resolver falls through to the
    # reasoning floor.  Patch the lookup to return None deterministically
    # rather than relying on importlib.reload of shared config modules, which
    # races other tests in the same xdist worker (#52217 flake).
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemotron-3-ultra-550b-a55b",
    )
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 600.0
    assert implicit is False, (
        "Reasoning-model floor must return uses_implicit_default=False "
        "so the local-endpoint short-circuit in "
        "_compute_non_stream_stale_timeout does not disable detection "
        "for users running reasoning models on a local NIM endpoint."
    )


def test_reasoning_floor_applies_to_opus_4_thinking(monkeypatch, tmp_path):
    """Anthropic Opus 4.x thinking gets the 240s floor without explicit config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    # Deterministic floor path — see test_reasoning_floor_applies_to_nemotron_3_ultra.
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-4-6",
    )
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 240.0
    assert implicit is False


def test_reasoning_floor_never_overrides_explicit_user_config(monkeypatch, tmp_path):
    """Explicit per-model stale_timeout_seconds wins over the floor.

    Regression guard for the invariant: explicit user config > reasoning
    floor > env var > default. If a user sets stale_timeout_seconds: 60
    on Nemotron 3 Ultra, that's what fires — even though the floor
    would otherwise be 600s.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    # Explicit per-model config resolves to 60s (priority 1). The resolver
    # must short-circuit on this and never consult the reasoning floor.
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 60.0)

    agent = _make_agent(
        tmp_path,
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemotron-3-ultra-550b-a55b",
    )
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 60.0, (
        "Explicit user stale_timeout_seconds must override the "
        "reasoning-model floor; the user knows their environment."
    )
    assert implicit is False


def test_reasoning_floor_loses_to_env_var_when_no_floor_match(monkeypatch, tmp_path):
    """For a non-reasoning model, env var still wins over the 90s default."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "300")
    _write_config(tmp_path, "")

    # No provider config -> resolver consults the env var (priority 3).
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",  # not in the floor allowlist
    )
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 300.0
    assert implicit is False


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


def test_stream_stale_timeout_floor_never_lowers_existing():
    """The floor raises; it never lowers the existing context-size tier."""
    # 120k-token conversation on a reasoning model -> context tier already
    # raises to 300s; floor (600s) takes it to 600s.
    timeout = _resolve_stream_stale_timeout(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        est_tokens=120_000,
    )
    assert timeout == 600.0

    # 60k tokens on Opus 4 -> context tier raises to 240s; floor keeps 240s.
    timeout = _resolve_stream_stale_timeout(
        model="anthropic/claude-opus-4-6",
        base_url="https://api.anthropic.com",
        est_tokens=60_000,
    )
    assert timeout == 240.0


def test_stream_stale_timeout_unchanged_for_non_reasoning_models():
    """gpt-4o on a small context still gets the 180s default — no behavior change."""
    timeout = _resolve_stream_stale_timeout(
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        est_tokens=5_000,
    )
    assert timeout == 180.0
