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



# ── pure-function resolver ────────────────────────────────────────────────









def test_floor_matching_is_vendor_prefix_and_variant_suffix_transparent():
    """A routed/vendor-prefixed or -variant/:sku slug resolves to its family's floor,
    while a look-alike community derivative does not match at all."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor as floor

    assert floor("openai/o3-mini") == floor("o3-mini") is not None
    assert floor("gpt-5.6-sol-900k") == floor("gpt-5.6-sol") is not None
    assert floor("thinkingmachines/inkling:free") == floor("thinkingmachines/inkling") is not None
    assert floor("openai/gpt-4o") is None


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


def test_gpt_5_6_floor_reaches_non_stream_and_stream_resolvers(monkeypatch, tmp_path):
    """Small GPT-5.6 requests get the reasoning floor on both request paths."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    import run_agent
    from agent.chat_completion_helpers import _cloud_stale_timeout
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, model="gpt-5.6-sol")

    assert agent._resolved_api_call_stale_timeout_base() == (600.0, False)
    assert _cloud_stale_timeout(180.0, {"model": "gpt-5.6-sol", "input": "small"}) == 600.0
    assert get_reasoning_stale_timeout_floor("vendor/my-gpt-5.6-sol") is None
    assert get_reasoning_stale_timeout_floor("gpt-5.60-sol") is None
    # Non-reasoning OpenAI chat line stays on the defaults (#103802 control case).
    for chat_model in ("gpt-4.1", "gpt-4o", "gpt-5.1-chat-latest"):
        assert get_reasoning_stale_timeout_floor(chat_model) is None
        assert _cloud_stale_timeout(180.0, {"model": chat_model, "input": "small"}) == 180.0

    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 900.0)
    assert agent._resolved_api_call_stale_timeout_base() == (900.0, False)
    assert _cloud_stale_timeout(900.0, {"model": "gpt-5.6-sol", "input": "small"}) == 900.0


def test_explicit_provider_stale_timeout_wins_over_context_tier_and_reasoning_floor(monkeypatch, tmp_path):
    """``providers.<id>.stale_timeout_seconds`` must be able to SHORTEN patience (#115024):
    a 60s explicit value on a >50k-token request for a reasoning model stays 60s, while the
    same request with no explicit value still gets the 240s tier / 600s floor (control)."""
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
    from agent.chat_completion_helpers import _derive_stream_stale_timeout

    api_kwargs = {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "word " * 60_000}]}
    agent = SimpleNamespace(provider="custom", model="gpt-5.6-sol", base_url="https://api.example.invalid/v1")

    _write_config(tmp_path, "providers:\n  custom:\n    stale_timeout_seconds: 60\n")
    assert _derive_stream_stale_timeout(agent, api_kwargs) == 60.0

    _write_config(tmp_path, "")
    assert _derive_stream_stale_timeout(agent, api_kwargs) == 600.0
