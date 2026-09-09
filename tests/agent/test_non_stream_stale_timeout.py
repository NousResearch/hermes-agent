"""Tests for the non-stream stale-call detector context estimator.

Covers:
- ``estimate_request_context_tokens`` for Chat Completions, Responses API,
  bare lists, and mixed-shape dicts.
- ``AIAgent._compute_non_stream_stale_timeout`` with both legacy ``messages``
  list and full ``api_kwargs`` dicts.
- The May 2026 default-base change (300s -> 90s) and the lowered
  context-tier ceilings (450/600 -> 150/240).
"""

from __future__ import annotations

from pathlib import Path



def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


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


# ── estimator ──────────────────────────────────────────────────────────────




def test_estimator_responses_api_input():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "instructions": "i" * 1000,
        "input": "x" * 4000,
        "tools": [{"name": "t", "description": "d" * 200}],
    }
    # input(4000) + instructions(1000) + tools (~stringified) -> well over 1000 tokens
    tokens = estimate_request_context_tokens(payload)
    assert tokens >= 1200, f"Responses API estimator returned {tokens}"






def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0




# ── default base + tier scaling ────────────────────────────────────────────


def test_default_base_is_90s(monkeypatch, tmp_path):
    """Default base stale timeout dropped from 300s to 90s (May 2026)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True










def test_explicit_user_config_overrides_default(monkeypatch, tmp_path):
    """If the user explicitly sets a stale_timeout, the new defaults don't apply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  openai-codex:
    stale_timeout_seconds: 1800
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    agent = _make_agent(tmp_path)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


# ── openai-codex gateway-scale stale floor ────────────────────────────────




def test_openai_codex_stale_floor_tiers():
    from agent.chat_completion_helpers import openai_codex_stale_timeout_floor

    assert openai_codex_stale_timeout_floor(55_000) == 900.0
    assert openai_codex_stale_timeout_floor(120_000) == 1200.0


# ── delegated-child 150s floor (t_5806fd2b) ───────────────────────────────


def _make_subagent_agent(tmp_path: Path, **overrides):
    """Child-shaped AIAgent: nous chat-completions, subagent platform stamp."""
    from run_agent import AIAgent
    kwargs = dict(
        model="z-ai/glm-5.3-flash",
        provider="nous",
        api_key="sk-dummy",
        base_url="https://inference-api.nousresearch.com/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="subagent",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


def test_delegated_child_implicit_floor_is_150s(monkeypatch, tmp_path):
    """Delegated children get a 150s implicit non-stream stale floor: their gateway's
    Cloudflare Proxy Read Timeout (120s) exceeds the 90s default, so the local watchdog
    must never win the race against the provider's actionable 524 (t_5806fd2b)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_subagent_agent(tmp_path)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 150.0
    assert implicit is True


def test_delegated_child_context_var_floor(monkeypatch, tmp_path):
    """The delegation ContextVar alone (no platform stamp) also raises the floor."""
    from agent.delegation_context import delegated_child_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_subagent_agent(tmp_path, platform="cli")
    with delegated_child_context("sess-x"):
        base, _ = agent._resolved_api_call_stale_timeout_base()
    assert base == 150.0


def test_delegated_child_explicit_config_still_wins(monkeypatch, tmp_path):
    """Explicit user/provider stale_timeout_seconds still beats the child floor."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  nous:
    stale_timeout_seconds: 300
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    try:
        agent = _make_subagent_agent(tmp_path)
        base, implicit = agent._resolved_api_call_stale_timeout_base()
        assert base == 300.0
        assert implicit is False
    finally:
        importlib.reload(to_mod)
