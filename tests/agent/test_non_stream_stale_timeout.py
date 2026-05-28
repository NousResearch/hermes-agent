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


def test_estimator_chat_completions_messages():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
        ],
    }
    # 800+ chars from messages -> ~200 tokens (char/4 estimate)
    assert estimate_request_context_tokens(payload) >= 200


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


def test_estimator_responses_api_long_session_triggers_tier():
    """A real long Codex session (large ``input``) should clear the 50k boundary."""
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "input": "x" * 240_000,  # ~60k tokens (240k chars / 4)
        "instructions": "s" * 4000,
    }
    assert estimate_request_context_tokens(payload) > 50_000


def test_estimator_bare_list_back_compat():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    messages = [
        {"role": "user", "content": "x" * 800},
    ]
    assert estimate_request_context_tokens(messages) >= 200


def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0


def test_estimator_unknown_dict_fallback():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {"random_field": "z" * 400}
    assert estimate_request_context_tokens(payload) > 50


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


def test_short_codex_request_uses_base_only(monkeypatch, tmp_path):
    """Codex payload below 50k tokens -> default 90s base."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "hi", "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


def test_long_codex_request_bumps_to_50k_tier(monkeypatch, tmp_path):
    """Codex payload > 50k tokens -> at least 150s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "x" * 240_000, "instructions": ""}
    timeout = agent._compute_non_stream_stale_timeout(payload)
    assert timeout >= 150.0
    assert timeout < 240.0


def test_very_long_codex_request_bumps_to_100k_tier(monkeypatch, tmp_path):
    """Codex payload > 100k tokens -> at least 240s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "x" * 500_000, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) >= 240.0


def test_chat_completions_long_messages_bumps_tier(monkeypatch, tmp_path):
    """Chat Completions estimator still works for the legacy messages path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.4",
    )
    payload = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "x" * 240_000}],
    }
    assert agent._compute_non_stream_stale_timeout(payload) >= 150.0


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
