"""A session pinned to its fallback returns to a Codex primary whose quota window reopened early.

A Codex 429 benches the pool entry and arms the session's ``_rate_limited_until`` until a
``resets_at`` that can be days out. The pool's throttled usage probe notices an early reopen, but
only on ``select()``; the fallback restore gates on the two cooldowns and never reached it.
"""

import json
import time
from unittest.mock import MagicMock, patch

from hermes_time import deadline_clock
from run_agent import AIAgent

CODEX_URL = "https://chatgpt.com/backend-api/codex"
FALLBACK = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}


def _write_exhausted_codex_pool(home, reset_in_seconds):
    reset_at = time.time() + reset_in_seconds
    entry = {
        "id": "codex-1",
        "label": "codex-login",
        "auth_type": "oauth",
        "priority": 0,
        "source": "device_code",
        "access_token": "token",
        "refresh_token": "refresh",
        "base_url": CODEX_URL,
        "last_status": "exhausted",
        "last_status_at": time.time(),
        "last_error_code": 429,
        "last_error_reason": "usage_limit_reached",
        "last_error_message": "The usage limit has been reached",
        "last_error_reset_at": reset_at,
    }
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"openai-codex": [entry]}}))


def _fallback_agent():
    tool_defs = [{"type": "function", "function": {
        "name": "web_search", "description": "x", "parameters": {"type": "object", "properties": {}}}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678", base_url="https://my-llm.example.com/v1", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=FALLBACK,
        )
    agent.client = MagicMock()
    agent._primary_runtime = {**agent._primary_runtime, "provider": "openai-codex", "base_url": CODEX_URL}
    agent._swap_credential = MagicMock()
    _pin_to_fallback(agent)
    return agent


def _pin_to_fallback(agent):
    agent._fallback_index = 0
    client = MagicMock(api_key="fallback-key-1234", base_url="https://openrouter.ai/api/v1")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)):
        assert agent._try_activate_fallback() is True
    # The weekly reset the provider declared, armed by _arm_rate_limit_cooldown.
    agent._rate_limited_until = deadline_clock() + 3 * 86400


def _restore(agent, quota_restored):
    with (
        patch("hermes_cli.auth._probe_codex_quota_restored", return_value=quota_restored),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        return agent._restore_primary_runtime()


def test_returns_to_primary_when_quota_reopens_before_declared_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_exhausted_codex_pool(tmp_path, reset_in_seconds=3 * 86400)
    agent = _fallback_agent()

    assert _restore(agent, quota_restored=True) is True
    assert agent._fallback_activated is False


def test_stays_on_fallback_while_quota_is_still_spent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_exhausted_codex_pool(tmp_path, reset_in_seconds=3 * 86400)
    agent = _fallback_agent()

    assert _restore(agent, quota_restored=False) is False
    assert agent._fallback_activated is True


def test_a_wrong_restored_verdict_does_not_flip_back_every_turn(tmp_path, monkeypatch):
    """The probe caches "restored" for its interval and a fresh 429 does not clear it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_exhausted_codex_pool(tmp_path, reset_in_seconds=3 * 86400)
    agent = _fallback_agent()
    assert _restore(agent, quota_restored=True) is True

    # The primary 429s again: the pool re-benches the entry and the session falls back.
    _write_exhausted_codex_pool(tmp_path, reset_in_seconds=3 * 86400)
    _pin_to_fallback(agent)

    assert _restore(agent, quota_restored=True) is False
    assert agent._fallback_activated is True
