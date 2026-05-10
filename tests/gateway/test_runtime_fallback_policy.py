"""Focused tests for gateway runtime credential fallback routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.auth import AuthError


def _source(platform: str = "weixin", chat_id: str = "chat-1", user_id: str = "user-1"):
    return SimpleNamespace(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_name="Test User",
        chat_name="Test Chat",
        chat_type="group",
        thread_id=None,
    )


def test_strict_gateway_route_fails_closed_without_fallback(monkeypatch):
    """A matched strict route must not silently fall back after primary auth failure."""
    import gateway.run as run

    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openai-codex")
    cfg = {
        "gateway_credential_routing": {
            "rules": [
                {
                    "platform": "weixin",
                    "chat_id": "chat-1",
                    "provider": "openai-codex",
                    "exclusive": True,
                }
            ]
        }
    }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=AuthError("expired", provider="openai-codex")), \
        patch.object(run, "_load_gateway_config", return_value=cfg), \
        patch.object(run, "_try_resolve_fallback_provider") as fallback:
        with pytest.raises(RuntimeError, match="fallback is disabled"):
            run._resolve_runtime_agent_kwargs(source=_source())

    fallback.assert_not_called()


def test_non_strict_gateway_route_uses_fallback_with_metadata(monkeypatch):
    """Non-strict routes retain existing fallback behavior and expose fallback metadata."""
    import gateway.run as run

    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openai-codex")
    fallback_config = {
        "api_key": "fb-key",
        "base_url": "https://fallback.example/v1",
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "fallback_used": True,
        "fallback_reason": "expired",
        "primary_provider": "openai-codex",
    }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=AuthError("expired", provider="openai-codex")), \
        patch.object(run, "_load_gateway_config", return_value={}), \
        patch.object(run, "_try_resolve_fallback_provider", return_value=fallback_config) as fallback:
        resolved = run._resolve_runtime_agent_kwargs(source=_source())

    assert resolved is fallback_config
    fallback.assert_called_once_with(primary_error="expired", primary_provider="openai-codex")


def test_runtime_footer_shows_fallback_warning_when_enabled():
    """Footer-enabled gateways should visibly warn when a fallback runtime handled the turn."""
    from gateway.runtime_footer import build_footer_line

    line = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "provider", "credential"],
                }
            }
        },
        platform_key="weixin",
        model="anthropic/claude-sonnet-4",
        context_tokens=0,
        context_length=None,
        cwd="",
        provider="openrouter",
        credential_label="fallback-key",
        fallback_used=True,
        fallback_reason="expired primary token",
    )

    assert line.startswith("⚠ fallback:")
    assert "claude-sonnet-4" in line
    assert "openrouter" in line
    assert "fallback-key" in line
    assert "expired primary token" in line


class _FakeBackgroundAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, *args, **kwargs):
        self.sent.append((args, kwargs))

    def extract_media(self, text):
        return [], text

    def extract_images(self, text):
        return [], text


async def _inline_executor(func):
    return func()


class _RecordingAgent:
    constructed_kwargs = []

    def __init__(self, **kwargs):
        type(self).constructed_kwargs.append(kwargs)

    def run_conversation(self, **kwargs):
        return {"final_response": "ok", "messages": [], "api_calls": 1, "tools": []}


def _runner_for_background(fallback_model):
    import gateway.run as run
    from gateway.config import Platform

    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters = {Platform.WEIXIN: _FakeBackgroundAdapter()}
    runner._session_db = None
    runner._provider_routing = {}
    runner._fallback_model = fallback_model
    runner._service_tier = None
    runner._reasoning_config = None
    runner._thread_metadata_for_source = lambda source, event_message_id=None: {}
    runner._resolve_session_agent_runtime = lambda *, source, user_config, session_key=None: (
        "primary/model",
        {
            "api_key": "primary-key",
            "base_url": "https://primary.example/v1",
            "provider": "openai-codex",
            "api_mode": "chat_completions",
        },
    )
    runner._resolve_session_reasoning_config = lambda *args, **kwargs: None
    runner._load_service_tier = lambda: None
    runner._run_in_executor_with_context = _inline_executor
    runner._cleanup_agent_resources = lambda agent: None
    return runner


@pytest.mark.asyncio
async def test_strict_gateway_route_disables_aiagent_fallback_after_runtime_success(monkeypatch):
    """Strict routes must disable AIAgent fallback for later API-time 401/403s."""
    import gateway.run as run
    from gateway.config import Platform

    fallback = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    cfg = {
        "gateway_credential_routing": {
            "rules": [
                {
                    "platform": "weixin",
                    "chat_id": "chat-1",
                    "provider": "openai-codex",
                    "fallback_policy": "fail_closed",
                }
            ]
        }
    }
    source = _source(platform=Platform.WEIXIN, chat_id="chat-1")
    runner = _runner_for_background(fallback)
    _RecordingAgent.constructed_kwargs = []

    with patch.object(run, "_load_gateway_config", return_value=cfg), \
        patch("run_agent.AIAgent", _RecordingAgent):
        await runner._run_background_task("hello", source, "task-1")

    assert _RecordingAgent.constructed_kwargs
    assert _RecordingAgent.constructed_kwargs[-1]["fallback_model"] is None


@pytest.mark.asyncio
async def test_non_strict_gateway_route_keeps_aiagent_fallback_after_runtime_success(monkeypatch):
    """Non-strict routes still pass the configured AIAgent fallback chain."""
    import gateway.run as run
    from gateway.config import Platform

    fallback = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    source = _source(platform=Platform.WEIXIN, chat_id="chat-1")
    runner = _runner_for_background(fallback)
    _RecordingAgent.constructed_kwargs = []

    with patch.object(run, "_load_gateway_config", return_value={}), \
        patch("run_agent.AIAgent", _RecordingAgent):
        await runner._run_background_task("hello", source, "task-2")

    assert _RecordingAgent.constructed_kwargs
    assert _RecordingAgent.constructed_kwargs[-1]["fallback_model"] == fallback
