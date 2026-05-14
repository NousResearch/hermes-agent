"""Tests for gateway per-channel default model routes."""

from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.run import GatewayRunner, _configured_channel_model_route


def test_discord_parent_channel_model_route_matches_thread_source():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="1504333465221464224",
    )
    cfg = {
        "discord": {
            "channel_model_routes": {
                "1504333465221464224": {
                    "model": "qwen3.6:27b",
                    "provider": "ollama",
                }
            }
        }
    }

    assert _configured_channel_model_route(source, cfg) == {
        "model": "qwen3.6:27b",
        "provider": "ollama",
    }


def test_configured_channel_model_route_applies_to_runtime_before_global_default():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)  # type: ignore[assignment]

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        user_id="user-1",
        thread_id="thread-1",
        parent_chat_id="1504333465221464224",
    )
    cfg = {
        "model": {"default": "gpt-5.5", "provider": "openai-codex"},
        "discord": {
            "channel_model_routes": {
                "1504333465221464224": {
                    "model": "qwen3.6:27b",
                    "provider": "ollama",
                }
            }
        },
    }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolve:
        resolve.return_value = {
            "provider": "custom",
            "api_key": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_mode": "chat_completions",
            "model": "qwen3.6:27b",
        }
        model, kwargs = runner._resolve_session_agent_runtime(source=source, user_config=cfg)

    assert model == "qwen3.6:27b"
    assert kwargs["provider"] == "custom"
    assert kwargs["base_url"] == "http://127.0.0.1:11434/v1"
    resolve.assert_called_once_with(
        requested="ollama",
        explicit_api_key=None,
        explicit_base_url=None,
        target_model="qwen3.6:27b",
    )


def test_session_model_override_beats_configured_channel_default():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)  # type: ignore[assignment]
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1504333465221464224",
        chat_type="channel",
        user_id="user-1",
    )
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides = {
        session_key: {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "api_key": "codex-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        }
    }
    cfg = {
        "discord": {
            "channel_model_routes": {
                "1504333465221464224": {
                    "model": "qwen3.6:27b",
                    "provider": "ollama",
                }
            }
        }
    }

    model, kwargs = runner._resolve_session_agent_runtime(source=source, user_config=cfg)

    assert model == "gpt-5.5"
    assert kwargs["provider"] == "openai-codex"
