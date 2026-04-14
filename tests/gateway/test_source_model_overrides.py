"""Tests for persistent source-aware model overrides in the gateway."""

from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _make_source(user_id: str = "1490536274782716089") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        user_name="Mario",
        chat_id="channel-1",
        chat_type="group",
    )


def test_source_model_override_applies_for_matching_user():
    """A matching source_model_overrides rule should replace the global default model."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}

    user_config = {
        "model": {"default": "gpt-5.4-mini"},
        "source_model_overrides": [
            {
                "match": {"platform": "discord", "user_id": "1490536274782716089"},
                "model": "gpt-5.4",
            }
        ],
    }

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }):
        model, kwargs = runner._resolve_session_agent_runtime(
            source=_make_source(),
            user_config=user_config,
        )

    assert model == "gpt-5.4"
    assert kwargs["provider"] == "openai-codex"


def test_session_model_override_beats_source_model_override():
    """An explicit /model session switch should still win over persistent source rules."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = _make_source()
    session_key = build_session_key(source)
    runner._session_model_overrides = {
        session_key: {
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": "***",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "chat_completions",
        }
    }

    user_config = {
        "model": {"default": "gpt-5.4-mini"},
        "source_model_overrides": [
            {
                "match": {"platform": "discord", "user_id": "1490536274782716089"},
                "model": "gpt-5.4",
            }
        ],
    }

    model, kwargs = runner._resolve_session_agent_runtime(
        source=source,
        user_config=user_config,
    )

    assert model == "gpt-4o"
    assert kwargs["provider"] == "openai"
