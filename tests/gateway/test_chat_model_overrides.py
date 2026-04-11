from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_source(chat_id: str = "c1", thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        thread_id=thread_id,
        user_id="u1",
        user_name="tester",
    )


def _make_runner(config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._session_model_overrides = {}
    runner.session_store = MagicMock()
    return runner


class TestGatewayConfigChatModelOverrides:
    def test_thread_override_beats_parent_chat_override(self):
        config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")},
            chat_model_overrides={
                "telegram:c1": {"model": "gpt-5.4-mini", "provider": "openai-codex"},
                "telegram:c1:t42": {"model": "claude-sonnet-4", "provider": "anthropic"},
            },
        )

        override = config.get_chat_model_override(_make_source(thread_id="t42"))

        assert override == {"model": "claude-sonnet-4", "provider": "anthropic"}

    def test_parent_chat_override_falls_back_for_threads(self):
        config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")},
            chat_model_overrides={
                "telegram:c1": {"model": "gpt-5.4-mini", "provider": "openai-codex"},
            },
        )

        override = config.get_chat_model_override(_make_source(thread_id="t42"))

        assert override == {"model": "gpt-5.4-mini", "provider": "openai-codex"}


class TestApplyConfiguredChatModelOverride:
    def test_applies_configured_override_via_model_switch(self):
        config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")},
            chat_model_overrides={
                "telegram:c1": {"model": "gpt-5.4-mini", "provider": "openai-codex"},
            },
        )
        runner = _make_runner(config)

        with patch("gateway.run._load_gateway_config", return_value={"providers": {}, "custom_providers": []}), patch(
            "hermes_cli.model_switch.switch_model",
            return_value=SimpleNamespace(
                success=True,
                new_model="gpt-5.4-mini",
                target_provider="openai-codex",
                api_key="chat-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="responses",
            ),
        ) as switch_model:
            model, runtime_kwargs = runner._apply_configured_chat_model_override(
                _make_source(),
                "gpt-5.4",
                {
                    "provider": "openai-codex",
                    "api_key": "global-key",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api_mode": "responses",
                },
            )

        assert model == "gpt-5.4-mini"
        assert runtime_kwargs["provider"] == "openai-codex"
        assert runtime_kwargs["api_key"] == "chat-key"
        assert runtime_kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert runtime_kwargs["api_mode"] == "responses"
        switch_model.assert_called_once()

    def test_session_override_wins_over_configured_chat_default(self):
        config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")},
            chat_model_overrides={
                "telegram:c1": {"model": "gpt-5.4-mini", "provider": "openai-codex"},
            },
        )
        runner = _make_runner(config)
        session_key = "agent:main:telegram:group:c1:u1"
        runner._session_key_for_source = MagicMock(return_value=session_key)
        runner._session_model_overrides[session_key] = {
            "model": "claude-sonnet-4",
            "provider": "anthropic",
            "api_key": "anthropic-key",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={
                "provider": "openai-codex",
                "api_key": "global-key",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "responses",
            },
        ), patch.object(
            runner,
            "_apply_configured_chat_model_override",
            return_value=(
                "gpt-5.4-mini",
                {
                    "provider": "openai-codex",
                    "api_key": "chat-key",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api_mode": "responses",
                },
            ),
        ):
            model, runtime_kwargs = runner._resolve_session_agent_runtime(source=_make_source())

        assert model == "claude-sonnet-4"
        assert runtime_kwargs["provider"] == "anthropic"
        assert runtime_kwargs["api_key"] == "anthropic-key"
