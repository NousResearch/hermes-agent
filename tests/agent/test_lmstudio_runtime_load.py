from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from run_agent import AIAgent


class TestEnsureLmstudioRuntimeLoaded:
    def _agent(self):
        return SimpleNamespace(
            provider="lmstudio",
            model="qwen3.6-35b-a3b-uncensored-heretic-native-mtp-preserved",
            base_url="http://192.168.1.209:1234/v1",
            api_key="",
            api_mode="chat_completions",
            _config_context_length=None,
            _custom_providers=None,
            context_compressor=None,
            quiet_mode=True,
        )

    def test_uses_detected_context_when_no_explicit_override(self):
        agent = cast(Any, self._agent())
        with patch("agent.model_metadata.get_model_context_length", return_value=262144) as mock_ctx:
            with patch("hermes_cli.models.ensure_lmstudio_model_loaded", return_value=262144) as mock_load:
                AIAgent._ensure_lmstudio_runtime_loaded(agent, None)

        mock_ctx.assert_called_once_with(
            agent.model,
            base_url=agent.base_url,
            api_key=agent.api_key,
            config_context_length=None,
            provider=agent.provider,
            custom_providers=agent._custom_providers,
        )
        mock_load.assert_called_once_with(
            agent.model,
            agent.base_url,
            agent.api_key,
            262144,
        )

    def test_explicit_override_beats_detected_context(self):
        agent = cast(Any, self._agent())
        with patch("agent.model_metadata.get_model_context_length", return_value=262144) as mock_ctx:
            with patch("hermes_cli.models.ensure_lmstudio_model_loaded", return_value=131072) as mock_load:
                AIAgent._ensure_lmstudio_runtime_loaded(agent, 131072)

        mock_ctx.assert_not_called()
        mock_load.assert_called_once_with(
            agent.model,
            agent.base_url,
            agent.api_key,
            131072,
        )

    def test_falls_back_to_minimum_when_detected_context_missing(self):
        agent = cast(Any, self._agent())
        with patch("agent.model_metadata.get_model_context_length", return_value=None) as mock_ctx:
            with patch("hermes_cli.models.ensure_lmstudio_model_loaded", return_value=64000) as mock_load:
                AIAgent._ensure_lmstudio_runtime_loaded(agent, None)

        mock_ctx.assert_called_once()
        mock_load.assert_called_once_with(
            agent.model,
            agent.base_url,
            agent.api_key,
            64000,
        )
