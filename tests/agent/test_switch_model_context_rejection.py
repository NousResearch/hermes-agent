"""An unusable live model selection leaves the working runtime intact."""

from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_switch_model_context import _make_agent_with_compressor


@pytest.mark.parametrize("main_context,aux_context", [(32768, None), (131072, 32768)])
def test_live_switch_rejects_small_context_without_mutation(main_context, aux_context):
    agent = _make_agent_with_compressor(config_context_length=131072)
    agent.compression_enabled = True
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    old_client = agent.client
    old_compressor = agent.context_compressor
    old_context = old_compressor.context_length
    old_runtime = agent._primary_runtime
    aux_client = MagicMock(base_url="https://aux.example/v1", api_key="dummy-test-key")
    aux_model = "auxiliary-model" if aux_context else "new-model"
    contexts = [main_context, aux_context] if aux_context else [main_context]
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("agent.model_metadata.get_model_context_length", side_effect=contexts),
        patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(aux_client, aux_model)),
    ):
        with pytest.raises(ValueError, match="below the minimum 64,000"):
            agent.switch_model("new-model", "openrouter", api_key="dummy-new-key",
                               base_url="https://openrouter.ai/api/v1")
    assert agent.model == "primary-model"
    assert agent.client is old_client
    assert agent.context_compressor is old_compressor
    assert old_compressor.model == "primary-model"
    assert old_compressor.context_length == old_context
    assert agent._primary_runtime is old_runtime
