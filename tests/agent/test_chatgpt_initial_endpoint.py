"""Direct agent construction respects provider endpoint contracts before probes."""

from unittest.mock import patch

import pytest

from run_agent import AIAgent


@pytest.mark.parametrize("api_mode", [None, "chat_completions", "codex_responses"])
@pytest.mark.parametrize("provider", ["openai-chatgpt", " OpenAI-ChatGPT "])
def test_direct_chatgpt_agent_pins_client_and_metadata_endpoint(api_mode, provider):
    with (
        patch("agent.process_bootstrap.OpenAI") as client,
        patch("agent.model_metadata.get_model_context_length", return_value=128000) as metadata,
        patch("agent.context_compressor.get_model_context_length", return_value=128000) as compression_metadata,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            provider=provider, api_key="local-bearer",
            base_url="https://relay.example/v1", model="local-unknown-model",
            api_mode=api_mode,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    assert agent.base_url == "https://api.openai.com/v1"
    assert agent.api_mode == "codex_responses"
    assert client.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
    assert metadata.called or compression_metadata.called
    assert all(call.kwargs.get("base_url") == "https://api.openai.com/v1"
               for call in metadata.call_args_list + compression_metadata.call_args_list)
