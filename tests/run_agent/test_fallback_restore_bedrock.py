"""restore_primary_runtime must rebuild Bedrock primaries without OpenAI (#102860).

A Bedrock primary that fell back previously fell through to
``_create_openai_client`` on restore, raising a spurious OPENAI_API_KEY
error and permanently stranding the fallback chain. Converse uses boto3
directly (no long-lived client); Claude-on-Bedrock uses the
AnthropicBedrock SDK — mirrored from ``_rebuild_anthropic_client``.
"""

from unittest.mock import MagicMock, patch


def _bedrock_agent(api_mode):
    agent = MagicMock()
    agent._primary_runtime = {
        "model": "anthropic.claude-sonnet-4-5",
        "provider": "bedrock",
        "base_url": "https://bedrock-runtime.eu-west-1.amazonaws.com",
        "api_mode": api_mode,
        "api_key": "aws-sdk",
        "bedrock_region": "eu-west-1",
        "client_kwargs": {},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "runtime_capabilities": {},
        "reasoning_config": None,
        "compressor_model": "anthropic.claude-sonnet-4-5",
        "compressor_base_url": "",
        "compressor_api_key": "",
        "compressor_provider": "bedrock",
        "compressor_context_length": 0,
        "compressor_api_mode": api_mode,
        "compressor_threshold_tokens": 0,
    }
    agent._fallback_activated = True
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._transport_cache = {}
    agent._rate_limited_until = 0
    agent.model = "fallback-model"
    agent.provider = "bedrock"
    agent.api_mode = api_mode
    agent.base_url = "https://bedrock-runtime.eu-west-1.amazonaws.com"
    agent.reasoning_config = None
    agent.runtime_capabilities = {}
    agent.context_compressor = MagicMock()
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent._create_openai_client = MagicMock()
    return agent


class TestRestoreBedrockPrimary:
    def test_converse_restore_leaves_no_openai_client(self):
        """bedrock_converse restore: client None, region kept, no OpenAI build."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = _bedrock_agent("bedrock_converse")
        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build:
            assert restore_primary_runtime(agent) is True
            mock_build.assert_not_called()
        agent._create_openai_client.assert_not_called()
        assert agent.client is None
        assert agent._bedrock_region == "eu-west-1"

    def test_anthropic_messages_restore_rebuilds_bedrock_sdk_client(self):
        """Claude-on-Bedrock restore rebuilds the SDK client for the region."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = _bedrock_agent("anthropic_messages")
        sentinel = MagicMock()
        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client",
            return_value=sentinel,
        ) as mock_build:
            assert restore_primary_runtime(agent) is True
            mock_build.assert_called_once_with("eu-west-1")
        agent._create_openai_client.assert_not_called()
        assert agent.client is None
        assert agent._anthropic_client is sentinel
        assert agent._bedrock_region == "eu-west-1"

    def test_missing_snapshot_region_falls_back_to_live_attr(self):
        """Pre-fix snapshots without bedrock_region still restore from the agent."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = _bedrock_agent("bedrock_converse")
        del agent._primary_runtime["bedrock_region"]
        agent._bedrock_region = "ap-south-1"
        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ):
            assert restore_primary_runtime(agent) is True
        assert agent._bedrock_region == "ap-south-1"
        assert agent.client is None
