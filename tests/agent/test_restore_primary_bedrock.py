"""Tests for the bedrock_converse carve-out in runtime restore/recovery (#102860).

``bedrock_converse`` talks to Bedrock through a lazily-created boto3
bedrock-runtime client (``agent.bedrock_adapter._get_bedrock_runtime_client``);
the request path never reads ``_anthropic_client``, and the Bedrock dependency
group installs boto3 but NOT the optional Anthropic SDK. Restoring this api_mode
must therefore rebuild the runtime *state* (region + client slots) without
constructing an Anthropic client — an earlier revision built an
``AnthropicBedrock`` client, which turned the reported OPENAI_API_KEY failure
into an Anthropic-dependency failure and still stranded the fallback chain
(review by ehz0ah).
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_bedrock_agent(*, base_url_region: str = "eu-west-1", set_region_attr=None):
    """Create a minimal AIAgent in bedrock_converse mode with a fallback active."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "anthropic.claude-sonnet-4-6"
    agent.provider = "bedrock"
    base_url = f"https://bedrock-runtime.{base_url_region}.amazonaws.com"
    agent.base_url = base_url
    agent.api_mode = "bedrock_converse"
    agent.api_key = ""
    agent._client_kwargs = {}
    agent._credential_pool = None
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.context_compressor = MagicMock()
    agent.context_compressor.update_model = MagicMock()
    if set_region_attr:
        agent._bedrock_region = set_region_attr
    # snapshot mirrors what agent init captured for a bedrock primary
    agent._primary_runtime = {
        "model": agent.model,
        "provider": "bedrock",
        "base_url": base_url,
        "api_mode": "bedrock_converse",
        "api_key": "",
        "client_kwargs": {},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "compressor_model": agent.model,
        "compressor_base_url": base_url,
        "compressor_api_key": "",
        "compressor_provider": "bedrock",
        "compressor_context_length": 200000,
        "compressor_api_mode": "bedrock_converse",
    }
    return agent


class TestRestorePrimaryBedrock:
    def test_restore_recovers_region_and_state_without_any_client(self):
        """Explicit _bedrock_region is honored; no client is constructed eagerly
        (the boto3 Converse client is created lazily on the next request)."""
        agent = _make_bedrock_agent(set_region_attr="eu-west-1")

        result = agent._restore_primary_runtime()

        assert result is True
        assert agent._bedrock_region == "eu-west-1"
        assert agent.client is None
        assert agent._anthropic_client is None
        assert agent._client_kwargs == {}

    def test_restore_never_falls_through_to_openai_client(self):
        """The old buggy path called _create_openai_client and died on the
        missing OPENAI_API_KEY; restore must not take that branch."""
        agent = _make_bedrock_agent()
        agent._create_openai_client = MagicMock(
            side_effect=AssertionError("_create_openai_client must not be called for bedrock")
        )

        assert agent._restore_primary_runtime() is True
        agent._create_openai_client.assert_not_called()

    def test_region_recovered_from_base_url_when_attribute_missing(self):
        agent = _make_bedrock_agent(base_url_region="ap-southeast-2")

        assert agent._restore_primary_runtime() is True
        assert agent._bedrock_region == "ap-southeast-2"

    def test_restore_does_not_require_the_anthropic_sdk(self):
        """Bedrock-only installs (boto3, no `anthropic` extra) must restore.

        Simulate an unavailable Anthropic SDK: building an AnthropicBedrock
        client raises. The restore must still succeed because that client is
        never part of the bedrock_converse path (ehz0ah's exact probe)."""
        agent = _make_bedrock_agent()

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client",
            side_effect=ImportError("The 'anthropic' package is required ..."),
        ) as mock_build:
            result = agent._restore_primary_runtime()

        assert result is True
        mock_build.assert_not_called()
        assert agent._anthropic_client is None


class TestTryRecoverBedrock:
    """try_recover_primary_transport must also route api_mode=bedrock_converse
    through the Converse state restore — never through _create_openai_client,
    and never by constructing an Anthropic client (#102860 class)."""

    def _bedrock_agent(self, *, region="eu-west-1"):
        agent = _make_bedrock_agent(base_url_region=region, set_region_attr=region)
        agent._fallback_activated = False
        agent.client = None
        agent._primary_runtime = {
            "model": agent.model,
            "provider": "bedrock",
            "base_url": agent.base_url,
            "api_mode": "bedrock_converse",
            "api_key": "",
            "client_kwargs": {},
        }
        agent._vprint = lambda *a, **k: None
        agent.log_prefix = ""
        agent._retire_shared_openai_client = MagicMock()
        agent._create_openai_client = MagicMock(
            side_effect=AssertionError("_create_openai_client must not be called for bedrock")
        )
        return agent

    def test_recovery_restores_converse_state_and_evicts_cached_client(self):
        from agent.agent_runtime_helpers import try_recover_primary_transport

        agent = self._bedrock_agent(region="eu-west-1")
        ReadTimeout = type("ReadTimeout", (Exception,), {})

        with patch(
            "agent.bedrock_adapter.invalidate_runtime_client"
        ) as mock_invalidate, patch("agent.agent_runtime_helpers.time.sleep"):
            ok = try_recover_primary_transport(
                agent, ReadTimeout("boom"), retry_count=0, max_retries=2
            )

        assert ok is True
        agent._create_openai_client.assert_not_called()
        assert agent._anthropic_client is None
        # A fresh connection pool for the retry: the cached boto3 bedrock-runtime
        # client for the restored region is evicted.
        mock_invalidate.assert_called_once_with("eu-west-1")

    def test_recovery_never_constructs_anthropic_client(self):
        from agent.agent_runtime_helpers import try_recover_primary_transport

        agent = self._bedrock_agent()
        ReadTimeout = type("ReadTimeout", (Exception,), {})

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client",
            side_effect=ImportError("The 'anthropic' package is required ..."),
        ) as mock_build, patch("agent.agent_runtime_helpers.time.sleep"), patch(
            "agent.bedrock_adapter.invalidate_runtime_client"
        ):
            ok = try_recover_primary_transport(
                agent, ReadTimeout("boom"), retry_count=0, max_retries=2
            )

        assert ok is True
        mock_build.assert_not_called()
        agent._create_openai_client.assert_not_called()
