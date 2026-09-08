"""Test that restore_primary_runtime() rebuilds the Bedrock client for the
bedrock_converse api mode instead of falling through to _create_openai_client().

Bug (#102860): when the primary provider is Bedrock and a turn fell back to
another provider, restoring the primary at the top of the next turn hit the
generic OpenAI-client branch and raised

    The api_key client option must be set either by passing api_key to the
    client or by setting the OPENAI_API_KEY environment variable

— Bedrock authenticates via the boto3 chain, so this error was spurious and
stranded the fallback chain for every subsequent turn.
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
    def test_bedrock_restore_rebuilds_bedrock_client_with_region(self):
        """Explicit _bedrock_region is honored when rebuilding the client."""
        agent = _make_bedrock_agent(set_region_attr="eu-west-1")

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            result = agent._restore_primary_runtime()

        assert result is True
        mock_build.assert_called_once_with("eu-west-1")
        assert agent._anthropic_client is not None
        assert agent.client is None  # Bedrock never uses the OpenAI-style client

    def test_bedrock_restore_never_falls_through_to_openai_client(self):
        """The old buggy path called _create_openai_client and died on the
        missing OPENAI_API_KEY; restore must route to the Bedrock client."""
        agent = _make_bedrock_agent()
        agent._create_openai_client = MagicMock(
            side_effect=AssertionError("_create_openai_client must not be called for bedrock")
        )

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            result = agent._restore_primary_runtime()

        assert result is True
        agent._create_openai_client.assert_not_called()

    def test_region_recovered_from_base_url_when_attribute_missing(self):
        """Without a _bedrock_region attribute the restore path re-extracts the
        region from the restored base_url (bedrock-runtime.<region>.amazonaws.com)."""
        agent = _make_bedrock_agent(base_url_region="ap-southeast-2")

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            result = agent._restore_primary_runtime()

        assert result is True
        mock_build.assert_called_once_with("ap-southeast-2")
        assert agent._bedrock_region == "ap-southeast-2"


class TestTryRecoverBedrock:
    """try_recover_primary_transport must route api_mode=bedrock_converse through
    the Bedrock client builder — the same carve-out _rebuild_primary_client has
    (#102860 class); otherwise the recovery path calls _create_openai_client
    for a provider that has no OpenAI key."""

    def _bedrock_agent(self, *, region="eu-west-1"):
        from unittest.mock import MagicMock

        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.model = "anthropic.claude-sonnet-4-6"
        agent.provider = "bedrock"
        agent.base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        agent.api_mode = "bedrock_converse"
        agent.api_key = ""
        agent._client_kwargs = {}
        agent._credential_pool = None
        agent._fallback_activated = False
        agent.client = None
        agent._bedrock_region = region
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

    def test_bedrock_recovery_rebuilds_bedrock_client(self):
        from unittest.mock import patch

        from agent.agent_runtime_helpers import try_recover_primary_transport

        agent = self._bedrock_agent(region="eu-west-1")
        ReadTimeout = type("ReadTimeout", (Exception,), {})

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build, patch("agent.agent_runtime_helpers.time.sleep"):
            mock_build.return_value = object()
            ok = try_recover_primary_transport(
                agent, ReadTimeout("boom"), retry_count=0, max_retries=2
            )

        assert ok is True
        mock_build.assert_called_once_with("eu-west-1")
        assert agent._anthropic_client is not None
        assert agent.client is None
        agent._create_openai_client.assert_not_called()

    def test_bedrock_recovery_never_touches_openai_client(self):
        from unittest.mock import patch

        from agent.agent_runtime_helpers import try_recover_primary_transport

        agent = self._bedrock_agent()
        ReadTimeout = type("ReadTimeout", (Exception,), {})

        with patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client"
        ) as mock_build, patch("agent.agent_runtime_helpers.time.sleep"):
            mock_build.return_value = object()
            ok = try_recover_primary_transport(
                agent, ReadTimeout("boom"), retry_count=0, max_retries=2
            )

        assert ok is True
        agent._create_openai_client.assert_not_called()
