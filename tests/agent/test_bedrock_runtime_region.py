"""Tests for Bedrock runtime region resolution in agent_init.

The runtime init paths in ``agent.agent_init`` previously hardcoded
``us-east-1`` whenever base_url carried no region. That diverged from the
rest of the Bedrock surface (model discovery, the picker, doctor, auth),
which all resolve region via ``resolve_bedrock_region()`` with the AWS-SDK
precedence (AWS_REGION env, then AWS_DEFAULT_REGION, then the ~/.aws/config
profile region, then us-east-1). The split meant an EU/AP user with a
profile-configured region saw their real region in discovery but routed to
us-east-1 at inference time.

These tests pin the behavior contract of the shared
``_resolve_bedrock_runtime_region`` helper that both runtime paths use:
  1. An explicit region in the endpoint host always wins.
  2. With no region in base_url, the helper consults resolve_bedrock_region().
"""

from unittest.mock import patch


class TestResolveBedrockRuntimeRegion:
    def test_explicit_base_url_region_wins(self):
        """A region in the endpoint host beats any env/profile resolution."""
        from agent.agent_init import _resolve_bedrock_runtime_region

        url = "https://bedrock-runtime.eu-west-1.amazonaws.com"
        # Even if resolve_bedrock_region would return something else, the
        # explicit endpoint region takes precedence and the helper is not
        # consulted for a value.
        with patch(
            "agent.bedrock_adapter.resolve_bedrock_region",
            return_value="ap-southeast-2",
        ):
            assert _resolve_bedrock_runtime_region(url) == "eu-west-1"

    def test_no_base_url_region_falls_back_to_resolver(self):
        """With no region in base_url, fall back to resolve_bedrock_region()."""
        from agent.agent_init import _resolve_bedrock_runtime_region

        with patch(
            "agent.bedrock_adapter.resolve_bedrock_region",
            return_value="eu-central-1",
        ):
            # Previously this returned the hardcoded "us-east-1".
            assert _resolve_bedrock_runtime_region(None) == "eu-central-1"
            assert _resolve_bedrock_runtime_region("") == "eu-central-1"

    def test_base_url_without_bedrock_host_falls_back_to_resolver(self):
        """A base_url that has no bedrock-runtime region also defers."""
        from agent.agent_init import _resolve_bedrock_runtime_region

        with patch(
            "agent.bedrock_adapter.resolve_bedrock_region",
            return_value="eu-central-1",
        ):
            assert (
                _resolve_bedrock_runtime_region("https://example.invalid/v1")
                == "eu-central-1"
            )

    def test_resolver_us_east_1_default_is_preserved(self):
        """When the resolver yields us-east-1, the helper returns us-east-1.

        This guards against a regression for the common no-env, no-profile
        case: behavior for those users is unchanged.
        """
        from agent.agent_init import _resolve_bedrock_runtime_region

        with patch(
            "agent.bedrock_adapter.resolve_bedrock_region",
            return_value="us-east-1",
        ):
            assert _resolve_bedrock_runtime_region(None) == "us-east-1"
