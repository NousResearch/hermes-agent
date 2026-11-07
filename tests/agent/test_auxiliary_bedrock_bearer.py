"""Claude auxiliary requests must select the same Bedrock auth path as chat."""

from unittest.mock import MagicMock, patch

import pytest


MODELS = [None, "anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-sonnet-5"]
DEFAULT_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-west-2"


@pytest.mark.parametrize("model", MODELS)
def test_claude_bearer_aux_uses_converse_without_anthropic_sdk(model):
    from agent.auxiliary_client import BedrockAuxiliaryClient, _build_bedrock_client

    with (
        patch("agent.bedrock_adapter.has_aws_credentials", return_value=True),
        patch("agent.bedrock_adapter.resolve_bedrock_runtime_region", return_value=REGION),
        patch("agent.bedrock_adapter.resolve_bedrock_bearer_token", return_value="test-bearer"),
        patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
              side_effect=AssertionError("Bearer auth must not build the SigV4-only SDK")) as sdk,
    ):
        client, resolved_model = _build_bedrock_client("bedrock", model, raw_codex=False)

    assert isinstance(client, BedrockAuxiliaryClient)
    assert resolved_model == (model or DEFAULT_MODEL)
    assert client._model == resolved_model
    assert client._region == REGION
    sdk.assert_not_called()


@pytest.mark.parametrize("model", MODELS)
def test_claude_iam_aux_keeps_anthropic_bedrock_sdk(model):
    from agent.auxiliary_client import _build_bedrock_client

    native_client = MagicMock()
    wrapped_client = MagicMock()
    with (
        patch("agent.bedrock_adapter.has_aws_credentials", return_value=True),
        patch("agent.bedrock_adapter.resolve_bedrock_runtime_region", return_value=REGION),
        patch("agent.bedrock_adapter.resolve_bedrock_bearer_token", return_value=""),
        patch("agent.anthropic_adapter.build_anthropic_bedrock_client", return_value=native_client) as sdk,
        patch("agent.auxiliary_client.AnthropicAuxiliaryClient", return_value=wrapped_client) as wrapper,
        patch("agent.auxiliary_client.BedrockAuxiliaryClient") as converse,
    ):
        client, resolved_model = _build_bedrock_client("bedrock", model, raw_codex=False)

    assert client is wrapped_client
    assert resolved_model == (model or DEFAULT_MODEL)
    sdk.assert_called_once_with(REGION)
    wrapper.assert_called_once_with(
        native_client, resolved_model, api_key="aws-sdk",
        base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com",
    )
    converse.assert_not_called()
