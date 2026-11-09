"""Kimi K3 Bedrock context regression (Leafy1, #149670)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("model_id", (
    "us.moonshotai.kimi-k3",
    "global.moonshotai.kimi-k3",
))
def test_kimi_k3_context_survives_non_numeric_probe_error(model_id):
    """The probe must respect the 16-token minimum and preserve the offline 1M fallback."""
    from agent.bedrock_adapter import get_bedrock_context_length

    client = MagicMock()

    def converse(**kwargs):
        output_cap = kwargs["inferenceConfig"]["maxTokens"]
        if output_cap < 16:
            raise RuntimeError(
                "ValidationException: Invalid 'max_output_tokens': integer below "
                f"minimum value. Expected a value >= 16, but got {output_cap} instead."
            )
        raise RuntimeError(
            "ValidationException: Input Tokens Exceeded: Number of input tokens "
            "exceeds maximum length. Please update the input to try again."
        )

    client.converse.side_effect = converse
    with patch("agent.bedrock_adapter._BEDROCK_PROBE_TIERS", (1_300, 2_200)), \
         patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
        assert get_bedrock_context_length(model_id, region="us-east-1") == 1_000_000

    assert client.converse.call_count == 2
    assert {call.kwargs["inferenceConfig"]["maxTokens"]
            for call in client.converse.call_args_list} == {16}
