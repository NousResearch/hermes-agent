"""Bedrock context windows for models the static table lagged on, and the family fall-through.

Field report: GPT-6.1 Terra on Bedrock resolved to the 128K default and compressed at 96K (75% small-window
floor) although its AWS card states 1,050,000 tokens. Card:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-1-terra.html
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

TERRA_CARD_WINDOW = 1_050_000  # "Context window: 1,050,000 tokens" (model-card-openai-gpt-6-1-terra)
GPT54_CARD_WINDOW = 1_050_000  # "Context window: 1,050,000 tokens, shared between input and output"
RUNTIME_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.fixture(autouse=True)
def _fresh_bedrock_state():
    from agent import bedrock_adapter
    bedrock_adapter.reset_client_cache()
    yield
    bedrock_adapter.reset_client_cache()


TERRA_FORMS = [
    "openai.gpt-6.1-terra",
    "us.openai.gpt-6.1-terra",
    "global.openai.gpt-6.1-terra",
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.openai.gpt-6.1-terra",
    "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6.1-terra",
]


class TestCardSizedOpenAIModels:
    @pytest.mark.parametrize("model_id", TERRA_FORMS)
    def test_gpt61_terra_resolves_to_its_card_window_in_every_id_form(self, model_id):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length(model_id, probe=False) == TERRA_CARD_WINDOW

    def test_application_profile_wrapping_terra_resolves_to_the_card_window(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/terraprof01"
        client = MagicMock()
        client.get_inference_profile.return_value = {"models": [
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/openai.gpt-6.1-terra"}]}
        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=client):
            assert get_bedrock_context_length(arn, probe=False) == TERRA_CARD_WINDOW

    @pytest.mark.parametrize("model_id", ["openai.gpt-5.4", "us.openai.gpt-5.4", "global.openai.gpt-5.4"])
    def test_gpt54_resolves_to_its_card_window(self, model_id):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length(model_id, probe=False) == GPT54_CARD_WINDOW

    @pytest.mark.parametrize("bare", ["openai.gpt-6.1-terra", "openai.gpt-5.4"])
    def test_terra_and_gpt54_route_like_their_siblings(self, bare):
        """Cards: bare ID on bedrock-mantle only, us./global. profiles on bedrock-runtime (Converse)."""
        from agent.bedrock_adapter import bedrock_openai_uses_mantle, is_openai_bedrock_model
        assert is_openai_bedrock_model(bare) and bedrock_openai_uses_mantle(bare)
        for profile in (f"us.{bare}", f"global.{bare}"):
            assert is_openai_bedrock_model(profile) and not bedrock_openai_uses_mantle(profile)

    def test_terra_keeps_the_gpt6_reasoning_replay_policy(self):
        from agent.bedrock_adapter import _converse_replay_policy
        assert _converse_replay_policy("us.openai.gpt-6.1-terra") == "issuer"

    def test_every_routed_openai_model_has_its_own_card_row(self):
        """Routed (allowlisted) models are sized from their AWS card, not borrowed from the catalog."""
        from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS, BEDROCK_OPENAI_RESPONSES_MODEL_IDS
        assert not [m for m in BEDROCK_OPENAI_RESPONSES_MODEL_IDS if m not in BEDROCK_CONTEXT_LENGTHS]

    def test_terra_compression_trigger_uses_the_card_window(self):
        """Through the real resolver and compressor with no live probe: main compressed at 96,000."""
        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(
            model="us.openai.gpt-6.1-terra", provider="bedrock", threshold_percent=0.5,
            base_url=RUNTIME_URL, api_key="aws-sdk", quiet_mode=True,
        )
        with patch("agent.bedrock_adapter.probe_bedrock_context_length", return_value=None):
            assert compressor.context_length == TERRA_CARD_WINDOW
            assert compressor.threshold_tokens == TERRA_CARD_WINDOW // 2


class TestThirdPartyCardRows:
    """Rows read from each model's AWS card ("Context window" field, 2026-11-04). On main all of these resolved to
    the 128K default, and the ones above 128K compressed early."""

    @pytest.mark.parametrize("model_id, card_window", [
        ("us.amazon.nova-2-lite-v1:0", 1_000_000),
        ("ai21.jamba-1-5-large-v1:0", 256_000),
        ("google.gemma-4-31b", 256_000),
        ("minimax.minimax-m2.5", 196_000),
        ("mistral.devstral-2-123b", 256_000),
        ("mistral.mistral-large-3-675b-instruct", 256_000),
        ("moonshotai.kimi-k2.5", 256_000),
        ("global.moonshotai.kimi-k3", 1_000_000),
        ("qwen.qwen3-coder-480b-a35b-v1:0", 256_000),
        ("zai.glm-4.7", 203_000),
        ("zai.glm-5", 200_000),
        ("global.zai.glm-5.3", 1_000_000),
    ])
    def test_card_window(self, model_id, card_window):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length(model_id, probe=False) == card_window

    def test_older_rows_are_not_shadowed(self):
        """Longest-key match: the new mistral-large-3 row must not capture Mistral Large (2402, card 32K, row 128K)."""
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length("mistral.mistral-large-2402-v1:0", probe=False) == 128_000
        assert get_bedrock_context_length("zai.glm-4.7-flash", probe=False) == 203_000

    def test_every_row_stays_within_the_model_metadata_window(self):
        """The table's stated invariant, now enforced for every row: a hosted window never exceeds the
        model_metadata.DEFAULT_CONTEXT_LENGTHS value for the same bare id when that table knows the model."""
        from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match
        above = []
        for key, hosted in BEDROCK_CONTEXT_LENGTHS.items():
            hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, key.split(".", 1)[1])
            if hit and hosted > hit[1]:
                above.append((key, hosted, hit))
        assert not above, above
