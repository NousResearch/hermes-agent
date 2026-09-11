from unittest.mock import patch

from hermes_cli.model_switch import _Switch, _validate_switch
from hermes_cli.model_switch_providers import _live_or_curated_ids


def test_generation_models_are_filtered_from_chat_catalogs_and_rejected_on_direct_switch():
    with patch(
        "hermes_cli.models.cached_provider_model_ids",
        return_value=["qwen3.8-max", "wan2.7-image-pro", "wan2.7-video-pro"],
    ):
        assert _live_or_curated_ids("alibaba-token-plan", {}) == ["qwen3.8-max"]

    state = _Switch(
        raw_input="wan2.7-image-pro", current_provider="alibaba-token-plan",
        current_model="qwen3.8-max", current_base_url="https://example.com/v1",
        current_api_key="", is_global=False, explicit_provider="",
        user_providers=None, custom_providers=None, new_model="wan2.7-image-pro",
        target_provider="alibaba-token-plan", provider_label="Alibaba Cloud (Token Plan)",
    )
    result = _validate_switch(state)

    assert result is not None and result.success is False
    assert "cannot be used for chat" in result.error_message