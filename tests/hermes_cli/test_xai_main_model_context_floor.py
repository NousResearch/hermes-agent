from unittest.mock import patch

import hermes_cli.models as models_mod
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length
from hermes_cli.model_switch import switch_model
from hermes_cli.models import validate_requested_model


def _xai_context_length(model_id: str, **_kwargs):
    context_windows = {
        "grok-imagine-video": 1_024,
        "grok-imagine-image-quality": 8_000,
        "grok-4.3": 1_000_000,
    }
    return context_windows.get(model_id, 131_072)


def test_validate_rejects_xai_media_models_below_main_context_floor():
    with (
        patch(
            "hermes_cli.models.provider_model_ids",
            return_value=[
                "grok-4.3",
                "grok-imagine-video",
                "grok-imagine-image-quality",
            ],
        ),
        patch("agent.model_metadata.get_model_context_length", side_effect=_xai_context_length),
    ):
        video = validate_requested_model("grok-imagine-video", "xai-oauth")
        image = validate_requested_model("grok-imagine-image-quality", "xai-oauth")

    assert video["accepted"] is False
    assert video["persist"] is False
    assert "context window of 1,024 tokens" in video["message"]
    assert "64,000-token minimum" in video["message"]

    assert image["accepted"] is False
    assert image["persist"] is False
    assert "context window of 8,000 tokens" in image["message"]
    assert "Grok Imagine" in image["message"]


def test_validate_rejects_direct_xai_media_model_below_main_context_floor():
    with patch("agent.model_metadata.get_model_context_length", side_effect=_xai_context_length):
        result = validate_requested_model("grok-imagine-video", "xai")

    assert result["accepted"] is False
    assert result["persist"] is False
    assert "context window of 1,024 tokens" in result["message"]


def test_validate_rejects_media_model_on_custom_provider_pointed_at_xai():
    with patch("agent.model_metadata.get_model_context_length", return_value=1_024):
        result = validate_requested_model(
            "grok-imagine-video",
            "custom:xai-direct",
            api_key="dummy",
            base_url="https://api.x.ai/v1",
        )

    assert result["accepted"] is False
    assert result["persist"] is False
    assert result.get("hard_reject") is True
    assert "context window of 1,024 tokens" in result["message"]


def test_validate_does_not_autocorrect_media_model_on_custom_xai_endpoint():
    with (
        patch(
            "hermes_cli.models.probe_api_models",
            return_value={
                "models": ["grok-4.3", "grok-imagine-video"],
                "probed_url": "https://api.x.ai/v1/models",
            },
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=1_024),
    ):
        result = validate_requested_model(
            "grok-iamgine-video",
            "custom:xai-direct",
            api_key="dummy",
            base_url="https://api.x.ai/v1",
        )

    assert result["accepted"] is False
    assert result["persist"] is False
    assert result.get("hard_reject") is True
    assert "context window of 1,024 tokens" in result["message"]


def test_validate_does_not_apply_xai_floor_to_other_custom_endpoints():
    with patch(
        "hermes_cli.models.probe_api_models",
        return_value={
            "models": ["grok-imagine-video"],
            "probed_url": "https://models.example.test/v1/models",
        },
    ):
        result = validate_requested_model(
            "grok-imagine-video",
            "custom:example",
            api_key="dummy",
            base_url="https://models.example.test/v1",
        )

    assert result["accepted"] is True


def test_validate_rejects_xai_media_model_when_context_lookup_raises():
    with patch(
        "agent.model_metadata.get_model_context_length",
        side_effect=RuntimeError("probe failed"),
    ):
        result = validate_requested_model("grok-imagine-video", "xai-oauth")

    assert result["accepted"] is False
    assert result["persist"] is False
    assert result.get("hard_reject") is True
    assert "Grok Imagine media-generation model" in result["message"]


def test_validate_still_accepts_xai_chat_model_above_main_context_floor():
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["grok-4.3", "grok-imagine-video"],
    ):
        result = validate_requested_model("grok-4.3", "xai-oauth")

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is True
    assert result["message"] is None


def test_validate_rejects_unlisted_xai_media_model_even_with_high_context_fallback():
    with (
        patch("hermes_cli.models.provider_model_ids", return_value=["grok-4.3"]),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
    ):
        result = validate_requested_model("grok-imagine-video-1.5-preview", "xai-oauth")

    assert result["accepted"] is False
    assert "Grok Imagine media-generation model" in result["message"]


def test_validate_rejects_direct_xai_live_model_below_main_context_floor():
    with (
        patch("hermes_cli.models.fetch_api_models", return_value=["grok-2-vision"]),
        patch("agent.model_metadata.get_model_context_length", return_value=8_192),
    ):
        result = validate_requested_model(
            "grok-2-vision",
            "xai",
            api_key="dummy",
            base_url="https://api.x.ai/v1",
        )

    assert result["accepted"] is False
    assert result.get("hard_reject") is True
    assert "context window of 8,192 tokens" in result["message"]


def test_validate_direct_xai_does_not_autocorrect_to_media_model():
    with (
        patch("hermes_cli.models.fetch_api_models", return_value=["grok-imagine-video"]),
        patch("agent.model_metadata.get_model_context_length", return_value=1_024),
    ):
        result = validate_requested_model(
            "grok-imagien-video",
            "xai",
            api_key="dummy",
            base_url="https://api.x.ai/v1",
        )

    assert result["accepted"] is False
    assert result.get("hard_reject") is True
    assert "context window of 1,024 tokens" in result["message"]


def test_switch_model_does_not_override_xai_context_floor_from_configured_models():
    with (
        patch(
            "hermes_cli.models.provider_model_ids",
            return_value=["grok-4.3", "grok-imagine-video"],
        ),
        patch("agent.model_metadata.get_model_context_length", side_effect=_xai_context_length),
    ):
        result = switch_model(
            "grok-imagine-video",
            current_provider="xai-oauth",
            current_model="grok-4.3",
            user_providers={"xai-oauth": {"models": {"grok-imagine-video": {}}}},
        )

    assert result.success is False
    assert "context window of 1,024 tokens" in result.error_message


def test_xai_media_context_defaults_stay_below_main_floor_without_models_dev():
    media_models = [
        "grok-imagine-image",
        "grok-imagine-image-quality",
        "grok-imagine-video",
        "grok-imagine-video-1.5-preview",
        "grok-imagine-video-1.5-2026-05-30",
    ]

    with patch("agent.models_dev.lookup_models_dev_context", return_value=None):
        for model_id in media_models:
            assert (
                get_model_context_length(model_id, provider="xai-oauth")
                < MINIMUM_CONTEXT_LENGTH
            )


def test_xai_curated_catalog_hides_grok_imagine_media_models():
    data = {
        "xai": {
            "models": {
                "grok-4.3": {},
                "grok-imagine-image": {},
                "grok-imagine-video": {},
                "grok-imagine-video-1.5-preview": {},
            }
        }
    }

    with patch("agent.models_dev._load_disk_cache", return_value=data):
        result = models_mod._xai_curated_models()

    assert "grok-4.3" in result
    assert not any(model.startswith("grok-imagine-") for model in result)
