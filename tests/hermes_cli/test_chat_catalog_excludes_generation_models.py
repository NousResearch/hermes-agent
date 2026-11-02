"""Chat catalogs must not offer generation models, and a direct switch must reject them.

Name shape and capability type are the catalog's own signals — not one model id.
"""

import json
from unittest.mock import MagicMock, patch

from providers.base import ProviderProfile


def _models_response(items):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({"data": items}).encode()
    return response


def test_live_catalog_drops_capability_typed_and_name_shaped_generation_models():
    profile = ProviderProfile(
        name="alibaba-token-plan",
        base_url="https://token-plan.example.test/v1",
    )
    items = [
        {"id": "wan2.7-image-pro"},
        {"id": "acme/text-to-image"},
        {"id": "gen-1", "capabilities": {"type": "image"}},
        {"id": "clip-gen", "capabilities": {"type": "video-generation"}},
        {"id": "router/flux-pro", "architecture": {"output_modalities": ["image"]}},
        {"id": "qwen3.7-plus", "capabilities": {"type": "chat"}},
        {"id": "qwen-vl-max"},
        {"id": "missing-capabilities"},
        {"id": "malformed-capabilities", "capabilities": "image"},
        {"id": "unknown-kind", "capabilities": {"type": "future-kind"}},
    ]

    with patch("hermes_cli.urllib_security.open_credentialed_url", return_value=_models_response(items)):
        assert profile.fetch_models(api_key="test-key") == [
            "qwen3.7-plus",
            "qwen-vl-max",
            "missing-capabilities",
            "malformed-capabilities",
            "unknown-kind",
        ]


def test_shared_chat_catalog_drops_cached_generation_ids():
    """A warm picker cache is the list model.options serves. Generation ids must not survive it."""
    from hermes_cli.models import cached_provider_model_ids

    entry = {
        "fp": "fp",
        "at": 10**10,
        "models": ["qwen3.7-plus", "wan2.7-image-pro", "acme/text-to-video", "qwen-vl-max"],
    }
    with (
        patch("hermes_cli.models._credential_fingerprint", return_value="fp"),
        patch("hermes_cli.models._load_provider_models_cache", return_value={"alibaba-token-plan": entry}),
        patch("hermes_cli.models._spawn_swr_refresh"),
    ):
        assert cached_provider_model_ids("alibaba-token-plan") == ["qwen3.7-plus", "qwen-vl-max"]


def test_direct_switch_rejects_a_generation_model_and_keeps_a_vision_chat_model():
    from hermes_cli.model_switch import _Switch, _validate_switch

    accepted = {"accepted": True, "persist": True, "recognized": True, "message": ""}

    def _state(model_id):
        return _Switch(
            raw_input=model_id,
            current_provider="alibaba-token-plan",
            current_model="qwen3.7-plus",
            current_base_url="https://token-plan.example.test/v1",
            current_api_key="",
            is_global=False,
            explicit_provider="",
            user_providers=None,
            custom_providers=None,
            new_model=model_id,
            target_provider="alibaba-token-plan",
            provider_label="Alibaba Cloud (Token Plan)",
        )

    with patch("hermes_cli.models_validate.validate_requested_model", return_value=accepted):
        rejected = _validate_switch(_state("vendor/image-pro"))
        kept = _validate_switch(_state("qwen-vl-max"))

    assert rejected is not None and rejected.success is False
    assert "chat" in (rejected.error_message or "").lower()
    assert kept is None


def test_capability_typed_id_is_rejected_once_the_catalog_has_seen_it():
    """An innocuous id is still non-chat when the catalog item says so."""
    from hermes_cli.model_switch import _Switch, _validate_switch

    profile = ProviderProfile(name="example", base_url="https://catalog.example.test/v1")
    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        return_value=_models_response([{"id": "gen-1", "capabilities": {"type": "image"}}]),
    ):
        assert profile.fetch_models(api_key="test-key") == []

    state = _Switch(
        raw_input="gen-1",
        current_provider="example",
        current_model="qwen3.7-plus",
        current_base_url="https://catalog.example.test/v1",
        current_api_key="",
        is_global=False,
        explicit_provider="",
        user_providers=None,
        custom_providers=None,
        new_model="gen-1",
        target_provider="example",
        provider_label="Example",
    )
    accepted = {"accepted": True, "persist": True, "recognized": True, "message": ""}
    with patch("hermes_cli.models_validate.validate_requested_model", return_value=accepted):
        rejected = _validate_switch(state)

    assert rejected is not None and rejected.success is False
