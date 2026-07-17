"""Contract tests for the bundled Together AI provider profile."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def together_profile():
    import providers

    profile = providers.get_provider_profile("together")
    assert profile is not None
    return profile


def test_profile_identity(together_profile):
    assert together_profile.name == "together"
    assert together_profile.auth_type == "api_key"
    assert together_profile.base_url == "https://api.together.ai/v1"
    assert together_profile.env_vars == ("TOGETHER_API_KEY",)
    assert together_profile.default_headers == {}


@pytest.mark.parametrize("alias", ["together-ai", "togetherai"])
def test_profile_aliases_resolve(together_profile, alias):
    import providers

    assert providers.get_provider_profile(alias) is together_profile


def test_defaults_use_namespaced_model_ids(together_profile):
    assert "/" in together_profile.default_aux_model
    assert together_profile.fallback_models
    assert all("/" in model for model in together_profile.fallback_models)


def test_tool_result_replay_strips_optional_name(together_profile):
    original = {
        "role": "tool",
        "name": "hermes_test_echo",
        "tool_call_id": "call-1",
        "content": '{"echo":"ping"}',
    }

    prepared = together_profile.prepare_messages([original])

    assert "name" not in prepared[0]
    assert original["name"] == "hermes_test_echo"


def test_catalog_filters_non_chat_surfaces(together_profile):
    payload = [
        {"id": "chat/model", "type": "chat"},
        {"id": "image/model", "type": "image"},
        {"id": "embed/model", "type": "embedding"},
        {"id": "private/model"},
    ]
    response = io.BytesIO(json.dumps(payload).encode())
    module = sys.modules[together_profile.__class__.__module__]

    with patch.object(
        module,
        "open_credentialed_url",
        return_value=response,
    ):
        models = together_profile.fetch_models(api_key="test-key")

    assert models == ["chat/model", "private/model"]
