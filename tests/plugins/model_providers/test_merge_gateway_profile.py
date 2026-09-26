"""Behavioral coverage for the Merge Gateway provider integration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from agent.models_dev import get_model_capabilities
from hermes_cli import runtime_provider as runtime_provider_mod
from hermes_cli.models import provider_model_ids
from hermes_cli.providers import get_provider, is_aggregator
from providers import get_provider_profile


@pytest.fixture
def merge_gateway_profile():
    profile = get_provider_profile("merge-gateway")
    assert profile is not None
    return profile


def test_profile_auto_wires_auth_runtime_and_aggregator_metadata(
    monkeypatch, merge_gateway_profile
):
    monkeypatch.setenv("MERGE_GATEWAY_API_KEY", "merge-test-key")
    monkeypatch.setattr(
        runtime_provider_mod,
        "_get_model_config",
        lambda: {
            "provider": "merge-gateway",
            "default": "anthropic/example-agent-model",
        },
    )

    resolved = runtime_provider_mod.resolve_runtime_provider(
        requested="merge-gateway"
    )
    provider_def = get_provider("merge-gateway", allow_network=False)

    assert merge_gateway_profile.env_vars == ("MERGE_GATEWAY_API_KEY",)
    assert resolved["provider"] == "merge-gateway"
    assert resolved["api_key"] == "merge-test-key"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://api-gateway.merge.dev/v1/openai"
    assert provider_def is not None
    assert provider_def.base_url == resolved["base_url"]
    assert "MERGE_GATEWAY_API_KEY" in provider_def.api_key_env_vars
    assert is_aggregator("merge-gateway") is True


def test_models_dev_provider_uses_router_overlay_endpoint_and_auth(
    monkeypatch, merge_gateway_profile
):
    monkeypatch.setattr(
        "agent.models_dev.get_provider_info",
        lambda *_args, **_kwargs: SimpleNamespace(
            name="Merge Gateway",
            env=("MODELS_DEV_MERGE_KEY",),
            api="https://api-gateway.merge.dev/v1/ai-sdk",
            doc="https://docs.merge.dev/merge-gateway/get-started",
        ),
    )

    provider_def = get_provider("merge-gateway", allow_network=False)

    assert provider_def is not None
    assert provider_def.source == "models.dev"
    assert provider_def.transport == "openai_chat"
    assert provider_def.base_url == merge_gateway_profile.base_url
    assert provider_def.api_key_env_vars == (
        "MODELS_DEV_MERGE_KEY",
        "MERGE_GATEWAY_API_KEY",
    )
    assert provider_def.auth_type == "api_key"
    assert provider_def.is_aggregator is True


def test_model_selector_uses_merge_gateway_profile_catalog(
    monkeypatch, merge_gateway_profile
):
    calls = []

    def fake_fetch_models(**kwargs):
        calls.append(kwargs)
        return ["moonshot/kimi-k3"]

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "merge-test-key",
            "base_url": "https://api-gateway.merge.dev/v1/openai",
        },
    )
    monkeypatch.setattr(
        merge_gateway_profile,
        "fetch_models",
        fake_fetch_models,
    )

    assert provider_model_ids("merge-gateway") == ["moonshot/kimi-k3"]
    assert calls == [
        {
            "api_key": "merge-test-key",
            "base_url": "https://api-gateway.merge.dev/v1/openai",
        }
    ]


def test_fetch_models_paginates_and_keeps_available_tool_routes(
    monkeypatch, merge_gateway_profile
):
    pages = {
        None: {
            "data": [
                {
                    "model": "anthropic/tool-model",
                    "availability_status": "available",
                    "vendors": {
                        "anthropic": {
                            "availability_status": "available",
                            "capabilities": {"supports_tool_calling": True},
                        }
                    },
                },
                {
                    "model": "openai/text-only-model",
                    "availability_status": "available",
                    "vendors": {
                        "openai": {
                            "availability_status": "available",
                            "capabilities": {"supports_tool_calling": False},
                        }
                    },
                },
                {
                    "model": "google/unavailable-model",
                    "availability_status": "unavailable",
                    "vendors": {
                        "google": {
                            "availability_status": "available",
                            "capabilities": {"supports_tool_calling": True},
                        }
                    },
                },
            ],
            "has_more": True,
            "next_cursor": "page-2",
        },
        "page-2": {
            "data": [
                {
                    "model": "ANTHROPIC/TOOL-MODEL",
                    "availability_status": "available",
                    "vendors": {
                        "bedrock": {
                            "availability_status": "available",
                            "capabilities": {"supports_tool_calling": True},
                        }
                    },
                },
                {
                    "model": "openai/second-tool-model",
                    "availability_status": "available",
                    "vendors": {
                        "openai": {
                            "availability_status": "unavailable",
                            "capabilities": {"supports_tool_calling": True},
                        },
                        "azure": {
                            "availability_status": "available",
                            "capabilities": {"supports_tool_calling": True},
                        },
                    },
                },
            ],
            "has_more": False,
            "next_cursor": None,
        },
    }
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_open(req, timeout=0):
        requests.append((req, timeout))
        cursor = parse_qs(urlparse(req.full_url).query).get("cursor", [None])[0]
        return Response(pages[cursor])

    monkeypatch.setattr(
        "hermes_cli.urllib_security.open_credentialed_url", fake_open
    )

    models = merge_gateway_profile.fetch_models(
        api_key="merge-test-key", timeout=2.5
    )

    assert models == ["anthropic/tool-model", "openai/second-tool-model"]
    assert len(requests) == 2
    assert all(
        urlparse(req.full_url).path == "/v1/models" for req, _ in requests
    )
    assert all(
        parse_qs(urlparse(req.full_url).query)["limit"] == ["500"]
        for req, _ in requests
    )
    assert requests[1][0].get_header("Authorization") == "Bearer merge-test-key"
    assert requests[1][1] == 2.5


def test_fetch_models_uses_matching_catalog_for_custom_base(
    monkeypatch, merge_gateway_profile
):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data": [], "has_more": false, "next_cursor": null}'

    def fake_open(req, **_kwargs):
        seen["url"] = req.full_url
        return Response()

    monkeypatch.setattr(
        "hermes_cli.urllib_security.open_credentialed_url", fake_open
    )

    assert (
        merge_gateway_profile.fetch_models(
            api_key="merge-test-key",
            base_url="https://gateway.staging.example/v1/openai",
        )
        == []
    )
    assert urlparse(seen["url"]).path == "/v1/models"
    assert urlparse(seen["url"]).hostname == "gateway.staging.example"


def test_models_dev_metadata_resolves_through_merge_gateway_mapping(monkeypatch):
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda **_kwargs: {
            "merge-gateway": {
                "models": {
                    "anthropic/example-agent-model": {
                        "tool_call": True,
                        "attachment": True,
                        "reasoning": True,
                        "limit": {"context": 123456, "output": 8192},
                    }
                }
            }
        },
    )

    capabilities = get_model_capabilities(
        "merge-gateway", "anthropic/example-agent-model"
    )

    assert capabilities is not None
    assert capabilities.supports_tools is True
    assert capabilities.supports_vision is True
    assert capabilities.context_window == 123456
