"""Behavior contracts for Databricks AI Gateway catalog discovery."""

from __future__ import annotations

import os
import sys

from agent.anthropic_endpoints import _requires_bearer_auth
from hermes_cli import model_switch_providers as providers
from hermes_cli import models


ANTHROPIC = "anthropic/v1/messages"
OPENAI = "openai/v1/responses"
MLFLOW = "mlflow/v1/chat/completions"
SERVING_PATH = "/api/2.0/serving-endpoints"
GATEWAY_PATH = "/api/ai-gateway/v2/endpoints?page_size=100"
UNITY_PATH = "/api/2.1/unity-catalog/model-services?view=FULL&page_size=100"


def _serving_endpoint(name: str, api_types: list[str]) -> dict:
    return {
        "name": name,
        "state": {"ready": "READY"},
        "capabilities": {"function_calling": True},
        "config": {
            "served_entities": [
                {
                    "foundation_model": {
                        "ai_gateway_v2_supported": True,
                        "api_types": api_types,
                    }
                }
            ]
        },
    }


def _gateway_alias(name: str, target: str, kind: str) -> dict:
    return {
        "name": name,
        "config": {
            "destinations": [
                {
                    "name": target,
                    "type": f"DESTINATION_TYPE_{kind}",
                    "traffic_percentage": 100,
                }
            ]
        },
    }


def test_catalog_relationships_place_each_endpoint_and_alias_on_one_surface(
    monkeypatch,
    caplog,
):
    workspace = "https://dbc-example.cloud.databricks.com"
    routes = {
        SERVING_PATH: {
            "endpoints": [
                _serving_endpoint("chat-anthropic", [MLFLOW, ANTHROPIC]),
                _serving_endpoint("chat-openai", [MLFLOW, OPENAI]),
                _serving_endpoint("chat-mlflow", [MLFLOW]),
            ]
        },
        GATEWAY_PATH: {
            "endpoints": [
                _gateway_alias(
                    "friendly-openai",
                    "system.ai.chat-openai",
                    "PAY_PER_TOKEN_FOUNDATION_MODEL",
                )
            ],
            "next_page_token": "second page",
        },
        GATEWAY_PATH + "&page_token=second%20page": {
            "endpoints": [
                _gateway_alias(
                    "external-openai",
                    "external-destination",
                    "EXTERNAL_FOUNDATION_MODEL",
                )
            ],
            "next_page_token": "second page",
        },
        "/api/ai-gateway/v2/endpoints/external-openai": {
            "supported_api_types": [MLFLOW, OPENAI]
        },
        UNITY_PATH: {
            "model_services": [
                {
                    "name": "model-services/catalog.schema.external-anthropic",
                    "config": {
                        "routing": {
                            "destinations": [
                                {
                                    "destination_type": "DESTINATION_TYPE_EXTERNAL_FOUNDATION_MODEL",
                                    "traffic_percentage": 100,
                                    "external_model_config": {
                                        "target": {
                                            "native_api_types": [MLFLOW, ANTHROPIC]
                                        }
                                    },
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    hits: list[str] = []

    def fake_get_json(url, *, headers, **_kwargs):
        assert headers["Authorization"] == "Bearer example-token"
        path = url.removeprefix(workspace)
        hits.append(path)
        return routes[path]

    monkeypatch.setattr(models, "_get_json", fake_get_json)
    monkeypatch.setattr(models, "_custom_provider_ssl_context", lambda _url: None)

    by_surface = {
        surface: set(
            models.probe_api_models(
                "example-token", f"{workspace}/ai-gateway/{surface}/v1"
            )["models"]
        )
        for surface in ("anthropic", "openai", "mlflow")
    }

    assert by_surface == {
        "anthropic": {"chat-anthropic", "catalog.schema.external-anthropic"},
        "openai": {"chat-openai", "friendly-openai", "external-openai"},
        "mlflow": {"chat-mlflow"},
    }
    all_models = [
        model for surface_models in by_surface.values() for model in surface_models
    ]
    assert len(all_models) == len(set(all_models))
    assert GATEWAY_PATH + "&page_token=second%20page" in hits
    assert "/api/ai-gateway/v2/endpoints/external-openai" in hits
    assert "repeated page token" in caplog.text
    assert _requires_bearer_auth(f"{workspace}/ai-gateway/anthropic")


def test_picker_materializes_credentials_before_empty_serving_catalog_alias_discovery(
    monkeypatch,
):
    cases = [
        ({"api_key": "${GATEWAY_TOKEN}"}, "dbc-env.cloud.databricks.com", "env-token"),
        (
            {"key_cmd": f'"{sys.executable}" -c "print(\'command-token\')"'},
            "dbc-command.cloud.databricks.com",
            "command-token",
        ),
    ]
    monkeypatch.setenv("GATEWAY_TOKEN", "env-token")
    monkeypatch.setattr(models, "_custom_provider_ssl_context", lambda _url: None)
    monkeypatch.setattr(
        models,
        "cached_fetch_api_models",
        lambda *_args, fetch_models, **_kwargs: fetch_models(),
    )

    for credential_config, workspace, expected_token in cases:
        root = f"https://{workspace}"
        gateway_url = f"{root}/ai-gateway/openai/v1"
        seen_tokens: list[str] = []

        def fake_get_json(url, *, headers, **_kwargs):
            seen_tokens.append(headers["Authorization"].removeprefix("Bearer "))
            path = url.removeprefix(root)
            if path == SERVING_PATH:
                return {"endpoints": []}
            if path == GATEWAY_PATH:
                alias = _gateway_alias(
                    "external-only",
                    "external-destination",
                    "EXTERNAL_FOUNDATION_MODEL",
                )
                alias["supported_api_types"] = [OPENAI]
                return {"endpoints": [alias]}
            if path == UNITY_PATH:
                return {"model_services": []}
            raise AssertionError(f"unexpected catalog path: {path}")

        monkeypatch.setattr(models, "_get_json", fake_get_json)
        inline_key, key_env, _identity = providers._entry_credentials(
            {"name": "Databricks AI Gateway", **credential_config}, "key_env"
        )
        probe_key = providers._resolve_probe_key(inline_key, key_env, os.environ.get)
        result = providers._fetch_picker_live_models(
            probe_key,
            gateway_url,
            "custom",
            False,
            cache=False,
        )

        assert result == ["external-only"]
        assert seen_tokens and set(seen_tokens) == {expected_token}
