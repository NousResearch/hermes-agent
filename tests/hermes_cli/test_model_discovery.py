"""Behavior contracts for noninteractive provider-scoped model discovery."""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.model_discovery import (
    ModelDiscoveryRequest,
    _refresh_provider_models,
    discover_models,
    models_command,
)
from hermes_cli.subcommands.models import build_models_parser


def _cache_entry(models, *, age=0):
    return {"fp": "fp", "at": time.time() - age, "models": list(models)}


def test_default_discovery_is_network_free_and_prefers_fresh_cache():
    request = ModelDiscoveryRequest(provider="openai-api")
    with (
        patch("hermes_cli.model_discovery._registered_provider_ids", return_value=["openai-api"]),
        patch("hermes_cli.model_discovery._provider_cache_entry", return_value=_cache_entry(["cached-model"])),
        patch("hermes_cli.model_discovery._catalog_model_ids", return_value=["catalog-model"]),
        patch("hermes_cli.model_discovery._refresh_provider_models") as refresh,
    ):
        response, exit_code = discover_models(request)

    assert exit_code == 0
    assert response["providers"][0]["source"] == "cache"
    assert [model["id"] for model in response["providers"][0]["models"]] == ["cached-model"]
    refresh.assert_not_called()


def test_offline_discovery_serves_stale_cache_without_refresh():
    request = ModelDiscoveryRequest(provider="openai-api", offline=True)
    with (
        patch("hermes_cli.model_discovery._registered_provider_ids", return_value=["openai-api"]),
        patch(
            "hermes_cli.model_discovery._provider_cache_entry",
            return_value=_cache_entry(["stale-model"], age=7200),
        ),
        patch("hermes_cli.model_discovery._refresh_provider_models") as refresh,
    ):
        response, exit_code = discover_models(request)

    provider = response["providers"][0]
    assert exit_code == 0
    assert provider["source"] == "cache"
    assert provider["stale"] is True
    assert provider["warnings"][0]["code"] == "stale_cache"
    refresh.assert_not_called()


def test_catalog_fallback_has_required_public_model_fields():
    request = ModelDiscoveryRequest(provider="openai-api", offline=True)
    with (
        patch("hermes_cli.model_discovery._registered_provider_ids", return_value=["openai-api"]),
        patch("hermes_cli.model_discovery._provider_cache_entry", return_value=None),
        patch("hermes_cli.model_discovery._catalog_model_ids", return_value=["catalog-model"]),
        patch("agent.models_dev.get_model_info", return_value=None),
    ):
        response, exit_code = discover_models(request)

    provider = response["providers"][0]
    assert exit_code == 0
    assert provider["source"] == "catalog"
    assert provider["warnings"] == []
    assert provider["models"] == [{
        "id": "catalog-model",
        "capabilities": ["chat"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "deprecated": False,
    }]


def test_live_refresh_uses_one_registered_profile_and_updates_one_cache_entry():
    profile = SimpleNamespace(
        auth_type="api_key",
        base_url="https://public.provider.example/v1",
        fetch_models=lambda **kwargs: ["live-a", "live-a", "live-b"],
    )
    with (
        patch("providers.get_provider_profile", return_value=profile) as get_profile,
        patch("hermes_cli.models._api_key_credentials", return_value=("secret", profile.base_url)),
        patch("hermes_cli.models.update_provider_cache_entry") as update_cache,
    ):
        assert _refresh_provider_models("openai-api") == ["live-a", "live-b"]

    get_profile.assert_called_once_with("openai-api")
    update_cache.assert_called_once_with("openai-api", ["live-a", "live-b"])


def test_refresh_is_scoped_and_falls_back_without_leaking_exception_details():
    request = ModelDiscoveryRequest(provider="openai-api", refresh=True)
    with (
        patch("hermes_cli.model_discovery._registered_provider_ids", return_value=["openai-api", "anthropic"]),
        patch(
            "hermes_cli.model_discovery._refresh_provider_models",
            side_effect=RuntimeError("Bearer secret-value at https://private.example/v1/models"),
        ) as refresh,
        patch(
            "hermes_cli.model_discovery._provider_cache_entry",
            return_value=_cache_entry(["cached-model"], age=7200),
        ),
    ):
        response, exit_code = discover_models(request)

    assert exit_code == 0
    refresh.assert_called_once_with("openai-api")
    assert response["providers"][0]["source"] == "cache"
    rendered = json.dumps(response)
    assert "secret-value" not in rendered
    assert "private.example" not in rendered
    assert response["errors"] == [{
        "code": "refresh_failed",
        "provider": "openai-api",
        "message": "Live model discovery failed; a fallback source was used.",
    }]


def test_argument_errors_are_structured_json(capsys):
    args = SimpleNamespace(json=True, provider=None, refresh=True, offline=False)
    assert models_command(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"][0]["code"] == "provider_required"

    args = SimpleNamespace(json=True, provider="openai-api", refresh=True, offline=True)
    assert models_command(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"][0]["code"] == "conflicting_options"


def test_unknown_provider_is_rejected_before_discovery(capsys):
    args = SimpleNamespace(json=True, provider="not-registered", refresh=False, offline=True)
    with patch("hermes_cli.model_discovery._registered_provider_ids", return_value=["openai-api"]):
        assert models_command(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["providers"] == []
    assert payload["errors"][0]["code"] == "unsupported_provider"


def test_models_parser_wires_machine_readable_flags():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_models_parser(subparsers, cmd_models=lambda args: 0)

    args = parser.parse_args(["models", "--json", "--provider", "openai-api", "--offline"])
    assert args.command == "models"
    assert args.provider == "openai-api"
    assert args.json is True
    assert args.offline is True
    assert args.refresh is False
