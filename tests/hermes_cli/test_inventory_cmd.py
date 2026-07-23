"""Tests for the non-interactive provider/model inventory CLI."""

from __future__ import annotations

import argparse
import json

import pytest

import hermes_cli.inventory_cmd as inventory_cmd
from hermes_cli.inventory import ConfigContext, build_offline_models_payload


def _handler(_args):  # pragma: no cover - parser identity only
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    inventory_cmd.build_inventory_parsers(
        parser.add_subparsers(dest="command"),
        cmd_models=_handler,
        cmd_providers=_handler,
    )
    return parser


def test_inventory_parsers_accept_scriptable_offline_commands():
    parser = _parser()

    models = parser.parse_args(["models", "list", "--provider", "openrouter", "--offline", "--json"])
    assert models.func is _handler
    assert models.models_action == "list"
    assert models.offline is True

    providers = parser.parse_args(["providers", "list", "--all", "--no-live"])
    assert providers.func is _handler
    assert providers.providers_action == "list"
    assert providers.offline is True


def test_status_disallows_combining_probe_with_offline():
    with pytest.raises(SystemExit):
        _parser().parse_args(["models", "status", "--offline", "--probe"])


def test_offline_inventory_uses_config_and_bundled_catalogs_only():
    context = ConfigContext(
        current_provider="openrouter",
        current_model="openai/gpt-5.6-sol",
        current_base_url="",
        user_providers={
            "local": {
                "name": "Local gateway",
                "models": {"qwen-local": {}},
            }
        },
        custom_providers=[
            {
                "name": "Legacy endpoint",
                "base_url": "http://localhost:8000/v1",
                "models": ["legacy-model"],
            }
        ],
    )

    configured = build_offline_models_payload(context)
    by_id = {row["slug"]: row for row in configured["providers"]}
    assert {"openrouter", "local", "custom:legacy-endpoint"}.issubset(by_id)
    assert by_id["openrouter"]["is_current"] is True
    assert "openai/gpt-5.6-sol" in by_id["openrouter"]["models"]
    assert by_id["local"]["models"] == ["qwen-local"]
    assert by_id["custom:legacy-endpoint"]["models"] == ["legacy-model"]

    all_providers = build_offline_models_payload(context, include_unconfigured=True)
    assert any(row["slug"] == "anthropic" for row in all_providers["providers"])


@pytest.mark.parametrize(
    "legacy_override",
    [
        {"key_env": "LEGACY_API_KEY"},
        {"api_mode": "responses"},
        {"extra_headers": {"X-Tenant": "legacy"}},
    ],
    ids=["credential", "api-mode", "headers"],
)
def test_offline_inventory_keeps_legacy_entries_with_distinct_endpoint_identity(
    legacy_override,
):
    shared_url = "https://gateway.example.test/v1"
    legacy_entry = {
        "name": "Legacy endpoint",
        "base_url": shared_url,
        "key_env": "KEYED_API_KEY",
        "api_mode": "chat_completions",
        "extra_headers": {"X-Tenant": "keyed"},
        "models": ["legacy-model"],
    }
    legacy_entry.update(legacy_override)
    context = ConfigContext(
        current_provider="",
        current_model="",
        current_base_url="",
        user_providers={
            "configured": {
                "name": "Configured endpoint",
                "base_url": shared_url,
                "key_env": "KEYED_API_KEY",
                "api_mode": "chat_completions",
                "extra_headers": {"X-Tenant": "keyed"},
                "models": ["configured-model"],
            }
        },
        custom_providers=[legacy_entry],
    )

    by_id = {row["slug"]: row for row in build_offline_models_payload(context)["providers"]}

    assert by_id["configured"]["models"] == ["configured-model"]
    assert by_id["custom:legacy-endpoint"]["models"] == ["legacy-model"]


def test_models_list_json_flattens_the_shared_provider_payload(monkeypatch, capsys):
    monkeypatch.setattr(
        inventory_cmd,
        "_inventory_payload",
        lambda _args, probe=False: {
            "model": "model-b",
            "providers": [
                {
                    "id": "example",
                    "name": "Example",
                    "auth_state": "configured",
                    "auth_type": "api_key",
                    "env_vars": ["EXAMPLE_API_KEY"],
                    "is_current": True,
                    "models": ["model-a", "model-b"],
                    "model_count": 2,
                    "source": "static",
                }
            ],
        },
    )
    args = argparse.Namespace(
        models_action="list",
        offline=True,
        probe=False,
        provider=None,
        json=True,
    )

    inventory_cmd.cmd_models(args)

    document = json.loads(capsys.readouterr().out)
    assert document["offline"] is True
    assert document["models"] == [
        {
            "id": "model-a",
            "is_current": False,
            "provider": "example",
            "provider_name": "Example",
        },
        {
            "id": "model-b",
            "is_current": True,
            "provider": "example",
            "provider_name": "Example",
        },
    ]
