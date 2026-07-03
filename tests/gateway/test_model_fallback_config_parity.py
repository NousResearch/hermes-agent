"""Non-runtime parity tests for gateway model and fallback config handling.

These tests use only in-memory/temp config data.  They do not launch the
messaging gateway, construct agents, resolve live providers, or read the real
profile config.
"""

from __future__ import annotations

import dataclasses
import importlib
import os

import pytest
import yaml

from gateway.config import GatewayConfig
from hermes_cli.fallback_config import get_fallback_chain


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point imports/helpers at an empty temp HERMES_HOME for this test."""

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _gateway_run_with_temp_home(isolated_hermes_home, monkeypatch):
    """Import gateway.run after HERMES_HOME is temp-scoped, then pin _hermes_home."""

    module = importlib.import_module("gateway.run")
    monkeypatch.setattr(module, "_hermes_home", isolated_hermes_home)
    return module


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model": "flat-string-model"}, "flat-string-model"),
        ({"model": {"default": "default-model"}}, "default-model"),
        ({"model": {"model": "legacy-model-key"}}, "legacy-model-key"),
        (
            {"model": {"default": "default-wins", "model": "legacy-loses"}},
            "default-wins",
        ),
        (
            {"model": {"default": "", "model": "legacy-fallback"}},
            "legacy-fallback",
        ),
        ({"model": {}}, ""),
        ({}, ""),
    ],
)
def test_gateway_model_resolution_shapes_and_precedence(
    isolated_hermes_home,
    monkeypatch,
    config,
    expected,
):
    gateway_run = _gateway_run_with_temp_home(isolated_hermes_home, monkeypatch)

    assert gateway_run._resolve_gateway_model(config) == expected


def test_gateway_model_resolution_reads_temp_config_only(isolated_hermes_home, monkeypatch):
    cfg_path = isolated_hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": {"default": "temp-config-model"}}),
        encoding="utf-8",
    )
    gateway_run = _gateway_run_with_temp_home(isolated_hermes_home, monkeypatch)

    assert gateway_run._resolve_gateway_model() == "temp-config-model"


def test_fallback_chain_merges_new_key_before_legacy_and_dedupes():
    config = {
        "fallback_providers": [
            {"provider": "openrouter", "model": "openai/gpt-5.5"},
            {
                "provider": "ollama",
                "model": "qwen3:30b",
                "base_url": " http://127.0.0.1:11434/v1/ ",
            },
        ],
        "fallback_model": [
            # Duplicate of the first primary entry: omitted.
            {"provider": "OPENROUTER", "model": "OpenAI/GPT-5.5"},
            # Distinct legacy entry: appended after fallback_providers.
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
    }

    chain = get_fallback_chain(config)

    assert [entry["provider"] for entry in chain] == [
        "openrouter",
        "ollama",
        "anthropic",
    ]
    assert [entry["model"] for entry in chain] == [
        "openai/gpt-5.5",
        "qwen3:30b",
        "claude-sonnet-4-6",
    ]
    assert chain[1]["base_url"] == "http://127.0.0.1:11434/v1"


@pytest.mark.parametrize(
    "legacy_value",
    [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        [{"provider": "openrouter", "model": "openai/gpt-5.5"}],
    ],
)
def test_fallback_chain_accepts_legacy_fallback_model_shapes(legacy_value):
    assert get_fallback_chain({"fallback_model": legacy_value}) == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"}
    ]


def test_fallback_chain_ignores_incomplete_or_non_mapping_entries():
    config = {
        "fallback_providers": [
            "not-a-dict",
            {"provider": "openrouter"},
            {"model": "missing-provider"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
        "fallback_model": "not-a-valid-legacy-shape",
    }

    assert get_fallback_chain(config) == [
        {"provider": "anthropic", "model": "claude-sonnet-4-6"}
    ]


def test_gateway_runner_load_fallback_model_reads_temp_config_only(
    isolated_hermes_home,
    monkeypatch,
):
    cfg_path = isolated_hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "fallback_providers": [
                    {"provider": "openrouter", "model": "openai/gpt-5.5"}
                ],
                "fallback_model": {"provider": "legacy", "model": "legacy-model"},
            }
        ),
        encoding="utf-8",
    )
    gateway_run = _gateway_run_with_temp_home(isolated_hermes_home, monkeypatch)

    assert gateway_run.GatewayRunner._load_fallback_model() == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "legacy", "model": "legacy-model"},
    ]


def test_gateway_config_contract_excludes_model_routing_fields():
    """GatewayConfig is platform/session config, not model routing config."""

    field_names = {field.name for field in dataclasses.fields(GatewayConfig)}
    forbidden_model_routing_fields = {
        "model",
        "provider",
        "fallback_model",
        "fallback_providers",
        "provider_routing",
        "api_key",
        "base_url",
    }

    assert field_names.isdisjoint(forbidden_model_routing_fields)
    cfg = GatewayConfig.from_dict(
        {
            "model": {"default": "should-not-become-gateway-field"},
            "fallback_providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.5"}
            ],
            "provider_routing": {"only": ["openrouter"]},
        }
    )
    as_dict = cfg.to_dict()
    assert as_dict.keys().isdisjoint(forbidden_model_routing_fields)


def test_this_test_module_does_not_pin_real_profile_home(
    isolated_hermes_home,
    monkeypatch,
):
    """Guard against accidentally using Nathan's real profile config path."""

    gateway_run = _gateway_run_with_temp_home(isolated_hermes_home, monkeypatch)

    assert os.environ["HERMES_HOME"] == str(isolated_hermes_home)
    assert str(isolated_hermes_home).startswith("/Users/neo/.hermes") is False
    assert gateway_run._hermes_home == isolated_hermes_home
