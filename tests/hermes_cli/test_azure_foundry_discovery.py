"""Azure Foundry deployment discovery + per-model route selection.

One Azure Foundry resource hosts Claude (Anthropic Messages route) and GPT
(OpenAI Responses / chat route) deployments side by side. The picker must list
the resource's *deployments* (not Microsoft's whole catalog, not nothing), and
the runtime must move the configured base URL onto the route each model needs.
"""

from __future__ import annotations

import io
import json

import pytest

from hermes_cli import models as models_mod
from hermes_cli.models import (
    azure_foundry_base_url_for_mode,
    azure_foundry_model_api_mode,
    azure_foundry_resource_root,
    fetch_azure_foundry_deployments,
)

SERVICES = "https://res.services.ai.azure.com"
OPENAI_HOST = "https://res.openai.azure.com"


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-sonnet-4-6", "anthropic_messages"),
        ("claude-opus-4-8-anthropic", "anthropic_messages"),
        ("anthropic/claude-haiku-4-5", "anthropic_messages"),
        ("gpt-5.3-codex", "codex_responses"),
        ("o3-mini", "codex_responses"),
        ("gpt-4o", None),
        ("Llama-3.3-70B-Instruct", None),
        ("", None),
    ],
)
def test_api_mode_inferred_from_deployment_name(model, expected):
    assert azure_foundry_model_api_mode(model) == expected


@pytest.mark.parametrize(
    "url, root",
    [
        (f"{SERVICES}/anthropic", SERVICES),
        (f"{SERVICES}/anthropic/v1/", SERVICES),
        (f"{OPENAI_HOST}/openai/v1", OPENAI_HOST),
        (f"{OPENAI_HOST}/openai", OPENAI_HOST),
        (f"{SERVICES}/models", SERVICES),
        ("https://gw.example.com/azure/openai/v1", "https://gw.example.com/azure"),
        ("", ""),
    ],
)
def test_resource_root_strips_route_suffix(url, root):
    assert azure_foundry_resource_root(url) == root


def test_base_url_rewritten_per_mode_on_azure_hosts():
    anth = f"{SERVICES}/anthropic"
    assert azure_foundry_base_url_for_mode(anth, "codex_responses") == f"{SERVICES}/openai/v1"
    assert azure_foundry_base_url_for_mode(anth, "chat_completions") == f"{SERVICES}/openai/v1"
    assert azure_foundry_base_url_for_mode(anth, "anthropic_messages") == anth
    oai = f"{OPENAI_HOST}/openai/v1"
    assert azure_foundry_base_url_for_mode(oai, "anthropic_messages") == f"{OPENAI_HOST}/anthropic"
    assert azure_foundry_base_url_for_mode(oai, "codex_responses") == oai


def test_base_url_untouched_for_non_azure_gateways():
    for url in ("http://localhost:4001/v1", "https://litellm.corp/anthropic"):
        for mode in ("anthropic_messages", "codex_responses", "chat_completions"):
            assert azure_foundry_base_url_for_mode(url, mode) == url


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_deployments_lists_resource_scoped_endpoint(monkeypatch):
    seen = {}

    def fake_open(req, *, timeout, ssl_context=None):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        body = {
            "data": [
                {"id": "claude-sonnet-4-6", "status": "succeeded"},
                {"id": "gpt-5.3-codex", "status": "succeeded"},
                {"id": "claude-opus-5", "status": "creating"},
                {"id": "", "status": "succeeded"},
                "garbage",
            ]
        }
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", fake_open)
    out = fetch_azure_foundry_deployments("k3y", f"{SERVICES}/anthropic")

    assert out == ["claude-sonnet-4-6", "gpt-5.3-codex", "claude-opus-5"]
    assert seen["url"].startswith(f"{SERVICES}/openai/deployments?api-version=")
    assert seen["headers"]["api-key"] == "k3y"


def test_fetch_deployments_returns_none_on_failure_or_missing_inputs(monkeypatch):
    def boom(req, *, timeout, ssl_context=None):
        raise OSError("down")

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", boom)
    assert fetch_azure_foundry_deployments("k", f"{SERVICES}/anthropic") is None
    assert fetch_azure_foundry_deployments("", f"{SERVICES}/anthropic") is None
    assert fetch_azure_foundry_deployments("k", "") is None


def test_provider_model_ids_uses_deployments_and_keeps_default(monkeypatch):
    monkeypatch.setattr(
        models_mod,
        "_get_model_config_dict",
        lambda: {"provider": "azure-foundry", "base_url": f"{SERVICES}/anthropic", "default": "claude-custom"},
    )
    monkeypatch.setattr(
        models_mod,
        "fetch_azure_foundry_deployments",
        lambda key, url, timeout=5.0: ["claude-sonnet-4-6", "gpt-5.3-codex"],
    )
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "get_env_value", lambda k: "secret" if k == "AZURE_FOUNDRY_API_KEY" else "")
    assert models_mod.provider_model_ids("azure-foundry") == [
        "claude-custom",
        "claude-sonnet-4-6",
        "gpt-5.3-codex",
    ]


def test_provider_model_ids_empty_when_probe_fails(monkeypatch):
    monkeypatch.setattr(models_mod, "_get_model_config_dict", lambda: {"provider": "anthropic"})
    monkeypatch.setattr(models_mod, "fetch_azure_foundry_deployments", lambda *a, **k: None)
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "get_env_value", lambda k: "x")
    assert models_mod.provider_model_ids("azure-foundry") == []


def test_runtime_routes_each_model_to_its_azure_route(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "_getenv", lambda name, default="": "k3y" if name == "AZURE_FOUNDRY_API_KEY" else default)
    model_cfg = {
        "provider": "azure-foundry",
        "default": "claude-sonnet-4-6",
        "base_url": f"{SERVICES}/anthropic",
        "auth_mode": "api_key",
    }

    claude = rp._resolve_azure_foundry_runtime(
        requested_provider="azure-foundry", model_cfg=model_cfg, target_model="claude-opus-5"
    )
    assert claude["api_mode"] == "anthropic_messages"
    assert claude["base_url"] == f"{SERVICES}/anthropic"

    codex = rp._resolve_azure_foundry_runtime(
        requested_provider="azure-foundry", model_cfg=model_cfg, target_model="gpt-5.3-codex"
    )
    assert codex["api_mode"] == "codex_responses"
    assert codex["base_url"] == f"{SERVICES}/openai/v1"

    # Config pointing at the OpenAI route still serves Claude on /anthropic.
    model_cfg["base_url"] = f"{OPENAI_HOST}/openai/v1"
    claude2 = rp._resolve_azure_foundry_runtime(
        requested_provider="azure-foundry", model_cfg=model_cfg, target_model="claude-haiku-4-5"
    )
    assert claude2["base_url"] == f"{OPENAI_HOST}/anthropic"

    # Explicit --base-url overrides are honoured verbatim (no route rewrite;
    # only the pre-existing trailing-/v1 strip for the Anthropic SDK applies).
    forced = rp._resolve_azure_foundry_runtime(
        requested_provider="azure-foundry",
        model_cfg=model_cfg,
        explicit_base_url="https://gw.corp/azure/openai/v1",
        target_model="claude-haiku-4-5",
    )
    assert forced["base_url"] == "https://gw.corp/azure/openai"
