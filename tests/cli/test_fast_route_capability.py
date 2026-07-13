"""Route-aware Fast/Priority capability contracts."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai-codex", "codex_responses"),
        ("anthropic", "anthropic_messages"),
        ("openai", "codex_responses"),
        ("openai-api", "codex_responses"),
        ("openrouter", "chat_completions"),
        (None, "chat_completions"),
    ],
)
def test_provider_only_api_mode_inference_contract(provider, expected):
    from hermes_cli.providers import infer_api_mode_from_provider

    assert infer_api_mode_from_provider(provider) == expected


@pytest.mark.parametrize(
    ("provider", "api_mode", "model", "supported", "family", "overrides"),
    [
        (
            "openai-api",
            "codex_responses",
            "gpt-5.6-sol",
            True,
            "openai_priority",
            {"service_tier": "priority"},
        ),
        (
            "openai-codex",
            "codex_responses",
            "gpt-5.6-sol",
            False,
            "codex_fast",
            {},
        ),
        (
            "openai-codex",
            "codex_responses",
            "gpt-5.5",
            True,
            "codex_fast",
            {"service_tier": "fast"},
        ),
        (
            "openai-codex",
            "codex_responses",
            "gpt-5.4-mini",
            False,
            "codex_fast",
            {},
        ),
        (
            "anthropic",
            "anthropic_messages",
            "claude-opus-4-6",
            False,
            "anthropic_fast",
            {},
        ),
        (
            "anthropic",
            "anthropic_messages",
            "claude-opus-4-8",
            True,
            "anthropic_fast",
            {"speed": "fast"},
        ),
        ("openrouter", "chat_completions", "gpt-5.6-sol", False, "unsupported", {}),
        ("custom:proxy", "codex_responses", "gpt-5.5", False, "unsupported", {}),
    ],
)
def test_fast_capability_is_route_specific(
    provider, api_mode, model, supported, family, overrides
):
    from hermes_cli.models import resolve_fast_mode_capability

    capability = resolve_fast_mode_capability(
        model=model,
        provider=provider,
        api_mode=api_mode,
    )

    assert capability.supported is supported
    assert capability.family == family
    assert capability.request_overrides == overrides


def test_opus_46_native_route_is_standard_speed_only():
    from hermes_cli.models import resolve_fast_mode_capability

    capability = resolve_fast_mode_capability(
        model="claude-opus-4-6",
        provider="anthropic",
        api_mode="anthropic_messages",
    )

    assert capability.supported is False
    assert capability.supported is False
    assert "not available" in capability.reason


def test_opus_48_proxy_route_stays_unsupported_but_keeps_separate_model_guidance():
    from hermes_cli.models import resolve_fast_mode_capability

    capability = resolve_fast_mode_capability(
        model="claude-opus-4-8",
        provider="claude-apr",
        api_mode="anthropic_messages",
    )

    assert capability.supported is False
    assert capability.family == "unsupported"
    assert capability.request_overrides == {}
    assert "claude-opus-4-8-fast" in capability.reason
    assert "speed=fast" in capability.reason


def test_fast_capability_catalog_entries_exist_in_provider_catalogs():
    from hermes_cli.fast_mode_contracts import OPENAI_PRIORITY_SOURCE_MODELS
    from hermes_cli.models import FAST_MODE_CAPABILITY_CATALOG, provider_model_ids

    catalog_provider = {
        "openai_priority": "openai-api",
        "codex_fast": "openai-codex",
        "anthropic_fast": "anthropic",
    }
    for family, contract in FAST_MODE_CAPABILITY_CATALOG.items():
        assert contract["source_url"].startswith("https://")
        assert contract["checked_date"] == "2026-07-12"
        assert "providers" not in contract
        catalog = set(provider_model_ids(catalog_provider[family]))
        assert set(contract["models"]) <= catalog

    # This is source-contract coverage, not merely agreement between two
    # Hermes-owned catalogs. Keep the official snapshot explicit and assert
    # the shipped contract is its exact intersection with Hermes inventory.
    official_snapshot = {
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
        "gpt-5.4-mini", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5",
        "gpt-5-mini", "gpt-5.1-codex", "gpt-5-codex", "gpt-4.1",
        "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o",
        "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
        "gpt-4o-mini", "o3", "o4-mini",
    }
    assert set(OPENAI_PRIORITY_SOURCE_MODELS) == official_snapshot
    hermes_openai = set(provider_model_ids("openai-api"))
    assert set(FAST_MODE_CAPABILITY_CATALOG["openai_priority"]["models"]) == (
        official_snapshot & hermes_openai
    )
    assert "gpt-5.5-pro" not in FAST_MODE_CAPABILITY_CATALOG["openai_priority"]["models"]
    assert "gpt-5.4-nano" not in FAST_MODE_CAPABILITY_CATALOG["openai_priority"]["models"]


def test_codex_fast_contract_retains_documented_request_field():
    from hermes_cli.models import (
        FAST_MODE_CAPABILITY_CATALOG,
        resolve_fast_mode_capability,
    )

    contract = FAST_MODE_CAPABILITY_CATALOG["codex_fast"]
    assert contract["source_url"] == "https://developers.openai.com/codex/speed"
    assert resolve_fast_mode_capability(
        model="gpt-5.5",
        provider="openai-codex",
        api_mode="codex_responses",
    ).request_overrides == {"service_tier": "fast"}


def test_anthropic_resolver_and_adapter_share_one_immutable_exact_catalog():
    from agent.anthropic_adapter import _supports_fast_mode
    from hermes_cli.fast_mode_contracts import FAST_MODE_CAPABILITY_CATALOG
    from hermes_cli.models import resolve_fast_mode_capability

    assert FAST_MODE_CAPABILITY_CATALOG["anthropic_fast"]["models"] == (
        "claude-opus-4-8",
    )
    for model, expected in (
        ("claude-opus-4-8", True),
        ("anthropic/claude-opus-4.8", True),
        ("claude-opus-4-7", False),
        ("claude-opus-4-80", False),
        ("claude-opus-4-8:fast", False),
        ("claude-opus-4-8-suffix", False),
        ("impostor/claude-opus-4-8", False),
    ):
        resolved = resolve_fast_mode_capability(
            model=model, provider="anthropic", api_mode="anthropic_messages"
        )
        assert resolved.supported is expected
        assert _supports_fast_mode(model) is expected


def test_request_enforcement_call_sites_do_not_use_model_only_wrapper():
    root = Path(__file__).resolve().parents[2]
    compatibility_definition = root / "hermes_cli" / "models.py"
    production_files = (
        path
        for path in root.rglob("*.py")
        if compatibility_definition != path
        and "tests" not in path.relative_to(root).parts
        and not any(
            part.startswith(".") or part in {"venv", "node_modules"}
            for part in path.relative_to(root).parts
        )
    )

    for path in production_files:
        source = path.read_text(encoding="utf-8")
        assert "resolve_fast_mode_overrides(" not in source, path
        assert "model_supports_fast_mode(" not in source, path
