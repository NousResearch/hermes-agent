"""Tests for ``hermes provider remove`` and the shared provider-removal helpers."""

from __future__ import annotations

import pytest

from hermes_cli.provider_removal import (
    build_removal_plan,
    execute_removal,
    format_plan,
)


def _config(**overrides):
    config = {
        "model": {"default": "m1", "provider": "cerebras", "base_url": "https://api.cerebras.ai/v1"},
        "providers": {
            "cerebras": {"api": "https://api.cerebras.ai/v1", "key_env": "CEREBRAS_API_KEY"},
            "other": {"api": "https://api.other.ai/v1", "key_env": "OTHER_API_KEY"},
        },
    }
    config.update(overrides)
    return config


def test_plan_finds_migrated_providers_entry():
    """The #5525 case: a v12+ providers entry must be found, not just the legacy list."""
    plan = build_removal_plan(_config(), "cerebras")
    assert plan.found is True
    assert plan.providers_key == "cerebras"
    assert plan.legacy_index is None
    assert "providers" in plan.stores


def test_plan_finds_legacy_custom_providers_entry():
    config = _config(custom_providers=[{"name": "Yunwu.ai", "base_url": "https://yunwu.ai/v1"}])
    plan = build_removal_plan(config, "Yunwu.ai")
    assert plan.found is True
    assert plan.legacy_index == 0
    assert plan.providers_key is None
    assert "custom_providers" in plan.stores


def test_plan_is_case_insensitive():
    assert build_removal_plan(_config(), "CEREBRAS").found is True


def test_plan_collects_every_reference_path():
    config = _config(
        fallback_providers=[{"provider": "cerebras", "model": "m2"}, {"provider": "other", "model": "m3"}],
        auxiliary={"vision": {"provider": "cerebras", "model": "m"}, "compression": {"provider": "other"}},
        moa={
            "aggregator": {"provider": "cerebras"},
            "reference_models": [{"provider": "other"}, {"provider": "cerebras"}],
        },
    )
    plan = build_removal_plan(config, "cerebras")
    paths = sorted(ref.path for ref in plan.references)
    assert paths == [
        "auxiliary.vision.provider",
        "fallback_providers[0].provider",
        "moa.aggregator.provider",
        "moa.reference_models[1].provider",
        "model.provider",
    ]


def test_references_match_legacy_custom_prefix():
    config = _config(model={"provider": "custom:cerebras"})
    plan = build_removal_plan(config, "cerebras")
    assert [ref.path for ref in plan.references] == ["model.provider"]


def test_dry_run_reports_without_mutating():
    config = _config(auxiliary={"vision": {"provider": "cerebras"}})
    plan = build_removal_plan(config, "cerebras")
    lines = format_plan(plan, dry_run=True)
    assert any("Would remove" in line for line in lines)
    assert any("auxiliary.vision.provider" in line for line in lines)
    # Nothing was written.
    assert "cerebras" in config["providers"]


def test_removal_purges_definition_and_clears_references():
    config = _config(
        fallback_providers=[{"provider": "cerebras"}],
        auxiliary={"vision": {"provider": "cerebras"}},
    )
    plan = build_removal_plan(config, "cerebras")
    result = execute_removal(
        config, plan,
        reassign={"model.provider": "other"},
        clear=["fallback_providers[0].provider", "auxiliary.vision.provider"],
        keep_credentials=True,
    )
    assert "cerebras" not in config["providers"]
    assert "other" in config["providers"]
    assert config["model"]["provider"] == "other"
    assert config["fallback_providers"] == []
    assert config["auxiliary"]["vision"]["provider"] == ""
    assert result["stores"] == ["providers"]


def test_clearing_model_provider_drops_base_url():
    config = _config()
    plan = build_removal_plan(config, "cerebras")
    execute_removal(config, plan, clear=["model.provider"], keep_credentials=True)
    assert config["model"]["provider"] == ""
    assert "base_url" not in config["model"]


def test_reassigning_model_provider_drops_stale_base_url():
    """A stale base_url would send every request to the removed provider's host."""
    config = _config()
    plan = build_removal_plan(config, "cerebras")
    execute_removal(
        config, plan, reassign={"model.provider": "other"}, keep_credentials=True)
    assert config["model"]["provider"] == "other"
    assert "base_url" not in config["model"]


def test_legacy_list_entry_is_removed_from_owning_schema():
    config = _config(custom_providers=[
        {"name": "keep.ai", "base_url": "https://keep.ai/v1"},
        {"name": "drop.ai", "base_url": "https://drop.ai/v1"},
    ])
    plan = build_removal_plan(config, "drop.ai")
    result = execute_removal(config, plan, keep_credentials=True)
    assert [entry["name"] for entry in config["custom_providers"]] == ["keep.ai"]
    assert result["stores"] == ["custom_providers"]


def test_empty_legacy_list_is_dropped_entirely():
    config = _config(custom_providers=[{"name": "solo.ai", "base_url": "https://solo.ai/v1"}])
    plan = build_removal_plan(config, "solo.ai")
    execute_removal(config, plan, keep_credentials=True)
    assert "custom_providers" not in config


def test_unresolved_references_are_kept_and_reported():
    config = _config(auxiliary={"vision": {"provider": "cerebras"}})
    plan = build_removal_plan(config, "cerebras")
    result = execute_removal(config, plan, keep_credentials=True)
    # Both the active model provider and the auxiliary slot still point at it.
    assert result["unresolved"] == ["model.provider", "auxiliary.vision.provider"]
    assert config["auxiliary"]["vision"]["provider"] == "cerebras"
    assert config["model"]["provider"] == "cerebras"


def test_reassigned_reference_reports_target():
    config = _config(auxiliary={"vision": {"provider": "cerebras"}})
    plan = build_removal_plan(config, "cerebras")
    result = execute_removal(
        config, plan, reassign={"auxiliary.vision.provider": "other"}, keep_credentials=True)
    assert any("-> other" in line for line in result["references"])


def test_plan_reports_env_var_name_only():
    plan = build_removal_plan(_config(), "cerebras")
    assert plan.exposed_env_vars == ["CEREBRAS_API_KEY"]


def test_unknown_provider_is_not_found():
    plan = build_removal_plan(_config(), "nonexistent")
    assert plan.found is False
    assert plan.stores == []
    assert plan.references == []
