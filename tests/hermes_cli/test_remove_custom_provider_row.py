"""The interactive "Remove a saved custom provider" row must see both config schemas.

Regression cover for the reported defect: a provider saved in the v12+ ``providers`` map
was invisible to the removal row, which only ever read the legacy ``custom_providers``
list — so removal silently appeared to work while the entry survived.
"""

from __future__ import annotations

from hermes_cli.main_provider_setup import _removable_custom_providers


def _migrated_config():
    return {
        "model": {"provider": "other"},
        "providers": {
            "migrated": {"api": "https://migrated.ai/v1", "key_env": "MIGRATED_API_KEY"},
            "other": {"api": "https://api.other.ai/v1", "key_env": "OTHER_API_KEY"},
        },
    }


def _legacy_config():
    return {
        "model": {"provider": "other"},
        "custom_providers": [{"name": "Legacy.ai", "base_url": "https://legacy.ai/v1"}],
    }


def test_migrated_providers_entry_is_removable():
    names = {e.get("name") or e.get("provider_key") for e in _removable_custom_providers(_migrated_config())}
    assert "migrated" in names


def test_legacy_entry_is_removable():
    names = {e.get("name") or e.get("provider_key") for e in _removable_custom_providers(_legacy_config())}
    assert "Legacy.ai" in names


def test_removable_list_is_empty_without_custom_providers():
    assert _removable_custom_providers({"model": {"provider": "openai"}}) == []


def test_removable_entries_are_all_dicts():
    for config in (_migrated_config(), _legacy_config()):
        assert all(isinstance(entry, dict) for entry in _removable_custom_providers(config))
