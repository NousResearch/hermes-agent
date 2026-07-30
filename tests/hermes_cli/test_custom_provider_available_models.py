"""Regression (gh-52266): ``available_models`` must declare provider models.

Hand-written custom-provider configs use ``available_models`` where Hermes'
own writer emits ``models``. The normalizer previously knew only ``models``,
so the alias hit the unknown-key warning and was dropped — leaving the entry
with no ``models`` at all. Downstream that means the picker lists the
provider with (0) models and every ``/model <id>`` against it fails to
resolve, because the routing paths all read the normalized ``models`` key.
"""

import copy

from hermes_cli.config import _normalize_custom_provider_entry

# The provider config from the issue report, verbatim in shape.
_ISSUE_ENTRY = {
    "name": "tokenplan",
    "api_mode": "chat_completions",
    "base_url": "https://token-plan.example/compatible-mode/v1",
    "model": "qwen3.7-plus",
    "available_models": [
        "qwen3.7-plus",
        "qwen3.7-max",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "kimi-k2.7-code",
        "glm-5.2",
    ],
}


def test_available_models_list_declares_models():
    out = _normalize_custom_provider_entry(_ISSUE_ENTRY, provider_key="tokenplan")
    assert out is not None
    assert sorted(out["models"]) == sorted(_ISSUE_ENTRY["available_models"])


def test_available_models_accepts_id_rows():
    entry = {
        "name": "p",
        "base_url": "https://x.example/v1",
        "available_models": [{"id": "m-1", "context_length": 8}, {"name": "m-2"}],
    }
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert out is not None
    assert out["models"] == {"m-1": {"context_length": 8}, "m-2": {}}


def test_models_metadata_wins_over_available_models():
    """Both keys may appear; the canonical ``models`` owns shared ids."""
    entry = {
        "name": "p",
        "base_url": "https://x.example/v1",
        "available_models": ["shared", "alias-only"],
        "models": {"shared": {"context_length": 128}, "models-only": {}},
    }
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert out is not None
    assert sorted(out["models"]) == ["alias-only", "models-only", "shared"]
    assert out["models"]["shared"] == {"context_length": 128}


def test_models_only_entry_is_unchanged():
    entry = {
        "name": "p",
        "base_url": "https://x.example/v1",
        "models": {"only": {"context_length": 4}},
    }
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert out is not None
    assert out["models"] == {"only": {"context_length": 4}}


def test_no_declaration_leaves_models_unset():
    entry = {"name": "p", "base_url": "https://x.example/v1"}
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert out is not None
    assert "models" not in out


def test_available_models_does_not_mutate_input():
    """Entries alias the shared read-only config cache — see
    test_custom_provider_normalize_no_mutate.py."""
    entry = copy.deepcopy(_ISSUE_ENTRY)
    snapshot = copy.deepcopy(entry)
    _normalize_custom_provider_entry(entry, provider_key="tokenplan")
    assert entry == snapshot
