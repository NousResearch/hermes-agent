"""Unit tests for hermes_cli.model_profiles."""

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli import model_profiles as mp


def _base_model_config():
    return {
        "default": "anthropic/claude-opus-4.6",
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "profiles": [
            {
                "name": mp.DEFAULT_PROFILE_NAME,
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.6",
                "base_url": OPENROUTER_BASE_URL,
                "enabled": True,
            }
        ],
        "active_profile": mp.DEFAULT_PROFILE_NAME,
        "scoped_profiles": [mp.DEFAULT_PROFILE_NAME],
    }


def test_normalize_model_config_migrates_legacy_string():
    cfg = mp.normalize_model_config("  openai/gpt-5.2  ")

    assert cfg["default"] == "openai/gpt-5.2"
    assert cfg["provider"] == "openrouter"
    assert cfg["base_url"] == OPENROUTER_BASE_URL
    assert cfg["active_profile"] == mp.DEFAULT_PROFILE_NAME
    assert cfg["scoped_profiles"] == [mp.DEFAULT_PROFILE_NAME]
    assert cfg["profiles"] == [
        {
            "name": mp.DEFAULT_PROFILE_NAME,
            "provider": "openrouter",
            "model": "openai/gpt-5.2",
            "base_url": OPENROUTER_BASE_URL,
            "enabled": True,
        }
    ]


def test_normalize_model_config_migrates_legacy_fields_and_syncs_active_profile():
    cfg = mp.normalize_model_config(
        {
            "default": "glm-5",
            "provider": " GLM ",
            "base_url": " https://endpoint.example/v1/ ",
        }
    )

    profile = cfg["profiles"][0]
    assert profile["name"] == mp.DEFAULT_PROFILE_NAME
    assert profile["provider"] == "zai"
    assert profile["model"] == "glm-5"
    assert profile["base_url"] == "https://endpoint.example/v1"
    assert cfg["active_profile"] == mp.DEFAULT_PROFILE_NAME
    assert cfg["scoped_profiles"] == [mp.DEFAULT_PROFILE_NAME]
    assert cfg["provider"] == "zai"
    assert cfg["default"] == "glm-5"
    assert cfg["base_url"] == "https://endpoint.example/v1"


def test_normalize_model_config_repairs_duplicate_profile_names_and_scoped_profiles():
    cfg = mp.normalize_model_config(
        {
            "profiles": [
                {"name": "work", "provider": "openrouter", "model": "model-a", "enabled": True},
                {"name": "work", "provider": "kimi", "model": "k2p5", "enabled": True},
            ],
            "active_profile": "missing-profile",
            "scoped_profiles": ["work", "missing-profile", " work-2 "],
        }
    )

    assert [p["name"] for p in cfg["profiles"]] == ["work", "work-2"]
    assert cfg["profiles"][1]["provider"] == "kimi-coding"
    assert cfg["active_profile"] == "work"
    assert cfg["scoped_profiles"] == ["work", "work-2"]
    assert cfg["default"] == "model-a"
    assert cfg["provider"] == "openrouter"


def test_sync_legacy_model_fields_uses_provider_resolver_when_active_base_url_missing(monkeypatch):
    monkeypatch.setattr(mp, "resolve_provider_base_url", lambda provider: f"https://{provider}.example/v1")
    synced = mp.sync_legacy_model_fields(
        {
            "default": "old-model",
            "provider": "openrouter",
            "base_url": None,
            "profiles": [
                {
                    "name": "zai-profile",
                    "provider": "zai",
                    "model": "glm-4.6",
                    "base_url": "",
                    "enabled": True,
                }
            ],
            "active_profile": "zai-profile",
            "scoped_profiles": ["zai-profile"],
        }
    )

    assert synced["default"] == "glm-4.6"
    assert synced["provider"] == "zai"
    assert synced["base_url"] == "https://zai.example/v1"


def test_upsert_profile_adds_profile_sets_active_and_syncs_legacy_fields():
    updated = mp.upsert_profile(
        _base_model_config(),
        {
            "name": "work",
            "provider": "kimi",
            "model": "k2p5",
            "enabled": True,
        },
    )

    assert any(p["name"] == "work" and p["provider"] == "kimi-coding" for p in updated["profiles"])
    assert updated["active_profile"] == "work"
    assert "work" in updated["scoped_profiles"]
    assert updated["default"] == "k2p5"
    assert updated["provider"] == "kimi-coding"


def test_upsert_profile_disabled_entry_does_not_take_active_slot():
    updated = mp.upsert_profile(
        _base_model_config(),
        {
            "name": "disabled-zai",
            "provider": "zai",
            "model": "glm-5",
            "enabled": False,
        },
    )

    assert updated["active_profile"] == mp.DEFAULT_PROFILE_NAME
    assert "disabled-zai" in updated["scoped_profiles"]
    assert updated["default"] == "anthropic/claude-opus-4.6"
    assert updated["provider"] == "openrouter"


def test_remove_profile_reassigns_active_profile_and_scoped_profiles():
    cfg = mp.normalize_model_config(
        {
            "profiles": [
                {"name": "a", "provider": "openrouter", "model": "model-a", "enabled": True},
                {"name": "b", "provider": "glm", "model": "model-b", "enabled": True},
            ],
            "active_profile": "a",
            "scoped_profiles": ["a", "b"],
        }
    )
    updated = mp.remove_profile(cfg, "a")

    assert [p["name"] for p in updated["profiles"]] == ["b"]
    assert updated["active_profile"] == "b"
    assert updated["scoped_profiles"] == ["b"]
    assert updated["default"] == "model-b"
    assert updated["provider"] == "zai"


def test_remove_profile_recreates_default_profile_when_last_profile_is_removed():
    cfg = mp.normalize_model_config(
        {
            "profiles": [{"name": "solo", "provider": "zai", "model": "glm-5", "enabled": True}],
            "active_profile": "solo",
            "scoped_profiles": ["solo"],
        }
    )
    updated = mp.remove_profile(cfg, "solo")

    assert [p["name"] for p in updated["profiles"]] == [mp.DEFAULT_PROFILE_NAME]
    assert updated["active_profile"] == mp.DEFAULT_PROFILE_NAME
    assert updated["scoped_profiles"] == [mp.DEFAULT_PROFILE_NAME]
    assert updated["default"] == mp.DEFAULT_MODEL_ID
    assert updated["provider"] == "openrouter"
    assert updated["base_url"] == OPENROUTER_BASE_URL


def test_set_active_profile_switches_active_profile_and_preserves_unknown_requests():
    cfg = mp.normalize_model_config(
        {
            "profiles": [
                {"name": "a", "provider": "openrouter", "model": "model-a", "enabled": True},
                {"name": "b", "provider": "zai", "model": "model-b", "enabled": True},
            ],
            "active_profile": "a",
            "scoped_profiles": ["a"],
        }
    )

    switched = mp.set_active_profile(cfg, "b")
    assert switched["active_profile"] == "b"
    assert switched["scoped_profiles"] == ["a", "b"]
    assert switched["default"] == "model-b"
    assert switched["provider"] == "zai"

    unchanged = mp.set_active_profile(switched, "missing")
    assert unchanged["active_profile"] == "b"
    assert unchanged["scoped_profiles"] == ["a", "b"]
