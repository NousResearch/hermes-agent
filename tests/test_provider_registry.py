"""Unit tests for hermes_cli.provider_registry."""

import pytest

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli import provider_registry as pr


def _env(mapping):
    return lambda key: mapping.get(key)


def test_normalize_provider_id_aliases_defaults_and_unknowns():
    assert pr.normalize_provider_id(" GLM ") == "zai"
    assert pr.normalize_provider_id("z-AI") == "zai"
    assert pr.normalize_provider_id("kimi") == "kimi-coding"
    assert pr.normalize_provider_id("", default="openrouter") == "openrouter"
    assert pr.normalize_provider_id(None, default="openrouter") == "openrouter"
    assert pr.normalize_provider_id("  Unknown-Provider  ") == "unknown-provider"


def test_resolve_provider_api_key_prefers_explicit_then_first_non_empty_env_var():
    env = {"GLM_API_KEY": " glm-primary ", "ZAI_API_KEY": "zai-secondary"}
    assert (
        pr.resolve_provider_api_key(
            "zai",
            env_get=_env(env),
            explicit_api_key="explicit-key",
        )
        == "explicit-key"
    )
    assert pr.resolve_provider_api_key("zai", env_get=_env(env)) == "glm-primary"

    env["GLM_API_KEY"] = "   "
    assert pr.resolve_provider_api_key("zai", env_get=_env(env)) == "zai-secondary"

    env["ZAI_API_KEY"] = "   "
    env["Z_AI_API_KEY"] = "zai-underscore-alias"
    assert pr.resolve_provider_api_key("zai", env_get=_env(env)) == "zai-underscore-alias"


def test_resolve_provider_base_url_precedence_and_normalization():
    env = {"GLM_BASE_URL": " https://api.example.test/v4/// "}

    assert (
        pr.resolve_provider_base_url(
            "zai",
            env_get=_env(env),
            explicit_base_url=" https://override.example/v1/ ",
        )
        == "https://override.example/v1/"
    )
    assert pr.resolve_provider_base_url("zai", env_get=_env(env)) == "https://api.example.test/v4"
    assert pr.resolve_provider_base_url("openrouter", env_get=_env({})) == OPENROUTER_BASE_URL
    assert pr.resolve_provider_base_url("missing-provider", env_get=_env({})) is None


def test_provider_cli_choices_and_list_filters():
    choices = pr.provider_cli_choices(include_auto=True)
    assert "auto" in choices
    assert "zai" in choices
    assert "custom" not in choices

    api_only = pr.list_provider_ids(include_oauth=False, include_api_key=True, include_custom=False)
    assert "nous" not in api_only
    assert "openrouter" in api_only


def test_setup_and_model_picker_provider_visibility_lists():
    setup_ids = pr.list_setup_provider_ids()
    model_ids = pr.list_model_picker_provider_ids()

    for expected in ("openrouter", "nous", "zai", "kimi-coding", "minimax", "minimax-cn", "custom"):
        assert expected in setup_ids
        assert expected in model_ids


def test_resolve_effective_base_url_precedence_and_custom_fallback_behavior():
    env = {"GLM_BASE_URL": "https://env.example/zai/v1"}

    assert (
        pr.resolve_effective_base_url(
            "zai",
            env_get=_env(env),
            explicit_base_url=" https://explicit.example/zai/v1/ ",
            profile_base_url="https://profile.example/zai/v1",
            model_base_url="https://model.example/zai/v1",
        )
        == "https://explicit.example/zai/v1"
    )
    assert (
        pr.resolve_effective_base_url(
            "zai",
            env_get=_env(env),
            profile_base_url="https://profile.example/zai/v1",
            model_base_url="https://model.example/zai/v1",
        )
        == "https://env.example/zai/v1"
    )
    assert (
        pr.resolve_effective_base_url(
            "zai",
            env_get=_env({}),
            profile_base_url="https://profile.example/zai/v1",
            model_base_url="https://model.example/zai/v1",
        )
        == "https://profile.example/zai/v1"
    )
    assert (
        pr.resolve_effective_base_url(
            "custom",
            env_get=_env({}),
            include_openrouter_fallback=False,
        )
        is None
    )


def test_get_provider_model_candidates_uses_loader_for_openrouter_and_curated_for_others():
    assert pr.get_provider_model_candidates("openrouter", openrouter_model_loader=lambda: ["a", "b"]) == [
        "a",
        "b",
    ]
    assert "glm-5" in pr.get_provider_model_candidates("zai")


def test_has_any_provider_key_checks_all_configured_env_vars():
    env = {"GLM_API_KEY": "   ", "ZAI_API_KEY": "   ", "Z_AI_API_KEY": "legacy-zai-key"}
    assert pr.has_any_provider_key("zai", env_get=_env(env)) is True
    assert pr.has_any_provider_key("minimax", env_get=_env(env)) is False


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"OPENAI_BASE_URL": " https://custom-endpoint/v1 ", "OPENROUTER_API_KEY": "or-key"}, "custom"),
        ({"OPENROUTER_API_KEY": "or-key", "GLM_API_KEY": "glm-key"}, "openrouter"),
        ({"ZAI_API_KEY": "zai-key"}, "zai"),
        ({"Z_AI_API_KEY": "z-ai-key"}, "zai"),
        ({"KIMI_API_KEY": "kimi-key"}, "kimi-coding"),
        ({"MINIMAX_API_KEY": "minimax-key", "MINIMAX_CN_API_KEY": "cn-key"}, "minimax"),
        ({"MINIMAX_CN_API_KEY": "cn-key"}, "minimax-cn"),
        ({}, "openrouter"),
    ],
)
def test_detect_auto_provider_priority_order(env, expected):
    assert pr.detect_auto_provider(env_get=_env(env)) == expected
