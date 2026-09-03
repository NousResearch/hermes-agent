"""Main-provider-scoped config policy tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


_BASE_CONFIG: dict[str, Any] = {
    "model": {"provider": "openai-codex", "default": "gpt-5.6-sol-900k"},
    "auxiliary": {
        "compression": {"provider": "anthropic", "model": "claude-opus-5"},
    },
    "fallback_providers": [
        {"provider": "anthropic", "model": "claude-opus-5"},
    ],
    "main_provider_policies": {
        "openrouter": {
            "model_overrides": {
                "z-ai/glm-5.3-flash": {"context_length": 1_048_576},
            },
            "provider_routing": {
                "require_parameters": True,
                "data_collection": "deny",
            },
            "auxiliary": {
                "compression": {
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-flash-0731",
                },
            },
            "fallback_providers": [
                {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"},
                {"provider": "anthropic", "model": "claude-opus-5"},
            ],
        },
    },
}


def test_matching_main_provider_policy_overlays_without_mutating_base():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openrouter", "z-ai/glm-5.3-flash"
    )

    assert resolved["model"]["default"] == "gpt-5.6-sol-900k"
    assert resolved["model"]["context_length"] == 1_048_576
    assert resolved["provider_routing"] == {
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert resolved["auxiliary"]["compression"] == {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
    }
    assert resolved["fallback_providers"][0]["provider"] == "openai-codex"

    assert _BASE_CONFIG["auxiliary"]["compression"]["provider"] == "anthropic"
    assert "context_length" not in _BASE_CONFIG["model"]


def test_managed_leaves_win_over_matching_user_policy(tmp_path: Path, monkeypatch):
    user_home = tmp_path / "user"
    managed_home = tmp_path / "managed"
    user_home.mkdir()
    managed_home.mkdir()
    user_config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    user_config["main_provider_policies"]["openrouter"]["provider_routing"][
        "data_collection"
    ] = "allow"
    user_config["main_provider_policies"]["openrouter"]["auxiliary"][
        "compression"
    ]["provider"] = "openrouter"
    (user_home / "config.yaml").write_text(
        yaml.safe_dump(user_config, sort_keys=False), encoding="utf-8"
    )
    (managed_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider_routing": {"data_collection": "deny"},
                "auxiliary": {"compression": {"provider": "anthropic"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(user_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))

    from hermes_cli import managed_scope
    from hermes_cli.config import (
        _LOAD_CONFIG_CACHE,
        load_config_readonly,
        resolve_main_provider_policy,
    )

    managed_scope.invalidate_managed_cache()
    _LOAD_CONFIG_CACHE.clear()
    loaded = load_config_readonly()
    resolved = resolve_main_provider_policy(
        loaded, "openrouter", "z-ai/glm-5.3-flash"
    )

    assert loaded["provider_routing"]["data_collection"] == "deny"
    assert resolved["provider_routing"]["data_collection"] == "deny"
    assert resolved["auxiliary"]["compression"]["provider"] == "anthropic"


def test_nonmatching_main_provider_leaves_base_policy_dormant():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openai-codex", "gpt-5.6-sol-900k"
    )

    assert resolved is _BASE_CONFIG
    assert resolved["auxiliary"]["compression"]["provider"] == "anthropic"
    assert resolved["fallback_providers"] == [
        {"provider": "anthropic", "model": "claude-opus-5"}
    ]
    assert "provider_routing" not in resolved


def test_model_override_does_not_leak_to_another_openrouter_model():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openrouter", "qwen/qwen3.8-flash"
    )

    assert "context_length" not in resolved["model"]
    assert resolved["provider_routing"]["require_parameters"] is True


def test_non_context_model_override_does_not_authorize_base_context(
    tmp_path: Path, monkeypatch
):
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["model"]["default"] = "z-ai/glm-5.3-flash"
    config["model"]["context_length"] = 222_222
    config["main_provider_policies"]["openrouter"]["model_overrides"][
        "z-ai/glm-5.3-flash"
    ] = {"max_tokens": 1_024}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )

    assert getattr(agent, "context_compressor").context_length != 222_222


def test_model_base_url_override_cannot_authorize_base_context(
    tmp_path: Path, monkeypatch
):
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["model"].update(
        {
            "default": "z-ai/glm-5.3-flash",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "context_length": 222_222,
        }
    )
    config["main_provider_policies"]["openrouter"]["model_overrides"][
        "z-ai/glm-5.3-flash"
    ] = {"base_url": "https://openrouter.ai/api/v1"}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )

    assert getattr(agent, "context_compressor").context_length != 222_222


def test_disabled_policy_is_dormant():
    from hermes_cli.config import resolve_main_provider_policy

    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["main_provider_policies"]["openrouter"]["enabled"] = False

    assert (
        resolve_main_provider_policy(config, "openrouter", "z-ai/glm-5.3-flash")
        is config
    )


@pytest.mark.parametrize(
    "malformed",
    [
        {"enabled": "false"},
        {"fallback_providers": "anthropic"},
        {"fallback_providers": [{"provider": "anthropic"}]},
        {"provider_routing": []},
        {"provider_routing": {"only": "Anthropic"}},
        {"auxiliary": []},
        {"auxiliary": {"review": "anthropic"}},
        {"model_overrides": []},
        {"model": "attacker/replacement-model"},
        {"model_overrides": {"z-ai/glm-5.3-flash": "1048576"}},
        {
            "model_overrides": {
                "z-ai/glm-5.3-flash": {"context_length": "1M"}
            }
        },
        {
            "model_overrides": {
                "z-ai/glm-5.3-flash": {"provider": "attacker"}
            }
        },
        {"agent": {"reasoning_effort": "high"}},
    ],
)
def test_malformed_matching_policy_is_wholly_inert(malformed):
    from hermes_cli.config import resolve_main_provider_policy

    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    policy = {
        "provider_routing": {"require_parameters": True},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"}
        ],
    }
    policy.update(malformed)
    config["main_provider_policies"]["openrouter"] = policy

    resolved = resolve_main_provider_policy(
        config, "openrouter", "z-ai/glm-5.3-flash"
    )

    assert resolved is config
    assert "provider_routing" not in resolved
    assert resolved["fallback_providers"] == [
        {"provider": "anthropic", "model": "claude-opus-5"}
    ]


def test_valid_open_dict_policy_extensions_are_preserved():
    from hermes_cli.config import resolve_main_provider_policy

    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["main_provider_policies"]["openrouter"] = {
        "provider_routing": {
            "require_parameters": True,
            "future_provider_scalar": "preferred",
        },
        "auxiliary": {
            "future_task": {
                "provider": "openrouter",
                "model": "future/model",
                "temperature": 0.2,
                "extra_body": {"vendor_flag": True},
            }
        },
        "model_overrides": {
            "z-ai/glm-5.3-flash": {
                "context_length": 1_048_576,
                "future_model_scalar": 7,
            }
        },
        "fallback_providers": [
            {
                "provider": "anthropic",
                "model": "claude-opus-5",
                "reasoning_echo": True,
            }
        ],
    }

    resolved = resolve_main_provider_policy(
        config, "openrouter", "z-ai/glm-5.3-flash"
    )

    assert resolved["provider_routing"]["future_provider_scalar"] == "preferred"
    assert resolved["auxiliary"]["future_task"]["extra_body"] == {
        "vendor_flag": True
    }
    assert resolved["model"]["future_model_scalar"] == 7
    assert resolved["fallback_providers"][0]["reasoning_echo"] is True


def test_scalar_model_selector_survives_exact_context_override():
    from hermes_cli.config import resolve_main_provider_policy

    config = {
        "model": "base/main-model",
        "main_provider_policies": {
            "openrouter": {
                "model_overrides": {
                    "base/main-model": {"context_length": 123_456}
                }
            }
        },
    }

    resolved = resolve_main_provider_policy(
        config, "openrouter", "base/main-model"
    )

    assert resolved["model"] == {
        "default": "base/main-model",
        "context_length": 123_456,
    }
    assert config["model"] == "base/main-model"


def test_explicit_reasoning_equal_to_base_is_not_projected_by_policy(
    tmp_path: Path, monkeypatch
):
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["agent"] = {"reasoning_effort": "medium"}
    config["main_provider_policies"]["openrouter"]["agent"] = {
        "reasoning_effort": "high"
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    explicit = {"enabled": True, "effort": "medium"}
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        reasoning_config=explicit,
    )

    assert getattr(agent, "reasoning_config") == explicit


def test_openrouter_agent_uses_matching_policy_at_runtime(tmp_path: Path, monkeypatch):
    config = {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol-900k"},
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude-opus-5"},
        ],
        "main_provider_policies": _BASE_CONFIG["main_provider_policies"],
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        fallback_model=config["fallback_providers"],
    )

    assert getattr(agent, "provider_require_parameters") is True
    assert getattr(agent, "provider_data_collection") == "deny"
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]
    assert getattr(agent, "context_compressor").context_length == 1_048_576


@pytest.mark.parametrize(
    ("base_url", "expected_provider"),
    [
        ("https://api.anthropic.com", "anthropic"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openai.com/v1", "openai-api"),
        ("https://api.meta.ai/v1", "meta"),
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "bedrock",
        ),
    ],
)
def test_url_canonicalization_precedes_initial_policy_selection(
    tmp_path: Path, monkeypatch, base_url: str, expected_provider: str
):
    config = {
        "model": {"default": "test/model"},
        "main_provider_policies": {
            expected_provider: {
                "provider_routing": {"only": [f"{expected_provider}-policy"]}
            }
        },
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        model="test/model",
        base_url=base_url,
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )

    assert getattr(agent, "provider") == expected_provider
    assert getattr(agent, "providers_allowed") == [f"{expected_provider}-policy"]
    assert getattr(agent, "_main_provider_policy_active") is True


def test_in_place_switch_refreshes_policy_and_restores_base(
    tmp_path: Path, monkeypatch
):
    model = "shared/model-id"
    config = {
        "model": {
            "provider": "openai-codex",
            "default": model,
            "context_length": 222_222,
        },
        "provider_routing": {
            "only": ["Base Route"],
            "require_parameters": False,
            "data_collection": "allow",
        },
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude-opus-5"},
        ],
        "main_provider_policies": {
            "openrouter": {
                "model_overrides": {model: {"context_length": 1_048_576}},
                "provider_routing": {
                    "only": ["Policy Route"],
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                "auxiliary": {
                    "compression": {
                        "provider": "openrouter",
                        "model": "deepseek/deepseek-v4-flash-0731",
                    }
                },
                "fallback_providers": [
                    {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"},
                    {"provider": "anthropic", "model": "claude-opus-5"},
                ],
            }
        },
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )
    monkeypatch.setattr(agent, "_create_openai_client", lambda *_a, **_k: object())

    assert getattr(agent, "providers_allowed") == ["Policy Route"]
    assert getattr(agent, "provider_require_parameters") is True
    assert getattr(agent, "provider_data_collection") == "deny"
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]
    assert getattr(agent, "context_compressor").context_length == 1_048_576

    agent.switch_model(
        new_model=model,
        new_provider="openai-codex",
        api_key="codex-key",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
    )

    assert getattr(agent, "providers_allowed") == ["Base Route"]
    assert getattr(agent, "provider_require_parameters") is False
    assert getattr(agent, "provider_data_collection") == "allow"
    assert getattr(agent, "_fallback_chain") == [
        {"provider": "anthropic", "model": "claude-opus-5"}
    ]
    assert getattr(agent, "context_compressor").context_length == 222_222

    agent.switch_model(
        new_model=model,
        new_provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
    )

    assert getattr(agent, "providers_allowed") == ["Policy Route"]
    assert getattr(agent, "provider_require_parameters") is True
    assert getattr(agent, "provider_data_collection") == "deny"
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]
    assert getattr(agent, "context_compressor").context_length == 1_048_576


def test_failed_policy_switch_restores_context_and_routing_atomically(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )
    monkeypatch.setattr(agent, "_create_openai_client", lambda *_a, **_k: object())
    compressor = getattr(agent, "context_compressor")
    original_update = compressor.update_model

    def fail_after_context_mutation(**kwargs):
        original_update(**kwargs)
        raise RuntimeError("context update failed")

    monkeypatch.setattr(compressor, "update_model", fail_after_context_mutation)

    with pytest.raises(RuntimeError, match="context update failed"):
        agent.switch_model(
            new_model="gpt-5.6-sol-900k",
            new_provider="openai-codex",
            api_key="codex-key",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
        )

    assert getattr(agent, "provider") == "openrouter"
    assert getattr(agent, "model") == "z-ai/glm-5.3-flash"
    assert compressor.provider == "openrouter"
    assert compressor.model == "z-ai/glm-5.3-flash"
    assert compressor.context_length == 1_048_576
    assert getattr(agent, "provider_require_parameters") is True
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]


def test_active_policy_switch_aborts_if_projection_cannot_be_read(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )

    def fail_config_read():
        raise RuntimeError("injected config read failure")

    monkeypatch.setattr(config_module, "load_config_readonly", fail_config_read)

    with pytest.raises(RuntimeError, match="policy projection"):
        agent.switch_model(
            new_model="gpt-5.6-sol-900k",
            new_provider="openai-codex",
            api_key="codex-key",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
        )

    assert getattr(agent, "provider") == "openrouter"
    assert getattr(agent, "model") == "z-ai/glm-5.3-flash"
    assert getattr(agent, "provider_require_parameters") is True
    assert getattr(agent, "provider_data_collection") == "deny"
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]
    assert getattr(agent, "context_compressor").context_length == 1_048_576


def test_initial_policy_projection_failure_is_atomic(tmp_path: Path, monkeypatch):
    model = "z-ai/glm-5.3-flash"
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["model"] = {
        "provider": "openrouter",
        "default": model,
        "context_length": 222_222,
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from hermes_cli import fallback_config
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()

    def fail_chain(_config):
        raise RuntimeError("injected fallback projection failure")

    monkeypatch.setattr(fallback_config, "get_fallback_chain", fail_chain)
    caller_fallback = [{"provider": "nous", "model": "caller-fallback"}]
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        providers_allowed=["Caller Only"],
        providers_ignored=["Caller Ignore"],
        providers_order=["Caller Order"],
        provider_sort="latency",
        provider_require_parameters=False,
        provider_data_collection="allow",
        fallback_model=caller_fallback,
    )

    assert getattr(agent, "providers_allowed") == ["Caller Only"]
    assert getattr(agent, "providers_ignored") == ["Caller Ignore"]
    assert getattr(agent, "providers_order") == ["Caller Order"]
    assert getattr(agent, "provider_sort") == "latency"
    assert getattr(agent, "provider_require_parameters") is False
    assert getattr(agent, "provider_data_collection") == "allow"
    assert getattr(agent, "_fallback_chain") == caller_fallback
    assert getattr(agent, "context_compressor").context_length == 222_222
    assert getattr(agent, "_main_provider_policy_active") is False


def test_malformed_policy_does_not_refresh_switch_state(
    tmp_path: Path, monkeypatch
):
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["provider_routing"] = {"only": ["Base Route"]}
    config["main_provider_policies"]["openrouter"]["provider_routing"] = []
    config["main_provider_policies"]["anthropic"] = {
        "auxiliary": {"review": {"provider": "anthropic", "model": "review-model"}}
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    caller_fallback = [{"provider": "nous", "model": "caller-fallback"}]
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        providers_allowed=["Caller Route"],
        fallback_model=caller_fallback,
    )
    monkeypatch.setattr(agent, "_create_openai_client", lambda *_a, **_k: object())

    agent.switch_model(
        new_model="z-ai/glm-5.3-flash",
        new_provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
    )

    assert getattr(agent, "providers_allowed") == ["Caller Route"]
    assert getattr(agent, "_fallback_chain") == caller_fallback


def test_auxiliary_policy_follows_live_main_provider(tmp_path: Path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.auxiliary_client import (
        _get_auxiliary_task_config,
        scoped_runtime_main,
    )
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    with scoped_runtime_main(
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}
    ):
        openrouter_compression = _get_auxiliary_task_config("compression")
    with scoped_runtime_main(
        {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"}
    ):
        codex_compression = _get_auxiliary_task_config("compression")

    assert openrouter_compression["provider"] == "openrouter"
    assert openrouter_compression["model"] == "deepseek/deepseek-v4-flash-0731"
    assert codex_compression["provider"] == "anthropic"
    assert codex_compression["model"] == "claude-opus-5"


def test_aux_openrouter_guards_follow_live_main_policy(tmp_path: Path, monkeypatch):
    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["auxiliary"]["free_only"] = False
    config["auxiliary"]["openrouter_model"] = "paid/base-model"
    config["main_provider_policies"]["openrouter"]["auxiliary"].update(
        {
            "free_only": True,
            "openrouter_model": "free/policy-model:free",
        }
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.auxiliary_client import _aux_openrouter_settings, scoped_runtime_main
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    with scoped_runtime_main(
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}
    ):
        policy_settings = _aux_openrouter_settings()
    with scoped_runtime_main(
        {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"}
    ):
        base_settings = _aux_openrouter_settings()

    assert policy_settings == (True, "free/policy-model:free")
    assert base_settings == (False, "paid/base-model")


def test_aux_policy_is_isolated_across_concurrent_profiles(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    cases = [
        (
            tmp_path / "profile-a",
            "openrouter",
            {"free_only": False, "openrouter_model": "paid/profile-a-base"},
            {"free_only": True, "openrouter_model": "free/profile-a:free"},
            (True, "free/profile-a:free"),
        ),
        (
            tmp_path / "profile-b",
            "openai-codex",
            {"free_only": False, "openrouter_model": "paid/profile-b-base"},
            {"free_only": True, "openrouter_model": "free/profile-b:free"},
            (False, "paid/profile-b-base"),
        ),
    ]
    for home, _provider, base_aux, policy_aux, _expected in cases:
        home.mkdir()
        config = {
            "model": {"provider": "openrouter", "default": "main/model"},
            "auxiliary": base_aux,
            "main_provider_policies": {
                "openrouter": {"auxiliary": policy_aux}
            },
        }
        (home / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    from agent.auxiliary_client import _aux_openrouter_settings, scoped_runtime_main
    from hermes_cli import config as config_module
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    config_module._LOAD_CONFIG_CACHE.clear()
    barrier = Barrier(2)

    def read_profile(case) -> tuple[bool, str]:
        home, provider, _base_aux, _policy_aux, _expected = case
        home_token = set_hermes_home_override(str(home))
        try:
            with scoped_runtime_main(
                {"provider": provider, "model": "main/model"}
            ):
                barrier.wait(timeout=5)
                return _aux_openrouter_settings()
        finally:
            reset_hermes_home_override(home_token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(read_profile, cases))

    assert results == [case[-1] for case in cases]


def test_auto_auxiliary_uses_matching_policy_fallback_chain(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent import auxiliary_client
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_fallback_entry",
        lambda entry: (object(), entry["model"]),
    )
    with auxiliary_client.scoped_runtime_main(
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}
    ):
        _, model, provider = auxiliary_client._try_main_fallback_chain(
            "title_generation", failed_provider="openrouter"
        )

    assert provider == "openai-codex"
    assert model == "gpt-5.6-sol-900k"


def test_main_provider_policy_paths_are_known_config_keys():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key(
        "main_provider_policies.openrouter.auxiliary.compression.model"
    ) == (True, None)
    assert _validate_config_key(
        "main_provider_policies.openrouter.auxiliary.future_task.vendor_flag"
    ) == (True, None)
    assert _validate_config_key(
        "main_provider_policies.openrouter.provider_routing.future_scalar"
    ) == (True, None)
    assert _validate_config_key(
        "main_provider_policies.openrouter.agent.reasoning_effort"
    )[0] is False
