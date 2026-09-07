"""Output limits must survive real config loading through request assembly."""

import logging
import socket
from unittest.mock import patch

import pytest
import yaml

import agent.models_dev as models_dev
from hermes_constants import get_hermes_home
from run_agent import AIAgent


@pytest.fixture
def build_agent(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Output-limit tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_network)
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)
    monkeypatch.setattr(models_dev, "_OVERRIDE_WARNED_KEYS", set())

    def build(*, overrides=None, config_cap=None, env_cap=None, constructor_cap=None,
              requested_provider="custom:local-test"):
        config = {
            "model": {
                "provider": "custom:local-test",
                "default": "test-model",
                "base_url": "http://local-test.invalid/v1",
                "context_length": 200000,
            },
            "custom_providers": [
                {"name": name, "base_url": f"http://{name}.invalid/v1", "model": "test-model"}
                for name in ("local-test", "other-local")
            ],
            "model_overrides": overrides or {},
        }
        if config_cap is not None:
            config["model"]["max_tokens"] = config_cap
        if env_cap is not None:
            monkeypatch.setenv("HERMES_MAX_TOKENS", env_cap)
        (get_hermes_home() / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8",
        )
        # Only external clients/tool discovery are replaced; config and request
        # resolution must run together or a dropped setting looks correct.
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            return AIAgent(
                model="test-model", provider="custom",
                requested_provider=requested_provider,
                base_url="http://local-test.invalid/v1", api_key="test-key",
                max_tokens=constructor_cap, quiet_mode=True,
                skip_context_files=True, skip_memory=True,
            )

    return build


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"overrides": {"local-test": {"test-model": {"max_output_tokens": 25000}}}}, 25000),
        ({"overrides": {"custom": {"test-model": {"max_output_tokens": 25000}}}}, 25000),
        ({"env_cap": "25000"}, 25000),
        ({"config_cap": 24000}, 24000),
        ({"config_cap": 24000, "constructor_cap": 23000}, 23000),
        ({"config_cap": 24000, "env_cap": "25000"}, 25000),
        ({"env_cap": "25000", "constructor_cap": 23000}, 23000),
        ({"overrides": {"local-test": {"test-model": {"max_output_tokens": 25000}}},
          "requested_provider": "custom"}, 25000),
        ({"overrides": {"local-test": {"test-model": {"max_output_tokens": 25000}}},
          "env_cap": "23000"}, 23000),
        ({"overrides": {"local-test": {"test-model": {"max_output_tokens": 25000}}},
          "env_cap": "30000"}, 25000),
    ],
    ids=["named-override", "custom-override", "environment", "config", "constructor",
         "env-over-config", "constructor-over-env", "endpoint-identity", "env-ceiling", "model-ceiling"],
)
@pytest.mark.parametrize("api_mode, output_path", [
    ("chat_completions", ("max_tokens",)),
    ("anthropic_messages", ("max_tokens",)),
    ("codex_responses", ("max_output_tokens",)),
    ("bedrock_converse", ("inferenceConfig", "maxTokens")),
])
def test_output_limit_reaches_request(build_agent, settings, expected, api_mode, output_path):
    agent = build_agent(**settings)
    agent.api_mode = api_mode
    request = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
    value = request
    for key in output_path:
        value = value[key]
    assert value == expected


def test_unresolved_override_warns(build_agent, caplog):
    with caplog.at_level(logging.WARNING):
        agent = build_agent(overrides={
            "misspelled-provider": {"test-model": {"max_output_tokens": 25000}},
        })
        agent._build_api_kwargs([{"role": "user", "content": "hello"}])
        agent._build_api_kwargs([{"role": "user", "content": "again"}])
    warnings = [record.message for record in caplog.records if "model_overrides" in record.message]
    assert len(warnings) == 1
    assert "section" in warnings[0] and "custom" in warnings[0]


def test_output_ceiling_tracks_runtime_without_changing_user_budget(build_agent):
    agent = build_agent(config_cap=30000, overrides={
        "local-test": {"test-model": {"max_output_tokens": 25000}},
        "custom": {"test-model": {"max_output_tokens": 10000}},
        "other-local": {"test-model": {"max_output_tokens": 17000}},
    })
    messages = [{"role": "user", "content": "hello"}]
    assert agent._build_api_kwargs(messages)["max_tokens"] == 25000
    agent._ephemeral_max_output_tokens = 40000
    assert agent._build_api_kwargs(messages)["max_tokens"] == 25000
    assert agent._ephemeral_max_output_tokens is None

    # A restored/fallback runtime may only retain the generic provider and URL.
    agent.requested_provider = "custom"
    agent.base_url = "http://other-local.invalid/v1/"
    assert agent._build_api_kwargs(messages)["max_tokens"] == 17000
    agent.model = "unconfigured-model"
    assert agent._build_api_kwargs(messages)["max_tokens"] == 30000
    assert agent.max_tokens == 30000
