"""A CLI /model switch must preserve its key_cmd credential on the following turn."""

import hermes_yaml as yaml

from agent.command_token_source import CommandTokenSource
from hermes_cli.runtime_provider import resolve_runtime_provider


def test_named_key_cmd_provider_keeps_callable_after_model_switch(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    endpoint = "https://gateway.example.test/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "qodo", "default": "other-model"},
        "providers": {"relay": {
            "name": "Relay", "base_url": endpoint, "key_cmd": "printf fixture-token",
            "models": ["selected-model"],
        }},
    }), encoding="utf-8")
    source = CommandTokenSource("printf fixture-token", "Relay")

    # /model stores its callable result as the CLI's explicit key. The next turn passes it
    # through the public runtime resolver before the agent sends any request.
    runtime = resolve_runtime_provider(
        requested="relay", explicit_api_key=source, explicit_base_url=endpoint,
        target_model="selected-model",
    )

    assert runtime["api_key"] is source
    assert runtime["base_url"] == endpoint
    assert runtime["model"] == "selected-model"


def test_direct_custom_alias_keeps_callable_after_model_switch():
    endpoint = "https://gateway.example.test/v1"
    source = CommandTokenSource("printf fixture-token", "Relay")

    runtime = resolve_runtime_provider(
        requested="custom", explicit_api_key=source, explicit_base_url=endpoint,
        target_model="selected-model",
    )

    assert runtime["api_key"] is source
    assert runtime["base_url"] == endpoint
