"""CLI integration tests for smart model routing."""

from hermes_cli import cli_agent_setup_mixin


class _CLI:
    model = "gpt-5.6-terra"
    api_key = "test-key"
    base_url = "https://chatgpt.com/backend-api/codex"
    provider = "openai-codex"
    api_mode = "chat_completions"
    acp_command = None
    acp_args = []
    service_tier = None
    _credential_pool = None


def test_cli_turn_route_uses_configured_cheap_model(monkeypatch):
    monkeypatch.setattr(
        cli_agent_setup_mixin,
        "load_config",
        lambda: {
            "smart_model_routing": {
                "enabled": True,
                "platforms": ["cli"],
                "cheap_model": {"model": "gpt-5.6-luna"},
            }
        },
    )

    route = cli_agent_setup_mixin.CLIAgentSetupMixin._resolve_turn_agent_config(_CLI(), "hi")

    assert route["model"] == "gpt-5.6-luna"
    assert route["signature"][0] == "gpt-5.6-luna"
    assert route["routing_label"] == "cheap"
