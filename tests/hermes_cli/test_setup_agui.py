"""Real config/secret writes for the standalone adapter's explicit setup section."""

import pytest

from hermes_cli import setup
from hermes_cli.config import get_env_value, load_config, save_config


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize(
    "port_input,expected_port", [("9123", 9123), (float("inf"), 8000)]
)
def test_setup_persists_listener_preserves_lifecycle_and_separates_secrets(
    monkeypatch, host, port_input, expected_port
):
    config = {
        "agui": {
            "frontend_tool_timeout": 17,
            "conversation_idle_timeout": 31,
            "max_conversations": 2,
            "future_setting": "preserved",
            "port": float("inf"),
            "provider": "custom",
            "model": "old-model",
            "api_mode": "chat_completions",
            "base_url": "http://localhost:4010/v1",
        }
    }
    save_config(config)
    fake_token = "a7G4p2N9s5T1u8V6c3D0f9H2j4K7m1Q5"
    replies = {
        "Bind host": host,
        "Listen port": port_input,
        "Toolsets (comma-separated)": "web, terminal",
        "Session token (minimum 16 characters)": fake_token,
    }
    monkeypatch.setattr(
        setup,
        "prompt",
        lambda question, default=None, **kwargs: replies.get(question, default or ""),
    )
    monkeypatch.setattr(setup, "prompt_choice", lambda *args, **kwargs: 0)
    setup._run_setup_section(config, "agui")
    saved = load_config()["agui"]
    assert (saved["host"], saved["port"], saved["toolsets"]) == (
        host,
        expected_port,
        ["web", "terminal"],
    )
    assert (
        saved["frontend_tool_timeout"],
        saved["conversation_idle_timeout"],
        saved["max_conversations"],
    ) == (17, 31, 2)
    assert saved["future_setting"] == "preserved"
    assert all(
        saved[key] == "" for key in ("provider", "model", "api_mode", "base_url")
    )
    assert fake_token not in str(saved)
    assert get_env_value("HERMES_AGUI_SESSION_TOKEN") == (
        fake_token if host == "0.0.0.0" else None
    )


def test_enter_keeps_override_and_dash_clears_only_endpoint(monkeypatch):
    from agui_adapter.session import AgentConfig, _resolve_agent_settings

    config = {
        "model": {"provider": "anthropic", "default": "main-model"},
        "agui": {"model": "kept-model", "base_url": "http://localhost:4010/v1"},
    }
    save_config(config)
    replies = iter(["", "", "", "", "", "", "-"])
    monkeypatch.setattr("hermes_cli.cli_output.line_input", lambda _: next(replies))
    monkeypatch.setattr(setup, "prompt_choice", lambda *args, **kwargs: 1)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-credential")

    setup._run_setup_section(config, "agui")

    saved = load_config()["agui"]
    assert saved["model"] == "kept-model"
    assert saved["base_url"] == ""
    settings = _resolve_agent_settings(AgentConfig())
    assert settings["provider"] == config["model"]["provider"]
    assert settings["model"] == saved["model"]
    assert settings["base_url"] != "http://localhost:4010/v1"
