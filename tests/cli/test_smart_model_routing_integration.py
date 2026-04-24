from __future__ import annotations

import cli as cli_module


def test_resolve_turn_agent_config_uses_smart_model_route(monkeypatch):
    cli = object.__new__(cli_module.HermesCLI)
    cli.api_key = "primary-key"
    cli.base_url = "https://primary.example/v1"
    cli.provider = "openrouter"
    cli.api_mode = "chat_completions"
    cli.acp_command = None
    cli.acp_args = []
    cli._credential_pool = None
    cli.service_tier = None
    cli.model = "primary-model"

    routing_cfg = {
        "enabled": True,
        "complex_keywords": ["research", "review", "audit", "研究", "審查", "審閱"],
        "cheap_model": {
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
        },
        "complex_model": {
            "provider": "openai-codex",
            "model": "gpt-5.4",
        },
    }
    monkeypatch.setattr(cli_module, "CLI_CONFIG", {"smart_model_routing": routing_cfg}, raising=False)

    calls = {}

    def fake_resolve_turn_route(user_message, cfg, primary):
        calls["user_message"] = user_message
        calls["cfg"] = cfg
        calls["primary"] = primary
        return {
            "model": "gpt-5.4",
            "runtime": {
                "api_key": "route-key",
                "base_url": "https://route.example/v1",
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
            "label": "smart route → gpt-5.4 (openai-codex)",
            "signature": ("gpt-5.4", "openai-codex", "https://route.example/v1", "codex_responses", None, ()),
        }

    monkeypatch.setattr(cli_module, "resolve_turn_route", fake_resolve_turn_route)

    result = cli._resolve_turn_agent_config("please review this PR diff carefully")

    assert calls["user_message"] == "please review this PR diff carefully"
    assert calls["cfg"] is routing_cfg
    assert calls["primary"]["model"] == "primary-model"
    assert result["model"] == "gpt-5.4"
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["label"] == "smart route → gpt-5.4 (openai-codex)"
    assert result["request_overrides"] is None
