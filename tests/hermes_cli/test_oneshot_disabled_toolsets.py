"""``hermes -z`` (oneshot) must pass ``agent.disabled_toolsets`` through to ``AIAgent`` like the
interactive CLI, the gateway, and the ACP adapter already do — otherwise a toolset the user
disabled in config stays fully executable in a one-shot run, with or without an explicit
``--toolsets`` list (mirrors the ACP fix in 2c12e7ae)."""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.oneshot as oneshot_mod


class _FakeAgent:
    def __init__(self, **kwargs):
        _FakeAgent.captured = kwargs

    def __setattr__(self, name, _value):
        pass

    def run_conversation(self, _prompt, conversation_history=None):
        return {"final_response": "pong", "session_id": "s"}

    def close(self):
        pass


def _run(monkeypatch, cfg, **run_agent_kwargs):
    def fake_resolve(**kw):
        return {"api_key": "k", "base_url": None, "provider": "openai-codex", "api_mode": "chat", "credential_pool": None}

    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    with patch.object(_FakeAgent, "captured", {}, create=True):
        oneshot_mod._run_agent("Health check: reply pong.", **run_agent_kwargs)
        return dict(_FakeAgent.captured)


def test_disabled_toolsets_passed_with_config_derived_toolsets(monkeypatch):
    cfg = {"model": {"default": "gpt-5.4", "provider": "openai-codex"}, "agent": {"disabled_toolsets": ["todo"]}}
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: {"todo", "search"})
    captured = _run(monkeypatch, cfg)
    assert captured["disabled_toolsets"] == ["todo"]


def test_disabled_toolsets_passed_with_explicit_toolsets_flag(monkeypatch):
    cfg = {"model": {"default": "gpt-5.4", "provider": "openai-codex"}, "agent": {"disabled_toolsets": ["todo"]}}
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: {"todo", "search"})
    captured = _run(monkeypatch, cfg, toolsets="search,todo")
    assert captured["disabled_toolsets"] == ["todo"]
    assert captured["enabled_toolsets"] == ["search", "todo"]


def test_disabled_toolsets_none_when_not_configured(monkeypatch):
    cfg = {"model": {"default": "gpt-5.4", "provider": "openai-codex"}}
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: {"search"})
    captured = _run(monkeypatch, cfg)
    assert captured["disabled_toolsets"] is None
