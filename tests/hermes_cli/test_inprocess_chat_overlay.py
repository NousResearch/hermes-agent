"""In-process chat (-q under HERMES_INPROCESS_CHAT) and per-run MCP overlay wiring."""

from __future__ import annotations

import argparse
import json

from hermes_cli import main as main_mod
from hermes_cli.oneshot import run_oneshot


def test_cmd_chat_inprocess_routes_to_oneshot(monkeypatch):
    captured = {}

    def fake_oneshot(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setenv("HERMES_INPROCESS_CHAT", "1")
    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", fake_oneshot)
    args = argparse.Namespace(
        query="hello",
        model="m1",
        provider="auto",
        toolsets="terminal",
        skills=["skill-a"],
        resume="sess-1",
        reasoning=None,
    )
    main_mod.cmd_chat(args)
    assert captured["prompt"] == "hello"
    assert captured["model"] == "m1"
    assert captured["provider"] is None  # auto → no explicit provider
    assert captured["toolsets"] == "terminal"
    assert captured["skills"] == ["skill-a"]
    assert captured["resume"] == "sess-1"


def test_cmd_chat_inprocess_skips_without_query(monkeypatch):
    called = []
    monkeypatch.setenv("HERMES_INPROCESS_CHAT", "1")
    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", lambda *a, **k: called.append("oneshot"))
    monkeypatch.setattr("hermes_cli.stream_json.stream_json_requested", lambda _args: False)
    monkeypatch.setattr(main_mod, "_bypass_chat_launch", lambda _args: False)
    monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda _args: None)
    monkeypatch.setattr(main_mod, "_apply_user_config_bypass", lambda _args: None)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _args: None)
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main_mod, "_resolve_chat_session_args", lambda *_a, **_k: None)
    monkeypatch.setattr(main_mod, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: False)
    monkeypatch.setattr(main_mod, "_first_run_setup_guard", lambda _args: called.append("setup"))

    main_mod.cmd_chat(argparse.Namespace(query=None, tui=False))
    assert called == ["setup"]


def test_run_agent_appends_overlay_toolsets_and_ephemeral_prompt(monkeypatch, tmp_path):
    import hermes_cli.oneshot as oneshot_mod

    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps({"board": {"url": "https://board.example/mcp"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MCP_OVERLAY", str(overlay))
    monkeypatch.setenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "runtime guidance")

    captured = {}

    class _FakeAgent:
        session_id = "agent-sid"

        def __init__(self, **kwargs):
            captured.update(kwargs)
            object.__setattr__(self, "session_id", "agent-sid")

        def run_conversation(self, _prompt, conversation_history=None):
            return {"final_response": "ok"}

        def close(self):
            pass

    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "m", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: {
            "api_key": "k",
            "base_url": None,
            "provider": "openrouter",
            "requested_provider": "openrouter",
            "api_mode": "chat",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda _cfg, _p: {"terminal"},
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kw: None,
    )
    monkeypatch.setattr(oneshot_mod, "_build_preloaded_skills_prompt", lambda _skills: "skills block")
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    text, result = oneshot_mod._run_agent("hi", toolsets=["terminal"], use_config_toolsets=False)
    assert text == "ok"
    assert result["session_id"] == "agent-sid"
    assert captured["enabled_toolsets"] == ["terminal", "mcp-board"]
    assert captured["ephemeral_system_prompt"] == "runtime guidance\n\nskills block"


def test_run_agent_leaves_none_toolsets_alone(monkeypatch, tmp_path):
    import hermes_cli.oneshot as oneshot_mod

    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"board": {"url": "https://board.example/mcp"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MCP_OVERLAY", str(overlay))

    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            object.__setattr__(self, "session_id", "s")

        def run_conversation(self, _prompt, conversation_history=None):
            return {"final_response": "ok"}

        def close(self):
            pass

    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "m", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: {
            "api_key": "k",
            "base_url": None,
            "provider": "openrouter",
            "api_mode": "chat",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kw: None,
    )
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    oneshot_mod._run_agent("hi", toolsets=None, use_config_toolsets=False)
    assert captured["enabled_toolsets"] is None


def test_run_oneshot_writes_session_id_line_when_inprocess(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_INPROCESS_CHAT", "1")

    def fake_run_agent(prompt, **kwargs):
        return "hello", {"final_response": "hello", "session_id": "sess-xyz", "completed": True}

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fake_run_agent)
    monkeypatch.setattr("hermes_cli.oneshot._validate_explicit_toolsets", lambda _t: (None, None))
    rc = run_oneshot("hello")
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello\n" in out
    assert "session_id: sess-xyz\n" in out
