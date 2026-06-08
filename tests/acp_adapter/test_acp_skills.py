from __future__ import annotations

import sys
from types import SimpleNamespace

import acp


def test_hermes_cli_acp_startup_skills_reach_agent_ephemeral_prompt(monkeypatch):
    from hermes_cli import main as main_mod
    from acp_adapter import entry as entry_mod

    captured = {}
    skill_calls = {}

    class CapturingAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")

    async def fake_run_agent(agent, **kwargs):
        captured["run_kwargs"] = kwargs
        agent.session_manager._make_agent(session_id="acp-session", cwd=".")

    def fake_build_preloaded_skills_prompt(skills, task_id=None):
        skill_calls["skills"] = list(skills)
        skill_calls["task_id"] = task_id
        return "SKILL PROMPT", list(skills), []

    monkeypatch.setattr(sys, "argv", ["hermes", "-s", "alpha,beta", "-s", "gamma", "acp"])
    monkeypatch.setattr(entry_mod, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry_mod, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)
    monkeypatch.setattr("run_agent.AIAgent", CapturingAgent)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "test-model"}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.example/v1",
            "api_key": "***",
            "command": None,
            "args": [],
        },
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        fake_build_preloaded_skills_prompt,
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: None,
        raising=False,
    )

    main_mod.main()

    assert captured["run_kwargs"]["use_unstable_protocol"] is True
    assert skill_calls == {"skills": ["alpha", "beta", "gamma"], "task_id": "acp-session"}
    assert captured["ephemeral_system_prompt"] == "SKILL PROMPT"
    assert "prefill_messages" not in captured
