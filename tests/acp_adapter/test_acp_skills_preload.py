"""Tests for ACP skills preloading via HERMES_ACP_SKILLS env var (fixes #24466)."""
import sys
from types import ModuleType

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.steers = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def _patch_module(monkeypatch, patches, name, **attrs):
    """Register a minimal fake module via monkeypatch — same pattern as test_acp_real_agent_gets_session_db_for_recall."""
    module = ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    patches[name] = module
    monkeypatch.setitem(sys.modules, name, module)


class TestAcpSkillsPreload:
    """Test HERMES_ACP_SKILLS env var → ephemeral_system_prompt flow in _make_agent."""

    def test_skills_env_var_injects_ephemeral_system_prompt(self, monkeypatch):
        """When HERMES_ACP_SKILLS is set, _make_agent passes it as ephemeral_system_prompt."""
        captured = {}
        sentinel_db = NoopDb()
        fake_skills_prompt = "[IMPORTANT: test-skill is preloaded]"
        called_with = {}

        def fake_build(skill_list, task_id=None):
            called_with["skill_list"] = skill_list
            called_with["task_id"] = task_id
            return (fake_skills_prompt, ["test-skill"], [])

        class CapturingAgent(FakeAgent):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        patches = {}
        _patch_module(
            monkeypatch, patches, "run_agent",
            AIAgent=CapturingAgent,
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.config",
            load_config=lambda: {"model": {"default": "m", "provider": "p"}},
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kw: {
                "provider": "p", "api_mode": "chat_completions",
                "base_url": "u", "api_key": "***", "command": None, "args": [],
            },
        )
        _patch_module(
            monkeypatch, patches, "agent.skill_commands",
            build_preloaded_skills_prompt=fake_build,
        )

        monkeypatch.setenv("HERMES_ACP_SKILLS", "test-skill")
        try:
            manager = SessionManager(db=sentinel_db)
            agent = manager._make_agent(session_id="acp-skills-session", cwd=".")

            assert isinstance(agent, CapturingAgent)
            assert captured.get("ephemeral_system_prompt") == fake_skills_prompt
            assert called_with["skill_list"] == ["test-skill"]
            assert called_with["task_id"] == "acp-skills-session"
        finally:
            monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)

    def test_no_skills_without_env_var(self, monkeypatch):
        """When HERMES_ACP_SKILLS is absent, no ephemeral_system_prompt is set."""
        captured = {}

        class CapturingAgent(FakeAgent):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        patches = {}
        _patch_module(
            monkeypatch, patches, "run_agent",
            AIAgent=CapturingAgent,
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.config",
            load_config=lambda: {"model": {"default": "m", "provider": "p"}},
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kw: {
                "provider": "p", "api_mode": "chat_completions",
                "base_url": "u", "api_key": "***", "command": None, "args": [],
            },
        )

        monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)

        manager = SessionManager(db=NoopDb())
        agent = manager._make_agent(session_id="acp-no-skills-session", cwd=".")

        assert isinstance(agent, CapturingAgent)
        assert "ephemeral_system_prompt" not in captured

    def test_comma_separated_skills_parsed_into_list(self, monkeypatch):
        """Comma-separated values in HERMES_ACP_SKILLS are parsed into a list."""
        called_with = {}

        def fake_build(skill_list, task_id=None):
            called_with["skill_list"] = skill_list
            return ("[skills loaded]", skill_list, [])

        class CapturingAgent(FakeAgent):
            def __init__(self, **kwargs):
                super().__init__()

        patches = {}
        _patch_module(
            monkeypatch, patches, "run_agent",
            AIAgent=CapturingAgent,
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.config",
            load_config=lambda: {"model": {"default": "m", "provider": "p"}},
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kw: {
                "provider": "p", "api_mode": "chat_completions",
                "base_url": "u", "api_key": "***", "command": None, "args": [],
            },
        )
        _patch_module(
            monkeypatch, patches, "agent.skill_commands",
            build_preloaded_skills_prompt=fake_build,
        )

        monkeypatch.setenv("HERMES_ACP_SKILLS", "skill-a,skill-b,skill-c")
        try:
            manager = SessionManager(db=NoopDb())
            agent = manager._make_agent(session_id="acp-comma-session", cwd=".")

            assert called_with["skill_list"] == ["skill-a", "skill-b", "skill-c"]
        finally:
            monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)

    def test_missing_skills_log_at_debug_only(self, monkeypatch):
        """If build_preloaded_skills_prompt raises, it logs at DEBUG and does not crash."""
        captured = {}
        sentinel_db = NoopDb()

        def fake_build_fail(skill_list, task_id=None):
            raise RuntimeError("skill load failed")

        class CapturingAgent(FakeAgent):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        patches = {}
        _patch_module(
            monkeypatch, patches, "run_agent",
            AIAgent=CapturingAgent,
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.config",
            load_config=lambda: {"model": {"default": "m", "provider": "p"}},
        )
        _patch_module(
            monkeypatch, patches, "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kw: {
                "provider": "p", "api_mode": "chat_completions",
                "base_url": "u", "api_key": "***", "command": None, "args": [],
            },
        )
        _patch_module(
            monkeypatch, patches, "agent.skill_commands",
            build_preloaded_skills_prompt=fake_build_fail,
        )

        monkeypatch.setenv("HERMES_ACP_SKILLS", "broken-skill")
        try:
            manager = SessionManager(db=sentinel_db)
            # Should not raise — failure is caught at DEBUG
            agent = manager._make_agent(session_id="acp-fail-session", cwd=".")
            assert isinstance(agent, CapturingAgent)
            # ephemeral_system_prompt should NOT be set on failure
            assert "ephemeral_system_prompt" not in captured
        finally:
            monkeypatch.delenv("HERMES_ACP_SKILLS", raising=False)