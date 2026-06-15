"""Integration guard for gateway owner-profile isolation."""

from types import SimpleNamespace

import pytest


def _fake_client_factory():
    class _FakeChatCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="ok",
                            reasoning=None,
                            tool_calls=[],
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeChatCompletions())

    return _FakeClient


@pytest.fixture
def memories_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "USER.md").write_text(
        "**Name:** Aldo Eliacim Alvarez Lemus\n"
        "**What to call them:** Aldo\n",
        encoding="utf-8",
    )
    (mem_dir / "MEMORY.md").write_text(
        "Contact-specific notes that should always survive.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: home)
    monkeypatch.setattr("agent.agent_init.get_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {
            "memory": {
                "memory_enabled": True,
                "user_profile_enabled": True,
                "memory_char_limit": 2200,
                "user_char_limit": 1375,
                "nudge_interval": 10,
            }
        },
    )
    return home


def _build_agent(monkeypatch, *, skip_user_profile):
    import run_agent

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: _fake_client_factory()())
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    return run_agent.AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="telegram",
        max_iterations=2,
        quiet_mode=True,
        skip_context_files=True,
        skip_user_profile=skip_user_profile,
    )


def test_skip_user_profile_true_suppresses_user_md_keeps_memory(
    memories_home,
    monkeypatch,
):
    from agent.system_prompt import build_system_prompt

    agent = _build_agent(monkeypatch, skip_user_profile=True)

    assert agent._user_profile_enabled is False
    assert agent._memory_enabled is True

    prompt = build_system_prompt(agent)
    assert "USER PROFILE (who the user is)" not in prompt
    assert "What to call them" not in prompt
    assert "Contact-specific notes" in prompt


def test_skip_user_profile_false_keeps_user_md(memories_home, monkeypatch):
    from agent.system_prompt import build_system_prompt

    agent = _build_agent(monkeypatch, skip_user_profile=False)

    assert agent._user_profile_enabled is True

    prompt = build_system_prompt(agent)
    assert "USER PROFILE (who the user is)" in prompt
    assert "Contact-specific notes" in prompt
