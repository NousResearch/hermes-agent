import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from agent import skill_commands
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.service_tier: str | None = None
        self.request_overrides: dict[str, str] | None = None
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self._supports_active_turn_redirect = True
        self.steers = []
        self.redirects = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def redirect(self, text):
        self.redirects.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        if kwargs:
            self.updates.append((kwargs.get("session_id"), kwargs.get("update")))
        else:
            self.updates.append((args[0], args[1]))

    async def request_permission(self, *args, **kwargs):
        return SimpleNamespace(outcome="allow")


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def make_agent_and_state():
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


def install_skill(tmp_path, monkeypatch, name="acp-helper"):
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Help with ACP tasks.
---

# ACP helper

Follow the ACP helper instructions.
"""
    )
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_root)
    skill_commands.scan_skill_commands()


def test_acp_advertises_installed_skill_commands(tmp_path, monkeypatch):
    install_skill(tmp_path, monkeypatch)

    commands = {command.name: command for command in HermesACPAgent._available_commands()}

    assert commands["acp-helper"].description == "Help with ACP tasks."


def test_acp_advertises_fast_command():
    commands = {command.name: command for command in HermesACPAgent._available_commands()}

    assert commands["fast"].input is not None
    assert commands["fast"].input.root.hint == "normal|fast|status"


@pytest.mark.asyncio
async def test_acp_fast_command_is_session_scoped_and_intercepted():
    acp_agent, state, fake, conn = make_agent_and_state()
    state.model = fake.model = "gpt-5.4"

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/fast fast")],
    )

    assert fake.runs == []
    assert state.fast_mode is True
    assert fake.service_tier == "priority"
    assert fake.request_overrides == {"service_tier": "priority"}
    assert any("Fast mode: fast" in update.content.text for _, update in conn.updates)


@pytest.mark.asyncio
async def test_acp_fast_rejects_unsupported_model_without_mutation():
    acp_agent, state, fake, conn = make_agent_and_state()
    state.model = fake.model = "unsupported-model"

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/fast fast")],
    )

    assert fake.runs == []
    assert state.fast_mode is False
    assert fake.service_tier is None
    assert any("not available" in update.content.text for _, update in conn.updates)


def test_acp_model_switch_preserves_fast_mode(monkeypatch):
    agents = [FakeAgent(), FakeAgent()]
    agents[0].model = agents[1].model = "gpt-5.4"
    manager = SessionManager(agent_factory=lambda: agents.pop(0), db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    monkeypatch.setattr(
        acp_agent,
        "_resolve_model_selection",
        lambda *_args: ("openai-codex", "gpt-5.4"),
    )

    assert "Fast mode: fast" in acp_agent._cmd_fast("fast", state)
    acp_agent._cmd_model("gpt-5.4", state)

    assert state.fast_mode is True
    assert state.agent.service_tier == "priority"
    assert state.agent.request_overrides == {"service_tier": "priority"}


def test_acp_reset_restores_configured_fast_mode(monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.model = fake.model = "gpt-5.4"
    state.fast_mode = True
    fake.service_tier = "priority"
    fake.request_overrides = {"service_tier": "priority"}
    monkeypatch.setattr("acp_adapter.server.configured_fast_mode", lambda: False)

    acp_agent._cmd_reset("", state)

    assert state.fast_mode is False
    assert fake.service_tier is None
    assert fake.request_overrides is None


@pytest.mark.asyncio
async def test_acp_help_lists_installed_skill_commands(tmp_path, monkeypatch):
    install_skill(tmp_path, monkeypatch)
    acp_agent, state, fake, conn = make_agent_and_state()

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/help")],
    )

    assert fake.runs == []
    assert any("/acp-helper" in update.content.text for _, update in conn.updates)


@pytest.mark.asyncio
async def test_acp_skill_slash_command_loads_skill_for_agent(tmp_path, monkeypatch):
    install_skill(tmp_path, monkeypatch)
    acp_agent, state, fake, _conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/acp-helper fix command discovery")],
    )

    assert response.stop_reason == "end_turn"
    assert len(fake.runs) == 1
    assert "Follow the ACP helper instructions." in fake.runs[0]
    assert "fix command discovery" in fake.runs[0]


@pytest.mark.asyncio
async def test_acp_builtin_slash_command_takes_precedence_over_same_named_skill(
    tmp_path, monkeypatch
):
    install_skill(tmp_path, monkeypatch, name="help")
    acp_agent, state, fake, conn = make_agent_and_state()

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/help")],
    )

    assert fake.runs == []
    assert any("Available commands:" in update.content.text for _, update in conn.updates)


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    captured = {}
    sentinel_db = NoopDb()

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m", "provider": "p"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "p",
                "api_mode": "chat_completions",
                "base_url": "u",
                "api_key": "k",
                "command": None,
                "args": [],
            },
        ),
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, CapturingAgent)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["session_id"] == "acp-session"


@pytest.mark.asyncio
async def test_acp_steer_slash_command_injects_into_running_agent():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer prefer the simpler fix")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == ["prefer the simpler fix"]
    assert fake.runs == []








@pytest.mark.asyncio
async def test_acp_cancel_publishes_hard_stop_while_holding_runtime_lock():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "original request"
    observed = {}

    def interrupt():
        acquired = state.runtime_lock.acquire(blocking=False)
        observed["lock_held"] = not acquired
        if acquired:
            state.runtime_lock.release()

    fake.interrupt = interrupt

    await acp_agent.cancel(state.session_id)

    assert observed["lock_held"] is True
    assert state.cancel_event.is_set()
    assert state.interrupted_prompt_text == "original request"






