import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, _expand_acp_enabled_toolsets


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
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


def _capture_real_agent_kwargs(monkeypatch, config):
    captured = {}

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    module = ModuleType("run_agent")
    setattr(module, "AIAgent", CapturingAgent)
    monkeypatch.setitem(sys.modules, "run_agent", module)

    import hermes_cli.config
    import hermes_cli.runtime_provider

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: config)
    monkeypatch.setattr(
        hermes_cli.runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "p",
            "api_mode": "chat_completions",
            "base_url": "u",
            "api_key": "k",
            "command": None,
            "args": [],
        },
    )
    return captured, CapturingAgent


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    sentinel_db = NoopDb()
    captured, agent_type = _capture_real_agent_kwargs(
        monkeypatch,
        {"model": {"default": "m", "provider": "p"}},
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, agent_type)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["session_id"] == "acp-session"


def test_acp_real_agent_honors_explicit_platform_toolsets(monkeypatch):
    captured, _ = _capture_real_agent_kwargs(
        monkeypatch,
        {
            "model": {"default": "m", "provider": "p"},
            "platform_toolsets": {"acp": ["web", "scoped"]},
            "mcp_servers": {"scoped": {"command": "example"}},
        },
    )

    SessionManager(db=NoopDb())._make_agent(session_id="acp-session", cwd=".")

    assert set(captured["enabled_toolsets"]) == {"web", "mcp-scoped"}


def test_acp_real_agent_preserves_default_toolsets_when_platform_config_absent(monkeypatch):
    captured, _ = _capture_real_agent_kwargs(
        monkeypatch,
        {
            "model": {"default": "m", "provider": "p"},
            "mcp_servers": {"global": {"command": "example"}},
        },
    )

    SessionManager(db=NoopDb())._make_agent(session_id="acp-session", cwd=".")

    assert captured["enabled_toolsets"] == ["hermes-acp", "mcp-global"]


def test_acp_real_agent_honors_disabled_toolsets(monkeypatch):
    captured, _ = _capture_real_agent_kwargs(
        monkeypatch,
        {
            "model": {"default": "m", "provider": "p"},
            "agent": {"disabled_toolsets": ["terminal", "memory"]},
        },
    )

    SessionManager(db=NoopDb())._make_agent(session_id="acp-session", cwd=".")

    assert captured["enabled_toolsets"] == ["hermes-acp"]
    assert captured["disabled_toolsets"] == ["terminal", "memory"]


def test_acp_expansion_preserves_explicit_empty_toolsets():
    assert _expand_acp_enabled_toolsets([], ["scoped"]) == ["mcp-scoped"]


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






