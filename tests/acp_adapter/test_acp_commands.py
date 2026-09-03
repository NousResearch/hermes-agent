import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB
from session_rollover import TurnBoundaryRollover


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


def test_acp_restart_parent_handle_restores_turn_boundary_child_tip(monkeypatch, tmp_path):
    """The ACP handle remains stable while its agent resumes the rollover tip."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "acp-parent",
        source="acp",
        model="parent-model",
        model_config={"cwd": "/parent"},
    )
    db.append_message("acp-parent", "user", "ended parent transcript")
    child_id = TurnBoundaryRollover(db).mark_pending("acp-parent", threshold_tokens=1)
    assert child_id is True
    child_id = TurnBoundaryRollover(db).adopt_at_turn_boundary("acp-parent", active_work=False)
    assert child_id
    db.update_session_meta(
        child_id,
        json.dumps({"cwd": "/child", "provider": "child-provider"}),
        "child-model",
    )
    db.append_message(child_id, "user", "child continuation transcript")

    made = []

    class _Agent:
        model = "factory-model"

    def factory(**_kwargs):
        return _Agent()

    manager = SessionManager(agent_factory=factory, db=db)
    original_make_agent = manager._make_agent

    def capture_make_agent(**kwargs):
        made.append(kwargs)
        return original_make_agent(**kwargs)

    manager._make_agent = capture_make_agent
    restored = manager.get_session("acp-parent")

    assert restored is not None
    assert restored.session_id == "acp-parent"
    assert restored.cwd == "/child"
    assert restored.model == "child-model"
    assert [message["content"] for message in restored.history] == [
        "child continuation transcript"
    ]
    assert made == [{
        "session_id": child_id,
        "cwd": "/child",
        "model": "child-model",
        "requested_provider": "child-provider",
        "base_url": None,
        "api_mode": None,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("switch_path", ("slash", "protocol"))
async def test_acp_model_switch_persists_to_rollover_child_not_stable_parent(
    monkeypatch, tmp_path, switch_path,
):
    """The ACP handle stays stable while model/history writes target the child tip."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "acp-parent", source="acp", model="parent-model", model_config={"cwd": "/parent"},
    )
    db.append_message("acp-parent", "user", "parent-only")
    assert TurnBoundaryRollover(db).mark_pending("acp-parent", threshold_tokens=1) is True
    child_id = TurnBoundaryRollover(db).adopt_at_turn_boundary("acp-parent", active_work=False)
    assert child_id
    db.update_session_meta(child_id, json.dumps({"cwd": "/child"}), "child-model")
    db.append_message(child_id, "user", "child-only")

    made = []

    class _Agent:
        model = "factory-model"
        provider = "factory-provider"

    manager = SessionManager(agent_factory=_Agent, db=db)
    original_make_agent = manager._make_agent

    def capture_make_agent(**kwargs):
        made.append(kwargs)
        return original_make_agent(**kwargs)

    manager._make_agent = capture_make_agent
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.get_session("acp-parent")
    assert state is not None

    if switch_path == "slash":
        assert "Model switched to: next-model" in acp_agent._cmd_model("next-model", state)
    else:
        assert await acp_agent.set_session_model("next-model", "acp-parent") is not None

    parent = db.get_session("acp-parent")
    child = db.get_session(child_id)
    assert parent is not None
    assert child is not None
    assert parent["model"] == "parent-model"
    assert [message["content"] for message in db.get_messages_as_conversation("acp-parent")] == ["parent-only"]
    assert child["model"] == "next-model"
    assert [message["content"] for message in db.get_messages_as_conversation(child_id)] == ["child-only"]
    assert made[-1]["session_id"] == child_id

    restarted = SessionManager(agent_factory=_Agent, db=db).get_session("acp-parent")
    assert restarted is not None
    assert restarted.session_id == "acp-parent"
    assert restarted.model == "next-model"
    assert [message["content"] for message in restarted.history] == ["child-only"]


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






