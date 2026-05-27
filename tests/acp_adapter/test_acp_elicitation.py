import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.elicitation import (
    build_clarify_requested_schema,
    create_form_elicitation,
    extract_clarify_answer,
    make_elicitation_clarify_callback,
    supports_form_elicitation,
)
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def test_supports_form_elicitation_with_object_form():
    caps = SimpleNamespace(elicitation=SimpleNamespace(form=SimpleNamespace()))
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_dict_form():
    caps = SimpleNamespace(elicitation={"form": {}})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_empty_elicitation_object():
    # ACP RFD-style shorthand: empty elicitation object means form supported.
    caps = SimpleNamespace(elicitation={})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_missing_is_false():
    assert supports_form_elicitation(SimpleNamespace()) is False
    assert supports_form_elicitation(None) is False


def test_supports_form_elicitation_url_only_is_false():
    caps = SimpleNamespace(elicitation={"url": {}})
    assert supports_form_elicitation(caps) is False


def test_open_ended_clarify_schema_has_required_answer_string():
    schema = build_clarify_requested_schema(question="What branch?", choices=None)
    assert schema["type"] == "object"
    assert schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["type"] == "string"


def test_choice_clarify_schema_preserves_other_path():
    schema = build_clarify_requested_schema(
        question="Which approach?",
        choices=["Strict", "Fallback"],
    )
    props = schema["properties"]
    assert "answer" in props
    assert "other_answer" in props
    assert "Strict" in props["answer"].get("enum", [])
    assert "Fallback" in props["answer"].get("enum", [])
    assert "Other (type your answer)" in props["answer"].get("enum", [])


def test_extract_answer_from_accept_dict():
    response = {"action": "accept", "content": {"answer": "Strict"}}
    assert extract_clarify_answer(response) == "Strict"


def test_extract_other_answer_from_accept_dict():
    response = {
        "action": "accept",
        "content": {"answer": "Other (type your answer)", "other_answer": "Use raw JSON-RPC"},
    }
    assert extract_clarify_answer(response) == "Use raw JSON-RPC"


def test_extract_answer_from_accept_object():
    response = SimpleNamespace(action="accept", content={"answer": "Fallback"})
    assert extract_clarify_answer(response) == "Fallback"


def test_decline_cancel_or_missing_returns_standard_sentinel():
    assert extract_clarify_answer({"action": "decline"}) == "[user declined the clarification]"
    assert extract_clarify_answer({"action": "cancel"}) == "[user cancelled the clarification]"
    assert extract_clarify_answer(None) == "[clarify prompt could not be delivered]"


class TypedConn:
    def __init__(self):
        self.calls = []

    async def create_elicitation(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": "accept", "content": {"answer": "typed"}}


class RawConn:
    def __init__(self):
        self.requests = []
        self._conn = self

    async def send_request(self, method, params):
        self.requests.append((method, params))
        return {"action": "accept", "content": {"answer": "raw"}}


@pytest.mark.asyncio
async def test_create_form_elicitation_prefers_typed_helper_if_available():
    conn = TypedConn()
    response = await create_form_elicitation(
        conn,
        session_id="s1",
        question="Q?",
        choices=None,
    )
    assert response["content"]["answer"] == "typed"
    assert conn.calls[0].get("session_id") == "s1" or conn.calls[0].get("sessionId") == "s1"


@pytest.mark.asyncio
async def test_create_form_elicitation_uses_raw_json_rpc_fallback():
    conn = RawConn()
    response = await create_form_elicitation(
        conn,
        session_id="s1",
        question="Q?",
        choices=["A", "B"],
    )
    assert response["content"]["answer"] == "raw"
    method, params = conn.requests[0]
    assert method == "elicitation/create"
    assert params["sessionId"] == "s1"
    assert params["mode"] == "form"
    assert params["message"] == "Q?"
    assert params["requestedSchema"]["required"] == ["answer"]


def test_elicitation_clarify_callback_schedules_on_loop():
    conn = RawConn()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(timeout=2)

    try:
        callback = make_elicitation_clarify_callback(conn, loop, "s1", timeout=2)
        assert callback("Q?", ["A", "B"]) == "raw"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


@pytest.mark.asyncio
async def test_initialize_records_elicitation_capability():
    agent = HermesACPAgent()
    caps = SimpleNamespace(elicitation={"form": {}})

    await agent.initialize(client_capabilities=caps)

    assert agent._supports_elicitation_form() is True
    assert agent._client_capabilities is caps


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def replace_messages(self, *_args, **_kwargs):
        return None


class ToolSurfaceFakeAgent:
    def __init__(self):
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.invalidated = 0
        self.model = "fake-model"
        self.provider = "fake-provider"

    def _invalidate_system_prompt(self):
        self.invalidated += 1


class ToolSurfaceConn:
    async def session_update(self, *_args, **_kwargs):
        return None

    async def request_permission(self, *_args, **_kwargs):
        return SimpleNamespace(outcome="allow")


def install_fake_model_tools(monkeypatch):
    calls = []

    def get_tool_definitions(*, enabled_toolsets, disabled_toolsets=None, quiet_mode=True):
        calls.append(list(enabled_toolsets))
        tools = []
        for toolset in enabled_toolsets:
            if toolset == "clarify":
                tools.append({"function": {"name": "clarify"}})
            elif toolset.startswith("mcp-"):
                tools.append({"function": {"name": f"{toolset}-tool"}})
            else:
                tools.append({"function": {"name": f"{toolset}-tool"}})
        return tools

    module = ModuleType("model_tools")
    setattr(module, "get_tool_definitions", get_tool_definitions)
    monkeypatch.setitem(sys.modules, "model_tools", module)
    return calls


@pytest.mark.asyncio
async def test_acp_toolsets_add_clarify_only_for_form_elicitation():
    supported = HermesACPAgent()
    await supported.initialize(client_capabilities=SimpleNamespace(elicitation={"form": {}}))
    assert "clarify" in supported._acp_toolsets_for_client(["hermes-acp"])

    unsupported = HermesACPAgent()
    await unsupported.initialize(client_capabilities=SimpleNamespace(elicitation={"url": {}}))
    assert "clarify" not in unsupported._acp_toolsets_for_client(["hermes-acp", "clarify"])


@pytest.mark.asyncio
async def test_refresh_agent_tool_surface_gates_clarify_and_preserves_mcp(monkeypatch):
    calls = install_fake_model_tools(monkeypatch)
    fake = ToolSurfaceFakeAgent()
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    await agent.initialize(client_capabilities=SimpleNamespace(elicitation={"form": {}}))
    state = manager.create_session(cwd=".")

    agent._refresh_agent_tool_surface(state, mcp_server_names=["demo"])

    assert "clarify" in state.agent.enabled_toolsets
    assert "mcp-demo" in state.agent.enabled_toolsets
    assert "clarify" in state.agent.valid_tool_names
    assert "mcp-demo-tool" in state.agent.valid_tool_names
    assert state.agent.invalidated == 1
    assert "clarify" in calls[-1]


@pytest.mark.asyncio
async def test_new_session_refreshes_supported_client_tool_surface(monkeypatch):
    install_fake_model_tools(monkeypatch)
    fake = ToolSurfaceFakeAgent()
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    agent.on_connect(ToolSurfaceConn())
    await agent.initialize(client_capabilities=SimpleNamespace(elicitation={"form": {}}))

    response = await agent.new_session(cwd=".")
    state = manager.get_session(response.session_id)

    assert state is not None
    assert "clarify" in state.agent.valid_tool_names


class ClarifyFakeAgent(ToolSurfaceFakeAgent):
    def __init__(self):
        super().__init__()
        self.clarify_callback: object = None
        self.clarify_answers = []

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        answer = None
        if callable(self.clarify_callback):
            answer = self.clarify_callback("Q?", ["A"])
            self.clarify_answers.append(answer)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": f"answer: {answer}"})
        return {"final_response": f"answer: {answer}", "messages": messages}


class ElicitationConn(ToolSurfaceConn):
    def __init__(self):
        self.updates = []
        self.requests = []
        self.request_permission_calls = []
        self._conn = self

    async def session_update(self, *args, **kwargs):
        self.updates.append((args, kwargs))

    async def request_permission(self, *args, **kwargs):
        self.request_permission_calls.append((args, kwargs))
        return SimpleNamespace(outcome="allow")

    async def send_request(self, method, params):
        self.requests.append((method, params))
        return {"action": "accept", "content": {"answer": "Elicited"}}


@pytest.mark.asyncio
async def test_prompt_wires_clarify_callback_to_elicitation(monkeypatch):
    install_fake_model_tools(monkeypatch)
    fake = ClarifyFakeAgent()
    original_callback = lambda _q, _c: "original"
    fake.clarify_callback = original_callback
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    conn = ElicitationConn()
    agent.on_connect(conn)
    await agent.initialize(client_capabilities=SimpleNamespace(elicitation={"form": {}}))
    response = await agent.new_session(cwd=".")

    await agent.prompt(
        session_id=response.session_id,
        prompt=[TextContentBlock(type="text", text="ask if needed")],
    )

    assert fake.clarify_answers == ["Elicited"]
    assert conn.requests[0][0] == "elicitation/create"
    assert conn.request_permission_calls == []
    assert fake.clarify_callback is original_callback


@pytest.mark.asyncio
async def test_unsupported_client_new_session_has_no_clarify_tool(monkeypatch):
    install_fake_model_tools(monkeypatch)
    fake = ToolSurfaceFakeAgent()
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    agent.on_connect(ToolSurfaceConn())
    await agent.initialize(client_capabilities=None)

    response = await agent.new_session(cwd=".")
    state = manager.get_session(response.session_id)

    assert state is not None
    assert "clarify" not in state.agent.enabled_toolsets
    assert "clarify" not in state.agent.valid_tool_names


@pytest.mark.asyncio
async def test_url_only_elicitation_client_has_no_clarify_tool(monkeypatch):
    install_fake_model_tools(monkeypatch)
    fake = ToolSurfaceFakeAgent()
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    agent.on_connect(ToolSurfaceConn())
    await agent.initialize(client_capabilities=SimpleNamespace(elicitation={"url": {}}))

    response = await agent.new_session(cwd=".")
    state = manager.get_session(response.session_id)

    assert state is not None
    assert "clarify" not in state.agent.enabled_toolsets
    assert "clarify" not in state.agent.valid_tool_names


@pytest.mark.asyncio
async def test_unsupported_client_does_not_overwrite_existing_clarify_callback(monkeypatch):
    install_fake_model_tools(monkeypatch)
    fake = ClarifyFakeAgent()
    original_callback = lambda _q, _c: "original"
    fake.clarify_callback = original_callback
    manager = SessionManager(agent_factory=lambda: fake, db=NoopDb())
    agent = HermesACPAgent(session_manager=manager)
    conn = ElicitationConn()
    agent.on_connect(conn)
    await agent.initialize(client_capabilities=SimpleNamespace(elicitation={"url": {}}))
    response = await agent.new_session(cwd=".")

    await agent.prompt(
        session_id=response.session_id,
        prompt=[TextContentBlock(type="text", text="ask normally if needed")],
    )

    assert fake.clarify_answers == ["original"]
    assert conn.requests == []
    assert fake.clarify_callback is original_callback
