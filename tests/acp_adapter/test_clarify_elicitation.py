"""ACP ``clarify`` over form elicitation (#29978): a client that advertises
``elicitation.form`` in ``initialize`` gets the question as ``elicitation/create`` on the real
ACP wire and its answer back in clarify's result; any other client keeps today's behavior."""

import asyncio
import contextlib
import json
import os
from types import SimpleNamespace

import acp
import pytest
from acp.connection import StreamDirection, StreamEvent

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.clarify_tool import clarify_tool

_FORM = {"clientCapabilities": {"elicitation": {"form": {}}}}


@contextlib.asynccontextmanager
async def _acp_wire(agent):
    """Run ``agent`` behind the SDK's real stdio connection; yield raw JSON-RPC send/receive."""
    loop = asyncio.get_running_loop()
    client_to_agent_r, client_to_agent_w = os.pipe()
    agent_to_client_r, agent_to_client_w = os.pipe()
    to_agent = os.fdopen(client_to_agent_w, "wb", buffering=0)
    agent_input = asyncio.StreamReader(loop=loop)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(agent_input, loop=loop),
                                 os.fdopen(client_to_agent_r, "rb", buffering=0))
    transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin,
                                                        os.fdopen(agent_to_client_w, "wb", buffering=0))
    agent_output = asyncio.StreamWriter(transport, protocol, None, loop)
    from_agent = asyncio.StreamReader(loop=loop)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(from_agent, loop=loop),
                                 os.fdopen(agent_to_client_r, "rb", buffering=0))
    task = asyncio.create_task(acp.run_agent(agent, input_stream=agent_output, output_stream=agent_input,
                                             use_unstable_protocol=True))

    def send(message: dict) -> None:
        to_agent.write((json.dumps(message) + "\n").encode())

    async def receive() -> dict:
        return json.loads(await asyncio.wait_for(from_agent.readline(), timeout=10))

    try:
        yield send, receive
    finally:
        to_agent.close()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, timeout=5)
        transport.close()


async def _clarify_over_acp(initialize_params: dict, tool_kwargs: dict, reply: dict | None = None):
    """Initialize with ``initialize_params``, wire one turn, run clarify on a worker thread and
    answer the elicitation it sends (if any) with ``reply``. Returns ``(elicitation, result)``."""
    agent = HermesACPAgent(session_manager=SessionManager(db=None))
    async with _acp_wire(agent) as (send, receive):
        send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1, **initialize_params}})
        assert "result" in await receive()
        state = SimpleNamespace(agent=SimpleNamespace(clarify_callback=None), message_ids=None)
        agent._wire_turn_callbacks(state, "sess-1", agent._conn, asyncio.get_running_loop())
        pending = asyncio.ensure_future(
            asyncio.to_thread(clarify_tool, callback=state.agent.clarify_callback, **tool_kwargs))
        elicitation = None
        if reply is not None:
            elicitation = await receive()
            send({"jsonrpc": "2.0", "id": elicitation["id"], "result": reply})
        return elicitation, json.loads(await asyncio.wait_for(pending, timeout=10))


_BATCH = [{"question": "Name?"}, {"question": "Color?", "choices": ["red", "blue"]}]


@pytest.mark.platforms("linux", "macos")
@pytest.mark.asyncio
@pytest.mark.parametrize("tool_kwargs, properties, content, answer", [
    ({"question": "Which db?", "choices": ["Postgres", "SQLite"]},
     {"answer": {"type": "string", "enum": ["Postgres (Recommended)", "SQLite"]}},
     {"answer": "Postgres (Recommended)"}, "Postgres"),
    ({"question": "Which?", "choices": ["a", "b", "c"], "multi_select": True},
     {"answer": {"type": "array", "items": {"type": "string", "enum": ["a (Recommended)", "b", "c"]}}},
     {"answer": ["a (Recommended)", "c"]}, ["a", "c"]),
    ({"question": "When?"}, {"answer": {"type": "string"}}, {"answer": "Tomorrow"}, "Tomorrow"),
    ({"question": "", "questions": _BATCH},
     {"q0": {"type": "string", "title": "Name?"},
      "q1": {"type": "string", "enum": ["red (Recommended)", "blue"], "title": "Color?"}},
     {"q0": "Ada", "q1": "blue"}, ["Ada", "blue"]),
], ids=["single", "multi", "open", "batch"])
async def test_form_client_answers_clarify_through_elicitation(tool_kwargs, properties, content, answer):
    elicitation, result = await _clarify_over_acp(_FORM, tool_kwargs, {"action": "accept", "content": content})

    assert elicitation["method"] == "elicitation/create"
    params = elicitation["params"]
    assert (params["sessionId"], params["mode"]) == ("sess-1", "form")
    assert params["requestedSchema"] == {"type": "object", "properties": properties, "required": list(properties)}
    if "questions" in tool_kwargs:
        assert [row["user_response"] for row in result["responses"]] == answer
        assert "timed_out" not in result
    else:
        assert params["message"] == tool_kwargs["question"]
        assert result["user_response"] == answer


@pytest.mark.platforms("linux", "macos")
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["decline", "cancel"])
@pytest.mark.parametrize("tool_kwargs", [{"question": "Which db?", "choices": ["Postgres", "SQLite"]},
                                         {"question": "", "questions": _BATCH}], ids=["single", "batch"])
async def test_declined_or_cancelled_elicitation_is_a_skip(action, tool_kwargs):
    """No answer is a skip (blank responses), not a timeout: the user did respond."""
    _, result = await _clarify_over_acp(_FORM, tool_kwargs, {"action": action})

    responses = result.get("responses") or [result]
    assert [row["user_response"] for row in responses] == [""] * len(responses)
    assert "timed_out" not in result


@pytest.mark.platforms("linux", "macos")
@pytest.mark.asyncio
async def test_client_without_form_elicitation_keeps_clarify_unavailable():
    _, result = await _clarify_over_acp({"clientCapabilities": {"elicitation": {"url": {}}}},
                                        {"question": "Which db?", "choices": ["Postgres", "SQLite"]})

    assert result == {"error": "Clarify tool is not available in this execution context."}


@pytest.mark.parametrize("initialize_params, offered", [
    (_FORM, True),
    ({"capabilities": {"elicitation": {"form": {}}}}, True),
    ({"clientCapabilities": {"elicitation": {"url": {}}}}, False),
    ({"clientCapabilities": {}}, False),
], ids=["v1-form", "v2-form", "url-only", "none"])
def test_fresh_session_offers_clarify_only_to_form_elicitation_clients(monkeypatch, initialize_params, offered):
    """``hermes-acp`` withholds clarify; a form-capable client gets it on every fresh agent build."""
    from model_tools import get_tool_definitions

    seen: list[dict] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: {})
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)
    agent = HermesACPAgent(session_manager=SessionManager(db=None))

    agent._observe_client_message(StreamEvent(
        StreamDirection.INCOMING, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": initialize_params}))
    agent.session_manager._make_agent(session_id="fresh", cwd=".")

    names = {tool["function"]["name"] for tool in get_tool_definitions(
        enabled_toolsets=seen[0]["enabled_toolsets"], disabled_toolsets=seen[0]["disabled_toolsets"], quiet_mode=True)}
    assert ("clarify" in names) is offered
