"""T3 appends runtime context as a separate text block, including for slash controls."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tests.acp_adapter.test_acp_commands import CaptureConn, FakeAgent, NoopDb


RUNTIME_CONTEXT = (
    "<runtime_info>In case you're asked: you are running in T3 Code through the "
    "Hermes harness, as gpt-6-sol. No need to mention this otherwise.</runtime_info>"
    "\n\n<pull_request_linking>When the t3-code MCP server exposes link_pull_request, "
    "register pull requests for this thread.</pull_request_linking>"
)


@pytest.fixture(params=["inprocess", "gateway"])
def control_session(request, tmp_path):
    info = {"model": "gpt-6-sol", "provider": "openai-codex", "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex", "reasoning_effort": "medium"}
    conn, mutations, prompts = CaptureConn(), [], []
    if request.param == "inprocess":
        runtime = FakeAgent()
        for key in ("model", "provider", "api_mode", "base_url"):
            setattr(runtime, key, info[key])
        runtime.reasoning_config = {"enabled": True, "effort": "medium"}
        manager = SessionManager(agent_factory=lambda: runtime, db=NoopDb())
        agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd=str(tmp_path))
        agent.on_connect(conn)
        return SimpleNamespace(agent=agent, sid=state.session_id, conn=conn, state=state,
                               effort=lambda: runtime.reasoning_config["effort"],
                               prompts=runtime.runs, mutations=mutations)

    agent = GatewayACPAgent()
    agent._conn, agent._gateway = conn, AsyncMock()
    snapshot = {"session_id": "s", "revision": 1, "execution_generation": 0, "info": info}
    agent._snapshots["s"] = snapshot

    async def rpc(method, **params):
        if method == "session.info":
            return dict(info)
        if method == "session.resume":
            return {**snapshot, "info": dict(info)}
        if method == "session.mutate":
            mutations.append((params["operation"], params["payload"]))
            if params["operation"] == "reasoning":
                info["reasoning_effort"] = params["payload"]["effort"]
            else:
                info["service_tier"] = None
            return {"status": "applied"}
        assert method == "prompt.submit", method
        prompts.append(params["text"])
        agent._terminals["a"] = {"outcome": "completed"}
        return {"admission_id": "a"}

    agent._gateway.rpc.side_effect = rpc
    return SimpleNamespace(agent=agent, sid="s", conn=conn, state=None,
                           effort=lambda: info["reasoning_effort"], prompts=prompts, mutations=mutations)


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "<runtime_info>Harness context</runtime_info>", RUNTIME_CONTEXT],
                         ids=["plain", "runtime-only", "t3-runtime-and-pr"])
async def test_text_controls_ignore_only_trailing_runtime_context(control_session, suffix):
    session = control_session
    for command, expected in [("/reasoning", "medium"), ("/reasoning status", "medium"),
                              ("/reasoning high --session", "high"), ("/fast normal", "high")]:
        blocks = [acp.text_block(command)] + ([acp.text_block(suffix)] if suffix else [])
        response = await session.agent.prompt(blocks, session.sid)
        assert response.stop_reason == "end_turn"
        messages = [update.content.text for _, update in session.conn.updates
                    if update.session_update == "agent_message_chunk"]
        assert messages[-1] == ("Fast mode: normal (this session)" if command.startswith("/fast") else
                                f"Reasoning: {expected} (this session)") + ("\n" if session.state is None else "")
        assert session.effort() == expected
    assert session.prompts == []
    if session.state is None:
        assert session.mutations == [("reasoning", {"effort": "high"}),
                                     ("service_tier", {"service_tier": "normal"})]


@pytest.mark.asyncio
async def test_multiline_prompt_payloads_retain_runtime_and_user_xml(control_session):
    session = control_session
    payload = "Keep both lines\n<example>user text</example>\n" + RUNTIME_CONTEXT
    await session.agent.prompt([acp.text_block(payload)], session.sid)
    assert session.prompts[-1] == payload
    for command in ("steer", "queue"):
        if session.state is not None:
            session.state.is_running = True
        await session.agent.prompt([acp.text_block(f"/{command} {payload}")], session.sid)
        if session.state is None:
            assert session.prompts[-1] == payload
        elif command == "steer":
            assert session.state.agent.steers[-1] == payload
        else:
            assert session.state.queued_prompts[-1] == payload


def test_control_context_is_a_suffix_not_arbitrary_xml_removal():
    from acp_adapter.config_options import parse_selection, text_selection

    for value in ("high\n<example>user text</example>", "high\n<runtime_info>unclosed",
                  "high\n" + RUNTIME_CONTEXT + "\nextra argument"):
        assert text_selection(value) == " ".join(value.split())
    info = {"model": "gpt-6-sol", "provider": "openai-codex", "api_mode": "codex_responses"}
    for value in ("show", "off"):
        with pytest.raises(ValueError, match=f"Unsupported reasoning effort for this route: {value}"):
            parse_selection("reasoning", text_selection(value + "\n" + RUNTIME_CONTEXT), info)
    with pytest.raises(ValueError, match="session-scoped"):
        text_selection("high --global\n" + RUNTIME_CONTEXT)
