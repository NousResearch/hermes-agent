"""Request-scoped no-tools must survive real construction and conversation reuse."""

import json
import socket
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from openai.types.chat import ChatCompletion

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from run_agent import AIAgent
from tools.registry import registry


@pytest.fixture
def inert_runtime(tmp_path, monkeypatch):
    import gateway.run
    import model_tools

    monkeypatch.chdir(tmp_path)
    real_connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"Unexpected external connection: {address}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setattr(gateway.run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "openai", "api_mode": "chat_completions", "model": "test-model",
        "api_key": "inert-key", "base_url": "http://127.0.0.1:1/v1",
    })
    calls, agents, requests, selections = [], [], [], []
    select_tools = model_tools.get_tool_definitions

    def record_selection(**kwargs):
        selections.append(kwargs)
        return select_tools(**kwargs)

    monkeypatch.setattr(model_tools, "get_tool_definitions", record_selection)
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    for name in ("probe_default", "probe_worker"):
        registry.register(
            name=name, toolset=name,
            schema={"name": name, "description": "Inert test counter", "parameters": {
                "type": "object", "properties": {},
            }},
            handler=lambda args, _name=name, **kwargs: calls.append(_name) or '{"ok": true}',
        )

    def respond(agent, kwargs, **stream_kwargs):
        if agent not in agents:
            agents.append(agent)
        requests.append(kwargs)
        first = not getattr(agent, "_test_called", False)
        agent._test_called = True
        name = "probe_worker" if "worker" in str(agent._session_db.db_path) else "probe_default"
        message = {"role": "assistant", "content": "ok"}
        if first:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_probe", "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }]}
        return ChatCompletion.model_validate({
            "id": "inert", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if first else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setattr(AIAgent, "_interruptible_api_call", respond)
    monkeypatch.setattr(AIAgent, "_interruptible_streaming_api_call", respond)
    yield calls, agents, requests, selections
    for agent in agents:
        agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_request_tool_choice_is_enforced(tmp_path, monkeypatch, inert_runtime, endpoint, stream):
    import yaml

    calls, agents, requests, selections = inert_runtime
    for profile in ("default", "worker", "default"):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr("gateway.run._hermes_home", home)
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": {"default": "test-model", "context_length": 131072},
            "platform_toolsets": {"api_server": [f"probe_{profile}"]},
            "agent": {"max_turns": 4},
            "compression": {"enabled": False},
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "tools": {"tool_search": {"enabled": "off"}},
        }))
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
        app = web.Application()
        app["api_server_adapter"] = adapter
        for method, path, handler in adapter._http_route_table():
            app.router.add_route(method, path, handler)
        payload = ({"messages": [{"role": "user", "content": "probe"}]}
                   if endpoint.endswith("completions") else {"input": "probe", "conversation": "probe"})
        headers = {"Authorization": "Bearer test-key", "X-Hermes-Session-Id": f"probe-{profile}"}
        async with TestClient(TestServer(app)) as client:
            unauthorized = await client.post(endpoint, json={**payload, "tool_choice": "none"})
            assert unauthorized.status == 401
            # Use one conversation: a full-tool turn must not re-enable tools on the next turn.
            for mode in ({"tool_choice": "auto"}, {"tool_choice": "none"}, {}):
                disabled = mode.get("tool_choice") == "none"
                before = len(calls)
                start = len(requests)
                selections_before = len(selections)
                response = await client.post(endpoint, headers=headers, json={
                    **payload, "stream": stream, **mode,
                })
                text = await response.text()
                assert response.status == 200, text
                agent = agents[-1]
                expected = set() if disabled else {f"probe_{profile}"}
                assert agent.valid_tool_names == expected
                assert {tool["function"]["name"] for tool in agent.tools} == expected
                for request in requests[start:]:
                    assert {tool["function"]["name"] for tool in request.get("tools", [])} == expected
                assert calls[before:] == sorted(expected)
                if disabled:
                    assert len(selections) == selections_before
                    _assert_no_tools_can_be_restored_or_dispatched(agent, f"probe_{profile}", calls)
                else:
                    assert len(selections) > selections_before

            for malformed in (None, False, 0, [], {}, "NONE", {"type": "function", "name": 1}):
                before = len(agents)
                response = await client.post(endpoint, headers=headers, json={
                    **payload, "stream": stream, "tool_choice": malformed,
                })
                assert response.status == 400, await response.text()
                assert len(agents) == before

            if not stream:
                # A restricted request cannot reuse an earlier full-tool result.
                key_headers = {**headers, "Idempotency-Key": f"policy-{profile}"}
                response = await client.post(endpoint, headers=key_headers, json={**payload, "tool_choice": "auto"})
                assert response.status == 200, await response.text()
                before = len(agents)
                calls_before = list(calls)
                response = await client.post(endpoint, headers=key_headers, json={**payload, "tool_choice": "none"})
                assert response.status == 200, await response.text()
                assert len(agents) == before + 1
                assert agents[-1].tools == []
                assert calls == calls_before
        if adapter._session_db is not None:
            adapter._session_db.close()


def _assert_no_tools_can_be_restored_or_dispatched(agent, name, calls):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools, restore_agent_tool_prefix

    assert refresh_agent_mcp_tools(agent, enabled_override=[name]) == set()
    assert not restore_agent_tool_prefix(agent, [name])
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    before = list(calls)
    # Bypass model-output validation to inject directly into both real executor paths.
    for dispatch in (agent._execute_tool_calls_sequential, agent._execute_tool_calls_concurrent):
        tool_calls = [SimpleNamespace(id=f"injected-{i}", type="function", function=SimpleNamespace(
            name=name, arguments="{}")) for i in range(2)]
        messages = []
        dispatch(SimpleNamespace(tool_calls=tool_calls), messages, agent.session_id, finalize=False)
        assert calls == before
        assert len(messages) == 2
        assert all("Tools are disabled" in json.loads(message["content"])["error"] for message in messages)
