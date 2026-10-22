"""Real-agent probes for process-local frontend continuation, no core patch."""

import copy
import json
import shlex

import httpx
import pytest

from fastapi.testclient import TestClient

from agui_adapter.server import create_app
from hermes_cli.config import save_config
from tests.agui_adapter.test_modernization_probe import body, config_for, events


def frontend_request(peer, thread="parked-front"):
    save_config({"auxiliary": {"title_generation": {"enabled": False}}})
    return body(
        thread,
        [
            {
                "name": peer.tool_name,
                "description": "Browser probe",
                "parameters": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                },
            }
        ],
    )


def result_request(request, call, content="CLIENT_RESULT"):
    resumed = copy.deepcopy(request)
    resumed["runId"] += "-next"
    resumed["messages"] += [
        {
            "id": "assistant-" + call["toolCallId"],
            "role": "assistant",
            "content": "",
            "toolCalls": [
                {
                    "id": call["toolCallId"],
                    "type": "function",
                    "function": {"name": call["toolCallName"], "arguments": "{}"},
                }
            ],
        },
        {
            "id": "result-" + call["toolCallId"],
            "role": "tool",
            "toolCallId": call["toolCallId"],
            "content": content,
        },
    ]
    return resumed


def error_events(response):
    assert response.status_code == 200
    result = [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert result[-1]["type"] == "RUN_ERROR", result
    return result


def test_orphan_result_never_starts_an_agent(peer):
    request = frontend_request(peer)
    request = result_request(
        request, {"toolCallId": "orphan", "toolCallName": peer.tool_name}
    )
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        error_events(client.post("/", json=request))
    assert peer.requests == []


@pytest.mark.parametrize("changed_context", [False, True])
@pytest.mark.parametrize("error", [None, "Browser failed to apply action"])
def test_wrong_id_does_not_consume_live_wait_and_duplicate_does_not_replay(
    peer, changed_context, error
):
    request = frontend_request(peer)
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call = next(item for item in first if item["type"] == "TOOL_CALL_START")
        lifecycle = [item for item in first if item["type"].startswith("TOOL_CALL_")]
        assert [item["type"] for item in lifecycle] == [
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
        ]
        assert {item["toolCallId"] for item in lifecycle} == {"probe-call-1"}
        assert json.loads(lifecycle[1]["delta"]) == peer.tool_args
        assert first.index(lifecycle[-1]) < len(first) - 1
        assert first[-1]["type"] == "RUN_FINISHED"
        good = result_request(request, call, "" if error else "CLIENT_RESULT")
        good["messages"][-1]["error"] = error
        if changed_context:
            good["state"] = {"updated": True}
        relay = client.app.state.executions.get(request["threadId"])
        wrong = copy.deepcopy(good)
        wrong["messages"][-1]["toolCallId"] = "wrong-id"
        error_events(client.post("/", json=wrong))
        assert len(peer.requests) == 1
        events(client.post("/", json=good))
        assert client.app.state.executions.conversations.get(request["threadId"])
        assert len(peer.requests) == 2
        error_events(client.post("/", json=good))
        assert len(peer.requests) == 2
        assert relay.agent._user_turn_count == 1
    before, after = peer.requests
    assert after["messages"][: len(before["messages"])] == before["messages"]
    returned = after["messages"][-1]
    assert (returned["role"], returned["tool_call_id"]) == ("tool", call["toolCallId"])
    content = returned["content"]
    if changed_context:
        envelope = json.loads(content)
        assert envelope["adapter_context"]["state"] == good["state"]
        content = envelope["client_tool_result"]
    if error:
        assert error in content and "error" in content.lower()
    else:
        assert content == "CLIENT_RESULT"


@pytest.mark.parametrize("declaration", ["tools", "stateWriterTools"])
def test_schema_change_and_fresh_run_do_not_consume_live_wait(peer, declaration):
    request = frontend_request(peer)
    request["state"] = {"nested": [1]}
    request["tools"][0]["parameters"]["const"] = 1
    request["forwardedProps"] = {
        "stateWriterTools": [
            {"name": "set_probe", "parameters": {"type": "object", "const": 1}}
        ]
    }
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call = next(item for item in first if item["type"] == "TOOL_CALL_START")
        good = result_request(request, call)
        changed = copy.deepcopy(good)
        definitions = (
            changed["tools"]
            if declaration == "tools"
            else changed["forwardedProps"][declaration]
        )
        definitions[0]["parameters"]["const"] = True
        error_events(client.post("/", json=changed))
        relay = client.app.state.executions.get(request["threadId"])
        pending = relay.pending
        assert not pending.done()
        malformed = copy.deepcopy(good)
        malformed["forwardedProps"] = {"stateWriterTools": [{"name": []}]}
        error_events(client.post("/", json=malformed))
        assert relay.pending is pending and not pending.done()
        error_events(client.post("/", json=request))
        assert len(peer.requests) == 1
        good["tools"][0]["parameters"] = dict(
            reversed(list(good["tools"][0]["parameters"].items()))
        )
        good["state"] = {"nested": [True]}
        events(client.post("/", json=good))
        assert len(peer.requests) == 2
    before, after = peer.requests
    assert after["messages"][: len(before["messages"])] == before["messages"]
    assert after["tools"] == before["tools"]
    result = json.loads(after["messages"][-1]["content"])
    assert result["adapter_context"]["state"]["nested"][0] is True


def model_call(call_id, name, **args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def script_calls(peer, calls):
    peer.script = lambda request, index: (
        {"role": "assistant", "content": None, "tool_calls": copy.deepcopy(calls)}
        if index == 0
        else {"role": "assistant", "content": "SCRIPT_DONE"}
    )


def test_invalid_frontend_arguments_are_not_dispatched_to_the_browser(peer):
    request = frontend_request(peer)
    invalid = model_call("invalid-front", peer.tool_name)
    invalid["function"]["arguments"] = "[]"
    script_calls(peer, [invalid])
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        result = events(client.post("/", json=request))
        assert not any(event["type"] == "TOOL_CALL_START" for event in result), (
            "Undispatched frontend call was handed to the browser"
        )
        assert "SCRIPT_DONE" in "".join(event.get("delta", "") for event in result)
    assert len(peer.requests) == 2
    returned = peer.requests[1]["messages"][-1]
    assert returned["role"] == "tool"
    assert returned["tool_call_id"] == "invalid-front"
    assert "error" in returned["content"].lower()


def test_repeated_frontend_name_in_one_batch_correlates_by_id(peer):
    request = frontend_request(peer)
    script_calls(
        peer,
        [
            model_call("front-1", peer.tool_name, color="blue"),
            model_call("front-2", peer.tool_name, color="green"),
        ],
    )
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call1 = next(event for event in first if event["type"] == "TOOL_CALL_START")
        assert call1["toolCallId"] == "front-1"
        request = result_request(request, call1, "BLUE_RESULT")
        second = events(client.post("/", json=request))
        call2 = next(event for event in second if event["type"] == "TOOL_CALL_START")
        assert call2["toolCallId"] == "front-2"
        assert len(peer.requests) == 1
        stale = copy.deepcopy(request)
        stale["runId"] += "-stale"
        error_events(client.post("/", json=stale))
        request = result_request(request, call2, "GREEN_RESULT")
        final = events(client.post("/", json=request))
        assert "SCRIPT_DONE" in "".join(event.get("delta", "") for event in final), (
            final
        )
    results = [
        message for message in peer.requests[1]["messages"] if message["role"] == "tool"
    ]
    assert [(message["tool_call_id"], message["content"]) for message in results] == [
        ("front-1", "BLUE_RESULT"),
        ("front-2", "GREEN_RESULT"),
    ]


def test_server_effects_before_and_after_frontend_execute_once(peer, tmp_path):
    from tools.registry import registry

    request = frontend_request(peer)
    marker = tmp_path / "effects.txt"

    def effect(args, **kwargs):
        with marker.open("a") as stream:
            stream.write(args["phase"] + "\n")
        return "SERVER_EFFECT_" + args["phase"]

    registry.register(
        name="agui_probe_server_effect",
        toolset="agui-probe",
        handler=effect,
        schema={
            "name": "agui_probe_server_effect",
            "description": "Record test effect",
            "parameters": {
                "type": "object",
                "properties": {"phase": {"type": "string"}},
            },
        },
        check_fn=lambda: True,
    )
    script_calls(
        peer,
        [
            model_call("before", "agui_probe_server_effect", phase="before"),
            model_call("browser", peer.tool_name, color="blue"),
            model_call("after", "agui_probe_server_effect", phase="after"),
        ],
    )
    try:
        with TestClient(
            create_app(config_for(peer, ["agui-probe"])), base_url="http://127.0.0.1"
        ) as client:
            first = events(client.post("/", json=request))
            assert marker.exists(), first
            assert marker.read_text().splitlines() == ["before"]
            assert sum(event["type"] == "TOOL_CALL_RESULT" for event in first) == 1
            call = next(
                event
                for event in first
                if event["type"] == "TOOL_CALL_START"
                and event["toolCallName"] == peer.tool_name
            )
            resumed = result_request(request, call)
            second = events(client.post("/", json=resumed))
            assert marker.read_text().splitlines() == ["before", "after"]
            assert sum(event["type"] == "TOOL_CALL_RESULT" for event in second) == 1
            error_events(client.post("/", json=resumed))
            assert marker.read_text().splitlines() == ["before", "after"]
    finally:
        registry.deregister("agui_probe_server_effect")
    assert len(peer.requests) == 2


@pytest.mark.parametrize("frontend_first", [True, False])
def test_frontend_and_approval_waits_share_one_live_execution(
    peer, tmp_path, frontend_first
):
    request = frontend_request(peer)
    save_config(
        {
            "approvals": {"mode": "manual", "timeout": 10},
            "auxiliary": {"title_generation": {"enabled": False}},
            "terminal": {"env": "local"},
        }
    )
    target = tmp_path / "disposable-removal"
    target.mkdir()
    calls = [
        model_call("browser", peer.tool_name, color="blue"),
        model_call(
            "approval", "terminal", command="rm -r -- " + shlex.quote(str(target))
        ),
    ]
    script_calls(peer, calls if frontend_first else list(reversed(calls)))
    original_script = peer.script
    peer.script = lambda incoming, index: (
        {"role": "assistant", "content": "WARM"}
        if index == 0
        else original_script(incoming, index - 1)
    )
    with TestClient(
        create_app(config_for(peer, ["terminal"])), base_url="http://127.0.0.1"
    ) as client:
        from tests.agui_adapter.test_conversations import next_user

        previous_id = request["runId"]
        events(client.post("/", json=request))
        request = next_user(request)
        first = events(client.post("/", json=request))
        assert target.exists()
        if frontend_first:
            call = next(event for event in first if event["type"] == "TOOL_CALL_START")
            request = result_request(request, call)
            reused = copy.deepcopy(request)
            reused["runId"] = previous_id
            error_events(client.post("/", json=reused))
            first = events(client.post("/", json=request))
        interrupt = first[-1]["outcome"]["interrupts"][0]
        wrong = copy.deepcopy(request)
        wrong["runId"] += "-invalid-approval"
        wrong["resume"] = [
            {
                "interruptId": "wrong-interrupt",
                "status": "resolved",
                "payload": {"approved": True},
            }
        ]
        error_events(client.post("/", json=wrong))
        assert target.exists()
        active_id = request["runId"]
        request["runId"] = wrong["runId"]
        request["resume"] = [
            {
                "interruptId": interrupt["id"],
                "status": "resolved",
                "payload": {"approved": True},
            }
        ]
        approval_id = request["runId"]
        for reused_id in (previous_id, active_id):
            request["runId"] = reused_id
            error_events(client.post("/", json=request))
            assert target.exists()
        request["runId"] = approval_id
        second = events(client.post("/", json=request))
        assert not target.exists()
        if not frontend_first:
            call = next(event for event in second if event["type"] == "TOOL_CALL_START")
            request.pop("resume")
            request = result_request(request, call)
            reused = copy.deepcopy(request)
            reused["runId"] = approval_id
            error_events(client.post("/", json=reused))
            events(client.post("/", json=request))
    assert len(peer.requests) == 3


def test_frontend_timeout_releases_worker_without_another_model_call(peer):
    request = frontend_request(peer)
    save_config(
        {
            "agui": {"frontend_tool_timeout": 2},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
    )
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = events(client.post("/", json=request))
        relay = app.state.executions.get(request["threadId"])
        assert relay.finished.wait(timeout=10)
        assert relay.cancelled == "Frontend result expired."
        call = next(event for event in first if event["type"] == "TOOL_CALL_START")
        error_events(client.post("/", json=result_request(request, call)))
    assert len(peer.requests) == 1


def test_shutdown_releases_pending_worker(peer):
    request = frontend_request(peer)
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        events(client.post("/", json=request))
        relay = app.state.executions.get(request["threadId"])
        assert not relay.finished.is_set()
    assert relay.finished.is_set()
    assert app.state.executions.get(request["threadId"]) is None
    assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_new_http_client_reconnects_but_separate_app_cannot_claim_wait(peer):
    request = frontend_request(peer)
    app = create_app(config_for(peer, []))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            first = events(await client.post("/", json=request))
        call = next(event for event in first if event["type"] == "TOOL_CALL_START")
        resumed = result_request(request, call)
        other = create_app(config_for(peer, []))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=other), base_url="http://127.0.0.1"
        ) as client:
            error_events(await client.post("/", json=resumed))
        assert len(peer.requests) == 1
        wrong_thread = copy.deepcopy(resumed)
        wrong_thread["threadId"] = "different-thread"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            error_events(await client.post("/", json=wrong_thread))
            events(await client.post("/", json=resumed))
    assert len(peer.requests) == 2


def test_multiple_model_rounds_keep_one_agent_and_prefix(peer, monkeypatch):
    from agui_adapter import server

    agents = []
    real_build = server.build_run_agent

    def observe(*args, **kwargs):
        agent = real_build(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(server, "build_run_agent", observe)
    request = frontend_request(peer)
    peer.script = lambda incoming, index: (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                model_call("round-" + str(index), peer.tool_name, color=str(index))
            ],
        }
        if index < 2
        else {"role": "assistant", "content": "DONE"}
    )
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        for index in range(2):
            leg = events(client.post("/", json=request))
            call = next(event for event in leg if event["type"] == "TOOL_CALL_START")
            assert call["toolCallId"] == "round-" + str(index)
            request = result_request(request, call, "RESULT_" + str(index))
        events(client.post("/", json=request))
    assert len(agents) == 1
    assert agents[0]._user_turn_count == 1
    assert len(peer.requests) == 3
    for before, after in zip(peer.requests, peer.requests[1:]):
        assert after["messages"][: len(before["messages"])] == before["messages"]
        assert after["tools"] == before["tools"]


@pytest.mark.asyncio
async def test_other_app_cannot_resolve_an_approval(peer, tmp_path):
    import asyncio

    request = frontend_request(peer)
    save_config(
        {
            "approvals": {"mode": "manual", "timeout": 10},
            "auxiliary": {"title_generation": {"enabled": False}},
            "terminal": {"env": "local"},
        }
    )
    target = tmp_path / "disposable-removal"
    target.mkdir()
    script_calls(
        peer,
        [
            model_call(
                "approval", "terminal", command="rm -r -- " + shlex.quote(str(target))
            )
        ],
    )
    app = create_app(config_for(peer, ["terminal"]))
    other = create_app(config_for(peer, ["terminal"]))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            first = events(await client.post("/", json=request))
            interrupt = first[-1]["outcome"]["interrupts"][0]
            request["runId"] = "approval-next"
            request["resume"] = [
                {
                    "interruptId": interrupt["id"],
                    "status": "resolved",
                    "payload": {"approved": True},
                }
            ]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=other), base_url="http://127.0.0.1"
            ) as foreign:
                async with asyncio.timeout(3):
                    error_events(await foreign.post("/", json=request))
            assert target.exists()
            events(await client.post("/", json=request))
    assert not target.exists()


@pytest.mark.parametrize("props", [None, "opaque-props"])
def test_opaque_forwarded_props_do_not_break_continuation(peer, props):
    request = frontend_request(peer)
    request["forwardedProps"] = props
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call = next(event for event in first if event["type"] == "TOOL_CALL_START")
        resumed = result_request(request, call)
        resumed["state"] = {"updated": True}
        events(client.post("/", json=resumed))
    assert len(peer.requests) == 2
