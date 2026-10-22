"""Whole-conversation cache and ownership contracts with real AIAgent calls."""

import copy
import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agui_adapter import server
from agui_adapter.server import create_app
from hermes_cli.config import save_config
from tests.agui_adapter.test_modernization_probe import config_for, events
from tests.agui_adapter.test_parked_frontend import (
    frontend_request,
    result_request,
    error_events,
)


@pytest.fixture
def agents(monkeypatch):
    result = []
    build = server.build_run_agent

    def observe(*args, **kwargs):
        agent = build(*args, **kwargs)
        result.append(agent)
        return agent

    monkeypatch.setattr(server, "build_run_agent", observe)
    return result


def next_user(request, text="Second real user turn", user_id="user-2"):
    request = copy.deepcopy(request)
    request["runId"] += "-user"
    request["messages"] += [
        {"id": "assistant-final", "role": "assistant", "content": "Browser projection"},
        {"id": user_id, "role": "user", "content": text},
    ]
    return request


def plain_model(peer):
    peer.script = lambda request, index: {
        "role": "assistant",
        "content": f"MODEL_REPLY_{index}",
    }


@pytest.mark.parametrize("boolean_change", [False, True])
def test_two_user_turns_keep_agent_system_prompt_history_and_tools(
    peer, agents, boolean_change
):
    plain_model(peer)
    request = frontend_request(peer)
    request["state"] = {"color": "red", "nested": [1]}
    request["context"] = [{"description": "current item", "value": "first-item"}]
    request["forwardedProps"] = {"tone": "brief"}
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        events(client.post("/", json=request))
        followup = next_user(request)
        followup["state"] = (
            {"nested": [True], "color": "red"}
            if boolean_change
            else {"color": "blue", "nested": [1]}
        )
        if not boolean_change:
            followup["context"] = [
                {"description": "current item", "value": "second-item"}
            ]
            followup["forwardedProps"] = {"tone": "detailed"}
        events(client.post("/", json=followup))
        assert len(agents) == 1, "a new user turn rebuilt the conversation agent"
        assert agents[0]._user_turn_count == 2
    before, after = peer.requests
    assert after["messages"][: len(before["messages"])] == before["messages"]
    assert after["tools"] == before["tools"]
    assert after["messages"][-1]["role"] == "user"
    assert ('"nested": [true]' if boolean_change else "blue") in after["messages"][-1][
        "content"
    ]
    if not boolean_change:
        assert "second-item" in after["messages"][-1]["content"]
        assert "detailed" in after["messages"][-1]["content"]
    assert any(message["content"] == "MODEL_REPLY_0" for message in after["messages"])
    assert not any(
        message["content"] == "Browser projection" for message in after["messages"]
    )


def test_frontend_result_context_survives_a_later_user_turn(peer, agents):
    request = frontend_request(peer)
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call = next(event for event in first if event["type"] == "TOOL_CALL_START")
        resumed = result_request(request, call, "BROWSER_RESULT")
        resumed["state"] = {"new_browser_state": "retained"}
        events(client.post("/", json=resumed))
        events(client.post("/", json=next_user(resumed)))
        assert len(agents) == 1
    previous, following = peer.requests[1:]
    assert following["messages"][: len(previous["messages"])] == previous["messages"]
    assert following["messages"][-1]["content"] == "Second real user turn"
    tool = next(
        message for message in following["messages"] if message["role"] == "tool"
    )
    assert "adapter_context" in tool["content"] and "retained" in tool["content"]


@pytest.mark.parametrize("invalid", ["tools", "user", "history", "run"])
def test_invalid_new_turn_does_not_mutate_or_replace_conversation(
    peer, agents, invalid
):
    plain_model(peer)
    request = frontend_request(peer)
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        events(client.post("/", json=request))
        good = next_user(request)
        bad = copy.deepcopy(good)
        if invalid == "tools":
            bad["tools"][0]["description"] = "changed tool definition"
        elif invalid == "user":
            bad["messages"][-1]["id"] = request["messages"][-1]["id"]
        elif invalid == "history":
            bad["messages"][0]["content"] = "rewrite accepted user"
        else:
            bad["runId"] = request["runId"]
        error_events(client.post("/", json=bad))
        assert len(peer.requests) == 1
        events(client.post("/", json=good))
        assert len(agents) == 1
    assert len(peer.requests) == 2


def test_capacity_does_not_evict_a_warm_conversation(peer, agents):
    plain_model(peer)
    request = frontend_request(peer)
    save_config(
        {
            "agui": {"max_conversations": 1},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
    )
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        events(client.post("/", json=request))
        other = copy.deepcopy(request)
        other["threadId"] = "other-conversation"
        error_events(client.post("/", json=other))
        assert len(peer.requests) == 1
        events(client.post("/", json=next_user(request)))
        assert len(agents) == 1


def test_idle_expiry_closes_once_and_refuses_reconstruction(peer, agents, monkeypatch):
    plain_model(peer)
    request = frontend_request(peer)
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        events(client.post("/", json=request))
        store = app.state.executions.conversations
        record = store.get(request["threadId"])
        close = Mock(wraps=agents[0].close)
        monkeypatch.setattr(agents[0], "close", close)
        now = store.clock()
        store.clock = lambda: now + store.idle_timeout + 1
        assert store.sweep() == 1
        assert record.closed
        assert close.call_count == 1
        assert store.sweep() == 0
        error_events(client.post("/", json=next_user(request)))
        assert len(peer.requests) == 1
    assert close.call_count == 1


def test_pending_frontend_is_not_idle_evicted(peer, agents):
    request = frontend_request(peer)
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = events(client.post("/", json=request))
        store = app.state.executions.conversations
        now = store.clock()
        store.clock = lambda: now + store.idle_timeout + 1
        assert store.sweep() == 0
        assert store.get(request["threadId"]).busy
        call = next(event for event in first if event["type"] == "TOOL_CALL_START")
        events(client.post("/", json=result_request(request, call)))
        assert len(agents) == 1


def test_failure_after_execution_retires_agent_without_replaying(
    peer, agents, monkeypatch
):
    plain_model(peer)
    request = frontend_request(peer)
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        events(client.post("/", json=request))
        run = agents[0].run_conversation
        close = Mock(wraps=agents[0].close)
        monkeypatch.setattr(agents[0], "close", close)
        failed_relays = []

        def fail_after_execution(*args, **kwargs):
            run(*args, **kwargs)
            failed_relays.append(app.state.executions.get(request["threadId"]))
            raise RuntimeError("injected post-execution failure")

        monkeypatch.setattr(agents[0], "run_conversation", fail_after_execution)
        followup = next_user(request)
        error_events(client.post("/", json=followup))
        assert failed_relays[0].finished.wait(timeout=10)
        assert app.state.executions.conversations.get(request["threadId"]) is None
        assert close.call_count == 1
        error_events(client.post("/", json=followup))
        assert len(peer.requests) == 2


@pytest.mark.asyncio
async def test_periodic_idle_cleanup_keeps_profile_scope(
    peer, agents, monkeypatch, tmp_path
):
    import asyncio
    import httpx
    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
        get_hermes_home,
    )

    plain_model(peer)
    owner, foreign = tmp_path / "owner", tmp_path / "foreign"
    owner.mkdir()
    foreign.mkdir()
    token = set_hermes_home_override(owner)
    closed = threading.Event()
    seen = []
    try:
        request = frontend_request(peer)
        save_config(
            {
                "agui": {"conversation_idle_timeout": 2},
                "auxiliary": {"title_generation": {"enabled": False}},
            }
        )
        app = create_app(config_for(peer, []))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                events(await client.post("/", json=request))
            real_close = agents[0].close

            def close():
                seen.append(get_hermes_home())
                real_close()
                closed.set()

            monkeypatch.setattr(agents[0], "close", close)
            other_token = set_hermes_home_override(foreign)
            try:
                assert await asyncio.to_thread(closed.wait, 10), (
                    "idle reaper did not close the agent"
                )
            finally:
                reset_hermes_home_override(other_token)
        assert seen == [owner]
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_disconnect_after_run_started_retires_live_agent(peer, agents):
    import asyncio
    from ag_ui.core import RunAgentInput
    from ag_ui.encoder import EventEncoder

    entered, release = threading.Event(), threading.Event()

    def blocked_model(request, index):
        entered.set()
        assert release.wait(timeout=10)
        return {"role": "assistant", "content": "finished after disconnect"}

    peer.script = blocked_model
    request = frontend_request(peer)
    registry = server.ExecutionRegistry()
    stream = server._event_stream(
        RunAgentInput.model_validate(request),
        EventEncoder(),
        config_for(peer, []),
        {},
        registry,
    )
    try:
        assert "RUN_STARTED" in await stream.__anext__()
        assert await asyncio.to_thread(entered.wait, 10)
        relay = registry.get(request["threadId"])
        await stream.aclose()
        release.set()
        assert await asyncio.to_thread(relay.finished.wait, 10)
        assert registry.conversations.get(request["threadId"]) is None
        assert len(peer.requests) == 1
    finally:
        release.set()
        await stream.aclose()
        await asyncio.to_thread(registry.close)


@pytest.mark.asyncio
async def test_interleaved_conversations_keep_history_and_tool_schemas_isolated(
    peer, agents
):
    import asyncio
    import httpx

    entered, release = threading.Event(), threading.Event()
    request = frontend_request(peer)

    def model(incoming, index):
        if index == 0:
            entered.set()
            assert release.wait(timeout=10)
        return {"role": "assistant", "content": "reply-" + str(index)}

    peer.script = model
    other = copy.deepcopy(request)
    other["threadId"] = "separate-conversation"
    other["messages"][0]["content"] = "conversation-B-private"
    other["tools"][0]["name"] = "agui_other_frontend_probe"
    app = create_app(config_for(peer, []))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            first = asyncio.create_task(client.post("/", json=request))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                events(await client.post("/", json=other))
                error_events(await client.post("/", json=next_user(request)))
                release.set()
                events(await first)
                events(await client.post("/", json=next_user(request)))
            finally:
                release.set()
                await first
    assert len(agents) == 2
    before, foreign, after = peer.requests
    assert after["messages"][: len(before["messages"])] == before["messages"]
    assert after["tools"] == before["tools"]
    assert not any(
        "conversation-B-private" in str(message) for message in after["messages"]
    )
    assert foreign["tools"] != after["tools"]


def test_changed_context_preserves_multimodal_new_user_content(peer, agents):
    plain_model(peer)
    request = frontend_request(peer)
    # A custom endpoint is not in the model catalog. Declare its native image
    # capability through real config so Hermes does not invoke its text fallback.
    save_config(
        {
            "model": {"supports_vision": True},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
    )
    image = {
        "type": "image",
        "source": {
            "type": "data",
            "mimeType": "image/png",
            "value": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2sAAAAASUVORK5CYII=",
        },
    }
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        events(client.post("/", json=request))
        followup = next_user(request)
        followup["messages"][-1]["content"] = [
            {"type": "text", "text": "Look at this image"},
            image,
        ]
        followup["state"] = {"image_context": "updated"}
        events(client.post("/", json=followup))
        assert len(agents) == 1
    content = peer.requests[-1]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert any(part["type"] == "image_url" for part in content)
    assert any("updated" in part.get("text", "") for part in content)
