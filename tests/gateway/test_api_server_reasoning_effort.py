"""Contract tests for request-scoped API server reasoning effort."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _parse_request_reasoning_config,
)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def _completed_agent() -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    return agent


async def _wait_for_calls(mock: MagicMock, expected: int) -> None:
    for _ in range(50):
        if mock.call_count >= expected:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"expected {expected} calls, observed {mock.call_count}")


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", {"enabled": False}),
        ("low", {"enabled": True, "effort": "low"}),
        (" HIGH ", {"enabled": True, "effort": "high"}),
        ("xhigh", {"enabled": True, "effort": "xhigh"}),
        ("max", {"enabled": True, "effort": "max"}),
        ("ultra", {"enabled": True, "effort": "ultra"}),
    ],
)
def test_parse_request_reasoning_config(effort, expected):
    assert _parse_request_reasoning_config({"reasoning_effort": effort}) == expected


def test_parse_request_reasoning_config_uses_default_when_omitted():
    assert _parse_request_reasoning_config({}) is None


@pytest.mark.parametrize(
    "effort",
    [None, False, 3, "", "extreme", "false", "disabled"],
)
def test_parse_request_reasoning_config_rejects_invalid_values(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        _parse_request_reasoning_config({"reasoning_effort": effort})


def test_create_agent_prefers_isolated_request_override(monkeypatch):
    captured = {}
    persisted = {"enabled": True, "effort": "medium"}
    request_override = {"enabled": True, "effort": "xhigh"}
    global_loader = MagicMock(return_value=persisted)

    class CapturingAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", CapturingAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "base_url": "https://example.test/v1",
            "api_mode": "responses",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-test")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 10)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(global_loader),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    adapter._create_agent(
        session_id="request-session",
        reasoning_config_override=request_override,
    )

    assert captured["reasoning_config"] == request_override
    assert captured["reasoning_config"] is not request_override
    assert request_override == {"enabled": True, "effort": "xhigh"}
    assert persisted == {"enabled": True, "effort": "medium"}
    global_loader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "low",
            },
        ),
        (
            "/v1/responses",
            {"input": "hello", "reasoning_effort": "low"},
        ),
    ],
)
async def test_synchronous_endpoints_forward_reasoning_override(endpoint, payload):
    adapter = _make_adapter()
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "done"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )

    with patch.object(adapter, "_run_agent", run_agent):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            response = await client.post(endpoint, json=payload)

    assert response.status == 200
    assert run_agent.await_args.kwargs["reasoning_config_override"] == {
        "enabled": True,
        "effort": "low",
    }


@pytest.mark.asyncio
async def test_streaming_chat_completion_forwards_reasoning_override():
    adapter = _make_adapter()
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "done"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )

    async def finish_stream(
        _request,
        _completion_id,
        _model,
        _created,
        _stream_q,
        agent_task,
        *_args,
        **_kwargs,
    ):
        await agent_task
        return web.json_response({"ok": True})

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "low",
        "stream": True,
    }
    with (
        patch.object(adapter, "_run_agent", run_agent),
        patch.object(
            adapter,
            "_write_sse_chat_completion",
            side_effect=finish_stream,
        ),
    ):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            response = await client.post("/v1/chat/completions", json=payload)

    assert response.status == 200
    assert run_agent.await_args.kwargs["reasoning_config_override"] == {
        "enabled": True,
        "effort": "low",
    }


@pytest.mark.asyncio
async def test_streaming_responses_forwards_reasoning_override():
    adapter = _make_adapter()
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "done"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )

    async def finish_stream(*_args, **kwargs):
        await kwargs["agent_task"]
        return web.json_response({"ok": True})

    payload = {
        "input": "hello",
        "reasoning_effort": "xhigh",
        "stream": True,
    }
    with (
        patch.object(adapter, "_run_agent", run_agent),
        patch.object(
            adapter,
            "_write_sse_responses",
            side_effect=finish_stream,
        ),
    ):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            response = await client.post("/v1/responses", json=payload)

    assert response.status == 200
    assert run_agent.await_args.kwargs["reasoning_config_override"] == {
        "enabled": True,
        "effort": "xhigh",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "base_payload"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        ),
        ("/v1/responses", {"input": "hello"}),
    ],
)
async def test_idempotency_distinguishes_reasoning_effort(endpoint, base_payload):
    adapter = _make_adapter()
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "done"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )
    headers = {"Idempotency-Key": f"reasoning-effort:{endpoint}"}

    with patch.object(adapter, "_run_agent", run_agent):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            for effort in ("low", "xhigh"):
                response = await client.post(
                    endpoint,
                    json={**base_payload, "reasoning_effort": effort},
                    headers=headers,
                )
                assert response.status == 200

    assert run_agent.await_count == 2
    assert [
        call.kwargs["reasoning_config_override"] for call in run_agent.await_args_list
    ] == [
        {"enabled": True, "effort": "low"},
        {"enabled": True, "effort": "xhigh"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": 3,
            },
        ),
        ("/v1/responses", {"input": "hello", "reasoning_effort": "extreme"}),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "false",
            },
        ),
        ("/v1/responses", {"input": "hello", "reasoning_effort": "disabled"}),
        ("/v1/runs", {"input": "hello", "reasoning_effort": None}),
    ],
)
async def test_endpoints_reject_invalid_reasoning_before_starting_work(
    endpoint, payload
):
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(endpoint, json=payload)
        data = await response.json()

    assert response.status == 400
    assert data["error"]["param"] == "reasoning_effort"
    assert adapter._run_streams == {}


@pytest.mark.asyncio
async def test_concurrent_runs_keep_reasoning_overrides_isolated():
    adapter = _make_adapter()
    create_agent = MagicMock(side_effect=lambda **_: _completed_agent())

    with patch.object(adapter, "_create_agent", create_agent):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            low_response, xhigh_response, ultra_response = await asyncio.gather(
                client.post(
                    "/v1/runs",
                    json={"input": "low task", "reasoning_effort": "low"},
                ),
                client.post(
                    "/v1/runs",
                    json={"input": "deep task", "reasoning_effort": "xhigh"},
                ),
                client.post(
                    "/v1/runs",
                    json={"input": "deepest task", "reasoning_effort": "ultra"},
                ),
            )
            assert low_response.status == 202
            assert xhigh_response.status == 202
            assert ultra_response.status == 202
            await _wait_for_calls(create_agent, 3)

    observed = {
        tuple(sorted(call.kwargs["reasoning_config_override"].items()))
        for call in create_agent.call_args_list
    }
    assert observed == {
        (("effort", "low"), ("enabled", True)),
        (("effort", "xhigh"), ("enabled", True)),
        (("effort", "ultra"), ("enabled", True)),
    }
