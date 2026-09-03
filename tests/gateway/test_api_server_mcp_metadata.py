import asyncio
import base64
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.mcp_run_context import (
    MAX_MCP_RUN_METADATA_BYTES,
    MAX_MCP_RUN_METADATA_DEPTH,
    decode_mcp_run_metadata_header,
    read_mcp_run_metadata,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _encode_metadata(value: dict) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _adapter(api_key: str = "sk-test-secret") -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    )


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def test_decoded_metadata_limit_is_enforced():
    json_overhead = len(b'{"value":""}')
    allowed = {"value": "a" * (MAX_MCP_RUN_METADATA_BYTES - json_overhead)}
    assert decode_mcp_run_metadata_header(_encode_metadata(allowed)) == allowed

    too_large = {"value": allowed["value"] + "a"}
    with pytest.raises(ValueError, match="too large"):
        decode_mcp_run_metadata_header(_encode_metadata(too_large))


def test_metadata_nesting_limit_is_explicit_and_stable():
    allowed = "leaf"
    for _ in range(MAX_MCP_RUN_METADATA_DEPTH):
        allowed = {"nested": allowed}
    assert decode_mcp_run_metadata_header(_encode_metadata(allowed)) == allowed

    too_deep = {"nested": allowed}
    with pytest.raises(ValueError, match="too deeply nested"):
        decode_mcp_run_metadata_header(_encode_metadata(too_deep))


async def _wait_for_run(adapter: APIServerAdapter, run_id: str) -> None:
    for _ in range(100):
        status = adapter._run_statuses.get(run_id, {})
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish")


@pytest.mark.asyncio
async def test_run_header_reaches_only_private_context_not_model_inputs():
    adapter = _adapter()
    metadata = {"magic.employee": {"binding": "signed-binding"}}
    observed = {}

    mock_agent = MagicMock()

    def _run_conversation(user_message, conversation_history, task_id):
        observed["metadata"] = read_mcp_run_metadata()
        observed["user_message"] = user_message
        observed["history"] = conversation_history
        return {"final_response": "done"}

    mock_agent.run_conversation.side_effect = _run_conversation
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    def _create_agent(**kwargs):
        observed["system_prompt"] = kwargs.get("ephemeral_system_prompt")
        return mock_agent

    with patch.object(adapter, "_create_agent", side_effect=_create_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "programul meu", "instructions": "Răspunde scurt"},
                headers={
                    "Authorization": "Bearer sk-test-secret",
                    "X-Hermes-MCP-Metadata": _encode_metadata(metadata),
                },
            )
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            await _wait_for_run(adapter, run_id)

    assert observed == {
        "metadata": metadata,
        "user_message": "programul meu",
        "history": [],
        "system_prompt": "Răspunde scurt",
    }
    assert "signed-binding" not in observed["user_message"]
    assert "signed-binding" not in json.dumps(observed["history"])
    assert "signed-binding" not in observed["system_prompt"]
    assert read_mcp_run_metadata() is None


@pytest.mark.asyncio
async def test_two_concurrent_api_runs_keep_private_metadata_isolated():
    adapter = _adapter()
    barrier = threading.Barrier(2)
    observed = {}

    class _Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, user_message, conversation_history, task_id):
            first = read_mcp_run_metadata()
            barrier.wait(timeout=5)
            second = read_mcp_run_metadata()
            observed[user_message] = (first, second)
            return {"final_response": "done"}

    with patch.object(adapter, "_create_agent", side_effect=lambda **_kwargs: _Agent()):
        async with TestClient(TestServer(_app(adapter))) as client:
            async def _start(key: str):
                response = await client.post(
                    "/v1/runs",
                    json={"input": key},
                    headers={
                        "Authorization": "Bearer sk-test-secret",
                        "X-Hermes-MCP-Metadata": _encode_metadata({"request": key}),
                    },
                )
                assert response.status == 202
                return (await response.json())["run_id"]

            run_a, run_b = await asyncio.gather(_start("a"), _start("b"))
            await asyncio.gather(
                _wait_for_run(adapter, run_a),
                _wait_for_run(adapter, run_b),
            )

    assert observed == {
        "a": ({"request": "a"}, {"request": "a"}),
        "b": ({"request": "b"}, {"request": "b"}),
    }
    assert read_mcp_run_metadata() is None


@pytest.mark.asyncio
async def test_run_idempotency_never_reuses_a_different_employee_binding():
    adapter = _adapter()
    observed = []

    class _Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, user_message, conversation_history, task_id):
            observed.append(read_mcp_run_metadata())
            return {"final_response": "done"}

    with patch.object(adapter, "_create_agent", side_effect=lambda **_kwargs: _Agent()):
        async with TestClient(TestServer(_app(adapter))) as client:
            run_ids = []
            for employee in ("a", "b"):
                response = await client.post(
                    "/v1/runs",
                    json={"input": "same message"},
                    headers={
                        "Authorization": "Bearer sk-test-secret",
                        "Idempotency-Key": "same-key",
                        "X-Hermes-MCP-Metadata": _encode_metadata(
                            {"employee": employee}
                        ),
                    },
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                run_ids.append(run_id)
                await _wait_for_run(adapter, run_id)

    assert run_ids[0] != run_ids[1]
    assert observed == [{"employee": "a"}, {"employee": "b"}]


@pytest.mark.asyncio
async def test_runs_preserve_two_employee_transcripts_without_cross_session_leak():
    adapter = _adapter()
    histories = {}
    observed = []

    class _SessionDB:
        def get_messages_as_conversation(self, session_id):
            return list(histories.get(session_id, []))

    class _Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            metadata = read_mcp_run_metadata()
            observed.append(
                {
                    "session_id": self.session_id,
                    "employee": metadata["employee"],
                    "history": list(conversation_history),
                    "message": user_message,
                }
            )
            response = f"reply-{metadata['employee']}-{user_message}"
            histories[self.session_id] = [
                *conversation_history,
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
            ]
            return {"final_response": response}

    def _create_agent(**kwargs):
        return _Agent(kwargs["session_id"])

    async def _start(client, employee, message, *, use_header):
        session_id = f"employee-session-{employee}"
        response = await client.post(
            "/v1/runs",
            json={
                "input": message,
                **({} if use_header else {"session_id": session_id}),
            },
            headers={
                "Authorization": "Bearer sk-test-secret",
                "X-Hermes-MCP-Metadata": _encode_metadata(
                    {"employee": employee}
                ),
                **(
                    {"X-Hermes-Session-Id": session_id}
                    if use_header
                    else {}
                ),
            },
        )
        assert response.status == 202
        run_id = (await response.json())["run_id"]
        await _wait_for_run(adapter, run_id)

    with patch.object(
        adapter, "_ensure_session_db_async", new_callable=AsyncMock
    ) as session_db, patch.object(
        adapter, "_create_agent", side_effect=_create_agent
    ):
        session_db.return_value = _SessionDB()
        async with TestClient(TestServer(_app(adapter))) as client:
            await _start(client, "a", "first-a", use_header=True)
            await _start(client, "b", "first-b", use_header=True)
            await _start(client, "a", "second-a", use_header=True)
            await _start(client, "b", "second-b", use_header=True)
            await _start(client, "bridge", "first-bridge", use_header=True)
            await _start(client, "bridge", "second-bridge", use_header=True)

    assert observed[0]["history"] == []
    assert observed[1]["history"] == []
    assert observed[2]["history"] == [
        {"role": "user", "content": "first-a"},
        {"role": "assistant", "content": "reply-a-first-a"},
    ]
    assert observed[3]["history"] == [
        {"role": "user", "content": "first-b"},
        {"role": "assistant", "content": "reply-b-first-b"},
    ]
    assert observed[4]["history"] == []
    assert observed[5]["history"] == [
        {"role": "user", "content": "first-bridge"},
        {"role": "assistant", "content": "reply-bridge-first-bridge"},
    ]
    assert read_mcp_run_metadata() is None


@pytest.mark.asyncio
async def test_runs_follow_compression_tip_and_publish_rotated_session_id():
    adapter = _adapter()
    observed = {}

    class _SessionDB:
        def resolve_resume_session_id(self, session_id):
            observed["resolved_from"] = session_id
            return "session-child"

        def get_messages_as_conversation(self, session_id):
            observed["loaded_from"] = session_id
            return [{"role": "user", "content": "compressed history"}]

    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "final_response": "done",
        "messages": [],
        "session_id": "session-grandchild",
    }
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    with patch.object(
        adapter, "_ensure_session_db_async", new_callable=AsyncMock
    ) as session_db, patch.object(
        adapter, "_create_agent", return_value=mock_agent
    ) as create_agent:
        session_db.return_value = _SessionDB()
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "next", "session_id": "session-old"},
                headers={
                    "Authorization": "Bearer sk-test-secret",
                    "X-Hermes-Session-Id": "session-old",
                },
            )
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            await _wait_for_run(adapter, run_id)

    assert observed == {
        "resolved_from": "session-old",
        "loaded_from": "session-child",
    }
    assert create_agent.call_args.kwargs["session_id"] == "session-old"
    assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == [
        {"role": "user", "content": "compressed history"}
    ]
    assert adapter._run_statuses[run_id]["session_id"] == "session-grandchild"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit_history",
    [
        [],
        [{"role": "user", "content": "explicit history"}],
    ],
)
async def test_explicit_run_history_is_authoritative_and_persisted(
    explicit_history,
):
    adapter = _adapter()
    session_db = MagicMock()
    returned_messages = [
        *explicit_history,
        {"role": "user", "content": "fresh turn"},
        {"role": "assistant", "content": "done"},
    ]
    mock_agent = MagicMock()
    mock_agent.session_id = "explicit-session"
    mock_agent._session_db = session_db
    mock_agent.run_conversation.return_value = {
        "final_response": "done",
        "messages": returned_messages,
        "session_id": "explicit-session",
    }
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    with patch.object(adapter, "_ensure_session_db_async") as load_db, patch.object(
        adapter, "_create_agent", return_value=mock_agent
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "fresh turn",
                    "session_id": "explicit-session",
                    "conversation_history": explicit_history,
                },
                headers={"Authorization": "Bearer sk-test-secret"},
            )
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            await _wait_for_run(adapter, run_id)

    load_db.assert_not_called()
    assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == (
        explicit_history
    )
    session_db.replace_messages.assert_called_once_with(
        "explicit-session",
        returned_messages,
        active_only=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_history", [None, {}, "", False, 0])
async def test_falsy_non_array_run_history_is_rejected(invalid_history):
    adapter = _adapter()
    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "hello",
                    "conversation_history": invalid_history,
                },
                headers={"Authorization": "Bearer sk-test-secret"},
            )

    assert response.status == 400
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_browser_preflight_allows_session_and_mcp_headers():
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": "sk-test-secret",
                "cors_origins": ["https://magic-team.example"],
            },
        )
    )
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)

    async with TestClient(TestServer(app)) as client:
        response = await client.options(
            "/v1/runs",
            headers={
                "Origin": "https://magic-team.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "X-Hermes-Session-Id, X-Hermes-Session-Key, "
                    "X-Hermes-MCP-Metadata"
                ),
            },
        )

    assert response.status == 200
    allowed = response.headers["Access-Control-Allow-Headers"]
    assert "X-Hermes-Session-Id" in allowed
    assert "X-Hermes-Session-Key" in allowed
    assert "X-Hermes-MCP-Metadata" in allowed


@pytest.mark.asyncio
async def test_invalid_metadata_header_is_rejected_before_agent_creation():
    adapter = _adapter()
    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={
                    "Authorization": "Bearer sk-test-secret",
                    "X-Hermes-MCP-Metadata": "not-base64!",
                },
            )

    assert response.status == 400
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_deeply_nested_metadata_is_rejected_before_agent_creation():
    adapter = _adapter()
    raw = (b'{"nested":' * 500) + b"null" + (b"}" * 500)
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={
                    "Authorization": "Bearer sk-test-secret",
                    "X-Hermes-MCP-Metadata": encoded,
                },
            )
            payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "invalid_mcp_metadata"
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_header_requires_api_key_authentication():
    adapter = _adapter(api_key="")
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello"},
            headers={"X-Hermes-MCP-Metadata": _encode_metadata({"private": "value"})},
        )

    assert response.status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body", "extract_text"),
    [
        (
            "/v1/chat/completions",
            {"model": "hermes-agent", "messages": [{"role": "user", "content": "hello"}]},
            lambda payload: payload["choices"][0]["message"]["content"],
        ),
        (
            "/v1/responses",
            {"model": "hermes-agent", "input": "hello"},
            lambda payload: payload["output"][0]["content"][0]["text"],
        ),
    ],
)
async def test_idempotency_cache_is_scoped_by_private_mcp_metadata(
    path, body, extract_text
):
    adapter = _adapter()

    async def _run_agent(**_kwargs):
        employee = read_mcp_run_metadata()["employee"]
        return (
            {
                "final_response": f"data-for-{employee}",
                "messages": [],
                "api_calls": 1,
            },
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    with patch.object(
        adapter, "_run_agent", new_callable=AsyncMock, side_effect=_run_agent
    ) as run_agent:
        async with TestClient(TestServer(_app(adapter))) as client:
            responses = []
            for employee in ("a", "b"):
                response = await client.post(
                    path,
                    json=body,
                    headers={
                        "Authorization": "Bearer sk-test-secret",
                        "Idempotency-Key": f"mcp-metadata-{path}",
                        "X-Hermes-MCP-Metadata": _encode_metadata(
                            {"employee": employee}
                        ),
                    },
                )
                assert response.status == 200
                responses.append(extract_text(await response.json()))

    assert responses == ["data-for-a", "data-for-b"]
    assert run_agent.await_count == 2
