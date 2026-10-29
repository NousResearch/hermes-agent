"""Behavior contracts for OpenAI API retry and session identity."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.platforms.api_server_idempotency import IdempotencyCache, scope


def _adapter(api_key=""):
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.mark.asyncio
async def test_cache_deduplicates_only_within_authenticated_request_scope(monkeypatch):
    adapter = _adapter("sk-cache-test")
    monkeypatch.setattr(
        adapter,
        "_expected_api_key",
        lambda: "sk-worker-test" if _api_request_profile.get() == "worker" else adapter._api_key,
    )
    base_headers = {"Idempotency-Key": "retry-1"}
    body = {"messages": [{"role": "user", "content": "hello"}]}
    calls = []
    outcomes = []

    async def invoke(headers, *, endpoint="chat_completions", profile="default", target=adapter):
        request = SimpleNamespace(headers=headers)

        async def compute():
            calls.append((endpoint, profile, headers.get("X-Hermes-Session-Id")))
            return len(calls), {"total_tokens": len(calls)}

        token = _api_request_profile.set(profile)
        try:
            return await target._run_idempotent(
                request, body, compute, log_label="test", fingerprint_keys=["messages"],
                route=endpoint)
        finally:
            _api_request_profile.reset(token)

    outcomes.append(await invoke(base_headers))
    outcomes.append(await invoke(base_headers, profile=None))
    outcomes.append(await invoke(base_headers, profile="worker"))
    outcomes.append(await invoke(base_headers, endpoint="responses"))
    outcomes.append(await invoke({**base_headers, "X-Hermes-Session-Id": "session-a"}))
    outcomes.append(await invoke({**base_headers, "X-Hermes-Session-Id": "session-b"}))
    outcomes.append(await invoke({**base_headers, "X-Hermes-Session-Key": "memory-a"}))
    outcomes.append(await invoke({**base_headers, "X-Hermes-Session-Key": "memory-b"}))
    outcomes.append(await invoke(base_headers, target=_adapter("sk-cache-test")))
    adapter._api_key = "sk-rotated-test"
    outcomes.append(await invoke(base_headers))

    assert len(calls) == 9
    assert outcomes[0] == outcomes[1]
    assert all(outcome[1] is None for outcome in outcomes)
    assert all(outcome != outcomes[0] for outcome in outcomes[2:])
    stored_scope = scope(adapter, SimpleNamespace(headers=base_headers), "responses", "default")
    assert "sk-rotated-test" not in repr(stored_scope)


@pytest.mark.asyncio
async def test_unconfigured_api_key_bypasses_retry_cache():
    adapter = _adapter()
    request = SimpleNamespace(headers={"Idempotency-Key": "retry-1"})
    calls = []

    async def compute():
        calls.append(True)
        return len(calls), {}

    for _ in range(2):
        outcome, error = await adapter._run_idempotent(
            request, {}, compute, log_label="test", fingerprint_keys=[], route="responses")
        assert error is None
        assert outcome[0] == len(calls)
    assert calls == [True, True]


@pytest.mark.asyncio
async def test_headerless_chat_requests_start_distinct_sessions():
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
            run_agent.side_effect = lambda **kwargs: (
                {"final_response": "ok", "session_id": kwargs["session_id"], "messages": []},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            responses = [
                await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "same opening"}]}),
                await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "same opening"}]}),
            ]

    session_ids = [response.headers["X-Hermes-Session-Id"] for response in responses]
    assert all(response.status == 200 for response in responses)
    assert session_ids[0] != session_ids[1]
    assert all(session_id.startswith("api-") for session_id in session_ids)
    assert all(UUID(session_id[4:]).version == 4 for session_id in session_ids)
    assert run_agent.await_count == 2


@pytest.mark.asyncio
async def test_responses_retry_fingerprint_tracks_normalized_history_and_truncation():
    adapter = _adapter("sk-responses-test")
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
            run_agent.return_value = (
                {"final_response": "ok", "messages": []},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            headers = {"Authorization": "Bearer sk-responses-test", "Idempotency-Key": "same"}
            common = {"model": "hermes-agent", "conversation_history": [
                {"role": "user", "content": "prior A"}]}
            responses = [
                await client.post("/v1/responses", headers=headers,
                                  json={**common, "input": "hello"}),
                await client.post("/v1/responses", headers=headers,
                                  json={**common, "input": [{"role": "user", "content": "hello"}]}),
                await client.post("/v1/responses", headers=headers,
                                  json={**common, "conversation_history": [
                                      {"role": "user", "content": "prior B"}], "input": "hello"}),
                await client.post("/v1/responses", headers=headers,
                                  json={**common, "truncation": "auto", "input": "hello"}),
            ]

    assert [response.status for response in responses] == [200, 200, 200, 200]
    assert run_agent.await_count == 3


@pytest.mark.asyncio
async def test_routed_response_retry_fingerprint_tracks_provider_credentials():
    from gateway.platforms.api_server_openai_routes import _responses_fingerprint_payload

    route = {"model": "provider-model", "provider": "openai", "api_key": "route-secret-a",
             "base_url": "https://a.example/v1"}
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True, extra={"key": "sk-route-test", "model_routes": {"alias": route}}))
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
            run_agent.return_value = (
                {"final_response": "ok", "messages": []},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            headers = {"Authorization": "Bearer sk-route-test", "Idempotency-Key": "same"}
            body = {"model": "alias", "input": "hello"}
            assert "route-secret-a" not in repr(_responses_fingerprint_payload(adapter, body))
            first = await client.post("/v1/responses", headers=headers, json=body)
            adapter._model_routes["alias"] = {
                **route, "api_key": "route-secret-b", "base_url": "https://b.example/v1"}
            second = await client.post("/v1/responses", headers=headers, json=body)

    assert first.status == second.status == 200
    assert run_agent.await_count == 2
    assert "route-secret-a" not in repr(adapter._model_routes)


@pytest.mark.asyncio
async def test_provider_auth_retry_keeps_the_session_id():
    from gateway.platforms.api_server import _ProviderAuthResolutionError

    adapter = _adapter("sk-auth-test")
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=_ProviderAuthResolutionError("fixture")) as create:
            headers = {"Authorization": "Bearer sk-auth-test", "Idempotency-Key": "same"}
            body = {"messages": [{"role": "user", "content": "hello"}]}
            first = await client.post("/v1/chat/completions", headers=headers, json=body)
            second = await client.post("/v1/chat/completions", headers=headers, json=body)

    assert first.status == second.status == 200
    assert first.headers["X-Hermes-Session-Id"] == second.headers["X-Hermes-Session-Id"]
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_responses_provider_auth_retry_keeps_the_session_id():
    from gateway.platforms.api_server import _ProviderAuthResolutionError

    adapter = _adapter("sk-responses-auth")
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=_ProviderAuthResolutionError("fixture")) as create:
            headers = {"Authorization": "Bearer sk-responses-auth", "Idempotency-Key": "same"}
            body = {"model": "hermes-agent", "input": "hello"}
            first = await client.post("/v1/responses", headers=headers, json=body)
            second = await client.post("/v1/responses", headers=headers, json=body)

    assert first.status == second.status == 200
    assert first.headers["X-Hermes-Session-Id"] == second.headers["X-Hermes-Session-Id"]
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_streaming_headerless_chat_keeps_fresh_ids_and_skips_cache():
    adapter = _adapter("sk-stream-test")

    async def run_agent(**kwargs):
        return (
            {"final_response": "ok", "session_id": kwargs["session_id"], "messages": []},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock, side_effect=run_agent) as mock_run:
            headers = {"Authorization": "Bearer sk-stream-test", "Idempotency-Key": "same"}
            body = {"model": "hermes-agent", "messages": [
                {"role": "user", "content": "hello"}], "stream": True}
            first = await client.post("/v1/chat/completions", headers=headers, json=body)
            second = await client.post("/v1/chat/completions", headers=headers, json=body)
            await first.text()
            await second.text()

    first_id = first.headers["X-Hermes-Session-Id"]
    second_id = second.headers["X-Hermes-Session-Id"]
    assert first.status == second.status == 200
    assert first_id != second_id
    assert UUID(first_id[4:]).version == UUID(second_id[4:]).version == 4
    assert mock_run.await_count == 2


@pytest.mark.asyncio
async def test_idempotency_cache_runs_concurrent_retries_once():
    cache = IdempotencyCache()
    gate = asyncio.Event()
    started = asyncio.Event()
    calls = []

    async def compute():
        calls.append(True)
        started.set()
        await gate.wait()
        return "response"

    first = asyncio.create_task(cache.get_or_set(("scope",), "key", "fp", compute))
    second = asyncio.create_task(cache.get_or_set(("scope",), "key", "fp", compute))
    await started.wait()
    gate.set()
    assert await first == await second == "response"
    assert calls == [True]
