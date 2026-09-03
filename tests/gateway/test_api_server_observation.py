"""Security-property tests for the fixed Hermes observation boundary."""

import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


GENERIC_KEY = "generic-key-for-tests-123456"
OBSERVATION_KEY = "observation-key-for-tests-123456"


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware, security_headers_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/observation", adapter._handle_observation)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


async def _target_server(*, redirect: bool = False):
    calls = {"health": 0, "secret": 0}
    target = web.Application()

    async def health(request):
        calls["health"] += 1
        if redirect:
            raise web.HTTPFound("/secret")
        return web.json_response({"status": "ok", "revision": "test-revision"})

    async def secret(request):
        calls["secret"] += 1
        return web.json_response({"secret": "must-not-be-followed"})

    target.router.add_get("/api/health", health)
    target.router.add_get("/secret", secret)
    server = TestServer(target)
    await server.start_server()
    return server, calls


async def _client(monkeypatch, *, observation_key=OBSERVATION_KEY, generic_key=GENERIC_KEY, redirect=False):
    target_server, calls = await _target_server(redirect=redirect)
    monkeypatch.setenv("HERMES_OBSERVATION_KEY", observation_key or "")
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": generic_key,
                "observation_target_url": str(target_server.make_url("/api/health")),
                "cors_origins": ["https://client.example"],
            },
        )
    )
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    return client, target_server, calls, adapter


@pytest.mark.asyncio
async def test_observation_positive_is_fixed_and_audited(monkeypatch, caplog):
    client, target_server, calls, adapter = await _client(monkeypatch)
    try:
        async with client:
            with caplog.at_level("INFO"):
                response = await client.post(
                    "/v1/observation",
                    json={"action_id": "observe.ai_country.health"},
                    headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
                )
            assert response.status == 200
            body = await response.json()
            assert body["action_id"] == "observe.ai_country.health"
            assert body["observation_identity"] == "hermes-observer"
            assert body["mutation_capability"] == "none"
            assert body["provenance"] == "real_observation"
            assert body["observed"] == {"status": "ok", "revision": "test-revision"}
            assert calls == {"health": 1, "secret": 0}
            assert OBSERVATION_KEY not in caplog.text
            assert GENERIC_KEY not in caplog.text
    finally:
        await target_server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get", "put", "patch", "delete", "options"])
async def test_observation_rejects_unsupported_methods_before_target(monkeypatch, method):
    client, target_server, calls, _ = await _client(monkeypatch)
    try:
        async with client:
            response = await getattr(client, method)(
                "/v1/observation",
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert response.status in {404, 405}
            assert calls == {"health": 0, "secret": 0}
    finally:
        await target_server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action_id": "unknown"},
        {"action_id": "observe.ai_country.health", "extra": True},
        ["observe.ai_country.health"],
    ],
)
async def test_observation_rejects_noncanonical_requests_before_target(monkeypatch, payload):
    client, target_server, calls, _ = await _client(monkeypatch)
    try:
        async with client:
            response = await client.post(
                "/v1/observation",
                data=json.dumps(payload),
                headers={
                    "Authorization": f"Bearer {OBSERVATION_KEY}",
                    "Content-Type": "application/json",
                },
            )
            assert response.status == 400
            assert calls == {"health": 0, "secret": 0}
    finally:
        await target_server.close()


@pytest.mark.asyncio
async def test_observation_rejects_generic_key_and_observation_key_rejects_generic_route(monkeypatch):
    client, target_server, calls, adapter = await _client(monkeypatch)
    try:
        async with client:
            generic_on_observation = await client.post(
                "/v1/observation",
                json={"action_id": "observe.ai_country.health"},
                headers={"Authorization": f"Bearer {GENERIC_KEY}"},
            )
            observation_on_generic = await client.post(
                "/v1/runs",
                json={"input": "must not reach agent"},
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert generic_on_observation.status == 401
            assert observation_on_generic.status == 401
            assert calls == {"health": 0, "secret": 0}
            assert adapter._check_auth(MagicMock(headers={})) is not None
    finally:
        await target_server.close()


@pytest.mark.asyncio
async def test_observation_missing_or_invalid_key_fails_closed(monkeypatch):
    for key in (None, "wrong-observation-key"):
        client, target_server, calls, _ = await _client(monkeypatch, observation_key=key)
        try:
            async with client:
                response = await client.post(
                    "/v1/observation",
                    json={"action_id": "observe.ai_country.health"},
                    headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
                )
                assert response.status == 401
                assert calls == {"health": 0, "secret": 0}
        finally:
            await target_server.close()


@pytest.mark.asyncio
async def test_observation_does_not_follow_redirects(monkeypatch):
    client, target_server, calls, _ = await _client(monkeypatch, redirect=True)
    try:
        async with client:
            response = await client.post(
                "/v1/observation",
                json={"action_id": "observe.ai_country.health"},
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert response.status in {502, 503}
            assert calls == {"health": 1, "secret": 0}
    finally:
        await target_server.close()


@pytest.mark.asyncio
async def test_observation_equal_to_generic_key_fails_closed(monkeypatch):
    client, target_server, calls, _ = await _client(
        monkeypatch, observation_key=GENERIC_KEY, generic_key=GENERIC_KEY
    )
    try:
        async with client:
            response = await client.post(
                "/v1/observation",
                json={"action_id": "observe.ai_country.health"},
                headers={"Authorization": f"Bearer {GENERIC_KEY}"},
            )
            assert response.status == 401
            generic_response = await client.post(
                "/v1/runs",
                json={"input": "must not reach agent"},
                headers={"Authorization": f"Bearer {GENERIC_KEY}"},
            )
            assert generic_response.status == 401
            assert calls == {"health": 0, "secret": 0}
    finally:
        await target_server.close()


@pytest.mark.asyncio
async def test_observation_key_cannot_use_generic_route_without_generic_key(monkeypatch):
    client, target_server, calls, _ = await _client(monkeypatch, generic_key="")
    try:
        async with client:
            response = await client.post(
                "/v1/runs",
                json={"input": "must not reach agent"},
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert response.status == 401
            assert calls == {"health": 0, "secret": 0}
    finally:
        await target_server.close()


@pytest.mark.asyncio
async def test_invalid_target_fails_closed_before_invocation(monkeypatch):
    target_server, calls = await _target_server()
    monkeypatch.setenv("HERMES_OBSERVATION_KEY", OBSERVATION_KEY)
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": GENERIC_KEY, "observation_target_url": "https://example.com/not-health"},
        )
    )
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    try:
        async with client:
            response = await client.post(
                "/v1/observation",
                json={"action_id": "observe.ai_country.health"},
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert response.status == 503
            assert calls == {"health": 0, "secret": 0}
    finally:
        await client.close()
        await target_server.close()


def test_observation_is_not_in_profile_mirrored_route_table():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": GENERIC_KEY}))
    paths = {path for _, path, _ in adapter._http_route_table()}
    assert "/v1/observation" not in paths


@pytest.mark.asyncio
async def test_observation_has_no_profile_prefix_variant(monkeypatch):
    client, target_server, calls, _ = await _client(monkeypatch)
    try:
        async with client:
            response = await client.post(
                "/p/default/v1/observation",
                json={"action_id": "observe.ai_country.health"},
                headers={"Authorization": f"Bearer {OBSERVATION_KEY}"},
            )
            assert response.status == 404
            assert calls == {"health": 0, "secret": 0}
    finally:
        await target_server.close()
