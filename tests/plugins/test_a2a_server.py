from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time

import pytest
from starlette.testclient import TestClient

from plugins.platforms.a2a import auth, server, setup, task_store


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    setup.ensure_a2a_platform_config(public_url="https://agent.example.test/a2a")
    return home


class GetTaskHandler:
    def __init__(self):
        self.context = None

    async def on_get_task(self, params, context):
        from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED, Task, TaskStatus

        self.context = context
        return Task(
            id=params.id,
            context_id="context",
            status=TaskStatus(state=TASK_STATE_COMPLETED),
        )


def _rpc(task_id="task-1"):
    return {"jsonrpc": "2.0", "id": "req-1", "method": "GetTask", "params": {"id": task_id}}


def _headers(token):
    return {"Authorization": f"Bearer {token}", "A2A-Version": "1.0"}


def test_public_card_uses_official_route_and_minimal_protocol_1_card(hermes_home):
    app = server.create_a2a_app(GetTaskHandler(), target_profile="default")
    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["supportedInterfaces"] == [
        {"url": "https://agent.example.test/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    assert card["capabilities"] == {"streaming": False, "pushNotifications": False, "extendedAgentCard": False}
    assert card["defaultInputModes"] == ["text/plain"]
    assert card["defaultOutputModes"] == ["text/plain"]
    assert card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_official_jsonrpc_route_authenticates_and_builds_owner_context(hermes_home):
    token = setup.add_principal("laptop", profile="default")
    handler = GetTaskHandler()
    app = server.create_a2a_app(handler, target_profile="default")

    response = TestClient(app).post("/a2a", json=_rpc(), headers=_headers(token))

    assert response.status_code == 200
    assert response.json()["result"]["id"] == "task-1"
    assert handler.context.user.is_authenticated
    assert handler.context.user.user_name == "laptop"
    assert "authorization" not in handler.context.state["headers"]


@pytest.mark.parametrize("authorization", [None, "Bearer invalid", "Bearer server-api-key"])
def test_missing_or_invalid_bearer_returns_401_challenge(hermes_home, authorization):
    headers = {"A2A-Version": "1.0"}
    if authorization:
        headers["Authorization"] = authorization
    response = TestClient(server.create_a2a_app(GetTaskHandler(), target_profile="default")).post(
        "/a2a", json=_rpc(), headers=headers
    )

    expected = 400 if authorization is None else 401
    assert response.status_code == expected
    if expected == 401:
        assert response.headers["www-authenticate"] == "Bearer"


def test_valid_bearer_for_another_profile_returns_403(hermes_home):
    token = setup.add_principal("laptop", profile="reviewer")
    response = TestClient(server.create_a2a_app(GetTaskHandler(), target_profile="default")).post(
        "/a2a", json=_rpc(), headers=_headers(token)
    )
    assert response.status_code == 403
    assert "www-authenticate" not in response.headers


def test_api_server_key_is_never_accepted_as_a2a_bearer(hermes_home, monkeypatch):
    api_key = "api-server-key-that-must-not-cross-auth-domains"
    monkeypatch.setenv("API_SERVER_KEY", api_key)
    response = TestClient(server.create_a2a_app(GetTaskHandler(), target_profile="default")).post(
        "/a2a", json=_rpc(), headers=_headers(api_key)
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "headers",
    [
        [("Authorization", "Bearer first"), ("Authorization", "Bearer second"), ("A2A-Version", "1.0")],
        [("Authorization", "Bearer first"), ("A2A-Version", "1.0"), ("A2A-Version", "1.0")],
        [("Authorization", "Bearer first")],
    ],
)
def test_rpc_rejects_duplicate_or_missing_singleton_headers(hermes_home, headers):
    response = TestClient(server.create_a2a_app(GetTaskHandler())).post("/a2a", json=_rpc(), headers=headers)
    assert response.status_code == 400


def test_target_profile_is_derived_and_mismatch_is_rejected(hermes_home):
    server.create_a2a_app(GetTaskHandler())
    with pytest.raises(ValueError, match="active profile"):
        server.create_a2a_app(GetTaskHandler(), target_profile="reviewer")


def test_production_rejects_loopback_http_advertised_url(hermes_home):
    setup.ensure_a2a_platform_config(public_url="http://127.0.0.1:8645/a2a")
    with pytest.raises(ValueError, match="HTTPS"):
        server.create_a2a_app(GetTaskHandler(), target_profile="default", production=True)


def test_body_header_and_pre_auth_rate_limits(hermes_home):
    body_client = TestClient(
        server.create_a2a_app(
            GetTaskHandler(),
            target_profile="default",
            limits=server.ServerLimits(max_body_bytes=64),
        )
    )
    assert body_client.post("/a2a", content=b"x" * 65, headers=_headers("invalid")).status_code == 413

    header_client = TestClient(
        server.create_a2a_app(
            GetTaskHandler(),
            target_profile="default",
            limits=server.ServerLimits(max_header_bytes=80),
        )
    )
    assert header_client.post("/a2a", content=b"{}", headers={"X-Large": "x" * 100}).status_code == 431

    rate_client = TestClient(
        server.create_a2a_app(
            GetTaskHandler(),
            target_profile="default",
            limits=server.ServerLimits(ip_requests_per_minute=1),
        )
    )
    assert rate_client.post("/a2a", json=_rpc()).status_code == 400
    assert rate_client.post("/a2a", json=_rpc()).status_code == 429


@pytest.mark.asyncio
async def test_ip_admission_happens_before_body_and_slow_body_times_out(hermes_home):
    app = server.create_a2a_app(
        GetTaskHandler(),
        limits=server.ServerLimits(ip_requests_per_minute=1, body_receive_timeout_seconds=0.02),
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/a2a",
        "raw_path": b"/a2a",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer invalid"), (b"a2a-version", b"1.0")],
        "client": ("192.0.2.10", 1234),
        "server": ("test", 80),
    }
    sent = []

    async def record(message):
        sent.append(message)

    async def immediate_body():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    await app(scope.copy(), immediate_body, record)
    body_called = False

    async def body_must_not_be_read():
        nonlocal body_called
        body_called = True
        return {"type": "http.request", "body": b"{}", "more_body": False}

    sent.clear()
    await app(scope.copy(), body_must_not_be_read, record)
    assert sent[0]["status"] == 429
    assert body_called is False

    slow_scope = scope.copy()
    slow_scope["client"] = ("192.0.2.11", 1234)

    async def slow_drip():
        await asyncio.sleep(0.1)
        return {"type": "http.request", "body": b"{", "more_body": True}

    sent.clear()
    await app(slow_scope, slow_drip, record)
    assert sent[0]["status"] == 408
    assert app.preauth_active == 0


@pytest.mark.asyncio
async def test_global_preauth_capacity_is_bounded_before_second_body_read(hermes_home):
    app = server.create_a2a_app(
        GetTaskHandler(),
        limits=server.ServerLimits(preauth_concurrency=1, body_receive_timeout_seconds=1),
    )
    base_scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/a2a",
        "raw_path": b"/a2a",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer invalid"), (b"a2a-version", b"1.0")],
        "server": ("test", 80),
    }
    entered = asyncio.Event()
    release = asyncio.Event()
    first_sent = []

    async def first_receive():
        entered.set()
        await release.wait()
        return {"type": "http.disconnect"}

    async def first_send(message):
        first_sent.append(message)

    first_scope = {**base_scope, "client": ("192.0.2.20", 1000)}
    first = asyncio.create_task(app(first_scope, first_receive, first_send))
    await entered.wait()
    second_body_read = False

    async def second_receive():
        nonlocal second_body_read
        second_body_read = True
        return {"type": "http.request", "body": b"{}", "more_body": False}

    second_sent = []

    async def second_send(message):
        second_sent.append(message)

    second_scope = {**base_scope, "client": ("192.0.2.21", 1001)}
    await app(second_scope, second_receive, second_send)
    assert second_sent[0]["status"] == 503
    assert second_body_read is False
    release.set()
    await first
    assert app.preauth_active == 0


def test_principal_rate_limit_and_streaming_are_disabled(hermes_home):
    token = setup.add_principal("laptop", profile="default")
    limits = server.ServerLimits(principal_requests_per_minute=1)
    client = TestClient(server.create_a2a_app(GetTaskHandler(), target_profile="default", limits=limits))

    assert client.post("/a2a", json=_rpc(), headers=_headers(token)).status_code == 200
    assert client.post("/a2a", json=_rpc(), headers=_headers(token)).status_code == 429

    fresh = TestClient(server.create_a2a_app(GetTaskHandler(), target_profile="default"))
    blocked = fresh.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req", "method": "SendStreamingMessage", "params": {}},
        headers=_headers(token),
    )
    assert blocked.status_code == 200
    assert blocked.json()["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_principal_concurrency_limit_and_request_timeout(hermes_home):
    import httpx

    token = setup.add_principal("laptop", profile="default")
    entered_handler = asyncio.Event()

    class SlowHandler(GetTaskHandler):
        async def on_get_task(self, params, context):
            entered_handler.set()
            await asyncio.sleep(0.1)
            return await super().on_get_task(params, context)

    limits = server.ServerLimits(principal_concurrency=1, request_timeout_seconds=0.03)
    app = server.create_a2a_app(SlowHandler(), target_profile="default", limits=limits)
    principal = server.ResolvedPrincipal(name="laptop", profile="default", credential_ref="test-ref")
    app.context_builder.authenticate_token = lambda _token: principal
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/a2a", json=_rpc("one"), headers=_headers(token)))
        await asyncio.wait_for(entered_handler.wait(), timeout=1)
        second = await client.post("/a2a", json=_rpc("two"), headers=_headers(token))
        timed_out = await first

    assert second.status_code == 429
    assert timed_out.status_code == 504


@pytest.mark.asyncio
async def test_auth_file_io_and_scrypt_do_not_block_event_loop(hermes_home, monkeypatch):
    import httpx

    token = setup.add_principal("laptop", profile="default")
    original = auth.resolve_inbound_token

    def slow_resolve(candidate):
        time.sleep(0.08)
        return original(candidate)

    monkeypatch.setattr(auth, "resolve_inbound_token", slow_resolve)
    app = server.create_a2a_app(GetTaskHandler())
    ticked = False

    async def ticker():
        nonlocal ticked
        await asyncio.sleep(0.01)
        ticked = True

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response, _ = await asyncio.gather(
            client.post("/a2a", json=_rpc(), headers=_headers(token)),
            ticker(),
        )
    assert ticked
    assert response.status_code == 200


def test_response_capture_is_bounded(hermes_home):
    token = setup.add_principal("laptop", profile="default")
    app = server.create_a2a_app(GetTaskHandler(), limits=server.ServerLimits(max_response_bytes=32))
    with TestClient(app) as client:
        response = client.post("/a2a", json=_rpc(), headers=_headers(token))
    assert response.status_code == 502
    assert response.json() == {"error": "Upstream response too large"}


@pytest.mark.asyncio
async def test_downstream_send_backpressure_is_outside_handler_timeout(hermes_home):
    token = setup.add_principal("laptop", profile="default")
    app = server.create_a2a_app(
        GetTaskHandler(), limits=server.ServerLimits(request_timeout_seconds=0.01)
    )
    body = json.dumps(_rpc()).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/a2a",
        "raw_path": b"/a2a",
        "query_string": b"",
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
            (b"a2a-version", b"1.0"),
        ],
        "client": ("192.0.2.30", 1000),
        "server": ("test", 80),
    }
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    sent = []

    async def slow_send(message):
        await asyncio.sleep(0.03)
        sent.append(message)

    started = time.monotonic()
    await app(scope, receive, slow_send)
    assert time.monotonic() - started >= 0.06
    assert sent[0]["status"] == 200


def test_limiter_is_bounded_and_transport_guidance_is_exposed():
    limiter = server._SlidingWindowLimiter(max_keys=3)
    for index in range(10):
        limiter.allow(f"192.0.2.{index}", 1)
    assert len(limiter) == 3
    assert "uvicorn" in server.UVICORN_TRANSPORT_GUIDANCE.lower()


@pytest.mark.asyncio
async def test_public_card_does_not_read_slow_body_and_releases_global_admission(hermes_home):
    app = server.create_a2a_app(
        GetTaskHandler(), limits=server.ServerLimits(preauth_concurrency=1)
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/.well-known/agent-card.json",
        "raw_path": b"/.well-known/agent-card.json",
        "query_string": b"",
        "headers": [],
        "client": ("192.0.2.40", 1000),
        "server": ("test", 80),
    }
    body_read = False

    async def slow_body():
        nonlocal body_read
        body_read = True
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"x", "more_body": True}

    sent = []

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope, slow_body, send), timeout=0.2)
    assert body_read is False
    assert sent[0]["status"] == 200
    assert app.preauth_active == 0


@pytest.mark.asyncio
async def test_unknown_paths_reject_without_body_hold_or_admission_leak(hermes_home):
    app = server.create_a2a_app(
        GetTaskHandler(),
        limits=server.ServerLimits(preauth_concurrency=1, ip_requests_per_minute=10),
    )
    body_reads = 0

    async def body_must_not_be_read():
        nonlocal body_reads
        body_reads += 1
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"x", "more_body": True}

    for index in range(3):
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/unknown-{index}",
            "raw_path": f"/unknown-{index}".encode(),
            "query_string": b"",
            "headers": [],
            "client": ("192.0.2.41", 1000),
            "server": ("test", 80),
        }
        sent = []

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, body_must_not_be_read, send), timeout=0.2)
        assert sent[0]["status"] == 404
        assert app.preauth_active == 0
    assert body_reads == 0


def test_server_lifespan_reconciles_same_wired_store_before_traffic(hermes_home):
    from a2a.types.a2a_pb2 import TASK_STATE_WORKING

    store = task_store.create_task_store()
    asyncio.run(store.save(
        __import__("a2a.types.a2a_pb2", fromlist=["Task"]).Task(
            id="orphan", context_id="context", status={"state": TASK_STATE_WORKING}
        ),
        __import__("a2a.server.context", fromlist=["ServerCallContext"]).ServerCallContext(
            user=server.AuthenticatedA2AUser("alice")
        ),
    ))
    asyncio.run(store.close())

    reopened = task_store.create_task_store()
    handler = GetTaskHandler()
    app = server.create_a2a_app(handler, task_store_instance=reopened)
    with TestClient(app):
        with sqlite3.connect(task_store.tasks_path()) as database:
            status = database.execute("SELECT status FROM tasks WHERE id = ?", ("orphan",)).fetchone()[0]
        assert "TASK_STATE_FAILED" in status
        assert handler.task_store is reopened


def test_sdk_errors_and_logs_do_not_echo_request_values(hermes_home, caplog):
    token = setup.add_principal("laptop", profile="default")
    secret = "request-secret-must-not-appear"
    app = server.create_a2a_app(GetTaskHandler(), target_profile="default")

    with caplog.at_level(logging.DEBUG):
        response = TestClient(app).post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "req", "method": "GetTask", "params": {"id": {"value": secret}}},
            headers=_headers(token),
        )

    assert response.status_code == 200
    assert secret not in response.text
    assert secret not in caplog.text


def test_factory_adopts_handler_instances_without_allocating(hermes_home, monkeypatch):
    store = task_store.create_task_store()
    card = server.build_agent_card("https://agent.example.test/a2a")
    handler = GetTaskHandler()
    handler.task_store = store
    handler._agent_card = card
    monkeypatch.setattr(
        task_store, "create_task_store", lambda: pytest.fail("unexpected allocation")
    )

    app = server.create_a2a_app(handler)

    assert app.task_store is store
    assert handler._agent_card is card
    asyncio.run(store.close())


def test_factory_rejects_identity_mismatch_before_allocation(hermes_home, monkeypatch):
    existing_store = task_store.create_task_store()
    supplied_store = task_store.create_task_store()
    existing_card = server.build_agent_card("https://agent.example.test/a2a")
    supplied_card = server.build_agent_card("https://agent.example.test/a2a")
    handler = GetTaskHandler()
    handler.task_store = existing_store
    handler._agent_card = existing_card
    allocations = 0

    def allocate():
        nonlocal allocations
        allocations += 1
        return task_store.create_task_store()

    monkeypatch.setattr(task_store, "create_task_store", allocate)
    with pytest.raises(ValueError, match="task store"):
        server.create_a2a_app(handler, task_store_instance=supplied_store)
    with pytest.raises(ValueError, match="agent card"):
        server.create_a2a_app(handler, agent_card=supplied_card)
    assert allocations == 0
    asyncio.run(existing_store.close())
    asyncio.run(supplied_store.close())


def test_stop_accepting_rejects_new_ingress(hermes_home):
    app = server.create_a2a_app(GetTaskHandler())
    app.stop_accepting()

    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 503
    assert response.json() == {"error": "Server is shutting down"}
