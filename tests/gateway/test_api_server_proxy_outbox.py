from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError, web
from aiohttp.test_utils import TestClient, TestServer

from gateway import proxy_outbox
from gateway.config import Platform, PlatformConfig
from gateway.delivery import resolve_delivery_transport
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult

API_KEY = "test-proxy-key-that-is-long-enough"


@pytest.fixture(autouse=True)
def _isolated_outbox(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"gateway": {"proxy_outbox_platforms": ["matrix"]}},
    )


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))


@pytest.mark.asyncio
async def test_api_adapter_queues_forwarded_text(monkeypatch):
    monkeypatch.setattr(proxy_outbox, "delivery_result", lambda _delivery_id: (True, None))
    adapter = _adapter()

    result = await adapter.send_for_platform(
        Platform.MATRIX,
        "!room:example.org",
        "scheduled result",
        metadata={"job_id": "job-1"},
    )

    assert result.success
    item = proxy_outbox.lease(platforms={"matrix"})[0]
    assert item["delivery_id"] == result.message_id
    assert item["content"] == "scheduled result"
    assert item["metadata"] == {"job_id": "job-1"}


@pytest.mark.asyncio
async def test_delivery_resolver_uses_api_forwarder_when_native_adapter_is_absent(
    monkeypatch,
):
    monkeypatch.setattr(proxy_outbox, "delivery_result", lambda _delivery_id: (True, None))
    adapter = _adapter()
    config = MagicMock()
    config.platforms = {}

    transport = resolve_delivery_transport(
        Platform.MATRIX, config, {Platform.API_SERVER: adapter}
    )

    assert transport is not None and transport.forwarded
    result = await transport.send(
        Platform.MATRIX,
        "!room:example.org",
        "forwarded",
        {"job_id": "job-2"},
    )
    assert result.success


class _NativeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="native-1")


@pytest.mark.asyncio
async def test_real_http_consumer_delivers_and_acknowledges_text():
    adapter = _adapter()
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/proxy/outbox"):
            app.router.add_route(method, path, handler)

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        delivery_id = proxy_outbox.enqueue(
            platform="matrix",
            chat_id="!room:example.org",
            content="scheduled result",
            metadata={"job_id": "job-3"},
        )
        native = _NativeAdapter()

        delivered = await proxy_outbox.deliver_once(
            str(client.make_url("/")).rstrip("/"),
            API_KEY,
            {Platform.MATRIX: native},
            session=client.session,
        )

        assert delivered == 1
        assert native.sent == [
            ("!room:example.org", "scheduled result", {"job_id": "job-3"})
        ]
        assert proxy_outbox.delivery_result(delivery_id) == (True, None)

        response = await client.get(
            "/v1/proxy/outbox?platforms=matrix",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ack_failure_does_not_strand_later_items_in_leased_batch():
    items = [
        {
            "delivery_id": "a" * 32,
            "platform": "matrix",
            "chat_id": "!one:example.org",
            "content": "one",
            "metadata": {},
            "attempt": 1,
        },
        {
            "delivery_id": "b" * 32,
            "platform": "matrix",
            "chat_id": "!two:example.org",
            "content": "two",
            "metadata": {},
            "attempt": 1,
        },
    ]
    ack_calls = 0

    async def lease_handler(_request):
        return web.json_response({"data": items})

    async def ack_handler(_request):
        nonlocal ack_calls
        ack_calls += 1
        return web.Response(status=503 if ack_calls == 1 else 200)

    app = web.Application()
    app.router.add_get("/v1/proxy/outbox", lease_handler)
    app.router.add_post("/v1/proxy/outbox/{delivery_id}/ack", ack_handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    native = _NativeAdapter()
    try:
        with pytest.raises(ClientResponseError):
            await proxy_outbox.deliver_once(
                str(client.make_url("/")).rstrip("/"),
                API_KEY,
                {Platform.MATRIX: native},
                session=client.session,
            )
        assert [sent[1] for sent in native.sent] == ["one", "two"]
        assert ack_calls == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_outbox_endpoint_requires_authentication():
    adapter = _adapter()
    request = MagicMock()
    request.headers = {}
    request.query = {"platforms": "matrix"}
    response = await adapter._handle_proxy_outbox(request)
    assert response.status == 401
