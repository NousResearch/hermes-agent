"""Per-route ``signature`` pin: a captured body-only MAC cannot be relabelled under another header family (#84265)."""
import base64
import hashlib
import hmac
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter

SECRET = "route-secret"
BODY = b'{"event_type":"push","n":1}'


def _adapter(route: dict) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "secret": SECRET, "routes": {"r": {"prompt": "x", **route}}}))


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _mac(data: bytes) -> str:
    return hmac.new(SECRET.encode(), data, hashlib.sha256).hexdigest()


def _v2_headers() -> dict:
    ts = str(int(time.time()))
    return {"X-Webhook-Signature-V2": _mac(ts.encode() + b"." + BODY), "X-Webhook-Timestamp": ts}


def _standard_headers() -> dict:
    msg_id, ts = "msg_1", str(int(time.time()))
    digest = hmac.new(SECRET.encode(), f"{msg_id}.{ts}.".encode() + BODY, hashlib.sha256).digest()
    return {"webhook-id": msg_id, "webhook-timestamp": ts, "webhook-signature": "v1," + base64.b64encode(digest).decode()}


# A captured V1 ``X-Webhook-Signature`` value re-presented under the other body-only header families.
_RELABELLED_V1 = {
    "linear": lambda: {"Linear-Signature": _mac(BODY)},
    "github": lambda: {"X-Hub-Signature-256": "sha256=" + _mac(BODY)},
}


async def _post(adapter: WebhookAdapter, headers: dict, request_id: str) -> int:
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/webhooks/r", data=BODY, headers={
            "Content-Type": "application/json", "X-Request-ID": request_id, **headers})
        return resp.status


@pytest.mark.parametrize("relabel", sorted(_RELABELLED_V1))
@pytest.mark.asyncio
async def test_relabelled_v1_mac_rejected_on_generic_route(relabel):
    adapter = _adapter({"signature": "generic"})
    assert await _post(adapter, {"X-Webhook-Signature": _mac(BODY)}, "v1") == 404
    assert await _post(adapter, _RELABELLED_V1[relabel](), "fresh-1") == 404
    assert await _post(adapter, _v2_headers(), "fresh-2") == 202


@pytest.mark.parametrize("relabel", sorted(_RELABELLED_V1))
@pytest.mark.asyncio
async def test_unpinned_route_still_accepts_relabelled_v1_mac(relabel):
    """Documented residual: unpinned routes keep accepting every provider family, so the relabel works
    until the route is pinned or its secret rotated."""
    adapter = _adapter({})
    assert await _post(adapter, {"X-Webhook-Signature": _mac(BODY)}, "v1") == 404
    assert await _post(adapter, _RELABELLED_V1[relabel](), "fresh-1") == 202


@pytest.mark.asyncio
async def test_github_pinned_route_accepts_only_github_signatures():
    adapter = _adapter({"signature": "github"})
    assert await _post(adapter, {"X-Hub-Signature-256": "sha256=" + _mac(BODY)}, "gh") == 202
    assert await _post(adapter, _v2_headers(), "v2") == 404
    assert await _post(adapter, {"Linear-Signature": _mac(BODY)}, "linear") == 404


@pytest.mark.asyncio
async def test_unknown_signature_pin_fails_closed():
    adapter = _adapter({"signature": "githubb"})
    with pytest.raises(ValueError, match="signature"):
        adapter._validate_route("r", adapter._routes["r"])
    assert await _post(adapter, {"X-Hub-Signature-256": "sha256=" + _mac(BODY)}, "gh") == 403


# Each supported header family, correctly signed with the route secret, and the families each pin admits.
_FAMILY_HEADERS = {
    "v2": _v2_headers, "standard": _standard_headers, "linear": lambda: {"Linear-Signature": _mac(BODY)},
    "github": lambda: {"X-Hub-Signature-256": "sha256=" + _mac(BODY)}, "gitlab": lambda: {"X-Gitlab-Token": SECRET}}
_PIN_ADMITS = {"generic": {"v2", "standard"}, "github": {"github"}, "gitlab": {"gitlab", "standard"},
               "linear": {"linear"}}


@pytest.mark.parametrize("family", sorted(_FAMILY_HEADERS))
@pytest.mark.parametrize("pin", sorted(_PIN_ADMITS))
@pytest.mark.asyncio
async def test_each_pin_admits_only_its_own_families(pin, family):
    expected = 202 if family in _PIN_ADMITS[pin] else 404
    assert await _post(_adapter({"signature": pin}), _FAMILY_HEADERS[family](), f"{pin}-{family}") == expected
