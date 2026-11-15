"""Behavior-contract tests for `secret_file` externalized HMAC secrets.

Background (2026-09-26, RCA-secret-watchdog-locksteal): a raw HMAC secret sat INLINE in
config.yaml and leaked via a transcript dump. `secret_file:` lets the route/global config
point at a file instead; the adapter resolves it once at startup so the raw value never
lives in config (or in any config dump).

Contracts:
- single-line file -> str secret; multi-line -> list (rotation)
- unreadable/empty file -> ValueError at startup (fail closed, crash early)
- same-level `secret` + `secret_file` -> ValueError (no silent stale-inline-wins)
- the ORIGINAL config dict is never mutated (secret_file key stays in config.extra)
- routes without `secret_file` keep their original dict object (zero behaviour change)
- dynamic subscriptions support `secret_file` too; a bad block skips the route, no crash
- end-to-end: a request signed with the FILE's value is accepted; a wrong secret is 403
"""

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _config(routes=None, **extra_keys):
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes or {}}
    extra.update(extra_keys)
    return PlatformConfig(enabled=True, extra=extra)


def _write_secret_file(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _generic_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _generic_v2_signature(body: bytes, secret: str, timestamp: str) -> str:
    signed_content = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), signed_content, hashlib.sha256).hexdigest()


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    mount_authority(app, adapter)
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


SECRET_A = "k3y-file-resolved-alpha-0001"
SECRET_B = "k3y-file-resolved-beta-0002"


class TestStaticRouteSecretFile:

    def test_single_line_file_resolves_to_str(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", SECRET_A + "\n")
        cfg_extra = {"secret_file": sf, "prompt": "hi"}
        routes = {"events": dict(cfg_extra)}
        adapter = WebhookAdapter(_config(routes=routes))
        assert adapter._static_routes["events"]["secret"] == SECRET_A
        assert "secret_file" not in adapter._static_routes["events"]

    def test_multi_line_file_resolves_to_rotation_list(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", f"{SECRET_A}\n{SECRET_B}\n")
        routes = {"events": {"secret_file": sf, "prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes))
        assert adapter._static_routes["events"]["secret"] == [SECRET_A, SECRET_B]

    def test_original_config_dict_not_mutated(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", SECRET_A)
        route_cfg = {"secret_file": sf, "prompt": "hi"}
        routes = {"events": route_cfg}
        WebhookAdapter(_config(routes=routes))
        assert "secret" not in route_cfg, "original config dict must stay untouched"
        assert route_cfg["secret_file"] == sf

    def test_route_without_secret_file_keeps_identity(self, tmp_path):
        route_cfg = {"secret": SECRET_A, "prompt": "hi"}
        routes = {"plain": route_cfg}
        adapter = WebhookAdapter(_config(routes=routes))
        assert adapter._static_routes["plain"] is route_cfg, \
            "routes without secret_file must keep the original dict object"

    def test_unreadable_file_fails_closed_at_startup(self, tmp_path):
        routes = {"events": {"secret_file": str(tmp_path / "missing.secret"), "prompt": "hi"}}
        with pytest.raises(ValueError, match="unreadable"):
            WebhookAdapter(_config(routes=routes))

    def test_empty_file_fails_closed_at_startup(self, tmp_path):
        sf = _write_secret_file(tmp_path, "empty.secret", "\n \n")
        routes = {"events": {"secret_file": sf, "prompt": "hi"}}
        with pytest.raises(ValueError, match="empty"):
            WebhookAdapter(_config(routes=routes))

    def test_secret_and_secret_file_same_level_refused(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", SECRET_A)
        routes = {"events": {"secret": SECRET_B, "secret_file": sf, "prompt": "hi"}}
        with pytest.raises(ValueError, match="pick one"):
            WebhookAdapter(_config(routes=routes))

    def test_resolved_secret_passes_route_validation(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", SECRET_A)
        routes = {"events": {"secret_file": sf, "prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes))
        adapter._validate_route("events", adapter._static_routes["events"])  # must not raise


    def test_insecure_no_auth_in_rotation_list_refused_non_loopback(self, tmp_path):
        # Reviewer-fynd 26/9 04:51: request path skips HMAC when INSECURE_NO_AUTH is IN the
        # secret list — startup validation (connect()) must refuse such a list on a public host.
        sf = _write_secret_file(tmp_path, "rot.secret", f"{SECRET_A}\nINSECURE_NO_AUTH\n")
        routes = {"events": {"secret_file": sf, "prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes))  # default host 0.0.0.0 = non-loopback
        with pytest.raises(ValueError, match="INSECURE_NO_AUTH"):
            adapter._validate_route("events", adapter._static_routes["events"])

    def test_directory_as_secret_file_refused(self, tmp_path):
        # Reviewer-fynd 26/9 04:51: FIFOs/devices/dirs must fail closed, not block or crash odd.
        routes = {"events": {"secret_file": str(tmp_path), "prompt": "hi"}}
        with pytest.raises(ValueError, match="not a regular file"):
            WebhookAdapter(_config(routes=routes))


class TestGlobalSecretFile:

    def test_global_file_applies_to_routes_without_secret(self, tmp_path):
        sf = _write_secret_file(tmp_path, "global.secret", SECRET_A)
        routes = {"events": {"prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes, secret_file=sf))
        assert adapter._global_secret == SECRET_A
        adapter._validate_route("events", routes["events"])  # falls back to global — must pass

    def test_global_and_route_file_both_supported(self, tmp_path):
        gsf = _write_secret_file(tmp_path, "global.secret", SECRET_A)
        rsf = _write_secret_file(tmp_path, "route.secret", SECRET_B)
        routes = {"events": {"secret_file": rsf, "prompt": "hi"},
                  "other": {"prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes, secret_file=gsf))
        assert adapter._static_routes["events"]["secret"] == SECRET_B
        assert adapter._global_secret == SECRET_A

    def test_global_secret_and_secret_file_refused(self, tmp_path):
        sf = _write_secret_file(tmp_path, "global.secret", SECRET_A)
        with pytest.raises(ValueError, match="pick one"):
            WebhookAdapter(_config(routes={}, secret=SECRET_B, secret_file=sf))


class TestDynamicRouteSecretFile:

    def test_dynamic_route_secret_file_resolves_on_reload(self, tmp_path, monkeypatch):
        sf = _write_secret_file(tmp_path, "dyn.secret", SECRET_A)
        adapter = WebhookAdapter(_config(routes={}))
        subs = {"dyn": {"secret_file": sf, "prompt": "hi"}}
        subs_path = tmp_path / "subscriptions.json"
        subs_path.write_text(json.dumps(subs), encoding="utf-8")

        import gateway.platforms.webhook as wh
        monkeypatch.setattr(wh, "_DYNAMIC_ROUTES_FILENAME", subs_path.name)
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

        adapter._reload_dynamic_routes()
        assert "dyn" in adapter._dynamic_routes
        assert adapter._dynamic_routes["dyn"]["secret"] == SECRET_A
        assert "secret_file" not in adapter._dynamic_routes["dyn"]

    def test_dynamic_route_bad_file_skips_route(self, tmp_path, monkeypatch):
        adapter = WebhookAdapter(_config(routes={}))
        subs = {"dyn": {"secret_file": str(tmp_path / "nope.secret"), "prompt": "hi"}}
        subs_path = tmp_path / "subscriptions.json"
        subs_path.write_text(json.dumps(subs), encoding="utf-8")

        import gateway.platforms.webhook as wh
        monkeypatch.setattr(wh, "_DYNAMIC_ROUTES_FILENAME", subs_path.name)
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

        adapter._reload_dynamic_routes()
        assert "dyn" not in adapter._dynamic_routes, "bad block must skip, not crash"


class TestSecretFileHttpRoundTrip:

    @pytest.mark.asyncio
    async def test_signed_with_file_value_accepted_wrong_secret_403(self, tmp_path):
        sf = _write_secret_file(tmp_path, "hmac.secret", SECRET_A)
        routes = {"events": {"secret_file": sf, "prompt": "hi"}}
        adapter = WebhookAdapter(_config(routes=routes))
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        body = json.dumps({"data": "value"}).encode()
        ts = str(int(time.time()))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/events", data=body,
                headers={"X-Webhook-Signature-V2": _generic_v2_signature(body, SECRET_A, ts),
                         "X-Webhook-Timestamp": ts,
                         "Content-Type": "application/json"})
            assert resp.status in (200, 202)
            adapter.handle_message.assert_called_once()

            adapter.handle_message.reset_mock()
            resp = await cli.post(
                "/webhooks/events", data=body,
                headers={"X-Webhook-Signature-V2": _generic_v2_signature(body, "wrong-secret", ts),
                         "X-Webhook-Timestamp": ts,
                         "Content-Type": "application/json"})
            assert resp.status == 404, "invalid signature must fail closed (404 anti-enumeration)"
            adapter.handle_message.assert_not_called()
