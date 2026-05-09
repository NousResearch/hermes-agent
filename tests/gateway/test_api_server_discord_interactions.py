"""Tests for the Discord interaction route mounted on the API server.

These tests intentionally keep the route local/test-only. They prove the
safety boundary before any live Discord endpoint registration:
raw request bytes and signature headers are captured first, invalid signatures
fail closed, and component callbacks only produce an ACK preview.
"""

from __future__ import annotations

import importlib
import json
import time

pytest = importlib.import_module("pytest")
web = importlib.import_module("aiohttp.web")
aiohttp_test_utils = importlib.import_module("aiohttp.test_utils")
TestClient = aiohttp_test_utils.TestClient
TestServer = aiohttp_test_utils.TestServer

from gateway.config import PlatformConfig
from gateway import discord_interactions as discord_interactions_module
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter
from gateway.discord_interactions import (
    DISCORD_INTERACTION_ROUTE,
    DiscordInteractionConfig,
    DiscordInteractionReplayCache,
    default_discord_signature_verifier,
    resolve_discord_interaction_config,
    validate_discord_dry_run_result,
    validate_discord_interaction_timestamp,
)


def _adapter(extra: dict | None = None) -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def _fresh_timestamp() -> str:
    return str(int(time.time()))


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"discord_interactions": {"enabled": True}},
        {"discord_interactions": {"enabled": True, "public_key_env": "DISCORD_BOT_TOKEN"}},
        {"discord_interactions": {"enabled": True, "public_key_env": "DISCORD_CLIENT_SECRET"}},
        {"discord_interactions": {"enabled": True, "public_key_env": "DISCORD_WEBHOOK_URL"}},
        {"discord_interactions": {"enabled": True, "public_key_env": "HERMES_WEBHOOK_SECRET"}},
        {"discord_interactions": {"enabled": True, "public_key_env": "TELEGRAM_TOKEN"}},
    ],
)
def test_discord_interaction_config_fails_closed_without_explicit_public_key(extra, monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "not-a-public-key")
    monkeypatch.setenv("DISCORD_CLIENT_SECRET", "not-a-public-key")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "not-a-public-key")
    monkeypatch.setenv("HERMES_WEBHOOK_SECRET", "not-a-public-key")
    monkeypatch.setenv("TELEGRAM_TOKEN", "not-a-public-key")

    config = resolve_discord_interaction_config(extra)

    assert config.enabled is False
    assert config.public_key == ""
    assert config.route_path == DISCORD_INTERACTION_ROUTE


def test_discord_interaction_config_accepts_only_public_key_source(monkeypatch):
    monkeypatch.setenv("DISCORD_APPLICATION_PUBLIC_KEY", "abc123")

    config = resolve_discord_interaction_config(
        {"discord_interactions": {"enabled": True, "public_key_env": "DISCORD_APPLICATION_PUBLIC_KEY"}}
    )

    assert config == DiscordInteractionConfig(
        enabled=True,
        public_key="abc123",
        route_path=DISCORD_INTERACTION_ROUTE,
    )


@pytest.mark.asyncio
async def test_route_is_not_mounted_when_config_is_disabled():
    adapter = _adapter()
    app = web.Application()

    mounted = adapter._register_discord_interaction_route(app)

    assert mounted is False
    assert all(getattr(route.resource, "canonical", "") != DISCORD_INTERACTION_ROUTE for route in app.router.routes())


@pytest.mark.asyncio
async def test_ping_reads_raw_body_and_signature_headers_before_ack_preview():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    seen = {}

    def verifier(*, public_key: str, timestamp: str, signature: str, body: bytes) -> bool:
        seen.update(
            {
                "public_key": public_key,
                "timestamp": timestamp,
                "signature": signature,
                "body": body,
            }
        )
        return True

    adapter._discord_interaction_verifier = verifier
    app = web.Application()
    assert adapter._register_discord_interaction_route(app) is True

    body = b'{"type":1,"id":"interaction-1"}'
    timestamp = _fresh_timestamp()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-1",
                "X-Signature-Timestamp": timestamp,
            },
        )
        assert resp.status == 200
        assert await resp.json() == {"type": 1}

    assert seen == {
        "public_key": "public-key",
        "timestamp": timestamp,
        "signature": "sig-1",
        "body": body,
    }


@pytest.mark.asyncio
async def test_invalid_signature_fails_before_json_parse_or_dry_run():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = []

    def verifier(**_kwargs) -> bool:
        return False

    adapter._discord_interaction_verifier = verifier
    adapter._discord_interaction_dry_run_handler = lambda _payload: calls.append(_payload)
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=b"not-json-but-signature-fails-first",
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "bad-sig",
                "X-Signature-Timestamp": _fresh_timestamp(),
            },
        )
        assert resp.status == 401
        data = await resp.json()
        assert data["error"] == "invalid_signature"

    assert calls == []


@pytest.mark.asyncio
async def test_component_interaction_returns_ephemeral_dry_run_ack_without_apply():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = []

    adapter._discord_interaction_verifier = lambda **_kwargs: True
    adapter._discord_interaction_dry_run_handler = lambda payload: calls.append(payload) or {
        "status": "dry_run",
        "action": "approve",
        "review_id": "review-123",
    }
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    payload = {
        "type": 3,
        "id": "interaction-2",
        "data": {"custom_id": "mim:soma-review:v1:approve:review-123"},
    }
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-2",
                "X-Signature-Timestamp": _fresh_timestamp(),
            },
        )
        assert resp.status == 200
        data = await resp.json()

    assert calls == []
    assert data["type"] == 4
    assert data["data"]["flags"] == 64
    assert "dry-run" in data["data"]["content"]
    assert "approve" in data["data"]["content"]
    assert "review-123" in data["data"]["content"]


@pytest.mark.asyncio
async def test_route_rejects_unknown_component_payload_without_dry_run_apply():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = []

    adapter._discord_interaction_verifier = lambda **_kwargs: True
    adapter._discord_interaction_dry_run_handler = lambda payload: calls.append(payload)
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    payload = {"type": 3, "id": "interaction-3", "data": {"custom_id": "bad:payload"}}
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-3",
                "X-Signature-Timestamp": _fresh_timestamp(),
            },
        )
        assert resp.status == 400
        data = await resp.json()

    assert data["error"] == "invalid_component"
    assert calls == []


def test_timestamp_freshness_rejects_stale_future_and_non_decimal_values():
    assert validate_discord_interaction_timestamp("1000", now=1200) == (True, "ok")
    assert validate_discord_interaction_timestamp("1000", now=1301) == (False, "stale_timestamp")
    assert validate_discord_interaction_timestamp("1006", now=1000) == (False, "future_timestamp")
    assert validate_discord_interaction_timestamp("١٢٣", now=123) == (False, "invalid_timestamp")
    assert validate_discord_interaction_timestamp("+123", now=123) == (False, "invalid_timestamp")


def test_default_ed25519_verifier_checks_timestamp_plus_raw_body():
    signing = pytest.importorskip("nacl.signing")
    key = signing.SigningKey.generate()
    public_key = key.verify_key.encode().hex()
    timestamp = "1700000000"
    body = b'{"type":1,"id":"abc"}'
    signature = key.sign(timestamp.encode("ascii") + body).signature.hex()

    assert default_discord_signature_verifier(
        public_key=public_key,
        timestamp=timestamp,
        signature=signature,
        body=body,
    ) is True
    assert default_discord_signature_verifier(
        public_key=public_key,
        timestamp=timestamp,
        signature=signature,
        body=b'{"id":"abc","type":1}',
    ) is False
    assert default_discord_signature_verifier(
        public_key="not-hex",
        timestamp=timestamp,
        signature=signature,
        body=body,
    ) is False


def test_replay_cache_rejects_duplicate_signature_body_without_db_writes():
    cache = DiscordInteractionReplayCache(max_entries=2, ttl_seconds=10)
    body = b'{"type":1}'

    assert cache.check_and_remember(timestamp="1000", signature="sig", body=body, now=1000) == (True, "ok")
    assert cache.check_and_remember(timestamp="1000", signature="sig", body=body, now=1001) == (
        False,
        "replay_detected",
    )
    assert cache.check_and_remember(timestamp="1001", signature="sig2", body=body, now=1001) == (True, "ok")
    assert cache.check_and_remember(timestamp="1012", signature="sig", body=body, now=1012) == (True, "ok")


def test_replay_cache_fails_closed_when_full_after_prune():
    cache = DiscordInteractionReplayCache(max_entries=1, ttl_seconds=10)

    assert cache.check_and_remember(timestamp="1000", signature="sig-1", body=b"one", now=1000) == (True, "ok")
    assert cache.check_and_remember(timestamp="1001", signature="sig-2", body=b"two", now=1001) == (
        False,
        "replay_cache_full",
    )


def test_dry_run_result_validator_blocks_live_or_runtime_apply_claims():
    assert validate_discord_dry_run_result({"status": "dry_run", "dry_run": True}) == (True, "ok")
    assert validate_discord_dry_run_result({"live_send": True}) == (False, "unsafe_dry_run_result")
    assert validate_discord_dry_run_result({"runtime_write": True}) == (False, "unsafe_dry_run_result")
    assert validate_discord_dry_run_result({"applied": True}) == (False, "unsafe_dry_run_result")
    assert validate_discord_dry_run_result({"apply": True}) == (False, "unsafe_dry_run_result")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timestamp", "expected_error"),
    [
        ("9999999999", "future_timestamp"),
        ("+123", "invalid_timestamp"),
        ("١٢٣", "invalid_timestamp"),
    ],
)
async def test_bad_timestamp_fails_before_signature_or_dry_run_handler(timestamp, expected_error):
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = {"verifier": 0, "dry_run": 0}

    def verifier(**_kwargs) -> bool:
        calls["verifier"] += 1
        return True

    adapter._discord_interaction_verifier = verifier
    adapter._discord_interaction_dry_run_handler = lambda _payload: calls.__setitem__("dry_run", calls["dry_run"] + 1)
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=b'{"type":1}',
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-bad-time",
                "X-Signature-Timestamp": timestamp,
            },
        )
        assert resp.status == 401
        data = await resp.json()

    assert data["error"] == expected_error
    assert calls == {"verifier": 0, "dry_run": 0}


@pytest.mark.asyncio
async def test_stale_timestamp_fails_before_signature_or_dry_run_handler():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = {"verifier": 0, "dry_run": 0}

    def verifier(**_kwargs) -> bool:
        calls["verifier"] += 1
        return True

    adapter._discord_interaction_verifier = verifier
    adapter._discord_interaction_dry_run_handler = lambda _payload: calls.__setitem__("dry_run", calls["dry_run"] + 1)
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=b'{"type":1}',
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-stale",
                "X-Signature-Timestamp": "1000",
            },
        )
        assert resp.status == 401
        data = await resp.json()

    assert data["error"] == "stale_timestamp"
    assert calls == {"verifier": 0, "dry_run": 0}


@pytest.mark.asyncio
async def test_replayed_interaction_fails_before_json_or_dry_run_handler():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = []
    timestamp = _fresh_timestamp()
    body = json.dumps(
        {"type": 3, "id": "interaction-replay", "data": {"custom_id": "mim:soma-review:v1:defer:review-456"}}
    ).encode("utf-8")

    adapter._discord_interaction_verifier = lambda **_kwargs: True
    adapter._discord_interaction_dry_run_handler = lambda payload: calls.append(payload) or {"status": "dry_run"}
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    headers = {
        "Content-Type": "application/json",
        "X-Signature-Ed25519": "sig-replay",
        "X-Signature-Timestamp": timestamp,
    }
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(DISCORD_INTERACTION_ROUTE, data=body, headers=headers)
        second = await cli.post(DISCORD_INTERACTION_ROUTE, data=body, headers=headers)
        assert first.status == 200
        assert second.status == 409
        second_data = await second.json()

    assert second_data["error"] == "replay_detected"
    assert calls == []


@pytest.mark.asyncio
async def test_dry_run_handler_is_not_invoked_until_live_apply_gate_exists():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = []
    adapter._discord_interaction_verifier = lambda **_kwargs: True
    adapter._discord_interaction_dry_run_handler = lambda payload: calls.append(payload) or {"status": "secret-token", "runtime_write": True}
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    payload = {"type": 3, "id": "interaction-unsafe", "data": {"custom_id": "mim:soma-review:v1:approve:review-789"}}
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            DISCORD_INTERACTION_ROUTE,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": "sig-unsafe",
                "X-Signature-Timestamp": _fresh_timestamp(),
            },
        )
        assert resp.status == 200
        data = await resp.json()

    assert calls == []
    assert data["type"] == 4
    assert "secret-token" not in data["data"]["content"]
    assert "runtime_write" not in data["data"]["content"]


def test_default_verifier_fails_closed_when_pynacl_is_unavailable(monkeypatch):
    def fake_import(name: str):
        if name in {"nacl.signing", "nacl.exceptions"}:
            raise ImportError("missing nacl")
        return importlib.import_module(name)

    monkeypatch.setattr(discord_interactions_module.importlib, "import_module", fake_import)

    assert default_discord_signature_verifier(
        public_key="00" * 32,
        timestamp="1700000000",
        signature="00" * 64,
        body=b"{}",
    ) is False


@pytest.mark.asyncio
async def test_missing_signature_headers_fail_before_verifier_or_dry_run_handler():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    calls = {"verifier": 0, "dry_run": 0}
    adapter._discord_interaction_verifier = lambda **_kwargs: calls.__setitem__("verifier", calls["verifier"] + 1) or True
    adapter._discord_interaction_dry_run_handler = lambda _payload: calls.__setitem__("dry_run", calls["dry_run"] + 1)
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(DISCORD_INTERACTION_ROUTE, data=b'{"type":1}', headers={"Content-Type": "application/json"})
        assert resp.status == 401
        data = await resp.json()

    assert data["error"] == "missing_signature"
    assert calls == {"verifier": 0, "dry_run": 0}


@pytest.mark.asyncio
async def test_connect_mounts_discord_route_between_responses_post_and_response_lookup(monkeypatch):
    class FakeRunner:
        def __init__(self, app):
            self.app = app

        async def setup(self):
            return None

        async def cleanup(self):
            return None

    class FakeSite:
        def __init__(self, runner, host, port):
            self.runner = runner
            self.host = host
            self.port = port

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(api_server_module.web, "AppRunner", FakeRunner)
    monkeypatch.setattr(api_server_module.web, "TCPSite", FakeSite)

    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "key": "sk-test-key",
                "discord_interactions": {"enabled": True, "public_key": "public-key"},
            },
        )
    )

    assert await adapter.connect() is True
    assert adapter._app is not None
    routes = []
    for route in adapter._app.router.routes():
        resource = route.resource
        assert resource is not None
        routes.append((route.method, resource.canonical))

    assert ("POST", DISCORD_INTERACTION_ROUTE) in routes
    assert routes.index(("POST", "/v1/responses")) < routes.index(("POST", DISCORD_INTERACTION_ROUTE))
    assert routes.index(("POST", DISCORD_INTERACTION_ROUTE)) < routes.index(("GET", "/v1/responses/{response_id}"))
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_replay_detection_happens_before_json_parsing_on_second_request():
    adapter = _adapter({"discord_interactions": {"enabled": True, "public_key": "public-key"}})
    adapter._discord_interaction_verifier = lambda **_kwargs: True
    app = web.Application()
    adapter._register_discord_interaction_route(app)

    headers = {
        "Content-Type": "application/json",
        "X-Signature-Ed25519": "sig-invalid-json-replay",
        "X-Signature-Timestamp": _fresh_timestamp(),
    }
    body = b"not-json-but-signature-is-valid-in-test"

    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(DISCORD_INTERACTION_ROUTE, data=body, headers=headers)
        second = await cli.post(DISCORD_INTERACTION_ROUTE, data=body, headers=headers)
        first_data = await first.json()
        second_data = await second.json()

    assert first.status == 400
    assert first_data["error"] == "invalid_json"
    assert second.status == 409
    assert second_data["error"] == "replay_detected"
