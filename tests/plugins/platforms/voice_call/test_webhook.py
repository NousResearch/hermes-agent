"""Webhook server security pipeline and inbound flow tests.

Runs the real aiohttp server on an ephemeral port and exercises it with an
aiohttp client — the same wire path a carrier would hit.
"""

import asyncio
import json

import pytest
import pytest_asyncio

aiohttp = pytest.importorskip("aiohttp")

from plugins.platforms.voice_call import runtime as runtime_mod
from plugins.platforms.voice_call.config import SecurityConfig
from plugins.platforms.voice_call.events import CallState
from plugins.platforms.voice_call.manager import CallManager
from plugins.platforms.voice_call.providers.base import (
    WebhookVerificationResult,
)
from plugins.platforms.voice_call.providers.mock import MockProvider
from plugins.platforms.voice_call.webhook import VoiceCallWebhookServer


class _ServerHarness:
    def __init__(self, server):
        self.server = server

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.bound_port}"

    async def post(self, path, *, data=None, json_body=None, headers=None):
        async with aiohttp.ClientSession() as session:
            kwargs = {"headers": headers or {}}
            if json_body is not None:
                kwargs["json"] = json_body
            else:
                kwargs["data"] = data
            async with session.post(self.base_url + path, **kwargs) as resp:
                return resp.status, await resp.text()

    async def get(self, path):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url + path) as resp:
                return resp.status, await resp.text()


@pytest_asyncio.fixture
async def harness(vc_config, provider, store):
    """Running webhook server wired to a real manager on the mock provider."""
    vc_config.serve.port = 0  # ephemeral
    vc_config.inbound_policy = "allowlist"
    vc_config.allow_from = ["+15555550009"]
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    server = VoiceCallWebhookServer(
        vc_config, provider, process_event=manager.process_event,
        admin_handler=None, admin_token="test-admin-token",
    )
    await server.start()
    h = _ServerHarness(server)
    h.manager = manager
    h.provider = provider
    h.config = vc_config
    yield h
    await server.stop()
    await manager.shutdown()


def _event_body(event_type="call.initiated", **fields):
    if event_type == "call.initiated":
        fields.setdefault("provider_call_id", "prov-in-1")
    return {"event": {"type": event_type, **fields}}


# -- pipeline gates -----------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_path_404(harness):
    status, _ = await harness.post("/somewhere/else", json_body={})
    assert status == 404


@pytest.mark.asyncio
async def test_wrong_method_405(harness):
    status, _ = await harness.get(harness.config.serve.path)
    assert status == 405


@pytest.mark.asyncio
async def test_oversized_body_rejected(harness):
    harness.config.security.max_body_bytes = 1024  # already-built app uses ctor value
    big = "x" * (2_000_000)
    status, _ = await harness.post(harness.config.serve.path, data=big)
    assert status in (408, 413)


@pytest.mark.asyncio
async def test_invalid_json_rejected_by_mock_parser(harness):
    status, _ = await harness.post(harness.config.serve.path, data=b"\xff\xfe not json")
    assert status == 400


@pytest.mark.asyncio
async def test_unknown_event_type_rejected(harness):
    status, _ = await harness.post(
        harness.config.serve.path, json_body=_event_body("call.made-up")
    )
    assert status == 400


@pytest.mark.asyncio
async def test_preauth_header_gate_for_signed_providers(harness):
    """Providers with signature schemes get a 401 before the body is read."""
    harness.provider.name = "telnyx"  # pretend: gate keys off provider name
    try:
        status, _ = await harness.post(
            harness.config.serve.path, json_body=_event_body()
        )
        assert status == 401
        status, _ = await harness.post(
            harness.config.serve.path,
            json_body=_event_body(
                event_type="call.speech", provider_call_id="x", text="hi"
            ),
            headers={"telnyx-signature-ed25519": "sig"},
        )
        assert status == 200
    finally:
        harness.provider.name = "mock"


@pytest.mark.asyncio
async def test_signature_verification_failure_403(harness):
    harness.provider.verify_webhook = lambda ctx: WebhookVerificationResult(
        ok=False, error="bad signature"
    )
    status, _ = await harness.post(
        harness.config.serve.path, json_body=_event_body()
    )
    assert status == 403


@pytest.mark.asyncio
async def test_skip_signature_verification_bypasses_verify(harness):
    harness.config.security.skip_signature_verification = True

    def explode(ctx):
        raise AssertionError("verify_webhook must not be called when skipped")

    harness.provider.verify_webhook = explode
    status, _ = await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            event_type="call.speech", provider_call_id="x", text="hi"
        ),
    )
    assert status == 200


@pytest.mark.asyncio
async def test_replay_returns_cached_response_without_reprocessing(harness):
    body = _event_body(
        event_type="call.initiated", direction="inbound",
        **{"from": "+15555550009", "to": "+15555550000"},
    )
    status1, text1 = await harness.post(harness.config.serve.path, json_body=body)
    assert status1 == 200
    assert len(harness.manager.get_active_calls()) == 1

    # Identical body → same dedupe key (body hash) → cached, not reprocessed.
    status2, text2 = await harness.post(harness.config.serve.path, json_body=body)
    assert status2 == 200 and text2 == text1
    assert len(harness.manager.get_active_calls()) == 1


# -- inbound policy ------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowlist_accepts_allowed_caller(harness):
    status, _ = await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            direction="inbound", **{"from": "+1 (555) 555-0009", "to": "+15555550000"}
        ),
    )
    assert status == 200
    record = harness.manager.call_for_chat("+15555550009")
    assert record is not None
    assert harness.provider.answered  # explicit answer issued


@pytest.mark.asyncio
async def test_allowlist_rejects_unknown_caller(harness):
    status, _ = await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            direction="inbound", **{"from": "+19998887777", "to": "+15555550000"}
        ),
    )
    assert status == 200  # carrier still gets its expected response
    assert harness.manager.get_active_calls() == []


@pytest.mark.asyncio
async def test_inbound_policy_disabled_rejects_everyone(harness):
    harness.config.inbound_policy = "disabled"
    status, _ = await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            direction="inbound", **{"from": "+15555550009", "to": "+15555550000"}
        ),
    )
    assert status == 200
    assert harness.manager.get_active_calls() == []


@pytest.mark.asyncio
async def test_inbound_policy_open_accepts_anyone(harness):
    harness.config.inbound_policy = "open"
    status, _ = await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            direction="inbound", **{"from": "+10000000001", "to": "+15555550000"}
        ),
    )
    assert status == 200
    assert len(harness.manager.get_active_calls()) == 1


@pytest.mark.asyncio
async def test_inbound_speech_flows_to_manager(harness):
    await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            direction="inbound", **{"from": "+15555550009", "to": "+15555550000"}
        ),
    )
    record = harness.manager.call_for_chat("+15555550009")
    await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            event_type="call.answered", provider_call_id=record.provider_call_id
        ),
    )
    assert record.state == CallState.LISTENING
    await harness.post(
        harness.config.serve.path,
        json_body=_event_body(
            event_type="call.speech",
            provider_call_id=record.provider_call_id,
            text="hello from the phone",
        ),
    )
    assert [t.text for t in record.transcript if t.speaker == "user"] == [
        "hello from the phone"
    ]


# -- per-IP limiter --------------------------------------------------------------


@pytest.mark.asyncio
async def test_inflight_limiter_429(vc_config, provider, store):
    vc_config.serve.port = 0
    vc_config.security = SecurityConfig(max_inflight_per_ip=1)
    manager = CallManager(vc_config, provider, store)

    gate = asyncio.Event()

    async def slow_process(event):
        await gate.wait()

    server = VoiceCallWebhookServer(vc_config, provider, process_event=slow_process)
    await server.start()
    h = _ServerHarness(server)
    body = _event_body(
        direction="inbound", **{"from": "+15555550009", "to": "+15555550000"}
    )
    vc_config.inbound_policy = "open"
    try:
        first = asyncio.create_task(
            h.post(vc_config.serve.path, json_body=body)
        )
        await asyncio.sleep(0.1)  # first request parked inside process_event
        status2, _ = await h.post(
            vc_config.serve.path, json_body={"event": {"type": "call.speech",
                                                       "provider_call_id": "x",
                                                       "text": "hi"}}
        )
        assert status2 == 429
        gate.set()
        status1, _ = await first
        assert status1 == 200
    finally:
        gate.set()
        await server.stop()
        await manager.shutdown()


# -- admin endpoint ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_requires_token(harness):
    status, _ = await harness.post("/voice/admin", json_body={"command": "status"})
    assert status == 401
    status, _ = await harness.post(
        "/voice/admin", json_body={"command": "status"},
        headers={"x-voice-call-admin-token": "wrong"},
    )
    assert status == 401


@pytest.mark.asyncio
async def test_stream_route_404_until_realtime_phase(harness):
    status, _ = await harness.get(harness.config.serve.stream_path + "/some-token")
    assert status == 404


# -- runtime integration ------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_starts_and_stops_server_releasing_port(tmp_path, make_config):
    cfg = make_config()
    cfg.serve.port = 0
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    port = runtime.webhook_server.bound_port
    assert port
    # Admin endpoint answers with the persisted token.
    token = (tmp_path / "admin.token").read_text(encoding="utf-8").strip()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/voice/admin",
            json={"command": "status"},
            headers={"x-voice-call-admin-token": token},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
    assert data["success"] is True and data["provider"] == "mock"

    await runtime_mod.stop_runtime()
    # Port is released: a fresh runtime can bind the same fixed port.
    cfg2 = make_config()
    cfg2.serve.port = port
    runtime2 = await runtime_mod.ensure_runtime(cfg2, store_dir=tmp_path)
    assert runtime2.webhook_server.bound_port == port
    await runtime_mod.stop_runtime()


@pytest.mark.asyncio
async def test_runtime_bind_failure_raises_oserror(tmp_path, make_config):
    cfg = make_config()
    cfg.serve.port = 0
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    taken = runtime.webhook_server.bound_port

    # Second runtime instance on the occupied port must raise OSError
    # (the adapter maps this to a retryable fatal error).
    cfg2 = make_config()
    cfg2.serve.port = taken
    other = runtime_mod.VoiceCallRuntime(cfg2, store_dir=tmp_path)
    with pytest.raises(OSError):
        await other.start()
    await other.stop()  # tolerates partial init
    await runtime_mod.stop_runtime()
