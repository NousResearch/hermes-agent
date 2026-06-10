"""Telnyx provider: credentials, Ed25519 verification, event parsing,
request shapes, host guard, and ngrok log parsing."""

import base64
import json
import time

import pytest

cryptography = pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from plugins.platforms.voice_call.config import VoiceCallConfig
from plugins.platforms.voice_call.events import CallRecord, EventType
from plugins.platforms.voice_call.providers import _http
from plugins.platforms.voice_call.providers.base import WebhookContext
from plugins.platforms.voice_call.providers.telnyx import (
    TelnyxProvider,
    build_streaming_fields,
)
from plugins.platforms.voice_call.tunnel import parse_ngrok_log_line, start_tunnel


@pytest.fixture
def keypair():
    private = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    public_b64 = base64.b64encode(
        private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    return private, public_b64


@pytest.fixture
def provider(keypair, monkeypatch):
    _, public_b64 = keypair
    monkeypatch.setenv("TELNYX_API_KEY", "test-key")
    monkeypatch.setenv("TELNYX_CONNECTION_ID", "conn-1")
    monkeypatch.setenv("TELNYX_PUBLIC_KEY", public_b64)
    cfg = VoiceCallConfig.from_extra(
        {"provider": "telnyx", "public_url": "https://hooks.example"}
    )
    p = TelnyxProvider(cfg)
    p.set_public_url("https://hooks.example")
    return p


def _signed_ctx(private, body: dict, timestamp: int | None = None) -> WebhookContext:
    raw = json.dumps(body).encode()
    ts = str(timestamp if timestamp is not None else int(time.time()))
    signature = private.sign(ts.encode() + b"|" + raw)
    return WebhookContext(
        method="POST",
        path="/voice/webhook",
        body=raw,
        headers={
            "telnyx-signature-ed25519": base64.b64encode(signature).decode(),
            "telnyx-timestamp": ts,
        },
    )


def _telnyx_event(event_type, **payload):
    return {"data": {"id": "evt-1", "event_type": event_type, "payload": payload}}


# -- credentials ---------------------------------------------------------------


def test_requires_api_key_and_connection_id(monkeypatch):
    for var in ("TELNYX_API_KEY", "TELNYX_CONNECTION_ID", "TELNYX_PUBLIC_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="API key"):
        TelnyxProvider(VoiceCallConfig.from_extra({"provider": "telnyx"}))
    monkeypatch.setenv("TELNYX_API_KEY", "k")
    with pytest.raises(ValueError, match="connection ID"):
        TelnyxProvider(VoiceCallConfig.from_extra({"provider": "telnyx"}))


# -- Ed25519 verification ---------------------------------------------------------


def test_verify_valid_signature(provider, keypair):
    private, _ = keypair
    ctx = _signed_ctx(private, _telnyx_event("call.answered"))
    result = provider.verify_webhook(ctx)
    assert result.ok is True
    assert result.dedupe_key.startswith("telnyx:")


def test_verify_invalid_signature(provider, keypair):
    private, _ = keypair
    ctx = _signed_ctx(private, _telnyx_event("call.answered"))
    ctx.body = b'{"tampered": true}'
    result = provider.verify_webhook(ctx)
    assert result.ok is False and "invalid signature" in result.error


def test_verify_wrong_key(provider):
    other = Ed25519PrivateKey.generate()
    ctx = _signed_ctx(other, _telnyx_event("call.answered"))
    assert provider.verify_webhook(ctx).ok is False


def test_verify_missing_headers(provider):
    ctx = WebhookContext(method="POST", path="/voice/webhook", body=b"{}", headers={})
    result = provider.verify_webhook(ctx)
    assert result.ok is False and "missing signature" in result.error


def test_verify_stale_timestamp(provider, keypair):
    private, _ = keypair
    ctx = _signed_ctx(
        private, _telnyx_event("call.answered"), timestamp=int(time.time()) - 3600
    )
    result = provider.verify_webhook(ctx)
    assert result.ok is False and "too old" in result.error


def test_verify_garbage_signature(provider):
    ctx = WebhookContext(
        method="POST", path="/voice/webhook", body=b"{}",
        headers={
            "telnyx-signature-ed25519": "!!!not-base64!!!",
            "telnyx-timestamp": str(int(time.time())),
        },
    )
    assert provider.verify_webhook(ctx).ok is False


# -- event parsing ---------------------------------------------------------------


def _parse_single(provider, body):
    ctx = WebhookContext(
        method="POST", path="/voice/webhook", body=json.dumps(body).encode()
    )
    result = provider.parse_webhook(ctx)
    assert result.response_status == 200
    return result.events


def test_parse_call_initiated_inbound(provider):
    events = _parse_single(
        provider,
        _telnyx_event(
            "call.initiated",
            call_control_id="v3:abc", direction="incoming",
            **{"from": "+15555550009", "to": "+15555550000"},
        ),
    )
    assert len(events) == 1
    event = events[0]
    assert event.type == EventType.CALL_INITIATED
    assert event.direction == "inbound"
    assert event.provider_call_id == "v3:abc"
    assert event.from_number == "+15555550009"
    assert event.dedupe_key == "evt-1"


def test_parse_client_state_roundtrips_call_id(provider):
    call_id = "vc-roundtrip"
    events = _parse_single(
        provider,
        _telnyx_event(
            "call.answered",
            call_control_id="v3:abc",
            client_state=base64.b64encode(call_id.encode()).decode(),
        ),
    )
    assert events[0].call_id == call_id


def test_parse_transcription_event(provider):
    events = _parse_single(
        provider,
        _telnyx_event(
            "call.transcription",
            call_control_id="v3:abc",
            transcription_data={"transcript": "hello world", "is_final": True},
        ),
    )
    event = events[0]
    assert event.type == EventType.CALL_SPEECH
    assert event.text == "hello world" and event.is_final is True


def test_parse_hangup_cause_mapping(provider):
    for cause, expected in (
        ("normal_clearing", "completed"),
        ("user_hangup", "hangup-user"),
        ("user_busy", "busy"),
        ("no_answer", "no-answer"),
        ("machine_detected", "voicemail"),
        ("service_unavailable", "failed"),
        ("some_future_cause", "completed"),
    ):
        events = _parse_single(
            provider,
            _telnyx_event("call.hangup", call_control_id="v3:x", hangup_cause=cause),
        )
        assert events[0].type == EventType.CALL_ENDED
        assert events[0].reason == expected, cause


def test_parse_dtmf_and_speak_started(provider):
    events = _parse_single(
        provider, _telnyx_event("call.dtmf.received", call_control_id="v3:x", digit="5")
    )
    assert events[0].type == EventType.CALL_DTMF and events[0].digits == "5"
    events = _parse_single(
        provider,
        _telnyx_event("call.speak.started", call_control_id="v3:x", text="hi"),
    )
    assert events[0].type == EventType.CALL_SPEAKING


def test_parse_ignores_streaming_and_unknown_events(provider):
    assert _parse_single(provider, _telnyx_event("streaming.started")) == []
    assert _parse_single(provider, _telnyx_event("call.fork.started")) == []
    assert _parse_single(provider, {"not": "telnyx"}) == []


def test_parse_bad_json_400(provider):
    ctx = WebhookContext(method="POST", path="/voice/webhook", body=b"\xff{nope")
    assert provider.parse_webhook(ctx).response_status == 400


# -- request shapes ----------------------------------------------------------------


@pytest.fixture
def api_calls(provider, monkeypatch):
    calls = []

    async def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return {"data": {"call_control_id": "v3:new", "is_alive": True}}

    import plugins.platforms.voice_call.providers.telnyx as telnyx_mod

    monkeypatch.setattr(telnyx_mod, "guarded_json_request", fake_request)
    return calls


def _call_record(**kwargs):
    return CallRecord(
        call_id="vc-test", provider="telnyx", direction="outbound",
        from_number="+15555550000", to_number="+15555550001",
        provider_call_id=kwargs.pop("provider_call_id", "v3:abc"), **kwargs,
    )


@pytest.mark.asyncio
async def test_initiate_call_request_shape(provider, api_calls):
    record = _call_record(provider_call_id=None)
    provider_call_id = await provider.initiate_call(record)
    assert provider_call_id == "v3:new"
    request = api_calls[0]
    assert request["url"].endswith("/v2/calls")
    body = request["json_body"]
    assert body["connection_id"] == "conn-1"
    assert body["to"] == "+15555550001" and body["from"] == "+15555550000"
    assert body["webhook_url"] == "https://hooks.example/voice/webhook"
    assert base64.b64decode(body["client_state"]).decode() == "vc-test"
    assert "stream_url" not in body
    assert request["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_initiate_call_includes_streaming_fields(provider, api_calls):
    record = _call_record(provider_call_id=None)
    record.metadata["stream_url"] = "wss://hooks.example/voice/stream/tok"
    record.metadata["stream_auth_token"] = "tok"
    await provider.initiate_call(record)
    body = api_calls[0]["json_body"]
    assert body["stream_url"] == "wss://hooks.example/voice/stream/tok"
    assert body["stream_codec"] == "PCMU"
    assert body["stream_bidirectional_mode"] == "rtp"
    assert body["stream_bidirectional_sampling_rate"] == 8000
    assert body["stream_auth_token"] == "tok"


@pytest.mark.asyncio
async def test_action_request_shapes(provider, api_calls):
    record = _call_record()
    await provider.answer_call(record)
    await provider.speak(record, "hello caller")
    await provider.send_dtmf(record, "12#")
    await provider.start_listening(record)
    await provider.stop_listening(record)
    await provider.hangup_call(record)
    urls = [c["url"] for c in api_calls]
    assert urls == [
        "https://api.telnyx.com/v2/calls/v3:abc/actions/answer",
        "https://api.telnyx.com/v2/calls/v3:abc/actions/speak",
        "https://api.telnyx.com/v2/calls/v3:abc/actions/send_dtmf",
        "https://api.telnyx.com/v2/calls/v3:abc/actions/transcription_start",
        "https://api.telnyx.com/v2/calls/v3:abc/actions/transcription_stop",
        "https://api.telnyx.com/v2/calls/v3:abc/actions/hangup",
    ]
    assert api_calls[1]["json_body"]["payload"] == "hello caller"
    assert api_calls[2]["json_body"]["digits"] == "12#"


@pytest.mark.asyncio
async def test_get_call_status_mapping(provider, monkeypatch):
    import plugins.platforms.voice_call.providers.telnyx as telnyx_mod

    async def alive(method, url, **kwargs):
        return {"data": {"is_alive": True, "state": "active"}}

    monkeypatch.setattr(telnyx_mod, "guarded_json_request", alive)
    status = await provider.get_call_status(_call_record())
    assert status.is_active and not status.is_terminal

    async def gone(method, url, **kwargs):
        return None  # 404

    monkeypatch.setattr(telnyx_mod, "guarded_json_request", gone)
    status = await provider.get_call_status(_call_record())
    assert status.is_terminal

    async def unknown(method, url, **kwargs):
        return {"data": {"state": "weird"}}  # no is_alive

    monkeypatch.setattr(telnyx_mod, "guarded_json_request", unknown)
    status = await provider.get_call_status(_call_record())
    assert status.is_unknown and not status.is_terminal

    async def boom(method, url, **kwargs):
        raise _http.ProviderApiError("Telnyx API error: HTTP 502")

    monkeypatch.setattr(telnyx_mod, "guarded_json_request", boom)
    status = await provider.get_call_status(_call_record())
    assert status.is_unknown


# -- host guard ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_guard_rejects_foreign_hosts():
    with pytest.raises(_http.ProviderApiError, match="refusing request"):
        await _http.guarded_json_request(
            "POST", "https://evil.example/v2/calls", allowed_host="api.telnyx.com"
        )


def test_streaming_fields_builder():
    fields = build_streaming_fields("wss://x/y", None)
    assert "stream_auth_token" not in fields
    assert fields["stream_bidirectional_target_legs"] == "self"


# -- tunnel -----------------------------------------------------------------------


def test_parse_ngrok_log_lines():
    assert (
        parse_ngrok_log_line(
            '{"msg":"started tunnel","url":"https://abc.ngrok-free.app"}'
        )
        == "https://abc.ngrok-free.app"
    )
    assert (
        parse_ngrok_log_line(
            '{"addr":"http://localhost:3334","url":"https://abc.ngrok.app","msg":"x"}'
        )
        == "https://abc.ngrok.app"
    )
    assert parse_ngrok_log_line('{"msg":"no url here"}') is None
    assert parse_ngrok_log_line("not json at all") is None


@pytest.mark.asyncio
async def test_start_tunnel_unknown_provider():
    cfg = VoiceCallConfig.from_extra({"provider": "mock"})
    cfg.tunnel.provider = "carrier-pigeon"
    with pytest.raises(ValueError):
        await start_tunnel(cfg)
