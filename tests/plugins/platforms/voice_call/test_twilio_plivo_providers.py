"""Twilio and Plivo providers: signatures, event parsing, request shapes."""

import base64
import hashlib
import hmac
from urllib.parse import urlencode

import pytest

from plugins.platforms.voice_call.config import VoiceCallConfig
from plugins.platforms.voice_call.events import CallRecord, EventType
from plugins.platforms.voice_call.providers.base import WebhookContext
from plugins.platforms.voice_call.providers.plivo import (
    PlivoProvider,
    compute_plivo_v2_signature,
    compute_plivo_v3_signature,
)
from plugins.platforms.voice_call.providers.twilio import (
    TwilioProvider,
    compute_twilio_signature,
)

PUBLIC = "https://hooks.example"
PATH = "/voice/webhook"


@pytest.fixture
def twilio(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tw-secret")
    cfg = VoiceCallConfig.from_extra({"provider": "twilio", "public_url": PUBLIC})
    provider = TwilioProvider(cfg)
    provider.set_public_url(PUBLIC)
    return provider


@pytest.fixture
def plivo(monkeypatch):
    monkeypatch.setenv("PLIVO_AUTH_ID", "MA123")
    monkeypatch.setenv("PLIVO_AUTH_TOKEN", "pl-secret")
    cfg = VoiceCallConfig.from_extra({"provider": "plivo", "public_url": PUBLIC})
    provider = PlivoProvider(cfg)
    provider.set_public_url(PUBLIC)
    return provider


def _form_ctx(params: dict, headers: dict = None, url: str = None, query: dict = None):
    body = urlencode(params).encode()
    return WebhookContext(
        method="POST", path=PATH, body=body,
        headers=headers or {}, query=query or {},
        url=url or f"{PUBLIC}{PATH}",
    )


def _call(provider_call_id="CA1", mode="conversation"):
    return CallRecord(
        call_id="vc-x", provider="x", direction="outbound",
        from_number="+15555550000", to_number="+15555550001",
        provider_call_id=provider_call_id, mode=mode,
    )


@pytest.fixture
def api_calls(monkeypatch):
    calls = []

    async def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return {
            "sid": "CA-new", "status": "queued",          # twilio shapes
            "request_uuid": "req-new", "call_status": "in-progress",  # plivo
        }

    import plugins.platforms.voice_call.providers.plivo as plivo_mod
    import plugins.platforms.voice_call.providers.twilio as twilio_mod

    monkeypatch.setattr(twilio_mod, "guarded_json_request", fake_request)
    monkeypatch.setattr(plivo_mod, "guarded_json_request", fake_request)
    return calls


# -- credentials ----------------------------------------------------------------


def test_credentials_required(monkeypatch):
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        TwilioProvider(VoiceCallConfig.from_extra({"provider": "twilio"}))
    with pytest.raises(ValueError):
        PlivoProvider(VoiceCallConfig.from_extra({"provider": "plivo"}))


# -- Twilio signature ---------------------------------------------------------------


def test_twilio_signature_valid_and_tampered(twilio):
    params = {"CallSid": "CA1", "CallStatus": "ringing", "From": "+15555550009"}
    url = f"{PUBLIC}{PATH}"
    signature = compute_twilio_signature("tw-secret", url, params)
    ctx = _form_ctx(params, headers={"X-Twilio-Signature": signature})
    assert twilio.verify_webhook(ctx).ok is True

    bad = _form_ctx({**params, "From": "+19999999999"},
                    headers={"X-Twilio-Signature": signature})
    assert twilio.verify_webhook(bad).ok is False


def test_twilio_signature_port_variant(twilio):
    params = {"CallSid": "CA1"}
    # Twilio signed the URL without the port; we received it with one.
    signature = compute_twilio_signature("tw-secret", f"{PUBLIC}{PATH}", params)
    ctx = _form_ctx(params, headers={"X-Twilio-Signature": signature},
                    url=f"https://hooks.example:443{PATH}")
    assert twilio.verify_webhook(ctx).ok is True


def test_twilio_signature_missing(twilio):
    assert twilio.verify_webhook(_form_ctx({"CallSid": "CA1"})).ok is False


# -- Twilio parsing -------------------------------------------------------------------


def test_twilio_parse_lifecycle_and_speech(twilio):
    result = twilio.parse_webhook(_form_ctx(
        {"CallSid": "CA1", "CallStatus": "ringing", "Direction": "inbound",
         "From": "+15555550009", "To": "+15555550000"},
    ))
    event = result.events[0]
    assert event.type == EventType.CALL_RINGING and event.direction == "inbound"
    assert "Pause" in result.response_body  # voice request → keep-alive TwiML

    result = twilio.parse_webhook(_form_ctx(
        {"CallSid": "CA1", "CallStatus": "in-progress"},
        query={"callId": "vc-x", "type": "status"},
    ))
    assert result.events[0].type == EventType.CALL_ANSWERED
    assert result.events[0].call_id == "vc-x"
    assert "Pause" not in result.response_body  # status callback → empty ack

    result = twilio.parse_webhook(_form_ctx(
        {"CallSid": "CA1", "SpeechResult": "hello there", "Confidence": "0.9"},
        query={"callId": "vc-x"},
    ))
    event = result.events[0]
    assert event.type == EventType.CALL_SPEECH and event.text == "hello there"

    result = twilio.parse_webhook(_form_ctx({"CallSid": "CA1", "Digits": "5"}))
    assert result.events[0].type == EventType.CALL_DTMF

    result = twilio.parse_webhook(_form_ctx(
        {"CallSid": "CA1", "CallStatus": "no-answer"}))
    assert result.events[0].type == EventType.CALL_ENDED
    assert result.events[0].reason == "no-answer"


# -- Twilio request shapes ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_twilio_initiate_and_speak_shapes(twilio, api_calls):
    record = _call(provider_call_id=None)
    record.call_id = "vc-x"
    sid = await twilio.initiate_call(record)
    assert sid == "CA-new"
    request = api_calls[0]
    assert request["url"].endswith("/Accounts/AC123/Calls.json")
    assert request["auth"] == ("AC123", "tw-secret")
    form = request["form_body"]
    assert form["To"] == "+15555550001"
    assert "callId=vc-x" in form["Url"]
    assert "type=status" in form["StatusCallback"]
    assert form["StatusCallbackEvent"] == ["initiated", "ringing", "answered",
                                           "completed"]

    record.provider_call_id = "CA-new"
    await twilio.speak(record, 'Hello <world> & "friends"')
    twiml = api_calls[1]["form_body"]["Twiml"]
    assert "&lt;world&gt;" in twiml and "&amp;" in twiml  # XML-escaped
    assert "<Gather" in twiml and "callId=vc-x" in twiml  # conversation embeds Gather

    notify = _call(provider_call_id="CA-new", mode="notify")
    await twilio.speak(notify, "bye")
    assert "<Gather" not in api_calls[2]["form_body"]["Twiml"]

    await twilio.send_dtmf(record, "1w2#x")  # x filtered out
    assert 'digits="1w2#"' in api_calls[3]["form_body"]["Twiml"]

    await twilio.hangup_call(record)
    assert api_calls[4]["form_body"] == {"Status": "completed"}


@pytest.mark.asyncio
async def test_twilio_status_mapping(twilio, monkeypatch):
    import plugins.platforms.voice_call.providers.twilio as twilio_mod

    async def status(value):
        async def fake(method, url, **kwargs):
            return {"status": value}
        monkeypatch.setattr(twilio_mod, "guarded_json_request", fake)
        return await twilio.get_call_status(_call())

    assert (await status("in-progress")).is_active
    assert (await status("completed")).is_terminal
    assert (await status("busy")).is_terminal


# -- Plivo signature -----------------------------------------------------------------


def test_plivo_v3_signature_valid_and_invalid(plivo):
    params = [("CallUUID", "uuid-1"), ("Event", "StartApp")]
    url = f"{PUBLIC}{PATH}?callId=vc-x&flow=answer"
    nonce = "nonce123"
    signature = compute_plivo_v3_signature("pl-secret", "POST", url, params, nonce)
    ctx = WebhookContext(
        method="POST", path=PATH, body=urlencode(params).encode(),
        headers={"X-Plivo-Signature-V3": signature,
                 "X-Plivo-Signature-V3-Nonce": nonce},
        query={"callId": "vc-x", "flow": "answer"}, url=url,
    )
    result = plivo.verify_webhook(ctx)
    assert result.ok is True and result.dedupe_key.startswith("plivo:v3:")

    ctx.body = urlencode([("CallUUID", "tampered")]).encode()
    assert plivo.verify_webhook(ctx).ok is False


def test_plivo_v3_multiple_signatures_header(plivo):
    params = [("CallUUID", "uuid-1")]
    url = f"{PUBLIC}{PATH}"
    nonce = "n1"
    good = compute_plivo_v3_signature("pl-secret", "POST", url, params, nonce)
    ctx = WebhookContext(
        method="POST", path=PATH, body=urlencode(params).encode(),
        headers={"X-Plivo-Signature-V3": f"bogus123==,{good}",
                 "X-Plivo-Signature-V3-Nonce": nonce},
        url=url,
    )
    assert plivo.verify_webhook(ctx).ok is True


def test_plivo_v2_fallback(plivo):
    url = f"{PUBLIC}{PATH}"
    nonce = "n2"
    signature = compute_plivo_v2_signature("pl-secret", url, nonce)
    ctx = WebhookContext(
        method="POST", path=PATH, body=b"",
        headers={"X-Plivo-Signature-V2": signature,
                 "X-Plivo-Signature-V2-Nonce": nonce},
        url=url,
    )
    result = plivo.verify_webhook(ctx)
    assert result.ok is True and result.dedupe_key.startswith("plivo:v2:")
    assert plivo.verify_webhook(
        WebhookContext(method="POST", path=PATH, body=b"", url=url)
    ).ok is False


# -- Plivo parsing --------------------------------------------------------------------


def test_plivo_parse_lifecycle_speech_dtmf(plivo):
    result = plivo.parse_webhook(_form_ctx(
        {"CallUUID": "uuid-1", "Event": "StartApp", "Direction": "inbound",
         "From": "+15555550009", "To": "+15555550000"},
        query={"flow": "answer"},
    ))
    event = result.events[0]
    assert event.type == EventType.CALL_ANSWERED
    assert "Wait" in result.response_body  # answer flow → keep-alive XML

    result = plivo.parse_webhook(_form_ctx(
        {"CallUUID": "uuid-1", "Speech": "hi bot"}, query={"flow": "getinput",
                                                           "callId": "vc-x"},
    ))
    event = result.events[0]
    assert event.type == EventType.CALL_SPEECH and event.text == "hi bot"
    assert event.call_id == "vc-x"

    result = plivo.parse_webhook(_form_ctx({"CallUUID": "uuid-1", "Digits": "7"}))
    assert result.events[0].type == EventType.CALL_DTMF

    result = plivo.parse_webhook(_form_ctx(
        {"CallUUID": "uuid-1", "CallStatus": "completed"}, query={"flow": "hangup"}))
    assert result.events[0].type == EventType.CALL_ENDED
    assert result.events[0].reason == "completed"


def test_plivo_xml_speak_flow_serves_pending_text(plivo):
    plivo._pending_speak["vc-x"] = 'Say <hello> & welcome'
    result = plivo.parse_webhook(_form_ctx(
        {"CallUUID": "uuid-1"}, query={"flow": "xml-speak", "callId": "vc-x"}))
    assert result.events == []
    assert "&lt;hello&gt;" in result.response_body
    assert "<GetInput" in result.response_body
    assert "flow=getinput" in result.response_body
    # Pending text is one-shot.
    again = plivo.parse_webhook(_form_ctx(
        {"CallUUID": "uuid-1"}, query={"flow": "xml-speak", "callId": "vc-x"}))
    assert "<Speak" not in again.response_body


# -- Plivo request shapes -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_plivo_initiate_speak_dtmf_hangup_shapes(plivo, api_calls):
    record = _call(provider_call_id=None)
    record.call_id = "vc-x"
    request_uuid = await plivo.initiate_call(record)
    assert request_uuid == "req-new"
    request = api_calls[0]
    assert request["url"].endswith("/Account/MA123/Call/")
    body = request["json_body"]
    assert body["to"] == "+15555550001"
    assert "callId=vc-x" in body["answer_url"] and "flow=answer" in body["answer_url"]
    assert "flow=hangup" in body["hangup_url"]

    record.provider_call_id = "uuid-1"
    await plivo.speak(record, "hello")
    transfer = api_calls[1]
    assert transfer["url"].endswith("/Call/uuid-1/")
    assert transfer["json_body"]["legs"] == "aleg"
    assert "flow=xml-speak" in transfer["json_body"]["aleg_url"]
    assert plivo._pending_speak["vc-x"] == "hello"

    await plivo.send_dtmf(record, "12w#")
    assert api_calls[2]["url"].endswith("/Call/uuid-1/DTMF/")

    await plivo.hangup_call(record)
    assert api_calls[3]["method"] == "DELETE"


# -- manager adopts swapped provider ids (Plivo request_uuid → CallUUID) -------------


@pytest.mark.asyncio
async def test_manager_adopts_new_provider_call_id(vc_config, provider, store):
    from plugins.platforms.voice_call.events import NormalizedEvent
    from plugins.platforms.voice_call.manager import CallManager

    manager = CallManager(vc_config, provider, store)
    provider.auto_answer = False
    record = await manager.initiate_call("+15555550001")
    old_id = record.provider_call_id

    await manager.process_event(NormalizedEvent(
        type=EventType.CALL_ANSWERED, provider="mock",
        call_id=record.call_id, provider_call_id="real-call-uuid",
    ))
    assert record.provider_call_id == "real-call-uuid"
    assert manager._by_provider_id.get("real-call-uuid") == record.call_id
    assert old_id not in manager._by_provider_id
    await manager.shutdown()
