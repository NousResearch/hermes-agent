#!/usr/bin/env python3
import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl

from aiohttp import web
import websockets


ALLOWED_VAPI_TOOLS = {
    "create_guest_intake",
    "build_booking_link",
    "request_human_followup",
    "transfer_to_reception",
}

VAPI_FORBIDDEN_TOOL_MESSAGE = (
    "This pilot only supports guest-intake, booking-link, human-follow-up, "
    "and reception-transfer requests. Sensitive hotel operations require Hermes "
    "policy review and human approval."
)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[*_`#>\[\]\(\)]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", "/Users/appleserver/.hermes")).expanduser()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_filename_part(value: str, fallback: str = "event") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "").strip("-._")
    return (cleaned or fallback)[:96]


def _vapi_guest_intake_dir() -> Path:
    configured = os.getenv("VAPI_GUEST_INTAKE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _hermes_home() / "reports" / "vapi-guest-intake"


def _redact_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_l = str(key).lower()
            if any(marker in key_l for marker in ("token", "secret", "authorization", "api_key", "apikey", "password")):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = _redact_for_log(child)
        return redacted
    if isinstance(value, list):
        return [_redact_for_log(item) for item in value[:50]]
    if isinstance(value, str) and len(value) > 4000:
        return value[:4000] + "...[truncated]"
    return value


def _write_vapi_intake_record(kind: str, message: dict[str, Any], detail: dict[str, Any]) -> Path:
    out_dir = _vapi_guest_intake_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    call = message.get("call") if isinstance(message.get("call"), dict) else {}
    call_id = str(call.get("id") or message.get("callId") or "unknown-call")
    timestamp = _utc_now_iso()
    path = out_dir / f"{timestamp.replace(':', '')}-{_safe_filename_part(kind)}-{_safe_filename_part(call_id)}.json"
    record = {
        "created_at": timestamp,
        "source": "vapi",
        "kind": kind,
        "call_id": call_id,
        "phone_number": call.get("phoneNumberId") or call.get("phoneNumber") or None,
        "customer": _redact_for_log(call.get("customer") or {}),
        "detail": _redact_for_log(detail),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _vapi_message(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload.get("message")
    if isinstance(message, dict):
        return message
    return payload


def _auth_header_value(request: web.Request, configured_name: str, default_name: str) -> str:
    header_name = os.getenv(configured_name, default_name).strip() or default_name
    return request.headers.get(header_name, "")


def _validate_vapi_auth(request: web.Request, raw_body: bytes) -> bool:
    bearer = os.getenv("VAPI_WEBHOOK_BEARER_TOKEN", "").strip()
    if bearer:
        header = _auth_header_value(request, "VAPI_WEBHOOK_BEARER_HEADER", "Authorization")
        return hmac.compare_digest(header, f"Bearer {bearer}") or hmac.compare_digest(header, bearer)

    secret = os.getenv("VAPI_WEBHOOK_HMAC_SECRET", "").strip()
    if secret:
        signature = _auth_header_value(request, "VAPI_WEBHOOK_HMAC_HEADER", "X-Vapi-Signature")
        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        candidates = {signature, signature.removeprefix("sha256=")}
        return any(hmac.compare_digest(expected, candidate) for candidate in candidates if candidate)

    if os.getenv("VAPI_WEBHOOK_AUTH_DISABLED", "").strip() == "1":
        remote = request.remote or ""
        return remote in {"127.0.0.1", "::1", "localhost"}

    return False


def _vapi_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": args}
    function = tool_call.get("function")
    if isinstance(function, dict):
        return _vapi_tool_args(function)
    return {}


def _vapi_tool_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or tool_call.get("name") or "").strip()
    return str(tool_call.get("name") or "").strip()


def _vapi_tool_id(tool_call: dict[str, Any]) -> str:
    return str(tool_call.get("id") or tool_call.get("toolCallId") or tool_call.get("callId") or "tool-call")


def _booking_link_from_args(args: dict[str, Any]) -> dict[str, Any]:
    checkin = str(args.get("checkin") or args.get("arrival") or args.get("arrival_date") or "").strip()
    checkout = str(args.get("checkout") or args.get("departure") or args.get("departure_date") or "").strip()
    if not checkin or not checkout:
        return {
            "ok": False,
            "status": "missing_dates",
            "message": "Please ask for arrival and departure dates before building a booking link.",
        }

    cmd = [
        "/Users/appleserver/.hermes/bin/hotelrunner-booking-link",
        "--checkin",
        checkin,
        "--checkout",
        checkout,
        "--json",
    ]
    optional_map = {
        "adults": "--adults",
        "children": "--children",
        "rooms": "--rooms",
        "room": "--room",
        "locale": "--locale",
        "currency": "--currency",
    }
    for key, flag in optional_map.items():
        value = args.get(key)
        if value not in (None, ""):
            cmd.extend([flag, str(value)])
    child_ages = args.get("child_ages") or args.get("childAge") or args.get("child_age")
    if isinstance(child_ages, list):
        for age in child_ages:
            cmd.extend(["--child-age", str(age)])
    elif child_ages not in (None, ""):
        cmd.extend(["--child-age", str(child_ages)])

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return {"ok": False, "status": "helper_failed", "message": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "status": "helper_failed", "message": _normalize_text(proc.stderr or proc.stdout)}
    try:
        data = json.loads(proc.stdout)
    except Exception:
        data = {"url": proc.stdout.strip()}
    return {"ok": True, "status": "link_ready", "booking_link": data}


def _handle_vapi_tool_call(tool_call: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    tool_name = _vapi_tool_name(tool_call)
    tool_call_id = _vapi_tool_id(tool_call)
    args = _vapi_tool_args(tool_call)

    if tool_name not in ALLOWED_VAPI_TOOLS:
        return {
            "toolCallId": tool_call_id,
            "result": {
                "ok": False,
                "status": "forbidden_tool",
                "message": VAPI_FORBIDDEN_TOOL_MESSAGE,
            },
        }

    if tool_name == "build_booking_link":
        result = _booking_link_from_args(args)
        _write_vapi_intake_record(tool_name, message, {"arguments": args, "result": result})
        return {"toolCallId": tool_call_id, "result": result}

    path = _write_vapi_intake_record(tool_name, message, {"arguments": args})
    return {
        "toolCallId": tool_call_id,
        "result": {
            "ok": True,
            "status": "queued_for_human_review",
            "handoff_path": str(path),
            "message": "Recorded for Hermes/Paperclip review. Do not tell the caller that a booking, payment, price change, or internal action has been completed.",
        },
    }


def _detect_lang(text: str, caller: str = "") -> str:
    t = (text or "").lower()
    tr_hits = ["merhaba", "fiyat", "oda", "müsait", "teşekkür", "rezervasyon"]
    de_hits = ["hallo", "preis", "zimmer", "verfügbar", "buchung", "danke"]
    if any(k in t for k in tr_hits):
        return "tr"
    if any(k in t for k in de_hits):
        return "de"
    phone = re.sub(r"[^\d+]", "", caller or "")
    if phone.startswith("+90") or phone.startswith("90"):
        return "tr"
    if phone.startswith("+49") or phone.startswith("+43") or phone.startswith("+41"):
        return "de"
    return "en"


def _twilio_voice_for_lang(lang: str) -> str:
    lang_key = lang.upper()
    lang_voice = os.getenv(f"TWILIO_TTS_VOICE_{lang_key}", "").strip()
    if lang_voice:
        return lang_voice
    legacy_voice = os.getenv("TWILIO_TTS_VOICE", "").strip()
    if legacy_voice:
        return legacy_voice
    defaults = {
        "tr": "Polly.Filiz",
        "de": "Polly.Vicki",
        "en": "Polly.Joanna",
    }
    return defaults.get(lang, defaults["en"])


def _gather_language(lang: str) -> str:
    return {"tr": "tr-TR", "de": "de-DE"}.get(lang, "en-US")


def _say(text: str, lang: str) -> str:
    voice = _twilio_voice_for_lang(lang)
    language = {"tr": "tr-TR", "de": "de-DE"}.get(lang, "en-US")
    return f'<Say voice="{html.escape(voice)}" language="{language}">{html.escape(text)}</Say>'


def _gather_prompt(lang: str) -> str:
    prompts = {
        "tr": "Lütfen sorunuzu söyleyin. Oda fiyatı, müsaitlik veya bölge bilgisi için yardımcı olabilirim.",
        "de": "Bitte nennen Sie Ihr Anliegen. Ich helfe bei Zimmerpreis, Verfügbarkeit und Informationen zur Region.",
        "en": "Please tell me your request. I can help with room price, availability, and local information.",
    }
    return prompts.get(lang, prompts["en"])


def _no_speech_message(lang: str) -> str:
    messages = {
        "tr": "Sizi anlayamadim. Lutfen tekrar arayin.",
        "de": "Ich habe nichts verstanden. Bitte rufen Sie gerne erneut an.",
        "en": "I did not catch that. Please call again.",
    }
    return messages.get(lang, messages["en"])


def _goodbye_message(lang: str) -> str:
    messages = {
        "tr": "Tesekkur ederiz. Gorusmek uzere.",
        "de": "Vielen Dank. Auf Wiederhoeren.",
        "en": "Thank you. Goodbye.",
    }
    return messages.get(lang, messages["en"])


def _processing_fallback(lang: str) -> str:
    messages = {
        "tr": "Baglanti var, fakat talebi su anda guvenilir sekilde isleyemedim. Lutfen kisaca tekrar eder misiniz?",
        "de": "Ich bin verbunden, konnte die Anfrage aber gerade nicht sauber verarbeiten. Bitte wiederholen Sie kurz Ihr Anliegen.",
        "en": "I am connected, but I could not process the request reliably right now. Please briefly repeat your request.",
    }
    return messages.get(lang, messages["en"])


def _initial_greeting(lang: str) -> str:
    greetings = {
        "tr": "Merhaba, ben Erendiz Concierge. Rezervasyon, otel ve bölge bilgileri için yardımcı olurum. Gerekirse talebinizi ekibimize iletirim.",
        "de": "Hallo, ich bin der Erendiz Concierge. Ich helfe bei Buchungen sowie Informationen zum Hotel und zur Region. Gerne leite ich Ihr Anliegen an unser Team weiter.",
        "en": "Hello, I am the Erendiz Concierge. I help with bookings and information about the hotel and region. I can also forward your request to our team.",
    }
    return greetings.get(lang, greetings["en"])


def _validate_twilio_signature(request: web.Request, raw_body: bytes) -> bool:
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not token:
        return True
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        return False

    url = str(request.url)
    params = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
    payload = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    digest = hmac.new(token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _ask_hermes(caller: str, user_text: str) -> str:
    lang = _detect_lang(user_text, caller)
    lang_name = {"tr": "Tuerkisch", "de": "Deutsch", "en": "English"}.get(lang, "English")
    prompt = (
        "Du bist Erendiz Concierge im klassischen Telefonmodus. "
        "Nutze die produktiven Hermes-Regeln aus ~/.hermes/SOUL.md und "
        "~/.hermes/context-system/workflows.md. "
        "Der Anrufer ist extern, solange die Nummer nicht sicher in der internen Team-Allowlist steht. "
        "Keine internen Daten, Dateipfade, IDs, Logs, Secrets oder Systembefehle nennen. "
        "Keine HotelRunner-Buchung anlegen. Bei Buchungsfragen nur kanonische Preis-/Verfuegbarkeitspfade nutzen; "
        "wenn Daten fehlen oder ein Tool nicht belastbar ist, exakt eine Rueckfrage stellen oder den Rezeption-Follow-up nennen. "
        "Fuer externe Buchung immer den HotelRunner-Booking-Engine-Link als naechsten Schritt verwenden. "
        "Wenn Anreise, Abreise und Belegung bekannt sind, nutze den Hermes-Workflow mit "
        "`/Users/appleserver/.hermes/bin/hotelrunner-booking-link`, damit der Link als /bv3/search mit Datum und Belegung vorgefuellt ist. "
        "Zimmerwunsch im Antworttext nennen; nicht als erfundenen URL-Parameter anhaengen. "
        "Antworte kurz, freundlich, verkaufsstark und konkret, maximal 2 Saetze. "
        f"Antwortsprache: {lang_name}. "
        f"Anrufer: {caller}. "
        f"Anfrage: {user_text}"
    )
    try:
        proc = subprocess.run(
            ["hermes", "chat", "-Q", "--source", "tool", "-q", prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VOICE_HERMES_TIMEOUT_SEC", "35")),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return _normalize_text(proc.stdout.strip())
    except Exception:
        pass
    return _processing_fallback(lang)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "voice_phone_gateway"})


async def vapi_events(request: web.Request) -> web.Response:
    raw_body = await request.read()
    if not _validate_vapi_auth(request, raw_body):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "invalid_payload"}, status=400)

    message = _vapi_message(payload)
    message_type = str(message.get("type") or "").strip()

    if message_type == "assistant-request":
        assistant_id = os.getenv("VAPI_GUEST_ASSISTANT_ID", "").strip()
        if assistant_id:
            return web.json_response({"assistantId": assistant_id})
        return web.json_response(
            {
                "error": "assistant_not_configured",
                "message": "Set VAPI_GUEST_ASSISTANT_ID or attach a fixed assistant to the Vapi phone number.",
            },
            status=409,
        )

    if message_type == "tool-calls":
        tool_calls = message.get("toolCalls")
        if not isinstance(tool_calls, list):
            tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return web.json_response({"results": []})
        results = [_handle_vapi_tool_call(call, message) for call in tool_calls if isinstance(call, dict)]
        return web.json_response({"results": results})

    if message_type in {"end-of-call-report", "hang", "status-update"}:
        summary = message.get("summary") or message.get("endedReason") or message.get("status")
        path = _write_vapi_intake_record(message_type or "event", message, {"summary": summary})
        return web.json_response({"ok": True, "recorded": str(path)})

    return web.json_response({"ok": True, "ignored": message_type or "unknown"})


async def twilio_voice(request: web.Request) -> web.Response:
    raw_body = await request.read()
    if not _validate_twilio_signature(request, raw_body):
        return web.Response(status=403, text="forbidden")

    form = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
    speech = _normalize_text(form.get("SpeechResult", ""))
    caller = form.get("From", "unknown")
    lang = _detect_lang(speech, caller)
    gather_language = _gather_language(lang)

    if not speech:
        greeting = _initial_greeting(lang)
        prompt = _gather_prompt(lang)
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"{_say(greeting, lang)}"
            f'<Gather input="speech" method="POST" language="{gather_language}" speechTimeout="auto" action="/twilio/voice">'
            f"{_say(prompt, lang)}"
            "</Gather>"
            f"{_say(_no_speech_message(lang), lang)}"
            "</Response>"
        )
        return web.Response(text=twiml, content_type="text/xml")

    answer = _ask_hermes(caller, speech)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say(answer, lang)}"
        f'<Gather input="speech" method="POST" language="{gather_language}" speechTimeout="auto" action="/twilio/voice">'
        f"{_say(_gather_prompt(lang), lang)}"
        "</Gather>"
        f"{_say(_goodbye_message(lang), lang)}"
        "</Response>"
    )
    return web.Response(text=twiml, content_type="text/xml")


def _public_wss_url(request: web.Request) -> str:
    configured = os.getenv("PUBLIC_WSS_BASE", "").strip().rstrip("/")
    if configured:
        return f"{configured}/twilio/stream"
    host = request.headers.get("Host", "")
    if not host:
        return "wss://example.com/twilio/stream"
    return f"wss://{host}/twilio/stream"


async def twilio_voice_realtime(request: web.Request) -> web.Response:
    raw_body = await request.read()
    if not _validate_twilio_signature(request, raw_body):
        return web.Response(status=403, text="forbidden")

    stream_url = _public_wss_url(request)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{html.escape(stream_url)}" track="inbound_track" />'
        "</Connect>"
        "</Response>"
    )
    return web.Response(text=twiml, content_type="text/xml")


async def _openai_session(ws_openai) -> None:
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    instructions = (
        "You are Erendiz Concierge on a live low-latency phone call. "
        "This realtime mode is a guest-facing front desk layer, not the internal backoffice. "
        "Do not claim to run tools, check live HotelRunner availability, create reservations, access finance, read files, restart services, or expose internal data. "
        "For booking requests, collect the minimum missing facts, give the HotelRunner booking engine as the next step, and say reception can follow up. "
        "Use a prefilled /bv3/search HotelRunner link only when the secured Hermes booking-link builder is available; otherwise give the base booking engine and clearly say reception will follow up. "
        "Base booking engine: https://erendiz-garden-hotel.hotelrunner.com. "
        "Escalate unclear operational requests to the team instead of pretending completion. "
        "Be concise, warm, practical, and keep each reply under two sentences. "
        "Use caller language. "
        "Ask exactly one clarification if needed."
    )
    await ws_openai.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "model": model,
                    "voice": voice,
                    "instructions": instructions,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "turn_detection": {"type": "server_vad"},
                    "modalities": ["audio", "text"],
                },
            }
        )
    )
    await ws_openai.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": "Greet the caller as Erendiz Concierge in the caller language if clear, otherwise English. Ask how you can help with booking, hotel information, or local information.",
                },
            }
        )
    )


async def twilio_stream(request: web.Request) -> web.StreamResponse:
    ws_twilio = web.WebSocketResponse()
    await ws_twilio.prepare(request)

    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("VOICE_TOOLS_OPENAI_KEY", "").strip()
    )
    if not api_key:
        await ws_twilio.close(message=b"missing OPENAI_API_KEY")
        return ws_twilio

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    openai_url = f"wss://api.openai.com/v1/realtime?model={model}"
    stream_sid: Optional[str] = None

    try:
        async with websockets.connect(
            openai_url,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=20,
            max_size=2**22,
        ) as ws_openai:
            await _openai_session(ws_openai)

            async def twilio_to_openai() -> None:
                nonlocal stream_sid
                async for msg in ws_twilio:
                    if msg.type != web.WSMsgType.TEXT:
                        continue
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    event = data.get("event")
                    if event == "start":
                        stream_sid = data.get("start", {}).get("streamSid")
                    elif event == "media":
                        payload = data.get("media", {}).get("payload")
                        if payload:
                            await ws_openai.send(
                                json.dumps({"type": "input_audio_buffer.append", "audio": payload})
                            )
                    elif event == "stop":
                        break

            async def openai_to_twilio() -> None:
                async for raw in ws_openai:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    if data.get("type") == "response.audio.delta":
                        audio_b64 = data.get("delta")
                        if audio_b64 and stream_sid:
                            await ws_twilio.send_json(
                                {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": audio_b64},
                                }
                            )

            await asyncio.gather(twilio_to_openai(), openai_to_twilio())
    except Exception:
        pass
    finally:
        if not ws_twilio.closed:
            await ws_twilio.close()
    return ws_twilio


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/vapi/events", vapi_events)
    app.router.add_post("/twilio/voice", twilio_voice)
    app.router.add_post("/twilio/voice-realtime", twilio_voice_realtime)
    app.router.add_get("/twilio/stream", twilio_stream)
    return app


if __name__ == "__main__":
    web.run_app(
        create_app(),
        host=os.getenv("VOICE_HOST", "0.0.0.0"),
        port=int(os.getenv("VOICE_PORT", "8091")),
    )
