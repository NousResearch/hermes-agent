"""FastAPI Twilio voice service for the Hermes voice_call plugin.

Run with:
    uvicorn plugins.voice_call.service:app --host 0.0.0.0 --port 8765

This module intentionally keeps the MVP small and dependency-light:
* Twilio REST calls use httpx basic auth instead of requiring twilio-python.
* STT/TTS/agent providers are represented by async seams so real streaming
  providers can be swapped in without changing routes.
* Call state is in-memory. Production deployments should replace CallStore
  with a durable store if transcripts must survive restarts.
"""

from __future__ import annotations

import asyncio
import html
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .config import CallerProfile
from .tools import normalize_phone, redact_phone


@dataclass
class CallSession:
    call_id: str
    to: str = ""
    from_number: str = ""
    purpose: str = ""
    context: str = ""
    escalation_policy: str = "no_escalation"
    caller_profile: dict[str, Any] = field(default_factory=dict)
    twilio_sid: str = ""
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["to"] = redact_phone(self.to)
        data["from_number"] = redact_phone(self.from_number)
        profile = dict(data.get("caller_profile") or {})
        for key in ("callback_number", "transfer_number"):
            if profile.get(key):
                profile[key] = redact_phone(str(profile[key]))
        data["caller_profile"] = profile
        return data


class CallStore:
    def __init__(self) -> None:
        self._calls: dict[str, CallSession] = {}
        self._sid_to_id: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: CallSession) -> CallSession:
        async with self._lock:
            self._calls[session.call_id] = session
            if session.twilio_sid:
                self._sid_to_id[session.twilio_sid] = session.call_id
        return session

    async def get(self, call_id: str) -> CallSession | None:
        async with self._lock:
            return self._calls.get(call_id) or self._calls.get(self._sid_to_id.get(call_id, ""))

    async def update_status(self, call_id: str, status: str, twilio_sid: str = "") -> CallSession | None:
        async with self._lock:
            session = self._calls.get(call_id) or self._calls.get(self._sid_to_id.get(call_id, ""))
            if not session:
                return None
            session.status = status or session.status
            if twilio_sid:
                session.twilio_sid = twilio_sid
                self._sid_to_id[twilio_sid] = session.call_id
            return session

    async def append_transcript(self, call_id: str, speaker: str, text: str, **meta: Any) -> None:
        async with self._lock:
            session = self._calls.get(call_id) or self._calls.get(self._sid_to_id.get(call_id, ""))
            if session:
                session.transcript.append({"ts": time.time(), "speaker": speaker, "text": text, **meta})


STORE = CallStore()
app = FastAPI(title="Hermes Voice Call Service", version="0.1.0")


def _public_base_url() -> str:
    return os.environ.get("VOICE_CALL_PUBLIC_BASE_URL") or os.environ.get("VOICE_CALL_SERVICE_URL", "")


def _twilio_configured() -> bool:
    return all(os.environ.get(name) for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"))


def _twiml(*body: str) -> Response:
    return Response("<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response>" + "".join(body) + "</Response>", media_type="application/xml")


def _say(text: str) -> str:
    return f"<Say>{html.escape(text)}</Say>"


def _stream(call_id: str) -> str:
    base = _public_base_url().rstrip("/")
    if not base:
        return ""
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    return f'<Connect><Stream url="{html.escape(ws_url)}/twilio/media-stream"><Parameter name="call_id" value="{html.escape(call_id)}" /></Stream></Connect>'


def _caller_profile(raw: Any) -> dict[str, Any]:
    profile = raw if isinstance(raw, dict) else {}
    default = CallerProfile().__dict__
    merged = {**default, **{k: str(v) for k, v in profile.items() if v is not None}}
    return merged


async def _read_payload(request: Request) -> dict[str, Any]:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    form = await request.form()
    return dict(form)


async def _create_twilio_call(session: CallSession) -> tuple[str, bool]:
    """Return (twilio_sid, dry_run)."""
    if not _twilio_configured():
        return "", True
    base = _public_base_url().rstrip("/")
    if not base:
        return "", True
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_FROM_NUMBER"]
    session.from_number = from_number
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
    callback = f"{base}/twilio/voice/inbound?call_id={session.call_id}"
    status_callback = f"{base}/twilio/status?call_id={session.call_id}"
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(
            url,
            data={
                "To": session.to,
                "From": from_number,
                "Url": callback,
                "StatusCallback": status_callback,
                "StatusCallbackEvent": "initiated ringing answered completed",
            },
            auth=(account_sid, auth_token),
        )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("sid") or ""), False


@app.post("/twilio/voice/outbound")
async def outbound(request: Request) -> JSONResponse:
    payload = await _read_payload(request)
    to = normalize_phone(str(payload.get("to") or ""))
    purpose = str(payload.get("purpose") or "").strip()
    context = str(payload.get("context") or "").strip()
    if not to or not purpose or not context:
        return JSONResponse({"success": False, "error": "to, purpose, and context are required"}, status_code=400)
    session = CallSession(
        call_id="vc_" + uuid.uuid4().hex[:16],
        to=to,
        purpose=purpose,
        context=context,
        escalation_policy=str(payload.get("escalation_policy") or "no_escalation"),
        caller_profile=_caller_profile(payload.get("caller_profile")),
        status="queued",
    )
    await STORE.create(session)
    try:
        sid, dry_run = await _create_twilio_call(session)
        if sid:
            session.twilio_sid = sid
            await STORE.update_status(session.call_id, "initiated", sid)
        return JSONResponse({"success": True, "dry_run": dry_run, "call": session.public_dict()})
    except httpx.HTTPStatusError as exc:
        await STORE.update_status(session.call_id, "failed")
        return JSONResponse({"success": False, "error": f"twilio HTTP {exc.response.status_code}", "detail": exc.response.text[:1000]}, status_code=502)


@app.api_route("/twilio/voice/inbound", methods=["GET", "POST"])
async def inbound(request: Request) -> Response:
    payload = await _read_payload(request) if request.method == "POST" else dict(request.query_params)
    call_id = str(payload.get("call_id") or payload.get("CallSid") or "")
    session = await STORE.get(call_id) if call_id else None
    if not session:
        # Unknown inbound calls still get a transparent disclosure and message path.
        return _twiml(_say("This is Hermes, an AI assistant. I do not have call context for this number. Please leave a message after the tone."), "<Record maxLength=\"120\" transcribe=\"false\" />")
    profile = session.caller_profile
    intro = profile.get("disclosure") or f"This is {profile.get('assistant_name', 'Hermes')}, calling on behalf of {profile.get('on_behalf_of', 'Jason Lai')}."
    await STORE.append_transcript(session.call_id, "system", intro, event="disclosure")
    stream = _stream(session.call_id)
    fallback = _say(f"{intro} The purpose of this call is: {session.purpose}. {session.context}")
    return _twiml(fallback, stream or "<Pause length=\"1\" />")


@app.websocket("/twilio/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    call_id = ""
    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")
            if event == "start":
                params = ((message.get("start") or {}).get("customParameters") or {})
                call_id = str(params.get("call_id") or (message.get("start") or {}).get("callSid") or "")
                if call_id:
                    await STORE.update_status(call_id, "in-progress", str((message.get("start") or {}).get("callSid") or ""))
                    await STORE.append_transcript(call_id, "system", "media stream started", event="start")
            elif event == "media":
                # TODO: feed message["media"]["payload"] to the configured STT provider.
                continue
            elif event == "stop":
                if call_id:
                    await STORE.append_transcript(call_id, "system", "media stream stopped", event="stop")
                break
    except WebSocketDisconnect:
        if call_id:
            await STORE.append_transcript(call_id, "system", "media stream disconnected", event="disconnect")


@app.api_route("/twilio/status", methods=["GET", "POST"])
async def status(request: Request) -> JSONResponse:
    payload = await _read_payload(request) if request.method == "POST" else dict(request.query_params)
    call_id = str(payload.get("call_id") or payload.get("CallSid") or "")
    status_value = str(payload.get("CallStatus") or payload.get("status") or "")
    session = await STORE.update_status(call_id, status_value, str(payload.get("CallSid") or "")) if call_id else None
    return JSONResponse({"success": True, "call": session.public_dict() if session else None})


@app.post("/sms/inbound")
async def sms_inbound(request: Request) -> Response:
    payload = await _read_payload(request)
    from_number = redact_phone(str(payload.get("From") or payload.get("from") or ""))
    body = str(payload.get("Body") or payload.get("body") or "")[:500]
    # SMS is often used by recipients to clarify before/after a call. Keep the
    # MVP transparent and non-autonomous: acknowledge receipt, do not take action.
    return _twiml(_say(f"Message received from {from_number}. Hermes will pass it to Jason. Message: {body}"))


@app.post("/twilio/voice/{call_id}/hangup")
async def hangup(call_id: str) -> JSONResponse:
    session = await STORE.update_status(call_id, "completed")
    # TODO: call Twilio Calls(call_sid).update(status='completed') when session.twilio_sid exists.
    return JSONResponse({"success": bool(session), "call": session.public_dict() if session else None})


@app.post("/twilio/voice/{call_id}/transfer")
async def transfer(call_id: str) -> JSONResponse:
    session = await STORE.get(call_id)
    if not session:
        return JSONResponse({"success": False, "error": "unknown call_id"}, status_code=404)
    transfer_number = str((session.caller_profile or {}).get("transfer_number") or os.environ.get("VOICE_CALL_TRANSFER_NUMBER", ""))
    if not transfer_number:
        return JSONResponse({"success": False, "error": "transfer target is not configured"}, status_code=400)
    await STORE.append_transcript(call_id, "system", "transfer_to_jason requested", event="transfer")
    session.status = "transfer-requested"
    return JSONResponse({"success": True, "transfer_to": redact_phone(transfer_number), "call": session.public_dict()})


@app.get("/twilio/voice/{call_id}/transcript")
async def transcript(call_id: str) -> JSONResponse:
    session = await STORE.get(call_id)
    if not session:
        return JSONResponse({"success": False, "error": "unknown call_id"}, status_code=404)
    return JSONResponse({"success": True, "call_id": session.call_id, "transcript": session.transcript, "call": session.public_dict()})
