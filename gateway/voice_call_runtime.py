"""Minimal production voice-call runtime for Render Telegram deployments.

The runtime mirrors the Labota/OpenClaw voice-call contract without importing
the whole plugin system:

- Hermes tool calls an internal localhost control API.
- Voximplant is started through Management API ``StartScenarios``.
- VoxEngine sends lifecycle callbacks to ``/voice/webhook``.
- VoxEngine bridges call audio to ``/voice/stream`` over WebSocket.
- A realtime voice provider handles the live speech-to-speech conversation.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import web, WSMsgType

try:
    import jwt
except Exception:  # pragma: no cover - dependency exists in Render all extra
    jwt = None


logger = logging.getLogger(__name__)


CALL_ENDED_EVENTS = {
    "call.ended",
    "call.completed",
    "call.failed",
    "call.error",
    "call.hangup",
    "ended",
    "completed",
    "failed",
    "error",
    "hangup",
    "disconnected",
    "timeout",
    "busy",
    "no_answer",
    "no-answer",
    "cancelled",
    "canceled",
}

CALENDAR_LINK_POSITIVE_RE = re.compile(
    r"(забронировал|забронирован[ао]?|бронь подтвержден[а]?|успешно заброниров|"
    r"ресторан подтвердил|подтвердил.*брон)",
    re.IGNORECASE,
)
CALENDAR_LINK_NEGATIVE_RE = re.compile(
    r"(не удалось|не подтвержден|нет подтверждения|не состоялся|не смог|не дозвони|"
    r"мест нет|к сожалению|отказ|ошибк)",
    re.IGNORECASE,
)
CALENDAR_TEMPLATE_LINK_RE = re.compile(
    r"📅 Добавить в календарь:\s*(https://calendar\.google\.com/calendar/render\?\S+)"
)
WEEKDAY_ALIASES = {
    "понедельник": 0,
    "понедельника": 0,
    "вторник": 1,
    "вторника": 1,
    "среда": 2,
    "среду": 2,
    "среды": 2,
    "четверг": 3,
    "четверга": 3,
    "пятница": 4,
    "пятницу": 4,
    "пятницы": 4,
    "суббота": 5,
    "субботу": 5,
    "субботы": 5,
    "воскресенье": 6,
    "воскресенья": 6,
}


def _sanitize_voice_task(raw: str) -> str:
    """Mirror Labota voice-call task cleanup before injecting it into Realtime."""
    task = raw.strip()
    task = re.sub(r"\s+", " ", task)
    task = re.sub(
        r"^позвонить\s+(?:по номеру\s+)?(?:[\d+()-]+[\s()-]*)+(?:\s*и\s+)?",
        "",
        task,
        flags=re.IGNORECASE,
    )
    task = task.strip()
    if task:
        task = task[0].upper() + task[1:]
    return task[:300]


@dataclass
class CallRecord:
    call_id: str
    to: str
    task: str
    user_id: str = ""
    chat_id: str = ""
    session_key: str = ""
    language: str = "ru"
    from_number: str = ""
    status: str = "created"
    provider_call_id: str = ""
    stream_token: str = ""
    control_url: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    raw_transcript: list[dict[str, Any]] = field(default_factory=list)
    transcript_source: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    summary_sent: bool = False
    summary_suppressed: bool = False

    def touch(self) -> None:
        self.updated_at = time.time()


def _get_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or "/opt/data")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _mask_phone(value: str) -> str:
    clean = str(value or "").strip()
    if len(clean) <= 4:
        return "***"
    return f"***{clean[-4:]}"


def _public_origin() -> str:
    explicit = os.environ.get("VOICE_CALL_PUBLIC_URL", "").strip().rstrip("/")
    if explicit:
        parsed = urlparse(explicit)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    telegram_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    if telegram_url:
        parsed = urlparse(telegram_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    return "https://giga-hermes-staging.onrender.com"


def _ws_origin(origin: str) -> str:
    return origin.replace("https://", "wss://", 1).replace("http://", "ws://", 1)


def _read_private_key() -> str:
    raw = os.environ.get("VOXIMPLANT_MANAGEMENT_PRIVATE_KEY", "").strip()
    if raw:
        return raw
    b64 = os.environ.get("VOXIMPLANT_MANAGEMENT_PRIVATE_KEY_B64", "").strip()
    if not b64:
        return ""
    try:
        return base64.b64decode(b64).decode("utf-8")
    except Exception:
        return ""


def _voice_realtime_provider() -> str:
    raw = os.environ.get("VOICE_CALL_REALTIME_PROVIDER", "openai").strip().lower()
    if raw in {"xai", "x-ai", "x.ai", "grok"}:
        return "xai"
    return "openai"


def _voice_realtime_api_key(provider: str) -> str:
    if provider == "xai":
        return os.environ.get("XAI_API_KEY", "").strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _voice_realtime_model(provider: str) -> str:
    default = "grok-voice-think-fast-1.0" if provider == "xai" else "gpt-realtime-2"
    return os.environ.get("VOICE_CALL_REALTIME_MODEL", default).strip() or default


def _voice_realtime_voice(provider: str) -> str:
    default = "ara" if provider == "xai" else "marin"
    return os.environ.get("VOICE_CALL_ASSISTANT_VOICE", default).strip() or default


def _voice_vad_eagerness() -> str:
    raw = os.environ.get("VOICE_CALL_VAD_EAGERNESS", "high").strip().lower()
    return raw if raw in {"low", "medium", "high", "auto"} else "high"


def _voice_transcription_provider() -> str:
    configured = os.environ.get("VOICE_CALL_TRANSCRIPTION_PROVIDER", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("SBER_SALUTE_AUTH_KEY", "").strip():
        return "sber_salute"
    return "realtime"


def _voice_metric_now() -> float:
    return time.time()


class SaluteSpeechTranscriber:
    """Post-call ASR for Russian phone audio using Sber SaluteSpeech."""

    def __init__(self) -> None:
        self.auth_key = os.environ.get("SBER_SALUTE_AUTH_KEY", "").strip()
        self.scope = os.environ.get("SBER_SALUTE_SCOPE", "SALUTE_SPEECH_CORP").strip()
        self.oauth_url = os.environ.get(
            "SBER_SALUTE_OAUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        ).strip()
        self.api_url = os.environ.get(
            "SBER_SALUTE_RECOGNIZE_URL",
            "https://smartspeech.sber.ru/rest/v1/speech:recognize",
        ).strip()
        self.model = os.environ.get("SBER_SALUTE_RECOGNITION_MODEL", "callcenter").strip()
        self.content_type = os.environ.get(
            "SBER_SALUTE_CONTENT_TYPE", "audio/pcmu;rate=8000"
        ).strip()
        self.user_agent = (
            os.environ.get("SBER_SALUTE_USER_AGENT", "GigaVoice").strip() or "GigaVoice"
        )
        self.insecure = _truthy(os.environ.get("SBER_SALUTE_INSECURE"))
        self.enabled = bool(self.auth_key)
        self._token: str = ""
        self._token_expires_at = 0.0

    async def transcribe_pcmu(self, audio: bytes) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "transcript": "", "error": "SBER_SALUTE_AUTH_KEY not set"}
        if not audio:
            return {"success": False, "transcript": "", "error": "empty audio"}
        try:
            token = await self._access_token()
            timeout = aiohttp.ClientTimeout(total=float(os.environ.get("SBER_SALUTE_TIMEOUT", "60")))
            ssl_arg = False if self.insecure else None
            params: dict[str, str] = {}
            if self.model:
                params["model"] = self.model
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.api_url,
                    params=params,
                    data=audio,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": self.content_type,
                        "Accept": "application/json",
                        "User-Agent": self.user_agent,
                    },
                    ssl=ssl_arg,
                ) as response:
                    text = await response.text()
                    if response.status >= 400:
                        return {
                            "success": False,
                            "transcript": "",
                            "error": f"Sber SaluteSpeech HTTP {response.status}: {text[:300]}",
                        }
            transcript = self._extract_transcript(text)
            if not transcript:
                return {
                    "success": False,
                    "transcript": "",
                    "error": "Sber SaluteSpeech returned empty transcript",
                }
            return {"success": True, "transcript": transcript, "provider": "sber_salute"}
        except Exception as exc:
            logger.exception("Sber SaluteSpeech transcription failed")
            return {
                "success": False,
                "transcript": "",
                "error": f"Sber SaluteSpeech failed: {exc}",
            }

    async def _access_token(self) -> str:
        now = time.time()
        if self._token and self._token_expires_at - 60 > now:
            return self._token
        timeout = aiohttp.ClientTimeout(total=float(os.environ.get("SBER_SALUTE_TIMEOUT", "60")))
        ssl_arg = False if self.insecure else None
        scopes = []
        for scope in (self.scope, "SALUTE_SPEECH_CORP", "SALUTE_SPEECH_PERS"):
            if scope and scope not in scopes:
                scopes.append(scope)
        last_error = ""
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload_text = ""
            for scope in scopes:
                async with session.post(
                    self.oauth_url,
                    data={"scope": scope},
                    headers={
                        "Authorization": f"Basic {self.auth_key}",
                        "RqUID": str(uuid.uuid4()),
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "User-Agent": self.user_agent,
                    },
                    ssl=ssl_arg,
                ) as response:
                    payload_text = await response.text()
                    if response.status < 400:
                        self.scope = scope
                        break
                    last_error = f"Sber OAuth HTTP {response.status}: {payload_text[:300]}"
            else:
                raise RuntimeError(last_error or "Sber OAuth failed")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Sber OAuth returned non-JSON response: {payload_text[:120]}") from exc
        token, expires_at_float = self._parse_oauth_payload(payload, now)
        self._token = token
        self._token_expires_at = expires_at_float
        return token

    @staticmethod
    def _parse_oauth_payload(payload: dict[str, Any], now: float) -> tuple[str, float]:
        token = str(
            payload.get("access_token")
            or payload.get("token")
            or payload.get("tok")
            or ""
        ).strip()
        if not token:
            raise RuntimeError("Sber OAuth response has no access_token")
        expires_at = (
            payload.get("expires_at")
            or payload.get("expiresAt")
            or payload.get("exp")
            or 0
        )
        try:
            expires_at_float = float(expires_at)
            if expires_at_float > 10_000_000_000:
                expires_at_float /= 1000.0
        except (TypeError, ValueError):
            expires_at_float = now + 25 * 60
        if expires_at_float <= now:
            expires_at_float = now + 25 * 60
        return token, expires_at_float

    @staticmethod
    def _extract_transcript(payload_text: str) -> str:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return payload_text.strip()
        parts: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(SaluteSpeechTranscriber._extract_from_dict(item))
        elif isinstance(payload, dict):
            parts.append(SaluteSpeechTranscriber._extract_from_dict(payload))
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()

    @staticmethod
    def _extract_from_dict(payload: dict[str, Any]) -> str:
        for key in ("text", "transcript", "normalized_text", "normalizedText"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        result = payload.get("result") or payload.get("results")
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(SaluteSpeechTranscriber._extract_from_dict(item))
            return " ".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""


class VoiceCallRuntime:
    def __init__(self) -> None:
        self.calls: dict[str, CallRecord] = {}
        self.calls_by_provider: dict[str, str] = {}
        self.calls_by_token: dict[str, str] = {}
        self.storage_dir = _get_hermes_home() / "voice-calls"
        self.public_origin = _public_origin()
        self.webhook_path = os.environ.get("VOICE_CALL_WEBHOOK_PATH", "/voice/webhook")
        self.stream_path = os.environ.get("VOICE_CALL_STREAM_PATH", "/voice/stream")
        self._management_jwt_cache: tuple[str, int] | None = None

    def health(self) -> dict[str, Any]:
        required = self.missing_requirements()
        realtime_provider = _voice_realtime_provider()
        return {
            "enabled": not required,
            "missing": required,
            "realtime_provider": realtime_provider,
            "realtime_model": _voice_realtime_model(realtime_provider),
            "vad_eagerness": _voice_vad_eagerness(),
            "transcription_provider": _voice_transcription_provider(),
            "public_origin": self.public_origin,
            "webhook_path": self.webhook_path,
            "stream_path": self.stream_path,
            "active_calls": len([c for c in self.calls.values() if not c.ended_at]),
            "warnings": self.config_warnings(),
        }

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        provider = _voice_realtime_provider()
        if not _voice_realtime_api_key(provider):
            missing.append("XAI_API_KEY" if provider == "xai" else "OPENAI_API_KEY")
        if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
            missing.append("TELEGRAM_BOT_TOKEN")
        if not os.environ.get("VOXIMPLANT_RULE_ID", "").strip():
            missing.append("VOXIMPLANT_RULE_ID")
        if not os.environ.get("VOXIMPLANT_WEBHOOK_SECRET", "").strip() and not _truthy(
            os.environ.get("VOICE_CALL_SKIP_WEBHOOK_VERIFICATION")
        ):
            missing.append("VOXIMPLANT_WEBHOOK_SECRET")
        has_static_jwt = bool(os.environ.get("VOXIMPLANT_MANAGEMENT_JWT", "").strip())
        has_service_account = all(
            [
                os.environ.get("VOXIMPLANT_MANAGEMENT_ACCOUNT_ID", "").strip(),
                os.environ.get("VOXIMPLANT_MANAGEMENT_KEY_ID", "").strip(),
                _read_private_key(),
            ]
        )
        if not has_static_jwt and not has_service_account:
            missing.append("VOXIMPLANT_MANAGEMENT_JWT or service account fields")
        return missing

    def config_warnings(self) -> list[str]:
        warnings: list[str] = []
        if not os.environ.get("VOICE_CALL_FROM_NUMBER", "").strip():
            warnings.append(
                "VOICE_CALL_FROM_NUMBER is not set; Voximplant scenario must provide a caller ID fallback."
            )
        return warnings

    # ------------------------------------------------------------------
    # Internal control API used by the Hermes tool
    # ------------------------------------------------------------------

    async def handle_control_initiate(self, request: web.Request) -> web.Response:
        data = await request.json()
        task = str(data.get("task") or data.get("prompt") or "").strip()
        to = str(data.get("to") or "").strip()
        if not to:
            return web.json_response({"success": False, "error": "to required"}, status=400)
        if not task:
            return web.json_response({"success": False, "error": "task required"}, status=400)
        missing = self.missing_requirements()
        if missing:
            return web.json_response(
                {"success": False, "error": "voice_call not configured", "missing": missing},
                status=503,
            )
        from_number = str(data.get("from") or os.environ.get("VOICE_CALL_FROM_NUMBER", "")).strip()
        if not from_number:
            return web.json_response(
                {
                    "success": False,
                    "error": "voice_call not configured",
                    "missing": ["VOICE_CALL_FROM_NUMBER"],
                    "detail": "Outbound Voximplant calls require a configured caller ID.",
                },
                status=503,
            )

        call = CallRecord(
            call_id=f"call_{uuid.uuid4().hex[:16]}",
            to=to,
            task=task,
            user_id=str(data.get("user_id") or ""),
            chat_id=str(data.get("chat_id") or ""),
            session_key=str(data.get("session_key") or ""),
            language=str(data.get("language") or "ru"),
            from_number=from_number,
            stream_token=uuid.uuid4().hex,
        )
        self.calls[call.call_id] = call
        self.calls_by_token[call.stream_token] = call.call_id
        self._persist(call)
        logger.info(
            "voice_call initiate call_id=%s to=%s from_set=%s user_id=%s",
            call.call_id,
            _mask_phone(call.to),
            bool(call.from_number),
            call.user_id,
        )

        try:
            await self._start_voximplant_call(call)
        except Exception as exc:
            call.status = "failed"
            call.error = str(exc)
            call.touch()
            self._persist(call)
            logger.exception("voice_call start failed call_id=%s", call.call_id)
            return web.json_response({"success": False, "error": str(exc)}, status=502)

        return web.json_response(
            {
                "success": True,
                "callId": call.call_id,
                "providerCallId": call.provider_call_id,
                "status": call.status,
            }
        )

    async def handle_control_status(self, request: web.Request) -> web.Response:
        call = self._resolve_call(request.match_info.get("call_id", ""))
        if not call:
            return web.json_response({"found": False}, status=404)
        return web.json_response({"found": True, "call": self._safe_call_dict(call)})

    async def handle_control_history(self, request: web.Request) -> web.Response:
        session_key = request.query.get("session_key", "")
        user_id = request.query.get("user_id", "")
        calls_by_id = {call.call_id: call for call in self._load_persisted_calls()}
        calls_by_id.update(self.calls)
        calls = list(calls_by_id.values())
        if session_key:
            calls = [call for call in calls if call.session_key == session_key]
        elif user_id:
            calls = [call for call in calls if call.user_id == user_id]
        calls.sort(key=lambda call: call.created_at, reverse=True)
        return web.json_response({"calls": [self._safe_call_dict(call) for call in calls[:20]]})

    async def handle_control_end(self, request: web.Request) -> web.Response:
        call = self._resolve_call(request.match_info.get("call_id", ""))
        if not call:
            return web.json_response({"success": False, "error": "call not found"}, status=404)
        if call.control_url:
            await self._post_control(call, {"action": "hangup", "reason": "user_requested"})
        await self._finalize_call(call, "ended")
        return web.json_response({"success": True})

    # ------------------------------------------------------------------
    # Public Voximplant routes
    # ------------------------------------------------------------------

    async def handle_webhook(self, request: web.Request) -> web.Response:
        raw = await request.text()
        payload: dict[str, Any]
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if not self._verify_webhook(request, payload):
            return web.Response(status=401, text="Unauthorized")

        call = self._call_from_payload(payload, request)
        if not call:
            logger.info(
                "voice_call webhook ignored event=%s payload_keys=%s",
                self._event_type(payload),
                sorted(payload.keys()),
            )
            return web.json_response({"ok": True, "ignored": True})

        event_type = self._event_type(payload)
        provider_call_id = self._pick(payload, "providerCallId", "call_session_history_id", "session_id")
        if provider_call_id:
            call.provider_call_id = provider_call_id
            self.calls_by_provider[provider_call_id] = call.call_id

        control_url = self._pick(
            payload,
            "media_session_access_secure_url",
            "mediaSessionAccessSecureUrl",
            "control_url",
            "controlUrl",
        )
        if control_url:
            call.control_url = control_url

        transcript = self._pick(payload, "transcript", "speech", "text")
        if transcript and event_type in {"call.speech", "speech", "transcript"}:
            self._add_transcript(call, "callee", transcript, source="voximplant_webhook")

        call.status = event_type or call.status
        call.touch()
        self._persist(call)
        logger.info(
            "voice_call webhook call_id=%s provider_call_id=%s event=%s control_url_set=%s transcript_items=%s",
            call.call_id,
            call.provider_call_id,
            event_type or "unknown",
            bool(call.control_url),
            len(call.transcript),
        )

        if event_type in CALL_ENDED_EVENTS or "end" in event_type or "hangup" in event_type:
            asyncio.create_task(self._finalize_call(call, event_type or "ended"))

        return web.json_response({"ok": True})

    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        token = request.query.get("token", "")
        call_id = self.calls_by_token.get(token)
        call = self.calls.get(call_id or "")
        if not call:
            logger.warning("voice_call stream rejected unknown token")
            return web.Response(status=401, text="Unknown stream token")

        vox_ws = web.WebSocketResponse(heartbeat=25)
        await vox_ws.prepare(request)
        self._mark_metric(call, "stream_opened_at")

        realtime_provider = _voice_realtime_provider()
        realtime = RealtimeVoiceBridge(
            provider=realtime_provider,
            api_key=_voice_realtime_api_key(realtime_provider),
            model=_voice_realtime_model(realtime_provider),
            voice=_voice_realtime_voice(realtime_provider),
            instructions=self._build_voice_instructions(call),
            language=call.language or "ru",
            call=call,
            on_assistant_audio=lambda audio: self._send_audio_to_vox(vox_ws, call, audio),
            on_transcript=lambda role, text: self._add_transcript(
                call, role, text, source=f"{realtime_provider}_realtime"
            ),
        )

        stream_sid = f"vox-{call.call_id}"
        seq = 0
        realtime_ready = asyncio.Event()
        pending_audio: list[bytes] = []
        pending_audio_bytes = 0
        vox_started = asyncio.Event()
        initial_response_sent = False
        try:
            max_pending_audio_bytes = int(
                os.environ.get("VOICE_CALL_REALTIME_AUDIO_BUFFER_BYTES", "80000")
            )
        except ValueError:
            max_pending_audio_bytes = 80000

        async def maybe_start_initial_response() -> None:
            nonlocal initial_response_sent
            if initial_response_sent:
                return
            if not realtime_ready.is_set() or not vox_started.is_set():
                return
            if not _truthy(os.environ.get("VOICE_CALL_INITIAL_RESPONSE", "0")):
                return
            initial_response_sent = True
            await realtime.create_initial_response()
            logger.info("voice_call initial response requested call_id=%s", call.call_id)

        async def connect_realtime() -> None:
            nonlocal pending_audio, pending_audio_bytes
            self._mark_metric(call, "realtime_connect_started_at")
            try:
                await realtime.connect()
            except Exception as exc:
                call.error = f"realtime_connect_failed: {exc}"
                call.status = "stream_error"
                call.touch()
                self._persist(call)
                logger.exception("voice_call realtime connect failed call_id=%s", call.call_id)
                return
            self._mark_metric(call, "realtime_connected_at")
            realtime_ready.set()
            buffered = pending_audio
            pending_audio = []
            pending_audio_bytes = 0
            for chunk in buffered:
                await realtime.send_audio(chunk)
            await maybe_start_initial_response()
            logger.info(
                "voice_call realtime connected call_id=%s provider=%s model=%s startup_ms=%.0f buffered_bytes=%s",
                call.call_id,
                realtime_provider,
                realtime.model,
                self._elapsed_ms(call, "stream_opened_at", "realtime_connected_at"),
                sum(len(chunk) for chunk in buffered),
            )

        async def forward_audio(audio: bytes) -> None:
            nonlocal pending_audio_bytes
            if not audio:
                return
            self._record_callee_audio(call, audio)
            if "first_inbound_audio_at" not in call.metrics:
                self._mark_metric(call, "first_inbound_audio_at")
            if realtime_ready.is_set():
                await realtime.send_audio(audio)
                return
            if max_pending_audio_bytes <= 0:
                return
            pending_audio.append(audio)
            pending_audio_bytes += len(audio)
            while pending_audio_bytes > max_pending_audio_bytes and pending_audio:
                dropped = pending_audio.pop(0)
                pending_audio_bytes -= len(dropped)

        connect_task = asyncio.create_task(connect_realtime())
        try:
            async for msg in vox_ws:
                if msg.type == WSMsgType.BINARY:
                    await forward_audio(msg.data)
                    continue
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                event_name = str(event.get("event") or "").lower()
                if event_name == "start":
                    stream_sid = (
                        event.get("streamSid")
                        or (event.get("start") or {}).get("streamSid")
                        or stream_sid
                    )
                    setattr(call, "_stream_sid", stream_sid)
                    call.status = "stream_connected"
                    self._mark_metric(call, "vox_start_received_at")
                    vox_started.set()
                    call.touch()
                    await vox_ws.send_json(
                        {
                            "event": "start",
                            "sequenceNumber": 0,
                            "start": {
                                "mediaFormat": {
                                    "encoding": "ULAW",
                                    "sampleRate": 8000,
                                    "channels": 1,
                                },
                                "customParameters": {},
                            },
                            "streamSid": stream_sid,
                        }
                    )
                    self._mark_metric(call, "vox_start_sent_at")
                    logger.info(
                        "voice_call vox stream start call_id=%s ack_ms=%.0f realtime_ready=%s",
                        call.call_id,
                        self._elapsed_ms(call, "vox_start_received_at", "vox_start_sent_at"),
                        realtime_ready.is_set(),
                    )
                    await maybe_start_initial_response()
                    continue
                if event_name == "media":
                    payload = ((event.get("media") or {}).get("payload") or "").strip()
                    if payload:
                        try:
                            await forward_audio(base64.b64decode(payload))
                        except Exception:
                            pass
                    continue
                if event_name == "stop":
                    break
                seq += 1
        finally:
            if not connect_task.done():
                connect_task.cancel()
                try:
                    await connect_task
                except asyncio.CancelledError:
                    pass
            else:
                try:
                    exc = connect_task.exception()
                except asyncio.CancelledError:
                    exc = None
                if exc:
                    logger.warning(
                        "voice_call realtime task ended with error call_id=%s error=%s",
                        call.call_id,
                        exc,
                    )
            await realtime.close()
            call.touch()
            self._persist(call)
            logger.info("voice_call stream closed call_id=%s", call.call_id)
        return vox_ws

    async def _send_audio_to_vox(
        self, vox_ws: web.WebSocketResponse, call: CallRecord, audio: bytes
    ) -> None:
        if vox_ws.closed:
            return
        if "first_assistant_audio_at" not in call.metrics:
            self._mark_metric(call, "first_assistant_audio_at")
        stream_sid = getattr(call, "_stream_sid", f"vox-{call.call_id}")
        for offset in range(0, len(audio), 160):
            chunk = audio[offset : offset + 160]
            if not chunk:
                continue
            count = getattr(call, "_audio_seq", 0) + 1
            setattr(call, "_audio_seq", count)
            timestamp_ms = (count - 1) * 20
            await vox_ws.send_json(
                {
                    "event": "media",
                    "sequenceNumber": count,
                    "streamSid": stream_sid,
                    "media": {
                        "chunk": count,
                        "timestamp": timestamp_ms,
                        "payload": base64.b64encode(chunk).decode("ascii"),
                    },
                }
            )

    # ------------------------------------------------------------------
    # Metrics and audio capture
    # ------------------------------------------------------------------

    def _mark_metric(self, call: CallRecord, name: str) -> None:
        call.metrics[name] = _voice_metric_now()
        call.touch()
        self._persist(call)

    def _elapsed_ms(self, call: CallRecord, start_name: str, end_name: str) -> float:
        start = call.metrics.get(start_name)
        end = call.metrics.get(end_name)
        if not start or not end:
            return 0.0
        return max((end - start) * 1000, 0.0)

    def _record_callee_audio(self, call: CallRecord, audio: bytes) -> None:
        chunks: list[bytes] = getattr(call, "_callee_audio_chunks", [])
        total = int(getattr(call, "_callee_audio_bytes", 0))
        try:
            max_seconds = int(os.environ.get("VOICE_CALL_TRANSCRIPTION_MAX_SECONDS", "240"))
        except ValueError:
            max_seconds = 240
        max_bytes = max(max_seconds, 15) * 8000
        chunks.append(audio)
        total += len(audio)
        while total > max_bytes and chunks:
            dropped = chunks.pop(0)
            total -= len(dropped)
        setattr(call, "_callee_audio_chunks", chunks)
        setattr(call, "_callee_audio_bytes", total)

    def _callee_audio_bytes(self, call: CallRecord) -> bytes:
        chunks = getattr(call, "_callee_audio_chunks", [])
        if not chunks:
            return b""
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Voximplant management
    # ------------------------------------------------------------------

    async def _start_voximplant_call(self, call: CallRecord) -> None:
        webhook_url = f"{self.public_origin}{self.webhook_path}?callId={call.call_id}"
        stream_url = f"{_ws_origin(self.public_origin)}{self.stream_path}?token={call.stream_token}"
        script_custom_data = {
            "provider": "voximplant",
            "callId": call.call_id,
            "from": call.from_number,
            "to": call.to,
            "webhookUrl": webhook_url,
            "webhookSecret": os.environ.get("VOXIMPLANT_WEBHOOK_SECRET", ""),
            "streamUrl": stream_url,
            "clientState": {
                "telegramChatId": call.chat_id,
                "telegramUserId": call.user_id,
                "sessionKey": call.session_key,
            },
        }
        logger.info(
            "voice_call StartScenarios call_id=%s rule_id=%s to=%s from_set=%s",
            call.call_id,
            os.environ["VOXIMPLANT_RULE_ID"],
            _mask_phone(call.to),
            bool(call.from_number),
        )
        response = await self._voximplant_api(
            "StartScenarios",
            {
                "rule_id": os.environ["VOXIMPLANT_RULE_ID"],
                "script_custom_data": json.dumps(script_custom_data, ensure_ascii=False),
            },
        )
        call.provider_call_id = self._pick(
            response,
            "call_session_history_id",
            "callSessionHistoryId",
            "session_id",
            "sessionId",
            "request_id",
            "requestId",
            "call_id",
            "callId",
        ) or call.call_id
        call.control_url = self._pick(
            response,
            "media_session_access_secure_url",
            "mediaSessionAccessSecureUrl",
            "control_url",
            "controlUrl",
        )
        call.status = "initiated"
        self.calls_by_provider[call.provider_call_id] = call.call_id
        call.touch()
        self._persist(call)
        logger.info(
            "voice_call initiated call_id=%s provider_call_id=%s control_url_set=%s",
            call.call_id,
            call.provider_call_id,
            bool(call.control_url),
        )

    async def _voximplant_api(self, action: str, params: dict[str, str]) -> dict[str, Any]:
        token = await self._management_jwt()
        base_url = os.environ.get("VOXIMPLANT_API_BASE_URL", "https://api.voximplant.com/platform_api")
        url = f"{base_url.rstrip('/')}/{action}/"
        async with aiohttp.ClientSession() as session:
            response, text, payload = await self._post_voximplant(session, url, params, token)
            if response == 401 and self._can_generate_management_jwt():
                token = await self._management_jwt(force_service_account=True)
                response, text, payload = await self._post_voximplant(session, url, params, token)
            if response >= 400:
                raise RuntimeError(f"Voximplant API error {response}: {text[:300]}")
            if isinstance(payload, dict):
                error = self._pick(payload, "error", "error_msg", "errorMsg")
                if error:
                    raise RuntimeError(f"Voximplant API error: {error}")
                if payload.get("result") == 0:
                    raise RuntimeError("Voximplant API returned unsuccessful result")
                return payload
            return {}

    async def _post_voximplant(
        self, session: aiohttp.ClientSession, url: str, params: dict[str, str], token: str
    ) -> tuple[int, str, dict[str, Any]]:
        async with session.post(
            url,
            data=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            text = await response.text()
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {}
            return response.status, text, payload if isinstance(payload, dict) else {}

    def _can_generate_management_jwt(self) -> bool:
        return bool(
            os.environ.get("VOXIMPLANT_MANAGEMENT_ACCOUNT_ID", "").strip()
            and os.environ.get("VOXIMPLANT_MANAGEMENT_KEY_ID", "").strip()
            and _read_private_key()
        )

    async def _management_jwt(self, *, force_service_account: bool = False) -> str:
        static = os.environ.get("VOXIMPLANT_MANAGEMENT_JWT", "").strip()
        if (
            not force_service_account
            and static
            and static not in {"AUTO", "__AUTO__", "__SERVICE_ACCOUNT__"}
        ):
            return static
        if self._management_jwt_cache and self._management_jwt_cache[1] - 60 > int(time.time()):
            return self._management_jwt_cache[0]
        if jwt is None:
            raise RuntimeError("PyJWT is required for Voximplant service-account JWT generation")
        now = int(time.time())
        exp = now + 3600
        token = jwt.encode(
            {"iat": now, "iss": os.environ["VOXIMPLANT_MANAGEMENT_ACCOUNT_ID"], "exp": exp},
            _read_private_key(),
            algorithm="RS256",
            headers={"kid": os.environ["VOXIMPLANT_MANAGEMENT_KEY_ID"]},
        )
        self._management_jwt_cache = (token, exp)
        return token

    async def _post_control(self, call: CallRecord, payload: dict[str, Any]) -> None:
        async with aiohttp.ClientSession() as session:
            await session.post(
                call.control_url,
                json={"provider": "voximplant", "callId": call.call_id, **payload},
                timeout=aiohttp.ClientTimeout(total=10),
            )

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    async def _finalize_call(self, call: CallRecord, reason: str) -> None:
        if call.summary_sent or call.summary_suppressed:
            return
        call.status = reason
        call.ended_at = call.ended_at or time.time()
        call.touch()
        self._persist(call)
        if not self._should_send_completion_summary(call):
            call.summary_suppressed = True
            call.touch()
            self._persist(call)
            logger.info(
                "voice_call async summary suppressed call_id=%s status=%s transcript_items=%s elapsed=%.1fs",
                call.call_id,
                call.status,
                len(call.transcript),
                (call.ended_at or time.time()) - call.created_at,
            )
            return
        await self._maybe_apply_salute_transcript(call)
        text = await self._build_summary(call)
        await self._send_telegram(call.chat_id, text)
        call.summary_sent = True
        call.touch()
        self._persist(call)

    def _should_send_completion_summary(self, call: CallRecord) -> bool:
        if not call.chat_id:
            return False
        if _truthy(os.environ.get("VOICE_CALL_DISABLE_ASYNC_SUMMARY")):
            return False
        if _truthy(os.environ.get("VOICE_CALL_NOTIFY_FAST_FAILURES")):
            return True
        status = str(call.status or "").lower()
        is_failure = any(
            marker in status
            for marker in ("error", "fail", "busy", "no_answer", "no-answer", "timeout", "cancel")
        )
        if not is_failure or call.transcript:
            return True
        try:
            suppress_seconds = float(os.environ.get("VOICE_CALL_SUPPRESS_FAST_FAILURE_SECONDS", "30"))
        except ValueError:
            suppress_seconds = 30.0
        elapsed = (call.ended_at or time.time()) - call.created_at
        return elapsed > suppress_seconds

    async def _maybe_apply_salute_transcript(self, call: CallRecord) -> None:
        if _voice_transcription_provider() not in {"sber", "sber_salute", "salute"}:
            return
        if call.transcript_source == "sber_salute":
            return
        audio = self._callee_audio_bytes(call)
        if not audio:
            logger.info("voice_call salute transcript skipped call_id=%s reason=no_audio", call.call_id)
            return
        transcriber = SaluteSpeechTranscriber()
        if not transcriber.enabled:
            logger.info("voice_call salute transcript skipped call_id=%s reason=not_configured", call.call_id)
            return
        result = await transcriber.transcribe_pcmu(audio)
        if not result.get("success"):
            logger.warning(
                "voice_call salute transcript failed call_id=%s error=%s",
                call.call_id,
                result.get("error"),
            )
            return
        text = str(result.get("transcript") or "").strip()
        if not text:
            return
        call.transcript = [
            item
            for item in call.transcript
            if item.get("role") != "callee" or item.get("source") == "sber_salute"
        ]
        self._add_transcript(call, "callee", text, source="sber_salute")
        call.transcript_source = "sber_salute"
        call.touch()
        self._persist(call)
        logger.info(
            "voice_call salute transcript applied call_id=%s chars=%s audio_bytes=%s",
            call.call_id,
            len(text),
            len(audio),
        )

    async def _build_summary(self, call: CallRecord) -> str:
        if call.transcript_source == "sber_salute":
            callee_text = " ".join(
                str(item.get("text") or "").strip()
                for item in call.transcript
                if item.get("role") == "callee" and item.get("source") == "sber_salute"
            ).strip()
            assistant_text = " ".join(
                str(item.get("text") or "").strip()
                for item in call.transcript[-80:]
                if item.get("role") == "assistant"
            ).strip()
            transcript = "\n".join(
                part
                for part in (
                    f"Собеседник (Sber SaluteSpeech): {callee_text}" if callee_text else "",
                    f"Ассистент: {assistant_text}" if assistant_text else "",
                )
                if part
            ).strip()
        else:
            transcript = "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('text', '')}"
                for item in call.transcript[-80:]
            ).strip()
        if not transcript:
            if "error" in call.status or "fail" in call.status:
                return (
                    "Звонок не состоялся: соединение завершилось ошибкой на стороне телефонии. "
                    "Бронь не подтверждена."
                )
            return (
                "Звонок завершён, но подробная расшифровка не пришла от голосового канала. "
                "Если нужно, могу попробовать ещё раз."
            )
        prompt = (
            "Составь отчёт для клиента Telegram после телефонного звонка консьержа.\n"
            "Верни только готовое сообщение клиенту, без служебных статусов и markdown-заголовков.\n"
            "Пиши по-русски, уважительно, профессионально, без первого лица множественного числа. "
            "Не пиши 'мы'. Используй стиль: 'уточнил', 'забронировал вам', 'ресторан сообщил'.\n"
            "Нельзя писать, что бронь подтверждена, если в расшифровке нет явного подтверждения "
            "от сотрудника. Если результат неясен, честно скажи, что подтверждения нет.\n"
            "Формат: сначала итог 1-2 предложениями, затем при наличии данных блок "
            "'Подробности:' со строками через длинное тире: Дата, Время, Количество персон, "
            "Имя брони, Условия. Если цель не достигнута, добавь короткий следующий шаг.\n"
            f"Задача: {call.task}\n\nРасшифровка:\n{transcript}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={
                        "model": os.environ.get("VOICE_CALL_SUMMARY_MODEL", "gpt-5.4"),
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Ты профессиональный консьерж. Формируешь только "
                                    "клиентский отчёт после звонка, без внутренних деталей."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=aiohttp.ClientTimeout(total=45),
                ) as response:
                    payload = await response.json()
                    content = (
                        payload.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if content:
                        return self._append_calendar_template_link(content, call)
        except Exception:
            pass
        return f"Звонок завершён. Краткая расшифровка:\n{transcript[:2500]}"

    async def _send_telegram(self, chat_id: str, text: str) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return
        message_text, entities = self._telegram_calendar_link_entities(text)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": message_text[:3900],
            "disable_web_page_preview": True,
        }
        if entities:
            payload["entities"] = entities
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20),
            )

    def _persist(self, call: CallRecord) -> None:
        try:
            user_dir = self.storage_dir / (call.user_id or "unknown")
            user_dir.mkdir(parents=True, exist_ok=True)
            path = user_dir / f"{call.call_id}.json"
            data = self._safe_call_dict(call, include_transcript=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_persisted_calls(self) -> list[CallRecord]:
        if not self.storage_dir.exists():
            return []
        calls: list[CallRecord] = []
        for path in self.storage_dir.glob("*/*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data.pop("stream_token", None)
                call_fields = {field.name for field in fields(CallRecord)}
                call = CallRecord(**{key: value for key, value in data.items() if key in call_fields})
            except Exception:
                continue
            calls.append(call)
        return calls

    def _add_transcript(
        self, call: CallRecord, role: str, text: str, *, source: str = "unknown"
    ) -> None:
        raw = text.strip()
        if not raw:
            return
        call.raw_transcript.append(
            {"role": role, "text": raw, "source": source, "timestamp": time.time()}
        )
        clean = self._clean_transcript_text(raw)
        if not clean:
            call.touch()
            self._persist(call)
            return
        if call.transcript and call.transcript[-1].get("role") == role:
            previous = str(call.transcript[-1].get("text") or "").strip().lower()
            if previous == clean.lower():
                call.touch()
                self._persist(call)
                return
        if role == "callee" and "first_callee_transcript_at" not in call.metrics:
            self._mark_metric(call, "first_callee_transcript_at")
        if role == "assistant" and "first_assistant_transcript_at" not in call.metrics:
            self._mark_metric(call, "first_assistant_transcript_at")
        call.transcript.append(
            {"role": role, "text": clean, "source": source, "timestamp": time.time()}
        )
        if source:
            call.transcript_source = source
        call.touch()
        self._persist(call)

    @staticmethod
    def _clean_transcript_text(text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ""
        if re.fullmatch(r"[\W_]+", clean, flags=re.UNICODE):
            return ""
        if len(clean) <= 3 and not re.search(r"[A-Za-zА-Яа-яЁё0-9]", clean):
            return ""
        junk = clean.lower()
        known_junk = {
            "😎",
            "🤖",
            "♪",
            "♫",
            "music",
            "субтитры сделал dima torzok",
            "спасибо за просмотр",
        }
        if junk in known_junk:
            return ""
        return clean

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_calendar_template_link(self, text: str, call: CallRecord) -> str:
        if "calendar.google.com/calendar/render" in text:
            return text
        if not CALENDAR_LINK_POSITIVE_RE.search(text) or CALENDAR_LINK_NEGATIVE_RE.search(text):
            return text
        event_start = self._infer_calendar_start(text, call)
        if not event_start:
            return text
        try:
            duration_min = int(os.environ.get("VOICE_CALL_CALENDAR_LINK_DURATION_MIN", "120"))
        except ValueError:
            duration_min = 120
        event_end = event_start + timedelta(minutes=max(duration_min, 15))
        title = self._calendar_event_title(call)
        location = self._calendar_event_location(call)
        link = self._google_calendar_template_url(
            title=title,
            start=event_start,
            end=event_end,
            details=text.strip(),
            location=location,
        )
        return f"{text.rstrip()}\n\n📅 Добавить в календарь: {link}"

    @staticmethod
    def _telegram_utf16_len(text: str) -> int:
        return len(text.encode("utf-16-le")) // 2

    @classmethod
    def _telegram_calendar_link_entities(cls, text: str) -> tuple[str, list[dict[str, Any]]]:
        match = CALENDAR_TEMPLATE_LINK_RE.search(text)
        if not match:
            return text, []
        label = "📅 Добавить в календарь"
        visible_text = f"{text[:match.start()]}{label}{text[match.end():]}"
        offset = cls._telegram_utf16_len(visible_text[: match.start()])
        length = cls._telegram_utf16_len(label)
        if cls._telegram_utf16_len(visible_text[:3900]) < offset + length:
            return visible_text, []
        return visible_text, [
            {"type": "text_link", "offset": offset, "length": length, "url": match.group(1)}
        ]

    def _infer_calendar_start(self, text: str, call: CallRecord) -> datetime | None:
        tz_name = os.environ.get("TZ") or "Europe/Moscow"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("Europe/Moscow")
        combined = f"{text}\n{call.task}"
        base = datetime.fromtimestamp(call.created_at, tz)
        date_value = self._infer_calendar_date(combined, base)
        time_value = self._infer_calendar_time(combined)
        if not date_value or not time_value:
            return None
        return datetime.combine(date_value, time_value, tzinfo=tz)

    @staticmethod
    def _infer_calendar_date(text: str, base: datetime):
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if iso_match:
            try:
                return datetime(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3)),
                ).date()
            except ValueError:
                return None
        dotted_match = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b", text)
        if dotted_match:
            day = int(dotted_match.group(1))
            month = int(dotted_match.group(2))
            year = int(dotted_match.group(3) or base.year)
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day).date()
            except ValueError:
                return None
        lowered = text.lower()
        if "послезавтра" in lowered:
            return (base + timedelta(days=2)).date()
        if "завтра" in lowered:
            return (base + timedelta(days=1)).date()
        if "сегодня" in lowered:
            return base.date()
        for word, weekday in WEEKDAY_ALIASES.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                days_ahead = (weekday - base.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (base + timedelta(days=days_ahead)).date()
        return None

    @staticmethod
    def _infer_calendar_time(text: str):
        detail_match = re.search(
            r"(?:Время|время)\s*[:—-]\s*(?:после\s*)?([01]?\d|2[0-3])[:.]([0-5]\d)",
            text,
        )
        if detail_match:
            return datetime.min.time().replace(
                hour=int(detail_match.group(1)),
                minute=int(detail_match.group(2)),
            )
        time_match = re.search(r"\b(?:в|на|после)\s+([01]?\d|2[0-3])[:.]([0-5]\d)\b", text, re.IGNORECASE)
        if time_match:
            return datetime.min.time().replace(hour=int(time_match.group(1)), minute=int(time_match.group(2)))
        hour_match = re.search(r"\b(?:в|на|после)\s+([01]?\d|2[0-3])\b", text, re.IGNORECASE)
        if hour_match:
            return datetime.min.time().replace(hour=int(hour_match.group(1)), minute=0)
        return None

    @staticmethod
    def _calendar_event_title(call: CallRecord) -> str:
        location = VoiceCallRuntime._calendar_event_location(call)
        return f"Бронь: {location}" if location else "Бронирование"

    @staticmethod
    def _calendar_event_location(call: CallRecord) -> str:
        match = re.search(
            r"\b(?:в|во)\s+(?:ресторане?\s+|кафе\s+|баре\s+)?"
            r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9 '&«»\".-]{1,60})"
            r"(?=\s+(?:на|завтра|сегодня|послезавтра|в\s+\d|после|для|по номеру)|[,.]|$)",
            call.task,
            re.IGNORECASE,
        )
        if not match:
            return ""
        candidate = match.group(1).strip(" «»\"'")
        if re.search(r"\b(завтра|сегодня|послезавтра|имя|человек|персон)\b", candidate, re.IGNORECASE):
            return ""
        return candidate[:80]

    @staticmethod
    def _google_calendar_template_url(
        *,
        title: str,
        start: datetime,
        end: datetime,
        details: str,
        location: str,
    ) -> str:
        tz_name = start.tzinfo.key if isinstance(start.tzinfo, ZoneInfo) else (os.environ.get("TZ") or "Europe/Moscow")
        params = {
            "action": "TEMPLATE",
            "text": title,
            "dates": f"{start.strftime('%Y%m%dT%H%M%S')}/{end.strftime('%Y%m%dT%H%M%S')}",
            "ctz": tz_name,
            "details": f"Добавлено из Гига Помощника.\n\n{details}",
        }
        if location:
            params["location"] = location
        return f"https://calendar.google.com/calendar/render?{urlencode(params)}"

    def _resolve_call(self, call_id: str) -> CallRecord | None:
        call = self.calls.get(call_id) or self.calls.get(self.calls_by_provider.get(call_id, ""))
        if call:
            return call
        for persisted in self._load_persisted_calls():
            if persisted.call_id == call_id or persisted.provider_call_id == call_id:
                return persisted
        return None

    def _call_from_payload(self, payload: dict[str, Any], request: web.Request) -> CallRecord | None:
        call_id = self._pick(payload, "callId") or request.query.get("callId", "")
        provider_id = self._pick(payload, "providerCallId", "call_session_history_id", "session_id")
        return self._resolve_call(call_id) or self._resolve_call(provider_id or "")

    def _verify_webhook(self, request: web.Request, payload: dict[str, Any]) -> bool:
        if _truthy(os.environ.get("VOICE_CALL_SKIP_WEBHOOK_VERIFICATION")):
            return True
        expected = os.environ.get("VOXIMPLANT_WEBHOOK_SECRET", "").strip()
        if not expected:
            return False
        provided = (
            request.headers.get("x-openclaw-voximplant-secret")
            or request.headers.get("x-voximplant-secret")
            or request.query.get("secret")
            or self._pick(payload, "secret", "webhookSecret")
            or ""
        )
        return hmac.compare_digest(expected, provided)

    def _event_type(self, payload: dict[str, Any]) -> str:
        raw = str(
            self._pick(payload, "type", "event", "eventType", "name", "status") or ""
        ).strip()
        normalized = raw.lower().replace("_", ".").replace("-", ".")
        aliases = {
            "connected": "call.answered",
            "answered": "call.answered",
            "ringing": "call.ringing",
            "disconnected": "call.ended",
            "completed": "call.completed",
            "failed": "call.failed",
        }
        return aliases.get(normalized, normalized)

    def _build_voice_instructions(self, call: CallRecord) -> str:
        task = _sanitize_voice_task(call.task)
        return "\n".join(
            [
                "Ты ведёшь реальный телефонный звонок от имени пользователя.",
                f"Твоя задача: {task}.",
                "Говори только по-русски, естественно, спокойно и вежливо.",
                "Каждая реплика должна быть короткой: один вопрос или одно уточнение за раз.",
                "После каждой своей реплики остановись и жди ответа человека.",
                "Если слышишь тишину, шум, 'алло' или 'вас не слышно', коротко повтори последнюю понятную фразу и снова жди.",
                "Если задача выполнена или человек отказал, коротко поблагодари и заверши разговор.",
            ]
        )

    def _safe_call_dict(self, call: CallRecord, *, include_transcript: bool = False) -> dict[str, Any]:
        data = asdict(call)
        data.pop("stream_token", None)
        if not include_transcript:
            data["transcript"] = data["transcript"][-10:]
        return data

    @staticmethod
    def _pick(data: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None and not isinstance(value, (dict, list)):
                return str(value).strip()
        return ""


class RealtimeVoiceBridge:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
        language: str,
        call: CallRecord,
        on_assistant_audio,
        on_transcript,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.language = language
        self.call = call
        self.on_assistant_audio = on_assistant_audio
        self.on_transcript = on_transcript
        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.reader_task: asyncio.Task | None = None
        self._assistant_text = ""

    def _mark_metric_once(self, name: str) -> None:
        if name not in self.call.metrics:
            self.call.metrics[name] = _voice_metric_now()
            self.call.touch()

    async def connect(self) -> None:
        self.session = aiohttp.ClientSession()
        if self.provider == "xai":
            await self._connect_xai()
        else:
            await self._connect_openai()
        self.reader_task = asyncio.create_task(self._read_events())

    async def _connect_openai(self) -> None:
        assert self.session is not None
        self.ws = await self.session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            heartbeat=20,
        )
        await self.ws.send_json(self._openai_session_update_event())

    def _openai_session_update_event(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": self.instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "transcription": {
                            "model": os.environ.get(
                                "VOICE_CALL_REALTIME_TRANSCRIPTION_MODEL", "whisper-1"
                            ),
                            "language": self.language or "ru",
                        },
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": _voice_vad_eagerness(),
                            "create_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": self.voice,
                    },
                },
            },
        }

    def _openai_response_create_event(self, instructions: str) -> dict[str, Any]:
        return {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": self.voice,
                    },
                },
                "instructions": instructions,
            },
        }

    async def _connect_xai(self) -> None:
        assert self.session is not None
        base_url = os.environ.get("XAI_REALTIME_URL", "wss://api.x.ai/v1/realtime").strip()
        separator = "&" if "?" in base_url else "?"
        self.ws = await self.session.ws_connect(
            f"{base_url}{separator}model={self.model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            heartbeat=20,
        )
        await self.ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "instructions": self.instructions,
                    "voice": self.voice,
                    "audio": {
                        "input": {
                            "format": {
                                "type": os.environ.get("VOICE_CALL_XAI_INPUT_FORMAT", "audio/pcmu"),
                            }
                        },
                        "output": {
                            "format": {
                                "type": os.environ.get("VOICE_CALL_XAI_OUTPUT_FORMAT", "audio/pcmu"),
                            },
                        },
                    },
                    "turn_detection": {
                        "type": os.environ.get("VOICE_CALL_XAI_TURN_DETECTION", "server_vad"),
                        "create_response": True,
                    },
                },
            }
        )

    async def send_audio(self, audio: bytes) -> None:
        if self.ws and not self.ws.closed and audio:
            await self.ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )

    async def create_initial_response(self) -> None:
        if not self.ws or self.ws.closed:
            return
        instructions = (
            "Скажи только первую короткую фразу звонка: поздоровайся и одним вопросом "
            "озвучь задачу. Затем полностью остановись и жди ответа собеседника. "
            "Не продолжай сценарий сам."
        )
        if self.provider == "xai":
            await self.ws.send_json({"type": "response.create", "instructions": instructions})
            return
        await self.ws.send_json(self._openai_response_create_event(instructions))

    async def close(self) -> None:
        if self.reader_task:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass
        if self.ws and not self.ws.closed:
            await self.ws.close()
        if self.session:
            await self.session.close()

    async def _read_events(self) -> None:
        assert self.ws is not None
        async for msg in self.ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            event_type = str(event.get("type") or "")
            if event_type == "input_audio_buffer.speech_started":
                self._mark_metric_once("first_speech_started_at")
                logger.info("voice_call realtime speech_started call_id=%s", self.call.call_id)
                continue
            if event_type == "input_audio_buffer.speech_stopped":
                self._mark_metric_once("first_speech_stopped_at")
                logger.info("voice_call realtime speech_stopped call_id=%s", self.call.call_id)
                continue
            if event_type == "input_audio_buffer.committed":
                self._mark_metric_once("first_audio_committed_at")
                logger.info("voice_call realtime audio_committed call_id=%s", self.call.call_id)
                continue
            if event_type == "response.created":
                self._mark_metric_once("first_response_created_at")
                logger.info("voice_call realtime response_created call_id=%s", self.call.call_id)
                continue
            if event_type == "response.done":
                self._mark_metric_once("first_response_done_at")
                response = event.get("response")
                status = response.get("status") if isinstance(response, dict) else None
                details = None
                if isinstance(response, dict):
                    details = response.get("status_details") or response.get("error")
                logger.info(
                    "voice_call realtime response_done call_id=%s status=%s details=%s",
                    self.call.call_id,
                    status or "unknown",
                    details,
                )
                continue
            if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                self._mark_metric_once("first_realtime_audio_delta_at")
                delta = event.get("delta")
                if isinstance(delta, str):
                    await self.on_assistant_audio(base64.b64decode(delta))
                continue
            if event_type in {
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
                "response.text.delta",
            }:
                delta = event.get("delta")
                if isinstance(delta, str):
                    self._assistant_text += delta
                continue
            if event_type in {
                "response.audio_transcript.done",
                "response.output_audio_transcript.done",
                "response.output_text.done",
                "response.text.done",
            }:
                text = str(event.get("transcript") or event.get("text") or self._assistant_text).strip()
                self._assistant_text = ""
                if text:
                    self.on_transcript("assistant", text)
                continue
            if event_type in {
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.done",
                "input_audio_buffer.transcription.completed",
            }:
                text = str(event.get("transcript") or "").strip()
                if text:
                    self.on_transcript("callee", text)
                continue
            if event_type == "error":
                self._mark_metric_once("first_realtime_error_at")
                error = event.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or error).strip()
                    code = str(error.get("code") or error.get("type") or "").strip()
                    self.call.error = f"realtime_error: {code}: {message}" if code else f"realtime_error: {message}"
                else:
                    self.call.error = f"realtime_error: {event}"
                self.call.touch()
                logger.warning("voice_call realtime error provider=%s event=%s", self.provider, event)
