"""OpenAI Realtime voice bridge for gateway voice channels.

This module owns the provider-facing session loop. Platform adapters keep
owning media transport, authorization, and delivery. The bridge consumes PCM
audio chunks, forwards them to a Realtime session, executes Hermes tools when
the model asks for a function call, and returns generated PCM audio via a
callback.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlencode

from tools.tool_backend_helpers import resolve_openai_audio_api_key

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop as _audioop
except ImportError:  # pragma: no cover - audioop is absent in newer Python runtimes.
    _audioop = None

logger = logging.getLogger(__name__)

DIRECT_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime-2"
DEFAULT_REALTIME_VOICE = "marin"
DEFAULT_INPUT_SAMPLE_RATE = 24000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_OUTPUT_FLUSH_BYTES = DEFAULT_OUTPUT_SAMPLE_RATE  # ~0.5s mono s16le
DEFAULT_VAD_THRESHOLD = 0.55
DEFAULT_VAD_PREFIX_PADDING_MS = 250
DEFAULT_VAD_SILENCE_DURATION_MS = 350
DEFAULT_MANUAL_TURN_TIMEOUT_MS = 700
DEFAULT_INPUT_SILENCE_THRESHOLD = 120
DEFAULT_RESPONSE_START_TIMEOUT_MS = 4500
MAX_RESPONSE_CREATE_ATTEMPTS = 2
DEFAULT_REALTIME_INSTRUCTIONS = (
    "You are Hermes speaking in a Discord voice channel. Keep replies concise "
    "and natural for live conversation. Use the available Hermes tools when a "
    "request requires current local state, files, commands, memory, or other "
    "tool-backed information. If the user asks you to use terminal, run a "
    "command, check pwd, inspect files, or answer what folder/process/state "
    "Hermes is running from, call the matching Hermes tool before speaking. "
    "Do not answer those requests from memory. After a tool result, answer "
    "the user in audio."
)
_VALID_AUTH_MODES = {"auto", "direct", "managed", "codex"}


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    enabled: bool = False
    model: str = DEFAULT_REALTIME_MODEL
    voice: str = DEFAULT_REALTIME_VOICE
    instructions: str = ""
    reasoning_effort: str = "low"
    auth_mode: str = "auto"
    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_prefix_padding_ms: int = DEFAULT_VAD_PREFIX_PADDING_MS
    vad_silence_duration_ms: int = DEFAULT_VAD_SILENCE_DURATION_MS
    manual_turn_timeout_ms: int = DEFAULT_MANUAL_TURN_TIMEOUT_MS
    input_silence_threshold: int = DEFAULT_INPUT_SILENCE_THRESHOLD
    response_start_timeout_ms: int = DEFAULT_RESPONSE_START_TIMEOUT_MS


@dataclass(frozen=True)
class RealtimeAuthConfig:
    api_key: str
    websocket_url: str
    mode: str


def _is_truthy(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def load_realtime_voice_config(user_config: Optional[Dict[str, Any]] = None) -> RealtimeVoiceConfig:
    """Read ``voice.realtime`` config with Discord-specific overrides."""
    cfg = user_config or {}
    voice_cfg = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
    realtime_cfg = voice_cfg.get("realtime") if isinstance(voice_cfg.get("realtime"), dict) else {}
    discord_cfg = realtime_cfg.get("discord") if isinstance(realtime_cfg.get("discord"), dict) else {}

    def pick(name: str, default: object = None) -> object:
        if name in discord_cfg:
            return discord_cfg.get(name)
        if name in realtime_cfg:
            return realtime_cfg.get(name)
        return default

    enabled = _is_truthy(pick("enabled", False), default=False)
    auth_mode = str(pick("auth_mode", "auto") or "auto").strip().lower()
    if auth_mode not in _VALID_AUTH_MODES:
        auth_mode = "auto"
    if auth_mode == "codex":
        auth_mode = "managed"

    return RealtimeVoiceConfig(
        enabled=enabled,
        model=str(pick("model", DEFAULT_REALTIME_MODEL) or DEFAULT_REALTIME_MODEL).strip(),
        voice=str(pick("voice", DEFAULT_REALTIME_VOICE) or DEFAULT_REALTIME_VOICE).strip(),
        instructions=str(pick("instructions", "") or ""),
        reasoning_effort=str(pick("reasoning_effort", "low") or "low").strip(),
        auth_mode=auth_mode,
        input_sample_rate=int(pick("input_sample_rate", DEFAULT_INPUT_SAMPLE_RATE) or DEFAULT_INPUT_SAMPLE_RATE),
        output_sample_rate=int(pick("output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE) or DEFAULT_OUTPUT_SAMPLE_RATE),
        vad_threshold=float(pick("vad_threshold", DEFAULT_VAD_THRESHOLD) or DEFAULT_VAD_THRESHOLD),
        vad_prefix_padding_ms=int(pick("vad_prefix_padding_ms", DEFAULT_VAD_PREFIX_PADDING_MS) or DEFAULT_VAD_PREFIX_PADDING_MS),
        vad_silence_duration_ms=int(pick("vad_silence_duration_ms", DEFAULT_VAD_SILENCE_DURATION_MS) or DEFAULT_VAD_SILENCE_DURATION_MS),
        manual_turn_timeout_ms=int(pick("manual_turn_timeout_ms", DEFAULT_MANUAL_TURN_TIMEOUT_MS) or DEFAULT_MANUAL_TURN_TIMEOUT_MS),
        input_silence_threshold=int(
            pick(
                "input_silence_threshold",
                pick("silence_threshold", voice_cfg.get("silence_threshold", DEFAULT_INPUT_SILENCE_THRESHOLD)),
            )
            or DEFAULT_INPUT_SILENCE_THRESHOLD
        ),
        response_start_timeout_ms=int(
            pick("response_start_timeout_ms", DEFAULT_RESPONSE_START_TIMEOUT_MS)
            or DEFAULT_RESPONSE_START_TIMEOUT_MS
        ),
    )


def discord_realtime_voice_enabled(user_config: Optional[Dict[str, Any]] = None) -> bool:
    return load_realtime_voice_config(user_config).enabled


def resolve_realtime_auth(config: RealtimeVoiceConfig) -> RealtimeAuthConfig:
    """Resolve direct OpenAI credentials or Hermes OpenAI Codex/GPT OAuth."""
    direct_key = resolve_openai_audio_api_key()

    if config.auth_mode == "direct":
        if direct_key:
            return RealtimeAuthConfig(
                api_key=direct_key,
                websocket_url=os.getenv("OPENAI_REALTIME_WS_URL", DIRECT_REALTIME_WS_URL).strip() or DIRECT_REALTIME_WS_URL,
                mode="direct",
            )
        raise ValueError(
            "OpenAI Realtime voice direct auth requires VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY."
        )

    if direct_key and config.auth_mode == "auto":
        return RealtimeAuthConfig(
            api_key=direct_key,
            websocket_url=os.getenv("OPENAI_REALTIME_WS_URL", DIRECT_REALTIME_WS_URL).strip() or DIRECT_REALTIME_WS_URL,
            mode="direct",
        )

    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        token = str(creds.get("api_key", "") or "").strip()
    except Exception:
        token = ""
    if token:
        return RealtimeAuthConfig(
            api_key=token,
            websocket_url=os.getenv("OPENAI_REALTIME_WS_URL", DIRECT_REALTIME_WS_URL).strip() or DIRECT_REALTIME_WS_URL,
            mode="managed",
        )

    raise ValueError(
        "OpenAI Realtime voice requires VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY "
        "or OpenAI Codex/GPT OAuth credentials from `hermes auth add openai-codex`."
    )


def _require_websocket_connect():
    try:
        from websockets.sync.client import connect as ws_connect  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "websockets package is required for OpenAI Realtime voice; "
            "install the messaging extra or run: pip install websockets"
        ) from exc
    return ws_connect


def discord_pcm_to_realtime_pcm(
    pcm: bytes,
    *,
    src_rate: int = 48000,
    src_channels: int = 2,
    dst_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
) -> bytes:
    """Convert Discord s16le PCM to mono s16le PCM for Realtime input.

    Discord provides 48kHz stereo PCM. Realtime voice sessions commonly use
    24kHz mono PCM.
    """
    if not pcm:
        return b""
    frame_width = 2 * max(1, int(src_channels))
    usable = len(pcm) - (len(pcm) % frame_width)
    if usable <= 0:
        return b""
    pcm = pcm[:usable]

    if _audioop is not None:
        try:
            if src_channels >= 2:
                mono = _audioop.tomono(pcm, 2, 0.5, 0.5)
            else:
                mono = pcm
            if src_rate != dst_rate:
                mono, _ = _audioop.ratecv(mono, 2, 1, int(src_rate), int(dst_rate), None)
            return mono
        except Exception:
            logger.debug(
                "Falling back to Python PCM conversion",
                exc_info=True,
            )

    samples: List[int] = []
    step_frames = 2 if src_rate == 48000 and dst_rate == 24000 else 1
    for frame_index in range(0, usable // frame_width, step_frames):
        offset = frame_index * frame_width
        if src_channels >= 2:
            left = int.from_bytes(pcm[offset:offset + 2], "little", signed=True)
            right = int.from_bytes(pcm[offset + 2:offset + 4], "little", signed=True)
            sample = int((left + right) / 2)
        else:
            sample = int.from_bytes(pcm[offset:offset + 2], "little", signed=True)
        samples.append(max(-32768, min(32767, sample)))

    return b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)


def pcm_rms(pcm: bytes) -> float:
    """Return RMS amplitude for s16le mono/stereo PCM."""
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    if _audioop is not None:
        return float(_audioop.rms(pcm[:usable], 2))
    total = 0
    count = 0
    for offset in range(0, usable, 2):
        sample = int.from_bytes(pcm[offset:offset + 2], "little", signed=True)
        total += sample * sample
        count += 1
    if not count:
        return 0.0
    return (total / count) ** 0.5


def hermes_tools_to_realtime_tools(tool_schemas: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI Chat-style function schemas to Realtime function tools."""
    realtime_tools: List[Dict[str, Any]] = []
    for schema in tool_schemas or []:
        fn = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        realtime_tools.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return realtime_tools


def realtime_session_instructions(config: RealtimeVoiceConfig) -> str:
    """Return configured Realtime instructions or the Discord voice default."""
    return config.instructions.strip() or DEFAULT_REALTIME_INSTRUCTIONS


class OpenAIRealtimeVoiceSession:
    """Threaded Realtime session for Discord voice-channel audio."""

    def __init__(
        self,
        *,
        config: RealtimeVoiceConfig,
        auth: RealtimeAuthConfig,
        tool_schemas: Iterable[Dict[str, Any]],
        on_audio_response: Callable[[bytes], None],
        on_user_audio_start: Optional[Callable[[], None]] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.auth = auth
        self._tools = hermes_tools_to_realtime_tools(tool_schemas)
        self._on_audio_response = on_audio_response
        self._on_user_audio_start = on_user_audio_start
        self._task_id = task_id
        self._session_id = session_id
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._response_audio = bytearray()
        self._errors: "queue.Queue[Exception]" = queue.Queue()
        self._handled_call_ids: set[str] = set()
        self._sent_audio_chunks = 0
        self._received_audio_chunks = 0
        self._session_updated_logged = False
        self._last_audio_flush_at = time.monotonic()
        self._turn_timer: Optional[threading.Timer] = None
        self._turn_has_audio = False
        self._last_input_audio_at = 0.0
        self._response_in_progress = False
        self._pending_barge_in = False
        self._barge_in_cancel_sent = False
        self._response_watchdog_timer: Optional[threading.Timer] = None
        self._awaiting_response_start = False
        self._response_create_attempts = 0

    @property
    def active(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        connect = _require_websocket_connect()
        query = urlencode({"model": self.config.model})
        sep = "&" if "?" in self.auth.websocket_url else "?"
        url = f"{self.auth.websocket_url}{sep}{query}"
        headers = [
            ("Authorization", f"Bearer {self.auth.api_key}"),
        ]
        try:
            self._ws = connect(url, additional_headers=headers)
        except TypeError:
            self._ws = connect(url, extra_headers=headers)

        logger.info(
            "OpenAI Realtime voice connected: mode=%s model=%s tools=%d",
            self.auth.mode,
            self.config.model,
            len(self._tools),
        )
        self._running.set()
        self._send_session_update()
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name="hermes-openai-realtime-voice",
            daemon=True,
        )
        self._recv_thread.start()

    def close(self) -> None:
        self._running.clear()
        ws = self._ws
        self._ws = None
        self._cancel_turn_timer()
        self._cancel_response_watchdog()
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def send_discord_pcm(self, pcm: bytes) -> None:
        if not self.active or not pcm:
            return
        converted = discord_pcm_to_realtime_pcm(
            pcm,
            src_rate=48000,
            src_channels=2,
            dst_rate=self.config.input_sample_rate,
        )
        if not converted:
            return
        rms = pcm_rms(converted)
        if rms < max(0, int(self.config.input_silence_threshold)):
            return
        self._sent_audio_chunks += 1
        if self._sent_audio_chunks <= 3 or self._sent_audio_chunks in {10, 25, 50, 100}:
            logger.info(
                "OpenAI Realtime voice input chunk #%d: discord_bytes=%d realtime_bytes=%d rms=%.1f",
                self._sent_audio_chunks,
                len(pcm),
                len(converted),
                rms,
            )
        if not self._turn_has_audio:
            if self._response_in_progress:
                self._pending_barge_in = True
                logger.info("OpenAI Realtime voice user barge-in detected during response")
                self._cancel_response_for_barge_in()
            self._notify_user_audio_start()
        self._send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(converted).decode("ascii"),
        })
        self._schedule_manual_turn_finalize()

    def _send_session_update(self) -> None:
        session: Dict[str, Any] = {
            "type": "realtime",
            "model": self.config.model,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.input_sample_rate,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.config.vad_threshold,
                        "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                        "silence_duration_ms": self.config.vad_silence_duration_ms,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.output_sample_rate,
                    },
                    "voice": self.config.voice,
                },
            },
            "tool_choice": "auto",
        }
        instructions = realtime_session_instructions(self.config)
        if instructions:
            session["instructions"] = instructions
        if self.config.reasoning_effort:
            session["reasoning"] = {"effort": self.config.reasoning_effort}
        if self._tools:
            session["tools"] = self._tools
        self._send_json({"type": "session.update", "session": session})

    def _recv_loop(self) -> None:
        while self._running.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv()
                frame = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                if isinstance(frame, dict):
                    self._handle_frame(frame)
            except Exception as exc:
                if self._running.is_set():
                    logger.warning("OpenAI Realtime voice receive loop stopped: %s", exc)
                    self._errors.put(exc)
                self._running.clear()
                break

    def _handle_frame(self, frame: Dict[str, Any]) -> None:
        ftype = frame.get("type")
        if ftype in {"response.audio.delta", "response.output_audio.delta"}:
            self._response_in_progress = True
            self._mark_response_started()
            if self._pending_barge_in:
                return
            self._turn_has_audio = False
            b64 = frame.get("delta") or frame.get("audio") or ""
            if b64:
                try:
                    decoded = base64.b64decode(b64)
                    self._received_audio_chunks += 1
                    if self._received_audio_chunks <= 3 or self._received_audio_chunks in {10, 25, 50, 100}:
                        logger.info(
                            "OpenAI Realtime voice output audio chunk #%d: bytes=%d",
                            self._received_audio_chunks,
                            len(decoded),
                        )
                    self._response_audio.extend(decoded)
                    self._maybe_flush_audio_response(reason="stream")
                except (TypeError, ValueError):
                    logger.debug("Ignoring invalid realtime audio delta")
        elif ftype == "session.updated":
            if not self._session_updated_logged:
                self._session_updated_logged = True
                session = frame.get("session") if isinstance(frame.get("session"), dict) else {}
                logger.info(
                    "OpenAI Realtime voice session updated: model=%s modalities=%s tools=%d turn_detection=%s",
                    session.get("model") or self.config.model,
                    session.get("output_modalities"),
                    len(self._tools),
                    (((session.get("audio") or {}).get("input") or {}).get("turn_detection")),
                )
        elif ftype in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}:
            logger.info("OpenAI Realtime voice VAD event: %s", ftype)
        elif ftype in {"response.created", "response.output_item.added"}:
            self._response_in_progress = True
            self._mark_response_started()
            if self._pending_barge_in:
                self._ensure_turn_timer()
            else:
                self._turn_has_audio = False
                self._cancel_turn_timer()
            logger.info("OpenAI Realtime voice response event: %s", ftype)
        elif ftype in {"response.done", "response.audio.done", "response.output_audio.done"}:
            self._maybe_flush_audio_response(force=True, reason=ftype)
            if ftype == "response.done":
                self._response_in_progress = False
                if self._turn_has_audio:
                    self._ensure_turn_timer()
        elif ftype == "response.function_call_arguments.done":
            self._mark_response_started()
            self._handle_function_call(
                name=str(frame.get("name") or ""),
                call_id=str(frame.get("call_id") or frame.get("item_id") or ""),
                arguments=frame.get("arguments") or "{}",
            )
        elif ftype == "response.output_item.done":
            item = frame.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                self._mark_response_started()
                self._handle_function_call(
                    name=str(item.get("name") or ""),
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    arguments=item.get("arguments") or "{}",
                )
        elif ftype == "error":
            self._cancel_response_watchdog()
            logger.warning("OpenAI Realtime voice error: %s", frame.get("error") or frame)

    def _schedule_manual_turn_finalize(self) -> None:
        self._turn_has_audio = True
        self._last_input_audio_at = time.monotonic()
        if self._turn_timer is not None:
            return
        timeout = max(100, int(self.config.manual_turn_timeout_ms)) / 1000.0
        self._arm_turn_timer(timeout)

    def _arm_turn_timer(self, delay: float) -> None:
        timer = threading.Timer(max(0.01, delay), self._finalize_input_turn)
        timer.daemon = True
        self._turn_timer = timer
        timer.start()

    def _ensure_turn_timer(self) -> None:
        if self._turn_timer is not None or not self._turn_has_audio:
            return
        timeout = max(100, int(self.config.manual_turn_timeout_ms)) / 1000.0
        elapsed = time.monotonic() - self._last_input_audio_at
        self._arm_turn_timer(max(0.01, timeout - elapsed))

    def _cancel_turn_timer(self) -> None:
        timer = self._turn_timer
        self._turn_timer = None
        if timer is not None:
            timer.cancel()

    def _mark_response_started(self) -> None:
        self._awaiting_response_start = False
        self._response_create_attempts = 0
        self._cancel_response_watchdog()

    def _cancel_response_watchdog(self) -> None:
        timer = self._response_watchdog_timer
        self._response_watchdog_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_response_watchdog(self) -> None:
        self._cancel_response_watchdog()
        timeout = max(500, int(self.config.response_start_timeout_ms)) / 1000.0
        timer = threading.Timer(timeout, self._response_start_timed_out)
        timer.daemon = True
        self._response_watchdog_timer = timer
        timer.start()

    def _response_start_timed_out(self) -> None:
        self._response_watchdog_timer = None
        if not self.active or not self._awaiting_response_start or self._response_in_progress:
            return
        if self._response_create_attempts >= MAX_RESPONSE_CREATE_ATTEMPTS:
            logger.warning(
                "OpenAI Realtime voice response did not start after %d create attempt(s)",
                self._response_create_attempts,
            )
            self._awaiting_response_start = False
            return
        self._response_create_attempts += 1
        logger.warning(
            "OpenAI Realtime voice response did not start within %dms; retrying response.create (attempt %d/%d)",
            self.config.response_start_timeout_ms,
            self._response_create_attempts,
            MAX_RESPONSE_CREATE_ATTEMPTS,
        )
        self._send_json({"type": "response.create"})
        self._arm_response_watchdog()

    def _finalize_input_turn(self) -> None:
        self._turn_timer = None
        if not self.active or not self._turn_has_audio:
            return
        if self._response_in_progress:
            self._arm_turn_timer(0.1)
            return
        timeout = max(100, int(self.config.manual_turn_timeout_ms)) / 1000.0
        elapsed = time.monotonic() - self._last_input_audio_at
        if elapsed < timeout:
            timer = threading.Timer(timeout - elapsed, self._finalize_input_turn)
            timer.daemon = True
            self._turn_timer = timer
            timer.start()
            return
        self._turn_has_audio = False
        self._pending_barge_in = False
        self._barge_in_cancel_sent = False
        logger.info(
            "OpenAI Realtime voice finalizing input turn after %dms silence",
            self.config.manual_turn_timeout_ms,
        )
        self._send_json({"type": "input_audio_buffer.commit"})
        self._awaiting_response_start = True
        self._response_create_attempts = 1
        self._send_json({"type": "response.create"})
        self._arm_response_watchdog()

    def _cancel_response_for_barge_in(self) -> None:
        if self._barge_in_cancel_sent:
            return
        self._barge_in_cancel_sent = True
        self._response_audio.clear()
        logger.info("OpenAI Realtime voice cancelling response for user barge-in")
        self._send_json({"type": "response.cancel"})

    def _notify_user_audio_start(self) -> None:
        callback = self._on_user_audio_start
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            logger.debug("Realtime voice user-audio callback failed: %s", exc)

    def _maybe_flush_audio_response(self, *, force: bool = False, reason: str = "stream") -> None:
        if not force and len(self._response_audio) < DEFAULT_OUTPUT_FLUSH_BYTES:
            return
        if not self._response_audio:
            return
        pcm = bytes(self._response_audio)
        self._response_audio.clear()
        self._last_audio_flush_at = time.monotonic()
        try:
            logger.info(
                "OpenAI Realtime voice flushing audio response: bytes=%d reason=%s",
                len(pcm),
                reason,
            )
            self._on_audio_response(pcm)
        except Exception as exc:
            logger.warning("Realtime voice audio callback failed: %s", exc)

    def _handle_function_call(self, *, name: str, call_id: str, arguments: object) -> None:
        if not name or not call_id:
            return
        if call_id in self._handled_call_ids:
            return
        self._handled_call_ids.add(call_id)
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                args = {}
        except ValueError:
            args = {}

        try:
            from model_tools import handle_function_call

            started_at = time.monotonic()
            logger.info("OpenAI Realtime voice executing Hermes tool: name=%s call_id=%s", name, call_id)
            result = handle_function_call(
                name,
                args,
                task_id=self._task_id,
                tool_call_id=call_id,
                session_id=self._session_id,
            )
            logger.info(
                "OpenAI Realtime voice completed Hermes tool: name=%s call_id=%s duration_ms=%d result_bytes=%d",
                name,
                call_id,
                int((time.monotonic() - started_at) * 1000),
                len(result.encode("utf-8") if isinstance(result, str) else str(result).encode("utf-8")),
            )
        except Exception as exc:
            result = json.dumps({"error": str(exc)}, ensure_ascii=False)
            logger.warning("OpenAI Realtime voice Hermes tool failed: name=%s call_id=%s error=%s", name, call_id, exc)

        self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        })
        self._send_json({"type": "response.create"})

    def _send_json(self, payload: Dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        with self._send_lock:
            ws.send(json.dumps(payload))
