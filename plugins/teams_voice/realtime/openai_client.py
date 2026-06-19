"""Realtime speech-to-speech client (OpenAI / Azure Realtime API).

A thin async wrapper over the Realtime WebSocket. It is provider-pure: it deals
only in the model's native PCM 24 kHz audio and fires callbacks; resampling to
the bridge's 16 kHz, frame chunking, expression/viseme emission, and barge-in
handling all live in the dialogue handler (``handlers.py``).

Mirrors the role of openclaw ``msteams-realtime.ts``. Uses aiohttp's WS client
(already a bridge dependency) rather than adding the ``websockets`` package.

Azure note: pass ``base_url`` like
``wss://<resource>.openai.azure.com/openai/realtime?api-version=...&deployment=...``
and ``api_key_header="api-key"``; the default is OpenAI's bearer auth.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "alloy"
DEFAULT_AZURE_API_VERSION = "2024-10-01-preview"
DEFAULT_INSTRUCTIONS = (
    "You are a helpful voice assistant on a Microsoft Teams call. Keep replies "
    "brief and conversational. For anything requiring real work (lookups, "
    "actions, files), delegate to the agent rather than guessing."
)

# The model emits/consumes PCM 16-bit mono at this rate over the Realtime API.
REALTIME_SAMPLE_RATE_HZ = 24_000

AsyncCb = Callable[..., Awaitable[None]]


@dataclass
class RealtimeConfig:
    """Resolved realtime-provider settings (OpenAI or Azure OpenAI)."""

    api_key: str
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    instructions: str = DEFAULT_INSTRUCTIONS
    base_url: str = DEFAULT_BASE_URL
    api_key_header: str = "Authorization"  # "api-key" for Azure
    # Server-VAD tuning (mirrors the openclaw realtime provider config).
    vad_threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    # Caller-audio transcription (for wake words / verbal interrupts). Empty
    # string disables it (some deployments don't support the field).
    input_transcribe_model: str = "whisper-1"
    # Bilingual Arabic/English mode (opt-in): pin the model to detect/mirror the
    # caller's language and translate on request.
    bilingual: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def _to_ws(url: str) -> str:
    """Normalize an http(s) endpoint to a ws(s) one; pass ws(s) through."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _pick(block: "dict[str, Any]", key: str, env: str, default: str = "") -> str:
    """Resolve one value: config.yaml block first, then env, then default."""
    val = block.get(key)
    if val is not None and str(val).strip() != "":
        return str(val).strip()
    return os.getenv(env, "").strip() or default


def realtime_config_from_env(block: "dict[str, Any] | None" = None) -> RealtimeConfig:
    """Build a :class:`RealtimeConfig` from config.yaml + environment.

    Both sources are supported per the Hermes docs: the per-plugin config.yaml
    block (``plugins.entries.teams_voice.config.realtime``) takes precedence,
    with environment variables as the fallback. ``block`` is read from config.yaml
    when omitted (pass ``{}`` to force env-only, e.g. in tests).

    Azure is selected when ``backend: azure`` / ``TEAMS_VOICE_REALTIME_BACKEND=azure``,
    an Azure endpoint is set, or an explicit ``*.azure.com`` URL is given;
    otherwise OpenAI (bearer auth). The Azure key falls back to
    ``AZURE_OPENAI_API_KEY`` / ``AZURE_FOUNDRY_API_KEY`` so the gateway key is reused.
    """
    if block is None:
        try:
            from ..config import plugin_config_block

            block = plugin_config_block().get("realtime", {}) or {}
        except Exception:  # noqa: BLE001
            block = {}

    backend = _pick(block, "backend", "TEAMS_VOICE_REALTIME_BACKEND").lower()
    explicit_url = _pick(block, "url", "TEAMS_VOICE_REALTIME_URL")
    azure_endpoint = _pick(block, "azure_endpoint", "TEAMS_VOICE_AZURE_ENDPOINT")
    voice = _pick(block, "voice", "TEAMS_VOICE_REALTIME_VOICE", DEFAULT_VOICE)
    instructions = _pick(block, "instructions", "TEAMS_VOICE_REALTIME_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)

    def _vad() -> "tuple[float, int, int]":
        try:
            thr = float(_pick(block, "vad_threshold", "TEAMS_VOICE_VAD_THRESHOLD", "0.5"))
            prefix = int(_pick(block, "prefix_padding_ms", "TEAMS_VOICE_PREFIX_PADDING_MS", "300"))
            silence = int(_pick(block, "silence_duration_ms", "TEAMS_VOICE_SILENCE_DURATION_MS", "500"))
        except ValueError:
            return 0.5, 300, 500
        return thr, prefix, silence

    vad_threshold, prefix_padding_ms, silence_duration_ms = _vad()
    # "" / "none" / "off" disables caller transcription.
    transcribe_model = _pick(block, "input_transcribe_model", "TEAMS_VOICE_INPUT_TRANSCRIBE_MODEL", "whisper-1")
    if transcribe_model.lower() in ("none", "off", "disabled"):
        transcribe_model = ""
    bilingual = _pick(block, "bilingual", "TEAMS_VOICE_BILINGUAL", "").lower() in ("1", "true", "yes", "on")
    is_azure = backend == "azure" or bool(azure_endpoint) or "azure.com" in explicit_url

    if is_azure:
        deployment = _pick(block, "azure_deployment", "TEAMS_VOICE_AZURE_DEPLOYMENT")
        api_version = _pick(
            block, "azure_api_version", "TEAMS_VOICE_AZURE_API_VERSION", DEFAULT_AZURE_API_VERSION
        )
        if explicit_url:
            base_url = _to_ws(explicit_url)
        else:
            base = _to_ws(azure_endpoint.rstrip("/"))
            base_url = f"{base}/openai/realtime?api-version={api_version}&deployment={deployment}"
        api_key = (
            _pick(block, "api_key", "TEAMS_VOICE_REALTIME_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            or os.getenv("AZURE_FOUNDRY_API_KEY", "").strip()
        )
        return RealtimeConfig(
            api_key=api_key,
            model=deployment or DEFAULT_MODEL,
            voice=voice,
            instructions=instructions,
            base_url=base_url,
            api_key_header="api-key",
            vad_threshold=vad_threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            input_transcribe_model=transcribe_model,
            bilingual=bilingual,
        )

    return RealtimeConfig(
        api_key=_pick(block, "api_key", "TEAMS_VOICE_REALTIME_API_KEY")
        or os.getenv("OPENAI_API_KEY", "").strip(),
        model=_pick(block, "model", "TEAMS_VOICE_REALTIME_MODEL", DEFAULT_MODEL),
        voice=voice,
        instructions=instructions,
        base_url=explicit_url or DEFAULT_BASE_URL,
        api_key_header="Authorization",
        vad_threshold=vad_threshold,
        prefix_padding_ms=prefix_padding_ms,
        silence_duration_ms=silence_duration_ms,
        input_transcribe_model=transcribe_model,
        bilingual=bilingual,
    )


class RealtimeSession:
    """One realtime model connection for a single call.

    Set the ``on_*`` callbacks before :meth:`connect`. All callbacks are async
    and best-effort — an exception in a callback is logged, never propagated into
    the receive loop.
    """

    def __init__(self, config: RealtimeConfig) -> None:
        self._cfg = config
        self._ws: Any = None
        self._session: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False
        self._response_active = False  # True between response.created and response.done

        # Function tools exposed to the model (realtime shape: flat
        # {type:"function", name, description, parameters}). Set before connect().
        self.tools: list[dict] = []

        # Callbacks (wired by the handler).
        self.on_audio_delta: Optional[AsyncCb] = None  # (pcm24k: bytes)
        self.on_transcript_delta: Optional[AsyncCb] = None  # (text: str) — bot reply
        self.on_input_transcript: Optional[AsyncCb] = None  # (text: str) — caller turn
        self.on_speech_started: Optional[AsyncCb] = None  # () -> barge-in
        self.on_response_done: Optional[AsyncCb] = None  # ()
        self.on_function_call: Optional[AsyncCb] = None  # (name, call_id, args_json)
        self.on_error: Optional[AsyncCb] = None  # (error: Any)

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the WS, configure the session, and start the receive loop."""
        import aiohttp

        if self._cfg.api_key_header.lower() == "authorization":
            headers = {
                "Authorization": f"Bearer {self._cfg.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }
        else:  # Azure-style
            headers = {self._cfg.api_key_header: self._cfg.api_key}

        url = self._cfg.base_url
        if "model=" not in url and "deployment=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}model={self._cfg.model}"

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url, headers=headers, max_msg_size=0)

        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "instructions": self._cfg.instructions,
            "voice": self._cfg.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": self._cfg.vad_threshold,
                "prefix_padding_ms": self._cfg.prefix_padding_ms,
                "silence_duration_ms": self._cfg.silence_duration_ms,
                "create_response": True,
            },
        }
        # Transcribe the caller's audio so the handler can detect wake words and
        # verbal interrupts. Configurable/optional: if unsupported, set the model
        # to "" and the gate/interrupts degrade gracefully (VAD barge-in still works).
        if self._cfg.input_transcribe_model:
            session["input_audio_transcription"] = {"model": self._cfg.input_transcribe_model}
        if self.tools:
            session["tools"] = self.tools
            session["tool_choice"] = "auto"
        await self._send({"type": "session.update", "session": session})
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("[teams_voice] realtime connected model=%s", self._cfg.model)

    async def close(self) -> None:
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
        finally:
            if self._session is not None and not self._session.closed:
                await self._session.close()

    # ── send paths ───────────────────────────────────────────────────────────

    async def push_audio(self, pcm24k: bytes) -> None:
        """Append a chunk of caller audio (PCM 24 kHz) to the input buffer."""
        if self._closed or not pcm24k:
            return
        await self._send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm24k).decode("ascii")}
        )

    async def cancel_response(self) -> None:
        """Cancel the in-flight model response (barge-in), if one is active.

        Guarded on ``_response_active`` so we don't spam ``response.cancel`` when
        nothing is playing (server-VAD fires speech_started on every utterance).
        """
        if not self._closed and self._response_active:
            self._response_active = False
            await self._send({"type": "response.cancel"})

    async def send_user_text(self, text: str, *, respond: bool = True) -> None:
        """Inject a user-role text item; optionally trigger a spoken response.

        ``respond`` is guarded on ``_response_active`` so we never hit
        'conversation already has an active response'.
        """
        if self._closed or not text:
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        if respond and not self._response_active:
            self._response_active = True
            await self._send({"type": "response.create"})

    async def request_say(self, instruction: str) -> None:
        """Inject an instruction and trigger a spoken response (outbound delivery
        / greeting, where there is no caller turn to respond to)."""
        await self.send_user_text(instruction, respond=True)

    async def send_image(self, image_url: str) -> None:
        """Push an ambient image into the conversation with NO forced response.

        Keeps the realtime model continuously visually aware (the openclaw
        ``sendImage`` equivalent). Best-effort.
        """
        if self._closed or not image_url:
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": image_url}],
                },
            }
        )

    async def send_function_result(self, call_id: str, output: str) -> None:
        """Return a tool result to the model and ask it to continue speaking.

        ``output`` is a plain string (the tool's spoken-result text). The model
        then generates a new audio response incorporating it.
        """
        if self._closed:
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": output},
            }
        )
        await self._send({"type": "response.create"})

    async def _send(self, obj: dict) -> None:
        if self._ws is None or self._ws.closed:
            return
        await self._ws.send_str(json.dumps(obj))

    # ── receive loop ─────────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                try:
                    evt = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(evt)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — keep the call alive on transport hiccups
            logger.error("[teams_voice] realtime recv loop error", exc_info=True)

    async def _dispatch(self, evt: dict) -> None:
        etype = evt.get("type", "")
        # Audio output (handle both the beta and GA event names).
        if etype in ("response.audio.delta", "response.output_audio.delta"):
            b64 = evt.get("delta")
            if b64:
                await self._safe(self.on_audio_delta, base64.b64decode(b64))
        elif etype in (
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        ):
            text = evt.get("delta") or ""
            if text:
                await self._safe(self.on_transcript_delta, text)
        elif etype == "response.created":
            self._response_active = True
        elif etype == "input_audio_buffer.speech_started":
            await self._safe(self.on_speech_started)
        elif etype == "conversation.item.input_audio_transcription.completed":
            text = evt.get("transcript") or ""
            if text:
                await self._safe(self.on_input_transcript, text)
        elif etype == "response.done":
            self._response_active = False
            # A response may carry function calls (no audio) and/or spoken output.
            resp = evt.get("response") or {}
            for item in resp.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    await self._safe(
                        self.on_function_call,
                        item.get("name", ""),
                        item.get("call_id", ""),
                        item.get("arguments", "") or "{}",
                    )
            await self._safe(self.on_response_done)
        elif etype == "error":
            logger.warning("[teams_voice] realtime error: %s", evt.get("error"))
            await self._safe(self.on_error, evt.get("error"))

    async def _safe(self, cb: Optional[AsyncCb], *args: Any) -> None:
        if cb is None:
            return
        try:
            await cb(*args)
        except Exception:  # noqa: BLE001 — a callback fault must not kill the loop
            logger.error("[teams_voice] realtime callback error", exc_info=True)
