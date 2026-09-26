"""OpenAI Realtime transcription support for Discord voice channels.

This module is deliberately independent from discord.py. The Discord adapter
owns authorization and lifecycle; this module owns PCM conversion, the
Realtime WebSocket protocol, and per-user turn accounting.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import sys
import time
from array import array
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Mapping, Optional


DEFAULT_LIVE_ENDPOINT = (
    "wss://api.openai.com/v1/realtime?intent=transcription"
)
LIVE_MODEL = "gpt-live-transcribe"
LIVE_DELAYS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
DISCORD_STT_MODES = frozenset(
    {"configured", "openai_contextual", "openai_live_high"}
)
_MODE_ALIASES = {
    "": "configured",
    "default": "configured",
    "configured": "configured",
    "contextual": "openai_contextual",
    "openai-contextual": "openai_contextual",
    "openai_contextual": "openai_contextual",
    "live-high": "openai_live_high",
    "openai-live-high": "openai_live_high",
    "openai_live_high": "openai_live_high",
}
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?$")
_OPENAI_WEBSOCKET_LOGGER = logging.Logger(
    "hermes.openai_realtime_transcription.transport",
    level=logging.WARNING,
)
_OPENAI_WEBSOCKET_LOGGER.addHandler(logging.NullHandler())
_OPENAI_WEBSOCKET_LOGGER.propagate = False


class LiveTranscriptionError(RuntimeError):
    """A Realtime transcription connection or event-contract failure."""


def _require_admitted(admit: Optional[Callable[[], bool]]) -> None:
    if admit is not None and not admit():
        raise LiveTranscriptionError("Live transcription operation is not admitted")


def _create_no_redirect_websocket_connect(uri: str, **kwargs: Any) -> Any:
    """Create a quiet OpenAI WebSocket connector that rejects every redirect."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import SecurityError

    class _NoRedirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception | str:
            result = super().process_redirect(exc)
            if isinstance(result, str):
                return SecurityError("OpenAI Realtime redirects are disabled")
            return result

    kwargs.setdefault("logger", _OPENAI_WEBSOCKET_LOGGER)
    return _NoRedirectConnect(uri, **kwargs)


def normalize_discord_stt_mode(value: Any) -> str:
    """Normalize supported mode aliases without silently guessing."""
    key = str(value or "").strip().lower()
    mode = _MODE_ALIASES.get(key)
    if mode is None:
        supported = ", ".join(sorted(DISCORD_STT_MODES))
        raise ValueError(
            f"unsupported Discord STT mode {value!r}; expected one of {supported}"
        )
    return mode


def _string_tuple(value: Any, field: str = "languages") -> tuple[str, ...]:
    """Keep live context decoding local; validate raw members before stripping."""
    if isinstance(value, tuple):
        value = list(value)
    if value is None:
        return ()
    if isinstance(value, str):
        if field == "keywords" and any(char in value for char in "<>\r\n"):
            raise ValueError("stt.openai.keywords: forbidden character (<, >, CR or LF)")
        if value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"stt.openai.{field}: expected a valid JSON list of strings") from exc
        else:
            value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"stt.openai.{field}: expected a string or list of strings")
    if field == "keywords" and any(char in item for item in value for char in "<>\r\n"):
        raise ValueError("stt.openai.keywords: forbidden character (<, >, CR or LF)")
    values = tuple(item.strip() for item in value if item.strip())
    if field == "languages" and any(not _LANGUAGE_RE.fullmatch(item) for item in values):
        raise ValueError("stt.openai.languages: invalid language code format")
    return values


@dataclass(frozen=True)
class LiveTranscriptionConfig:
    """Validated configuration for one OpenAI transcription session."""

    model: str = LIVE_MODEL
    delay: str = "high"
    prompt: str = ""
    keywords: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    endpoint: str = DEFAULT_LIVE_ENDPOINT
    completion_timeout_seconds: float = 20.0
    send_timeout_seconds: float = 5.0
    max_session_seconds: float = 3300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", str(self.model or LIVE_MODEL).strip())
        object.__setattr__(self, "delay", str(self.delay or "high").strip().lower())
        if not isinstance(self.prompt, str):
            raise ValueError("live transcription prompt must be a string")
        if len(self.prompt) > 5000:
            raise ValueError("live transcription prompt exceeds 5000 characters")
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "keywords", _string_tuple(self.keywords, "keywords"))
        object.__setattr__(self, "languages", _string_tuple(self.languages))
        object.__setattr__(self, "endpoint", str(self.endpoint or "").strip())
        object.__setattr__(
            self,
            "completion_timeout_seconds",
            float(self.completion_timeout_seconds),
        )
        object.__setattr__(
            self,
            "send_timeout_seconds",
            float(self.send_timeout_seconds),
        )
        object.__setattr__(
            self,
            "max_session_seconds",
            float(self.max_session_seconds),
        )

        if self.model != LIVE_MODEL:
            raise ValueError(f"unsupported live transcription model: {self.model}")
        if self.delay not in LIVE_DELAYS:
            raise ValueError(f"unsupported live transcription delay: {self.delay}")
        if self.delay != "high":
            raise ValueError("openai_live_high requires delay=high")
        if self.endpoint != DEFAULT_LIVE_ENDPOINT:
            raise ValueError(
                "openai_live_high endpoint must be "
                "wss://api.openai.com/v1/realtime?intent=transcription"
            )
        if not 0 < self.completion_timeout_seconds <= 20:
            raise ValueError("completion timeout must be positive and at most 20 seconds")
        if not 0 < self.send_timeout_seconds <= 5:
            raise ValueError("send timeout must be positive and at most 5 seconds")
        if not 60 <= self.max_session_seconds <= 3300:
            raise ValueError("max session duration must be between 60 and 3300 seconds")
        if len(self.prompt) > 5000:
            raise ValueError("live transcription prompt exceeds 5000 characters")
        for keyword in self.keywords:
            if any(character in keyword for character in "<>\r\n"):
                raise ValueError(
                    "live transcription keyword contains a forbidden character"
                )
        for language in self.languages:
            if not _LANGUAGE_RE.fullmatch(language):
                raise ValueError(
                    f"invalid live transcription language code: {language!r}"
                )

    @classmethod
    def from_hermes_config(cls, config: Mapping[str, Any]) -> "LiveTranscriptionConfig":
        """Build settings from ``stt.openai`` and ``discord.voice_stt``."""
        stt = config.get("stt") if isinstance(config, Mapping) else {}
        stt = stt if isinstance(stt, Mapping) else {}
        openai_cfg = stt.get("openai")
        openai_cfg = openai_cfg if isinstance(openai_cfg, Mapping) else {}

        discord_cfg = config.get("discord") if isinstance(config, Mapping) else {}
        discord_cfg = discord_cfg if isinstance(discord_cfg, Mapping) else {}
        voice_stt = discord_cfg.get("voice_stt")
        voice_stt = voice_stt if isinstance(voice_stt, Mapping) else {}
        live_cfg = voice_stt.get("openai_live")
        live_cfg = live_cfg if isinstance(live_cfg, Mapping) else {}

        languages: tuple[str, ...] = ()
        for configured_languages in (
            live_cfg.get("languages"),
            live_cfg.get("language"),
            openai_cfg.get("languages"),
            openai_cfg.get("language"),
            stt.get("language"),
        ):
            languages = _string_tuple(configured_languages)
            if languages:
                break

        return cls(
            model=live_cfg.get("model", LIVE_MODEL),
            delay=live_cfg.get("delay", "high"),
            prompt=live_cfg.get("prompt", openai_cfg.get("prompt", "")),
            keywords=live_cfg.get("keywords", openai_cfg.get("keywords", ())),
            languages=_string_tuple(languages),
            endpoint=live_cfg.get("endpoint", DEFAULT_LIVE_ENDPOINT),
            completion_timeout_seconds=live_cfg.get(
                "completion_timeout_seconds", 20.0
            ),
            send_timeout_seconds=live_cfg.get("send_timeout_seconds", 5.0),
            max_session_seconds=live_cfg.get("max_session_seconds", 3300.0),
        )


def resolve_openai_realtime_api_key(config: Mapping[str, Any]) -> str:
    """Resolve a direct OpenAI key without routing through managed gateways."""
    stt = config.get("stt") if isinstance(config, Mapping) else {}
    stt = stt if isinstance(stt, Mapping) else {}
    openai_cfg = stt.get("openai")
    openai_cfg = openai_cfg if isinstance(openai_cfg, Mapping) else {}
    from tools.transcription_cloud import _is_native_openai_transcription_endpoint
    endpoint = openai_cfg.get("base_url")
    if endpoint and not _is_native_openai_transcription_endpoint(endpoint):
        raise ValueError("openai_live_high refuses the configured non-native OpenAI endpoint")
    configured = str(openai_cfg.get("api_key") or "").strip()
    if configured:
        return configured

    from hermes_cli.config import get_env_value
    for name in ("VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"):
        value = str(get_env_value(name) or "").strip()
        if value:
            return value
    raise ValueError(
        "openai_live_high requires a direct OpenAI API key in "
        "stt.openai.api_key, VOICE_TOOLS_OPENAI_KEY, or OPENAI_API_KEY"
    )


def build_live_session_update(config: LiveTranscriptionConfig) -> Dict[str, Any]:
    """Build the documented dedicated Realtime transcription session update."""
    transcription: Dict[str, Any] = {
        "model": config.model,
        "delay": config.delay,
    }
    if config.prompt:
        transcription["prompt"] = config.prompt
    if config.keywords:
        transcription["keywords"] = list(config.keywords)
    if config.languages:
        transcription["languages"] = list(config.languages)
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": transcription,
                    "turn_detection": None,
                }
            },
        },
    }


def pcm48_stereo_to_pcm24_mono(pcm: bytes) -> bytes:
    """Downmix signed 16-bit 48 kHz stereo PCM to 24 kHz mono PCM.

    Discord's decoder emits interleaved little-endian stereo frames. Each output
    sample averages both channels across two consecutive 48 kHz frames. This is
    a deterministic 2-tap low-pass before the exact 2:1 downsample and avoids
    the worst aliasing of plain frame decimation without adding a DSP dependency.
    """
    if len(pcm) % 8:
        raise ValueError("PCM must contain whole pairs of 48 kHz stereo frames")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    converted = array(
        "h",
        (
            (
                int(samples[index])
                + int(samples[index + 1])
                + int(samples[index + 2])
                + int(samples[index + 3])
            )
            // 4
            for index in range(0, len(samples), 4)
        ),
    )
    if sys.byteorder != "little":
        converted.byteswap()
    return converted.tobytes()


@dataclass
class _PendingTurn:
    future: asyncio.Future
    item_id: Optional[str] = None
    delta_parts: list[str] = field(default_factory=list)


class OpenAIRealtimeTranscriptionSession:
    """One persistent, multi-turn OpenAI transcription WebSocket session."""

    MAX_EARLY_ITEM_IDS = 16
    MAX_EARLY_EVENTS_PER_ITEM = 8
    MAX_ITEM_TOMBSTONES = 64

    def __init__(
        self,
        *,
        api_key: str,
        config: LiveTranscriptionConfig,
        websocket_connect: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError("OpenAI Realtime API key is required")
        self._api_key = str(api_key).strip()
        self.config = config
        self._websocket_connect = websocket_connect
        self._clock = clock
        self._ws = None
        self._reader_task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()
        self._rollover_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._connected_at: Optional[float] = None
        self._rollover_count = 0
        self._generation = 0
        self._terminal_error: Optional[LiveTranscriptionError] = None
        self._pending_unassigned: Deque[_PendingTurn] = deque()
        self._pending_by_item: Dict[str, _PendingTurn] = {}
        self._early_events: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        self._item_tombstones: set[str] = set()
        self._item_tombstone_order: Deque[str] = deque()
        self._closed = False

    async def start(self, *, admit: Optional[Callable[[], bool]] = None) -> None:
        _require_admitted(admit)
        if self._closed:
            raise LiveTranscriptionError("OpenAI Realtime session is closed")
        if self._ws is not None and self._reader_task and not self._reader_task.done():
            return
        async with self._start_lock:
            _require_admitted(admit)
            if self._closed:
                raise LiveTranscriptionError("OpenAI Realtime session is closed")
            if self._ws is not None and self._reader_task and not self._reader_task.done():
                return
            stale_ws = self._ws
            self._ws = None
            self._reader_task = None
            self._connected_at = None
            if stale_ws is not None:
                try:
                    await stale_ws.close()
                except Exception:
                    pass
            connect = self._websocket_connect
            if connect is None:
                connect = _create_no_redirect_websocket_connect

            async def open_transport():
                _require_admitted(admit)
                return await connect(
                    self.config.endpoint,
                    additional_headers={"Authorization": f"Bearer {self._api_key}"},
                    compression=None,
                    open_timeout=10,
                    close_timeout=5,
                    max_size=4 * 1024 * 1024,
                    logger=_OPENAI_WEBSOCKET_LOGGER,
                    proxy=None,
                )

            self._ws = await asyncio.wait_for(open_transport(), timeout=10.0)
            try:
                _require_admitted(admit)
                await self._send_json(build_live_session_update(self.config), admit=admit)
                await asyncio.wait_for(self._wait_for_session_updated(), timeout=10.0)
                _require_admitted(admit)
            except BaseException:
                failed_ws = self._ws
                self._ws = None
                if failed_ws is not None:
                    try:
                        await failed_ws.close()
                    except Exception:
                        pass
                raise
            self._connected_at = self._clock()
            self._generation += 1
            self._terminal_error = None
            self._reader_task = asyncio.create_task(self._reader_loop())

    @property
    def rollover_count(self) -> int:
        return self._rollover_count

    @property
    def generation(self) -> int:
        return self._generation

    async def rollover_if_due(self, *, admit: Optional[Callable[[], bool]] = None) -> bool:
        """Reconnect between turns before the provider's 60-minute limit."""
        _require_admitted(admit)
        connected_at = self._connected_at
        if connected_at is None:
            return False
        if self._clock() - connected_at < self.config.max_session_seconds:
            return False
        if self._pending_unassigned or self._pending_by_item:
            return False
        async with self._rollover_lock:
            connected_at = self._connected_at
            if connected_at is None:
                return False
            if self._clock() - connected_at < self.config.max_session_seconds:
                return False
            if self._pending_unassigned or self._pending_by_item:
                return False
            async with self._send_lock:
                await self._close_transport()
            _require_admitted(admit)
            await self.start(admit=admit)
            self._rollover_count += 1
            return True

    async def _wait_for_session_updated(self) -> None:
        assert self._ws is not None
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            event = json.loads(raw)
            event_type = str(event.get("type") or "")
            if event_type == "error":
                raise LiveTranscriptionError(self._safe_error_type(event))
            if event_type in {"session.updated", "transcription_session.updated"}:
                return

    @staticmethod
    def _safe_error_type(event: Mapping[str, Any]) -> str:
        error = event.get("error")
        if isinstance(error, Mapping):
            code = str(error.get("code") or error.get("type") or "api_error")
        else:
            code = "api_error"
        return f"OpenAI Realtime transcription error: {code}"

    async def _send_json(
        self, event: Mapping[str, Any], *, admit: Optional[Callable[[], bool]] = None,
    ) -> None:
        ws = self._ws
        if ws is None:
            raise LiveTranscriptionError("OpenAI Realtime session is not connected")

        async def send() -> None:
            # wait_for schedules a task: check inside it immediately before wire admission.
            _require_admitted(admit)
            if self._closed or self._ws is not ws:
                raise LiveTranscriptionError("OpenAI Realtime session is closed")
            await ws.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

        await asyncio.wait_for(send(), timeout=self.config.send_timeout_seconds)
        if self._closed:
            raise LiveTranscriptionError("OpenAI Realtime session is closed")
        _require_admitted(admit)

    async def append_pcm24(self, pcm: bytes, *, admit: Optional[Callable[[], bool]] = None) -> None:
        if not pcm or len(pcm) % 2:
            raise ValueError("live PCM must be non-empty signed 16-bit mono audio")
        self._raise_terminal_error()
        _require_admitted(admit)
        await self.start(admit=admit)
        _require_admitted(admit)
        self._raise_terminal_error()
        async with self._send_lock:
            _require_admitted(admit)
            await self._send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }, admit=admit,
            )

    async def commit(self, *, admit: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        pending = await self._send_commit(admit=admit)
        return await self._wait_for_commit(pending)

    async def _send_commit(self, *, admit: Optional[Callable[[], bool]] = None) -> _PendingTurn:
        """Seal the input buffer before allowing the next turn's PCM to be sent."""
        self._raise_terminal_error()
        _require_admitted(admit)
        await self.start(admit=admit)
        _require_admitted(admit)
        self._raise_terminal_error()
        loop = asyncio.get_running_loop()
        pending = _PendingTurn(future=loop.create_future())
        async with self._send_lock:
            _require_admitted(admit)
            self._pending_unassigned.append(pending)
            try:
                await self._send_json({"type": "input_audio_buffer.commit"}, admit=admit)
            except BaseException:
                self._remove_pending(pending)
                raise
        return pending

    async def _wait_for_commit(self, pending: _PendingTurn) -> Dict[str, Any]:
        """Await the existing item ledger without holding an audio-send barrier."""
        try:
            return await asyncio.wait_for(
                asyncio.shield(pending.future),
                timeout=self.config.completion_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._remove_pending(pending)
            raise LiveTranscriptionError(
                "OpenAI Realtime transcription completion timed out"
            ) from exc

    async def clear(self) -> None:
        if self._ws is None:
            return
        async with self._send_lock:
            await self._send_json({"type": "input_audio_buffer.clear"})

    async def _reader_loop(self) -> None:
        try:
            while self._ws is not None:
                raw = await self._ws.recv()
                event = json.loads(raw)
                await self._observe_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_terminal_error(
                exc
                if isinstance(exc, LiveTranscriptionError)
                else LiveTranscriptionError(
                    f"OpenAI Realtime receive failed: {type(exc).__name__}"
                )
            )

    async def _observe_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "error":
            self._record_terminal_error(
                LiveTranscriptionError(self._safe_error_type(event))
            )
            return
        if event_type == "input_audio_buffer.committed":
            item_id = str(event.get("item_id") or "").strip()
            if not item_id:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime commit acknowledgement had no item_id"
                    )
                )
                return
            if item_id in self._pending_by_item or item_id in self._item_tombstones:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime duplicate commit acknowledgement"
                    )
                )
                return
            if not self._pending_unassigned:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime unexpected commit acknowledgement"
                    )
                )
                return
            pending = self._pending_unassigned.popleft()
            pending.item_id = item_id
            self._pending_by_item[item_id] = pending
            for early in self._early_events.pop(item_id, []):
                await self._observe_event(early)
            return
        if event_type not in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        }:
            return
        item_id = str(event.get("item_id") or "").strip()
        if item_id in self._item_tombstones:
            return
        pending = self._pending_by_item.get(item_id)
        if pending is None:
            if not item_id:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime transcription event had no item_id"
                    )
                )
                return
            if not self._pending_unassigned:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime transcription event had no pending turn"
                    )
                )
                return
            if (
                item_id not in self._early_events
                and len(self._early_events) >= self.MAX_EARLY_ITEM_IDS
            ):
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime early-item limit exceeded"
                    )
                )
                return
            events = self._early_events[item_id]
            if len(events) >= self.MAX_EARLY_EVENTS_PER_ITEM:
                self._record_terminal_error(
                    LiveTranscriptionError(
                        "OpenAI Realtime early-event limit exceeded"
                    )
                )
                return
            events.append(event)
            return
        if event_type.endswith(".delta"):
            delta = str(event.get("delta") or "")
            if delta:
                pending.delta_parts.append(delta)
            return
        self._pending_by_item.pop(item_id, None)
        self._remember_item_tombstone(item_id)
        if event_type.endswith(".failed"):
            if not pending.future.done():
                pending.future.set_exception(
                    LiveTranscriptionError("OpenAI Realtime transcription failed")
                )
            return
        if not pending.future.done():
            pending.future.set_result(
                {
                    "success": True,
                    "transcript": str(event.get("transcript") or "").strip(),
                    "partial_transcript": "".join(pending.delta_parts),
                    "provider": "openai_realtime",
                    "model": self.config.model,
                    "delay": self.config.delay,
                    "item_id": item_id,
                    "usage": event.get("usage"),
                    "session_rollovers": self._rollover_count,
                }
            )

    def _remember_item_tombstone(self, item_id: str) -> None:
        if not item_id or item_id in self._item_tombstones:
            return
        self._item_tombstones.add(item_id)
        self._item_tombstone_order.append(item_id)
        while len(self._item_tombstone_order) > self.MAX_ITEM_TOMBSTONES:
            expired = self._item_tombstone_order.popleft()
            self._item_tombstones.discard(expired)

    def _remove_pending(self, pending: _PendingTurn) -> None:
        try:
            self._pending_unassigned.remove(pending)
        except ValueError:
            pass
        if pending.item_id:
            self._pending_by_item.pop(pending.item_id, None)
        if not pending.future.done():
            pending.future.cancel()

    def _fail_all(self, error: BaseException) -> None:
        pending = list(self._pending_unassigned) + list(
            self._pending_by_item.values()
        )
        self._pending_unassigned.clear()
        self._pending_by_item.clear()
        self._early_events.clear()
        seen = set()
        for turn in pending:
            if id(turn) in seen:
                continue
            seen.add(id(turn))
            if not turn.future.done():
                turn.future.set_exception(error)

    def _record_terminal_error(self, error: LiveTranscriptionError) -> None:
        self._terminal_error = error
        self._fail_all(error)

    def _raise_terminal_error(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error

    async def _close_transport(self) -> None:
        reader = self._reader_task
        self._reader_task = None
        if reader is not None and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        ws = self._ws
        self._ws = None
        self._connected_at = None
        self._early_events.clear()
        self._item_tombstones.clear()
        self._item_tombstone_order.clear()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def close(self) -> None:
        async with self._start_lock:
            self._closed = True
            async with self._send_lock:
                await self._close_transport()
        self._fail_all(LiveTranscriptionError("OpenAI Realtime session closed"))


class DiscordLiveTranscriptionController:
    """Per-user persistent sessions plus full-utterance source accounting."""

    MAX_SESSIONS = 4

    def __init__(
        self,
        *,
        api_key: str,
        config: LiveTranscriptionConfig,
        session_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self._api_key = api_key
        self._session_factory = session_factory
        self._sessions: Dict[int, Any] = {}
        self._source_bytes: Dict[int, int] = defaultdict(int)
        self._turn_errors: Dict[int, str] = {}
        self._turn_generations: Dict[int, int] = {}
        self._turn_admissions: Dict[int, Callable[[], bool]] = {}
        self._operation_lock = asyncio.Lock()
        self._closed = False

    def _get_session(self, user_id: int):
        if self._closed:
            raise LiveTranscriptionError("Discord Live STT controller is closed")
        session = self._sessions.get(user_id)
        if session is None:
            if len(self._sessions) >= self.MAX_SESSIONS:
                raise LiveTranscriptionError("Discord Live STT session limit reached")
            if self._session_factory is not None:
                session = self._session_factory()
            else:
                session = OpenAIRealtimeTranscriptionSession(
                    api_key=self._api_key,
                    config=self.config,
                )
            self._sessions[user_id] = session
        return session

    async def append_pcm48(
        self, user_id: int, pcm: bytes, *, admit: Optional[Callable[[], bool]] = None,
    ) -> None:
        async with self._operation_lock:
            await self._append_pcm48(user_id, pcm, admit=admit)

    async def _append_pcm48(
        self, user_id: int, pcm: bytes, *, admit: Optional[Callable[[], bool]] = None,
    ) -> None:
        if self._closed:
            raise LiveTranscriptionError("Discord Live STT controller is closed")
        if not pcm or user_id in self._turn_errors:
            return
        try:
            _require_admitted(admit)
            previous_admission = self._turn_admissions.get(user_id)
            _require_admitted(previous_admission)
            pcm24 = pcm48_stereo_to_pcm24_mono(pcm)
            session = self._get_session(user_id)

            def admitted() -> bool:
                return (not self._closed and self._sessions.get(user_id) is session
                        and (admit is None or admit())
                        and (previous_admission is None or previous_admission()))

            if admit is not None and previous_admission is None:
                self._turn_admissions[user_id] = admit
            if self._source_bytes[user_id] == 0:
                rollover = getattr(session, "rollover_if_due", None)
                if rollover is not None:
                    if admit is None:
                        await rollover()
                    else:
                        await rollover(admit=admitted)
            _require_admitted(admitted)
            if admit is None:
                await session.append_pcm24(pcm24)
            else:
                await session.append_pcm24(pcm24, admit=admitted)
            if self._closed:
                raise LiveTranscriptionError("Discord Live STT controller is closed")
            _require_admitted(admitted)
            generation = int(getattr(session, "generation", 0))
            previous_generation = self._turn_generations.get(user_id)
            if previous_generation is None:
                self._turn_generations[user_id] = generation
            elif previous_generation != generation:
                self._turn_errors[user_id] = "session_generation_changed"
                return
            self._source_bytes[user_id] += len(pcm)
        except Exception as exc:
            if self._closed:
                raise LiveTranscriptionError(
                    "Discord Live STT controller is closed"
                ) from exc
            self._turn_errors[user_id] = type(exc).__name__

    async def finish_utterance(
        self,
        user_id: int,
        *,
        expected_source_bytes: int,
        admit: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        prepared = await self._prepare_utterance(
            user_id, expected_source_bytes=expected_source_bytes, admit=admit)
        return await self._wait_for_utterance(user_id, prepared)

    async def _prepare_utterance(
        self, user_id: int, *, expected_source_bytes: int,
        admit: Optional[Callable[[], bool]] = None,
    ):
        """Seal one turn under the append barrier, leaving its result to the caller."""
        async with self._operation_lock:
            return await self._finish_utterance(
                user_id,
                expected_source_bytes=expected_source_bytes,
                admit=admit,
            )

    async def _wait_for_utterance(self, user_id: int, prepared) -> Dict[str, Any]:
        """Wait on the captured session/generation, without blocking later PCM."""
        if isinstance(prepared, dict):
            return prepared
        session, pending, generation, admitted = prepared
        try:
            result = await session._wait_for_commit(pending)
            _require_admitted(admitted)
            return result
        except (Exception, asyncio.CancelledError) as exc:
            await self._retire_prepared_utterance(user_id, prepared)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return self._failure(f"live_commit_failed:{type(exc).__name__}")

    async def _retire_prepared_utterance(self, user_id: int, prepared) -> None:
        """Also retire a sealed handle whose lexical waiter never started."""
        if isinstance(prepared, dict):
            return
        session, pending, generation, admitted = prepared
        async with self._operation_lock:
            if session.generation != generation:
                return
            session._remove_pending(pending)
            # A late completion owns only its original transport, never a replacement.
            if self._sessions.get(user_id) is session:
                if self._source_bytes.get(user_id):
                    # Retain later accounting, but reject audio lost with this session.
                    self._turn_errors[user_id] = "live_commit_failed"
                await self._drop_session(user_id)

    async def _finish_utterance(
        self,
        user_id: int,
        *,
        expected_source_bytes: int,
        admit: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any] | tuple[OpenAIRealtimeTranscriptionSession, _PendingTurn, Optional[int], Callable[[], bool]]:
        if self._closed:
            return self._failure("live_controller_closed")
        streamed = self._source_bytes.pop(user_id, 0)
        turn_error = self._turn_errors.pop(user_id, None)
        generation = self._turn_generations.pop(user_id, None)
        previous_admission = self._turn_admissions.pop(user_id, None)
        session = self._sessions.get(user_id)
        if (admit is not None and not admit()) or (previous_admission is not None and not previous_admission()):
            await self._drop_session(user_id)
            return self._failure("live_operation_not_admitted")
        if turn_error:
            await self._drop_session(user_id)
            return self._failure("live_stream_failed")
        if streamed != int(expected_source_bytes) or session is None:
            await self._discard_current_buffer(session)
            return self._failure("incomplete_live_stream")
        try:
            def admitted() -> bool:
                return (not self._closed and self._sessions.get(user_id) is session
                        and int(getattr(session, "generation", 0)) == generation
                        and (admit is None or admit())
                        and (previous_admission is None or previous_admission()))

            _require_admitted(admitted)
            if isinstance(session, OpenAIRealtimeTranscriptionSession):
                pending = await session._send_commit(admit=admitted)
                return session, pending, generation, admitted
            # Preserve the injected session factory's historical commit contract.
            result = await session.commit(admit=admitted) if admit is not None else await session.commit()
            _require_admitted(admitted)
            return result
        except (Exception, asyncio.CancelledError) as exc:
            await self._drop_session(user_id)
            if isinstance(exc, asyncio.CancelledError):
                # SEND may already have committed before cancellation, without a waiter.
                raise
            return self._failure(f"live_commit_failed:{type(exc).__name__}")

    async def _discard_current_buffer(self, session: Any) -> None:
        if session is None:
            return
        try:
            await session.clear()
        except Exception:
            for user_id, candidate in list(self._sessions.items()):
                if candidate is session:
                    await self._drop_session(user_id)
                    break

    def _failure(self, error: str) -> Dict[str, Any]:
        return {
            "success": False,
            "transcript": "",
            "error": error,
            "provider": "openai_realtime",
            "model": self.config.model,
            "delay": self.config.delay,
        }

    async def _drop_session(self, user_id: int) -> None:
        session = self._sessions.pop(user_id, None)
        if session is not None:
            cancellation = None
            try:
                # The map no longer owns this transport. Keep its close alive until
                # reader and socket retirement settle, even if the waiter is cancelled.
                close_task = asyncio.create_task(session.close())
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                close_task.result()
            except Exception:
                pass
            if cancellation is not None:
                raise cancellation

    async def abort_user(self, user_id: int) -> None:
        """Discard all accounting and provider state for one user."""
        async with self._operation_lock:
            self._source_bytes.pop(user_id, None)
            self._turn_errors.pop(user_id, None)
            self._turn_generations.pop(user_id, None)
            self._turn_admissions.pop(user_id, None)
            await self._drop_session(user_id)

    async def abort_pending(self) -> None:
        """Abort discarded captures while the receiver is paused; keep idle sessions."""
        async with self._operation_lock:
            users = (self._source_bytes.keys() | self._turn_errors.keys()
                     | self._turn_generations.keys() | self._turn_admissions.keys())
            users.update(user_id for user_id, session in self._sessions.items()
                         if isinstance(session, OpenAIRealtimeTranscriptionSession)
                         and (session._pending_unassigned or session._pending_by_item))
        for user_id in users:
            await self.abort_user(user_id)

    async def close(self) -> None:
        self._closed = True
        async with self._operation_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._source_bytes.clear()
            self._turn_errors.clear()
            self._turn_generations.clear()
            self._turn_admissions.clear()
            await asyncio.gather(
                *(session.close() for session in sessions),
                return_exceptions=True,
            )
