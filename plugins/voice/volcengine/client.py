"""Low-level WebSocket client for Volcengine voice APIs.

Implements Volcengine/ByteDance bidirectional streaming TTS and streaming ASR
(bigmodel) protocol framing, event handling, and async audio streaming.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import os
import struct
import uuid
import wave
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import AsyncIterator, Literal

import websockets
from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class VolcengineVoiceError(Exception):
    """Base error for the Volcengine voice client."""

    def __init__(self, message: str, *, code: int | None = None, payload: bytes | str | dict | None = None):
        super().__init__(message)
        self.code = code
        self.payload = payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = super().__str__()
        if self.code is not None:
            base = f"[{self.code}] {base}"
        return base


class VolcengineAuthError(VolcengineVoiceError):
    """Authentication / authorization failure (45000001, 401, 403...)."""


class VolcengineParamError(VolcengineVoiceError):
    """Invalid request parameter (45000002...)."""


class VolcengineServerError(VolcengineVoiceError):
    """Server-side failure (55000000, 5xxxxxxx...)."""


class VolcengineTimeoutError(VolcengineVoiceError):
    """Operation timed out."""


def _raise_for_code(code: int, message: str, payload: bytes | str | dict | None = None) -> None:
    """Map Volcengine error code to the appropriate exception."""
    if code == 45000001 or code in (401, 403):
        raise VolcengineAuthError(message, code=code, payload=payload)
    if code == 45000002 or 45000000 <= code < 46000000:
        raise VolcengineParamError(message, code=code, payload=payload)
    if code >= 55000000 or 500 <= code < 600 or code >= 50000000:
        raise VolcengineServerError(message, code=code, payload=payload)
    raise VolcengineVoiceError(message, code=code, payload=payload)


# ---------------------------------------------------------------------------
# Protocol enums & Message
# ---------------------------------------------------------------------------
class MsgType(IntEnum):
    Invalid = 0
    FullClientRequest = 0b0001
    AudioOnlyClient = 0b0010
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111


class Flags(IntEnum):
    """Type-specific flags (low 4 bits of byte 1)."""

    NoSeq = 0b0000
    PositiveSeq = 0b0001
    LastNoSeq = 0b0010
    NegativeSeq = 0b0011
    WithEvent = 0b0100


class Serialization(IntEnum):
    Raw = 0
    JSON = 0b0001


class Compression(IntEnum):
    None_ = 0
    Gzip = 0b0001


class Event(IntEnum):
    None_ = 0
    # Upstream connection
    StartConnection = 1
    FinishConnection = 2
    # Downstream connection
    ConnectionStarted = 50
    ConnectionFailed = 51
    ConnectionFinished = 52
    # Upstream session
    StartSession = 100
    CancelSession = 101
    FinishSession = 102
    # Downstream session
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    # Upstream generic
    TaskRequest = 200
    UpdateConfig = 201
    # Downstream TTS
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359
    # Downstream ASR
    ASRInfo = 450
    ASRResponse = 451
    ASREnded = 459


# Events that do NOT carry a session_id field on the wire.
_CONNECTION_EVENTS = frozenset({
    Event.StartConnection,
    Event.FinishConnection,
    Event.ConnectionStarted,
    Event.ConnectionFailed,
    Event.ConnectionFinished,
})


@dataclass
class Message:
    """Unified wire message for both TTS and ASR.

    Only one dataclass — marshal/unmarshal inspect (type, flag) to decide
    which optional fields are present.
    """

    type: MsgType = MsgType.Invalid
    flag: Flags = Flags.NoSeq
    serialization: Serialization = Serialization.JSON
    compression: Compression = Compression.None_

    event: Event = Event.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0

    payload: bytes = field(default=b"")

    # -- helpers -----------------------------------------------------------
    def json_payload(self) -> dict | None:
        """Return payload decoded as JSON (handling gzip). None if payload empty."""
        if not self.payload:
            return None
        raw = self.payload
        try:
            if self.compression == Compression.Gzip:
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    # -- marshal -----------------------------------------------------------
    def marshal(self) -> bytes:
        """Serialize to bytes. 4-byte header + optional fields + payload."""
        buf = io.BytesIO()
        buf.write(bytes([
            (1 << 4) | 1,  # version=1, header_size=1 (=> 4 bytes)
            (int(self.type) << 4) | int(self.flag),
            (int(self.serialization) << 4) | int(self.compression),
            0x00,
        ]))

        # Optional fields. Order matters on the wire.
        if self.flag == Flags.WithEvent:
            buf.write(struct.pack(">i", int(self.event)))
            if self.event not in _CONNECTION_EVENTS:
                sid = self.session_id.encode("utf-8")
                buf.write(struct.pack(">I", len(sid)))
                if sid:
                    buf.write(sid)

        if self.type == MsgType.Error:
            buf.write(struct.pack(">I", self.error_code & 0xFFFFFFFF))
        elif self.flag in (Flags.PositiveSeq, Flags.NegativeSeq) and self.type in (
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ):
            buf.write(struct.pack(">i", self.sequence))

        payload = self.payload
        buf.write(struct.pack(">I", len(payload)))
        if payload:
            buf.write(payload)
        return buf.getvalue()

    # -- unmarshal ---------------------------------------------------------
    @classmethod
    def unmarshal(cls, data: bytes) -> "Message":
        """Deserialize from bytes. Raises ValueError on malformed input."""
        if len(data) < 4:
            raise ValueError(f"frame too short: {len(data)} bytes")

        version = data[0] >> 4
        header_size = (data[0] & 0x0F) * 4
        if version != 1:
            logger.debug("unexpected protocol version %d", version)
        if header_size < 4:
            raise ValueError(f"invalid header_size {header_size}")

        msg = cls(
            type=MsgType(data[1] >> 4),
            flag=Flags(data[1] & 0x0F),
            serialization=Serialization(data[2] >> 4),
            compression=Compression(data[2] & 0x0F),
        )

        buf = io.BytesIO(data[header_size:])  # skip header + any padding

        # Flags-specific header extensions come BEFORE seq/err on servers too.
        # Server FullServerResponse with Pos/NegSeq emits seq+size+payload.
        # Server WithEvent emits event, session_id(len+bytes), connect_id(len+bytes).
        if msg.flag == Flags.WithEvent:
            msg.event = Event(struct.unpack(">i", buf.read(4))[0])
            if msg.event not in _CONNECTION_EVENTS:
                sid_size = struct.unpack(">I", buf.read(4))[0]
                if sid_size:
                    msg.session_id = buf.read(sid_size).decode("utf-8", errors="replace")
            else:
                # Server connection events carry connect_id(size+bytes) on response side.
                if msg.type == MsgType.FullServerResponse:
                    cid_size_bytes = buf.read(4)
                    if cid_size_bytes:
                        cid_size = struct.unpack(">I", cid_size_bytes)[0]
                        if cid_size:
                            msg.connect_id = buf.read(cid_size).decode("utf-8", errors="replace")

        if msg.type == MsgType.Error:
            msg.error_code = struct.unpack(">I", buf.read(4))[0]
        elif msg.flag in (Flags.PositiveSeq, Flags.NegativeSeq) and msg.type in (
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ):
            msg.sequence = struct.unpack(">i", buf.read(4))[0]
        elif msg.flag == Flags.LastNoSeq:
            # No sequence field; nothing to read.
            pass

        size_bytes = buf.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size:
                msg.payload = buf.read(size)
        return msg


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
def _build_headers(
    app_id: str,
    access_token: str,
    resource_id: str,
    request_id: str | None,
) -> dict[str, str]:
    """Construct the X-Api-* headers required by Volcengine voice endpoints.

    X-Api-Connect-Id is always fresh (per the protocol spec).
    """
    return {
        "X-Api-App-Id": app_id,
        "X-Api-App-Key": app_id,  # Some endpoints use App-Key; send both for compat.
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id or str(uuid.uuid4()),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }


def _resolve_credentials(app_id: str | None, access_token: str | None) -> tuple[str, str]:
    app_id = app_id or os.environ.get("VOLCENGINE_APP_ID")
    access_token = access_token or os.environ.get("VOLCENGINE_ACCESS_TOKEN")
    if not app_id or not access_token:
        raise VolcengineAuthError(
            "missing credentials: set VOLCENGINE_APP_ID and VOLCENGINE_ACCESS_TOKEN "
            "environment variables, or pass app_id/access_token explicitly"
        )
    return app_id, access_token


# ---------------------------------------------------------------------------
# Low-level send/recv helpers
# ---------------------------------------------------------------------------
async def _recv(ws, timeout: float) -> Message:
    """Receive one binary frame and parse into a Message. Raises on errors."""
    try:
        data = await asyncio.wait_for(ws.recv(), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise VolcengineTimeoutError(f"timed out waiting for server frame ({timeout}s)") from e
    except WebSocketException as e:
        raise VolcengineVoiceError(f"connection closed while waiting for server frame: {e}") from e
    if isinstance(data, str):
        raise VolcengineVoiceError(f"unexpected text frame: {data[:200]!r}")
    if not data:
        raise VolcengineVoiceError("empty server frame")
    try:
        msg = Message.unmarshal(data)
    except (ValueError, struct.error) as e:
        raise VolcengineVoiceError(f"malformed server frame: {e}") from e
    logger.debug("recv: type=%s event=%s payload=%r",
                 msg.type.name, msg.event.name if msg.event else "-",
                 msg.payload[:100] if msg.payload else b"")
    if msg.type == MsgType.Error:
        body = msg.payload
        try:
            body_str = body.decode("utf-8", errors="replace")
        except Exception:
            body_str = repr(body)
        _raise_for_code(msg.error_code, f"server error: {body_str}", payload=body_str)
    if msg.event in (Event.ConnectionFailed, Event.SessionFailed):
        body = msg.json_payload() or msg.payload
        raise VolcengineServerError(
            f"{msg.event.name}: {body!r}",
            code=msg.error_code or None,
            payload=body,
        )
    return msg


async def _send(ws, msg: Message) -> None:
    logger.debug("send: type=%s event=%s payload=%r",
                 msg.type.name, msg.event.name if msg.event else "-",
                 msg.payload[:100] if msg.payload else b"")
    await ws.send(msg.marshal())


# ---------------------------------------------------------------------------
# TTS bidirectional streaming
# ---------------------------------------------------------------------------
async def tts_stream(
    text: str,
    *,
    speaker: str = "zh_female_vv_uranus_bigtts",
    audio_format: Literal["mp3", "pcm", "ogg_opus"] = "mp3",
    sample_rate: int = 24000,
    speed_ratio: float = 1.0,
    emotion: str | None = None,
    emotion_scale: int | None = None,
    app_id: str | None = None,
    access_token: str | None = None,
    resource_id: str = "seed-tts-2.0",
    request_id: str | None = None,
    timeout: float = 30.0,
) -> AsyncIterator[bytes]:
    """Stream TTS audio chunks from Volcengine bidirectional TTS.

    Yields raw audio bytes (in the requested ``audio_format``) as they arrive.
    Handles the full event dance:
        StartConnection -> ConnectionStarted
        StartSession    -> SessionStarted
        TaskRequest     -> (AudioOnlyServer...) TTSEnded
        FinishSession   -> SessionFinished
        FinishConnection-> ConnectionFinished

    Accepts ``speed_ratio`` as a *multiplier* (1.0 = normal); Volcengine's
    ``speech_rate`` field is integer percent-delta in [-50, 100], so we
    translate.

    Raises:
        VolcengineAuthError: bad credentials / 401 / 45000001.
        VolcengineParamError: invalid parameters (45000002 etc).
        VolcengineServerError: backend failure / 55000000.
        VolcengineTimeoutError: any phase exceeded the timeout.
    """
    app_id, access_token = _resolve_credentials(app_id, access_token)
    headers = _build_headers(app_id, access_token, resource_id, request_id)

    # speed_ratio (1.0 multiplier) -> speech_rate (integer percent delta)
    speech_rate = int(round((speed_ratio - 1.0) * 100))
    speech_rate = max(-50, min(100, speech_rate))

    req_params: dict = {
        "speaker": speaker,
        "audio_params": {
            "format": audio_format,
            "sample_rate": sample_rate,
            "speech_rate": speech_rate,
        },
    }
    # Optional emotion controls (Seed-TTS 2.0 emo_v2 voices).
    # emotion: happy|sad|angry|surprised|fear|hate|excited|coldness|neutral|...
    # emotion_scale: 1..5 (default 4 server-side)
    if emotion:
        req_params["audio_params"]["emotion"] = emotion
    if emotion_scale is not None:
        req_params["audio_params"]["emotion_scale"] = int(emotion_scale)
    base_payload = {
        "user": {"uid": headers["X-Api-Request-Id"]},
        "namespace": "BidirectionalTTS",
        "req_params": req_params,
    }

    session_id = str(uuid.uuid4())
    logger.info("tts: connecting (session=%s, speaker=%s, format=%s)",
                session_id, speaker, audio_format)

    try:
        ws = await asyncio.wait_for(
            websockets.connect(
                TTS_URL,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                open_timeout=timeout,
                ping_interval=20,
                ping_timeout=20,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        raise VolcengineTimeoutError(f"tts connect timed out ({timeout}s)") from e
    except WebSocketException as e:
        # Surface auth-looking failures as auth errors.
        text_ = str(e)
        if "401" in text_ or "403" in text_ or "auth" in text_.lower():
            raise VolcengineAuthError(f"tts handshake rejected: {e}") from e
        raise VolcengineVoiceError(f"tts handshake failed: {e}") from e
    except OSError as e:
        raise VolcengineVoiceError(f"tts connection failed: {e}") from e

    try:
        # 1. StartConnection
        await _send(ws, Message(
            type=MsgType.FullClientRequest, flag=Flags.WithEvent,
            event=Event.StartConnection, payload=b"{}",
        ))
        msg = await _recv(ws, timeout)
        if msg.event != Event.ConnectionStarted:
            raise VolcengineServerError(
                f"expected ConnectionStarted, got {msg.event.name}",
                payload=msg.json_payload() or msg.payload,
            )

        # 2. StartSession
        start_payload = dict(base_payload, event=int(Event.StartSession))
        await _send(ws, Message(
            type=MsgType.FullClientRequest, flag=Flags.WithEvent,
            event=Event.StartSession, session_id=session_id,
            payload=json.dumps(start_payload, ensure_ascii=False).encode("utf-8"),
        ))
        msg = await _recv(ws, timeout)
        if msg.event != Event.SessionStarted:
            raise VolcengineServerError(
                f"expected SessionStarted, got {msg.event.name}",
                payload=msg.json_payload() or msg.payload,
            )

        # 3. TaskRequest (carries the text)
        task_payload = dict(base_payload, event=int(Event.TaskRequest))
        task_payload["req_params"] = dict(req_params, text=text)
        await _send(ws, Message(
            type=MsgType.FullClientRequest, flag=Flags.WithEvent,
            event=Event.TaskRequest, session_id=session_id,
            payload=json.dumps(task_payload, ensure_ascii=False).encode("utf-8"),
        ))

        # 3b. Immediately signal FinishSession so the server knows no more text
        # is coming and should finalize the audio stream. (Bidirectional TTS
        # otherwise keeps the session open waiting for more TaskRequests.)
        await _send(ws, Message(
            type=MsgType.FullClientRequest, flag=Flags.WithEvent,
            event=Event.FinishSession, session_id=session_id, payload=b"{}",
        ))

        # 4. Receive loop: audio chunks until TTSEnded / SessionFinished
        # NB: Seed-TTS 2.0 often signals completion via SessionFinished(152) alone,
        # without sending TTSEnded(359) first. Accept either as terminal.
        session_done = False
        while True:
            msg = await _recv(ws, timeout)
            if msg.type == MsgType.AudioOnlyServer:
                if msg.payload:
                    yield msg.payload
            elif msg.type == MsgType.FullServerResponse:
                if msg.event in (Event.TTSEnded, Event.SessionFinished):
                    logger.debug("tts: stream complete (%s)", msg.event.name)
                    if msg.event == Event.SessionFinished:
                        session_done = True
                    break
                if msg.event in (Event.TTSResponse, Event.TTSSentenceStart, Event.TTSSentenceEnd):
                    continue  # progress / metadata frames
                logger.debug("tts: unhandled server event %s", msg.event.name)
            else:
                logger.debug("tts: unexpected msg type %s", msg.type.name)

        # 5. FinishSession (skip if server already sent SessionFinished) / FinishConnection
        if not session_done:
            await _send(ws, Message(
                type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                event=Event.FinishSession, session_id=session_id, payload=b"{}",
            ))
            try:
                msg = await _recv(ws, timeout)
                if msg.event != Event.SessionFinished:
                    logger.debug("tts: expected SessionFinished, got %s", msg.event.name)
            except VolcengineTimeoutError:
                logger.warning("tts: SessionFinished not received in time")

        await _send(ws, Message(
            type=MsgType.FullClientRequest, flag=Flags.WithEvent,
            event=Event.FinishConnection, payload=b"{}",
        ))
        try:
            msg = await _recv(ws, timeout)
            if msg.event != Event.ConnectionFinished:
                logger.debug("tts: expected ConnectionFinished, got %s", msg.event.name)
        except VolcengineTimeoutError:
            logger.warning("tts: ConnectionFinished not received in time")
    finally:
        try:
            await ws.close()
        except WebSocketException:
            pass


async def tts_to_file(text: str, output_path: str | Path, **kwargs) -> Path:
    """Convenience wrapper: stream TTS and write every chunk to a file.

    Returns the resolved output Path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    # Open in binary append-less mode; we overwrite any prior content.
    try:
        with open(out, "wb") as f:
            async for chunk in tts_stream(text, **kwargs):
                f.write(chunk)
                bytes_written += len(chunk)
    except Exception:
        try:
            out.unlink()
        except OSError:
            pass
        raise
    if bytes_written <= 0:
        try:
            out.unlink()
        except OSError:
            pass
        raise VolcengineVoiceError("tts returned no audio data")
    logger.info("tts: wrote %d bytes to %s", bytes_written, out)
    return out


# ---------------------------------------------------------------------------
# Audio source handling (for ASR)
# ---------------------------------------------------------------------------
async def _audio_chunks_from_source(
    source: "AsyncIterator[bytes] | bytes | str | Path",
    *,
    audio_format: str,
    segment_bytes: int,
    sample_rate: int,
) -> AsyncIterator[bytes]:
    """Yield audio chunks of size ``segment_bytes`` from many source shapes."""

    # Case 1: already an async iterator.
    if hasattr(source, "__aiter__"):
        buf = bytearray()
        async for chunk in source:  # type: ignore[union-attr]
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= segment_bytes:
                yield bytes(buf[:segment_bytes])
                del buf[:segment_bytes]
        if buf:
            yield bytes(buf)
        return

    # Case 2: raw bytes in memory.
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        for i in range(0, len(data), segment_bytes):
            yield data[i:i + segment_bytes]
        return

    # Case 3: file path.
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = audio_format.lower()
    if fmt == "wav" or path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != sample_rate:
                    # Re-encode via ffmpeg.
                    raw = await _ffmpeg_to_pcm16(path, sample_rate)
                else:
                    raw = w.readframes(w.getnframes())
        except wave.Error:
            raw = await _ffmpeg_to_pcm16(path, sample_rate)
        for i in range(0, len(raw), segment_bytes):
            yield raw[i:i + segment_bytes]
        return

    if fmt == "pcm":
        raw = path.read_bytes()
        for i in range(0, len(raw), segment_bytes):
            yield raw[i:i + segment_bytes]
        return

    # Case 4: other formats -> ffmpeg transcode to PCM16 mono.
    raw = await _ffmpeg_to_pcm16(path, sample_rate)
    for i in range(0, len(raw), segment_bytes):
        yield raw[i:i + segment_bytes]


async def _ffmpeg_to_pcm16(path: Path, sample_rate: int) -> bytes:
    """Transcode any audio file to raw PCM16 mono at the given sample rate."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-y", "-i", str(path),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise VolcengineParamError(
            f"ffmpeg failed ({proc.returncode}): {stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return stdout


# ---------------------------------------------------------------------------
# ASR streaming
# ---------------------------------------------------------------------------
async def asr_stream(
    audio_source: "AsyncIterator[bytes] | bytes | str | Path",
    *,
    audio_format: str = "wav",
    codec: str = "raw",
    sample_rate: int = 16000,
    bits: int = 16,
    channel: int = 1,
    model_name: str = "bigmodel",
    enable_itn: bool = True,
    enable_punc: bool = True,
    enable_ddc: bool = True,
    show_utterances: bool = True,
    segment_duration_ms: int = 200,
    app_id: str | None = None,
    access_token: str | None = None,
    resource_id: str = "volc.bigasr.sauc.duration",
    request_id: str | None = None,
    timeout: float = 30.0,
) -> AsyncIterator[dict]:
    """Stream ASR recognition. Yields ``{text, is_final, utterances}`` dicts.

    The function concurrently pumps audio out and reads results in, so the
    consumer starts seeing partial transcripts as soon as the server emits
    them. Terminates when ``is_last_package`` is set, ``is_final`` is True,
    or the socket closes cleanly.
    """
    app_id, access_token = _resolve_credentials(app_id, access_token)
    headers = _build_headers(app_id, access_token, resource_id, request_id)

    # When the source is a file or bytes, our chunk generator decodes it into
    # raw PCM16 mono at `sample_rate`. The wire-level `format` we advertise to
    # the server must reflect *what we actually send*, not the source format.
    # Only async-iterator callers keep the user-supplied format/codec verbatim.
    wire_format = audio_format
    wire_codec = codec
    if not hasattr(audio_source, "__aiter__"):
        wire_format = "pcm"
        wire_codec = "raw"

    # Derive segment byte size from sample_rate/bits/channel and duration.
    segment_bytes = max(1, sample_rate * (bits // 8) * channel * segment_duration_ms // 1000)

    init_payload = {
        "user": {"uid": headers["X-Api-Request-Id"]},
        "audio": {
            "format": wire_format,
            "codec": wire_codec,
            "rate": sample_rate,
            "bits": bits,
            "channel": channel,
        },
        "request": {
            "model_name": model_name,
            "enable_itn": enable_itn,
            "enable_punc": enable_punc,
            "enable_ddc": enable_ddc,
            "show_utterances": show_utterances,
            "enable_nonstream": False,
        },
    }

    logger.info("asr: connecting (model=%s, rate=%d, wire_format=%s)", model_name, sample_rate, wire_format)
    try:
        ws = await asyncio.wait_for(
            websockets.connect(
                ASR_URL,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                open_timeout=timeout,
                ping_interval=20,
                ping_timeout=20,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        raise VolcengineTimeoutError(f"asr connect timed out ({timeout}s)") from e
    except WebSocketException as e:
        text_ = str(e)
        if "401" in text_ or "403" in text_ or "auth" in text_.lower():
            raise VolcengineAuthError(f"asr handshake rejected: {e}") from e
        raise VolcengineVoiceError(f"asr handshake failed: {e}") from e
    except OSError as e:
        raise VolcengineVoiceError(f"asr connection failed: {e}") from e

    seq = 1
    init_bytes = gzip.compress(json.dumps(init_payload, ensure_ascii=False).encode("utf-8"))
    init_msg = Message(
        type=MsgType.FullClientRequest,
        flag=Flags.PositiveSeq,
        serialization=Serialization.JSON,
        compression=Compression.Gzip,
        sequence=seq,
        payload=init_bytes,
    )

    recv_queue: asyncio.Queue[dict | Exception | None] = asyncio.Queue()

    async def receiver() -> None:
        try:
            while True:
                msg = await _recv(ws, timeout)
                payload = msg.json_payload()
                if msg.type == MsgType.FullServerResponse and isinstance(payload, dict):
                    result = payload.get("result")
                    if isinstance(result, dict):
                        out = {
                            "text": result.get("text", ""),
                            "is_final": bool(result.get("is_final", False)) or bool(msg.flag & Flags.LastNoSeq),
                            "utterances": result.get("utterances", []) or [],
                            "raw": payload,
                        }
                        await recv_queue.put(out)
                        if out["is_final"] or (msg.flag & Flags.LastNoSeq):
                            break
                    elif msg.flag & Flags.LastNoSeq:
                        break
                elif msg.type == MsgType.Error:
                    # _recv already raises; unreachable
                    break
        except (VolcengineVoiceError, WebSocketException, asyncio.TimeoutError) as e:
            await recv_queue.put(e if isinstance(e, Exception) else Exception(str(e)))
        except Exception as e:
            await recv_queue.put(VolcengineVoiceError(f"asr receiver failed: {e}"))
        finally:
            await recv_queue.put(None)

    async def sender() -> None:
        nonlocal seq
        try:
            await _send(ws, init_msg)
            seq += 1

            # Buffer one chunk ahead so we can flag the *last* one correctly.
            prev: bytes | None = None
            async for chunk in _audio_chunks_from_source(
                audio_source,
                audio_format=audio_format,
                segment_bytes=segment_bytes,
                sample_rate=sample_rate,
            ):
                if prev is not None:
                    await _send(ws, Message(
                        type=MsgType.AudioOnlyClient,
                        flag=Flags.PositiveSeq,
                        serialization=Serialization.JSON,
                        compression=Compression.Gzip,
                        sequence=seq,
                        payload=gzip.compress(prev),
                    ))
                    seq += 1
                prev = chunk

            # Final chunk (empty if source was empty)
            final_bytes = prev if prev is not None else b""
            await _send(ws, Message(
                type=MsgType.AudioOnlyClient,
                flag=Flags.NegativeSeq,
                serialization=Serialization.JSON,
                compression=Compression.Gzip,
                sequence=-seq,
                payload=gzip.compress(final_bytes),
            ))
        except (VolcengineVoiceError, WebSocketException, asyncio.TimeoutError) as e:
            await recv_queue.put(e if isinstance(e, Exception) else Exception(str(e)))
        except Exception as e:
            await recv_queue.put(VolcengineVoiceError(f"asr sender failed: {e}"))

    recv_task = asyncio.create_task(receiver(), name="volc-asr-recv")
    send_task = asyncio.create_task(sender(), name="volc-asr-send")

    try:
        while True:
            item = await recv_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
            if item.get("is_final"):
                break
    finally:
        for t in (send_task, recv_task):
            if not t.done():
                t.cancel()
        for t in (send_task, recv_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await ws.close()
        except WebSocketException:
            pass


