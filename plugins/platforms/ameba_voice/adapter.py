"""
Hermes Agent voice device platform adapter.

WebSocket gateway adapter for embedded voice devices (Realtek Ameba, etc.).
Relays audio bidirectionally between the device and Hermes Agent.

Protocol (consistent with docs/Ameba-Hermes Cloud Voice Assistant Design §3.3):

    Uplink (device -> server):
        TEXT  {"type":"hello", "device_id":..., "sr":16000,
               "codec":"wav"|"pcm", ["channels":1, "bits":16]}
        TEXT  {"type":"wake",  "keyword":..., "ts":<ms>}
        BIN   <wav | pcm audio chunks>...
        TEXT  {"type":"end_of_speech"}

    Downlink (server -> device):
        TEXT  {"type":"asr_final",  "text":"..."}
        TEXT  {"type":"reply_text",    "text":"...",
               "model":"claude-sonnet-4-6",
               "ctx_used":"12450", "ctx_total":"1000000"}
        TEXT  {"type":"system_notify", "text":"..."}
        TEXT  {"type":"tts_begin",  "codec":"mp3", "size":N}
        BIN   <mp3 audio>...
        TEXT  {"type":"tts_end"}
        TEXT  {"type":"error",     "detail":"..."}

Architecture:
    - Each device maps to one WebSocket connection. device_id doubles as
      Hermes chat_id and user_id; session, history, skills, and memory
      inherit the standard gateway pipeline.
    - The adapter calls transcribe_audio inside _dispatch_utterance (STT),
      sends back {"type":"asr_final","text":"..."} on success, and writes
      the transcript into MessageEvent.text (message_type -> TEXT) to skip
      re-transcription in run.py. On STT failure it falls back to
      MessageType.VOICE, using run.py's error path (shared with Telegram /
      Discord voice messages).
    - The adapter does NOT call text_to_speech_tool. It overrides play_tts
      and send_voice: when the base class auto-TTS pipeline
      (BasePlatformAdapter._process_message) produces the agent reply as an
      mp3 file, the adapter streams the bytes over WebSocket, bracketed by
      tts_begin / tts_end JSON frames.

``~/.hermes/config.yaml`` example::

    gateway:
      platforms:
        voice_device:
          enabled: true
          extra:
            host: 0.0.0.0
            port: 8080
            path: /v1/voice/ws
            allowed_devices:
              - "ameba-amebalite-001"
              - "ameba-amebasmart-001"
            allow_all: false

    voice:
      auto_tts: true            # Required — replies only return as audio when enabled
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import json
import logging
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Strip 4-byte emoji and common BMP misc symbols — embedded device fonts
# don't support them. Used for tool_status messages sent to the device.
_EMOJI4_RE     = re.compile(r'[\U00010000-\U0010FFFF]+')
_MISC_SYM_RE   = re.compile(r'[☀-➿⌀-⏿]+')
_MULTI_SPC_RE  = re.compile(r'  +')

def _clean_tool_text(s: str) -> str:
    """Strip emoji, leading ``┊``, and excess whitespace for embedded display."""
    s = s.lstrip('┊ \t')
    s = _EMOJI4_RE.sub('', s)
    s = _MISC_SYM_RE.sub('', s)
    s = _MULTI_SPC_RE.sub(' ', s)
    return s.strip()


def _to_simplified(text: str) -> str:
    """Convert traditional Chinese to simplified; returns original if zhconv unavailable."""
    try:
        import zhconv
        return zhconv.convert(text, "zh-hans")
    except ImportError:
        return text


from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.config import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Track utterance epoch per connection/task chain so stale output from a
# pre-interrupt task can be identified and dropped.
utterance_epoch_var: contextvars.ContextVar[int] = contextvars.ContextVar("utterance_epoch", default=0)


def _wrap_pcm_in_wav(pcm: bytes, sr: int, channels: int, bits: int) -> bytes:
    """Prepend a RIFF/WAVE header to raw little-endian PCM bytes."""
    byte_rate   = sr * channels * bits // 8
    block_align = channels * bits // 8
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sr,
                                  byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", len(pcm))
    )
    return header + pcm


def _bool_env(name: str) -> bool:
    """Parse an env var as boolean (accepts 1/true/yes/on)."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Voice Device Adapter
# ---------------------------------------------------------------------------

def _resolve_platform() -> Platform:
    """Return the voice_device Platform enum member.

    Platform("voice_device") only works after platform_registry is fully
    populated, but adapter imports may fire earlier (e.g. during test suite
    discovery or plugin-manager preloading). If standard lookup fails, build
    a pseudo-member manually so startup doesn't crash due to import order.
    """
    try:
        return Platform("voice_device")
    except ValueError:
        pseudo = object.__new__(Platform)
        pseudo._value_ = "voice_device"
        pseudo._name_ = "VOICE_DEVICE"
        Platform._value2member_map_["voice_device"] = pseudo
        Platform._member_map_["VOICE_DEVICE"] = pseudo
        return pseudo


class VoiceDeviceAdapter(BasePlatformAdapter):
    """WebSocket platform adapter for embedded voice devices."""

    def __init__(self, config, **kwargs):
        logger.warning("voice_device: VoiceDeviceAdapter.__init__ entered "
                       "(extra=%s)",
                       getattr(config, "extra", None))
        platform = _resolve_platform()
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self.host: str = os.getenv("VOICE_DEVICE_HOST") or extra.get("host", "0.0.0.0")
        self.port: int = int(os.getenv("VOICE_DEVICE_PORT") or extra.get("port", 8080))
        self.ws_path: str = (
            os.getenv("VOICE_DEVICE_PATH") or extra.get("path", "/v1/voice/ws")
        )

        # Device whitelist (modelled after IRC platform's allowed_users / allow_all_users)
        env_allow = os.getenv("VOICE_DEVICE_ALLOWED_DEVICES", "")
        cfg_allow = extra.get("allowed_devices", []) or []
        if env_allow:
            cfg_allow = [s.strip() for s in env_allow.split(",") if s.strip()]
        self._allowed_devices: set = {str(d) for d in cfg_allow}
        self._allow_all_devices: bool = (
            _bool_env("VOICE_DEVICE_ALLOW_ALL")
            if os.getenv("VOICE_DEVICE_ALLOW_ALL")
            else bool(extra.get("allow_all", False))
        )

        self._connections: Dict[str, Any] = {}
        # One send lock per connection — prevents interleaving between
        # concurrent send_voice / send / send_text calls on the same socket.
        self._send_locks: Dict[str, asyncio.Lock] = {}

        # Devices that have sent {"type":"interrupt"}. Stale replies from the
        # agent (play_tts / send) are silently dropped until the next
        # end_of_speech clears this flag.
        self._interrupted: set = set()

        self._device_epochs: Dict[str, int] = {}

        self._ws_server = None

        # HTTP file server for TTS audio distribution
        # http_host must be reachable by the device. If ws host != 0.0.0.0 it
        # is inherited automatically; otherwise configure extra.http_host.
        _ws_host = self.host
        self.http_host: str = (
            os.getenv("VOICE_DEVICE_HTTP_HOST")
            or extra.get("http_host")
            or (_ws_host if _ws_host not in ("0.0.0.0", "") else "")
        )
        self.http_port: int = int(
            os.getenv("VOICE_DEVICE_HTTP_PORT")
            or extra.get("http_port")
            or (self.port + 1)
        )
        self._http_dir: str = ""
        self._http_server = None
        self._http_thread: Optional[threading.Thread] = None

    # Connection lifecycle

    async def connect(self) -> bool:
        logger.warning("voice_device: connect() entered host=%s port=%s path=%s",
                       self.host, self.port, self.ws_path)
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error("voice_device: 'websockets' 包未安装，"
                         "请执行 `pip install websockets`")
            self._set_fatal_error(
                "missing_dep",
                "websockets package not installed",
                retryable=False,
            )
            return False

        try:
            from websockets.asyncio.server import serve  # websockets >= 13
        except ImportError:
            try:
                # Fallback to legacy serve interface for websockets <= 12
                from websockets import serve  # type: ignore
            except ImportError as e:
                logger.error("voice_device: 无法找到 websockets.serve: %s", e)
                self._set_fatal_error("missing_dep", str(e), retryable=False)
                return False

        try:
            self._ws_server = await serve(
                self._on_connection, self.host, self.port,
                max_size=8 * 1024 * 1024,
                ping_interval=60, ping_timeout=20,
            )
        except Exception as e:
            logger.error("voice_device: WebSocket 服务器绑定 %s:%d 失败 — %s",
                         self.host, self.port, e)
            self._set_fatal_error("connect_failed", str(e), retryable=True)
            return False

        if not self.http_host:
            logger.error("voice_device: http_host 未配置。"
                         "请在 extra.http_host 中填写设备能访问的服务器 IP。")
            self._set_fatal_error("config", "http_host not set", retryable=False)
            return False

        self._http_dir = tempfile.mkdtemp(prefix="voice_tts_http_")
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        import functools as _functools

        _dir = self._http_dir
        class _Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=_dir, **kw)
            def log_message(self, *a):
                pass

        self._http_server = ThreadingHTTPServer(("0.0.0.0", self.http_port), _Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="voice-tts-http",
        )
        self._http_thread.start()
        logger.warning("voice_device: ✓ TTS HTTP 服务已启动，设备访问 http://%s:%d/",
                       self.http_host, self.http_port)

        self._mark_connected()
        logger.warning("voice_device: ✓ 已监听 ws://%s:%d%s",
                       self.host, self.port, self.ws_path)

        # Pre-warm STT model in background so the first _transcribe_and_notify
        # call doesn't suffer a 60-180s cold-start.
        asyncio.ensure_future(self._warmup_stt())
        return True

    async def _warmup_stt(self) -> None:
        """Load the STT model into memory in the background to avoid cold-start latency."""
        try:
            import wave, struct, tempfile, os as _os
            # Generate 0.5s of 16kHz mono silence — enough to trigger model loading
            samples = b'\x00\x00' * 8000
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                warmup_path = f.name
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(samples)
            t0 = __import__('time').time()
            from tools.transcription_tools import transcribe_audio
            await asyncio.to_thread(transcribe_audio, warmup_path)
            logger.warning("voice_device: ✓ STT 预热完成（%.1fs）", __import__('time').time() - t0)
            try:
                _os.unlink(warmup_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning("voice_device: STT 预热失败（首次对话仍会冷启动）: %s", e)

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for chat_id, ws in list(self._connections.items()):
            try:
                await ws.close(code=1001, reason="server shutting down")
            except Exception:
                logger.debug("voice_device: error closing ws for %s during disconnect", chat_id, exc_info=True)
        self._connections.clear()
        self._send_locks.clear()
        self._interrupted.clear()
        self._device_epochs.clear()
        if self._ws_server is not None:
            try:
                self._ws_server.close()
                if hasattr(self._ws_server, "wait_closed"):
                    await self._ws_server.wait_closed()
            except Exception:
                logger.debug("voice_device: error closing ws_server", exc_info=True)
            self._ws_server = None
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
        if self._http_dir:
            shutil.rmtree(self._http_dir, ignore_errors=True)
            self._http_dir = ""

    # Session key helpers

    def _device_session_key(self, device_id: str) -> str:
        """Return the session_key for this device in BasePlatformAdapter.

        Equivalent to calling gateway.session.build_session_key() with
        voice_device platform, chat_type="dm".
        """
        return f"agent:main:{_resolve_platform().value}:dm:{device_id}"

    # Uplink: per-connection driver loop

    async def _on_connection(self, ws) -> None:
        """Accept one device, drive its session, dispatch voice events.

        This coroutine lives for the full connection lifetime. It reads
        incoming frames and groups each "wake → binary chunks → end_of_speech"
        sequence into a single MessageEvent. The connection is reused across
        turns; the device does not need to reconnect.
        """
        path = (
            getattr(getattr(ws, "request", None), "path", None)
            or getattr(ws, "path", "")
        )
        if path != self.ws_path:
            logger.warning("voice_device: 拒绝非法路径: %s", path)
            try:
                await ws.close(code=1008, reason="bad path")
            except Exception:
                logger.debug("voice_device: error closing bad-path connection from %s", path, exc_info=True)
            return

        peer = self._format_peer(ws)
        device_id: Optional[str] = None
        hello_meta: Dict[str, Any] = {}
        audio_buf = bytearray()

        # Send init_info immediately upon connection (before hello arrives),
        # so the device's top bar and version display update as soon as it
        # connects, well before the user speaks a wake word.
        await self._send_model_init(peer, ws)
        asyncio.ensure_future(self._send_tools_skills(peer, ws))

        try:
            async for raw in ws:
                if isinstance(raw, str):
                    msg = self._parse_text_frame(raw, peer)
                    if msg is None:
                        continue
                    mtype = msg.get("type")

                    if mtype == "hello":
                        hello_meta = dict(msg)
                        device_id = self._authorize_hello(hello_meta, ws, peer)
                        if device_id is None:
                            return
                        self._connections[device_id] = ws
                        self._send_locks.setdefault(device_id, asyncio.Lock())
                        logger.info("voice_device: hello from %s (peer=%s)",
                                    device_id, peer)

                    elif mtype == "wake":
                        audio_buf.clear()
                        logger.debug("voice_device: wake %s ts=%s",
                                     device_id, msg.get("ts"))

                    elif mtype == "interrupt":
                        # Two-layer abort strategy:
                        # 1. cancel_session_processing() cancels the asyncio
                        #    Task holding the STT/LLM/TTS pipeline.
                        # 2. _interrupted flag serves as a safety net for the
                        #    race window before cancellation takes effect,
                        #    filtering stale output in _stream_audio_to_device
                        #    and send().
                        if device_id:
                            self._interrupted.add(device_id)
                            session_key = self._device_session_key(device_id)
                            asyncio.ensure_future(
                                self.cancel_session_processing(session_key)
                            )
                            try:
                                await ws.send(json.dumps(
                                    {"type": "thinking",
                                     "text": "[Request interrupted by user]"},
                                    ensure_ascii=False,
                                ))
                            except Exception:
                                logger.debug("voice_device: [%s] failed to send interrupt notification", device_id, exc_info=True)
                        audio_buf.clear()
                        logger.info("voice_device: [%s] interrupt -- "
                                    "agent 会话已取消", device_id)

                    elif mtype == "end_of_speech":
                        if not device_id:
                            await self._send_error(ws, "hello missing")
                            continue
                        if not audio_buf:
                            await self._send_error(ws, "empty audio")
                            audio_buf.clear()
                            continue
                        # New utterance starts — lift interrupt suppression so
                        # this turn's replies can flow through.
                        self._interrupted.discard(device_id)
                        await self._dispatch_utterance(
                            device_id, hello_meta, bytes(audio_buf)
                        )
                        audio_buf.clear()

                    else:
                        logger.debug("voice_device: 忽略未知 text type=%s",
                                     mtype)
                else:
                    audio_buf.extend(raw)

        except Exception as _exc:
            # ConnectionClosedError is normal when embedded WS clients drop
            # without a close frame — log at INFO, not ERROR.
            try:
                from websockets.exceptions import ConnectionClosedError as _CCE
                _is_normal_close = isinstance(_exc, _CCE)
            except ImportError:
                _is_normal_close = type(_exc).__name__ == "ConnectionClosedError"
            if _is_normal_close:
                logger.info("voice_device: 设备断开（无 close frame）peer=%s", peer)
            else:
                logger.exception("voice_device: 连接驱动崩溃 (peer=%s)", peer)
        finally:
            if device_id and self._connections.get(device_id) is ws:
                self._connections.pop(device_id, None)
                self._send_locks.pop(device_id, None)
            logger.info("voice_device: 设备断开 device=%s peer=%s",
                        device_id, peer)

    # MessageEvent dispatch

    async def _dispatch_utterance(
        self,
        device_id: str,
        hello: Dict[str, Any],
        audio_bytes: bytes,
    ) -> None:
        """Write utterance to temp WAV, run STT, and forward to gateway pipeline."""
        if not self._message_handler:
            logger.warning("voice_device: 未绑定消息处理器，丢弃 %s 的话语",
                           device_id)
            return

        # Advance epoch so stale output from a prior interrupted turn is
        # identifiable and dropped.
        current_epoch = self._device_epochs.get(device_id, 0) + 1
        self._device_epochs[device_id] = current_epoch
        utterance_epoch_var.set(current_epoch)

        codec = str(hello.get("codec", "wav")).lower()
        if codec == "pcm":
            sr       = int(hello.get("sr", 16000))
            # Device firmware sends "ch" (M5-style); also accept "channels"
            channels = int(hello.get("ch") or hello.get("channels") or 1)
            bits     = int(hello.get("bits", 16))
            wav_bytes = _wrap_pcm_in_wav(audio_bytes, sr, channels, bits)
        else:
            wav_bytes = audio_bytes

        # Write to temp file — transcribe_audio needs a path for faster-whisper etc.
        with tempfile.NamedTemporaryFile(
            prefix=f"voice_{device_id}_",
            suffix=".wav",
            delete=False,
        ) as f:
            f.write(wav_bytes)
            wav_path = f.name

        # Run STT first: on success send asr_final back to device and use
        # TEXT message_type to skip re-transcription in run.py; on failure
        # fall back to VOICE type so run.py's error path handles it.
        transcript = await self._transcribe_and_notify(device_id, wav_path)

        source = self.build_source(
            chat_id=device_id,
            chat_name=device_id,
            chat_type="dm",
            user_id=device_id,
            user_name=hello.get("device_name") or f"voice-{device_id}",
        )
        if transcript:
            event = MessageEvent(
                text=f'[The user sent a voice message~ Here\'s what they said: "{transcript}"]',
                message_type=MessageType.TEXT,
                source=source,
                message_id=str(int(time.time() * 1000)),
                media_urls=[wav_path],
                media_types=["voice"],
                timestamp=datetime.datetime.now(),
            )
        else:
            event = MessageEvent(
                text="",
                message_type=MessageType.VOICE,
                source=source,
                message_id=str(int(time.time() * 1000)),
                media_urls=[wav_path],
                media_types=["voice"],
                timestamp=datetime.datetime.now(),
            )
        await self.handle_message(event)
        # Inject thinking_callback after the agent appears in _running_agents
        # so LLM thinking states are streamed to the device in real time.
        asyncio.ensure_future(
            self._inject_thinking_callback(
                device_id, self._device_session_key(device_id)
            )
        )

    # STT + asr_final notification

    async def _transcribe_and_notify(self, device_id: str, wav_path: str) -> str:
        """Transcribe wav_path via STT and send asr_final frame to device on success.

        Returns the transcript on success, or empty string on failure.
        Caller uses the return to pick MessageType: non-empty -> TEXT (skip
        run.py re-transcription), empty -> VOICE (run.py error path).
        """
        try:
            from tools.transcription_tools import transcribe_audio
            result = await asyncio.to_thread(transcribe_audio, wav_path)
        except Exception:
            logger.exception("voice_device: [%s] STT 调用异常", device_id)
            return ""

        if not result.get("success"):
            logger.warning("voice_device: [%s] STT 失败: %s",
                           device_id, result.get("error", "unknown"))
            return ""

        transcript = (result.get("transcript") or "").strip()
        if not transcript:
            return ""

        # Convert traditional to simplified — Whisper may output traditional glyphs
        try:
            import zhconv
            transcript = zhconv.convert(transcript, "zh-hans")
        except ImportError:
            pass

        ws = self._connections.get(device_id)
        if ws:
            try:
                await ws.send(json.dumps(
                    {"type": "asr_final", "text": transcript},
                    ensure_ascii=False,
                ))
                logger.info("voice_device: [%s] asr_final: %r", device_id, transcript)
            except Exception:
                logger.warning("voice_device: [%s] 发送 asr_final 失败", device_id)

        return transcript

    # Downlink: TTS audio push

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        """Voice devices have always-on speakers — stream TTS in real time."""
        caption = kwargs.get("caption")
        return await self._stream_audio_to_device(chat_id, audio_path, caption=caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Voice messages and auto-TTS share the same path: push mp3/opus as binary frames."""
        return await self._stream_audio_to_device(chat_id, audio_path,
                                                  caption=caption)

    async def _stream_audio_to_device(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        if chat_id in self._interrupted:
            logger.info("voice_device: [%s] 打断后丢弃陈旧 TTS", chat_id)
            return SendResult(success=False, error="stale: device interrupted")

        task_epoch = utterance_epoch_var.get()
        latest_epoch = self._device_epochs.get(chat_id, 0)
        if 0 < task_epoch < latest_epoch:
            logger.info("voice_device: [%s] 丢弃旧世代 TTS (task_epoch=%d < latest_epoch=%d)",
                        chat_id, task_epoch, latest_epoch)
            return SendResult(success=False, error="stale: older generation TTS")

        ws = self._connections.get(chat_id)
        if ws is None:
            return SendResult(success=False, error=f"device {chat_id} offline")
        if not os.path.isfile(audio_path):
            return SendResult(success=False, error=f"missing audio: {audio_path}")
        if not self._http_dir:
            return SendResult(success=False, error="HTTP file service not running")

        codec = (os.path.splitext(audio_path)[1].lstrip(".") or "mp3").lower()
        filename = f"{uuid.uuid4().hex}.{codec}"
        dest_path = os.path.join(self._http_dir, filename)
        url = f"http://{self.http_host}:{self.http_port}/{filename}"
        msg_id = uuid.uuid4().hex[:12]

        try:
            await asyncio.to_thread(shutil.copy2, audio_path, dest_path)
        except Exception as e:
            return SendResult(success=False, error=f"copy failed: {e}")

        async with self._send_lock(chat_id):
            try:
                if caption:
                    reply_payload = {"type": "reply_text",
                                     "text": _to_simplified(caption)}
                    meta = self._get_session_meta(chat_id)
                    if meta:
                        reply_payload.update(meta)
                    await ws.send(json.dumps(reply_payload, ensure_ascii=False))
                await ws.send(json.dumps(
                    {"type": "tts_begin", "codec": codec, "url": url},
                    ensure_ascii=False,
                ))
                await ws.send(json.dumps({"type": "tts_end"}))
            except Exception as e:
                logger.exception("voice_device: 向 %s 发送 TTS URL 失败", chat_id)
                try:
                    os.unlink(dest_path)
                except OSError:
                    pass
                return SendResult(success=False, error=str(e), retryable=True)

        # Clean up temp file after 5 minutes
        def _deferred_delete(path: str) -> None:
            time.sleep(300)
            try:
                os.unlink(path)
            except OSError:
                pass

        threading.Thread(target=_deferred_delete, args=(dest_path,),
                         daemon=True, name="voice-tts-gc").start()
        return SendResult(success=True, message_id=msg_id)

    # Downlink: text replies

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text to the device — used for both LLM replies and system notifications.

        Distinction: base._process_message_background injects "notify": True
        into metadata for LLM replies; system notifications lack this key.
          - LLM reply  → type="reply_text" with model/ctx metadata
          - Notification → type="system_notify" (device may ignore)
        """
        if chat_id in self._interrupted:
            logger.info("voice_device: [%s] 打断后丢弃陈旧文本回复", chat_id)
            return SendResult(success=False, error="stale: device interrupted")

        task_epoch = utterance_epoch_var.get()
        latest_epoch = self._device_epochs.get(chat_id, 0)
        if 0 < task_epoch < latest_epoch:
            logger.info("voice_device: [%s] 丢弃旧世代文本 (task_epoch=%d < latest_epoch=%d)",
                        chat_id, task_epoch, latest_epoch)
            return SendResult(success=False, error="stale: older generation text")

        ws = self._connections.get(chat_id)
        if ws is None:
            return SendResult(success=False, error=f"device {chat_id} offline")

        msg_id = uuid.uuid4().hex[:12]
        content = _to_simplified(content)

        # metadata["notify"]=True is injected exclusively by
        # base._process_message_background for LLM replies; system
        # notifications never carry this key.
        is_llm_reply = bool(metadata and metadata.get("notify"))
        if is_llm_reply:
            reply_payload: Dict[str, Any] = {"type": "reply_text", "text": content}
            session_meta = self._get_session_meta(chat_id)
            if session_meta:
                reply_payload.update(session_meta)
        else:
            reply_payload = {"type": "system_notify", "text": content}
            logger.debug("voice_device: [%s] system_notify: %r", chat_id, content[:80])

        async with self._send_lock(chat_id):
            try:
                await ws.send(json.dumps(reply_payload, ensure_ascii=False))
            except Exception as e:
                logger.exception("voice_device: 向 %s 发送文本失败", chat_id)
                return SendResult(success=False, error=str(e), retryable=True)
        return SendResult(success=True, message_id=msg_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"chat_id": chat_id, "online": chat_id in self._connections}

    # Internal helpers

    async def _send_model_init(self, device_id: str, ws) -> None:
        """Send init_info frame with model name and version string immediately after connect."""
        try:
            model = self._get_default_model()
            if not model:
                logger.warning("voice_device: [%s] init_info skipped: no model", device_id)
                return
            # format_banner_version_label calls git subprocess — offload to thread
            version = ""
            try:
                from hermes_cli.banner import format_banner_version_label
                version = await asyncio.to_thread(format_banner_version_label)
            except Exception:
                logger.warning("voice_device: [%s] version label failed", device_id, exc_info=True)
            payload: Dict[str, Any] = {"type": "init_info", "model": model}
            if version:
                payload["version"] = version
            await ws.send(json.dumps(payload, ensure_ascii=False))
            logger.warning("voice_device: [%s] init_info model=%s version=%r",
                           device_id, model, version)
        except Exception:
            logger.warning("voice_device: [%s] _send_model_init failed", device_id, exc_info=True)

    def _get_session_meta(self, device_id: str) -> Dict[str, str]:
        """Look up the agent for this device's session and collect metadata.

        Returns a dict that may contain model / ctx_used / ctx_total as
        string fields (device uses json_get_str). Returns empty dict when
        the agent is unavailable. Each field is independently present or
        absent:
          - model is always included when the agent exists
          - ctx_used/ctx_total are omitted when context upper bound is unknown

        The agent lives in _running_agents during processing and falls to
        _agent_cache after completion—both locations are checked.
        """
        handler = getattr(self, "_message_handler", None)
        runner = getattr(handler, "__self__", None)
        if runner is None:
            return {}
        session_key = self._device_session_key(device_id)

        # Check _running_agents first; sentinel placeholders (bare object())
        # have no context_compressor and will get None below.
        agent = getattr(runner, "_running_agents", {}).get(session_key)
        if getattr(agent, "context_compressor", None) is None:
            cache = getattr(runner, "_agent_cache", None)
            lock = getattr(runner, "_agent_cache_lock", None)
            if cache is not None and lock is not None:
                with lock:
                    cached = cache.get(session_key)
                if cached:
                    agent = cached[0] if isinstance(cached, tuple) else cached
        if agent is None or not hasattr(agent, "context_compressor"):
            return {}

        meta: Dict[str, str] = {}

        model = str(getattr(agent, "model", "") or "")
        if model:
            meta["model"] = model

        cc = getattr(agent, "context_compressor", None)
        total = int(getattr(cc, "context_length", 0) or 0) if cc else 0
        if total > 0:
            meta["ctx_used"] = str(int(getattr(cc, "last_prompt_tokens", 0) or 0))
            meta["ctx_total"] = str(total)

        return meta

    def _get_tools_skills_payload(self) -> Optional[Dict[str, Any]]:
        """Build tools_skills payload matching the CLI banner format:
        - Tools: grouped by toolset, pipe-separated "toolset: t1, t2|toolset2: t3"
        - Skills: grouped by category, same format, right-aligned ":category"
        """
        tools_lines: List[str] = []
        tool_total = 0
        try:
            from model_tools import get_tool_definitions, get_toolset_for_tool as _gts
            from hermes_cli.banner import _display_toolset_name
            raw_tools = get_tool_definitions(quiet_mode=True)
            tool_total = len(raw_tools)
            toolsets: Dict[str, list] = {}
            for t in raw_tools:
                name = t["function"]["name"]
                ts = _display_toolset_name(_gts(name) or "other")
                toolsets.setdefault(ts, []).append(name)
            for ts in sorted(toolsets.keys()):
                tools_lines.append(ts)
        except Exception:
            logger.warning("voice_device: tools grouping failed", exc_info=True)

        skills_lines: List[str] = []
        skill_total = 0
        try:
            from hermes_cli.banner import get_available_skills
            skills_by_cat = get_available_skills()
            skill_total = sum(len(v) for v in skills_by_cat.values())
            for cat in sorted(skills_by_cat.keys()):
                skills_lines.append(cat)
        except Exception:
            logger.warning("voice_device: skills grouping failed", exc_info=True)

        if not tools_lines and not skills_lines:
            return None

        return {
            "type":         "tools_skills",
            "tools":        "|".join(tools_lines),
            "tools_total":  str(tool_total),
            "skills":       "|".join(skills_lines),
            "skills_total": str(skill_total),
        }

    async def _send_tools_skills(self, device_id: str, ws) -> None:
        """Build and send tools_skills message to device."""
        try:
            payload = await asyncio.to_thread(self._get_tools_skills_payload)
            if not payload:
                return
            await ws.send(json.dumps(payload, ensure_ascii=False))
            logger.debug("voice_device: [%s] tools_skills tools=%s skills=%s",
                         device_id,
                         payload["tools_total"],
                         payload["skills_total"])
        except Exception:
            logger.warning("voice_device: [%s] _send_tools_skills failed", device_id, exc_info=True)

    def _get_default_model(self) -> str:
        """Get the default model name for the device's top-bar display.

        Tries runner._model first (loaded runtime value), then falls back to
        gateway.run._resolve_gateway_model() (reads config.yaml). Returns
        empty string when both fail (device still shows "⚕ Hermes").
        """
        handler = getattr(self, "_message_handler", None)
        runner  = getattr(handler, "__self__", None)
        if runner is not None:
            m = getattr(runner, "_model", None)
            if m and isinstance(m, str):
                return m
        try:
            from gateway.run import _resolve_gateway_model
            m = _resolve_gateway_model() or ""
            logger.warning("voice_device: _get_default_model via config: %r", m)
            return m
        except Exception:
            logger.warning("voice_device: _get_default_model failed", exc_info=True)
            return ""

    async def _inject_thinking_callback(
        self, device_id: str, session_key: str
    ) -> None:
        """Wait for the agent to appear in _running_agents, then inject thinking_callback.

        The thinking_callback is invoked by conversation_loop.py before each
        API call with a random face+verb string. Once injected, that string
        is pushed to the device as {"type":"thinking","text":"..."}.
        Polls up to 2.5s (50 × 50ms); silently gives up on timeout.
        """
        handler = getattr(self, "_message_handler", None)
        runner  = getattr(handler, "__self__", None)
        if runner is None:
            return

        for _ in range(50):
            agent = getattr(runner, "_running_agents", {}).get(session_key)
            if agent is not None and hasattr(agent, "thinking_callback"):
                did = device_id
                # Capture the running event loop so _cb (called from agent
                # thread pool workers) can schedule coroutines safely.
                # run_coroutine_threadsafe is the correct API for posting
                # coroutines from a non-async thread to an async loop.
                ev_loop = asyncio.get_running_loop()

                def _make_cb(d: str, loop: asyncio.AbstractEventLoop):
                    def _cb(text: str) -> None:
                        if d in self._interrupted:
                            return
                        ws = self._connections.get(d)
                        if ws is None:
                            return
                        # Device font subset lacks emoji / geometric-shape
                        # Unicode blocks — strip everything before the first
                        # ASCII alpha char to keep just the verb portion.
                        # e.g. "(¬‿¬) ruminating..." → "ruminating..."
                        # Empty string (clear signal) is passed through.
                        display = text
                        if text:
                            for i, ch in enumerate(text):
                                if ch.isascii() and ch.isalpha():
                                    display = text[i:]
                                    break
                        asyncio.run_coroutine_threadsafe(
                            ws.send(
                                json.dumps({"type": "thinking", "text": display},
                                           ensure_ascii=False)
                            ),
                            loop,
                        )
                    return _cb

                agent.thinking_callback = _make_cb(did, ev_loop)

                # Also inject tool_progress_callback for tool execution status
                if hasattr(agent, "tool_progress_callback"):
                    def _make_tool_cb(d: str, loop: asyncio.AbstractEventLoop):
                        pending_args: Dict[str, list] = {}

                        def _tool_cb(event_type: str,
                                     function_name: str = None,
                                     preview: str = None,
                                     function_args: dict = None,
                                     duration: float = 0.0,
                                     is_error: bool = False,
                                     **kw) -> None:
                            if d in self._interrupted:
                                return
                            ws = self._connections.get(d)
                            if ws is None:
                                return

                            if event_type == "tool.started":
                                name = function_name or "tool"
                                # Cache args for retrieval on tool.completed
                                if function_name:
                                    pending_args.setdefault(function_name, []).append(
                                        function_args or {}
                                    )
                                text = f"preparing {name}"
                            elif event_type == "tool.completed":
                                stored = pending_args.get(function_name, [])
                                stored_args = stored.pop(0) if stored else (function_args or {})
                                if function_name in pending_args and not pending_args[function_name]:
                                    del pending_args[function_name]
                                try:
                                    from agent.display import get_cute_tool_message
                                    raw = get_cute_tool_message(
                                        function_name or "",
                                        stored_args,
                                        duration,
                                        result=kw.get("result"),
                                    )
                                    text = _clean_tool_text(raw)
                                except Exception:
                                    logger.debug("voice_device: get_cute_tool_message failed", exc_info=True)
                                    text = f"{function_name or 'tool'}  {duration:.1f}s"
                            else:
                                return

                            if not text:
                                return
                            asyncio.run_coroutine_threadsafe(
                                ws.send(json.dumps(
                                    {"type": "tool_status", "text": text},
                                    ensure_ascii=False,
                                )),
                                loop,
                            )
                        return _tool_cb

                    agent.tool_progress_callback = _make_tool_cb(did, ev_loop)
                    logger.debug("voice_device: [%s] tool_progress_callback 注入成功",
                                 device_id)

                logger.debug("voice_device: [%s] thinking_callback 注入成功",
                             device_id)
                return
            await asyncio.sleep(0.05)

        logger.debug("voice_device: [%s] thinking_callback 注入超时（agent 未出现）",
                     device_id)

    def _send_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create the send lock for chat_id."""
        lock = self._send_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._send_locks[chat_id] = lock
        return lock

    @staticmethod
    def _format_peer(ws) -> str:
        """Format remote address as host:port for logging."""
        try:
            host, port = ws.remote_address[0], ws.remote_address[1]
            return f"{host}:{port}"
        except Exception:
            return "unknown"

    @staticmethod
    def _parse_text_frame(raw: str, peer: str) -> Optional[Dict[str, Any]]:
        """Parse a text frame as JSON; logs warning and returns None on failure."""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("voice_device: 来自 %s 的非 JSON 文本: %r",
                           peer, raw[:120])
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _authorize_hello(
        self,
        hello: Dict[str, Any],
        ws,
        peer: str,
    ) -> Optional[str]:
        """Validate device_id from hello frame against the whitelist.

        Returns the device_id on success, or None (close frame already sent).
        """
        device_id = str(hello.get("device_id") or "").strip()
        if not device_id:
            asyncio.ensure_future(self._send_error(ws, "missing device_id"))
            asyncio.ensure_future(ws.close(code=1008, reason="missing device_id"))
            return None
        if self._allow_all_devices:
            return device_id
        if not self._allowed_devices:
            logger.warning("voice_device: 拒绝 %s (来自 %s) -- "
                           "未配置白名单且 VOICE_DEVICE_ALLOW_ALL 未开启",
                           device_id, peer)
            asyncio.ensure_future(self._send_error(ws, "device not authorized"))
            asyncio.ensure_future(ws.close(code=1008, reason="not authorized"))
            return None
        if device_id not in self._allowed_devices:
            logger.warning("voice_device: 拒绝未授权设备 %s (peer=%s)",
                           device_id, peer)
            asyncio.ensure_future(self._send_error(ws, "device not authorized"))
            asyncio.ensure_future(ws.close(code=1008, reason="not authorized"))
            return None
        return device_id

    @staticmethod
    async def _send_error(ws, detail: str) -> None:
        """Send error frame; silently ignore send failures."""
        try:
            await ws.send(json.dumps({"type": "error", "detail": detail}))
        except Exception:
            logger.debug("voice_device: failed to send error frame: %r", detail, exc_info=True)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def _config_yaml_port() -> Optional[int]:
    """Read port from ~/.hermes/config.yaml so YAML-only users aren't blocked."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        plat = (cfg.get("platforms") or {}).get("voice_device") or {}
        port = (plat.get("extra") or {}).get("port") or plat.get("port")
        return int(port) if port else None
    except Exception:
        logger.debug("voice_device: failed to read port from config.yaml", exc_info=True)
        return None


def check_requirements() -> bool:
    """Whether this platform should be shown in setup/status.

    Either VOICE_DEVICE_PORT env or a port in config.yaml is sufficient.
    Does not require websockets to be installed so install_hint shows properly.
    """
    if os.getenv("VOICE_DEVICE_PORT"):
        return True
    if _config_yaml_port() is not None:
        return True
    return False


def validate_config(config) -> bool:
    """Pre-instantiation config validation, same env/YAML logic as check_requirements."""
    extra = getattr(config, "extra", {}) or {}
    port = (
        os.getenv("VOICE_DEVICE_PORT")
        or extra.get("port")
        or _config_yaml_port()
    )
    return bool(port)


def is_connected(adapter) -> bool:
    """Check whether the adapter is connected and running."""
    return bool(getattr(adapter, "_ws_server", None) and getattr(adapter, "_running", False))


def _env_enablement() -> Optional[dict]:
    """Expose env-driven config to 'hermes gateway status' before adapter instantiation."""
    port = os.getenv("VOICE_DEVICE_PORT")
    if not port:
        return None
    extra = {"port": int(port)}
    host = os.getenv("VOICE_DEVICE_HOST")
    if host:
        extra["host"] = host
    path = os.getenv("VOICE_DEVICE_PATH")
    if path:
        extra["path"] = path
    return extra


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at gateway startup."""
    ctx.register_platform(
        name="voice_device",
        label="Voice Device",
        adapter_factory=lambda cfg: VoiceDeviceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["VOICE_DEVICE_PORT"],
        install_hint="pip install websockets",
        env_enablement_fn=_env_enablement,
        allowed_users_env="VOICE_DEVICE_ALLOWED_DEVICES",
        allow_all_env="VOICE_DEVICE_ALLOW_ALL",
        emoji="🎙️",
        pii_safe=True,
    )
