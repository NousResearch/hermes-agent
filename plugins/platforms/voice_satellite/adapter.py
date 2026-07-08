"""Voice satellite platform adapter (Wyoming protocol).

Each configured satellite (a wyoming-satellite device on the LAN) becomes
one gateway session: chat_id == user_id == the satellite's configured
name. Wake word runs on the satellite; Hermes owns endpointing, STT,
the agent turn, TTS, and playback streaming.
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


def _import_sibling(name: str):
    """Load a module that lives next to this file.

    plugins/platforms/ has no __init__.py on purpose (plugins are not
    importable as dotted packages), so siblings load by file path — the
    same mechanism tests/gateway/_plugin_adapter_loader.py uses.
    """
    mod_key = f"hermes_voice_satellite_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(mod_key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = module
    spec.loader.exec_module(module)
    return module


def check_requirements() -> bool:
    """Wyoming framing lib present (lazy-installed on first use)."""
    try:
        import wyoming  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from tools.lazy_deps import ensure

        ensure("platform.voice_satellite")
        import wyoming  # noqa: F401
        return True
    except Exception:
        return False


def validate_config(config) -> bool:
    return bool((config.extra or {}).get("satellites"))


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Translate the top-level `voice_satellite:` config.yaml section."""
    section = yaml_cfg.get("voice_satellite") or {}
    if not isinstance(section, dict) or not section.get("satellites"):
        return None
    if section.get("enabled", True):
        platform_cfg["enabled"] = True
    return {
        "satellites": section.get("satellites", []),
        "endpointing": section.get("endpointing", {}) or {},
        "listen_timeout_seconds": section.get("listen_timeout_seconds", 30.0),
        "tts_sample_rate": section.get("tts_sample_rate", 22050),
    }


class VoiceSatelliteAdapter(BasePlatformAdapter):
    """Gateway adapter for Wyoming voice satellites."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("voice_satellite"))
        extra = config.extra or {}
        self._satellite_cfgs = list(extra.get("satellites", []))
        self._endpointing = dict(extra.get("endpointing", {}))
        self._listen_timeout = float(extra.get("listen_timeout_seconds", 30.0))
        self._tts_sample_rate = int(extra.get("tts_sample_rate", 22050))
        self._links: Dict[str, Any] = {}
        self._machines: Dict[str, Any] = {}
        self._audio = _import_sibling("audio")
        self._tm = _import_sibling("turn_machine")

    # -- auto-TTS: this surface is voice-only, always speak replies --------
    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        return True

    # -- lifecycle ----------------------------------------------------------
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._satellite_cfgs:
            logger.warning("[voice_satellite] no satellites configured")
            return False
        link_mod = _import_sibling("satellite_link")
        for cfg in self._satellite_cfgs:
            name = str(cfg.get("name") or f"{cfg.get('host')}:{cfg.get('port')}")
            if name in self._links:
                continue
            self._machines[name] = self._tm.TurnMachine(
                self._detector_factory, listen_timeout_seconds=self._listen_timeout
            )
            link = link_mod.SatelliteLink(
                name,
                str(cfg.get("host", "")),
                int(cfg.get("port", 10700)),
                on_pipeline_start=self._on_pipeline_start,
                on_audio_chunk=self._on_audio_chunk,
                on_played=self._on_played,
                tts_sample_rate=int(cfg.get("tts_sample_rate", self._tts_sample_rate)),
            )
            self._links[name] = link
            await link.start()
        self._running = True
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        for link in self._links.values():
            await link.stop()
        self._links.clear()
        self._machines.clear()
        self._mark_disconnected()

    def _detector_factory(self):
        return self._audio.EndpointDetector(
            silence_threshold=int(self._endpointing.get("silence_threshold", 200)),
            silence_duration=float(self._endpointing.get("silence_duration", 1.2)),
            min_speech_seconds=float(self._endpointing.get("min_speech_seconds", 0.5)),
            max_utterance_seconds=float(
                self._endpointing.get("max_utterance_seconds", 20.0)
            ),
        )

    # -- inbound: satellite -> agent ----------------------------------------
    async def _on_pipeline_start(self, name: str) -> None:
        machine = self._machines[name]
        if machine.on_pipeline_start(now=time.monotonic()):
            logger.info("[voice_satellite:%s] listening", name)

    async def _on_audio_chunk(
        self, name: str, pcm: bytes, seconds: float, rate: int
    ) -> None:
        machine = self._machines[name]
        action = machine.on_audio(pcm, seconds, rate, now=time.monotonic())
        if action is None:
            return
        if action[0] == "abort":
            await self._abort_turn(name)
        elif action[0] == "transcribe":
            _, utterance, utt_rate = action
            asyncio.create_task(self._transcribe_and_dispatch(name, utterance, utt_rate))

    async def _on_played(self, name: str) -> None:
        """Satellite acknowledged that queued audio finished playing.

        No-op: ``play_tts`` already marks the turn machine's
        playback-done once the stream write completes (M1). This hook
        exists so ``SatelliteLink`` always has a callback to invoke for
        the wyoming "played" event; M2 may use it for follow-up-window
        timing.
        """

    async def _abort_turn(self, name: str) -> None:
        self._machines[name].to_idle()
        try:
            await self._links[name].send_transcript("")
        except ConnectionError:
            pass

    async def _transcribe_and_dispatch(
        self, name: str, utterance: bytes, rate: int
    ) -> None:
        from tools.transcription_tools import transcribe_audio
        from tools.voice_mode import is_whisper_hallucination

        machine = self._machines[name]
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="satellite_utt_")
        os.close(fd)
        try:
            self._audio.pcm_to_wav(utterance, wav_path, rate=rate)
            result = await asyncio.to_thread(transcribe_audio, wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        text = ""
        if isinstance(result, dict) and result.get("success"):
            text = (result.get("transcript") or "").strip()
        if text and is_whisper_hallucination(text):
            text = ""

        action = machine.on_transcript_ready(text)
        if action[0] == "abort":
            await self._abort_turn(name)
            return

        # End satellite mic streaming; it returns to wake-word detection
        # while the agent thinks. (M2 withholds this for the follow-up window.)
        try:
            await self._links[name].send_transcript(text)
        except ConnectionError:
            machine.to_idle()
            return

        logger.info("[voice_satellite:%s] heard: %s", name, text)
        source = self.build_source(
            chat_id=name, chat_name=name, chat_type="dm", user_id=name, user_name=name
        )
        event = MessageEvent(
            text=text, message_type=MessageType.VOICE, source=source
        )
        await self.handle_message(event)

    # -- outbound: agent -> satellite ----------------------------------------
    def prepare_tts_text(self, text: str) -> str:
        from tools.tts_tool import _strip_markdown_for_tts

        return _strip_markdown_for_tts(text)[:4000].strip()

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        link = self._links.get(chat_id)
        machine = self._machines.get(chat_id)
        if link is None or not link.connected:
            return SendResult(success=False, error=f"satellite {chat_id} not connected")
        try:
            pcm = await asyncio.to_thread(
                self._audio.transcode_to_pcm, audio_path, link.snd_rate
            )
            if machine is not None:
                machine.on_reply_started()
            await link.play_pcm(pcm, rate=link.snd_rate)
            return SendResult(success=True)
        except Exception as err:  # noqa: BLE001 - surface as failed send
            logger.warning("[voice_satellite:%s] playback failed: %s", chat_id, err)
            return SendResult(success=False, error=str(err), retryable=True)
        finally:
            if machine is not None:
                machine.on_playback_done()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        machine = self._machines.get(chat_id)
        if machine is not None and machine.phase is not self._tm.TurnPhase.IDLE:
            # Mid-turn text reply: the spoken reply arrives via play_tts;
            # a voice-only surface has nowhere to render text.
            return SendResult(success=True)
        # Idle announce (cron delivery, background completion): speak it.
        return await self._announce(chat_id, content)

    async def _announce(self, chat_id: str, content: str) -> SendResult:
        from tools.tts_tool import check_tts_requirements, text_to_speech_tool

        link = self._links.get(chat_id)
        if link is None or not link.connected:
            return SendResult(success=False, error=f"satellite {chat_id} not connected")
        if not check_tts_requirements():
            return SendResult(success=False, error="no TTS provider configured")
        speech = self.prepare_tts_text(content)
        if not speech:
            return SendResult(success=True)
        tts_raw = await asyncio.to_thread(text_to_speech_tool, text=speech)
        tts_data = json.loads(tts_raw)
        audio_path = tts_data.get("file_path")
        if not tts_data.get("success") or not audio_path:
            return SendResult(
                success=False, error=str(tts_data.get("error", "TTS failed"))
            )
        try:
            return await self.play_tts(chat_id, audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    # -- misc required surface -----------------------------------------------
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass  # no visual surface

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "satellite", "chat_id": chat_id}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="voice_satellite",
        label="Voice Satellite",
        adapter_factory=lambda cfg: VoiceSatelliteAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        apply_yaml_config_fn=_apply_yaml_config,
        install_hint="pip install wyoming==1.10.0",
        emoji="🎙️",
        pii_safe=True,
        platform_hint=(
            "You are speaking aloud through a home voice assistant "
            "speaker. Keep replies brief and conversational — one to "
            "three sentences. Never use markdown, code blocks, tables, "
            "or URLs; they will be read out loud."
        ),
    )
