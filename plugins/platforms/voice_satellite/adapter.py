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

    The runtime plugin loader imports this plugin as a package (it ships an
    __init__.py), but the TEST loader
    (tests/gateway/_plugin_adapter_loader.py) loads adapter.py standalone with
    no package context, so a relative/dotted sibling import would fail there.
    Loading siblings by file path works under both loaders — the same
    mechanism the test loader itself uses.
    """
    mod_key = f"hermes_voice_satellite_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
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

        # prompt=False: called from the gateway's platform registry, which
        # must never block on an interactive install confirmation.
        ensure("platform.voice_satellite", prompt=False)
        import wyoming  # noqa: F401
        return True
    except Exception:
        return False


def validate_config(config) -> bool:
    satellites = (config.extra or {}).get("satellites")
    if not isinstance(satellites, list) or not satellites:
        return False
    for entry in satellites:
        # Each entry must be dialable: connect() would otherwise spin the
        # reconnect loop forever against an empty host.
        if not isinstance(entry, dict) or not str(entry.get("host") or "").strip():
            return False
        try:
            port = int(entry.get("port", 10700))
        except (TypeError, ValueError):
            return False
        if not 0 < port < 65536:
            return False
    return True


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Translate the user's `voice_satellite` config.yaml block.

    Only seeds ``PlatformConfig.extra`` (satellites, endpointing, timeouts).
    Enablement comes from the block's own ``enabled: true`` key, which the
    loader's generic shared-key loop bridges onto ``PlatformConfig.enabled``.

    ``load_gateway_config()`` binds ``platform_cfg`` to whichever block the
    user wrote — the top-level ``voice_satellite:`` section, or the nested
    ``platforms.voice_satellite`` / ``gateway.platforms.voice_satellite``
    fallbacks — so this hook must read from it rather than re-reading
    ``yaml_cfg`` (a top-level key may not exist). The loader only merges this
    function's *return value* into ``extra``; any in-place mutation of
    ``platform_cfg`` is discarded and never reaches ``PlatformConfig.enabled``.
    """
    section = platform_cfg if isinstance(platform_cfg, dict) else {}
    if not section.get("satellites"):
        return None
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
        self._reply_timeout = float(extra.get("reply_timeout_seconds", 120.0))
        self._links: Dict[str, Any] = {}
        self._machines: Dict[str, Any] = {}
        self._transcribe_tasks: set = set()
        # chat_ids whose turn reply was already spoken by play_tts, so the
        # base path's follow-up send() must not re-speak the same text.
        self._reply_spoken: set = set()
        # per-satellite reply watchdog tasks (frees a stuck THINKING machine).
        self._watchdogs: Dict[str, asyncio.Task] = {}
        self._audio = _import_sibling("audio")
        self._tm = _import_sibling("turn_machine")

    # -- authorization: satellites are trusted by being listed in config ----
    @property
    def authorization_is_upstream(self) -> bool:
        """Voice satellites are authorized UPSTREAM by operator configuration.

        A satellite has no user-account identity the gateway could match against
        a ``{PLATFORM}_ALLOWED_USERS`` allowlist: its identity IS its configured
        name, and it only exists because the operator listed it in
        ``config.yaml`` on a trusted LAN. Listing a satellite is the
        authorization grant — mirroring the relay adapter's upstream-trust
        posture (see ``BasePlatformAdapter.authorization_is_upstream`` and the
        carve-out in ``gateway/authz_mixin.py``). Keep satellites LAN-only; they
        are not network-exposed and carry no per-sender allowlist.
        """
        return True

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
                on_disconnect=self._on_link_disconnect,
                tts_sample_rate=int(cfg.get("tts_sample_rate", self._tts_sample_rate)),
            )
            self._links[name] = link
            await link.start()
        self._running = True
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        for task in list(self._transcribe_tasks):
            task.cancel()
        self._transcribe_tasks.clear()
        for name in list(self._watchdogs):
            self._cancel_watchdog(name)
        self._reply_spoken.clear()
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
        # New turn: clear any stale "already spoke" flag from a prior turn whose
        # base follow-up send() never arrived (e.g. text-only fallback path).
        self._reply_spoken.discard(name)
        if machine.on_pipeline_start(now=time.monotonic()):
            logger.info("[voice_satellite:%s] listening", name)
        else:
            # Wake fired mid-turn (THINKING/SPEAKING/...): the satellite is now
            # streaming mic audio and will ONLY stop when it receives a
            # transcript. Send an empty one so it returns to wake mode instead
            # of streaming forever into a busy machine.
            logger.info(
                "[voice_satellite:%s] wake ignored (phase=%s)",
                name, machine.phase.value,
            )
            try:
                await self._links[name].send_transcript("")
            except ConnectionError:
                pass

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
            _, utterance, utt_rate, turn_id = action
            task = asyncio.create_task(
                self._transcribe_and_dispatch(name, utterance, utt_rate, turn_id)
            )
            self._transcribe_tasks.add(task)
            task.add_done_callback(self._transcribe_tasks.discard)

    async def _abort_turn(self, name: str) -> None:
        self._machines[name].to_idle()
        try:
            await self._links[name].send_transcript("")
        except ConnectionError:
            pass

    def _cancel_watchdog(self, name: str) -> None:
        task = self._watchdogs.pop(name, None)
        if task is not None:
            task.cancel()

    def _arm_watchdog(self, name: str) -> None:
        """Free a machine that never reaches playback (agent turn produced no
        reply, or the base reply path never called play_tts).

        The transcript was already sent, so the satellite itself is fine and
        back in wake mode; this only unsticks the local machine so the next
        wake can start a turn instead of being rejected forever as THINKING.
        """
        self._cancel_watchdog(name)
        task = asyncio.create_task(self._reply_watchdog(name))
        self._watchdogs[name] = task
        task.add_done_callback(
            lambda t, n=name: self._watchdogs.pop(n, None) if self._watchdogs.get(n) is t else None
        )

    async def _reply_watchdog(self, name: str) -> None:
        try:
            await asyncio.sleep(self._reply_timeout)
        except asyncio.CancelledError:
            return
        machine = self._machines.get(name)
        if machine is not None and machine.phase is self._tm.TurnPhase.THINKING:
            logger.warning(
                "[voice_satellite:%s] no reply within %.0fs; freeing turn",
                name, self._reply_timeout,
            )
            machine.to_idle()

    async def _on_link_disconnect(self, name: str) -> None:
        """Reset per-satellite state when the link drops mid-turn.

        A dropped connection strands the machine in whatever phase it held; on
        reconnect the satellite is back in wake mode, so the machine must be
        idle to accept the next wake.
        """
        machine = self._machines.get(name)
        if machine is not None:
            machine.to_idle()
        self._reply_spoken.discard(name)
        self._cancel_watchdog(name)

    async def _transcribe_and_dispatch(
        self, name: str, utterance: bytes, rate: int, turn_id: int
    ) -> None:
        try:
            await self._transcribe_and_dispatch_inner(name, utterance, rate, turn_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[voice_satellite:%s] transcription turn failed", name
            )
            machine = self._machines.get(name)
            if machine is None or machine.turn_id != turn_id:
                # This failure belongs to a superseded turn (link dropped or
                # a new wake started); the current turn owns the satellite
                # stream — recovering here would clobber it.
                return
            machine.to_idle()
            link = self._links.get(name)
            if link is not None:
                try:
                    await link.send_transcript("")
                except ConnectionError:
                    pass

    async def _transcribe_and_dispatch_inner(
        self, name: str, utterance: bytes, rate: int, turn_id: int
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

        action = machine.on_transcript_ready(text, turn_id)
        if action[0] == "stale":
            # A newer turn (or a reset) superseded this one while STT ran;
            # nothing here may touch the machine or the satellite stream.
            return
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
        # If the agent turn never reaches playback (no reply, or a base path
        # that skipped play_tts), free the machine after a timeout so the next
        # wake works. play_tts / the send() text-only fallback cancel this once
        # the turn is genuinely resolved.
        if machine.phase is self._tm.TurnPhase.THINKING:
            self._arm_watchdog(name)

    # -- outbound: agent -> satellite ----------------------------------------
    def prepare_tts_text(self, text: str) -> str:
        from tools.tts_tool import _strip_markdown_for_tts

        return _strip_markdown_for_tts(text)[:4000].strip()

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        link = self._links.get(chat_id)
        machine = self._machines.get(chat_id)
        if link is None or not link.connected:
            return SendResult(success=False, error=f"satellite {chat_id} not connected")

        TurnPhase = self._tm.TurnPhase
        if machine is not None and machine.phase not in (
            TurnPhase.THINKING, TurnPhase.IDLE
        ):
            # Busy (listening/transcribing/already speaking): refuse WITHOUT
            # touching the machine so the in-flight turn is left intact.
            return SendResult(
                success=False,
                error=f"satellite {chat_id} busy ({machine.phase.value})",
                retryable=True,
            )

        was_turn_reply = machine is not None and machine.phase is TurnPhase.THINKING
        started = False
        played = False
        try:
            # Transcode BEFORE touching the machine: a transcode failure then
            # leaves the machine untouched (no on_reply_started without a
            # matching on_playback_done in finally).
            pcm = await asyncio.to_thread(
                self._audio.transcode_to_pcm, audio_path, link.snd_rate
            )
            if machine is not None:
                self._cancel_watchdog(chat_id)
                machine.on_reply_started()
                started = True
            await link.play_pcm(pcm, rate=link.snd_rate)
            # TCP writes drain far faster than the speaker plays back. Hold the
            # turn SPEAKING for the audio's real duration so a wake during
            # playback is rejected — opening the mic now would capture our own
            # TTS audio bleeding from the speaker.
            await asyncio.sleep(min(len(pcm) / (link.snd_rate * 2), 120.0))
            played = True
            return SendResult(success=True)
        except Exception as err:  # noqa: BLE001 - surface as failed send
            logger.warning("[voice_satellite:%s] playback failed: %s", chat_id, err)
            return SendResult(success=False, error=str(err), retryable=True)
        finally:
            if started and machine is not None:
                machine.on_playback_done()
                if was_turn_reply and played:
                    # This turn's reply was spoken here; the base path's
                    # follow-up send() of the same text must not re-speak it.
                    self._reply_spoken.add(chat_id)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if chat_id in self._reply_spoken:
            # Base reply path: the text portion follows a play_tts that
            # already spoke this turn's reply.
            self._reply_spoken.discard(chat_id)
            return SendResult(success=True)
        machine = self._machines.get(chat_id)
        if machine is not None and machine.phase is not self._tm.TurnPhase.IDLE:
            if machine.phase is self._tm.TurnPhase.THINKING:
                # Auto-TTS failed upstream and base fell back to text-only:
                # nothing to play on a voice-only surface; end the turn so
                # the next wake works.
                logger.warning(
                    "[voice_satellite:%s] text-only reply (TTS unavailable); "
                    "turn ended without audio", chat_id,
                )
                self._cancel_watchdog(chat_id)
                machine.to_idle()
                return SendResult(success=True)
            logger.warning(
                "[voice_satellite:%s] dropping message during active turn "
                "(phase=%s)", chat_id, machine.phase.value,
            )
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
        # is_connected = "satellites listed in config", exactly what
        # validate_config checks. Without this hook the setup wizard falls
        # back to check_fn and reports the platform configured whenever the
        # wyoming dep happens to be importable.
        is_connected=validate_config,
        validate_config=validate_config,
        apply_yaml_config_fn=_apply_yaml_config,
        install_hint="pip install 'hermes-agent[satellite]'",
        emoji="🎙️",
        pii_safe=True,
        platform_hint=(
            "You are speaking aloud through a home voice assistant "
            "speaker. Keep replies brief and conversational — one to "
            "three sentences. Never use markdown, code blocks, tables, "
            "or URLs; they will be read out loud."
        ),
    )
