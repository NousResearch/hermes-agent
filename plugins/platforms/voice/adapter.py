"""Voice platform plugin — realtime Daily (WebRTC) voice calls.

Structural template: plugins/platforms/ntfy/adapter.py.
Turn orchestration lives in turn_loop.py; this module owns plugin
registration, requirement checks, and the adapter lifecycle.

Standalone mode: this agent holds DAILY_API_KEY, creates its own private
Daily room + a single-use meeting token at connect time, joins immediately,
and logs the room URL for the owner to share. caller == owner.

Opinionated stack: Deepgram Flux (streaming STT + model-integrated turn
detection), Cartesia (streaming TTS, default — the per-turn TTS is built
behind a factory so another provider can be slotted in), and the agent's
voice_model: for generation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

logger = logging.getLogger(__name__)

DAILY_API = "https://api.daily.co/v1"
STANDALONE_ROOM_TTL_S = 3600

# Call watchdog: an abandoned call — tab closed, or the room expired and
# ejected the agent — would otherwise leave the billable STT stream running.
# The watchdog tears the call down when humans are gone, the call ended
# remotely, or a hard age cap is hit.
DEFAULT_IDLE_TEARDOWN_S = 60.0   # extra.idle_teardown_s
DEFAULT_MAX_CALL_S = 1800.0      # extra.max_call_s
WATCHDOG_POLL_S = 0.5


def _voice_modules():
    """Sibling voice modules via the discord dual-import pattern
    (plugins/platforms/discord/adapter.py:1991-1993): flat import for the
    test loader, relative import for the production package
    (hermes_plugins.platforms__voice)."""
    try:
        import cartesia_tts
        import daily_transport
        import stt as stt_mod
        import turn_loop
    except ImportError:
        from . import (
            cartesia_tts,
            daily_transport,
            turn_loop,
        )
        from . import stt as stt_mod
    return (daily_transport, turn_loop, cartesia_tts, stt_mod)


def _daily_available() -> bool:
    try:
        import daily  # noqa: F401
        return True
    except ImportError:
        return False


def _websockets_available() -> bool:
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


def check_requirements() -> bool:
    """Deps importable + the three call keys present (cheap env read, no
    config load): Daily (transport), Deepgram (Flux STT), Cartesia (TTS)."""
    if not _daily_available() or not _websockets_available():
        return False
    return bool(
        os.getenv("DAILY_API_KEY", "").strip()
        and os.getenv("DEEPGRAM_API_KEY", "").strip()
        and os.getenv("CARTESIA_API_KEY", "").strip()
    )


def _resolve_float_extra(extra: Dict[str, Any], key: str, default: float) -> float:
    raw = extra.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("voice: invalid %s=%r; using default %s",
                       key, raw, default)
        return default


def _resolve_opt_float_extra(extra: Dict[str, Any], key: str) -> Optional[float]:
    """Like _resolve_float_extra but returns None (not a default) when unset —
    so a Cartesia generation_config field is only sent when the user set it."""
    raw = extra.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("voice: invalid %s=%r; ignoring", key, raw)
        return None


def validate_config(config) -> bool:
    return bool(os.getenv("DAILY_API_KEY", "").strip())


def is_connected(config) -> bool:
    return check_requirements() and validate_config(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="voice",
        label="Voice",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["DAILY_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"],
        install_hint='uv sync --extra voice-platform   # daily-python + websockets',
        pii_safe=True,
        emoji="📞",
        allow_update_command=False,
        platform_hint=(
            "You are on a live voice call. Speak naturally and BRIEFLY — "
            "1-3 short sentences per reply unless asked for detail. Never use "
            "markdown, bullet lists, code blocks, or URLs; everything you "
            "write is read aloud. If a tool call will take a while, say so "
            "in a few words first."
        ),
    )


class VoiceAdapter(BasePlatformAdapter):
    """Daily-room voice call adapter (standalone mode).

    This agent holds DAILY_API_KEY, creates its own private room + meeting
    token via the Daily REST API at connect time, joins immediately, and logs
    the room URL for the owner to share.

    One active call at a time (daily-python virtual devices are process-level
    singletons — see daily_transport.py).
    """

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice")
        super().__init__(config=config, platform=platform)
        self._call_lock = asyncio.Lock()
        self._active_call: Optional[Dict[str, Any]] = None

    async def connect(self) -> bool:
        # Standalone: create our own private room + tokens, join immediately.
        # The room is private, so the shareable URL must carry an owner token —
        # the bare room URL alone cannot join a private room.
        room_url, agent_token, share_url = await self._create_standalone_room()
        await self._start_call(room_url, agent_token)
        self._mark_connected()
        # WARNING level so the URL is visible at the default gateway log level —
        # this is how the owner discovers where to call. The token in the URL is
        # the join permission (the room is private); share it only with people
        # you want to reach your agent.
        logger.warning(
            "voice: standalone call ready — open this URL to talk to your "
            "agent (keep the token private):\n    %s", share_url)
        return True

    async def _start_call(self, room_url: str, token: str) -> None:
        daily_transport, turn_loop, cartesia_tts, stt_mod = _voice_modules()
        async with self._call_lock:
            if self._active_call is not None:
                await self._end_call_locked("replaced-by-new-call")
            loop = asyncio.get_running_loop()
            extra = self.config.extra or {}

            # STT: streaming ASR + model-integrated turn detection
            # (start/eager-end/resumed/end-of-turn events the turn loop reacts
            # to). Deepgram Flux is the default; the make_stt factory is the
            # seam so another provider (e.g. Cartesia Ink-2) can be A/B'd
            # behind the same port. Inbound caller audio is at the Daily
            # SPEAKER_RATE; each adapter handles its own rate (Flux resamples
            # 24k->16k internally; Ink-2 accepts the native rate). The factory
            # resolves the provider's API key and raises if it's missing.
            stt_provider = (extra.get("stt_provider")
                            or "deepgram_flux").strip().lower()
            stt = stt_mod.make_stt(
                stt_provider,
                input_rate=daily_transport.SPEAKER_RATE,
                extra=extra,
            )
            await stt.start()
            logger.info("voice: STT provider=%s", stt.provider)

            async def on_audio_in(pcm: bytes) -> None:
                await stt.send_audio(pcm)

            transport = daily_transport.DailyTransport(loop, on_audio_in)
            await transport.join(room_url, token)

            # TTS: Cartesia (default). ONE persistent socket for the whole call
            # (connect is too costly to pay per turn), opened here so turn 1 is
            # fast. The per-turn tts_factory is the seam for another provider.
            tts_provider = (extra.get("tts_provider") or "cartesia").strip().lower()
            if tts_provider != "cartesia":
                raise RuntimeError(
                    f"unsupported tts_provider={tts_provider!r}; "
                    "'cartesia' is the bundled provider")
            cartesia_key = os.getenv("CARTESIA_API_KEY", "").strip()
            if not cartesia_key:
                raise RuntimeError("CARTESIA_API_KEY is not set")
            cartesia_voice = (
                extra.get("cartesia_voice_id")
                or os.getenv("CARTESIA_VOICE_ID", "").strip()
            )
            tts_client = cartesia_tts.CartesiaTTSClient(
                cartesia_key,
                cartesia_voice,
                model=extra.get("cartesia_model") or cartesia_tts.DEFAULT_MODEL,
                speed=_resolve_opt_float_extra(extra, "tts_speed"),
                volume=_resolve_opt_float_extra(extra, "tts_volume"),
                emotion=(extra.get("tts_emotion") or None),
            )
            await tts_client.connect()
            logger.info("voice: TTS provider=cartesia model=%s voice=%s",
                        tts_client._model, cartesia_voice)

            async def tts_factory(on_audio):
                turn = tts_client.new_turn(on_audio)
                await turn.open()
                return turn

            vloop = turn_loop.VoiceTurnLoop(
                stt, tts_factory, transport, extra=extra)
            task = asyncio.create_task(vloop.run())
            self._active_call = {
                "stt": stt, "transport": transport, "loop": vloop,
                "task": task, "tts_client": tts_client,
                "started_at": time.monotonic(), "room_url": room_url}
            self._active_call["watchdog"] = asyncio.create_task(
                self._call_watchdog(transport))
            logger.info("voice: call started in %s", room_url)

    async def _call_watchdog(self, transport) -> None:
        """Tear the call down when it is no longer worth paying for:
          - no human (remote) participant for extra.idle_teardown_s
            (tab closed / never joined),
          - the call ended remotely (room expired, agent ejected, fatal
            client error),
          - call age exceeds extra.max_call_s (hard cost cap).
        Polls every WATCHDOG_POLL_S. asr_seconds_est in the teardown summary
        makes the cost of every call auditable."""
        extra = self.config.extra or {}
        idle_teardown_s = _resolve_float_extra(
            extra, "idle_teardown_s", DEFAULT_IDLE_TEARDOWN_S)
        max_call_s = _resolve_float_extra(
            extra, "max_call_s", DEFAULT_MAX_CALL_S)
        idle_since: Optional[float] = None
        while True:
            await asyncio.sleep(WATCHDOG_POLL_S)
            call = self._active_call
            if call is None or call.get("transport") is not transport:
                return
            now = time.monotonic()
            age_s = now - call["started_at"]
            reason = None
            abnormal = transport.abnormal_end
            if abnormal is not None:
                reason = "remote-end"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s detail=%r age_s=%.0f "
                    "— call ended remotely (ejection/expiry/error)",
                    reason, abnormal, age_s)
            elif age_s >= max_call_s:
                reason = "max-call-duration"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s age_s=%.0f "
                    "max_call_s=%.0f — hard cost cap hit",
                    reason, age_s, max_call_s)
            elif transport.remote_participant_count == 0:
                if idle_since is None:
                    idle_since = now
                elif now - idle_since >= idle_teardown_s:
                    reason = "no-human-participants"
                    logger.warning(
                        "voice: WATCHDOG teardown reason=%s idle_s=%.0f "
                        "idle_teardown_s=%.0f age_s=%.0f — caller gone "
                        "(tab closed / never joined)",
                        reason, now - idle_since, idle_teardown_s, age_s)
            else:
                idle_since = None
            if reason is not None:
                await self._end_call(reason)
                return

    async def _end_call(self, reason: str) -> None:
        async with self._call_lock:
            await self._end_call_locked(reason)

    async def _end_call_locked(self, reason: str) -> None:
        call = self._active_call
        if call is None:
            return
        self._active_call = None
        # FIRST: kill the billable keep-alive feed. Every await below can take
        # real time, and the keep-alive must not pump paid STT audio while the
        # call winds down.
        call["transport"].begin_teardown()
        watchdog = call.get("watchdog")
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):
                pass
        await call["loop"].stop()
        call["task"].cancel()
        try:
            await call["task"]
        except (asyncio.CancelledError, Exception):
            pass
        await call["stt"].stop()
        tts_client = call.get("tts_client")
        if tts_client is not None:
            await tts_client.close()
        await call["transport"].leave()
        summary = {
            "event": "voice_call_teardown",
            "reason": reason,
            "room_url": call.get("room_url"),
            "call_s": round(time.monotonic() - call["started_at"], 1),
            "asr_seconds_est": round(call["stt"].asr_seconds_est, 1),
            "stt_provider": call["stt"].provider,
        }
        # Durable + WARNING-level emit: this is the per-call cost record; it
        # must survive log levels/rotation on the agent volume.
        (_, turn_loop, _, _) = _voice_modules()
        turn_loop.emit_telemetry(summary)
        logger.info("voice: call ended reason=%s", reason)

    async def _create_standalone_room(self) -> Tuple[str, str, str]:
        """Create a private room + two short-lived meeting tokens: one the agent
        joins with, and one embedded in the shareable URL so the owner can join
        the private room (the bare URL cannot). Returns
        (room_url, agent_token, share_url)."""
        import httpx

        daily_key = os.environ["DAILY_API_KEY"]
        headers = {"Authorization": f"Bearer {daily_key}"}
        exp = int(time.time()) + STANDALONE_ROOM_TTL_S
        async with httpx.AsyncClient(timeout=15.0) as client:
            room_resp = await client.post(
                f"{DAILY_API}/rooms", headers=headers,
                json={"privacy": "private", "properties": {"exp": exp}},
            )
            room_resp.raise_for_status()
            room = room_resp.json()

            async def _mint_token(is_owner: bool) -> str:
                resp = await client.post(
                    f"{DAILY_API}/meeting-tokens", headers=headers,
                    json={"properties": {"room_name": room["name"],
                                         "is_owner": is_owner, "exp": exp}},
                )
                resp.raise_for_status()
                return resp.json()["token"]

            agent_token = await _mint_token(False)
            human_token = await _mint_token(True)
        share_url = f"{room['url']}?t={human_token}"
        return room["url"], agent_token, share_url

    async def disconnect(self) -> None:
        await self._end_call("adapter-disconnect")

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        """Out-of-band sends (cron etc.): speak if a call is live.

        The gateway never routes normal chat through this adapter (the turn
        loop owns the conversation), but the abstract method must exist
        (base.py:2257). Honest failure beats a hidden queue.
        """
        call = self._active_call
        if call is None:
            return SendResult(success=False, error="no active voice call")
        tts = call["loop"]._tts
        if tts is None:
            return SendResult(
                success=False,
                error="agent not mid-utterance; queueing not supported in v0",
            )
        await tts.send_text(content)
        return SendResult(success=True, message_id="voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Voice Call", "type": "dm"}
