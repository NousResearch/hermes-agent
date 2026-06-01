from __future__ import annotations

import asyncio
import base64
import fractions
import inspect
import json
import logging
import math
import os
import re
import secrets
import subprocess
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from hermes_constants import get_hermes_home

from .simplex_sidecar import (
    SimplexNativeSidecarError,
    SimplexSidecarAnswer,
    SimplexSidecarOffer,
)
from .tracing import NativeCallTraceWriter
from .voice_turn import HermesVoiceTurnPipeline, VoiceTurnResult
from .webrtc_media import AudioUtteranceAccumulator
from .webrtc_media import pcm16_rms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimplexAiortcConfig:
    ice_servers: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun.simplex.im:443"]},
            {
                "urls": ["turns:turn.simplex.im:443?transport=tcp"],
                "username": "private2",
                "credential": "Hxuq2QxUjnhj96Zq2r4HjqHRj",
            },
        ]
    )
    ice_gather_timeout: float = 10.0
    sample_rate: int = 48000
    voice_rms_threshold: float = 500.0
    silence_seconds: float = 0.8
    ice_transport_policy: str = "relay"
    prune_non_relay_candidates: bool = False
    no_inbound_audio_timeout: float = 5.0
    no_inbound_stats_timeout: float = 1.0
    enable_simplex_media_e2ee: bool = False


@dataclass(frozen=True)
class AiortcLoopbackProbeResult:
    ok: bool
    remote_audio_frames: int
    local_sdp: dict[str, Any]
    remote_sdp: dict[str, Any]
    message: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    voice_turns: int = 0
    voice_pcm_bytes: int = 0


@dataclass(frozen=True)
class AiortcVoiceTurnSimulationResult:
    ok: bool
    code: str
    message: str
    call_id: str
    contact_id: str
    trace_path: Path
    offer_sent: bool = False
    answer_applied: bool = False
    connected: bool = False
    inbound_audio_frames: int = 0
    transcript_chars: int = 0
    expected_transcript_present: bool | None = None
    agent_response_chars: int = 0
    tts_audio_bytes: int = 0
    remote_received_audio_frames: int = 0
    remote_received_non_silent_frames: int = 0
    local_sdp: dict[str, Any] = field(default_factory=dict)
    remote_sdp: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


def generate_x25519_public_key_b64() -> str:
    private_key = X25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    return base64.urlsafe_b64encode(public_bytes).decode("ascii").rstrip("=")


def _safe_call_id(call_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id or "unknown"))
    return safe[:128] or "unknown"


class SimplexAiortcMediaEngine:
    def __init__(
        self,
        *,
        config: SimplexAiortcConfig | None = None,
        peer_factory: Callable[[], Any] | None = None,
        pipeline_factory: Callable[[], Any] | None = None,
        dh_public_key_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config or SimplexAiortcConfig()
        self.peer_factory = peer_factory or (lambda: AiortcAudioPeer(self.config))
        self.pipeline_factory = pipeline_factory or HermesVoiceTurnPipeline
        self.dh_public_key_factory = (
            dh_public_key_factory or generate_x25519_public_key_b64
        )
        self._sessions: dict[str, dict[str, Any]] = {}
        self._event_sink: Callable[[dict[str, Any]], Any] | None = None

    def set_event_sink(self, event_sink: Callable[[dict[str, Any]], Any] | None) -> None:
        self._event_sink = event_sink

    async def start_incoming(self, request: dict[str, Any]) -> SimplexSidecarOffer:
        call_id = str(request.get("callId") or "")
        if not call_id:
            raise ValueError("callId is required")
        previous = self._sessions.pop(call_id, None)
        if previous is not None:
            await _close_session(previous)

        peer = self.peer_factory()
        set_terminal_callback = getattr(peer, "set_terminal_callback", None)
        if callable(set_terminal_callback):
            async def _on_terminal(
                status: str,
                reason: str,
                details: dict[str, Any] | None = None,
            ) -> None:
                await self._handle_peer_terminal(
                    call_id=call_id,
                    status=status,
                    reason=reason,
                    details=details or {},
                )

            set_terminal_callback(_on_terminal)
        set_event_callback = getattr(peer, "set_event_callback", None)
        if callable(set_event_callback):
            async def _on_peer_event(
                event: str,
                details: dict[str, Any] | None = None,
            ) -> None:
                await self._handle_peer_event(
                    call_id=call_id,
                    event=event,
                    details=details or {},
                )

            set_event_callback(_on_peer_event)
        pipeline = self.pipeline_factory()
        await peer.start(call_id, pipeline)
        e2ee_unavailable_reason = ""
        if bool(self.config.enable_simplex_media_e2ee):
            shared_key_text = str(request.get("sharedKey") or "")
            try:
                peer._shared_media_key = decode_simplex_media_key(shared_key_text)
            except ValueError:
                peer._shared_media_key = None
                e2ee_unavailable_reason = "invalid_shared_key"
                await self._handle_peer_event(
                    call_id=call_id,
                    event="simplex_media_e2ee_unavailable",
                    details={"reason": e2ee_unavailable_reason},
                )
        sdp, ice_candidates = await peer.create_offer()
        self._sessions[call_id] = {
            "peer": peer,
            "pipeline": pipeline,
        }
        requested_encrypted = bool(request.get("encrypted"))
        offer_encrypted = requested_encrypted and bool(
            self.config.enable_simplex_media_e2ee
        ) and bool(getattr(peer, "_shared_media_key", None))
        if requested_encrypted and not offer_encrypted:
            reason = e2ee_unavailable_reason or "simplex_media_e2ee_not_enabled"
            await self._handle_peer_event(
                call_id=call_id,
                event="simplex_media_e2ee_disabled",
                details={
                    "requested": True,
                    "reason": reason,
                },
            )
        return SimplexSidecarOffer(
            sdp=sdp,
            ice_candidates=ice_candidates,
            capabilities={"encryption": offer_encrypted},
            call_dh_pub_key=self.dh_public_key_factory() if offer_encrypted else None,
        )

    async def start_outgoing_answer(
        self,
        request: dict[str, Any],
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> SimplexSidecarAnswer:
        call_id = str(request.get("callId") or "")
        if not call_id:
            raise ValueError("callId is required")
        previous = self._sessions.pop(call_id, None)
        if previous is not None:
            await _close_session(previous)

        peer = self.peer_factory()
        set_terminal_callback = getattr(peer, "set_terminal_callback", None)
        if callable(set_terminal_callback):
            async def _on_terminal(
                status: str,
                reason: str,
                details: dict[str, Any] | None = None,
            ) -> None:
                await self._handle_peer_terminal(
                    call_id=call_id,
                    status=status,
                    reason=reason,
                    details=details or {},
                )

            set_terminal_callback(_on_terminal)
        set_event_callback = getattr(peer, "set_event_callback", None)
        if callable(set_event_callback):
            async def _on_peer_event(
                event: str,
                details: dict[str, Any] | None = None,
            ) -> None:
                await self._handle_peer_event(
                    call_id=call_id,
                    event=event,
                    details=details or {},
                )

            set_event_callback(_on_peer_event)
        pipeline = self.pipeline_factory()
        await peer.start(call_id, pipeline)
        e2ee_unavailable_reason = ""
        if bool(self.config.enable_simplex_media_e2ee):
            shared_key_text = str(request.get("sharedKey") or "")
            try:
                peer._shared_media_key = decode_simplex_media_key(shared_key_text)
            except ValueError:
                peer._shared_media_key = None
                e2ee_unavailable_reason = "invalid_shared_key"
                await self._handle_peer_event(
                    call_id=call_id,
                    event="simplex_media_e2ee_unavailable",
                    details={"reason": e2ee_unavailable_reason},
                )
        requested_encrypted = bool(request.get("encrypted"))
        if (
            requested_encrypted
            and bool(self.config.enable_simplex_media_e2ee)
            and not bool(getattr(peer, "_shared_media_key", None))
        ):
            await self._handle_peer_event(
                call_id=call_id,
                event="simplex_media_e2ee_disabled",
                details={
                    "requested": True,
                    "reason": e2ee_unavailable_reason or "missing_shared_key",
                },
            )
        if requested_encrypted and not bool(self.config.enable_simplex_media_e2ee):
            await self._handle_peer_event(
                call_id=call_id,
                event="simplex_media_e2ee_disabled",
                details={
                    "requested": True,
                    "reason": "simplex_media_e2ee_not_enabled",
                },
            )
        answer_sdp, answer_ice_candidates = await peer.create_answer(
            sdp,
            ice_candidates,
        )
        self._sessions[call_id] = {
            "peer": peer,
            "pipeline": pipeline,
        }
        return SimplexSidecarAnswer(
            sdp=answer_sdp,
            ice_candidates=answer_ice_candidates,
        )

    async def apply_answer(
        self,
        call_id: str,
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> None:
        peer = self._require_peer(call_id)
        await peer.apply_answer(sdp, ice_candidates)

    async def add_extra_ice(self, call_id: str, ice_candidates: list[Any]) -> None:
        peer = self._require_peer(call_id)
        await peer.add_extra_ice(ice_candidates)

    async def stop(self, call_id: str) -> None:
        session = self._sessions.pop(call_id, None)
        if session is not None:
            await _close_session(session)

    async def process_audio_file(self, call_id: str, audio_path: str) -> VoiceTurnResult:
        session = self._sessions.get(call_id)
        if session is None:
            return VoiceTurnResult(
                ok=False,
                code="call_voice_no_active_session",
                message="No active native call session.",
            )
        pipeline = session["pipeline"]
        pcm16, sample_rate = _read_wav_pcm16(Path(audio_path))
        return await pipeline.process_pcm16(
            call_id=call_id,
            pcm16=pcm16,
            sample_rate=sample_rate,
        )

    def _require_peer(self, call_id: str) -> Any:
        session = self._sessions.get(str(call_id or ""))
        if session is None:
            raise RuntimeError("No active native call session")
        return session["peer"]

    async def _handle_peer_terminal(
        self,
        *,
        call_id: str,
        status: str,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        session = self._sessions.pop(call_id, None)
        if session is not None:
            await _close_session(session)
        await self._emit_event(
            {
                "type": "status",
                "callId": call_id,
                "status": status,
                "reasonCode": reason,
                "details": details,
            }
        )

    async def _handle_peer_event(
        self,
        *,
        call_id: str,
        event: str,
        details: dict[str, Any],
    ) -> None:
        await self._emit_event(
            {
                "type": "event",
                "callId": call_id,
                "event": str(event or "media"),
                "details": details,
            }
        )

    async def _emit_event(self, event: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        result = self._event_sink(event)
        if inspect.isawaitable(result):
            await result


async def _close_session(session: dict[str, Any]) -> None:
    peer = session.get("peer")
    close = getattr(peer, "close", None)
    if callable(close):
        await close()


def _read_wav_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError("debug audio WAV must be 16-bit PCM")
    if channels == 1:
        return frames, sample_rate
    samples = []
    frame_width = sample_width * channels
    for offset in range(0, len(frames), frame_width):
        samples.append(frames[offset : offset + sample_width])
    return b"".join(samples), sample_rate


class AiortcAudioPeer:
    def __init__(self, config: SimplexAiortcConfig) -> None:
        self.config = config
        self._pc = None
        self._call_id = ""
        self._ice_candidates: list[dict[str, Any]] = []
        self._ice_gather_complete = asyncio.Event()
        self._audio_relay_task: asyncio.Task | None = None
        self._output_track = None
        self._accumulator: Any | None = None
        self._pipeline: Any | None = None
        self._remote_audio_frame_count = 0
        self._terminal_callback: Callable[[str, str, dict[str, Any]], Any] | None = None
        self._event_callback: Callable[[str, dict[str, Any]], Any] | None = None
        self._terminal_reported = False
        self._closing = False
        self._no_inbound_watchdog_task: asyncio.Task | None = None
        self._shared_media_key: bytes | None = None

    def set_terminal_callback(
        self,
        callback: Callable[[str, str, dict[str, Any]], Any] | None,
    ) -> None:
        self._terminal_callback = callback

    def set_event_callback(
        self,
        callback: Callable[[str, dict[str, Any]], Any] | None,
    ) -> None:
        self._event_callback = callback

    async def start(self, call_id: str, pipeline: Any) -> None:
        self._call_id = str(call_id or "")
        apply_aiortc_transport_policy(self.config.ice_transport_policy)
        RTCPeerConnection, RTCConfiguration, RTCIceServer = _load_aiortc_peer_types()
        ice_servers = [
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in self.config.ice_servers
        ]
        self._pc = RTCPeerConnection(
            RTCConfiguration(iceServers=ice_servers)
        )
        self._ice_candidates = []
        self._ice_gather_complete.clear()
        if getattr(pipeline, "is_streaming", False):
            # Live streaming path (Slice 6b): the session owns turn detection and
            # outbound audio. Build a live PCM track, wire it as the transport's
            # outbound sink + barge-in drop hook, and feed inbound frames straight
            # into the pipeline via the direct-feed accumulator.
            self._pipeline = pipeline
            self._output_track = _create_pcm_streaming_track(self.config.sample_rate)
            pipeline.transport.set_outbound_sink(
                self._output_track.enqueue,
                drop=self._output_track.drop_pending,
            )
            self._accumulator = _DirectFeedAccumulator(
                pipeline,
                call_id,
                self.config.sample_rate,
            )
        else:
            self._pipeline = None
            self._output_track = _create_queued_audio_track(
                self.config.sample_rate,
                self._schedule_event,
            )
            self._accumulator = AudioUtteranceAccumulator(
                call_id=call_id,
                pipeline=pipeline,
                output_track=self._output_track,
                sample_rate=self.config.sample_rate,
                voice_rms_threshold=self.config.voice_rms_threshold,
                silence_seconds=self.config.silence_seconds,
            )
        self._install_event_handlers()
        self._pc.addTrack(self._output_track)
        logger.info("SimpleX native WebRTC peer started: call_id=%s", self._call_id)

    async def create_offer(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self._pc is None:
            raise RuntimeError("RTCPeerConnection has not been started")
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        await self._wait_for_ice_gathering()
        local = self._pc.localDescription
        sdp_text = str(local.sdp or "")
        sdp_text, ice_candidates = prepare_local_offer_transport(
            sdp_text,
            self._ice_candidates,
            self.config,
        )
        logger.info(
            "SimpleX native WebRTC local offer diagnostics: call_id=%s %s",
            self._call_id,
            json.dumps(describe_sdp(sdp_text), sort_keys=True),
        )
        await self._emit_event("local_offer_sdp", {"sdp": describe_sdp(sdp_text)})
        return (
            {"type": local.type, "sdp": sdp_text},
            ice_candidates,
        )

    async def create_answer(
        self,
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self._pc is None:
            raise RuntimeError("RTCPeerConnection has not been started")
        RTCSessionDescription = _load_rtc_session_description()
        sdp_text = str(sdp.get("sdp") or "")
        if ice_candidates and "a=candidate:" not in sdp_text:
            sdp_text = inject_ice_into_sdp(sdp_text, ice_candidates)
        if not ice_candidates and "a=candidate:" not in sdp_text:
            sdp_text = (
                sdp_text.replace("a=end-of-candidates\r\n", "")
                .replace("a=end-of-candidates\n", "")
            )
        logger.info(
            "SimpleX native WebRTC applying remote offer: call_id=%s sdp_chars=%d ice_candidates=%d",
            self._call_id,
            len(sdp_text),
            len(ice_candidates or []),
        )
        logger.info(
            "SimpleX native WebRTC remote offer diagnostics: call_id=%s %s",
            self._call_id,
            json.dumps(describe_sdp(sdp_text), sort_keys=True),
        )
        await self._emit_event("remote_offer_sdp", {"sdp": describe_sdp(sdp_text)})
        await self._pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=sdp_text,
                type=str(sdp.get("type") or "offer"),
            )
        )
        await self._apply_simplex_media_e2ee()
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        await self._wait_for_ice_gathering()
        local = self._pc.localDescription
        local_sdp_text = str(local.sdp or "")
        local_sdp_text, local_ice_candidates = prepare_local_offer_transport(
            local_sdp_text,
            self._ice_candidates,
            self.config,
        )
        logger.info(
            "SimpleX native WebRTC local answer diagnostics: call_id=%s %s",
            self._call_id,
            json.dumps(describe_sdp(local_sdp_text), sort_keys=True),
        )
        await self._emit_event("local_answer_sdp", {"sdp": describe_sdp(local_sdp_text)})
        return (
            {"type": local.type, "sdp": local_sdp_text},
            local_ice_candidates,
        )

    async def apply_answer(
        self,
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> None:
        if self._pc is None:
            raise RuntimeError("RTCPeerConnection has not been started")
        RTCSessionDescription = _load_rtc_session_description()
        sdp_text = str(sdp.get("sdp") or "")
        if ice_candidates and "a=candidate:" not in sdp_text:
            sdp_text = inject_ice_into_sdp(sdp_text, ice_candidates)
        if not ice_candidates and "a=candidate:" not in sdp_text:
            sdp_text = (
                sdp_text.replace("a=end-of-candidates\r\n", "")
                .replace("a=end-of-candidates\n", "")
            )
        logger.info(
            "SimpleX native WebRTC applying answer: call_id=%s sdp_chars=%d ice_candidates=%d",
            self._call_id,
            len(sdp_text),
            len(ice_candidates or []),
        )
        logger.info(
            "SimpleX native WebRTC remote answer diagnostics: call_id=%s %s",
            self._call_id,
            json.dumps(describe_sdp(sdp_text), sort_keys=True),
        )
        await self._emit_event("remote_answer_sdp", {"sdp": describe_sdp(sdp_text)})
        await self._pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=sdp_text,
                type=str(sdp.get("type") or "answer"),
            )
        )
        await self._apply_simplex_media_e2ee()
        logger.info("SimpleX native WebRTC answer applied: call_id=%s", self._call_id)

    async def add_extra_ice(self, ice_candidates: list[Any]) -> None:
        if self._pc is None:
            raise RuntimeError("RTCPeerConnection has not been started")
        added = 0
        for candidate in ice_candidates:
            rtc_candidate = _candidate_from_simplex(candidate)
            if rtc_candidate is not None:
                allow_trickle_ice_after_remote_description(self._pc)
                await self._pc.addIceCandidate(rtc_candidate)
                added += 1
        logger.info(
            "SimpleX native WebRTC extra ICE applied: call_id=%s candidates=%d",
            self._call_id,
            added,
        )
        await self._emit_event("extra_ice_applied", {"candidates": added})

    async def close(self) -> None:
        self._closing = True
        if self._no_inbound_watchdog_task is not None:
            self._no_inbound_watchdog_task.cancel()
            try:
                await self._no_inbound_watchdog_task
            except asyncio.CancelledError:
                pass
            self._no_inbound_watchdog_task = None
        current_task = asyncio.current_task()
        if self._audio_relay_task is not None:
            if self._audio_relay_task is not current_task:
                self._audio_relay_task.cancel()
                try:
                    await self._audio_relay_task
                except asyncio.CancelledError:
                    pass
            self._audio_relay_task = None
        pipeline = self._pipeline
        if pipeline is not None and getattr(pipeline, "is_streaming", False):
            await pipeline.aclose()
        if self._pc is not None:
            await self._pc.close()
            self._pc = None

    def _install_event_handlers(self) -> None:
        @self._pc.on("icecandidate")
        def on_ice_candidate(candidate):
            if candidate:
                self._ice_candidates.append(
                    {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    }
                )

        @self._pc.on("icegatheringstatechange")
        def on_ice_gathering_state_change():
            if not self._pc:
                return
            state = self._pc.iceGatheringState
            logger.info(
                "SimpleX native WebRTC ICE gathering state: call_id=%s state=%s",
                self._call_id,
                state,
            )
            self._schedule_event("ice_gathering_state", {"status": str(state)})
            if state == "complete":
                self._ice_gather_complete.set()

        @self._pc.on("iceconnectionstatechange")
        def on_ice_connection_state_change():
            if self._pc:
                state = self._pc.iceConnectionState
                logger.info(
                    "SimpleX native WebRTC ICE connection state: call_id=%s state=%s",
                    self._call_id,
                    state,
                )
                self._schedule_event("ice_connection_state", {"status": str(state)})
                if state in {"failed", "closed"} and not self._closing:
                    self._schedule_terminal(
                        "failed",
                        f"ice_connection_{state}",
                    )

        @self._pc.on("connectionstatechange")
        def on_connection_state_change():
            if self._pc:
                state = self._pc.connectionState
                logger.info(
                    "SimpleX native WebRTC peer connection state: call_id=%s state=%s",
                    self._call_id,
                    state,
                )
                self._schedule_event("connection_state", {"status": str(state)})
                if state == "connected":
                    self._schedule_no_inbound_watchdog()
                if state in {"failed", "closed"} and not self._closing:
                    self._schedule_terminal(
                        "failed",
                        f"peer_connection_{state}",
                    )

        @self._pc.on("signalingstatechange")
        def on_signaling_state_change():
            if self._pc:
                state = self._pc.signalingState
                logger.info(
                    "SimpleX native WebRTC signaling state: call_id=%s state=%s",
                    self._call_id,
                    state,
                )
                self._schedule_event("signaling_state", {"status": str(state)})

        @self._pc.on("track")
        def on_track(track):
            logger.info(
                "SimpleX native WebRTC remote track received: call_id=%s kind=%s",
                self._call_id,
                getattr(track, "kind", "unknown"),
            )
            self._schedule_event(
                "remote_track",
                {"kind": str(getattr(track, "kind", "unknown"))},
            )
            if track.kind == "audio" and self._accumulator is not None:
                self._schedule_simplex_media_e2ee()
                self._audio_relay_task = asyncio.create_task(
                    self._relay_audio(track)
                )

    async def _relay_audio(self, track) -> None:
        logger.info("SimpleX native WebRTC audio relay started: call_id=%s", self._call_id)
        try:
            while True:
                frame = await track.recv()
                self._remote_audio_frame_count += 1
                if self._remote_audio_frame_count == 1:
                    logger.info(
                        "SimpleX native WebRTC first remote audio frame: call_id=%s",
                        self._call_id,
                    )
                    await self._emit_event(
                        "first_remote_audio_frame",
                        {"remoteAudioFrames": self._remote_audio_frame_count},
                    )
                elif self._remote_audio_frame_count % 500 == 0:
                    logger.info(
                        "SimpleX native WebRTC remote audio frames: call_id=%s frames=%d",
                        self._call_id,
                        self._remote_audio_frame_count,
                    )
                pcm16 = _audio_frame_to_pcm16(frame)
                if pcm16 and self._accumulator is not None:
                    await self._accumulator.accept_pcm16(
                        pcm16,
                        sample_rate=int(
                            getattr(frame, "sample_rate", 0)
                            or self.config.sample_rate
                        ),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            reason = (
                "remote_audio_ended_before_first_frame"
                if self._remote_audio_frame_count == 0
                else "remote_audio_ended"
            )
            logger.warning(
                "SimpleX native WebRTC audio relay stopped with error: call_id=%s",
                self._call_id,
                exc_info=True,
            )
            self._schedule_terminal("failed", reason)

    async def _wait_for_ice_gathering(self) -> None:
        try:
            await asyncio.wait_for(
                self._ice_gather_complete.wait(),
                timeout=self.config.ice_gather_timeout,
            )
        except TimeoutError:
            logger.warning("SimpleX native WebRTC ICE gathering timed out")

    def _schedule_no_inbound_watchdog(self) -> None:
        if self._no_inbound_watchdog_task is not None:
            return
        self._no_inbound_watchdog_task = asyncio.create_task(
            self._no_inbound_watchdog()
        )

    def _schedule_simplex_media_e2ee(self) -> None:
        if self.config.enable_simplex_media_e2ee:
            asyncio.create_task(self._apply_simplex_media_e2ee())

    async def _apply_simplex_media_e2ee(self) -> None:
        if not self.config.enable_simplex_media_e2ee or self._pc is None:
            return
        shared_key = getattr(self, "_shared_media_key", None)
        if not shared_key:
            return
        try:
            transceivers = self._pc.getTransceivers()
        except Exception:
            logger.debug("Failed to inspect transceivers for SimpleX media E2EE")
            return
        for transceiver in transceivers or []:
            _patch_aiortc_sender_for_simplex_audio_e2ee(
                getattr(transceiver, "sender", None),
                shared_key,
            )
            _patch_aiortc_receiver_for_simplex_audio_e2ee(
                getattr(transceiver, "receiver", None),
                shared_key,
            )

    async def _no_inbound_watchdog(self) -> None:
        try:
            await asyncio.sleep(max(0.0, float(self.config.no_inbound_audio_timeout)))
            if (
                self._pc is None
                or self._closing
                or self._remote_audio_frame_count > 0
            ):
                return
            stats = await collect_webrtc_stats_summary_with_timeout(
                self._pc,
                timeout=self.config.no_inbound_stats_timeout,
            )
            peer_state = describe_peer_connection_state(self._pc)
            logger.warning(
                "SimpleX native WebRTC connected without inbound audio frames: "
                "call_id=%s stats=%s peer_state=%s",
                self._call_id,
                json.dumps(stats, sort_keys=True),
                json.dumps(peer_state, sort_keys=True),
            )
            await self._emit_event(
                "no_inbound_audio_frames",
                {"remoteAudioFrames": 0, "stats": stats, "peerState": peer_state},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "SimpleX native WebRTC no-inbound watchdog failed",
                extra={"call_id": self._call_id},
                exc_info=True,
            )

    def _schedule_terminal(self, status: str, reason: str) -> None:
        if self._terminal_reported:
            return
        self._terminal_reported = True
        asyncio.create_task(self._report_terminal(status, reason))

    def _schedule_event(self, event: str, details: dict[str, Any]) -> None:
        asyncio.create_task(self._emit_event(event, details))

    async def _emit_event(self, event: str, details: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        result = self._event_callback(event, details)
        if inspect.isawaitable(result):
            await result

    async def _report_terminal(self, status: str, reason: str) -> None:
        details: dict[str, Any] = {
            "remoteAudioFrames": self._remote_audio_frame_count,
        }
        if self._pc is not None:
            details["stats"] = await collect_webrtc_stats_summary_with_timeout(
                self._pc,
                timeout=self.config.no_inbound_stats_timeout,
            )
            details["peerState"] = describe_peer_connection_state(self._pc)
        logger.warning(
            "SimpleX native WebRTC terminal state: call_id=%s status=%s reason=%s details=%s",
            self._call_id,
            status,
            reason,
            json.dumps(details, sort_keys=True),
        )
        if self._terminal_callback is None:
            return
        result = self._terminal_callback(status, reason, details)
        if inspect.isawaitable(result):
            await result


def _load_aiortc_peer_types():
    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native calls require aiortc. Install the native call extra "
            "before enabling platforms.simplex.extra.native_calls."
        ) from exc
    return RTCPeerConnection, RTCConfiguration, RTCIceServer


def _load_rtc_session_description():
    try:
        from aiortc import RTCSessionDescription
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native calls require aiortc.",
        ) from exc
    return RTCSessionDescription


def apply_aiortc_transport_policy(policy: str) -> None:
    if str(policy or "").lower() != "relay":
        return
    try:
        from aioice.ice import TransportPolicy
        import aiortc.rtcicetransport as rtcice
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native relay ICE policy requires aiortc/aioice.",
        ) from exc
    patch_aiortc_connection_kwargs_for_relay(rtcice, TransportPolicy.RELAY)


def patch_aiortc_connection_kwargs_for_relay(
    rtcice_module: Any,
    relay_policy: Any,
) -> None:
    current = getattr(rtcice_module, "connection_kwargs", None)
    if not callable(current):
        return
    if getattr(current, "_hermes_simplex_relay_patch", False):
        return

    def _connection_kwargs_with_relay(servers):
        kwargs = current(servers)
        kwargs["transport_policy"] = relay_policy
        return kwargs

    _connection_kwargs_with_relay._hermes_simplex_relay_patch = True
    _connection_kwargs_with_relay._hermes_original_connection_kwargs = current
    rtcice_module.connection_kwargs = _connection_kwargs_with_relay


def allow_trickle_ice_after_remote_description(peer_connection: Any) -> None:
    try:
        transceivers = peer_connection.getTransceivers()
    except Exception:
        return
    for transceiver in transceivers or []:
        for endpoint_name in ("receiver", "sender"):
            endpoint = getattr(transceiver, endpoint_name, None)
            transport = getattr(endpoint, "transport", None)
            ice_transport = getattr(transport, "transport", None)
            connection = getattr(ice_transport, "_connection", None)
            if connection is None:
                continue
            if getattr(connection, "_remote_candidates_end", False):
                try:
                    connection._remote_candidates_end = False
                except Exception:
                    logger.debug(
                        "Failed to reset aiortc remote candidate end marker",
                        exc_info=True,
                    )


def describe_sdp(sdp: str) -> dict[str, Any]:
    """Return redacted SDP facts useful for SimpleX media debugging."""
    lines = [line.strip() for line in str(sdp or "").splitlines() if line.strip()]
    audio_m_line = next((line for line in lines if line.startswith("m=audio")), "")
    direction = next(
        (
            line[2:]
            for line in lines
            if line in {"a=sendrecv", "a=sendonly", "a=recvonly", "a=inactive"}
        ),
        "",
    )
    codec_by_payload: dict[str, str] = {}
    for line in lines:
        if not line.startswith("a=rtpmap:"):
            continue
        payload, _, codec = line[len("a=rtpmap:") :].partition(" ")
        codec_by_payload[payload] = codec.split("/", 1)[0].lower()
    candidate_types: dict[str, int] = {}
    for line in lines:
        if not line.startswith("a=candidate:"):
            continue
        parts = line.split()
        candidate_type = "unknown"
        if "typ" in parts:
            type_index = parts.index("typ") + 1
            if type_index < len(parts):
                candidate_type = parts[type_index]
        candidate_types[candidate_type] = candidate_types.get(candidate_type, 0) + 1
    setup = next((line[8:] for line in lines if line.startswith("a=setup:")), "")
    return {
        "audioMLinePayloads": audio_m_line.split()[3:] if audio_m_line else [],
        "candidateTypes": candidate_types,
        "codecs": sorted(set(codec_by_payload.values())),
        "direction": direction,
        "endOfCandidates": any(line == "a=end-of-candidates" for line in lines),
        "extmapCount": sum(1 for line in lines if line.startswith("a=extmap:")),
        "iceUfragPresent": any(line.startswith("a=ice-ufrag:") for line in lines),
        "mediaSections": sum(1 for line in lines if line.startswith("m=")),
        "midCount": sum(1 for line in lines if line.startswith("a=mid:")),
        "setup": setup,
    }


def retain_only_relay_candidates_in_sdp(sdp: str) -> str:
    sep = "\r\n" if "\r\n" in str(sdp or "") else "\n"
    result: list[str] = []
    for line in str(sdp or "").split(sep):
        if line.startswith("a=candidate:") and " typ relay" not in line:
            continue
        result.append(line)
    return sep.join(result)


def prepare_local_offer_transport(
    sdp: str,
    candidates: list[dict[str, Any]],
    config: SimplexAiortcConfig,
) -> tuple[str, list[dict[str, Any]]]:
    if bool(getattr(config, "prune_non_relay_candidates", False)):
        return retain_only_relay_candidates_in_sdp(sdp), _retain_only_relay_candidates(
            candidates
        )
    return sdp, list(candidates or [])


def _retain_only_relay_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in candidates or []:
        candidate_text = str(candidate.get("candidate") or "")
        if " typ relay" in candidate_text:
            retained.append(candidate)
    return retained


async def collect_webrtc_stats_summary(peer_connection: Any) -> dict[str, Any]:
    try:
        report = await peer_connection.getStats()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Failed to collect SimpleX native WebRTC stats", exc_info=True)
        return _empty_webrtc_stats_summary(collection_error="failed")

    summary = _empty_webrtc_stats_summary()
    for stat in report.values():
        stat_type = str(getattr(stat, "type", "") or "")
        kind = str(getattr(stat, "kind", "") or getattr(stat, "mediaType", "") or "")
        if stat_type == "inbound-rtp" and kind in {"audio", ""}:
            summary["inboundRtp"].append(
                {
                    "packetsReceived": int(getattr(stat, "packetsReceived", 0) or 0),
                    "bytesReceived": int(getattr(stat, "bytesReceived", 0) or 0),
                    "packetsLost": int(getattr(stat, "packetsLost", 0) or 0),
                    "jitter": float(getattr(stat, "jitter", 0.0) or 0.0),
                }
            )
        elif stat_type == "outbound-rtp" and kind in {"audio", ""}:
            summary["outboundRtp"].append(
                {
                    "packetsSent": int(getattr(stat, "packetsSent", 0) or 0),
                    "bytesSent": int(getattr(stat, "bytesSent", 0) or 0),
                }
            )
        elif stat_type == "candidate-pair":
            selected = bool(
                getattr(stat, "selected", False)
                or getattr(stat, "nominated", False)
                or getattr(stat, "state", "") == "succeeded"
            )
            if selected:
                summary["candidatePairs"].append(
                    {
                        "state": str(getattr(stat, "state", "") or ""),
                        "nominated": bool(getattr(stat, "nominated", False)),
                        "bytesSent": int(getattr(stat, "bytesSent", 0) or 0),
                        "bytesReceived": int(getattr(stat, "bytesReceived", 0) or 0),
                        "currentRoundTripTime": float(
                            getattr(stat, "currentRoundTripTime", 0.0) or 0.0
                        ),
                    }
                )
        elif stat_type in {"local-candidate", "remote-candidate"}:
            candidate_type = str(getattr(stat, "candidateType", "") or "unknown")
            key = (
                "localCandidateTypes"
                if stat_type == "local-candidate"
                else "remoteCandidateTypes"
            )
            bucket = summary[key]
            bucket[candidate_type] = bucket.get(candidate_type, 0) + 1
    summary["selectedIcePairs"] = selected_ice_pair_summaries(peer_connection)
    return summary


async def collect_webrtc_stats_summary_with_timeout(
    peer_connection: Any,
    *,
    timeout: float,
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            collect_webrtc_stats_summary(peer_connection),
            timeout=max(0.0, float(timeout)),
        )
    except TimeoutError:
        logger.warning("Timed out collecting SimpleX native WebRTC stats")
        return _empty_webrtc_stats_summary(collection_error="timeout")


def _empty_webrtc_stats_summary(
    *,
    collection_error: str = "",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "inboundRtp": [],
        "outboundRtp": [],
        "candidatePairs": [],
        "selectedIcePairs": [],
        "localCandidateTypes": {},
        "remoteCandidateTypes": {},
    }
    if collection_error:
        summary["collectionError"] = collection_error
    return summary


def selected_ice_pair_summaries(peer_connection: Any) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str, str]] = set()
    try:
        transceivers = peer_connection.getTransceivers()
    except Exception:
        return pairs
    for transceiver in transceivers or []:
        for endpoint_name in ("receiver", "sender"):
            endpoint = getattr(transceiver, endpoint_name, None)
            transport = getattr(endpoint, "transport", None)
            ice_transport = getattr(transport, "transport", None)
            connection = getattr(ice_transport, "_connection", None)
            nominated = getattr(connection, "_nominated", None)
            if not isinstance(nominated, dict):
                continue
            for component, pair in nominated.items():
                local = _candidate_summary(getattr(pair, "local_candidate", None))
                remote = _candidate_summary(getattr(pair, "remote_candidate", None))
                key = (
                    int(component or 0),
                    local.get("type", ""),
                    local.get("protocol", ""),
                    remote.get("type", ""),
                    remote.get("protocol", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                state = getattr(pair, "state", "")
                pairs.append(
                    {
                        "component": int(component or 0),
                        "state": str(getattr(state, "name", state) or ""),
                        "nominated": bool(getattr(pair, "nominated", False)),
                        "local": local,
                        "remote": remote,
                    }
                )
    return pairs


def _candidate_summary(candidate: Any) -> dict[str, Any]:
    return {
        "type": str(getattr(candidate, "type", "") or ""),
        "protocol": str(getattr(candidate, "transport", "") or ""),
        "component": int(getattr(candidate, "component", 0) or 0),
    }


def decode_simplex_media_key(shared_key: str) -> bytes:
    value = str(shared_key or "").strip()
    if not value:
        raise ValueError("SimpleX shared media key is empty")
    padded = value + ("=" * (-len(value) % 4))
    try:
        key = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise ValueError("SimpleX shared media key is not base64url") from exc
    if len(key) != 32:
        raise ValueError("SimpleX shared media key must decode to 32 bytes")
    return key


def encrypt_simplex_audio_payload(
    payload: bytes,
    key: bytes,
    *,
    iv: bytes | None = None,
) -> bytes:
    nonce = iv or secrets.token_bytes(12)
    if len(nonce) != 12:
        raise ValueError("SimpleX media IV must be 12 bytes")
    initial = payload[:1]
    plaintext = payload[1:]
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None) if plaintext else b""
    return initial + ciphertext + nonce


def decrypt_simplex_audio_payload(payload: bytes, key: bytes) -> bytes:
    if len(payload) <= 1:
        return payload
    if len(payload) < 13:
        raise ValueError("SimpleX encrypted audio payload is too short")
    initial = payload[:1]
    ciphertext = payload[1:-12]
    nonce = payload[-12:]
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None) if ciphertext else b""
    return initial + plaintext


def _patch_aiortc_sender_for_simplex_audio_e2ee(sender: Any, key: bytes) -> None:
    if sender is None or getattr(sender, "_hermes_simplex_media_e2ee", False):
        return
    original = getattr(sender, "_next_encoded_frame", None)
    if not callable(original):
        return

    async def _next_encoded_frame_with_simplex_e2ee(codec):
        encoded_frame = await original(codec)
        if encoded_frame is None:
            return None
        encrypted_payloads = [
            encrypt_simplex_audio_payload(bytes(payload), key)
            for payload in getattr(encoded_frame, "payloads", []) or []
        ]
        encoded_frame.payloads = encrypted_payloads
        return encoded_frame

    sender._next_encoded_frame = _next_encoded_frame_with_simplex_e2ee
    sender._hermes_simplex_media_e2ee = True


def _patch_aiortc_receiver_for_simplex_audio_e2ee(receiver: Any, key: bytes) -> None:
    if receiver is None or getattr(receiver, "_hermes_simplex_media_e2ee", False):
        return
    original = getattr(receiver, "_handle_rtp_packet", None)
    if not callable(original):
        return

    async def _handle_rtp_packet_with_simplex_e2ee(packet, arrival_time_ms):
        payload = getattr(packet, "payload", b"")
        if payload:
            try:
                packet.payload = decrypt_simplex_audio_payload(bytes(payload), key)
            except Exception:
                logger.debug("Failed to decrypt SimpleX audio RTP payload", exc_info=True)
                return
        return await original(packet, arrival_time_ms)

    receiver._handle_rtp_packet = _handle_rtp_packet_with_simplex_e2ee
    receiver._hermes_simplex_media_e2ee = True


def describe_peer_connection_state(peer_connection: Any) -> dict[str, Any]:
    """Return redacted WebRTC peer state for SimpleX media diagnostics."""
    summary: dict[str, Any] = {
        "connectionState": str(getattr(peer_connection, "connectionState", "") or ""),
        "iceConnectionState": str(
            getattr(peer_connection, "iceConnectionState", "") or ""
        ),
        "iceGatheringState": str(
            getattr(peer_connection, "iceGatheringState", "") or ""
        ),
        "signalingState": str(getattr(peer_connection, "signalingState", "") or ""),
        "transceivers": [],
    }
    try:
        transceivers = peer_connection.getTransceivers()
    except Exception:
        return summary
    for transceiver in transceivers or []:
        summary["transceivers"].append(
            {
                "mid": str(getattr(transceiver, "mid", "") or ""),
                "direction": str(getattr(transceiver, "direction", "") or ""),
                "currentDirection": str(
                    getattr(transceiver, "currentDirection", "") or ""
                ),
                "stopped": bool(getattr(transceiver, "stopped", False)),
                "receiver": _rtp_endpoint_state(
                    getattr(transceiver, "receiver", None)
                ),
                "sender": _rtp_endpoint_state(getattr(transceiver, "sender", None)),
            }
        )
    return summary


def _rtp_endpoint_state(endpoint: Any) -> dict[str, Any]:
    track = getattr(endpoint, "track", None)
    dtls_transport = getattr(endpoint, "transport", None)
    ice_transport = getattr(dtls_transport, "transport", None)
    connection = getattr(ice_transport, "_connection", None)
    return {
        "trackKind": str(getattr(track, "kind", "") or ""),
        "trackReadyState": str(getattr(track, "readyState", "") or ""),
        "dtlsState": str(getattr(dtls_transport, "state", "") or ""),
        "iceRole": str(getattr(ice_transport, "role", "") or ""),
        "iceState": str(getattr(ice_transport, "state", "") or ""),
        "remoteCandidatesEnd": bool(
            getattr(connection, "_remote_candidates_end", False)
        ),
    }


def _candidate_from_simplex(candidate: Any):
    if not isinstance(candidate, dict) or "candidate" not in candidate:
        return None
    try:
        from aioice.candidate import Candidate
        from aiortc import RTCIceCandidate
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native trickle ICE requires aiortc/aioice.",
        ) from exc
    parsed = Candidate.from_sdp(str(candidate["candidate"]))
    return RTCIceCandidate(
        component=parsed.component,
        foundation=parsed.foundation,
        ip=parsed.host,
        port=parsed.port,
        priority=parsed.priority,
        protocol=parsed.transport,
        type=parsed.type,
        relatedAddress=parsed.related_address,
        relatedPort=parsed.related_port,
        sdpMid=str(candidate.get("sdpMid") or "0"),
        sdpMLineIndex=int(candidate.get("sdpMLineIndex") or 0),
        tcpType=parsed.tcptype,
    )


def _audio_frame_to_pcm16(frame: Any) -> bytes:
    try:
        from av import AudioResampler

        sample_rate = int(getattr(frame, "sample_rate", 0) or 48000)
        resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
        converted_frames = resampler.resample(frame)
        chunks: list[bytes] = []
        for converted in converted_frames:
            samples = int(getattr(converted, "samples", 0) or 0)
            if samples <= 0 or not converted.planes:
                continue
            # PyAV planes can include padding beyond the valid samples.
            chunks.append(bytes(converted.planes[0])[: samples * 2])
        return b"".join(chunks)
    except Exception:
        logger.debug("Failed to convert WebRTC audio frame to PCM16", exc_info=True)
        return b""


def inject_ice_into_sdp(sdp: str, ice_candidates: list[Any]) -> str:
    sep = "\r\n" if "\r\n" in sdp else "\n"
    lines = sdp.split(sep)
    candidate_lines: list[str] = []
    for candidate in ice_candidates:
        if not isinstance(candidate, dict) or "candidate" not in candidate:
            continue
        candidate_text = str(candidate["candidate"])
        if not candidate_text.startswith(("candidate:", "a=candidate:")):
            candidate_text = f"candidate:{candidate_text}"
        if candidate_text.startswith("candidate:"):
            candidate_text = f"a={candidate_text}"
        candidate_lines.append(candidate_text)
    if not candidate_lines:
        return sdp

    result: list[str] = []
    in_media = False
    for line in lines:
        if line.startswith("m=") and in_media:
            result.extend(candidate_lines)
        result.append(line)
        if line.startswith(("m=audio", "m=video")):
            in_media = True
    if in_media:
        if result and result[-1] == "":
            result = result[:-1] + candidate_lines + [""]
        else:
            result.extend(candidate_lines)
    return sep.join(result)


def _file_size(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _create_queued_audio_track(
    sample_rate: int,
    event_sink: Callable[[str, dict[str, Any]], Any] | None = None,
):
    try:
        from aiortc.mediastreams import MediaStreamTrack
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native calls require aiortc.",
        ) from exc

    class QueuedFileAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._sample_rate = sample_rate
            self._pts = 0
            self._time_base = fractions.Fraction(1, sample_rate)
            self._queue: asyncio.Queue[str] = asyncio.Queue()
            self._current_frames: list[Any] = []
            self._current_audio_path = ""
            self._current_frames_sent = 0
            self._frame_index = 0
            self._start_time = 0.0
            self._frame_count = 0
            self._logged_first_frame = False

        async def queue_file(self, audio_path: str) -> None:
            await self._queue.put(audio_path)
            self._emit_playback_event(
                "outbound_tts_audio_queued",
                {
                    "audioBytes": _file_size(audio_path),
                },
            )
            logger.info("SimpleX native WebRTC queued TTS audio for playback")

        async def recv(self):
            await self._pace()
            if not self._logged_first_frame:
                logger.info("SimpleX native WebRTC first outbound audio frame")
                self._logged_first_frame = True
            if self._frame_index < len(self._current_frames):
                frame = self._current_frames[self._frame_index]
                self._frame_index += 1
                self._current_frames_sent += 1
                if self._current_frames_sent == 1:
                    self._emit_playback_event(
                        "outbound_tts_playback_started",
                        {
                            "audioBytes": _file_size(self._current_audio_path),
                            "frames": len(self._current_frames),
                        },
                    )
                if self._frame_index >= len(self._current_frames):
                    self._emit_playback_event(
                        "outbound_tts_playback_completed",
                        {
                            "audioBytes": _file_size(self._current_audio_path),
                            "frames": self._current_frames_sent,
                        },
                    )
                return frame
            try:
                audio_path = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return self._silence_frame()
            self._current_frames = await self._load_audio_frames(audio_path)
            self._current_audio_path = audio_path
            self._current_frames_sent = 0
            self._frame_index = 0
            if self._current_frames:
                frame = self._current_frames[self._frame_index]
                self._frame_index += 1
                self._current_frames_sent += 1
                self._emit_playback_event(
                    "outbound_tts_playback_started",
                    {
                        "audioBytes": _file_size(self._current_audio_path),
                        "frames": len(self._current_frames),
                    },
                )
                if self._frame_index >= len(self._current_frames):
                    self._emit_playback_event(
                        "outbound_tts_playback_completed",
                        {
                            "audioBytes": _file_size(self._current_audio_path),
                            "frames": self._current_frames_sent,
                        },
                    )
                return frame
            return self._silence_frame()

        def _emit_playback_event(self, event: str, details: dict[str, Any]) -> None:
            if event_sink is None:
                return
            try:
                event_sink(event, details)
            except Exception:
                logger.debug(
                    "SimpleX native WebRTC playback event sink failed",
                    exc_info=True,
                )

        async def _pace(self) -> None:
            if self._start_time == 0.0:
                self._start_time = time.monotonic()
            else:
                target = self._start_time + (self._frame_count * 0.02)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
            self._frame_count += 1

        async def _load_audio_frames(self, audio_path: str) -> list[Any]:
            raw_path = f"{audio_path}.raw"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(self._sample_rate),
                    "-ac",
                    "1",
                    raw_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except FileNotFoundError:
                logger.warning("ffmpeg is required for SimpleX call TTS playback")
                return []
            if not Path(raw_path).exists():
                return []
            raw_data = Path(raw_path).read_bytes()
            Path(raw_path).unlink(missing_ok=True)
            frame_bytes = (self._sample_rate // 50) * 2
            frames = []
            for offset in range(0, len(raw_data), frame_bytes):
                chunk = raw_data[offset : offset + frame_bytes]
                if len(chunk) < frame_bytes:
                    chunk += bytes(frame_bytes - len(chunk))
                frames.append(self._frame_from_pcm(chunk))
            return frames

        def _silence_frame(self):
            frame_bytes = (self._sample_rate // 50) * 2
            return self._frame_from_pcm(bytes(frame_bytes))

        def _frame_from_pcm(self, pcm16: bytes):
            from av import AudioFrame

            frame_samples = self._sample_rate // 50
            frame = AudioFrame(format="s16", layout="mono", samples=frame_samples)
            frame.sample_rate = self._sample_rate
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += frame_samples
            for plane in frame.planes:
                plane.update(pcm16)
            return frame

    return QueuedFileAudioTrack()


class _CapturingVoiceTurnPipeline:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.result: Any = None
        self.event = asyncio.Event()

    async def process_pcm16(
        self,
        *,
        call_id: str,
        pcm16: bytes,
        sample_rate: int,
    ) -> Any:
        self.result = await self.pipeline.process_pcm16(
            call_id=call_id,
            pcm16=pcm16,
            sample_rate=sample_rate,
        )
        self.event.set()
        return self.result


def _frame_from_pcm16(pcm16: bytes, *, sample_rate: int, pts: int):
    from av import AudioFrame

    samples = len(pcm16) // 2
    frame = AudioFrame(format="s16", layout="mono", samples=samples)
    frame.sample_rate = sample_rate
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, sample_rate)
    for plane in frame.planes:
        plane.update(pcm16)
    return frame


# Max buffered outbound frames before drop-oldest kicks in (back-pressure, I2).
_PCM_STREAMING_QUEUE_MAXSIZE = 50


def _create_pcm_streaming_track(target_rate: int):
    """Build a live outbound PCM ``MediaStreamTrack`` for the streaming pipeline.

    Plays the session's 16k TTS ``av.AudioFrame``s out the aiortc peer at the
    SimpleX rate (``target_rate``, 48k). A bounded queue (drop-oldest on overflow
    with a ``logger.warning``) provides back-pressure; a PERSISTENT
    ``av.AudioResampler`` keeps resample phase continuous across ``recv()`` calls.
    Wall-clock pacing is allowed here (outside ``streaming/**``).
    """
    try:
        from aiortc.mediastreams import MediaStreamTrack
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native calls require aiortc.",
        ) from exc

    class PcmStreamingTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            from av import AudioResampler

            self._target_rate = target_rate
            self._queue: asyncio.Queue[Any] = asyncio.Queue(
                maxsize=_PCM_STREAMING_QUEUE_MAXSIZE
            )
            # Persistent resampler: phase continuity across frames (M1).
            # frame_size chunks output into 20ms frames; a single large TTS
            # chunk therefore resamples to MULTIPLE output frames in one
            # resample() call (e.g. 12000 samples -> 12x 960-sample frames).
            self._frame_size = target_rate // 50
            self._resampler = AudioResampler(
                format="s16",
                layout="mono",
                rate=target_rate,
                frame_size=self._frame_size,
            )
            # Buffer for resampled frames not yet emitted: resample() can return
            # more than one frame, and recv() emits one per call (drained FIFO).
            # Note: with frame_size set, the resampler buffers (<frame_size)
            # samples on the first input, so the very first recv() emits one
            # 20ms silence frame before real audio (inaudible one-time latency).
            self._pending_out: deque[Any] = deque()
            self._pts = 0
            self._time_base = fractions.Fraction(1, target_rate)
            self._start_time = 0.0
            self._frame_count = 0

        async def enqueue(self, frame: Any) -> None:
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.warning(
                    "SimpleX native WebRTC outbound PCM queue full; dropping oldest frame"
                )
                self._queue.put_nowait(frame)

        def drop_pending(self) -> int:
            count = 0
            while True:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                count += 1
            # Barge-in must also drop already-resampled-but-unsent frames, else
            # buffered audio keeps playing after the flush. The transport derives
            # its own dropped-frame metric from its pending list and ignores this
            # return value, so counting these toward the total is safe.
            count += len(self._pending_out)
            self._pending_out.clear()
            return count

        async def recv(self):
            await self._pace()
            # Drain any frames left over from a prior multi-frame resample().
            if self._pending_out:
                return self._pending_out.popleft()
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return self._silence_frame()
            converted_frames = self._resampler.resample(frame)
            for converted in converted_frames:
                samples = int(getattr(converted, "samples", 0) or 0)
                if samples <= 0:
                    continue
                converted.pts = self._pts
                converted.time_base = self._time_base
                self._pts += samples
                self._pending_out.append(converted)
            if self._pending_out:
                return self._pending_out.popleft()
            # Resampler buffered everything (0 output frames): emit silence.
            return self._silence_frame()

        async def _pace(self) -> None:
            if self._start_time == 0.0:
                self._start_time = time.monotonic()
            else:
                target = self._start_time + (self._frame_count * 0.02)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
            self._frame_count += 1

        def _silence_frame(self):
            frame_bytes = (self._target_rate // 50) * 2
            frame = _frame_from_pcm16(
                bytes(frame_bytes), sample_rate=self._target_rate, pts=self._pts
            )
            self._pts += self._target_rate // 50
            return frame

    return PcmStreamingTrack()


class _DirectFeedAccumulator:
    """Direct-feed inbound accumulator for the streaming pipeline.

    Bypasses the turn-based RMS ``AudioUtteranceAccumulator``: the session is the
    turn detector, so every relayed frame is fed straight to ``process_pcm16``.
    The ack is discarded — outbound audio flows out via the transport sink.
    Signature matches the relay's ``await accumulator.accept_pcm16(pcm16)``.
    """

    def __init__(self, pipeline: Any, call_id: str, native_rate: int) -> None:
        self._pipeline = pipeline
        self._call_id = str(call_id or "")
        self._native_rate = int(native_rate)

    async def accept_pcm16(
        self,
        pcm16: bytes,
        *,
        now: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        await self._pipeline.process_pcm16(
            call_id=self._call_id,
            pcm16=pcm16,
            sample_rate=sample_rate or self._native_rate,
        )


def _read_wav_pcm16_mono(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError("simulation audio WAV must be 16-bit PCM")
    if channels == 1:
        return frames, sample_rate
    frame_width = sample_width * channels
    mono = []
    for offset in range(0, len(frames), frame_width):
        mono.append(frames[offset : offset + sample_width])
    return b"".join(mono), sample_rate


def _load_simulation_audio_pcm16(path: Path, sample_rate: int) -> bytes:
    if path.suffix.lower() == ".wav":
        pcm16, source_rate = _read_wav_pcm16_mono(path)
        if source_rate == sample_rate:
            return pcm16

    raw_path = path.with_name(
        f"{path.name}.{secrets.token_hex(4)}.{sample_rate}.s16le"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(raw_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return raw_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for non-native simulation audio") from exc
    finally:
        raw_path.unlink(missing_ok=True)


def _create_file_audio_track(audio_path: Path, sample_rate: int):
    try:
        from aiortc.mediastreams import MediaStreamTrack
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native voice-turn simulation requires aiortc.",
        ) from exc

    pcm16 = _load_simulation_audio_pcm16(audio_path, sample_rate)
    frame_bytes = (sample_rate // 50) * 2
    chunks = [
        pcm16[offset : offset + frame_bytes]
        for offset in range(0, len(pcm16), frame_bytes)
    ]
    if not chunks:
        chunks = [bytes(frame_bytes)]
    if len(chunks[-1]) < frame_bytes:
        chunks[-1] += bytes(frame_bytes - len(chunks[-1]))
    silence = bytes(frame_bytes)

    class FileAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._sample_rate = sample_rate
            self._pts = 0
            self._index = 0
            self._start_time = 0.0
            self._frame_count = 0

        async def recv(self):
            await self._pace()
            if self._index < len(chunks):
                chunk = chunks[self._index]
                self._index += 1
            else:
                chunk = silence
            frame = _frame_from_pcm16(
                chunk,
                sample_rate=self._sample_rate,
                pts=self._pts,
            )
            self._pts += sample_rate // 50
            return frame

        async def _pace(self) -> None:
            if self._start_time == 0.0:
                self._start_time = time.monotonic()
            else:
                target = self._start_time + (self._frame_count * 0.02)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
            self._frame_count += 1

    return FileAudioTrack()


def _parse_tts_audio_path(raw: str) -> Path:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("caller audio TTS returned invalid JSON") from exc
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(str(payload.get("error") if isinstance(payload, dict) else "") or "caller audio TTS failed")
    file_path = payload.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise RuntimeError("caller audio TTS did not return a file path")
    return Path(file_path)


async def _generate_simulation_caller_audio(
    *,
    call_id: str,
    caller_text: str,
) -> Path:
    from tools.tts_tool import text_to_speech_tool

    audio_dir = get_hermes_home() / "cache" / "calls" / "simulation"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{_safe_call_id(call_id)}-caller.wav"
    raw = await asyncio.to_thread(
        text_to_speech_tool,
        text=caller_text,
        output_path=str(audio_path),
    )
    return _parse_tts_audio_path(str(raw or ""))


def _transcript_expectation(
    result: Any,
    expected_transcript: str | None,
) -> bool | None:
    expected = str(expected_transcript or "").strip().lower()
    if not expected:
        return None
    transcript = str(getattr(result, "transcript", "") or "").strip().lower()
    return expected in transcript


async def run_aiortc_voice_turn_simulation(
    *,
    call_id: str = "simulated-voice-call",
    contact_id: str = "simulated-contact",
    audio_path: str | Path | None = None,
    caller_text: str = "Hermes simulation check.",
    expected_transcript: str | None = None,
    trace_root: Path | None = None,
    timeout_seconds: float = 12.0,
    pipeline_factory: Callable[[], Any] | None = None,
) -> AiortcVoiceTurnSimulationResult:
    """Run the full native WebRTC voice loop without SimpleX or a phone."""
    try:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:
        trace_path = NativeCallTraceWriter(root=trace_root).record(
            call_id,
            "simulation_failed",
            code="call_simplex_voice_turn_dependency_missing",
            message=str(exc),
        )
        return AiortcVoiceTurnSimulationResult(
            ok=False,
            code="call_simplex_voice_turn_dependency_missing",
            message=f"aiortc unavailable: {exc}",
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
        )

    tracer = NativeCallTraceWriter(root=trace_root)
    events: list[str] = []

    def record(event: str, **fields: Any) -> Path:
        events.append(event)
        return tracer.record(call_id, event, contact_id=contact_id, **fields)

    trace_path = record("simulation_started", caller_text_chars=len(caller_text or ""))
    try:
        caller_audio = (
            Path(audio_path).expanduser()
            if audio_path is not None
            else await _generate_simulation_caller_audio(
                call_id=call_id,
                caller_text=caller_text,
            )
        )
    except Exception as exc:
        trace_path = record(
            "simulation_failed",
            code="call_simplex_voice_turn_audio_unavailable",
            message=str(exc),
        )
        return AiortcVoiceTurnSimulationResult(
            ok=False,
            code="call_simplex_voice_turn_audio_unavailable",
            message=str(exc),
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
            events=events,
        )

    capture: _CapturingVoiceTurnPipeline | None = None

    def _pipeline_factory() -> _CapturingVoiceTurnPipeline:
        nonlocal capture
        base = (
            pipeline_factory()
            if pipeline_factory is not None
            else HermesVoiceTurnPipeline(tracer=tracer)
        )
        capture = _CapturingVoiceTurnPipeline(base)
        return capture

    config = SimplexAiortcConfig(
        ice_servers=[],
        ice_gather_timeout=2.0,
        ice_transport_policy="all",
        voice_rms_threshold=100.0,
        silence_seconds=0.25,
        no_inbound_audio_timeout=max(3.0, float(timeout_seconds)),
    )
    engine = SimplexAiortcMediaEngine(
        config=config,
        pipeline_factory=_pipeline_factory,
    )

    def _record_media_event(event: dict[str, Any]) -> None:
        if event.get("type") == "event":
            record(
                "native_media_event",
                media_event=str(event.get("event") or ""),
                details=event.get("details") if isinstance(event.get("details"), dict) else {},
            )
        elif event.get("type") == "status":
            record(
                "native_media_event",
                media_event="status",
                status=str(event.get("status") or ""),
                reason=str(event.get("reasonCode") or ""),
                details=event.get("details") if isinstance(event.get("details"), dict) else {},
            )

    engine.set_event_sink(_record_media_event)
    remote = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    remote_received_audio_frames = 0
    remote_received_non_silent_frames = 0
    remote_non_silent_event = asyncio.Event()
    remote_consumer: asyncio.Task | None = None

    async def _consume_remote_audio(track) -> None:
        nonlocal remote_received_audio_frames, remote_received_non_silent_frames
        while True:
            frame = await track.recv()
            remote_received_audio_frames += 1
            pcm16 = _audio_frame_to_pcm16(frame)
            if pcm16_rms(pcm16) > 50.0:
                remote_received_non_silent_frames += 1
                remote_non_silent_event.set()

    @remote.on("track")
    def on_track(track):
        nonlocal remote_consumer
        if getattr(track, "kind", "") == "audio":
            remote_consumer = asyncio.create_task(_consume_remote_audio(track))

    offer_sent = False
    answer_applied = False
    connected = False
    local_sdp_summary: dict[str, Any] = {}
    remote_sdp_summary: dict[str, Any] = {}
    result: Any = None
    try:
        offer = await engine.start_incoming(
            {"callId": call_id, "contactId": contact_id}
        )
        trace_path = record("native_call_registered")
        local_sdp_summary = describe_sdp(str(offer.sdp.get("sdp") or ""))
        trace_path = record(
            "simulation_offer_sent",
            payload={
                "rtcSession": offer.sdp,
                "rtcIceCandidates": offer.ice_candidates,
            },
        )
        offer_sent = True
        remote.addTrack(_create_file_audio_track(caller_audio, config.sample_rate))
        await remote.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(offer.sdp.get("sdp") or ""),
                type=str(offer.sdp.get("type") or "offer"),
            )
        )
        answer = await remote.createAnswer()
        await remote.setLocalDescription(answer)
        await _wait_for_remote_ice_gathering(remote, timeout=2.0)
        remote_description = remote.localDescription
        remote_sdp_summary = describe_sdp(str(remote_description.sdp or ""))
        answer_payload = {
            "type": str(remote_description.type or "answer"),
            "sdp": str(remote_description.sdp or ""),
        }
        trace_path = record(
            "native_signal_received",
            signal_type="answer",
            payload={"rtcSession": answer_payload, "rtcIceCandidates": []},
        )
        await engine.apply_answer(call_id, answer_payload, [])
        answer_applied = True
        trace_path = record("simulation_answer_applied")

        peer = engine._require_peer(call_id)
        local_pc = getattr(peer, "_pc", None)
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            local_state = str(getattr(local_pc, "connectionState", "") or "")
            remote_state = str(getattr(remote, "connectionState", "") or "")
            connected = connected or local_state == "connected" or remote_state == "connected"
            if capture is not None and capture.event.is_set():
                result = capture.result
            if (
                connected
                and result is not None
                and remote_non_silent_event.is_set()
            ):
                trace_path = record("simulation_outbound_audio_received")
                break
            await asyncio.sleep(0.05)

        inbound_frames = int(getattr(peer, "_remote_audio_frame_count", 0) or 0)
        if result is None and capture is not None and capture.event.is_set():
            result = capture.result
        expected_present = _transcript_expectation(result, expected_transcript)
        transcript_chars = len(str(getattr(result, "transcript", "") or ""))
        response_chars = len(str(getattr(result, "response_text", "") or ""))
        tts_audio_path = getattr(result, "audio_path", None)
        tts_audio_bytes = (
            Path(tts_audio_path).stat().st_size
            if tts_audio_path and Path(tts_audio_path).exists()
            else 0
        )
        ok = (
            offer_sent
            and answer_applied
            and connected
            and inbound_frames > 0
            and bool(getattr(result, "ok", False))
            and transcript_chars > 0
            and response_chars > 0
            and tts_audio_bytes > 0
            and remote_received_non_silent_frames > 0
            and expected_present is not False
        )
        if ok:
            trace_path = record(
                "simulation_completed",
                inbound_audio_frames=inbound_frames,
                transcript_chars=transcript_chars,
                agent_response_chars=response_chars,
                tts_audio_bytes=tts_audio_bytes,
                remote_received_audio_frames=remote_received_audio_frames,
                remote_received_non_silent_frames=remote_received_non_silent_frames,
                expected_transcript_present=expected_present,
            )
            return AiortcVoiceTurnSimulationResult(
                ok=True,
                code="call_simplex_voice_turn_simulation_passed",
                message="SimpleX simulated voice turn passed.",
                call_id=call_id,
                contact_id=contact_id,
                trace_path=trace_path,
                offer_sent=offer_sent,
                answer_applied=answer_applied,
                connected=connected,
                inbound_audio_frames=inbound_frames,
                transcript_chars=transcript_chars,
                expected_transcript_present=expected_present,
                agent_response_chars=response_chars,
                tts_audio_bytes=tts_audio_bytes,
                remote_received_audio_frames=remote_received_audio_frames,
                remote_received_non_silent_frames=remote_received_non_silent_frames,
                local_sdp=local_sdp_summary,
                remote_sdp=remote_sdp_summary,
                events=events,
            )
        if expected_present is False:
            code = "call_simplex_voice_turn_transcript_mismatch"
            message = "Simulated STT transcript did not contain the expected text."
        else:
            code = "call_simplex_voice_turn_simulation_failed"
            message = (
                "Timed out before the simulated caller received Hermes response audio."
            )
        trace_path = record(
            "simulation_failed",
            code=code,
            message=message,
            offer_sent=offer_sent,
            answer_applied=answer_applied,
            connected=connected,
            inbound_audio_frames=inbound_frames,
            transcript_chars=transcript_chars,
            agent_response_chars=response_chars,
            tts_audio_bytes=tts_audio_bytes,
            remote_received_audio_frames=remote_received_audio_frames,
            remote_received_non_silent_frames=remote_received_non_silent_frames,
            expected_transcript_present=expected_present,
        )
        return AiortcVoiceTurnSimulationResult(
            ok=False,
            code=code,
            message=message,
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
            offer_sent=offer_sent,
            answer_applied=answer_applied,
            connected=connected,
            inbound_audio_frames=inbound_frames,
            transcript_chars=transcript_chars,
            expected_transcript_present=expected_present,
            agent_response_chars=response_chars,
            tts_audio_bytes=tts_audio_bytes,
            remote_received_audio_frames=remote_received_audio_frames,
            remote_received_non_silent_frames=remote_received_non_silent_frames,
            local_sdp=local_sdp_summary,
            remote_sdp=remote_sdp_summary,
            events=events,
        )
    finally:
        if remote_consumer is not None:
            remote_consumer.cancel()
            try:
                await remote_consumer
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        await engine.stop(call_id)
        await remote.close()


async def run_aiortc_loopback_probe(
    *,
    timeout_seconds: float = 8.0,
    require_voice_turn: bool = False,
) -> AiortcLoopbackProbeResult:
    """Run an in-process aiortc media probe against the native receiver path."""
    try:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:
        return AiortcLoopbackProbeResult(
            ok=False,
            remote_audio_frames=0,
            local_sdp={},
            remote_sdp={},
            message=f"aiortc unavailable: {exc}",
        )

    class _ProbePipeline:
        def __init__(self) -> None:
            self.voice_turns = 0
            self.voice_pcm_bytes = 0
            self.voice_turn_event = asyncio.Event()

        async def process_pcm16(self, *, call_id: str, pcm16: bytes, sample_rate: int):
            self.voice_turns += 1
            self.voice_pcm_bytes += len(pcm16)
            self.voice_turn_event.set()
            return VoiceTurnResult(
                ok=False,
                code="call_aiortc_loopback_probe",
                message="Loopback probe does not invoke the live agent.",
            )

    config = SimplexAiortcConfig(
        ice_servers=[],
        ice_gather_timeout=2.0,
        ice_transport_policy="all",
        voice_rms_threshold=100.0,
        silence_seconds=0.25,
    )
    probe_pipeline = _ProbePipeline()
    engine = SimplexAiortcMediaEngine(
        config=config,
        pipeline_factory=lambda: probe_pipeline,
    )
    remote = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    local_sdp_summary: dict[str, Any] = {}
    remote_sdp_summary: dict[str, Any] = {}
    try:
        offer = await engine.start_incoming(
            {"callId": "loopback-probe", "contactId": "loopback-probe"}
        )
        local_sdp_summary = describe_sdp(str(offer.sdp.get("sdp") or ""))
        remote.addTrack(
            _create_probe_audio_track(
                config.sample_rate,
                tone_frames=25 if require_voice_turn else None,
            )
        )
        await remote.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(offer.sdp.get("sdp") or ""),
                type=str(offer.sdp.get("type") or "offer"),
            )
        )
        answer = await remote.createAnswer()
        await remote.setLocalDescription(answer)
        await _wait_for_remote_ice_gathering(remote, timeout=2.0)
        remote_description = remote.localDescription
        remote_sdp_summary = describe_sdp(str(remote_description.sdp or ""))
        await engine.apply_answer(
            "loopback-probe",
            {
                "type": str(remote_description.type or "answer"),
                "sdp": str(remote_description.sdp or ""),
            },
            [],
        )
        peer = engine._require_peer("loopback-probe")
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            frames = int(getattr(peer, "_remote_audio_frame_count", 0) or 0)
            if frames > 0 and (
                not require_voice_turn or probe_pipeline.voice_turn_event.is_set()
            ):
                stats = {}
                pc = getattr(peer, "_pc", None)
                if pc is not None:
                    stats = await collect_webrtc_stats_summary(pc)
                return AiortcLoopbackProbeResult(
                    ok=True,
                    remote_audio_frames=frames,
                    local_sdp=local_sdp_summary,
                    remote_sdp=remote_sdp_summary,
                    stats=stats,
                    voice_turns=probe_pipeline.voice_turns,
                    voice_pcm_bytes=probe_pipeline.voice_pcm_bytes,
                )
            await asyncio.sleep(0.05)
        frames = int(getattr(peer, "_remote_audio_frame_count", 0) or 0)
        return AiortcLoopbackProbeResult(
            ok=False,
            remote_audio_frames=frames,
            local_sdp=local_sdp_summary,
            remote_sdp=remote_sdp_summary,
            message=(
                "Timed out waiting for loopback voice turn input."
                if require_voice_turn
                else "Timed out waiting for loopback RTP audio frames."
            ),
            voice_turns=probe_pipeline.voice_turns,
            voice_pcm_bytes=probe_pipeline.voice_pcm_bytes,
        )
    finally:
        await engine.stop("loopback-probe")
        await remote.close()


async def _wait_for_remote_ice_gathering(peer_connection: Any, *, timeout: float) -> None:
    if getattr(peer_connection, "iceGatheringState", "") == "complete":
        return
    complete = asyncio.Event()

    @peer_connection.on("icegatheringstatechange")
    def on_ice_gathering_state_change():
        if getattr(peer_connection, "iceGatheringState", "") == "complete":
            complete.set()

    try:
        await asyncio.wait_for(complete.wait(), timeout=timeout)
    except TimeoutError:
        logger.warning("SimpleX native loopback probe ICE gathering timed out")


def _create_probe_audio_track(sample_rate: int, *, tone_frames: int | None = None):
    try:
        from aiortc.mediastreams import MediaStreamTrack
        from av import AudioFrame
    except ImportError as exc:
        raise SimplexNativeSidecarError(
            "call_simplex_native_dependency_missing",
            "SimpleX native loopback probe requires aiortc and PyAV.",
        ) from exc

    class ProbeToneTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._sample_rate = sample_rate
            self._pts = 0
            self._time_base = fractions.Fraction(1, sample_rate)
            self._start_time = 0.0
            self._frame_count = 0
            self._tone_frames = tone_frames

        async def recv(self):
            await self._pace()
            frame_samples = self._sample_rate // 50
            frame = AudioFrame(format="s16", layout="mono", samples=frame_samples)
            frame.sample_rate = self._sample_rate
            frame.pts = self._pts
            frame.time_base = self._time_base
            base = self._pts
            self._pts += frame_samples
            pcm = bytearray()
            for offset in range(frame_samples):
                value = 0
                if self._tone_frames is None or self._frame_count <= self._tone_frames:
                    value = int(
                        12000
                        * math.sin(
                            2 * math.pi * 440 * ((base + offset) / self._sample_rate)
                        )
                    )
                pcm.extend(value.to_bytes(2, "little", signed=True))
            for plane in frame.planes:
                plane.update(bytes(pcm))
            return frame

        async def _pace(self) -> None:
            if self._start_time == 0.0:
                self._start_time = time.monotonic()
            else:
                target = self._start_time + (self._frame_count * 0.02)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
            self._frame_count += 1

    return ProbeToneTrack()


async def run_simplex_aiortc_sidecar() -> None:
    from .simplex_sidecar import run_jsonl_sidecar
    from .streaming.engine import build_native_pipeline, select_call_engine

    hermes_cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        hermes_cfg = load_config() or {}
    except Exception:
        pass  # config unavailable — selector falls back to turn_based

    selected_engine = select_call_engine(hermes_cfg)
    logger.info("native call engine: %s", selected_engine)

    config = SimplexAiortcConfig(
        ice_transport_policy=os.getenv("SIMPLEX_NATIVE_ICE_POLICY", "relay"),
        prune_non_relay_candidates=str(
            os.getenv("SIMPLEX_NATIVE_PRUNE_NON_RELAY", "")
        ).lower()
        in {"1", "true", "yes", "on"},
        enable_simplex_media_e2ee=str(
            os.getenv("SIMPLEX_NATIVE_ENABLE_MEDIA_E2EE", "")
        ).lower()
        in {"1", "true", "yes", "on"},
    )
    # Wire the engine selector: turn_based delegates to HermesVoiceTurnPipeline
    # (identical behavior to before); streaming routes to the deferred Pipecat seam.
    engine = SimplexAiortcMediaEngine(
        config=config,
        pipeline_factory=lambda: build_native_pipeline(
            hermes_cfg,
            turn_based_factory=HermesVoiceTurnPipeline,
        ),
    )
    await run_jsonl_sidecar(engine)
