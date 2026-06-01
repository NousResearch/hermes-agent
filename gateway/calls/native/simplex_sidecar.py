from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO

from .simplex_session_codec import (
    decode_webrtc_session,
    encode_webrtc_session,
)

logger = logging.getLogger(__name__)


class SimplexNativeSidecarError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SimplexSidecarOffer:
    sdp: dict[str, Any]
    ice_candidates: list[Any]
    capabilities: dict[str, Any] = field(default_factory=dict)
    call_dh_pub_key: str | None = None


@dataclass(frozen=True)
class SimplexSidecarAnswer:
    sdp: dict[str, Any]
    ice_candidates: list[Any]


class SimplexNativeMediaEngine(Protocol):
    async def start_incoming(self, request: dict[str, Any]) -> SimplexSidecarOffer:
        raise NotImplementedError

    async def start_outgoing_answer(
        self,
        request: dict[str, Any],
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> SimplexSidecarAnswer:
        raise NotImplementedError

    async def apply_answer(
        self,
        call_id: str,
        sdp: dict[str, Any],
        ice_candidates: list[Any],
    ) -> None:
        raise NotImplementedError

    async def add_extra_ice(self, call_id: str, ice_candidates: list[Any]) -> None:
        raise NotImplementedError

    async def stop(self, call_id: str) -> None:
        raise NotImplementedError

    async def process_audio_file(self, call_id: str, audio_path: str) -> Any:
        raise NotImplementedError

    def set_event_sink(self, event_sink) -> None:
        raise NotImplementedError


class SimplexNativeSidecarRunner:
    def __init__(self, media_engine: SimplexNativeMediaEngine) -> None:
        self.media_engine = media_engine

    async def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        message_type = str(message.get("type") or "")
        call_id = str(message.get("callId") or "")

        if message_type == "start_incoming":
            return await self._handle_start_incoming(message)
        if message_type == "start_outgoing_answer":
            return await self._handle_start_outgoing_answer(message)
        if message_type == "apply_answer":
            await self._handle_apply_answer(call_id, message.get("answer"))
            return None
        if message_type == "answer":
            await self._handle_apply_answer(call_id, message)
            return None
        if message_type == "add_extra_ice":
            await self._handle_add_extra_ice(call_id, message.get("extra"))
            return None
        if message_type == "extra":
            await self._handle_add_extra_ice(call_id, message)
            return None
        if message_type == "stop":
            await self.media_engine.stop(call_id)
            return None
        if message_type == "debug_process_audio":
            return await self._handle_debug_process_audio(call_id, message.get("audioPath"))

        logger.warning("Ignoring unknown SimpleX native sidecar message")
        return None

    async def _handle_start_incoming(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = {
            "callId": str(message.get("callId") or ""),
            "contactId": str(message.get("contactId") or ""),
            "media": str(message.get("media") or "audio"),
            "encrypted": bool(message.get("encrypted")),
            "sharedKey": message.get("sharedKey"),
        }
        try:
            offer = await self.media_engine.start_incoming(request)
        except SimplexNativeSidecarError as exc:
            logger.warning(
                "Native SimpleX media sidecar start failed",
                extra={"code": exc.code},
            )
            return {
                "ok": False,
                "code": exc.code,
                "message": exc.message,
            }
        except Exception:
            logger.warning("Native SimpleX media sidecar failed to start", exc_info=True)
            return {
                "ok": False,
                "code": "call_simplex_native_sidecar_failed",
                "message": "native SimpleX media sidecar failed to start",
            }

        encoded_offer = encode_webrtc_session(offer.sdp, offer.ice_candidates)
        payload: dict[str, Any] = {
            "rtcSession": encoded_offer["rtcSession"],
            "rtcIceCandidates": encoded_offer["rtcIceCandidates"],
            "capabilities": dict(offer.capabilities or {}),
        }
        if offer.call_dh_pub_key:
            payload["callDhPubKey"] = offer.call_dh_pub_key
        return {"ok": True, "offer": payload}

    async def _handle_start_outgoing_answer(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = {
            "callId": str(message.get("callId") or ""),
            "contactId": str(message.get("contactId") or ""),
            "media": str(message.get("media") or "audio"),
            "encrypted": bool(message.get("encrypted")),
            "sharedKey": message.get("sharedKey"),
        }
        try:
            offer = message.get("offer")
            if not isinstance(offer, Mapping):
                raise ValueError("offer must be an object")
            sdp, ice_candidates = decode_webrtc_session(offer)
            answer = await self.media_engine.start_outgoing_answer(
                request,
                sdp,
                ice_candidates,
            )
        except SimplexNativeSidecarError as exc:
            logger.warning(
                "Native SimpleX media sidecar outgoing answer failed",
                extra={"code": exc.code},
            )
            return {
                "ok": False,
                "code": exc.code,
                "message": exc.message,
            }
        except Exception:
            logger.warning(
                "Native SimpleX media sidecar failed to answer outgoing call",
                exc_info=True,
            )
            return {
                "ok": False,
                "code": "call_simplex_native_sidecar_failed",
                "message": "native SimpleX media sidecar failed to answer call",
            }

        encoded_answer = encode_webrtc_session(answer.sdp, answer.ice_candidates)
        return {"ok": True, "answer": encoded_answer}

    async def _handle_apply_answer(self, call_id: str, answer: Any) -> None:
        if not isinstance(answer, Mapping):
            answer = {
                "rtcSession": answer,
                "rtcIceCandidates": "",
            } if isinstance(answer, str) else None
        if not isinstance(answer, Mapping):
            answer = None
        if answer is None:
            logger.warning("Ignoring malformed SimpleX native call answer")
            return
        if (
            "rtcSession" not in answer
            and isinstance(answer.get("answer") if isinstance(answer, Mapping) else None, str)
        ):
            answer = {
                "rtcSession": answer.get("answer"),
                "rtcIceCandidates": answer.get("iceCandidates", ""),
            }
        try:
            sdp, ice_candidates = decode_webrtc_session(answer)
            await self.media_engine.apply_answer(call_id, sdp, ice_candidates)
        except Exception:
            logger.warning(
                "Failed to apply SimpleX native call answer",
                extra={"call_id": call_id},
                exc_info=True,
            )

    async def _handle_add_extra_ice(self, call_id: str, extra: Any) -> None:
        if not isinstance(extra, Mapping):
            extra = {
                "rtcIceCandidates": extra,
            } if isinstance(extra, str) else None
        if not isinstance(extra, Mapping):
            logger.warning("Ignoring malformed SimpleX native call extra ICE")
            return
        try:
            payload = extra.get("rtcExtraInfo", extra)
            if not isinstance(payload, Mapping):
                raise ValueError("rtcExtraInfo must be an object")
            ice_value = payload.get("rtcIceCandidates")
            if not isinstance(ice_value, str):
                raise ValueError("rtcIceCandidates must be a compressed string")
            decoded = decode_extra_ice_candidates(ice_value)
            await self.media_engine.add_extra_ice(call_id, decoded)
        except Exception:
            logger.warning(
                "Failed to add SimpleX native call extra ICE",
                extra={"call_id": call_id},
                exc_info=True,
            )

    async def _handle_debug_process_audio(
        self,
        call_id: str,
        audio_path: Any,
    ) -> dict[str, Any]:
        if not isinstance(audio_path, str) or not audio_path:
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_invalid",
                "message": "debug audio path is required",
            }
        process_audio_file = getattr(self.media_engine, "process_audio_file", None)
        if not callable(process_audio_file):
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_unavailable",
                "message": "native SimpleX media engine cannot process debug audio",
            }
        try:
            result = process_audio_file(call_id, audio_path)
            if inspect.isawaitable(result):
                result = await result
            return _voice_turn_result_payload(result)
        except Exception:
            logger.warning(
                "Failed to process SimpleX native debug audio",
                extra={"call_id": call_id},
                exc_info=True,
            )
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_failed",
                "message": "native SimpleX debug audio processing failed",
            }


def decode_extra_ice_candidates(compressed_ice_candidates: str) -> list[Any]:
    from .simplex_session_codec import decompress_json

    decoded = decompress_json(compressed_ice_candidates)
    if not isinstance(decoded, list):
        raise ValueError("SimpleX extra ICE decoded to an invalid shape")
    return decoded


def _voice_turn_result_payload(result: Any) -> dict[str, Any]:
    ok = bool(getattr(result, "ok", False))
    transcript = str(getattr(result, "transcript", "") or "")
    response_text = str(getattr(result, "response_text", "") or "")
    audio_path = getattr(result, "audio_path", None)
    payload: dict[str, Any] = {
        "ok": ok,
        "code": str(getattr(result, "code", "") or "call_voice_turn_failed"),
        "message": str(getattr(result, "message", "") or ""),
        "transcriptChars": len(transcript),
        "responseChars": len(response_text),
    }
    if audio_path is not None:
        payload["audioPath"] = str(audio_path)
    stt_provider = str(getattr(result, "stt_provider", "") or "")
    if stt_provider:
        payload["sttProvider"] = stt_provider
    tts_provider = str(getattr(result, "tts_provider", "") or "")
    if tts_provider:
        payload["ttsProvider"] = tts_provider
    return payload


async def run_jsonl_sidecar(
    media_engine: SimplexNativeMediaEngine,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    runner = SimplexNativeSidecarRunner(media_engine)
    write_lock = asyncio.Lock()
    start_event_buffer: list[dict[str, Any]] = []
    buffering_start_events = False

    def _is_event_payload(payload: dict[str, Any]) -> bool:
        return payload.get("type") == "event" or payload.get("type") in {
            "audio",
            "ice",
            "status",
        }

    async def _write_protocol_message(payload: dict[str, Any]) -> None:
        nonlocal buffering_start_events
        if buffering_start_events and _is_event_payload(payload):
            start_event_buffer.append(dict(payload))
            return
        async with write_lock:
            output_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            output_stream.flush()

    set_event_sink = getattr(media_engine, "set_event_sink", None)
    if callable(set_event_sink):
        set_event_sink(_write_protocol_message)

    while True:
        line = await asyncio.to_thread(input_stream.readline)
        if not line:
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed SimpleX native sidecar JSON")
            continue
        if not isinstance(message, dict):
            logger.warning("Ignoring non-object SimpleX native sidecar JSON")
            continue
        is_start = str(message.get("type") or "") in {
            "start_incoming",
            "start_outgoing_answer",
        }
        if is_start:
            start_event_buffer.clear()
            buffering_start_events = True
        with contextlib.redirect_stdout(sys.stderr):
            try:
                response = await runner.handle_message(message)
            finally:
                if is_start:
                    buffering_start_events = False
        if response is not None:
            await _write_protocol_message(response)
        if is_start:
            for event_payload in start_event_buffer:
                await _write_protocol_message(event_payload)
            start_event_buffer.clear()
