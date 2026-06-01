from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ports import NativeMediaOffer, NativeMediaStartRequest
from .sidecar import SidecarMediaPort
from .tracing import NativeCallTraceWriter


_DEFAULT_ANSWER = {
    "rtcSession": "simulated-answer-sdp",
    "rtcIceCandidates": "simulated-answer-ice",
}
_DEFAULT_EXTRA = {"rtcIceCandidates": "simulated-extra-ice"}


@dataclass(frozen=True)
class NativeCallSimulationResult:
    ok: bool
    code: str
    message: str
    call_id: str
    contact_id: str
    trace_path: Path
    events: list[str] = field(default_factory=list)


def _offer_payload(offer: NativeMediaOffer | None) -> dict[str, Any]:
    if offer is None:
        return {}
    return {
        "rtcSession": offer.rtc_session,
        "rtcIceCandidates": offer.rtc_ice_candidates,
        "capabilities": dict(offer.capabilities or {}),
    }


async def run_native_call_simulation(
    *,
    command: Sequence[str],
    call_id: str = "simulated-call",
    contact_id: str = "simulated-contact",
    trace_root: Path | None = None,
    media: str = "audio",
    encrypted: bool = False,
    shared_key: str | None = None,
    answer: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    audio_path: Path | str | None = None,
    timeout_seconds: float = 10.0,
) -> NativeCallSimulationResult:
    tracer = NativeCallTraceWriter(root=trace_root)
    events: list[str] = []

    def record(event: str, **fields: Any) -> Path:
        events.append(event)
        return tracer.record(call_id, event, contact_id=contact_id, **fields)

    trace_path = record(
        "simulation_started",
        media=media,
        encrypted=encrypted,
        sharedKey=shared_key,
    )
    sidecar = SidecarMediaPort(command=command, timeout_seconds=timeout_seconds)
    start_result = await sidecar.start_incoming(
        NativeMediaStartRequest(
            call_id=call_id,
            contact_id=contact_id,
            media=media,
            encrypted=encrypted,
            shared_key=shared_key,
        )
    )
    if not start_result.ok:
        trace_path = record(
            "simulation_failed",
            code=start_result.code or "call_simplex_native_media_failed",
            message=start_result.message,
        )
        return NativeCallSimulationResult(
            ok=False,
            code=start_result.code or "call_simplex_native_media_failed",
            message=start_result.message,
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
            events=events,
        )

    trace_path = record(
        "simulation_offer_ready",
        payload=_offer_payload(start_result.offer),
    )
    answer_payload = dict(answer or _DEFAULT_ANSWER)
    if not await sidecar.apply_answer(call_id, answer_payload):
        await sidecar.stop(call_id)
        trace_path = record(
            "simulation_failed",
            code="call_simplex_native_answer_failed",
            message="simulated answer could not be applied",
        )
        return NativeCallSimulationResult(
            ok=False,
            code="call_simplex_native_answer_failed",
            message="simulated answer could not be applied",
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
            events=events,
        )
    trace_path = record("simulation_answer_applied", payload=answer_payload)

    extra_payload = dict(extra or _DEFAULT_EXTRA)
    if not await sidecar.add_extra_ice(call_id, extra_payload):
        await sidecar.stop(call_id)
        trace_path = record(
            "simulation_failed",
            code="call_simplex_native_extra_failed",
            message="simulated extra ICE could not be applied",
        )
        return NativeCallSimulationResult(
            ok=False,
            code="call_simplex_native_extra_failed",
            message="simulated extra ICE could not be applied",
            call_id=call_id,
            contact_id=contact_id,
            trace_path=trace_path,
            events=events,
        )
    trace_path = record("simulation_extra_applied", payload=extra_payload)

    if audio_path is not None:
        debug_result = await sidecar.debug_process_audio(call_id, str(audio_path))
        voice_turn_trace = {
            key: value
            for key, value in debug_result.items()
            if key != "audioPath"
        }
        if not bool(debug_result.get("ok")):
            await sidecar.stop(call_id)
            trace_path = record(
                "simulation_failed",
                code=str(
                    debug_result.get("code")
                    or "call_simplex_native_voice_turn_failed"
                ),
                message=str(
                    debug_result.get("message")
                    or "simulated voice turn could not be processed"
                ),
                voiceTurn=voice_turn_trace,
            )
            return NativeCallSimulationResult(
                ok=False,
                code=str(
                    debug_result.get("code")
                    or "call_simplex_native_voice_turn_failed"
                ),
                message=str(
                    debug_result.get("message")
                    or "simulated voice turn could not be processed"
                ),
                call_id=call_id,
                contact_id=contact_id,
                trace_path=trace_path,
                events=events,
            )
        trace_path = record(
            "simulation_voice_turn_completed",
            voiceTurn=voice_turn_trace,
        )

    await sidecar.stop(call_id)
    trace_path = record("simulation_stopped")
    trace_path = record("simulation_completed")
    return NativeCallSimulationResult(
        ok=True,
        code="call_simplex_native_simulation_passed",
        message="SimpleX-native sidecar simulation passed.",
        call_id=call_id,
        contact_id=contact_id,
        trace_path=trace_path,
        events=events,
    )
