#!/usr/bin/env python3
"""Verify Hermes WhatsApp Calling control flow against a real local sidecar.

This smoke does not contact Meta. It feeds a synthetic WhatsApp Calling
``connect`` webhook into Hermes, lets Hermes POST the real SDP offer to a
voice WebRTC sidecar, fakes only the Graph ``/calls`` actions, and verifies
WebRTC audio can move both ways. By default it starts the sidecar in-process;
pass ``--sidecar-url`` to target an already-running local sidecar service.

Run it with a Python environment that has the voice sidecar extras installed.
The local stack installer exposes that as ``--webrtc-python-bin``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from aiohttp import ClientSession, web

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


DEFAULT_CALL_ID = "wacid.local-sidecar-smoke"
DEFAULT_CALLER = "13557825698"
DEFAULT_CALLEE = "15551797781"
DEFAULT_CONTACT_NAME = "Hermes Voice Smoke"
DEFAULT_PHONE_NUMBER_ID = "7794189252778687"
DEFAULT_APP_SECRET = "synthetic-whatsapp-calling-secret"
DEFAULT_TIMEOUT = 12.0


@dataclass
class RecordedResponse:
    status_code: int
    body: dict[str, Any]

    @property
    def text(self) -> str:
        return json.dumps(self.body, sort_keys=True)

    def json(self) -> dict[str, Any]:
        return self.body


class SignedWebhookRequest:
    """Small aiohttp.web.Request stand-in for the Cloud webhook handler."""

    def __init__(self, raw: bytes, signature: str) -> None:
        self._raw = raw
        self.headers = {"X-Hub-Signature-256": signature}

    async def read(self) -> bytes:
        return self._raw


def signed_webhook_request(payload: dict[str, Any], *, app_secret: str) -> SignedWebhookRequest:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hmac.new(app_secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return SignedWebhookRequest(raw, f"sha256={digest}")


def load_sidecar_module(voice_repo: Path):
    path = voice_repo.expanduser().resolve() / "examples" / "webrtc-sidecar" / "sidecar.py"
    if not path.is_file():
        raise SystemExit(f"voice WebRTC sidecar not found: {path}")
    spec = importlib.util.spec_from_file_location("voice_webrtc_sidecar_live_smoke", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load voice WebRTC sidecar module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pcm_frame(sidecar: Any, sample: int = 12_000) -> bytes:
    return sample.to_bytes(2, byteorder="little", signed=True) * int(
        sidecar.SAMPLES_PER_FRAME
    )


def make_pcm_track_class(sidecar: Any, *, frame_bytes: bytes):
    class SyntheticPcmTrack(sidecar.MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.pts = 0

        async def recv(self) -> Any:
            await asyncio.sleep(sidecar.FRAME_MS / 1_000)
            frame = sidecar.av.AudioFrame(
                format="s16",
                layout="mono",
                samples=sidecar.SAMPLES_PER_FRAME,
            )
            frame.planes[0].update(frame_bytes)
            frame.sample_rate = sidecar.SAMPLE_RATE
            frame.time_base = sidecar.Fraction(1, sidecar.SAMPLE_RATE)
            frame.pts = self.pts
            self.pts += sidecar.SAMPLES_PER_FRAME
            return frame

    return SyntheticPcmTrack


async def start_sidecar_app(sidecar: Any) -> tuple[web.AppRunner, str]:
    source = sidecar.PcmSource(None)
    app = sidecar.create_app(source, None)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


class RecordingHybridHttpClient:
    """httpx-like async client: real sidecar HTTP, fake Graph /calls."""

    def __init__(
        self,
        *,
        sidecar_url: str,
        graph_calls_url: str,
        timeout: float,
    ) -> None:
        self.sidecar_url = sidecar_url.rstrip("/")
        self.graph_calls_url = graph_calls_url
        self.timeout = timeout
        self.session = ClientSession()
        self.requests: list[dict[str, Any]] = []
        self.sidecar_offer_response: dict[str, Any] | None = None
        self.sidecar_close_responses: list[dict[str, Any]] = []

    async def aclose(self) -> None:
        await self.session.close()

    async def get(self, url: str, **kwargs: Any) -> RecordedResponse:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> RecordedResponse:
        return await self._request("POST", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs: Any) -> RecordedResponse:
        request = {"method": method, "url": url, "kwargs": kwargs}
        self.requests.append(request)

        if url == self.graph_calls_url and method == "POST":
            payload = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}
            action = str(payload.get("action") or "")
            if action in {"pre_accept", "accept", "reject", "terminate"}:
                return RecordedResponse(200, {"success": True, "action": action})
            return RecordedResponse(400, {"error": {"message": "unknown action"}})

        if url.startswith(self.sidecar_url + "/"):
            async with self.session.request(
                method,
                url,
                json=kwargs.get("json"),
                params=kwargs.get("params"),
                timeout=self.timeout,
            ) as response:
                text = await response.text()
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    body = {"raw": text}
            if method == "POST" and url == f"{self.sidecar_url}/offer":
                self.sidecar_offer_response = body
            if (
                method == "POST"
                and url.startswith(f"{self.sidecar_url}/calls/")
                and url.endswith("/close")
            ):
                self.sidecar_close_responses.append(body)
            return RecordedResponse(response.status, body)

        return RecordedResponse(404, {"error": "not found"})


def call_connect_payload(
    *,
    call_id: str,
    caller: str,
    callee: str,
    contact_name: str,
    remote_sdp: str,
) -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "215589313241560883",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": callee,
                                "phone_number_id": DEFAULT_PHONE_NUMBER_ID,
                            },
                            "contacts": [
                                {
                                    "profile": {"name": contact_name},
                                    "wa_id": caller,
                                }
                            ],
                            "calls": [
                                {
                                    "id": call_id,
                                    "from": caller,
                                    "to": callee,
                                    "event": "connect",
                                    "timestamp": "1762216151",
                                    "direction": "USER_INITIATED",
                                    "session": {
                                        "sdp_type": "offer",
                                        "sdp": remote_sdp,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def call_terminate_payload(*, call_id: str, caller: str, callee: str) -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "215589313241560883",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": callee,
                                "phone_number_id": DEFAULT_PHONE_NUMBER_ID,
                            },
                            "calls": [
                                {
                                    "id": call_id,
                                    "from": caller,
                                    "to": callee,
                                    "event": "terminate",
                                    "timestamp": "1762216199",
                                    "direction": "USER_INITIATED",
                                    "status": "COMPLETED",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


async def wait_for_ice_complete(sidecar: Any, pc: Any, timeout: float) -> None:
    await sidecar.wait_for_ice_complete(pc, timeout=timeout)
    if pc.localDescription is None:
        raise RuntimeError("local WebRTC peer did not produce a local description")


async def wait_for_non_silent_track(track: Any, *, timeout_s: float) -> int:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        frame = await asyncio.wait_for(track.recv(), timeout=min(2.0, timeout_s))
        frame_bytes = b"".join(bytes(plane) for plane in frame.planes)
        if any(frame_bytes):
            return len(frame_bytes)
    raise TimeoutError("local WebRTC peer only received silence")


async def wait_for_non_silent_sidecar_drain(
    adapter: WhatsAppCloudAdapter,
    call_id: str,
    *,
    timeout_s: float,
) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        audio = await adapter._receive_calling_sidecar_audio(  # noqa: SLF001
            call_id,
            max_bytes=1_920,
            wait_ms=500,
        )
        if audio is not None and any(audio.pcm_s16le):
            return audio
    raise TimeoutError("sidecar inbound drain stayed silent")


def graph_actions(requests: list[dict[str, Any]], graph_calls_url: str) -> list[str]:
    return [
        str(request["kwargs"]["json"].get("action") or "")
        for request in requests
        if request["method"] == "POST"
        and request["url"] == graph_calls_url
        and isinstance(request["kwargs"].get("json"), dict)
    ]


async def run_live_sidecar_smoke(args: argparse.Namespace) -> dict[str, Any]:
    voice_repo = args.voice_repo.expanduser().resolve()
    sidecar = load_sidecar_module(voice_repo)
    if args.sidecar_url:
        runner = None
        sidecar_url = args.sidecar_url.rstrip("/")
        sidecar_mode = "running"
    else:
        runner, sidecar_url = await start_sidecar_app(sidecar)
        sidecar_mode = "in_process"
    pc = sidecar.RTCPeerConnection()
    http_client: RecordingHybridHttpClient | None = None
    try:
        inbound_frame = pcm_frame(sidecar, sample=-10_000)
        pc.addTrack(make_pcm_track_class(sidecar, frame_bytes=inbound_frame)())
        track_future = asyncio.get_running_loop().create_future()

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio" and not track_future.done():
                track_future.set_result(track)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_for_ice_complete(sidecar, pc, args.timeout)

        config = PlatformConfig(
            enabled=True,
            extra={
                "phone_number_id": args.phone_number_id,
                "access_token": "synthetic-token",
                "app_secret": args.app_secret,
                "calling_sidecar_url": sidecar_url,
                "calling_sidecar_timeout": args.timeout,
            },
        )
        adapter = WhatsAppCloudAdapter(config)
        graph_calls_url = (
            f"https://graph.facebook.com/{adapter._api_version}/{args.phone_number_id}/calls"  # noqa: SLF001
        )
        http_client = RecordingHybridHttpClient(
            sidecar_url=sidecar_url,
            graph_calls_url=graph_calls_url,
            timeout=args.timeout,
        )
        adapter._http_client = http_client  # noqa: SLF001

        dispatched_messages: list[Any] = []

        async def capture_message(event: Any) -> None:
            dispatched_messages.append(event)

        drain_starts: list[dict[str, str]] = []

        def capture_drain(call_id: str, chat_id: str, sender_name: str = "") -> None:
            drain_starts.append(
                {"call_id": call_id, "chat_id": chat_id, "sender_name": sender_name}
            )

        adapter.handle_message = capture_message
        adapter._start_calling_sidecar_drain = capture_drain  # type: ignore[method-assign]  # noqa: SLF001

        connect_response = await adapter._handle_webhook(  # noqa: SLF001
            signed_webhook_request(
                call_connect_payload(
                    call_id=args.call_id,
                    caller=args.caller,
                    callee=args.callee,
                    contact_name=args.contact_name,
                    remote_sdp=pc.localDescription.sdp,
                ),
                app_secret=args.app_secret,
            )
        )
        if connect_response.status != 200:
            raise RuntimeError(
                f"signed connect webhook returned HTTP {connect_response.status}"
            )

        if dispatched_messages:
            raise RuntimeError("calling connect dispatched an unexpected MessageEvent")
        if http_client.sidecar_offer_response is None:
            raise RuntimeError("Hermes did not POST a real offer to the sidecar")
        if http_client.sidecar_offer_response.get("type") != "answer":
            raise RuntimeError(
                f"sidecar did not return an SDP answer: {http_client.sidecar_offer_response}"
            )
        sidecar_state = http_client.sidecar_offer_response.get("state")
        if not isinstance(sidecar_state, dict):
            raise RuntimeError(
                f"sidecar answer did not include call state: {http_client.sidecar_offer_response}"
            )
        if sidecar_state.get("ready_for_accept") is not True:
            raise RuntimeError(f"sidecar was not ready for accept: {sidecar_state}")
        if args.call_id not in adapter._calling_sidecar_call_ids:  # noqa: SLF001
            raise RuntimeError("adapter did not track the active sidecar call id")

        await pc.setRemoteDescription(
            sidecar.RTCSessionDescription(
                sdp=str(http_client.sidecar_offer_response["sdp"]),
                type="answer",
            )
        )

        outbound_result = await adapter._send_calling_sidecar_audio(  # noqa: SLF001
            args.call_id,
            pcm_frame(sidecar, sample=12_000),
        )
        if not outbound_result.success:
            raise RuntimeError(f"sidecar outbound audio send failed: {outbound_result}")

        track = await asyncio.wait_for(track_future, timeout=args.timeout)
        outbound_webrtc_bytes = await wait_for_non_silent_track(
            track,
            timeout_s=args.timeout,
        )
        inbound_audio = await wait_for_non_silent_sidecar_drain(
            adapter,
            args.call_id,
            timeout_s=args.timeout,
        )

        clear_probe = await adapter._send_calling_sidecar_audio(  # noqa: SLF001
            args.call_id,
            pcm_frame(sidecar, sample=14_000),
            sequence=1,
        )
        if not clear_probe.success:
            raise RuntimeError(
                f"sidecar clear probe audio send failed: {clear_probe}"
            )
        clear_result = await adapter._clear_calling_sidecar_audio(  # noqa: SLF001
            args.call_id,
        )
        if not clear_result.success:
            raise RuntimeError(f"sidecar clear audio failed: {clear_result}")
        clear_body = clear_result.raw_response
        if not isinstance(clear_body, dict):
            raise RuntimeError(f"sidecar clear response was not an object: {clear_body}")
        if clear_body.get("skipped"):
            raise RuntimeError(f"sidecar clear endpoint was skipped: {clear_body}")
        try:
            queued_after_clear = int(clear_body.get("queued_tx_bytes"))
            dropped_tx_bytes = int(clear_body.get("dropped_tx_bytes"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"sidecar clear response missing byte telemetry: {clear_body}"
            ) from exc
        if queued_after_clear != 0 or dropped_tx_bytes < 0:
            raise RuntimeError(
                f"sidecar clear response did not drain outbound queue: {clear_body}"
            )

        actions = graph_actions(http_client.requests, graph_calls_url)
        if actions[:2] != ["pre_accept", "accept"]:
            raise RuntimeError(f"expected pre_accept, accept; got {actions}")
        if drain_starts != [
            {
                "call_id": args.call_id,
                "chat_id": args.caller,
                "sender_name": args.contact_name,
            }
        ]:
            raise RuntimeError(f"unexpected drain starts: {drain_starts}")

        terminate_response = await adapter._handle_webhook(  # noqa: SLF001
            signed_webhook_request(
                call_terminate_payload(
                    call_id=args.call_id,
                    caller=args.caller,
                    callee=args.callee,
                ),
                app_secret=args.app_secret,
            )
        )
        if terminate_response.status != 200:
            raise RuntimeError(
                f"signed terminate webhook returned HTTP {terminate_response.status}"
            )
        if adapter._calling_sidecar_call_ids:  # noqa: SLF001
            raise RuntimeError(
                f"call ids should be empty after terminate: {adapter._calling_sidecar_call_ids}"  # noqa: SLF001
            )

        close_requests = [
            request
            for request in http_client.requests
            if request["method"] == "POST"
            and request["url"] == f"{sidecar_url}/calls/{args.call_id}/close"
        ]
        if not close_requests:
            raise RuntimeError("Hermes did not close the real sidecar call session")
        sidecar_close = (
            http_client.sidecar_close_responses[-1]
            if http_client.sidecar_close_responses
            else None
        )
        if not isinstance(sidecar_close, dict) or sidecar_close.get("closed") is not True:
            raise RuntimeError(f"sidecar close did not report closed=true: {sidecar_close}")

        return {
            "success": True,
            "call_id": args.call_id,
            "caller": args.caller,
            "sidecar_url": sidecar_url,
            "sidecar_mode": sidecar_mode,
            "graph_calls_url": graph_calls_url,
            "graph_actions": actions,
            "drain_starts": drain_starts,
            "webhook_statuses": {
                "connect": connect_response.status,
                "terminate": terminate_response.status,
            },
            "sidecar_offer_url": f"{sidecar_url}/offer",
            "sidecar_close_url": close_requests[-1]["url"],
            "sidecar_close": sidecar_close,
            "sidecar_ready_for_accept": sidecar_state["ready_for_accept"],
            "sidecar_readiness": sidecar_state.get("readiness"),
            "outbound_webrtc_bytes": outbound_webrtc_bytes,
            "inbound_drain_bytes": inbound_audio.returned_bytes,
            "queued_rx_ms": inbound_audio.queued_rx_ms,
            "sent_audio": outbound_result.raw_response,
            "clear_probe": clear_probe.raw_response,
            "clear_audio": clear_body,
            "audio": sidecar.audio_contract(),
        }
    finally:
        await pc.close()
        if http_client is not None:
            await http_client.aclose()
        if runner is not None:
            await runner.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-repo", type=Path, required=True)
    parser.add_argument(
        "--sidecar-url",
        help=(
            "Existing local voice WebRTC sidecar URL. When omitted, the smoke "
            "starts an in-process sidecar from --voice-repo."
        ),
    )
    parser.add_argument("--phone-number-id", default=DEFAULT_PHONE_NUMBER_ID)
    parser.add_argument("--call-id", default=DEFAULT_CALL_ID)
    parser.add_argument("--caller", default=DEFAULT_CALLER)
    parser.add_argument("--callee", default=DEFAULT_CALLEE)
    parser.add_argument("--contact-name", default=DEFAULT_CONTACT_NAME)
    parser.add_argument("--app-secret", default=DEFAULT_APP_SECRET)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser.parse_args()


def main() -> int:
    result = asyncio.run(run_live_sidecar_smoke(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
