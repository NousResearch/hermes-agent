#!/usr/bin/env python3
"""Verify the WhatsApp Cloud Calling control plane without contacting Meta.

This is a synthetic smoke for the Hermes side of WhatsApp Calling. It feeds a
realistic ``calls`` webhook payload into the Cloud adapter, records the local
sidecar and Graph API requests the adapter would make, and validates the
expected sequence:

1. Fetch the optional ``voice.webrtc_sidecar`` contract.
2. POST the WhatsApp SDP offer to the local sidecar.
3. POST ``pre_accept`` to Graph with the sidecar SDP answer.
4. POST ``accept`` to Graph with the same answer.
5. Start the inbound sidecar drain for the caller.
6. On a terminate webhook, close the local sidecar call session.

The script uses an in-process fake HTTP client. It does not require WhatsApp
credentials, a public webhook URL, or a running sidecar.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from typing import Any

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


DEFAULT_CALL_ID = "wacid.ABGGFjFVU2AfAgo6V-Hc5eCgK5Gh"
DEFAULT_CALLER = "13557825698"
DEFAULT_CALLEE = "15551797781"
DEFAULT_CONTACT_NAME = "Jessica Laverdetman"
DEFAULT_REMOTE_SDP = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
DEFAULT_ANSWER_SDP = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8787"
DEFAULT_PHONE_NUMBER_ID = "7794189252778687"

CALLING_AUDIO_CONTRACT = {
    "sample_rate": 48_000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le",
    "bytes_per_sample": 2,
    "samples_per_frame": 960,
    "frame_bytes": 1_920,
    "default_drain_bytes": 96_000,
    "max_outbound_queue_bytes": 960_000,
    "max_inbound_queue_bytes": 960_000,
    "max_drain_wait_ms": 5_000,
}
CALLING_READY_STATE = {
    "ready_for_accept": True,
    "readiness": {
        "not_closed": True,
        "local_sdp_answer": True,
        "signaling_stable": True,
        "ice_gathering_complete": True,
        "outbound_audio_track": True,
    },
}


@dataclass
class FakeResponse:
    status_code: int
    body: dict[str, Any]

    @property
    def text(self) -> str:
        return json.dumps(self.body, sort_keys=True)

    def json(self) -> dict[str, Any]:
        return self.body


class RecordingHttpClient:
    """Small async httpx-like fake for sidecar and Graph requests."""

    def __init__(
        self,
        *,
        sidecar_url: str,
        api_version: str,
        phone_number_id: str,
        call_id: str,
        answer_sdp: str,
    ) -> None:
        self.sidecar_url = sidecar_url.rstrip("/")
        self.graph_calls_url = (
            f"https://graph.facebook.com/{api_version}/{phone_number_id}/calls"
        )
        self.call_id = call_id
        self.answer_sdp = answer_sdp
        self.requests: list[dict[str, Any]] = []

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": "GET", "url": url, "kwargs": kwargs})
        if url == f"{self.sidecar_url}/contract":
            return FakeResponse(
                200,
                {
                    "contract": "voice.webrtc_sidecar",
                    "version": 1,
                    "audio": CALLING_AUDIO_CONTRACT,
                    "endpoints": {
                        "offer": {"method": "POST", "path": "/offer"},
                        "close_call": {
                            "method": "POST",
                            "path": "/calls/{call_id}/close",
                        },
                    },
                },
            )
        return FakeResponse(404, {"error": "not found"})

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": "POST", "url": url, "kwargs": kwargs})
        payload = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}

        if url == f"{self.sidecar_url}/offer":
            return FakeResponse(
                200,
                {
                    "call_id": payload.get("call_id") or self.call_id,
                    "type": "answer",
                    "sdp": self.answer_sdp,
                    "audio": CALLING_AUDIO_CONTRACT,
                    "state": CALLING_READY_STATE,
                },
            )

        if url == self.graph_calls_url:
            action = str(payload.get("action") or "")
            if action in {"pre_accept", "accept", "reject", "terminate"}:
                return FakeResponse(200, {"success": True, "action": action})
            return FakeResponse(400, {"error": {"message": "unknown action"}})

        if url == f"{self.sidecar_url}/calls/{self.call_id}/close":
            return FakeResponse(200, {"call_id": self.call_id, "closed": True})

        return FakeResponse(404, {"error": "not found"})


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


async def run_control_plane_smoke(args: argparse.Namespace) -> dict[str, Any]:
    sidecar_url = args.sidecar_url.rstrip("/")
    config = PlatformConfig(
        enabled=True,
        extra={
            "phone_number_id": args.phone_number_id,
            "access_token": "synthetic-token",
            "calling_sidecar_url": sidecar_url,
            "calling_sidecar_timeout": args.timeout,
        },
    )
    adapter = WhatsAppCloudAdapter(config)
    http_client = RecordingHttpClient(
        sidecar_url=sidecar_url,
        api_version=adapter._api_version,
        phone_number_id=args.phone_number_id,
        call_id=args.call_id,
        answer_sdp=args.answer_sdp,
    )
    adapter._http_client = http_client

    dispatched_messages: list[Any] = []

    async def capture_message(event: Any) -> None:
        dispatched_messages.append(event)

    drain_starts: list[dict[str, str]] = []

    def capture_drain(call_id: str, chat_id: str, sender_name: str = "") -> None:
        drain_starts.append(
            {"call_id": call_id, "chat_id": chat_id, "sender_name": sender_name}
        )

    adapter.handle_message = capture_message
    adapter._start_calling_sidecar_drain = capture_drain  # type: ignore[method-assign]

    await adapter._dispatch_payload(
        call_connect_payload(
            call_id=args.call_id,
            caller=args.caller,
            callee=args.callee,
            contact_name=args.contact_name,
            remote_sdp=args.remote_sdp,
        )
    )

    await adapter._dispatch_payload(
        call_terminate_payload(
            call_id=args.call_id,
            caller=args.caller,
            callee=args.callee,
        )
    )

    graph_actions = [
        request["kwargs"]["json"]["action"]
        for request in http_client.requests
        if request["method"] == "POST"
        and request["url"] == http_client.graph_calls_url
        and isinstance(request["kwargs"].get("json"), dict)
    ]
    sidecar_offer = next(
        (
            request
            for request in http_client.requests
            if request["method"] == "POST"
            and request["url"] == f"{sidecar_url}/offer"
        ),
        None,
    )
    sidecar_close = next(
        (
            request
            for request in http_client.requests
            if request["method"] == "POST"
            and request["url"] == f"{sidecar_url}/calls/{args.call_id}/close"
        ),
        None,
    )

    failures: list[str] = []
    if dispatched_messages:
        failures.append("calling payload should not dispatch a text MessageEvent")
    if sidecar_offer is None:
        failures.append("sidecar offer request was not sent")
    elif sidecar_offer["kwargs"]["json"] != {
        "call_id": args.call_id,
        "type": "offer",
        "sdp": args.remote_sdp,
    }:
        failures.append("sidecar offer payload did not match expected SDP offer")
    if graph_actions[:2] != ["pre_accept", "accept"]:
        failures.append(f"expected graph actions pre_accept, accept; got {graph_actions}")
    if sidecar_close is None:
        failures.append("sidecar close request was not sent after terminate")
    if drain_starts != [
        {
            "call_id": args.call_id,
            "chat_id": args.caller,
            "sender_name": args.contact_name,
        }
    ]:
        failures.append(f"unexpected sidecar drain starts: {drain_starts}")
    if adapter._calling_sidecar_call_ids:
        failures.append(
            f"call ids should be empty after terminate: {adapter._calling_sidecar_call_ids}"
        )

    session_payloads = [
        request["kwargs"]["json"].get("session")
        for request in http_client.requests
        if request["method"] == "POST"
        and request["url"] == http_client.graph_calls_url
        and request["kwargs"]["json"].get("action") in {"pre_accept", "accept"}
    ]
    if session_payloads != [
        {"sdp_type": "answer", "sdp": args.answer_sdp},
        {"sdp_type": "answer", "sdp": args.answer_sdp},
    ]:
        failures.append("pre_accept and accept did not carry the sidecar SDP answer")

    if failures:
        raise SystemExit("control-plane smoke failed:\n- " + "\n- ".join(failures))

    return {
        "success": True,
        "call_id": args.call_id,
        "caller": args.caller,
        "sidecar_url": sidecar_url,
        "contract_loaded": adapter._calling_sidecar_contract is not None,
        "sidecar_offer_url": sidecar_offer["url"] if sidecar_offer else None,
        "graph_calls_url": http_client.graph_calls_url,
        "graph_actions": graph_actions,
        "sidecar_close_url": sidecar_close["url"] if sidecar_close else None,
        "drain_starts": drain_starts,
        "audio": CALLING_AUDIO_CONTRACT,
        "sidecar_ready_for_accept": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-url", default=DEFAULT_SIDECAR_URL)
    parser.add_argument("--phone-number-id", default=DEFAULT_PHONE_NUMBER_ID)
    parser.add_argument("--call-id", default=DEFAULT_CALL_ID)
    parser.add_argument("--caller", default=DEFAULT_CALLER)
    parser.add_argument("--callee", default=DEFAULT_CALLEE)
    parser.add_argument("--contact-name", default=DEFAULT_CONTACT_NAME)
    parser.add_argument("--remote-sdp", default=DEFAULT_REMOTE_SDP)
    parser.add_argument("--answer-sdp", default=DEFAULT_ANSWER_SDP)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    result = asyncio.run(run_control_plane_smoke(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
