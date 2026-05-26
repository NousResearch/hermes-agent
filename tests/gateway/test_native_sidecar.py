from __future__ import annotations

import json
import sys
import textwrap

import pytest

from gateway.calls.native.ports import NativeMediaStartRequest
from gateway.calls.native.sidecar import SidecarMediaPort


def _request() -> NativeMediaStartRequest:
    return NativeMediaStartRequest(
        call_id="call_123",
        contact_id="contact_456",
        media="audio",
        encrypted=True,
        shared_key="shared-secret",
    )


@pytest.mark.asyncio
async def test_start_incoming_fails_when_sidecar_command_missing():
    port = SidecarMediaPort(command=[])

    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_start_failed"
    assert "sidecar command is not configured" in result.message


@pytest.mark.asyncio
async def test_start_incoming_sends_request_and_returns_offer(tmp_path):
    request_path = tmp_path / "request.json"
    child_path = tmp_path / "fake_sidecar.py"
    child_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import pathlib
            import sys

            line = sys.stdin.readline()
            pathlib.Path({str(request_path)!r}).write_text(line)
            sys.stdout.write(json.dumps({{
                "ok": True,
                "offer": {{
                    "rtcSession": "offer-b64",
                    "rtcIceCandidates": "ice-b64",
                    "capabilities": {{"encryption": True}},
                }},
            }}, separators=(",", ":")) + "\\n")
            sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    port = SidecarMediaPort(command=[sys.executable, str(child_path)])

    result = await port.start_incoming(_request())

    assert result.ok is True
    assert result.offer is not None
    assert result.offer.rtc_session == "offer-b64"
    assert result.offer.rtc_ice_candidates == "ice-b64"
    assert result.offer.capabilities == {"encryption": True}
    assert json.loads(request_path.read_text(encoding="utf-8")) == {
        "type": "start_incoming",
        "callId": "call_123",
        "contactId": "contact_456",
        "media": "audio",
        "encrypted": True,
        "sharedKey": "shared-secret",
    }


@pytest.mark.asyncio
async def test_start_incoming_returns_protocol_failure_when_child_exits_without_output():
    port = SidecarMediaPort(command=[sys.executable, "-c", ""])

    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_protocol_failed"
    assert result.message


@pytest.mark.asyncio
async def test_start_incoming_returns_timeout_when_child_does_not_reply():
    port = SidecarMediaPort(
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_seconds=0.01,
    )

    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_simplex_native_timeout"
    assert result.message
