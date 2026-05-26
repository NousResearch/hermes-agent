from __future__ import annotations

import json
import os
import sys
import textwrap
import time

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


def _write_child(tmp_path, source: str) -> list[str]:
    child_path = tmp_path / "fake_sidecar.py"
    child_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return [sys.executable, str(child_path)]


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
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
            """,
        )
    )

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


@pytest.mark.parametrize(
    ("line_expr", "raw_fragment"),
    [
        ("json.dumps([]) + '\\n'", "[]"),
        ("'not-json\\n'", "not-json"),
    ],
)
@pytest.mark.asyncio
async def test_start_incoming_returns_protocol_failure_for_invalid_response_shape(
    tmp_path,
    line_expr,
    raw_fragment,
    caplog,
):
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
            f"""
            import json
            import sys

            sys.stdout.write({line_expr})
            sys.stdout.flush()
            """,
        )
    )

    caplog.set_level("WARNING", logger="gateway.calls.native.sidecar")
    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_protocol_failed"
    assert result.message
    assert any(
        record.reason == "invalid_response"
        for record in caplog.records
        if record.message == "SimpleX native call sidecar protocol failure"
    )
    assert raw_fragment not in caplog.text


@pytest.mark.asyncio
async def test_start_incoming_returns_protocol_failure_for_invalid_utf8_response(
    tmp_path,
    caplog,
):
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
            """
            import sys

            sys.stdout.buffer.write(b"sidecar-secret\\xff\\n")
            sys.stdout.flush()
            """,
        )
    )

    caplog.set_level("WARNING", logger="gateway.calls.native.sidecar")
    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_protocol_failed"
    assert result.message
    assert any(
        record.reason == "invalid_response_encoding"
        for record in caplog.records
        if record.message == "SimpleX native call sidecar protocol failure"
    )
    assert "sidecar-secret" not in caplog.text
    assert "0xff" not in caplog.text


@pytest.mark.parametrize(
    "offer",
    [
        {"rtcSession": ["offer-b64"], "rtcIceCandidates": "ice-b64"},
        {"rtcSession": "offer-b64", "rtcIceCandidates": {"candidate": "ice-b64"}},
    ],
)
@pytest.mark.asyncio
async def test_start_incoming_returns_protocol_failure_for_wrong_offer_field_types(
    tmp_path,
    offer,
):
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
            f"""
            import json
            import sys

            sys.stdout.write(json.dumps({{"ok": True, "offer": {offer!r}}}) + "\\n")
            sys.stdout.flush()
            """,
        )
    )

    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_protocol_failed"
    assert result.message


@pytest.mark.asyncio
async def test_start_incoming_defaults_safely_for_non_string_error_fields(
    tmp_path,
    caplog,
):
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
            """
            import json
            import sys

            sys.stdout.write(json.dumps({
                "ok": False,
                "code": ["call_custom"],
                "message": {"detail": "failed"},
            }) + "\\n")
            sys.stdout.flush()
            """,
        )
    )

    caplog.set_level("WARNING", logger="gateway.calls.native.sidecar")
    result = await port.start_incoming(_request())

    assert result.ok is False
    assert result.code == "call_sidecar_protocol_failed"
    assert result.message == "call sidecar returned a failed response"
    assert any(
        record.reason == "invalid_error_response"
        for record in caplog.records
        if record.message == "SimpleX native call sidecar protocol failure"
    )
    assert "call_custom" not in caplog.text


@pytest.mark.asyncio
async def test_start_incoming_returns_timeout_and_reaps_child(tmp_path):
    pid_path = tmp_path / "sidecar.pid"
    port = SidecarMediaPort(
        command=_write_child(
            tmp_path,
            f"""
            import os
            import pathlib
            import time

            pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))
            time.sleep(10)
            """,
        ),
        timeout_seconds=0.2,
    )

    result = await port.start_incoming(_request())
    pid = int(pid_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.code == "call_simplex_native_timeout"
    assert result.message
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and _process_exists(pid):
        time.sleep(0.01)
    assert not _process_exists(pid)
