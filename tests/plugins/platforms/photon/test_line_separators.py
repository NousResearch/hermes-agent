"""NDJSON framing guarantees for the Photon sidecar inbound stream.

``JSON.stringify`` emits U+2028/U+2029/U+0085 raw inside JSON strings. The
Python adapter reads the stream with ``httpx`` ``aiter_lines()``, whose
``LineDecoder`` follows ``str.splitlines()`` semantics and splits on those
same code points — so an event containing them used to arrive as two
fragments, both failing ``json.loads`` and being dropped at DEBUG with no
visible trace (an inbound iMessage containing an iOS Enter was silently
lost).

The fix escapes the separators at write time in
``plugins/platforms/photon/sidecar/line-separators.mjs`` (used by
``deliver()``), and promotes the adapter's drop path to WARNING. The node
tests below execute the real escape module; the adapter tests drive
``_on_inbound_line`` directly, following test_inbound.py's harness.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter

_MODULE = Path("plugins/platforms/photon/sidecar/line-separators.mjs").resolve()

# Representative payloads exercising every escaped code point, including the
# exact shape iOS Messages produces (U+2028 for Enter inside a multi-line
# message body).
_PAYLOADS = [
    {"kind": "message", "text": "line one\u2028line two", "messageId": "m-1"},
    {"kind": "message", "text": "a\u2028b\u2029c\u0085d", "messageId": "m-2"},
    {"kind": "message", "text": "plain message", "messageId": "m-3"},
]


def _escape_under_node(payloads: List[Dict]) -> List[str]:
    """Run the real escape module over each payload in one node call."""
    harness = (
        f"import {{ escapeLineSeparators }} from {json.dumps(_MODULE.as_uri())};\n"
        "const chunks = [];\n"
        "process.stdin.on('data', (c) => chunks.push(c));\n"
        "process.stdin.on('end', () => {\n"
        "  const payloads = JSON.parse(Buffer.concat(chunks).toString('utf-8'));\n"
        "  const out = payloads.map((p) => escapeLineSeparators(JSON.stringify(p)));\n"
        "  process.stdout.write(JSON.stringify(out));\n"
        "});\n"
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        input=json.dumps(payloads),
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_frames_contain_no_raw_line_separators() -> None:
    frames = _escape_under_node(_PAYLOADS)
    for frame in frames:
        assert "\u2028" not in frame
        assert "\u2029" not in frame
        assert "\u0085" not in frame
        # Both consumer families must see exactly one line per frame:
        # splitlines()-semantics readers (httpx LineDecoder) and \n-splitting
        # readers alike.
        assert len(frame.splitlines()) == 1
        assert len(frame.split("\n")) == 1


def test_frames_round_trip_through_json_parse() -> None:
    frames = _escape_under_node(_PAYLOADS)
    for frame, payload in zip(frames, _PAYLOADS):
        assert json.loads(frame) == payload


def test_plain_payload_is_untouched() -> None:
    frames = _escape_under_node(_PAYLOADS)
    plain = next(p for p in _PAYLOADS if p["messageId"] == "m-3")
    # JSON.stringify's compact form — no escaping added for separator-free payloads
    assert frames[_PAYLOADS.index(plain)] == json.dumps(plain, separators=(",", ":"))


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


@pytest.mark.asyncio
async def test_fragmented_inbound_line_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """What a splitlines()-semantics reader hands the adapter when the sidecar
    emits a raw U+2028: half a JSON document. Dropping it silently is an
    invisible loss of a user message — it must log at WARNING, and the log
    must carry the payload's shape (length + digest), never its content:
    inbound lines are user messages (SMS/iMessage), and a 120-char prefix
    would usually be the whole message."""
    adapter = _make_adapter(monkeypatch)
    fragment = '{"text": "line one'
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.photon.adapter"):
        await adapter._on_inbound_line(fragment)
    warnings = [
        r for r in caplog.records
        if "skipping non-JSON inbound line" in r.message and r.levelno == logging.WARNING
    ]
    assert warnings
    logged = warnings[0].getMessage()
    assert "len=" in logged and "sha256=" in logged
    assert "line one" not in logged, "user content leaked into logs"


@pytest.mark.asyncio
async def test_valid_inbound_line_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = _make_adapter(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.photon.adapter"):
        await adapter._on_inbound_line(
            json.dumps({"messageId": "m-ok", "kind": "ping"})
        )
    assert not any("skipping non-JSON" in r.message for r in caplog.records)
