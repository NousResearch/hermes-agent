"""Buzz video uploads must pass the relay's MP4 metadata validator, and a rejection must be reported.

The Buzz relay (block/buzz ``crates/buzz-media/src/validation.rs::validate_mp4_metadata_free``) answers
HTTP 422 "media contains metadata or a non-canonical metadata channel" for any MP4 whose ``udta`` box is not
the exact empty box ffmpeg writes under ``-map_metadata -1 -fflags +bitexact ...``. ffmpeg's default output
carries an ``encoder`` tag, so an ordinary ffmpeg-made MP4 was rejected and the adapter reported only a
generic failure. ``_relay_accepts_mp4_boxes`` below is the relay's box walk, used as the oracle.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.gateway.test_buzz_adapter import CHANNEL, _make_adapter

_buzz_mod = load_plugin_adapter("buzz")

_EMPTY_FFMPEG_UDTA = bytes([
    0, 0, 0, 0x35, *b"meta", 0, 0, 0, 0, 0, 0, 0, 0x21, *b"hdlr", 0, 0, 0, 0, 0, 0, 0, 0, *b"mdirappl",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, *b"ilst",
])
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"sinf", b"schi"}
_ALLOWED = _CONTAINERS | {
    b"ftyp", b"mdat", b"free", b"skip", b"wide", b"udta", b"mvhd", b"tkhd", b"mdhd", b"hdlr", b"vmhd", b"smhd",
    b"dref", b"url ", b"urn ", b"stsd", b"stts", b"stss", b"ctts", b"stsc", b"stsz", b"stco", b"co64", b"sgpd",
    b"sbgp", b"sdtp", b"elst",
}
_RELAY_REJECTION = json.dumps({
    "error": "other",
    "message": "upload failed for {path}: relay error 422: "
               '{{"error":"media contains metadata or a non-canonical metadata channel"}}',
})


def _relay_accepts_mp4_boxes(data: bytes) -> bool:
    """Port of the relay's ``validate_mp4_metadata_free`` walk plus its moov-before-mdat check."""
    def walk(start: int, end: int) -> bool:
        off = start
        while off < end:
            size, kind = int.from_bytes(data[off:off + 4], "big"), data[off + 4:off + 8]
            header = 8
            if size == 1:
                size, header = int.from_bytes(data[off + 8:off + 16], "big"), 16
            elif size == 0:
                size = end - off
            if size < header or off + size > end or kind not in _ALLOWED:
                return False
            if kind == b"udta" and data[off + header:off + size] != _EMPTY_FFMPEG_UDTA:
                return False
            if kind in _CONTAINERS and not walk(off + header, off + size):
                return False
            off += size
        return True

    return walk(0, len(data)) and data.find(b"moov") < data.find(b"mdat")


class _CapturingCli:
    """Fake ``buzz`` CLI: snapshots each ``--file`` as it is uploaded, then answers with *reply*."""

    def __init__(self, reply=(0, json.dumps({"accepted": True, "event_id": "evt-video"}), "")):
        self.reply, self.uploads = reply, []

    async def __call__(self, args, *, input_text=None):
        path = Path(args[args.index("--file") + 1])
        self.uploads.append((path, path.read_bytes()))
        code, out, err = self.reply
        return code, out, err.replace("{path}", str(path))


@pytest.fixture
def ffmpeg_mp4(tmp_path):
    """A real H.264/AAC MP4 as plain ffmpeg writes it (``encoder`` tag in ``udta``)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    clip = tmp_path / "clip.mp4"
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-f", "lavfi",
         "-i", "sine=frequency=440", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(clip)], capture_output=True)
    if proc.returncode != 0:
        pytest.skip("ffmpeg could not produce an H.264/AAC sample")
    assert not _relay_accepts_mp4_boxes(clip.read_bytes()), "sample must reproduce the relay rejection"
    return clip


@pytest.mark.asyncio
async def test_send_video_uploads_relay_canonical_mp4_and_cleans_up(ffmpeg_mp4):
    adapter = _make_adapter()
    adapter._run_cli = cli = _CapturingCli()

    result = await adapter.send_video(CHANNEL, str(ffmpeg_mp4), caption="clip")

    assert result.success is True
    [(uploaded, data)] = cli.uploads
    assert _relay_accepts_mp4_boxes(data)
    assert b"Lavf" not in data  # the ffmpeg ``encoder`` tag is gone
    assert uploaded.name == ffmpeg_mp4.name  # the chat still shows the user's filename
    assert not uploaded.exists()  # the temp copy is removed after the send
    assert ffmpeg_mp4.read_bytes() != data  # the caller's file is left untouched


@pytest.mark.asyncio
async def test_failed_remux_sends_original_and_reports_metadata_rejection(tmp_path, monkeypatch):
    broken = tmp_path / "ffmpeg"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    monkeypatch.setattr(_buzz_mod, "_find_ffmpeg", lambda: str(broken))
    private = tmp_path / "private-dir"
    private.mkdir()
    video = private / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16)
    adapter = _make_adapter()
    adapter._run_cli = cli = _CapturingCli(reply=(4, "", _RELAY_REJECTION))

    result = await adapter.send_video(CHANNEL, str(video))

    assert [path for path, _ in cli.uploads] == [video]
    assert result.success is False
    assert "rejected clip.mp4" in result.error and "container metadata" in result.error
    assert str(private) not in result.error
