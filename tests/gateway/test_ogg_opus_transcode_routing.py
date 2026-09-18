"""Every voice-bubble transcode routes through ``gateway.platforms.base.transcode_to_ogg_opus``.

Matrix, WhatsApp Cloud and the TTS tool each used to run their own ffmpeg argv; the shared
helper is the only place the codec flags, the timeout and the in-place-safe write live.
"""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace


def _capture_ffmpeg(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        out = argv[-1]
        with open(out, "wb") as fh:
            fh.write(b"OggS")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("gateway.platforms.base.subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    return calls


def _bitrate(argv):
    return argv[argv.index("-b:a") + 1]


def test_all_sites_share_one_ffmpeg_invocation(monkeypatch, tmp_path):
    from gateway.platforms import whatsapp_cloud
    from plugins.platforms.matrix import adapter as matrix
    from tools import tts_tool_delivery as tts

    calls = _capture_ffmpeg(monkeypatch)
    src = tmp_path / "speech.mp3"
    src.write_bytes(b"ID3")

    # Matrix: temp output at 48k.
    matrix_out = asyncio.run(asyncio.to_thread(matrix.transcode_to_ogg_opus, str(src), bitrate="48k", timeout=30))
    assert matrix_out and matrix_out.endswith(".ogg") and _bitrate(calls[-1][0]) == "48k"
    assert calls[-1][1]["timeout"] == 30

    # WhatsApp Cloud: sibling .ogg at the 32k default, warn-once on missing ffmpeg untouched.
    monkeypatch.setattr(whatsapp_cloud, "_FFMPEG_PATH", "/usr/bin/ffmpeg")
    wa = object.__new__(whatsapp_cloud.WhatsAppCloudAdapter)
    wa._warned_no_ffmpeg = False
    wa_out = asyncio.run(wa._convert_to_opus(str(src)))
    assert wa_out == str(tmp_path / "speech.ogg") and _bitrate(calls[-1][0]) == "32k"

    # TTS tool: sibling .ogg at 48k, and in-place repair never writes straight over the source.
    assert tts._convert_to_opus(str(src)) == str(tmp_path / "speech.ogg") and _bitrate(calls[-1][0]) == "48k"
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"ID3")
    assert tts._repair_ogg_container(str(bad)) == str(bad)
    assert calls[-1][0][-1] == str(bad) + ".tmp.ogg" and bad.read_bytes() == b"OggS"

    # Every argv carries the voice-tuned flag set exactly once.
    for argv, kwargs in calls:
        assert argv[argv.index("-application") + 1] == "voip" and "-compression_level" in argv
        assert kwargs["stdin"] is subprocess.DEVNULL


def test_failed_in_place_repair_keeps_the_source(monkeypatch, tmp_path):
    from gateway.platforms.base import transcode_to_ogg_opus

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("gateway.platforms.base.subprocess.run",
                        lambda argv, **kw: SimpleNamespace(returncode=1, stderr=b"boom"))
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"ID3")
    assert transcode_to_ogg_opus(str(bad), output_path=str(bad)) is None
    assert bad.read_bytes() == b"ID3" and not (tmp_path / "bad.ogg.tmp.ogg").exists()
