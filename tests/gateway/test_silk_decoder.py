"""Tests for the shared SILK / ffmpeg audio decoding helpers."""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

from gateway.platforms import _silk


# ---------------------------------------------------------------------------
# looks_like_silk
# ---------------------------------------------------------------------------

class TestLooksLikeSilk:
    def test_classic_silk_v3_magic(self):
        assert _silk.looks_like_silk(b"#!SILK_V3\x01\x02\x03")

    def test_short_silk_magic(self):
        assert _silk.looks_like_silk(b"#!SILK\x00\x00")

    def test_framing_byte_variant(self):
        # Tencent's wire format sometimes prefixes with 0x02 0x21
        assert _silk.looks_like_silk(b"\x02!\x00\x01\x02")

    def test_weixin_framed_silk_variant(self):
        # Weixin payloads sometimes have 0x02 then #!SILK
        assert _silk.looks_like_silk(b"\x02#!SILK_V3\x01\x02")

    def test_wav_is_not_silk(self):
        assert not _silk.looks_like_silk(b"RIFF\x00\x00\x00\x00WAVE")

    def test_mp3_is_not_silk(self):
        assert not _silk.looks_like_silk(b"\xff\xfb\x90\x00")

    def test_empty_is_not_silk(self):
        assert not _silk.looks_like_silk(b"")


# ---------------------------------------------------------------------------
# silk_to_wav
# ---------------------------------------------------------------------------

class TestSilkToWav:
    def test_returns_none_when_pilk_missing(self, monkeypatch, tmp_path, caplog):
        # Force ImportError on `import pilk`
        monkeypatch.setitem(sys.modules, "pilk", None)

        src = tmp_path / "v.silk"
        src.write_bytes(b"#!SILK_V3 fake")
        wav = tmp_path / "v.wav"

        with caplog.at_level("WARNING"):
            result = _silk.silk_to_wav(str(src), str(wav), log_tag="test")
        assert result is None
        assert "pilk not installed" in caplog.text

    def test_returns_none_when_src_missing(self, tmp_path):
        wav = tmp_path / "v.wav"
        result = _silk.silk_to_wav(str(tmp_path / "missing.silk"), str(wav))
        assert result is None

    def test_calls_pilk_and_returns_wav_path(self, monkeypatch, tmp_path):
        # Fake a pilk that writes a valid (small) WAV header to wav_path.
        called = {}

        def fake_silk_to_wav(src, dst, rate=16000):
            called["src"] = src
            called["dst"] = dst
            called["rate"] = rate
            _write_minimal_wav(dst)

        fake_pilk = type(sys)("pilk")
        fake_pilk.silk_to_wav = fake_silk_to_wav
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)

        src = tmp_path / "v.silk"
        src.write_bytes(b"#!SILK_V3 ...")
        wav = tmp_path / "v.wav"

        result = _silk.silk_to_wav(str(src), str(wav))
        assert result == str(wav)
        assert called["rate"] == 16000

    def test_retries_with_silk_suffix_when_direct_fails(self, monkeypatch, tmp_path):
        # First call raises (pilk's extension check), second (.silk path) succeeds.
        call_log = []

        def fake_silk_to_wav(src, dst, rate=16000):
            call_log.append(src)
            if src.endswith(".silk"):
                _write_minimal_wav(dst)
                return
            raise RuntimeError("pilk: bad extension")

        fake_pilk = type(sys)("pilk")
        fake_pilk.silk_to_wav = fake_silk_to_wav
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)

        # Source has .bin extension to force the rename branch.
        src = tmp_path / "v.bin"
        src.write_bytes(b"#!SILK_V3 ...")
        wav = tmp_path / "v.wav"

        result = _silk.silk_to_wav(str(src), str(wav))
        assert result == str(wav)
        assert len(call_log) == 2
        assert call_log[0].endswith(".bin")
        assert call_log[1].endswith(".silk")

    def test_returns_none_when_pilk_produces_empty_wav(self, monkeypatch, tmp_path):
        def fake_silk_to_wav(src, dst, rate=16000):
            Path(dst).write_bytes(b"")  # empty output

        fake_pilk = type(sys)("pilk")
        fake_pilk.silk_to_wav = fake_silk_to_wav
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)

        src = tmp_path / "v.silk"
        src.write_bytes(b"#!SILK_V3 ...")
        wav = tmp_path / "v.wav"

        result = _silk.silk_to_wav(str(src), str(wav))
        assert result is None


# ---------------------------------------------------------------------------
# ffmpeg_to_wav (uses asyncio.create_subprocess_exec — must be mocked)
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode=0, stderr_bytes=b"", emit_wav_to=None):
        self.returncode = returncode
        self._stderr_bytes = stderr_bytes
        self._emit_wav_to = emit_wav_to
        self.stderr = self  # let .read() on the same object work

    async def wait(self):
        if self._emit_wav_to:
            _write_minimal_wav(self._emit_wav_to)
        return self.returncode

    async def read(self):
        return self._stderr_bytes

    def kill(self):
        pass


class TestFfmpegToWav:
    def test_returns_none_when_ffmpeg_missing(self, monkeypatch, tmp_path, caplog):
        async def fake_exec(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not on PATH")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        src = tmp_path / "in.ogg"
        src.write_bytes(b"OggS fake")
        wav = tmp_path / "out.wav"

        with caplog.at_level("WARNING"):
            result = asyncio.run(_silk.ffmpeg_to_wav(str(src), str(wav), log_tag="t"))
        assert result is None
        assert "ffmpeg not installed" in caplog.text

    def test_returns_wav_on_success(self, monkeypatch, tmp_path):
        wav = tmp_path / "out.wav"

        async def fake_exec(*args, **kwargs):
            # Pull the output wav path from the positional args
            return _FakeProc(returncode=0, emit_wav_to=str(wav))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        src = tmp_path / "in.ogg"
        src.write_bytes(b"OggS")

        result = asyncio.run(_silk.ffmpeg_to_wav(str(src), str(wav)))
        assert result == str(wav)

    def test_returns_none_on_ffmpeg_nonzero_exit(self, monkeypatch, tmp_path, caplog):
        async def fake_exec(*args, **kwargs):
            return _FakeProc(returncode=1, stderr_bytes=b"Invalid data found")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        src = tmp_path / "in.silk"
        src.write_bytes(b"#!SILK_V3")
        wav = tmp_path / "out.wav"

        with caplog.at_level("WARNING"):
            result = asyncio.run(_silk.ffmpeg_to_wav(str(src), str(wav), log_tag="t"))
        assert result is None
        assert "ffmpeg failed" in caplog.text


# ---------------------------------------------------------------------------
# ensure_wav (the integrated SILK-first / ffmpeg-fallback path)
# ---------------------------------------------------------------------------

class TestEnsureWav:
    def test_silk_path_tries_pilk_first(self, monkeypatch, tmp_path):
        # pilk succeeds, ffmpeg should not be called
        async def fake_exec(*args, **kwargs):
            raise AssertionError("ffmpeg should not be invoked when pilk succeeds")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        def fake_silk_to_wav(src, dst, rate=16000):
            _write_minimal_wav(dst)

        fake_pilk = type(sys)("pilk")
        fake_pilk.silk_to_wav = fake_silk_to_wav
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)

        src = tmp_path / "v.silk"
        src.write_bytes(b"#!SILK_V3 abc")

        result = asyncio.run(_silk.ensure_wav(str(src), sniff_bytes=b"#!SILK_V3 abc"))
        assert result is not None
        assert result.endswith(".wav")

    def test_falls_back_to_ffmpeg_when_pilk_returns_none(self, monkeypatch, tmp_path):
        # Force pilk to fail, ffmpeg to succeed
        def fake_silk_to_wav(src, dst, rate=16000):
            raise RuntimeError("decode failed")

        fake_pilk = type(sys)("pilk")
        fake_pilk.silk_to_wav = fake_silk_to_wav
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)

        wav_target = tmp_path / "v.wav"

        async def fake_exec(*args, **kwargs):
            return _FakeProc(returncode=0, emit_wav_to=str(wav_target))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        src = tmp_path / "v.silk"
        src.write_bytes(b"#!SILK_V3 abc")

        result = asyncio.run(
            _silk.ensure_wav(str(src), wav_path=str(wav_target), sniff_bytes=b"#!SILK_V3")
        )
        assert result == str(wav_target)

    def test_non_silk_source_skips_pilk(self, monkeypatch, tmp_path):
        # If we never set up fake pilk, importing it would fail — which is
        # fine, because we expect ensure_wav to skip the silk branch entirely
        # for non-silk payloads and go straight to ffmpeg.
        wav_target = tmp_path / "out.wav"

        async def fake_exec(*args, **kwargs):
            return _FakeProc(returncode=0, emit_wav_to=str(wav_target))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        src = tmp_path / "in.ogg"
        src.write_bytes(b"OggS\x00fakeogg")

        result = asyncio.run(
            _silk.ensure_wav(
                str(src), wav_path=str(wav_target), sniff_bytes=b"OggS\x00fakeogg"
            )
        )
        assert result == str(wav_target)

    def test_returns_none_when_src_missing(self, tmp_path):
        result = asyncio.run(_silk.ensure_wav(str(tmp_path / "nope.silk")))
        assert result is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_minimal_wav(path: str) -> None:
    """Write a minimal but non-empty WAV (>44 bytes) file at ``path``."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 100 frames of silence == 200 bytes of sample data
        wf.writeframes(b"\x00\x00" * 100)
