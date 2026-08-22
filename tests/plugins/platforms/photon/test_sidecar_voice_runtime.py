"""Regression contract for Photon outbound voice identity.

Live failure: Hermes generated a distinct xAI TTS MP3, the gateway extracted
that MEDIA path, and Spectrum voice() still made iMessage play the inbound CAF
voice note. Spectrum 8 uploadVoice() transcodes through ensureM4a() but keeps
the source basename (tts_*.mp3) as the Photon upload filename, so isAudioMessage
delivery can collapse onto the conversation's original voice-memo slot.

The sidecar must materialize a unique AAC/M4A object whose bytes and filename
are distinct from the inbound CAF before calling voice().
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from gateway.platforms.base import BasePlatformAdapter


def _sidecar_dir() -> Path:
    return Path(__file__).parents[4] / "plugins" / "platforms" / "photon" / "sidecar"


def _ffmpeg() -> Path:
    return _sidecar_dir() / "node_modules" / "ffmpeg-static" / "ffmpeg"


def _write_wav(path: Path, *, seconds: float, sample: bytes = b"\x00\x00") -> None:
    frames = int(24_000 * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(sample * frames)


def test_sidecar_declares_ffmpeg_static_for_voice_transcoding() -> None:
    sidecar = _sidecar_dir()
    package = json.loads((sidecar / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((sidecar / "package-lock.json").read_text(encoding="utf-8"))

    assert "ffmpeg-static" in package["dependencies"], (
        "Photon voice() transcodes MP3/WAV input via ffmpeg; the sidecar must "
        "ship ffmpeg-static instead of assuming a host binary."
    )
    assert lock["packages"][""]["dependencies"]["ffmpeg-static"] == package[
        "dependencies"
    ]["ffmpeg-static"]
    assert "node_modules/ffmpeg-static" in lock["packages"]


def test_extract_media_keeps_tts_path_not_sibling_inbound_caf(tmp_path: Path) -> None:
    """The live 121-char reply tagged the TTS file; inbound CAF must not ride along."""
    inbound = tmp_path / "audio_01f0c3748bc6.caf"
    tts = tmp_path / "tts_20260821_021929_492327.mp3"
    inbound.write_bytes(b"caffinbound")
    tts.write_bytes(b"id3tts-bytes")
    media, text = BasePlatformAdapter.extract_media(
        f"Here she is, Tom.\n\nMEDIA:{tts}"
    )
    assert [path for path, _is_voice in media] == [str(tts)]
    assert str(inbound) not in text
    assert all(str(inbound) != path for path, _is_voice in media)


@pytest.mark.skipif(
    shutil.which("node") is None or not _ffmpeg().is_file(),
    reason="Photon sidecar npm dependencies are not installed",
)
def test_spectrum_voice_conversion_uses_bundled_ffmpeg(tmp_path: Path) -> None:
    """A non-M4A voice input must convert through the shipped static binary."""
    source = tmp_path / "voice.wav"
    _write_wav(source, seconds=0.25)

    script = f"""
import {{ readFile }} from 'node:fs/promises';
import {{ ensureM4a }} from '@spectrum-ts/core/authoring';
const source = await readFile({json.dumps(str(source))});
const result = await ensureM4a(source, 'audio/wav');
const brand = result.buffer.toString('ascii', 8, 12);
console.log(JSON.stringify({{ brand, bytes: result.buffer.length }}));
if (!['M4A ', 'M4B ', 'M4P ', 'mp42', 'mp41', 'isom', 'iso2'].includes(brand)) process.exit(1);
"""
    run = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_sidecar_dir(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["brand"] in {
        "M4A ",
        "M4B ",
        "M4P ",
        "mp42",
        "mp41",
        "isom",
        "iso2",
    }


@pytest.mark.skipif(
    shutil.which("node") is None or not _ffmpeg().is_file(),
    reason="Photon sidecar npm dependencies are not installed",
)
def test_prepare_voice_attachment_does_not_reuse_inbound_caf_identity(
    tmp_path: Path,
) -> None:
    """TTS upload bytes and filename must be distinct from the inbound CAF.

    This is the live loopback: inbound CAF (7.56s) vs xAI TTS MP3 (16.6s).
    Spectrum voice() used to upload the transcoded TTS under tts_*.mp3, and
    iMessage played the original voice memo. The sidecar must emit a unique
    M4A whose hash is not the inbound file's hash.
    """
    inbound_wav = tmp_path / "inbound.wav"
    tts_wav = tmp_path / "tts.wav"
    inbound_caf = tmp_path / "inbound.caf"
    _write_wav(inbound_wav, seconds=0.4, sample=b"\x11\x00")
    _write_wav(tts_wav, seconds=0.9, sample=b"\x7f\xff")
    transcode = subprocess.run(
        [
            str(_ffmpeg()),
            "-y",
            "-i",
            str(inbound_wav),
            "-f",
            "caf",
            str(inbound_caf),
        ],
        cwd=_sidecar_dir(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert transcode.returncode == 0, transcode.stderr
    inbound_sha = hashlib.sha256(inbound_caf.read_bytes()).hexdigest()

    module = (_sidecar_dir() / "voice-send.mjs").resolve()
    script = f"""
import {{ prepareVoiceAttachment }} from {json.dumps(module.as_uri())};
const caf = await prepareVoiceAttachment({json.dumps(str(inbound_caf))});
const tts = await prepareVoiceAttachment({json.dumps(str(tts_wav))});
try {{
  console.log(JSON.stringify({{
    caf: {{
      name: caf.opts.name,
      mimeType: caf.opts.mimeType,
      uploadSha: caf.uploadSha,
      brand: caf.brand,
      sourceSha: caf.sourceSha,
    }},
    tts: {{
      name: tts.opts.name,
      mimeType: tts.opts.mimeType,
      uploadSha: tts.uploadSha,
      brand: tts.brand,
      sourceSha: tts.sourceSha,
    }},
  }}));
}} finally {{
  await caf.cleanup();
  await tts.cleanup();
}}
"""
    run = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_sidecar_dir(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    caf = payload["caf"]
    tts = payload["tts"]

    assert caf["mimeType"] == "audio/mp4"
    assert tts["mimeType"] == "audio/mp4"
    assert caf["name"].endswith(".m4a")
    assert tts["name"].endswith(".m4a")
    assert caf["name"] != tts["name"]
    assert not caf["name"].endswith(".caf")
    assert not tts["name"].endswith(".mp3")
    assert caf["brand"] in {"M4A ", "M4B ", "M4P ", "mp42", "mp41", "isom", "iso2"}
    assert tts["brand"] in {"M4A ", "M4B ", "M4P ", "mp42", "mp41", "isom", "iso2"}
    assert caf["sourceSha"] == inbound_sha
    assert tts["uploadSha"] != inbound_sha
    assert tts["uploadSha"] != caf["uploadSha"]
    assert tts["uploadSha"] != tts["sourceSha"]
