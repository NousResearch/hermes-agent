"""Regression coverage for Photon native voice-note transcoding.

Hermes TTS commonly emits MP3. Spectrum's ``voice()`` builder accepts AAC/M4A
natively and uses ffmpeg for other formats. The Photon sidecar must therefore
declare ffmpeg-static directly rather than relying on Spectrum's optional peer
dependency, which npm is free to omit.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIDECAR_PACKAGE = ROOT / "plugins" / "platforms" / "photon" / "sidecar" / "package.json"


def test_photon_sidecar_installs_voice_transcoder() -> None:
    package = json.loads(SIDECAR_PACKAGE.read_text(encoding="utf-8"))

    assert "ffmpeg-static" in package["dependencies"], (
        "Photon sends MP3 TTS through Spectrum's voice() builder, which needs "
        "ffmpeg-static to transcode it to AAC/M4A"
    )
