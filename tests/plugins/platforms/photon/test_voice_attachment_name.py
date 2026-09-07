"""Behavior tests for Photon voice-attachment upload naming (#88083).

spectrum-ts's iMessage ``voice()`` path re-encodes non-m4a audio to an m4a
container (``ensureM4a``) but uploads the result under the original basename
of the delivered path — e.g. a TTS-cache ``1234.mp3``. iOS receives a voice
note whose ``.mp3`` file name disagrees with its m4a payload and renders it
as a short, unplayable bubble, while the same bytes hand-posted to the sidecar
as a real ``.m4a`` play end-to-end. The sidecar must therefore name every
voice upload what the payload will actually be.

The naming decision lives in
``plugins/platforms/photon/sidecar/voice-attachment-name.mjs`` (imported by
index.mjs's ``/send-attachment`` handler). These tests *execute* that real
module under node and assert the resulting upload name for representative
inputs — they do not read the sidecar source.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

_MODULE = Path("plugins/platforms/photon/sidecar/voice-attachment-name.mjs").resolve()

_CASES: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    # name: (path, explicit name, expected upload name)
    "tts_cache_mp3_renamed_to_m4a": (
        "/home/u/.hermes/audio_cache/1757212345.mp3",
        None,
        "1757212345.m4a",
    ),
    "ogg_voice_renamed_to_m4a": (
        "/tmp/out.ogg",
        None,
        "out.m4a",
    ),
    "wav_voice_renamed_to_m4a": (
        "/tmp/clip.wav",
        "clip.wav",
        "clip.m4a",
    ),
    "aac_adts_renamed_to_m4a": (
        # Bare ADTS .aac has no ftyp header, so spectrum-ts re-encodes it too.
        "/tmp/note.aac",
        None,
        "note.m4a",
    ),
    "m4a_keeps_its_name": (
        "/tmp/voice.m4a",
        None,
        None,
    ),
    "explicit_m4a_name_kept_verbatim": (
        "/tmp/anything.mp3",
        "reply.m4a",
        "reply.m4a",
    ),
    "extensionless_name_gets_m4a": (
        "/tmp/blob",
        "voiceclip",
        "voiceclip.m4a",
    ),
}


@pytest.fixture(scope="module")
def verdicts() -> Dict[str, Optional[str]]:
    """Run every case through the real voice-attachment-name module in one node call."""
    harness = (
        f"import {{ voiceAttachmentName }} from {json.dumps(_MODULE.as_uri())};\n"
        "const chunks = [];\n"
        "process.stdin.on('data', (c) => chunks.push(c));\n"
        "process.stdin.on('end', () => {\n"
        "  const cases = JSON.parse(Buffer.concat(chunks).toString('utf-8'));\n"
        "  const out = {};\n"
        "  for (const [name, [path, explicit]] of Object.entries(cases)) {\n"
        "    out[name] = voiceAttachmentName(path, explicit ?? undefined) ?? null;\n"
        "  }\n"
        "  process.stdout.write(JSON.stringify(out));\n"
        "});\n"
    )
    payload = {name: [path, explicit] for name, (path, explicit, _) in _CASES.items()}
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        input=json.dumps(payload),
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_voice_upload_name(name: str, verdicts: Dict[str, Optional[str]]) -> None:
    _, _, expected = _CASES[name]
    assert verdicts[name] == expected
