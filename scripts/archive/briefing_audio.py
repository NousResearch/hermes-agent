#!/usr/bin/env python3
"""Render briefing text to an MP3 via on-box Kokoro. Run under the repo .venv.

Usage: briefing_audio.py <text_file> <out_mp3> [voice]
Kept separate from the briefing because Kokoro (kokoro_onnx + the onnx model)
only loads under the repo .venv, while the briefing runs under system python3.
The briefing shells out to this with the .venv interpreter when, and only when,
BRIEFING_AUDIO is enabled.

This helper must stay at ~/.hermes/scripts/ (symlinked to the repo scripts/) and
the repo must stay at /home/kensei/repos/KenseiAgent; the briefing hardcodes both
paths. If either moves, audio degrades to off (fail-safe), it does not crash.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/kensei/repos/KenseiAgent")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: briefing_audio.py <text_file> <out_mp3> [voice]", file=sys.stderr)
        return 2
    text_file, out_mp3 = sys.argv[1], sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "kokoro:bf_emma"
    engine, vid = voice.split(":", 1) if ":" in voice else ("kokoro", voice)

    text = ""
    try:
        text = open(text_file, encoding="utf-8").read().strip()
    except OSError as e:
        print(f"cannot read text file: {e}", file=sys.stderr)
        return 2
    if not text:
        print("empty text, nothing to synthesise", file=sys.stderr)
        return 2

    from plugins.tts.kenseivoice import _to_wav, _transcode

    wav = out_mp3 + ".wav"
    _to_wav(engine, vid, text, wav, 1.0)
    _transcode(wav, out_mp3, "mp3")
    try:
        os.remove(wav)
    except OSError:
        pass
    return 0 if os.path.exists(out_mp3) else 1


if __name__ == "__main__":
    sys.exit(main())
