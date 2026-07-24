#!/usr/bin/env python3
"""Generate synthetic positive samples of a wake phrase for openWakeWord.

OpenAI TTS (``gpt-4o-mini-tts``) speaks the phrase across many voices with small
text/pacing variations, resampled to the 16 kHz mono WAV openWakeWord trains on.
This is the higher-quality alternative to openWakeWord's built-in Piper
generator; use it when ``OPENAI_API_KEY`` is set.

Heavy deps (openai, soundfile, numpy, scipy) are imported lazily inside the
synthesis path so the pure planning helpers stay import-light and unit-testable.
"""

from __future__ import annotations

import argparse
import re

# OpenAI's standard TTS voice set — cycling these gives speaker diversity, which
# is what drives custom wake-word robustness (openWakeWord's own models lean on
# exactly this kind of multi-voice synthetic data).
OPENAI_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "onyx", "nova", "sage", "shimmer",
)

TARGET_SAMPLE_RATE = 16_000


def slugify(phrase: str) -> str:
    """`"Hey Morgane!"` → `"hey_morgane"` — the model filename stem."""
    s = re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower())
    return s.strip("_") or "wake_word"


def build_variations(phrase: str) -> list[str]:
    """Small textual variations so the TTS doesn't render one frozen prosody."""
    core = phrase.strip().rstrip(".!?,")
    # Punctuation nudges cadence/intonation; duplicates are dropped by dict order.
    seen = {core: None, f"{core}.": None, f"{core}!": None, f"{core}...": None}
    return list(seen)


def voice_list(engine: str = "openai") -> list[str]:
    if engine == "openai":
        return list(OPENAI_VOICES)
    raise ValueError(f"unknown engine: {engine!r}")


def plan_clips(phrase: str, count: int, voices: list[str]) -> list[tuple[str, str, str]]:
    """Deterministic (voice, text, filename) plan for *count* clips.

    Voices and text variations are cycled independently so the set stays evenly
    balanced across speakers regardless of *count*. Filenames are zero-padded and
    unique, ready to drop into openWakeWord's positive-samples directory.
    """
    if count < 1:
        return []
    if not voices:
        raise ValueError("need at least one voice")
    variations = build_variations(phrase)
    stem = slugify(phrase)
    width = max(4, len(str(count - 1)))
    plan = []
    for i in range(count):
        voice = voices[i % len(voices)]
        text = variations[i % len(variations)]
        plan.append((voice, text, f"{stem}_{i:0{width}d}.wav"))
    return plan


def _resample_to_16k_mono(pcm, src_rate: int):
    """int16 mono PCM at *src_rate* → int16 mono PCM at 16 kHz (lazy scipy)."""
    import numpy as np
    from scipy.signal import resample

    if src_rate == TARGET_SAMPLE_RATE:
        return pcm
    n = round(len(pcm) * TARGET_SAMPLE_RATE / src_rate)
    out = resample(pcm.astype("float32"), n)
    return np.clip(out, -32768, 32767).astype("int16")


def _synthesize(plan, out_dir, api_key):  # pragma: no cover - network/heavy deps
    """Call OpenAI TTS for each planned clip and write a 16 kHz mono WAV."""
    import io
    import wave
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for voice, text, name in plan:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice=voice, input=text, response_format="wav"
        )
        data, rate = sf.read(io.BytesIO(resp.read()), dtype="int16")
        if getattr(data, "ndim", 1) == 2:  # stereo → mono
            data = data.mean(axis=1).astype("int16")
        data = _resample_to_16k_mono(np.asarray(data), rate)
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SAMPLE_RATE)
            w.writeframes(np.asarray(data).tobytes())
    return len(plan)


def main(argv=None) -> int:
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrase", required=True, help='e.g. "hey morgane"')
    ap.add_argument("--out-dir", required=True, help="positive-samples directory")
    ap.add_argument("--count", type=int, default=500, help="clips to generate")
    ap.add_argument("--engine", default="openai", choices=["openai"])
    args = ap.parse_args(argv)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        ap.error(
            "OPENAI_API_KEY not set. Set it, or use openWakeWord's offline Piper "
            "generator instead (see the skill's references/platforms.md)."
        )

    plan = plan_clips(args.phrase, args.count, voice_list(args.engine))
    written = _synthesize(plan, args.out_dir, api_key)
    print(f"wrote {written} positive samples to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
