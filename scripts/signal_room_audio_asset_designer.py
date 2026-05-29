#!/usr/bin/env python3
"""Create procedural review SFX WAV assets from a Signal Room cue sheet."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
from pathlib import Path
from typing import Any


SAMPLE_RATE = 24000
CHANNELS = 1
BITS_PER_SAMPLE = 16


def read_cue_sheet(cue_sheet: Path) -> dict[str, Any]:
    return json.loads(cue_sheet.read_text())


def _clamp_sample(value: float) -> int:
    return max(-32767, min(32767, int(value)))


def _envelope(index: int, sample_count: int, attack: float = 0.015, release: float = 0.18) -> float:
    attack_samples = max(1, int(SAMPLE_RATE * attack))
    release_samples = max(1, int(SAMPLE_RATE * release))
    fade_in = min(1.0, index / attack_samples)
    fade_out = min(1.0, max(0, sample_count - index) / release_samples)
    return min(fade_in, fade_out)


def _tone(t: float, frequency: float, amount: float = 1.0) -> float:
    return amount * math.sin(2 * math.pi * frequency * t)


def _noise(seed: random.Random) -> float:
    return seed.uniform(-1.0, 1.0)


def _cue_sample(cue_id: str, index: int, sample_count: int, seed: random.Random) -> float:
    t = index / SAMPLE_RATE
    env = _envelope(index, sample_count)
    if "room" in cue_id or "memory" in cue_id:
        pulse = 0.35 * _tone(t, 72.0) + 0.16 * _tone(t, 123.0)
        texture = 0.08 * _noise(seed)
        return env * 4200 * (pulse + texture)
    if "paper" in cue_id or "snap" in cue_id:
        transient = max(0.0, 1.0 - index / max(1, int(SAMPLE_RATE * 0.08)))
        return 10500 * transient * (0.65 * _noise(seed) + 0.35 * _tone(t, 1240.0))
    if "tick" in cue_id:
        base = 620.0 + (sum(ord(char) for char in cue_id) % 5) * 85.0
        transient = max(0.0, 1.0 - index / max(1, int(SAMPLE_RATE * 0.09)))
        body = _tone(t, base) + 0.45 * _tone(t, base * 1.7)
        return 9200 * transient * body
    if "thump" in cue_id or "wall" in cue_id:
        drop = max(0.0, 1.0 - index / max(1, int(SAMPLE_RATE * 0.55)))
        return 12500 * drop * (_tone(t, 88.0) + 0.24 * _noise(seed))
    if "machine" in cue_id:
        rhythm = 1.0 if int(t * 8) % 2 == 0 else 0.42
        motor = _tone(t, 145.0) + 0.4 * _tone(t, 290.0) + 0.18 * _noise(seed)
        return env * 7200 * rhythm * motor
    if "point" in cue_id or "gesture" in cue_id:
        transient = max(0.0, 1.0 - index / max(1, int(SAMPLE_RATE * 0.11)))
        return 6500 * transient * (0.5 * _noise(seed) + 0.5 * _tone(t, 760.0))
    if "lever" in cue_id or "click" in cue_id:
        transient = max(0.0, 1.0 - index / max(1, int(SAMPLE_RATE * 0.16)))
        latch = _tone(t, 210.0) + 0.7 * _tone(t, 940.0)
        return 11000 * transient * latch
    return env * 5500 * (_tone(t, 440.0) + 0.12 * _noise(seed))


def wav_bytes(duration: float, cue_id: str) -> bytes:
    sample_count = max(1, int(SAMPLE_RATE * duration))
    seed = random.Random(cue_id)
    samples = [
        struct.pack("<h", _clamp_sample(_cue_sample(cue_id, index, sample_count, seed)))
        for index in range(sample_count)
    ]
    data = b"".join(samples)
    byte_rate = SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE // 8
    block_align = CHANNELS * BITS_PER_SAMPLE // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, CHANNELS, SAMPLE_RATE, byte_rate, block_align, BITS_PER_SAMPLE)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def create_review_audio_assets(cue_sheet: Path, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sheet = read_cue_sheet(cue_sheet)
    assets = []
    for cue in sheet.get("cues", []):
        cue_id = str(cue["id"])
        duration = float(cue.get("duration", 0.1))
        filename = f"{cue_id}.wav"
        (out_dir / filename).write_bytes(wav_bytes(duration, cue_id))
        assets.append(
            {
                "cue_id": cue_id,
                "filename": filename,
                "start": cue.get("start"),
                "duration": duration,
                "sound": cue.get("sound"),
                "placeholder": False,
                "replacement_required": "final mix/master approval",
            }
        )

    manifest = {
        "status": "review-only",
        "public_release": False,
        "source": "procedural review sound design",
        "sample_rate": SAMPLE_RATE,
        "assets": assets,
    }
    (out_dir / "audio_asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-sheet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace existing output directory")
    args = parser.parse_args()

    manifest = create_review_audio_assets(args.cue_sheet, args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
