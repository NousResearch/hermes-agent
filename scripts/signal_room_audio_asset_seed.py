#!/usr/bin/env python3
"""Create review-only placeholder WAV assets from a Signal Room cue sheet."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16


def cue_frequency(cue_id: str) -> float:
    if "room" in cue_id or "memory" in cue_id:
        return 110.0
    if "tick" in cue_id or "snap" in cue_id:
        return 880.0
    if "thump" in cue_id or "lever" in cue_id:
        return 180.0
    if "machine" in cue_id:
        return 240.0
    return 440.0


def read_cue_sheet(cue_sheet: Path) -> dict[str, Any]:
    return json.loads(cue_sheet.read_text())


def wav_bytes(duration: float, cue_id: str) -> bytes:
    sample_count = max(1, int(SAMPLE_RATE * duration))
    frequency = cue_frequency(cue_id)
    amplitude = 9000
    samples = []
    for index in range(sample_count):
        t = index / SAMPLE_RATE
        envelope = min(1.0, index / max(1, int(SAMPLE_RATE * 0.015)))
        if index > sample_count * 0.72:
            envelope *= max(0.0, (sample_count - index) / max(1, sample_count * 0.28))
        sample = int(amplitude * envelope * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", sample))
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


def create_audio_asset_seed(cue_sheet: Path, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
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
                "duration": duration,
                "placeholder": True,
                "replacement_required": "final sound design asset",
            }
        )

    manifest = {
        "status": "review-only",
        "public_release": False,
        "source": "placeholder seed; replace with final sound design",
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

    manifest = create_audio_asset_seed(args.cue_sheet, args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
