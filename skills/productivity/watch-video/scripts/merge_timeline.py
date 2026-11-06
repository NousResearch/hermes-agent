#!/usr/bin/env python3
"""
Timeline merger — combines transcript segments and vision results into unified timeline.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import TranscriptSegment, VisionResult, TimelineEntry, write_json, print_json


def load_transcript(path: str) -> list[TranscriptSegment]:
    """Load transcript segments from JSON."""
    data = json.loads(Path(path).read_text())
    segments = data.get("captions", data.get("transcript", []))
    return [TranscriptSegment.from_dict(s) for s in segments]


def load_vision(path: str) -> list[VisionResult]:
    """Load vision results from JSON."""
    data = json.loads(Path(path).read_text())
    results = data.get("results", data.get("vision", []))
    return [VisionResult.from_dict(r) for r in results]


def merge_timeline(
    transcript: list[TranscriptSegment],
    vision: list[VisionResult],
) -> list[TimelineEntry]:
    """Merge transcript and vision into unified timeline."""
    # Collect all unique timestamps
    timestamps = set()

    # Add transcript timestamps (segment boundaries)
    for seg in transcript:
        timestamps.add(round(seg.start, 3))
        timestamps.add(round(seg.end, 3))

    # Add vision timestamps
    for vis in vision:
        timestamps.add(round(vis.timestamp, 3))

    # Sort timestamps
    sorted_times = sorted(timestamps)

    # Build timeline entries
    timeline = []

    for ts in sorted_times:
        entry = TimelineEntry(timestamp=ts)

        # Find matching transcript (segment containing this timestamp)
        for seg in transcript:
            if seg.start <= ts <= seg.end:
                entry.transcript = seg.text
                break

        # Find matching vision (closest timestamp)
        best_vis = None
        best_diff = float("inf")
        for vis in vision:
            diff = abs(vis.timestamp - ts)
            if diff < best_diff:
                best_diff = diff
                best_vis = vis
        # Only attach vision if within reasonable window (5 seconds)
        if best_vis and best_diff <= 5.0:
            entry.visual = best_vis

        timeline.append(entry)

    return timeline


def main():
    parser = argparse.ArgumentParser(description="Merge transcript and vision into timeline")
    parser.add_argument("--transcript", required=True, help="Transcript JSON file")
    parser.add_argument("--vision", required=True, help="Vision results JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    try:
        transcript = load_transcript(args.transcript)
        vision = load_vision(args.vision)

        timeline = merge_timeline(transcript, vision)

        if args.json:
            print_json({"timeline": timeline})
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())