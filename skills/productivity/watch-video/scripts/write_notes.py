#!/usr/bin/env python3
"""
Notes writer — generates structured markdown notes from timeline.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import TimelineEntry, VisionResult, NotesResult, write_json, print_json


def load_timeline(path: str) -> list[TimelineEntry]:
    """Load timeline from JSON."""
    data = json.loads(Path(path).read_text())
    timeline = data.get("timeline", [])
    return [TimelineEntry.from_dict(t) for t in timeline]


def load_meta(path: str) -> dict:
    """Load metadata from JSON."""
    return json.loads(Path(path).read_text())


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def generate_synthesis(timeline: list[TimelineEntry], meta: dict) -> dict:
    """Generate summary, TL;DR, key quotes using LLM (placeholder for now)."""
    # This is a placeholder - in production, this would call an LLM
    # For now, use rule-based extraction

    # Collect all transcript text
    transcript_texts = [e.transcript for e in timeline if e.transcript]
    full_transcript = " ".join(transcript_texts)

    # Simple summary: first 200 chars of transcript or visual descriptions
    if full_transcript:
        summary = full_transcript[:200] + ("..." if len(full_transcript) > 200 else "")
    else:
        visual_descs = [e.visual.description for e in timeline if e.visual]
        summary = " ".join(visual_descs[:3])[:200] + "..."

    # TL;DR: key points from transcript
    tldr = []
    sentences = [s.strip() for s in full_transcript.split(".") if len(s.strip()) > 20]
    for s in sentences[:5]:
        tldr.append(s[:150] + ("..." if len(s) > 150 else ""))

    # Key quotes: longer transcript segments
    key_quotes = []
    for e in timeline:
        if e.transcript and len(e.transcript) > 50:
            key_quotes.append([e.timestamp, e.transcript[:200]])
            if len(key_quotes) >= 5:
                break

    # Visual highlights
    visual_highlights = []
    for e in timeline:
        if e.visual and e.visual.description:
            visual_highlights.append([e.timestamp, e.visual.description[:200]])

    return {
        "summary": summary,
        "tldr": tldr,
        "key_quotes": key_quotes,
        "visual_highlights": visual_highlights,
    }


def render_notes(
    title: str,
    url: str,
    synthesis: dict,
    timeline: list[TimelineEntry],
    visual_notes: list[VisionResult],
) -> str:
    """Render markdown notes from template."""
    lines = []

    # Title
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Source:** {url}")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(synthesis["summary"])
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    lines.append("")
    for point in synthesis["tldr"]:
        lines.append(f"- {point}")
    lines.append("")

    # Timeline
    lines.append("## Timeline")
    lines.append("")

    for entry in timeline:
        ts = format_timestamp(entry.timestamp)
        lines.append(f"### {ts}")

        if entry.transcript:
            lines.append(f"**Transcript:** {entry.transcript}")

        if entry.visual:
            v = entry.visual
            lines.append(f"**Visual:** {v.description}")
            if v.ocr_text:
                lines.append(f"**OCR:** {v.ocr_text}")
            if v.ui_elements:
                lines.append(f"**UI Elements:** {', '.join(v.ui_elements)}")
            lines.append(f"**Scene:** {v.scene_type}")
            if v.speaker_change:
                lines.append("**Speaker Change:** Yes")

        lines.append("")

    # Key Quotes
    lines.append("## Key Quotes")
    lines.append("")
    for ts, quote in synthesis["key_quotes"]:
        lines.append(f"- [{format_timestamp(ts)}] {quote}")
    lines.append("")

    # Visual Notes
    lines.append("## Visual Notes")
    lines.append("")
    for v in visual_notes:
        if v.description:
            ts = format_timestamp(v.timestamp)
            lines.append(f"### {ts} (Frame {v.frame_index})")
            lines.append(v.description)
            if v.ocr_text:
                lines.append(f"**Text:** {v.ocr_text}")
            if v.ui_elements:
                lines.append(f"**UI:** {', '.join(v.ui_elements)}")
            lines.append(f"**Scene:** {v.scene_type}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Write markdown notes from timeline")
    parser.add_argument("--timeline", required=True, help="Timeline JSON file")
    parser.add_argument("--meta", required=True, help="Metadata JSON file")
    parser.add_argument("--out-dir", default="~/.hermes/watch-video-notes", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    try:
        timeline = load_timeline(args.timeline)
        meta = load_meta(args.meta)

        title = meta.get("title", "Video Notes")
        url = meta.get("url", meta.get("path", "Unknown"))

        # Generate synthesis
        synthesis = generate_synthesis(timeline, meta)

        # Collect visual notes
        visual_notes = [e.visual for e in timeline if e.visual]

        # Render markdown
        markdown = render_notes(title, url, synthesis, timeline, visual_notes)

        # Write file
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        video_id = meta.get("video_id", "video")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{video_id}-{timestamp}.md"
        notes_path = out_dir / filename

        notes_path.write_text(markdown, encoding="utf-8")

        result = NotesResult(
            notes_path=str(notes_path),
            title=title,
            summary=synthesis["summary"],
            tldr=synthesis["tldr"],
            timeline=timeline,
            key_quotes=synthesis["key_quotes"],
            visual_notes=visual_notes,
        )

        if args.json:
            print_json(result)
        else:
            print(str(notes_path))

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())