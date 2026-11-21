import json
import sys
from pathlib import Path

import pytest

SONOS_ROOT = Path.home() / ".hermes" / "sonos-brain"
if str(SONOS_ROOT) not in sys.path:
    sys.path.insert(0, str(SONOS_ROOT))

from sonos_brain import (
    build_podcast_readout_text,
    _prepare_sonos_audio,
    podcast_summary_records,
)


def _write_report(root: Path, name: str, published: str, show: str, episode: str) -> Path:
    report = root / "reports" / "show" / name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# Podcast Intelligence Brief\n\n"
        f"Podcast: {show}\n"
        f"Episode: {episode}\n"
        f"Published: {published}\n\n"
        "## Executive TL;DR\n"
        "- The durable signal is clear and useful.\n\n"
        "## High-leverage takeaway\n"
        "- Test the claim in a small real-world experiment.\n",
        encoding="utf-8",
    )
    return report


def test_podcast_summary_records_are_recent_stable_and_bounded(tmp_path):
    _write_report(tmp_path, "old.md", "2026-05-01", "Old Show", "Old episode")
    _write_report(tmp_path, "new.md", "2026-06-01", "New Show", "New episode")

    records = podcast_summary_records(tmp_path, limit=20)

    assert [record["title"] for record in records] == ["New episode", "Old episode"]
    assert all(len(str(record["id"])) == 8 for record in records)
    assert all(str(record["id"]).isalnum() for record in records)
    assert len(podcast_summary_records(tmp_path, limit=1)) == 1


def test_podcast_readout_is_curated_and_contains_no_internal_metadata(tmp_path):
    _write_report(tmp_path, "readout.md", "2026-06-01", "Signal Show", "A useful episode")
    record = podcast_summary_records(tmp_path, limit=1)[0]

    text = build_podcast_readout_text(record, tmp_path)

    assert "Signal Show" in text
    assert "A useful episode" in text
    assert "The durable signal is clear and useful." in text
    assert "http://" not in text
    assert "processed-episodes" not in text
    assert "Samantha" not in text


def test_podcast_summary_records_ignore_non_markdown_and_bad_reports(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "not-a-report.txt").write_text("no", encoding="utf-8")
    (report_dir / "bad.md").write_text("# no metadata", encoding="utf-8")

    assert podcast_summary_records(tmp_path, limit=20) == []


def test_neural_ogg_is_converted_to_sonos_safe_mp3(tmp_path):
    source = tmp_path / "readout.ogg"
    output = tmp_path / "readout.mp3"
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=0.2", str(source),
        ],
        check=True,
    )

    assert _prepare_sonos_audio(source) == output
    assert output.stat().st_size > 0
