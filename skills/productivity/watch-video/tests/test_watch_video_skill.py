#!/usr/bin/env python3
"""
Tests for watch-video skill.
Run with: scripts/run_tests.sh tests/skills/test_watch_video_skill.py -q
"""
from __future__ import annotations
import json
import tempfile
import pytest
import subprocess
import sys
from pathlib import Path

# Add scripts to path
SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))   # `from models import ...`
sys.path.insert(0, str(SKILL_ROOT))               # `from scripts.download import ...`
from models import (
    PipelineOptions, VideoAsset, TranscriptSegment, Frame,
    VisionResult, TimelineEntry, NotesResult,
)
from download import parse_timestamp as parse_timestamp_download
from extract_frames import parse_time_arg as parse_time_arg_frames


class TestModels:
    """Test data model serialization."""

    def test_transcript_segment_roundtrip(self):
        seg = TranscriptSegment(start=1.5, end=4.2, text="Hello world")
        data = seg.to_dict()
        restored = TranscriptSegment.from_dict(data)
        assert restored.start == 1.5
        assert restored.end == 4.2
        assert restored.text == "Hello world"

    def test_video_asset_roundtrip(self):
        asset = VideoAsset(
            path="/tmp/video.mp4",
            title="Test Video",
            duration=120.5,
            captions=[TranscriptSegment(0, 5, "Intro"), TranscriptSegment(5, 10, "Main")],
            metadata={"uploader": "test"},
        )
        data = asset.to_dict()
        restored = VideoAsset.from_dict(data)
        assert restored.path == "/tmp/video.mp4"
        assert restored.title == "Test Video"
        assert len(restored.captions) == 2

    def test_frame_roundtrip(self):
        frame = Frame(path="/tmp/frame_0001.jpg", timestamp=10.5, index=1, width=512, height=288)
        data = frame.to_dict()
        restored = Frame.from_dict(data)
        assert restored.timestamp == 10.5
        assert restored.index == 1

    def test_vision_result_roundtrip(self):
        vis = VisionResult(
            frame_index=0, timestamp=10.0, description="A person speaking",
            ocr_text="Hello", ui_elements=["button"], scene_type="talking_head", speaker_change=False
        )
        data = vis.to_dict()
        restored = VisionResult.from_dict(data)
        assert restored.frame_index == 0
        assert restored.ocr_text == "Hello"

    def test_timeline_entry_roundtrip(self):
        vis = VisionResult(frame_index=0, timestamp=10.0, description="Test")
        entry = TimelineEntry(timestamp=10.0, transcript="Hello", visual=vis)
        data = entry.to_dict()
        restored = TimelineEntry.from_dict(data)
        assert restored.timestamp == 10.0
        assert restored.transcript == "Hello"
        assert restored.visual is not None
        assert restored.visual.description == "Test"

    def test_parse_timestamp(self):
        assert parse_timestamp_download("00:01:30.500") == 90.5
        assert parse_timestamp_download("1:30.500") == 90.5
        assert parse_timestamp_download("90.5") == 90.5
        assert parse_timestamp_download("3600") == 3600.0


class TestDownload:
    """Test download.py script logic (mocked)."""

    def test_parse_vtt(self):
        from scripts.download import parse_vtt
        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world

00:00:05.000 --> 00:00:08.000
This is a test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            f.write(vtt_content)
            vtt_path = Path(f.name)

        try:
            segments = parse_vtt(vtt_path)
            assert len(segments) == 2
            assert segments[0].start == 1.0
            assert segments[0].end == 4.0
            assert segments[0].text == "Hello world"
            assert segments[1].text == "This is a test"
        finally:
            vtt_path.unlink()


class TestExtractFrames:
    """Test extract_frames.py logic."""

    def test_calculate_fps(self):
        from scripts.extract_frames import calculate_fps
        # Duration bands, with max_frames set high enough not to bind
        assert calculate_fps(100, 1000) == 1.0     # < 5 min
        assert calculate_fps(600, 1000) == 0.2     # 5-30 min
        assert calculate_fps(2000, 1000) == 0.1    # 30-60 min
        assert calculate_fps(4000, 1000) == 0.05   # > 60 min
        # max_frames caps the estimate: 100s at 1 fps would be 100 frames
        assert calculate_fps(100, 80) == pytest.approx(0.8)
        assert calculate_fps(600, 80) == pytest.approx(80 / 600)
        # Explicit fps is honoured but capped at 2
        assert calculate_fps(100, 1000, target_fps=5.0) == 2.0
        assert calculate_fps(100, 1000, target_fps=0.5) == 0.5
        # Very long video: capped by max_frames
        fps = calculate_fps(5000, 80)
        assert fps * 5000 <= 80
        # FPS override
        assert calculate_fps(100, 80, target_fps=0.5) == 0.5
        # Cap at 2 fps
        assert calculate_fps(100, 80, target_fps=5.0) == 2.0

    def test_parse_time_arg(self):
        from scripts.extract_frames import parse_time_arg
        assert parse_time_arg("30", 100) == 30.0
        assert parse_time_arg("00:01:30", 100) == 90.0
        assert parse_time_arg("1:30", 100) == 90.0
        assert parse_time_arg(None, 100) is None


class TestMergeTimeline:
    """Test merge_timeline.py logic."""

    def test_merge_basic(self):
        from scripts.merge_timeline import merge_timeline
        transcript = [
            TranscriptSegment(0, 5, "Intro"),
            TranscriptSegment(5, 10, "Main content"),
        ]
        vision = [
            VisionResult(0, 2.0, "Person speaking"),
            VisionResult(1, 7.0, "Code on screen", scene_type="screen_demo"),
        ]
        timeline = merge_timeline(transcript, vision)
        assert len(timeline) >= 4  # At least the unique timestamps
        # Check transcript alignment
        for entry in timeline:
            if entry.transcript:
                assert entry.transcript in ["Intro", "Main content"]


class TestPipelineIntegration:
    """Integration tests (require Ollama running)."""

    def test_pipeline_help(self):
        """Test pipeline script shows help."""
        script = Path(__file__).parent.parent / "scripts" / "pipeline.py"
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_setup_help(self):
        """Test setup script runs."""
        script = Path(__file__).parent.parent / "scripts" / "setup.py"
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0


if __name__ == "__main__":
    # Run tests manually if not using pytest
    import unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [TestModels, TestDownload, TestExtractFrames, TestMergeTimeline, TestPipelineIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)