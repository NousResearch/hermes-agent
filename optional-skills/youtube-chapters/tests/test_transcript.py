from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from utils.transcript import group_segments, normalize_segments


class TranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "transcript_basic.json"
        self.segments = json.loads(fixture.read_text(encoding="utf-8"))

    def test_normalizes_duration_and_text(self) -> None:
        normalized = normalize_segments(self.segments)
        self.assertEqual(normalized[0]["start"], 0.0)
        self.assertEqual(normalized[0]["end"], 20.0)
        self.assertEqual(normalized[0]["text"], "Welcome and introduction to the project.")
        self.assertEqual(normalized[-1]["end"], 230.0)

    def test_groups_chronologically(self) -> None:
        chunks = group_segments(self.segments, target_seconds=100)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[-1]["end"], 230.0)
        self.assertTrue(all(chunks[index]["start"] < chunks[index + 1]["start"] for index in range(len(chunks) - 1)))

    def test_rejects_untimestamped_segment(self) -> None:
        with self.assertRaises(ValueError):
            normalize_segments([{"start": 0, "text": "No duration"}])


if __name__ == "__main__":
    unittest.main()
