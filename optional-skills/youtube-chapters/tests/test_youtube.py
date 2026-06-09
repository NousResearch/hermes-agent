from __future__ import annotations

import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from utils.youtube import extract_video_id, normalize_video_id


class YouTubeIdTests(unittest.TestCase):
    def test_extracts_supported_url_forms(self) -> None:
        expected = "dQw4w9WgXcQ"
        inputs = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/live/dQw4w9WgXcQ",
            "Create chapters for https://youtube.com/watch?v=dQw4w9WgXcQ please",
        ]
        for value in inputs:
            with self.subTest(value=value):
                self.assertEqual(extract_video_id(value), expected)

    def test_accepts_raw_video_id(self) -> None:
        self.assertEqual(normalize_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_rejects_invalid_input(self) -> None:
        invalid_inputs = [
            "https://example.com/video",
            "too-short",
            "invalid-url",
        ]
        for value in invalid_inputs:
            with self.subTest(value=value), self.assertRaises(ValueError):
                extract_video_id(value)


if __name__ == "__main__":
    unittest.main()
