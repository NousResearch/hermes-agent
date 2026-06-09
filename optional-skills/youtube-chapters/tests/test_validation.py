from __future__ import annotations

import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from utils.validation import (
    Chapter,
    format_chapters,
    format_timestamp,
    parse_chapter_lines,
    parse_timestamp,
    validate_chapters,
)


class TimestampTests(unittest.TestCase):
    def test_parses_and_formats_timestamp_forms(self) -> None:
        self.assertEqual(parse_timestamp("01:42"), 102)
        self.assertEqual(parse_timestamp("75:00"), 4500)
        self.assertEqual(parse_timestamp("01:15:00"), 4500)
        self.assertEqual(format_timestamp(102), "01:42")
        self.assertEqual(format_timestamp(4500), "01:15:00")

    def test_rejects_invalid_timestamp(self) -> None:
        for value in ["1", "01:60", "01:75:00", "-01:00"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_timestamp(value)


class ChapterValidationTests(unittest.TestCase):
    def test_valid_chapters_are_paste_ready(self) -> None:
        raw = "00:00 Introduction\n01:42 Setup\n04:18 Architecture"
        chapters = parse_chapter_lines(raw)
        self.assertEqual(validate_chapters(chapters, duration_seconds=300), [])
        self.assertEqual(format_chapters(chapters), raw)

    def test_invalid_output_produces_repair_errors(self) -> None:
        chapters = [
            Chapter(5, "Introduction"),
            Chapter(5, "Part 1"),
            Chapter(400, "Conclusion"),
        ]
        errors = validate_chapters(chapters, duration_seconds=300)
        self.assertIn("First chapter must start at 00:00", errors)
        self.assertIn("Chapter 2 has a generic title", errors)
        self.assertIn("Chapter 2 timestamp must be strictly increasing", errors)
        self.assertIn("Chapter 3 exceeds the known video duration", errors)

    def test_minimum_can_be_disabled_for_short_transcript(self) -> None:
        chapters = [Chapter(0, "Only supported topic")]
        self.assertEqual(validate_chapters(chapters, require_minimum=False), [])

    def test_empty_output_is_invalid(self) -> None:
        self.assertEqual(validate_chapters([]), ["Chapter list is empty"])


if __name__ == "__main__":
    unittest.main()
