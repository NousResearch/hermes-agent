#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for word_metrics.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from word_metrics import section_metrics, word_count

DOC = """Intro line before any heading.

# Title

Two words here indeed.

## Section A

Alpha beta gamma.

```
# not a heading
fenced words ignored as headings
```

## Section B

Delta epsilon.
"""


class WordMetricsTest(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(word_count("one two  three\nfour"), 4)
        self.assertEqual(word_count(""), 0)

    def test_sections_split_on_headings(self):
        sections = section_metrics(DOC)
        headings = [s["heading"] for s in sections]
        self.assertEqual(headings, ["(preamble)", "Title", "Section A", "Section B"])

    def test_fenced_heading_not_a_section(self):
        sections = section_metrics(DOC)
        self.assertNotIn("not a heading", [s["heading"] for s in sections])

    def test_section_words_counted(self):
        sections = {s["heading"]: s["words"] for s in section_metrics(DOC)}
        self.assertEqual(sections["Section B"], 2)
        # Section A body includes the fenced block's tokens
        self.assertGreater(sections["Section A"], 3)

    def test_empty_preamble_dropped(self):
        sections = section_metrics("# Only\n\nwords here\n")
        self.assertEqual([s["heading"] for s in sections], ["Only"])


if __name__ == "__main__":
    unittest.main()
