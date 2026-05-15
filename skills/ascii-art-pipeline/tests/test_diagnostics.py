import json
import tempfile
import unittest
from pathlib import Path

from ascii_pipeline.diagnostics import analyze_lines, diagnose_path, parse_eikon_text


class DiagnosticsTests(unittest.TestCase):
    def test_analyze_lines_basic_metrics(self):
        metrics = analyze_lines([
            "@@@",
            "@ @",
            "@@@",
        ], expected_width=3, expected_height=3)
        self.assertEqual(metrics.width, 3)
        self.assertEqual(metrics.height, 3)
        self.assertEqual(metrics.unique_glyphs, 1)
        self.assertGreater(metrics.heavy_ratio, 0.9)
        self.assertTrue(metrics.dimensions_ok)

    def test_parse_legacy_eikon(self):
        text = "\n".join([
            json.dumps({"eikon": 1, "version": 1, "name": "demo", "width": 3, "height": 2}),
            json.dumps({"state": "idle", "fps": 12}),
            json.dumps({"data": "@@@\\n@ @"}),
            json.dumps({"data": "###\\n# #"}),
        ])
        frames = parse_eikon_text(text)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].state, "idle")
        self.assertEqual(frames[0].lines[0], "@@@")

    def test_diagnose_path_for_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("@@@\n@ @\n@@@\n", encoding="utf-8")
            summary = diagnose_path(path, expected_width=3, expected_height=3)
        self.assertEqual(summary["kind"], "text")
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["dimensions"]["widths"], [3])
        self.assertEqual(summary["verdict"], "high-contrast")


if __name__ == "__main__":
    unittest.main()
