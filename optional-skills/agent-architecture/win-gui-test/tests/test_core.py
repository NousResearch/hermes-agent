"""Unit tests for core functions and analyzers."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils.config import load_config, DEFAULT_CONFIG
from scripts.utils.screenshot import capture
from scripts.analyzers.size_analyzer import collect_sizes, detect_inconsistencies
from scripts.analyzers.color_analyzer import dominant_colors, detect_edge_colors


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(None)
        self.assertIn("screenshot_dir", cfg)
        self.assertIn("retry", cfg)
        self.assertEqual(cfg["retry"]["count"], 3)
        self.assertEqual(cfg["retry"]["delay"], 1.0)

    def test_json_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"screenshot_dir": "/tmp/test_ss", "retry": {"count": 5}}, f)
            fname = f.name
        try:
            cfg = load_config(fname)
            self.assertEqual(cfg["screenshot_dir"], "/tmp/test_ss")
            self.assertEqual(cfg["retry"]["count"], 5)
            self.assertEqual(cfg["retry"]["delay"], 1.0)  # inherited from default
        finally:
            os.unlink(fname)

    def test_env_override(self):
        with patch.dict(os.environ, {"WG_RETRY_COUNT": "7"}):
            cfg = load_config(None)
            self.assertEqual(cfg["retry"]["count"], 7)


class TestScreenshot(unittest.TestCase):
    @patch("scripts.utils.screenshot._screenshot_mss", return_value=False)
    @patch("scripts.utils.screenshot._screenshot_pil", return_value=False)
    def test_capture_failure_returns_none(self, mock_pil, mock_mss):
        """With both mss and PIL mocked to fail, capture should return None."""
        result = capture("/nonexistent/dir/file.png", fallback=True)
        self.assertIsNone(result)


class TestSizeAnalyzer(unittest.TestCase):
    def test_collect_sizes_empty(self):
        stats = collect_sizes([])
        self.assertEqual(stats["count"], 0)

    def test_collect_sizes(self):
        elements = [
            {"name": "btn1", "rect": "(0,0) 100x30"},
            {"name": "btn2", "rect": "(0,0) 120x30"},
            {"name": "btn3", "rect": "(0,0) 110x32"},
        ]
        stats = collect_sizes(elements)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["width"]["min"], 100)
        self.assertEqual(stats["width"]["max"], 120)
        self.assertEqual(stats["height"]["min"], 30)
        self.assertEqual(stats["height"]["all_same"], False)

    def test_detect_inconsistencies(self):
        elements = [
            {"name": "a", "rect": "(0,0) 10x30"},
            {"name": "b", "rect": "(0,0) 10x30"},
            {"name": "c", "rect": "(0,0) 10x60"},  # outlier
        ]
        out = detect_inconsistencies(elements)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "c")


class TestColorAnalyzer(unittest.TestCase):
    def test_dominant_colors(self):
        import numpy as np
        # 100x100 pure red image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # BGR = blue in BGR
        colors = dominant_colors(img, n=3)
        self.assertTrue(len(colors) >= 1)

    def test_detect_edge_colors(self):
        import numpy as np
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        edges = detect_edge_colors(img)
        self.assertIn("top", edges)
        self.assertIn("left", edges)
        self.assertEqual(edges["top"], [200, 200, 200])


if __name__ == "__main__":
    unittest.main()
