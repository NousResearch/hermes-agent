"""Unit tests for core functions and analyzers."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "optional-skills", "agent-architecture", "win-gui-test"))

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
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = capture(os.path.join(td, "file.png"), fallback=True)
        self.assertIsNone(result)

class TestScreenshotDefaults(unittest.TestCase):
    """Regression: screenshot default output_dir must not raise NameError (os)."""

    def test_default_output_dir_uses_os_path(self):
        """Verify core.py has os imported so default path resolution works."""
        import ast, pathlib
        core_src = (pathlib.Path(__file__).resolve().parent.parent.parent / "optional-skills" / "agent-architecture" / "win-gui-test" / "scripts" / "core.py")
        tree = ast.parse(core_src.read_text())
        # Verify 'import os' exists at module level
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
        os_imported = any('os' in [alias.name for alias in imp.names] for imp in imports)
        self.assertTrue(os_imported, "core.py must have 'import os' at module level")

        # Also verify _get_config helper exists
        funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        self.assertIn('_get_config', funcs, "core.py must have _get_config helper")

        # Verify screenshot() has config parameter
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'screenshot':
                args = [a.arg for a in node.args.args]
                self.assertIn('config', args, "screenshot() must accept config= parameter")
                break
        else:
            self.fail("screenshot() function not found")

        # Verify click() has config parameter
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'click':
                args = [a.arg for a in node.args.args]
                self.assertIn('config', args, "click() must accept config= parameter")
                break
        else:
            self.fail("click() function not found")


class TestOsImport(unittest.TestCase):
    """Regression: core.py must import os (used for screenshot default path)."""

    def test_os_imported_in_core(self):
        """Verify 'import os' exists in core.py source without importing pywinauto."""
        import ast, pathlib
        core_src = (pathlib.Path(__file__).resolve().parent.parent.parent / "optional-skills" / "agent-architecture" / "win-gui-test" / "scripts" / "core.py")
        tree = ast.parse(core_src.read_text())
        # Collect all import names at module level
        imported = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        self.assertIn('os', imported, "core.py must have 'import os' at module level")


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
