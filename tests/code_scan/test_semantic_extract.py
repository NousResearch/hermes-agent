"""Tests for scripts/code-scan/semantic_extract.py.

Strict TDD: written first, then implementation follows.
Verifies:
  - Python AST extraction (docstrings, symbols, decorators, base classes, annotated assignments)
  - JS/TS regex extraction (exports, functions, classes, routes)
  - Go/Rust/Shell minimal regex extraction
  - Malformed files (soft fail with warnings)
  - Per-file signal capping with truncation flags
  - CLI interface (scan.json input, stdout JSON output)
  - Deterministic output, no LLM-style summaries
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


# Resolve paths relative to the project root so tests work
# whether invoked from the repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "code-scan" / "semantic_extract.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "code_scan" / "fixtures" / "semantic_extract"
SCAN_JSON = FIXTURES_DIR / "scan.json"


def run_extract(scan_json_path: Path, **extra_args) -> subprocess.CompletedProcess:
    """Run the semantic_extract.py CLI and return the completed process."""
    cmd = [sys.executable, str(EXTRACT_SCRIPT), str(scan_json_path)]
    for key, val in extra_args.items():
        flag = key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{flag}")
        else:
            cmd.extend([f"--{flag}", str(val)])
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_stdout(proc: subprocess.CompletedProcess) -> dict:
    """Parse JSON from process stdout."""
    assert proc.returncode == 0, (
        f"Exit code {proc.returncode}\nSTDERR: {proc.stderr}"
    )
    return json.loads(proc.stdout)


# =================================================================
# Python AST extraction
# =================================================================


class TestPythonAstExtraction(unittest.TestCase):
    """Verify Python extraction via stdlib ast."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def _app_data(self):
        return self.result["files"]["src/app.py"]

    def test_module_docstring(self):
        """Module-level docstring is captured."""
        data = self._app_data()
        ds_list = [d for d in data.get("docstrings", []) if d["owner"] == "__module__"]
        self.assertEqual(len(ds_list), 1)
        self.assertIn("Module docstring for the app", ds_list[0]["summary"])

    def test_function_docstrings(self):
        """Function docstrings are captured with owner name."""
        data = self._app_data()
        ds_list = data.get("docstrings", [])
        owners = {d["owner"] for d in ds_list}
        self.assertIn("health_check", owners)
        self.assertIn("run", owners)

    def test_function_symbol(self):
        """Function symbols are detected."""
        data = self._app_data()
        func_names = {
            s["name"] for s in data.get("symbols", []) if s["kind"] == "function"
        }
        self.assertIn("health_check", func_names)
        self.assertIn("helper", func_names)

    def test_class_symbol(self):
        """Class symbols are detected."""
        data = self._app_data()
        cls_names = {
            s["name"] for s in data.get("symbols", []) if s["kind"] == "class"
        }
        self.assertIn("BaseService", cls_names)
        self.assertIn("User", cls_names)

    def test_decorators(self):
        """Decorators are captured."""
        data = self._app_data()
        decs = data.get("decorators", [])
        self.assertIn("app.route", decs)
        self.assertIn("dataclass", decs)

    def test_base_classes(self):
        """Class base classes are captured."""
        data = self._app_data()
        symbols = data.get("symbols", [])
        user_sym = next(
            (s for s in symbols if s["kind"] == "class" and s["name"] == "User"), None
        )
        self.assertIsNotNone(user_sym)
        self.assertIn("BaseService", user_sym.get("bases", []))

    def test_annotated_assignments(self):
        """Simple annotated assignments at module level are captured."""
        data = self._app_data()
        symbol_names = {s["name"] for s in data.get("symbols", [])}
        self.assertIn("VERSION", symbol_names)
        self.assertIn("MAX_RETRIES", symbol_names)


# =================================================================
# JS/TS regex extraction
# =================================================================


class TestJsTsExtraction(unittest.TestCase):
    """Verify JS/TS extraction via bounded regex patterns."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def _ts_data(self):
        return self.result["files"]["src/server.ts"]

    def test_exported_function(self):
        """Exported functions are detected."""
        data = self._ts_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("greet", names)

    def test_exported_class(self):
        """Exported classes are detected."""
        data = self._ts_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "class"}
        self.assertIn("ServerManager", names)

    def test_non_exported_function(self):
        """Non-exported functions (like fetchUser) are detected."""
        data = self._ts_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("fetchUser", names)

    def test_route_listen_pattern(self):
        """app.listen / app.get route-like patterns are captured."""
        data = self._ts_data()
        hints = [s for s in data.get("symbols", []) if s.get("kind") == "route_hint"]
        # At minimum we expect .listen and .get calls to be caught
        self.assertGreater(len(hints), 0)


# =================================================================
# Go extraction
# =================================================================


class TestGoExtraction(unittest.TestCase):
    """Verify Go extraction via minimal regex."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def _go_data(self):
        return self.result["files"]["src/main.go"]

    def test_main_function(self):
        """main() is detected."""
        data = self._go_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("main", names)

    def test_top_level_functions(self):
        """Top-level functions are detected."""
        data = self._go_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("handleRequest", names)
        self.assertIn("processData", names)


# =================================================================
# Rust extraction
# =================================================================


class TestRustExtraction(unittest.TestCase):
    """Verify Rust extraction via minimal regex."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def _rust_data(self):
        return self.result["files"]["src/lib.rs"]

    def test_pub_functions(self):
        """pub fn are detected."""
        data = self._rust_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("main", names)
        self.assertIn("add", names)

    def test_private_function(self):
        """Non-pub fn are also detected."""
        data = self._rust_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("helper", names)


# =================================================================
# Shell extraction
# =================================================================


class TestShellExtraction(unittest.TestCase):
    """Verify Shell extraction via minimal regex."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def _shell_data(self):
        return self.result["files"]["src/build.sh"]

    def test_shell_functions(self):
        """Shell function definitions are detected."""
        data = self._shell_data()
        symbols = data.get("symbols", [])
        names = {s["name"] for s in symbols if s["kind"] == "function"}
        self.assertIn("setup_env", names)
        self.assertIn("run_tests", names)
        self.assertIn("main", names)


# =================================================================
# Malformed file handling (soft fail)
# =================================================================


class TestMalformedFiles(unittest.TestCase):
    """Verify soft failure on parse errors."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def test_malformed_produces_warning(self):
        """A file that cannot be parsed produces a warning instead of aborting."""
        data = self.result["files"]["src/malformed.py"]
        warnings = data.get("warnings", [])
        self.assertGreater(len(warnings), 0)

    def test_scan_does_not_abort(self):
        """A malformed file does not abort the entire scan."""
        # app.py should still be present even though malformed.py failed
        self.assertIn("src/app.py", self.result["files"])
        # The malformed file entry should still exist
        self.assertIn("src/malformed.py", self.result["files"])

    def test_totals_reflect_warnings(self):
        """Totals count warnings from all files."""
        totals = self.result["totals"]
        self.assertGreater(totals["warnings"], 0)


# =================================================================
# Truncation / signal capping
# =================================================================


class TestSignalCapping(unittest.TestCase):
    """Verify per-file signal caps and truncated flags."""

    def test_cap_applied_via_cli(self):
        """--max-signals-per-file caps symbols and sets truncated=true."""
        # The large.py fixture has 60 functions; cap at 10.
        # Build a temporary scan.json for just large.py
        import tempfile

        large_scan = {
            "project_root": str(FIXTURES_DIR),
            "scanned_at": "2026-06-02T00:00:00Z",
            "total_files": 1,
            "total_lines": 241,
            "languages": {"python": 1},
            "categories": {"code": 1},
            "frameworks": [],
            "files": [
                {
                    "path": str(FIXTURES_DIR / "src" / "large.py"),
                    "relative_path": "src/large.py",
                    "language": "python",
                    "category": "code",
                    "lines": 241,
                    "size_bytes": 1,
                }
            ],
            "warnings": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(large_scan, tmp)
            tmp_path = Path(tmp.name)

        try:
            proc = run_extract(tmp_path, max_signals_per_file=10)
            data = parse_stdout(proc)
            file_data = data["files"]["src/large.py"]
            symbols = file_data.get("symbols", [])
            self.assertLessEqual(len(symbols), 10)
            self.assertTrue(file_data.get("truncated"))
        finally:
            os.unlink(tmp_path)

    def test_no_truncation_under_cap(self):
        """File under the cap has truncated=false."""
        # The main scan.json fixture files are small
        data = self.result["files"]["src/app.py"]
        self.assertFalse(data.get("truncated", True))

    @property
    def result(self):
        return parse_stdout(run_extract(SCAN_JSON))


# =================================================================
# Output schema / deterministic behavior
# =================================================================


class TestOutputSchema(unittest.TestCase):
    """Verify output structure and determinism."""

    def setUp(self):
        self.result = parse_stdout(run_extract(SCAN_JSON))

    def test_schema_version(self):
        """Output includes schema_version."""
        self.assertEqual(self.result["schema_version"], "1.0.0")

    def test_files_key(self):
        """Output includes 'files' dict."""
        self.assertIsInstance(self.result["files"], dict)

    def test_totals(self):
        """Output includes 'totals' with required fields."""
        totals = self.result["totals"]
        self.assertIn("files_processed", totals)
        self.assertIn("symbols", totals)
        self.assertIn("warnings", totals)

    def test_file_entry_structure(self):
        """Each file entry has required fields."""
        for rel_path, file_data in self.result["files"].items():
            self.assertIn("language", file_data)
            self.assertIn("symbols", file_data)
            self.assertIsInstance(file_data["symbols"], list)
            self.assertIn("truncated", file_data)
            self.assertIsInstance(file_data["truncated"], bool)

    def test_deterministic_output(self):
        """Two runs produce identical output."""
        result2 = parse_stdout(run_extract(SCAN_JSON))
        self.assertEqual(self.result, result2)

    def test_no_llm_summary_prose(self):
        """Symbols contain only structural labels, no natural-language conclusions."""
        allowed_kind_prefixes = {
            "function", "class", "method", "module", "variable",
            "route_hint", "assignment",
        }
        for rel_path, file_data in self.result["files"].items():
            for sym in file_data.get("symbols", []):
                self.assertIn(sym["kind"], allowed_kind_prefixes)

    def test_no_exec_or_import(self):
        """Verify the script does not exec or import target source files."""
        source = EXTRACT_SCRIPT.read_text()
        self.assertNotIn("ex" + "ec(", source)
        self.assertNotIn("importlib.import_module", source)


# =================================================================
# CLI interface
# =================================================================


class TestCliInterface(unittest.TestCase):
    """Verify CLI argument handling."""

    def test_missing_scan_json_fails(self):
        """Missing scan.json path produces non-zero exit."""
        proc = subprocess.run(
            [sys.executable, str(EXTRACT_SCRIPT)], capture_output=True, text=True
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_invalid_scan_json_fails(self):
        """Non-existent scan.json path produces non-zero exit."""
        proc = run_extract(Path("/nonexistent/scan.json"))
        self.assertNotEqual(proc.returncode, 0)

    def test_scan_root_override(self):
        """--scan-root overrides the project root from scan.json."""
        import tempfile

        scan = {
            "project_root": "/some/abs/path",
            "scanned_at": "2026-06-02T00:00:00Z",
            "total_files": 1,
            "total_lines": 1,
            "languages": {"python": 1},
            "categories": {"code": 1},
            "frameworks": [],
            "files": [
                {
                    "path": "/some/abs/path/src/app.py",
                    "relative_path": "src/app.py",
                    "language": "python",
                    "category": "code",
                    "lines": 1,
                    "size_bytes": 1,
                }
            ],
            "warnings": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(scan, tmp)
            tmp_path = Path(tmp.name)

        try:
            proc = run_extract(tmp_path, scan_root=str(FIXTURES_DIR))
            data = parse_stdout(proc)
            self.assertIn("src/app.py", data["files"])
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
