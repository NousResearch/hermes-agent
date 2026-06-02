"""Tests for scripts/code-scan/detect_entrypoints.py.

Tests follow strict TDD: written first, then implementation follows.
Covers all supported languages (Python, JS/TS, Go, Rust, Shell)
plus negative tests for false positives.
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure scripts/code-scan is on sys.path for sibling imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts" / "code-scan"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from detect_entrypoints import (
    detect_entrypoints,
    main as detect_main,
    _detect_python_entrypoints,
    _detect_js_ts_entrypoints,
    _detect_go_entrypoints,
    _detect_rust_entrypoints,
    _detect_shell_entrypoints,
    _read_file_content,
)


# ── Fixtures directory ──────────────────────────────────────────────────

FIXTURES = _PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "entrypoints"


def _load_scan(name: str) -> str:
    """Return absolute path to a fixture scan.json."""
    return str(FIXTURES / name / "scan.json")


def _load_scan_data(name: str) -> dict:
    """Load and return the fixture scan.json as a dict."""
    with open(_load_scan(name), "r") as f:
        return json.load(f)


# ── Output schema tests ─────────────────────────────────────────────────

class TestOutputSchema:
    """Verify the output matches the required JSON shape."""

    def test_output_has_schema_version(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        assert "schema_version" in result
        assert result["schema_version"] == "1.0.0"

    def test_output_has_entrypoints_list(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        assert "entrypoints" in result
        assert isinstance(result["entrypoints"], list)

    def test_output_has_totals(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        assert "totals" in result
        assert "entrypoints_found" in result["totals"]
        assert "by_type" in result["totals"]

    def test_entrypoint_has_required_fields(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        assert len(result["entrypoints"]) > 0
        ep = result["entrypoints"][0]
        for key in ("file", "language", "type", "signals", "confidence"):
            assert key in ep, f"Missing key: {key}"

    def test_confidence_is_float(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        ep = result["entrypoints"][0]
        assert isinstance(ep["confidence"], float)
        assert 0.0 <= ep["confidence"] <= 1.0

    def test_signals_is_list(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        ep = result["entrypoints"][0]
        assert isinstance(ep["signals"], list)


# ── Python CLI detection ────────────────────────────────────────────────

class TestPythonCLI:
    """Detect Python CLI entrypoints via __name__ == '__main__' patterns."""

    def test_detects_if_name_main(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        # The cli.py file should be detected
        files = [e["file"] for e in eps]
        assert "src/cli.py" in files

    def test_detects_def_main(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/cli.py"]
        assert len(ep_list) >= 1
        ep = ep_list[0]
        assert "def main" in ep["signals"] or "if __name__" in " ".join(ep["signals"])

    def test_detects_argparse(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/cli.py"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "argparse" in signals_str

    def test_helper_utils_not_entrypoint(self):
        """utils.py in the python_cli fixture has no entrypoint patterns."""
        result = detect_entrypoints(_load_scan("python_cli"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/utils.py" not in files

    def test_totals_correct(self):
        result = detect_entrypoints(_load_scan("python_cli"))
        assert result["totals"]["entrypoints_found"] >= 1
        assert result["totals"]["by_type"].get("python_cli", 0) >= 1


# ── Python FastAPI detection ────────────────────────────────────────────

class TestPythonFastAPI:
    """Detect FastAPI + Uvicorn app startup."""

    def test_detects_fastapi_app(self):
        result = detect_entrypoints(_load_scan("python_fastapi"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/app.py" in files

    def test_detects_uvicorn_signal(self):
        result = detect_entrypoints(_load_scan("python_fastapi"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/app.py"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "uvicorn" in signals_str.lower()


# ── Python Typer detection ──────────────────────────────────────────────

class TestPythonTyper:
    """Detect typer.Typer apps."""

    def test_detects_typer(self):
        result = detect_entrypoints(_load_scan("python_typer"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/main.py" in files

    def test_typer_signal_present(self):
        result = detect_entrypoints(_load_scan("python_typer"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/main.py"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "typer" in signals_str.lower()


# ── Python Click detection ──────────────────────────────────────────────

class TestPythonClick:
    """Detect click.command apps."""

    def test_detects_click(self):
        result = detect_entrypoints(_load_scan("python_click"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/cli.py" in files

    def test_click_signal_present(self):
        result = detect_entrypoints(_load_scan("python_click"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/cli.py"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "click" in signals_str.lower()


# ── Python __main__.py detection ────────────────────────────────────────

class TestPythonMainPy:
    """Detect __main__.py as entrypoint."""

    def test_detects_main_py(self):
        result = detect_entrypoints(_load_scan("python_main_py"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/__main__.py" in files

    def test_runner_not_entrypoint(self):
        result = detect_entrypoints(_load_scan("python_main_py"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/subpkg/runner.py" not in files


# ── JS/Node detection ───────────────────────────────────────────────────

class TestJSNode:
    """Detect JS/Node entrypoints."""

    def test_detects_index_js(self):
        result = detect_entrypoints(_load_scan("js_node"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "app/index.js" in files

    def test_detects_app_listen(self):
        result = detect_entrypoints(_load_scan("js_node"))
        ep_list = [e for e in result["entrypoints"] if "index.js" in e["file"]]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "app.listen" in signals_str

    def test_utils_not_entrypoint(self):
        result = detect_entrypoints(_load_scan("js_node"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/utils.js" not in files


# ── TS/Express detection ────────────────────────────────────────────────

class TestTSExpress:
    """Detect TypeScript/Express entrypoints."""

    def test_detects_main_ts(self):
        result = detect_entrypoints(_load_scan("ts_express"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/main.ts" in files

    def test_detects_listen_signal(self):
        result = detect_entrypoints(_load_scan("ts_express"))
        ep_list = [e for e in result["entrypoints"] if "main.ts" in e["file"]]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "listen" in signals_str


# ── Go detection ────────────────────────────────────────────────────────

class TestGoEntrypoints:
    """Detect Go entrypoints (package main + func main)."""

    def test_detects_package_main_and_func_main(self):
        result = detect_entrypoints(_load_scan("go_app"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "main.go" in files

    def test_go_signals_present(self):
        result = detect_entrypoints(_load_scan("go_app"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "main.go"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "package main" in signals_str
        assert "func main" in signals_str

    def test_secondary_main_file_detected(self):
        """cmd/tool.go also has package main + func main."""
        result = detect_entrypoints(_load_scan("go_app"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "cmd/tool.go" in files


# ── Rust detection ──────────────────────────────────────────────────────

class TestRustEntrypoints:
    """Detect Rust entrypoints (fn main in src/main.rs or src/bin/*.rs)."""

    def test_detects_main_rs(self):
        result = detect_entrypoints(_load_scan("rust_app"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "src/main.rs" in files

    def test_detects_bin_rust_file(self):
        result = detect_entrypoints(_load_scan("rust_app"))
        eps = result["entrypoints"]
        files = [e["file"] for e in eps]
        assert "src/bin/extra_tool.rs" in files

    def test_lib_rs_not_entrypoint(self):
        """src/lib.rs has fn helper but no fn main."""
        result = detect_entrypoints(_load_scan("rust_app"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/lib.rs" not in files

    def test_rust_signal_present(self):
        result = detect_entrypoints(_load_scan("rust_app"))
        ep_list = [e for e in result["entrypoints"] if e["file"] == "src/main.rs"]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "fn main" in signals_str


# ── Shell script detection ──────────────────────────────────────────────

class TestShellEntrypoints:
    """Detect shell scripts with shebang in root, bin/, or scripts/."""

    def test_detects_bin_shell_script(self):
        result = detect_entrypoints(_load_scan("shell_scripts"))
        eps = result["entrypoints"]
        assert len(eps) >= 1
        files = [e["file"] for e in eps]
        assert "bin/deploy.sh" in files

    def test_detects_shebang(self):
        result = detect_entrypoints(_load_scan("shell_scripts"))
        ep_list = [e for e in result["entrypoints"] if "deploy.sh" in e["file"]]
        assert len(ep_list) >= 1
        signals_str = " ".join(ep_list[0]["signals"])
        assert "shebang" in signals_str.lower()

    def test_scripts_dir_python_entrypoint(self):
        """Python script in scripts/ with __name__ == '__main__'."""
        result = detect_entrypoints(_load_scan("shell_scripts"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "scripts/build.py" in files


# ── Negative tests ──────────────────────────────────────────────────────

class TestNegativeCases:
    """False-positive prevention: helpers, docs, non-executable files."""

    def test_main_helper_not_entrypoint(self):
        """A function named main_helper should not be an entrypoint."""
        result = detect_entrypoints(_load_scan("negative_tests"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/helpers.py" not in files

    def test_docs_not_entrypoint(self):
        """Doc files mentioning 'main' should not be entrypoints."""
        result = detect_entrypoints(_load_scan("negative_tests"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "docs/README.md" not in files
        assert "docs/ARCHITECTURE.md" not in files

    def test_config_loader_not_entrypoint(self):
        """A function named main_util is not an entrypoint."""
        result = detect_entrypoints(_load_scan("negative_tests"))
        files = [e["file"] for e in result["entrypoints"]]
        assert "src/config_loader.py" not in files

    def test_empty_project_no_entrypoints(self):
        """A scan with no files should produce empty entrypoints."""
        result = detect_entrypoints(_load_scan("empty_project"))
        assert result["entrypoints"] == []
        assert result["totals"]["entrypoints_found"] == 0


# ── _read_file_content tests ────────────────────────────────────────────

class TestFileReading:
    """Test _read_file_content helper."""

    def test_read_existing_file(self):
        content = _read_file_content(str(FIXTURES / "python_cli/src/cli.py"))
        assert "if __name__" in content
        assert "def main" in content

    def test_read_nonexistent_file(self):
        content = _read_file_content("/nonexistent/file.py")
        assert content is None


# ── CLI integration test ────────────────────────────────────────────────

class TestCLI:
    """Test the CLI: python detect_entrypoints.py <scan.json>."""

    def test_cli_exits_zero(self, capfd):
        """CLI should exit with code 0 on valid input."""
        test_scan = _load_scan("python_cli")
        sys.argv = ["detect_entrypoints.py", test_scan]
        exit_code = detect_main()
        assert exit_code == 0
        captured = capfd.readouterr()
        output = json.loads(captured.out)
        assert output["schema_version"] == "1.0.0"

    def test_cli_errors_on_missing_file(self, capfd):
        """CLI should exit non-zero when scan.json doesn't exist."""
        sys.argv = ["detect_entrypoints.py", "/nonexistent/scan.json"]
        exit_code = detect_main()
        assert exit_code != 0

    def test_cli_errors_on_no_args(self, capfd):
        """CLI should exit non-zero when no args given."""
        sys.argv = ["detect_entrypoints.py"]
        exit_code = detect_main()
        assert exit_code != 0
