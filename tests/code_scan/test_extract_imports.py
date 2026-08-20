"""Tests for scripts/code-scan/extract_imports.py — Phase 2 D1."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow importing the module under test
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "code-scan"
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_imports import (
    build_import_map,
    extract_go_imports,
    extract_imports_for_file,
    extract_js_ts_imports,
    extract_python_imports,
    extract_rust_imports,
    extract_shell_imports,
    iter_scanned_files,
    load_scan_output,
    main,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "imports"


# ── load_scan_output ───────────────────────────────────────────────

def test_load_scan_output_valid(tmp_path: Path) -> None:
    scan = {"project_root": "/foo", "total_files": 3, "files": []}
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(scan))
    result = load_scan_output(str(p))
    assert result == scan


def test_load_scan_output_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_scan_output("/no/such/file.json")


def test_load_scan_output_invalid_schema(tmp_path: Path) -> None:
    # Missing 'files' key — invalid schema
    p = tmp_path / "bad.json"
    p.write_text('{"project_root": "/x"}')
    with pytest.raises(ValueError, match="Invalid scan schema"):
        load_scan_output(str(p))


def test_load_scan_output_empty_scan(tmp_path: Path) -> None:
    scan = {"project_root": "/empty", "total_files": 0, "files": []}
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(scan))
    result = load_scan_output(str(p))
    assert result["total_files"] == 0
    assert result["files"] == []


# ── iter_scanned_files ─────────────────────────────────────────────

def test_iter_scanned_files_yields_pairs() -> None:
    scan = {
        "files": [
            {"path": "src/main.py", "language": "python"},
            {"path": "src/index.ts", "language": "typescript"},
        ]
    }
    pairs = list(iter_scanned_files(scan))
    assert pairs == [
        ("src/main.py", "python"),
        ("src/index.ts", "typescript"),
    ]


def test_iter_scanned_files_empty() -> None:
    scan = {"files": []}
    assert list(iter_scanned_files(scan)) == []


# ── extract_python_imports ─────────────────────────────────────────

def test_extract_python_imports_simple() -> None:
    src = "import os\nimport sys\n"
    assert extract_python_imports(src) == ["os", "sys"]


def test_extract_python_imports_from() -> None:
    src = "from pathlib import Path\n"
    assert extract_python_imports(src) == ["pathlib"]


def test_extract_python_imports_dotted() -> None:
    src = "import os.path\n"
    assert extract_python_imports(src) == ["os"]


def test_extract_python_imports_comments_ignored() -> None:
    src = "# import fake\nimport os\n"
    assert extract_python_imports(src) == ["os"]


def test_extract_python_imports_fixture() -> None:
    src = (FIXTURES / "python_sample.py").read_text()
    result = extract_python_imports(src)
    assert result == ["os", "sys", "json", "pathlib"]


# ── extract_js_ts_imports ──────────────────────────────────────────

def test_extract_js_ts_imports_default() -> None:
    src = 'import React from "react";\n'
    assert extract_js_ts_imports(src) == ["react"]


def test_extract_js_ts_imports_named() -> None:
    src = 'import { helper } from "./utils";\n'
    assert extract_js_ts_imports(src) == ["./utils"]


def test_extract_js_ts_imports_require() -> None:
    src = 'const lodash = require("lodash");\n'
    assert extract_js_ts_imports(src) == ["lodash"]


def test_extract_js_ts_imports_dynamic() -> None:
    src = 'const mod = import("./lazy");\n'
    imports, warnings = extract_js_ts_imports(src, return_warnings=True)
    assert imports == ["./lazy"]
    assert any("dynamic" in w.lower() for w in warnings)


def test_extract_js_ts_imports_fixture(tmp_path: Path) -> None:
    src = (FIXTURES / "ts_sample.ts").read_text()
    imports, warnings = extract_js_ts_imports(src, return_warnings=True)
    # react, react-dom, ./App, ./utils from static imports
    assert "react" in imports
    assert "react-dom" in imports
    assert "./App" in imports
    assert "./utils" in imports


# ── extract_rust_imports ───────────────────────────────────────────

def test_extract_rust_imports_use() -> None:
    src = "use std::io::Read;\n"
    assert extract_rust_imports(src) == ["std"]


def test_extract_rust_imports_extern_crate() -> None:
    src = "extern crate tokio;\n"
    assert extract_rust_imports(src) == ["tokio"]


def test_extract_rust_imports_fixture() -> None:
    src = (FIXTURES / "rust_sample.rs").read_text()
    result = extract_rust_imports(src)
    assert result == ["std", "serde", "tokio"]


# ── extract_go_imports ─────────────────────────────────────────────

def test_extract_go_imports_single() -> None:
    src = 'import "fmt"\n'
    assert extract_go_imports(src) == ["fmt"]


def test_extract_go_imports_grouped() -> None:
    src = 'import (\n\t"net/http"\n\t"os"\n)\n'
    result = extract_go_imports(src)
    assert "net/http" in result
    assert "os" in result


def test_extract_go_imports_fixture() -> None:
    src = (FIXTURES / "go_sample.go").read_text()
    result = extract_go_imports(src)
    assert "fmt" in result
    assert "net/http" in result
    assert "github.com/gin-gonic/gin" in result


# ── extract_shell_imports ──────────────────────────────────────────

def test_extract_shell_imports_source() -> None:
    src = "source env.sh\n"
    assert extract_shell_imports(src) == ["env.sh"]


def test_extract_shell_imports_dot() -> None:
    src = ". ~/.bashrc\n"
    assert extract_shell_imports(src) == ["~/.bashrc"]


def test_extract_shell_imports_fixture() -> None:
    src = (FIXTURES / "shell_sample.sh").read_text()
    result = extract_shell_imports(src)
    assert result == ["env.sh", "~/.bashrc"]


# ── extract_imports_for_file ───────────────────────────────────────

def test_extract_imports_for_file_python(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text("import os\n")
    imports, warnings = extract_imports_for_file(
        str(f), "python", str(tmp_path)
    )
    assert imports == ["os"]
    assert warnings == []


def test_extract_imports_for_file_unsupported_language(tmp_path: Path) -> None:
    f = tmp_path / "x.zzz"
    f.write_text("hello\n")
    imports, warnings = extract_imports_for_file(str(f), "zig", str(tmp_path))
    assert imports == []
    assert len(warnings) == 1
    assert "unsupported" in warnings[0].lower()


# ── build_import_map ───────────────────────────────────────────────

def test_build_import_map_schema(tmp_path: Path) -> None:
    scan = {
        "project_root": str(tmp_path),
        "total_files": 1,
        "files": [
            {"path": "main.py", "language": "python"},
        ],
    }
    (tmp_path / "main.py").write_text("import os\n")
    result = build_import_map(scan, str(tmp_path))

    # Top-level required keys
    for key in ("schema_version", "source_scan", "generated_at", "files", "totals"):
        assert key in result, f"Missing key: {key}"
    assert result["schema_version"] == "1.0.0"
    assert result["source_scan"]["project_root"] == str(tmp_path)
    assert result["source_scan"]["total_files"] == 1
    assert isinstance(result["generated_at"], str)
    assert "main.py" in result["files"]
    assert result["files"]["main.py"]["imports"] == ["os"]


def test_build_import_map_totals(tmp_path: Path) -> None:
    scan = {
        "project_root": str(tmp_path),
        "total_files": 2,
        "files": [
            {"path": "a.py", "language": "python"},
            {"path": "b.py", "language": "python"},
        ],
    }
    (tmp_path / "a.py").write_text("import os\n")
    (tmp_path / "b.py").write_text("# no imports\n")
    result = build_import_map(scan, str(tmp_path))

    t = result["totals"]
    assert t["files_with_imports"] == 1
    assert t["files_without_imports"] == 1
    assert t["unique_modules"] == 1  # just "os"
    assert t["total_warnings"] == 0


# ── main() ─────────────────────────────────────────────────────────

def test_main_stdout_output(tmp_path: Path, capsys) -> None:
    scan = {
        "project_root": str(tmp_path),
        "total_files": 1,
        "files": [{"path": "m.py", "language": "python"}],
    }
    (tmp_path / "m.py").write_text("import sys\n")
    scan_path = tmp_path / "scan.json"
    scan_path.write_text(json.dumps(scan))

    sys.argv = ["extract_imports.py", str(scan_path)]
    exit_code = main()
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert exit_code == 0
    assert out["schema_version"] == "1.0.0"
    assert "m.py" in out["files"]


def test_main_invalid_input(capsys) -> None:
    sys.argv = ["extract_imports.py", "/no/such/file.json"]
    exit_code = main()
    captured = capsys.readouterr()
    assert exit_code != 0
