"""Tests for scripts/code-scan/classify_imports.py — Phase 4 D1 import classification."""

import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "code-scan"
sys.path.insert(0, str(SCRIPTS_DIR))

from classify_imports import (
    classify_import,
    build_classified_map,
    load_scan_and_imports,
    detect_local_roots,
    main,
)

# ── load_scan_and_imports ──────────────────────────────────────────


def test_load_scan_and_imports_valid(tmp_path: Path) -> None:
    scan = {"project_root": "/foo", "total_files": 1, "files": []}
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {"project_root": "/foo"},
        "generated_at": "2026-01-01T00:00:00Z",
        "files": {},
        "totals": {},
    }
    scan_p = tmp_path / "scan.json"
    imports_p = tmp_path / "imports.json"
    scan_p.write_text(json.dumps(scan))
    imports_p.write_text(json.dumps(imports_artifact))

    scan_data, imports_data = load_scan_and_imports(str(scan_p), str(imports_p))
    assert scan_data["project_root"] == "/foo"
    assert imports_data["schema_version"] == "1.0.0"


def test_load_scan_and_imports_missing_scan(tmp_path: Path) -> None:
    """load_scan_and_imports raises FileNotFoundError for missing scan file."""
    imports_p = tmp_path / "imports.json"
    imports_p.write_text("{}")
    with pytest.raises(FileNotFoundError, match="Scan file not found"):
        load_scan_and_imports("/no/such/scan.json", str(imports_p))


def test_load_scan_and_imports_missing_imports(tmp_path: Path) -> None:
    """load_scan_and_imports raises FileNotFoundError for missing imports file."""
    scan_p = tmp_path / "scan.json"
    scan_p.write_text('{"files": []}')
    with pytest.raises(FileNotFoundError, match="Imports file not found"):
        load_scan_and_imports(str(scan_p), "/no/such/imports.json")


def test_main_no_args_usage_exit(capsys) -> None:
    """CLI with no args exits non-zero via argparse."""
    sys.argv = ["classify_imports.py"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "scan_json" in captured.err.lower() or "usage" in captured.err.lower()


# ── detect_local_roots ─────────────────────────────────────────────


def test_detect_local_roots_from_scan() -> None:
    scan = {
        "files": [
            {"path": "src/main.py", "language": "python"},
            {"path": "src/utils.py", "language": "python"},
            {"path": "tests/test_main.py", "language": "python"},
        ]
    }
    roots = detect_local_roots(scan)
    assert "src" in roots
    assert "tests" in roots


def test_detect_local_roots_nested() -> None:
    scan = {
        "files": [
            {"path": "pkg/sub/mod.py", "language": "python"},
        ]
    }
    roots = detect_local_roots(scan)
    # Should include top-level dir
    assert "pkg" in roots


def test_detect_local_roots_empty() -> None:
    scan: dict = {"files": []}
    roots = detect_local_roots(scan)
    assert roots == set()


# ── classify_import (core logic) ────────────────────────────────────


def test_classify_relative_js_dot() -> None:
    local_roots: set[str] = set()
    assert classify_import("./utils", "javascript", local_roots) == "relative"


def test_classify_relative_js_dotdot() -> None:
    local_roots: set[str] = set()
    assert classify_import("../shared", "javascript", local_roots) == "relative"


def test_classify_relative_ts() -> None:
    local_roots: set[str] = set()
    assert classify_import("./components/App", "typescript", local_roots) == "relative"


def test_classify_relative_python_dot() -> None:
    local_roots: set[str] = set()
    # Python relative import strings start with .
    assert classify_import(".models", "python", local_roots) == "relative"


def test_classify_relative_python_dotdot() -> None:
    local_roots: set[str] = set()
    assert classify_import("..", "python", local_roots) == "relative"


# ── stdlib ─────────────────────────────────────────────────────────


def test_classify_python_stdlib() -> None:
    local_roots: set[str] = set()
    assert classify_import("os", "python", local_roots) == "stdlib"
    assert classify_import("sys", "python", local_roots) == "stdlib"
    assert classify_import("pathlib", "python", local_roots) == "stdlib"
    assert classify_import("json", "python", local_roots) == "stdlib"


def test_classify_python_non_stdlib() -> None:
    # requests is definitely NOT stdlib
    local_roots: set[str] = set()
    assert classify_import("requests", "python", local_roots) == "third_party"


def test_classify_go_stdlib() -> None:
    local_roots: set[str] = set()
    assert classify_import("fmt", "go", local_roots) == "stdlib"
    assert classify_import("net/http", "go", local_roots) == "stdlib"
    assert classify_import("os", "go", local_roots) == "stdlib"


def test_classify_go_struct_is_not_stdlib() -> None:
    """'struct' is not a real Go top-level stdlib package — should be third_party."""
    local_roots: set[str] = set()
    assert classify_import("struct", "go", local_roots) == "third_party"


def test_classify_rust_stdlib() -> None:
    local_roots: set[str] = set()
    assert classify_import("std", "rust", local_roots) == "stdlib"


def test_classify_unknown_language_stdlib() -> None:
    local_roots: set[str] = set()
    # For languages without stdlib mapping → unknown
    assert classify_import("whatever", "unknown_lang", local_roots) == "unknown"


# ── local ──────────────────────────────────────────────────────────


def test_classify_local_python() -> None:
    local_roots = {"src", "app"}
    assert classify_import("src", "python", local_roots) == "local"
    assert classify_import("app", "python", local_roots) == "local"


def test_classify_local_not_root() -> None:
    local_roots = {"src"}
    # 'os' is stdlib even if there are dirs named weirdly
    # But 'src_helpers' is NOT in roots
    assert classify_import("src_helpers", "python", local_roots) == "third_party"


# ── third_party ────────────────────────────────────────────────────


def test_classify_third_party_python() -> None:
    local_roots: set[str] = set()
    assert classify_import("flask", "python", local_roots) == "third_party"
    assert classify_import("numpy", "python", local_roots) == "third_party"
    assert classify_import("requests", "python", local_roots) == "third_party"


def test_classify_third_party_js() -> None:
    local_roots: set[str] = set()
    assert classify_import("react", "javascript", local_roots) == "third_party"
    assert classify_import("lodash", "javascript", local_roots) == "third_party"
    assert classify_import("express", "javascript", local_roots) == "third_party"


def test_classify_third_party_rust() -> None:
    local_roots: set[str] = set()
    assert classify_import("serde", "rust", local_roots) == "third_party"
    assert classify_import("tokio", "rust", local_roots) == "third_party"


def test_classify_third_party_go() -> None:
    local_roots: set[str] = set()
    # Go packages with github.com prefix are third-party
    assert classify_import("github.com/gin-gonic/gin", "go", local_roots) == "third_party"


# ── unknown ────────────────────────────────────────────────────────


def test_classify_empty_module() -> None:
    local_roots: set[str] = set()
    assert classify_import("", "python", local_roots) == "unknown"


def test_classify_unknown_language() -> None:
    local_roots: set[str] = set()
    assert classify_import("something", "brainfuck", local_roots) == "unknown"


# ── build_classified_map ───────────────────────────────────────────


def test_build_classified_map_basic() -> None:
    scan = {
        "project_root": "/tmp/test",
        "total_files": 2,
        "files": [
            {"path": "src/app.py", "language": "python"},
            {"path": "src/utils.py", "language": "python"},
        ],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {"project_root": "/tmp/test"},
        "generated_at": "2026-01-01T00:00:00Z",
        "files": {
            "src/app.py": {
                "imports": ["os", "flask", "src"],
                "warnings": ["some warning"],
            },
            "src/utils.py": {
                "imports": ["json"],
            },
        },
        "totals": {
            "files_with_imports": 2,
            "files_without_imports": 0,
            "unique_modules": 3,
            "total_warnings": 1,
        },
    }
    result = build_classified_map(scan, imports_artifact)

    assert result["schema_version"] == "1.0.0"
    assert "src/app.py" in result["files"]
    assert "src/utils.py" in result["files"]

    app_imports = result["files"]["src/app.py"]["imports"]
    assert any(i["module"] == "os" and i["classification"] == "stdlib" for i in app_imports)
    assert any(i["module"] == "flask" and i["classification"] == "third_party" for i in app_imports)
    assert any(i["module"] == "src" and i["classification"] == "local" for i in app_imports)

    # Warnings preserved
    assert "warnings" in result["files"]["src/app.py"]
    assert result["files"]["src/app.py"]["warnings"] == ["some warning"]

    utils_imports = result["files"]["src/utils.py"]["imports"]
    assert any(i["module"] == "json" and i["classification"] == "stdlib" for i in utils_imports)

    # Totals at file level
    app_totals = result["files"]["src/app.py"]["totals"]
    assert app_totals["stdlib"] == 1
    assert app_totals["third_party"] == 1
    assert app_totals["local"] == 1

    # Global totals
    assert result["totals"]["stdlib"] == 2  # os, json
    assert result["totals"]["third_party"] == 1  # flask
    assert result["totals"]["local"] == 1  # src


def test_build_classified_map_totals_accumulate() -> None:
    scan = {
        "project_root": "/t",
        "total_files": 1,
        "files": [{"path": "a.py", "language": "python"}],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {},
        "generated_at": "2026-01-01",
        "files": {
            "a.py": {"imports": ["os", "flask", "numpy"]},
        },
        "totals": {},
    }
    result = build_classified_map(scan, imports_artifact)

    t = result["totals"]
    assert t["stdlib"] == 1  # os
    assert t["third_party"] == 2  # flask, numpy
    assert t["local"] == 0
    assert t["relative"] == 0
    assert t["unknown"] == 0


# ── idempotency: already-classified input should fail loudly ──────


def test_build_classified_map_already_classified() -> None:
    """If input already has classified dicts {module, classification}, reject."""
    scan = {
        "project_root": "/t",
        "total_files": 1,
        "files": [{"path": "a.py", "language": "python"}],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {},
        "generated_at": "2026-01-01",
        "files": {
            "a.py": {"imports": [{"module": "os", "classification": "stdlib"}]},
        },
        "totals": {},
    }
    with pytest.raises(ValueError, match="already classified"):
        build_classified_map(scan, imports_artifact)


def test_build_classified_map_already_classified_later_position() -> None:
    """Already-classified dict in a later import position must also be rejected."""
    scan = {
        "project_root": "/t",
        "total_files": 1,
        "files": [{"path": "a.py", "language": "python"}],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {},
        "generated_at": "2026-01-01",
        "files": {
            "a.py": {
                "imports": [
                    "os",
                    "json",
                    {"module": "flask", "classification": "third_party"},
                ],
            },
        },
        "totals": {},
    }
    with pytest.raises(ValueError, match="already classified"):
        build_classified_map(scan, imports_artifact)


def test_build_classified_map_already_classified_in_second_file() -> None:
    """Already-classified dict in a later file must also be rejected."""
    scan = {
        "project_root": "/t",
        "total_files": 2,
        "files": [
            {"path": "a.py", "language": "python"},
            {"path": "b.py", "language": "python"},
        ],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {},
        "generated_at": "2026-01-01",
        "files": {
            "a.py": {"imports": ["os", "json"]},
            "b.py": {"imports": [{"module": "requests", "classification": "third_party"}]},
        },
        "totals": {},
    }
    with pytest.raises(ValueError, match="already classified"):
        build_classified_map(scan, imports_artifact)


# ── CLI main() ─────────────────────────────────────────────────────
# (test_main_no_args_usage_exit lives earlier, alongside load tests)


def test_main_valid_input(tmp_path: Path, capsys) -> None:
    scan = {
        "project_root": str(tmp_path),
        "total_files": 1,
        "files": [
            {"path": "app.py", "language": "python"},
        ],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {"project_root": str(tmp_path)},
        "generated_at": "2026-01-01T00:00:00Z",
        "files": {
            "app.py": {
                "imports": ["os", "requests"],
            },
        },
        "totals": {"files_with_imports": 1, "files_without_imports": 0, "unique_modules": 2, "total_warnings": 0},
    }
    scan_p = tmp_path / "scan.json"
    imports_p = tmp_path / "imports.json"
    scan_p.write_text(json.dumps(scan))
    imports_p.write_text(json.dumps(imports_artifact))

    sys.argv = ["classify_imports.py", str(scan_p), str(imports_p)]
    rc = main()
    captured = capsys.readouterr()
    assert rc == 0
    out = json.loads(captured.out)
    assert out["schema_version"] == "1.0.0"
    assert "app.py" in out["files"]
    assert "totals" in out


def test_main_missing_file(tmp_path: Path, capsys) -> None:
    sys.argv = ["classify_imports.py", "/no/such/scan.json", f"{tmp_path}/imports.json"]
    rc = main()
    assert rc != 0


def test_main_malformed_json(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("<<< not json >>>")
    sys.argv = ["classify_imports.py", str(bad), str(bad)]
    rc = main()
    assert rc != 0


def test_main_output_totals(tmp_path: Path, capsys) -> None:
    scan = {
        "project_root": str(tmp_path),
        "total_files": 1,
        "files": [{"path": "x.py", "language": "python"}],
    }
    imports_artifact = {
        "schema_version": "1.0.0",
        "source_scan": {},
        "generated_at": "2026-01-01",
        "files": {
            "x.py": {"imports": ["os", "flask", "./local_mod"]},
        },
        "totals": {},
    }
    scan_p = tmp_path / "scan.json"
    imports_p = tmp_path / "imports.json"
    scan_p.write_text(json.dumps(scan))
    imports_p.write_text(json.dumps(imports_artifact))

    sys.argv = ["classify_imports.py", str(scan_p), str(imports_p)]
    rc = main()
    captured = capsys.readouterr()
    out = json.loads(captured.out)

    global_t = out["totals"]
    assert global_t["stdlib"] == 1
    assert global_t["third_party"] == 1
    assert global_t["relative"] == 1
    assert global_t["local"] == 0
    assert global_t["unknown"] == 0


# ── shell relative paths ───────────────────────────────────────────


def test_classify_relative_shell() -> None:
    local_roots: set[str] = set()
    assert classify_import("./env.sh", "bash", local_roots) == "relative"
    assert classify_import("../common.sh", "shell", local_roots) == "relative"


# ── JS relative from scan (./ and ../) ──────────────────────────────


def test_classify_relative_typescript_nested() -> None:
    local_roots: set[str] = set()
    # JS/TS relative imports start with ./ or ../
    assert classify_import("../utils/helpers", "typescript", local_roots) == "relative"
    assert classify_import("./index", "typescript", local_roots) == "relative"
