"""Tests for coverage-report.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\coverage-report.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("cov", SCRIPT)
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)


# --- parse_total_coverage ---

def test_parse_total_coverage_happy():
    out = "tests/hermes_cli/test_foo.py 12 1 92%\ntests/hermes_cli/test_bar.py 8 0 100%\nTOTAL                       20 1 95%\n"
    result = cov.parse_total_coverage(out)
    assert result["total_statements"] == 20
    assert result["missing"] == 1
    assert result["covered"] == 19
    assert result["pct"] == 95


def test_parse_total_coverage_returns_empty_when_not_found():
    out = "some pytest output without a TOTAL line\n"
    result = cov.parse_total_coverage(out)
    assert result == {}


def test_parse_total_coverage_handles_zero_missing():
    out = "TOTAL 100 0 100%\n"
    result = cov.parse_total_coverage(out)
    assert result["covered"] == 100
    assert result["pct"] == 100


# --- parse_top_uncovered ---

def test_parse_top_uncovered_sorts_lowest_first(tmp_path):
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps({
        "files": {
            "tools/foo.py": {"summary": {"percent_covered": 90.0, "num_statements": 100, "missing_lines": 10}},
            "tools/bar.py": {"summary": {"percent_covered": 30.0, "num_statements": 50, "missing_lines": 35}},
            "tools/baz.py": {"summary": {"percent_covered": 60.0, "num_statements": 20, "missing_lines": 8}},
        }
    }))
    result = cov.parse_top_uncovered(cov_json, n=5)
    assert len(result) == 3
    assert result[0]["file"] == "tools/bar.py"  # 30% first
    assert result[0]["pct"] == 30.0
    assert result[1]["file"] == "tools/baz.py"  # 60%
    assert result[2]["file"] == "tools/foo.py"  # 90%


def test_parse_top_uncovered_skips_zero_statement_files(tmp_path):
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps({
        "files": {
            "tools/foo.py": {"summary": {"percent_covered": 100.0, "num_statements": 0, "missing_lines": 0}},
            "tools/bar.py": {"summary": {"percent_covered": 50.0, "num_statements": 10, "missing_lines": 5}},
        }
    }))
    result = cov.parse_top_uncovered(cov_json, n=5)
    assert len(result) == 1
    assert result[0]["file"] == "tools/bar.py"


def test_parse_top_uncovered_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    result = cov.parse_top_uncovered(missing, n=5)
    assert result == []


def test_parse_top_uncovered_handles_invalid_json(tmp_path):
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text("not json {")
    result = cov.parse_top_uncovered(cov_json, n=5)
    assert result == []


def test_parse_top_uncovered_respects_n(tmp_path):
    cov_json = tmp_path / "coverage.json"
    files = {}
    for i in range(10):
        files[f"tools/f{i}.py"] = {"summary": {"percent_covered": float(i*10), "num_statements": 10, "missing_lines": 10-i}}
    cov_json.write_text(json.dumps({"files": files}))
    result = cov.parse_top_uncovered(cov_json, n=3)
    assert len(result) == 3


# --- compose_message ---

def test_compose_message_includes_pct_and_count():
    coverage = {"pct": 75, "covered": 150, "total_statements": 200, "missing": 50}
    msg = cov.compose_message(coverage, [], 100)
    assert "75%" in msg
    assert "150/200" in msg
    assert "100 tests" in msg


def test_compose_message_high_coverage_green():
    coverage = {"pct": 90, "covered": 180, "total_statements": 200, "missing": 20}
    msg = cov.compose_message(coverage, [], 50)
    assert "🟢" in msg


def test_compose_message_low_coverage_red():
    coverage = {"pct": 30, "covered": 60, "total_statements": 200, "missing": 140}
    msg = cov.compose_message(coverage, [], 50)
    assert "🔴" in msg


def test_compose_message_includes_top_uncovered():
    coverage = {"pct": 75, "covered": 150, "total_statements": 200, "missing": 50}
    top = [{"file": "C:/Users/bbask/path/to/tools/foo.py", "pct": 10, "missing": 90, "statements": 100}]
    msg = cov.compose_message(coverage, top, 100)
    assert "top under-covered" in msg
    assert "foo.py" in msg
    assert "10%" in msg


def test_compose_message_empty_coverage():
    msg = cov.compose_message({}, [], 50)
    assert "couldn't parse" in msg


# --- run_pytest_cov (smoke test) ---

def test_run_pytest_cov_calls_subprocess():
    with patch.object(cov.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        rc, output = cov.run_pytest_cov()
    assert rc == 0
    assert output == "ok"
    # Verify --cov flag was passed
    call_args = mock_run.call_args[0][0]
    assert any("--cov=hermes_cli" in str(a) for a in call_args)
    assert any("--cov=tools" in str(a) for a in call_args)