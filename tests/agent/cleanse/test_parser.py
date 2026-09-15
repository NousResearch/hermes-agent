"""Tests for agent.cleanse.parser — diagnostic normalization."""

import json
from pathlib import Path

import pytest

from agent.cleanse.parser import (
    Diagnostic,
    normalize_diagnostics,
    parse_actionlint_output,
    parse_eslint_json,
    parse_ruff_json,
    parse_tsc_output,
)


def test_diagnostic_to_dict():
    """Diagnostic.to_dict() serializes all fields."""
    diag = Diagnostic(
        file="src/app.ts",
        line=42,
        col=10,
        code="TS2304",
        severity="error",
        message="Cannot find name 'foo'",
    )
    assert diag.to_dict() == {
        "file": "src/app.ts",
        "line": 42,
        "col": 10,
        "code": "TS2304",
        "severity": "error",
        "message": "Cannot find name 'foo'",
    }


def test_parse_tsc_output(tmp_path):
    """parse_tsc_output() extracts TypeScript errors."""
    output = """
src/app.ts(10,5): error TS2304: Cannot find name 'foo'.
src/utils.ts(42,15): warning TS6133: 'bar' is declared but never used.
"""
    result = parse_tsc_output(output, tmp_path)
    assert len(result) == 2
    assert result[0].file == "src/app.ts"
    assert result[0].line == 10
    assert result[0].col == 5
    assert result[0].code == "TS2304"
    assert result[0].severity == "error"
    assert "Cannot find name 'foo'" in result[0].message


def test_parse_eslint_json(tmp_path):
    """parse_eslint_json() extracts ESLint diagnostics from JSON."""
    eslint_output = json.dumps(
        [
            {
                "filePath": str(tmp_path / "src/app.js"),
                "messages": [
                    {
                        "line": 5,
                        "column": 10,
                        "ruleId": "no-unused-vars",
                        "severity": 2,
                        "message": "'foo' is defined but never used",
                    }
                ],
            }
        ]
    )
    result = parse_eslint_json(eslint_output, tmp_path)
    assert len(result) == 1
    assert result[0].file == "src/app.js"
    assert result[0].line == 5
    assert result[0].col == 10
    assert result[0].code == "no-unused-vars"
    assert result[0].severity == "error"


def test_parse_actionlint_output(tmp_path):
    """parse_actionlint_output() extracts actionlint diagnostics."""
    output = """
.github/workflows/ci.yml:10:5: shellcheck reported issue [shellcheck]
.github/workflows/test.yml:20:8: unknown action [action-name]
"""
    result = parse_actionlint_output(output, tmp_path)
    assert len(result) == 2
    assert result[0].file == ".github/workflows/ci.yml"
    assert result[0].line == 10
    assert result[0].col == 5
    assert result[0].code == "shellcheck"
    assert result[0].severity == "error"


def test_parse_ruff_json(tmp_path):
    """parse_ruff_json() extracts ruff diagnostics from JSON."""
    ruff_output = json.dumps(
        [
            {
                "filename": str(tmp_path / "main.py"),
                "location": {"row": 15, "column": 5},
                "code": "F401",
                "severity": "E",
                "message": "unused import",
            }
        ]
    )
    result = parse_ruff_json(ruff_output, tmp_path)
    assert len(result) == 1
    assert result[0].file == "main.py"
    assert result[0].line == 15
    assert result[0].col == 5
    assert result[0].code == "F401"
    assert result[0].severity == "error"


def test_normalize_diagnostics_unknown_tool(tmp_path):
    """normalize_diagnostics() returns empty list for unknown tools."""
    result = normalize_diagnostics("unknown-tool", "some output", tmp_path)
    assert result == []


def test_normalize_diagnostics_tsc(tmp_path):
    """normalize_diagnostics() dispatches to tsc parser."""
    output = "src/app.ts(10,5): error TS2304: Cannot find name 'foo'."
    result = normalize_diagnostics("tsc", output, tmp_path)
    assert len(result) == 1
    assert result[0].code == "TS2304"
