"""Tests for the diagnostic reporter (formatting layer)."""
from __future__ import annotations

from agent.lsp.reporter import (
    DEFAULT_SEVERITIES,
    MAX_PER_FILE,
    format_diagnostic,
    report_for_file,
    truncate,
)


def _diag(line=0, col=0, sev=1, code="E001", source="ls", msg="oops"):
    return {
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + 1},
        },
        "severity": sev,
        "code": code,
        "source": source,
        "message": msg,
    }


def test_format_diagnostic_uses_one_indexed_position():
    line = format_diagnostic(_diag(line=4, col=2))
    assert "[5:3]" in line  # +1 on both


def test_format_diagnostic_includes_severity_label():
    assert format_diagnostic(_diag(sev=1)).startswith("ERROR")
    assert format_diagnostic(_diag(sev=2)).startswith("WARN")
    assert format_diagnostic(_diag(sev=3)).startswith("INFO")
    assert format_diagnostic(_diag(sev=4)).startswith("HINT")


def test_format_diagnostic_includes_code_and_source():
    line = format_diagnostic(_diag(code="X42", source="src"))
    assert "[X42]" in line
    assert "(src)" in line


def test_format_diagnostic_omits_missing_optional_fields():
    line = format_diagnostic(
        {
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 0},
            },
            "severity": 1,
            "message": "bare",
        }
    )
    assert "[" not in line.split("]", 1)[1]  # no extra brackets after the position
    assert "(" not in line


def test_report_for_file_returns_empty_when_only_warnings():
    """Default severity filter is ERROR-only."""
    report = report_for_file("/x.py", [_diag(sev=2)])
    assert report == ""


def test_report_for_file_emits_block_with_errors():
    diag = _diag(msg="real error")
    report = report_for_file("/x.py", [diag])
    assert "<diagnostics file=\"/x.py\">" in report
    assert "real error" in report
    assert "</diagnostics>" in report


def test_report_for_file_caps_at_max_per_file():
    diags = [_diag(line=i) for i in range(MAX_PER_FILE + 5)]
    report = report_for_file("/x.py", diags)
    assert "and 5 more" in report


def test_report_for_file_respects_custom_severities():
    diag = _diag(sev=2, msg="warn")
    report = report_for_file("/x.py", [diag], severities=frozenset({1, 2}))
    assert "warn" in report


def test_truncate_below_limit_unchanged():
    s = "abc" * 100
    assert truncate(s, limit=4000) == s


def test_truncate_above_limit_appends_marker():
    s = "x" * 10000
    out = truncate(s, limit=200)
    assert out.endswith("[truncated]")
    assert len(out) <= 200
