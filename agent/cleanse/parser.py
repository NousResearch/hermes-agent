"""Diagnostic parsing and normalization.

Parsers convert tool-specific output formats into a unified Diagnostic schema:
{file, line, col, code, severity, message}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Diagnostic:
    """Normalized diagnostic from any checker."""

    file: str
    line: int
    col: int | None
    code: str
    severity: str  # "error", "warning", "info"
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


def parse_tsc_output(output: str, root: Path) -> list[Diagnostic]:
    """Parse TypeScript compiler output.

    Format: path/to/file.ts(line,col): error TSxxxx: message
    """
    diagnostics = []
    pattern = re.compile(r"^(.+?)\((\d+),(\d+)\):\s+(error|warning|info)\s+(\w+):\s+(.+)$")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        file_path, line_num, col_num, severity, code, message = match.groups()
        try:
            abs_file = (root / file_path).resolve()
            rel_file = abs_file.relative_to(root)
        except (ValueError, OSError):
            rel_file = Path(file_path)
        diagnostics.append(
            Diagnostic(
                file=str(rel_file),
                line=int(line_num),
                col=int(col_num),
                code=code,
                severity=severity.lower(),
                message=message.strip(),
            )
        )
    return diagnostics


def parse_eslint_json(output: str, root: Path) -> list[Diagnostic]:
    """Parse ESLint JSON output (--format json)."""
    diagnostics = []
    try:
        results = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return diagnostics

    if not isinstance(results, list):
        return diagnostics

    for result in results:
        if not isinstance(result, dict):
            continue
        file_path = result.get("filePath", "")
        if not file_path:
            continue
        try:
            abs_file = Path(file_path).resolve()
            rel_file = abs_file.relative_to(root)
        except (ValueError, OSError):
            rel_file = Path(file_path)

        messages = result.get("messages", [])
        if not isinstance(messages, list):
            continue

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            diagnostics.append(
                Diagnostic(
                    file=str(rel_file),
                    line=msg.get("line", 1),
                    col=msg.get("column"),
                    code=msg.get("ruleId", "unknown"),
                    severity={"1": "warning", "2": "error"}.get(str(msg.get("severity", 1)), "warning"),
                    message=msg.get("message", ""),
                )
            )
    return diagnostics


def parse_actionlint_output(output: str, root: Path) -> list[Diagnostic]:
    """Parse actionlint output.

    Format: .github/workflows/file.yml:line:col: message [rule-id]
    """
    diagnostics = []
    pattern = re.compile(r"^(.+?):(\d+):(\d+):\s+(.+?)(?:\s+\[(.+?)\])?$")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        file_path, line_num, col_num, message, code = match.groups()
        try:
            abs_file = (root / file_path).resolve()
            rel_file = abs_file.relative_to(root)
        except (ValueError, OSError):
            rel_file = Path(file_path)
        diagnostics.append(
            Diagnostic(
                file=str(rel_file),
                line=int(line_num),
                col=int(col_num) if col_num else None,
                code=code or "actionlint",
                severity="error",
                message=message.strip(),
            )
        )
    return diagnostics


def parse_ruff_json(output: str, root: Path) -> list[Diagnostic]:
    """Parse ruff JSON output (--format json)."""
    diagnostics = []
    try:
        results = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return diagnostics

    if not isinstance(results, list):
        return diagnostics

    for item in results:
        if not isinstance(item, dict):
            continue
        file_path = item.get("filename", "")
        if not file_path:
            continue
        try:
            abs_file = Path(file_path).resolve()
            rel_file = abs_file.relative_to(root)
        except (ValueError, OSError):
            rel_file = Path(file_path)

        location = item.get("location", {})
        if not isinstance(location, dict):
            location = {}

        diagnostics.append(
            Diagnostic(
                file=str(rel_file),
                line=location.get("row", 1),
                col=location.get("column"),
                code=item.get("code", "unknown"),
                severity="error" if item.get("severity") == "E" else "warning",
                message=item.get("message", ""),
            )
        )
    return diagnostics


def normalize_diagnostics(tool: str, output: str, root: Path) -> list[Diagnostic]:
    """Normalize tool output into unified Diagnostic format.

    Args:
        tool: Tool name ("tsc", "eslint", "actionlint", "ruff", etc.)
        output: Raw tool output
        root: Project root for path resolution

    Returns:
        List of normalized diagnostics
    """
    parsers = {
        "tsc": parse_tsc_output,
        "eslint": parse_eslint_json,
        "actionlint": parse_actionlint_output,
        "ruff": parse_ruff_json,
    }
    parser = parsers.get(tool)
    if parser is None:
        return []
    return parser(output, root)
