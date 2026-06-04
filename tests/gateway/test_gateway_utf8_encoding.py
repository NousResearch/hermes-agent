"""Static guard: every ``read_text`` / ``write_text`` call under ``gateway/``
must pass an explicit ``encoding=`` keyword argument so non-UTF-8 Windows
locales don't corrupt file IPC.  Mirrors the AST-based guard pattern in
``tests/tools/test_windows_compat.py``.
"""

import ast
import pathlib
import pytest

GATEWAY_DIR = pathlib.Path(__file__).resolve().parents[2] / "gateway"
METHODS = {"read_text", "write_text"}
SUPPRESSION = "# gateway-utf8: ok"


def _find_violations():
    violations = []
    for py_file in sorted(GATEWAY_DIR.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in METHODS:
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            lineno = node.lineno
            if lineno <= len(source_lines) and SUPPRESSION in source_lines[lineno - 1]:
                continue
            rel = py_file.relative_to(GATEWAY_DIR.parent)
            violations.append(f"{rel}:{lineno}")
    return violations


def test_all_read_write_text_pass_encoding():
    violations = _find_violations()
    assert not violations, (
        "Bare read_text()/write_text() calls found (missing encoding= kwarg).\n"
        "Add encoding=\"utf-8\" or suppress with '# gateway-utf8: ok':\n"
        + "\n".join(f"  {v}" for v in violations)
    )
