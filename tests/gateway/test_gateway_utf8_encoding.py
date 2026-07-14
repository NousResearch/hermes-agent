"""Static guard: every ``read_text`` / ``write_text`` call in the gateway and
bundled update-response adapters must pass an explicit ``encoding=`` keyword
argument so non-UTF-8 Windows locales don't corrupt file IPC.  Mirrors the
AST-based guard pattern in
``tests/tools/test_windows_compat.py``.
"""

import ast
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GATEWAY_DIR = REPO_ROOT / "gateway"
UPDATE_RESPONSE_FILES = (
    REPO_ROOT / "plugins/platforms/discord/adapter.py",
    REPO_ROOT / "plugins/platforms/telegram/adapter.py",
    REPO_ROOT / "plugins/platforms/feishu/adapter.py",
)
METHODS = {"read_text", "write_text"}
SUPPRESSION = "# gateway-utf8: ok"


def _find_violations():
    violations = []
    py_files = list(GATEWAY_DIR.rglob("*.py")) + list(UPDATE_RESPONSE_FILES)
    for py_file in sorted(py_files):
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
            rel = py_file.relative_to(REPO_ROOT)
            violations.append(f"{rel}:{lineno}")
    return violations


def test_all_read_write_text_pass_encoding():
    violations = _find_violations()
    assert not violations, (
        "Bare read_text()/write_text() calls found (missing encoding= kwarg).\n"
        "Add encoding=\"utf-8\" or suppress with '# gateway-utf8: ok':\n"
        + "\n".join(f"  {v}" for v in violations)
    )
