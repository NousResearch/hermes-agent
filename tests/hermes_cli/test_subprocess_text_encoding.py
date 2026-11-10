"""Guard: every text-mode subprocess call in hermes_cli pins utf-8 decoding (#55658).

On Windows, ``text=True`` without ``encoding`` decodes child output with the ANSI
code page (e.g. 'gbk'), and non-ASCII bytes raise UnicodeDecodeError inside
``subprocess._readerthread``, killing the backend before it becomes ready.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HERMES_CLI = Path(__file__).resolve().parents[2] / "hermes_cli"


def _iter_call_text_nodes(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


@pytest.mark.parametrize(
    "path",
    sorted(p for p in HERMES_CLI.rglob("*.py") if "__pycache__" not in p.parts and "test" not in p.name),
    ids=lambda p: str(p.relative_to(HERMES_CLI)),
)
def test_text_mode_subprocess_calls_pin_utf8(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for call in _iter_call_text_nodes(tree):
        names = {kw.arg for kw in call.keywords}
        for kw in call.keywords:
            if (
                kw.arg == "text"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                assert "encoding" in names, (
                    f"{path.relative_to(HERMES_CLI)}:{kw.lineno}: subprocess text=True "
                    "without encoding= — on Windows this decodes with the ANSI code page "
                    "and can raise UnicodeDecodeError on non-ASCII child output (#55658)"
                )
