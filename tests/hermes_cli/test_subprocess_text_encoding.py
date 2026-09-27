"""Guard: every text-mode subprocess call in hermes_cli and pm pins utf-8 decoding (#55658, #122772).

On Windows, ``text=True`` without ``encoding`` decodes child output with the ANSI
code page (e.g. 'gbk'), and non-ASCII bytes raise UnicodeDecodeError inside
``subprocess._readerthread``, killing the backend before it becomes ready. Strict
decoding is the same crash in disguise — the exception still kills the reader
thread, ``run()`` returns ``stdout=None``, and call sites' ``OSError``/
``SubprocessError`` handlers never catch it — so text-mode sites must also pin a
non-strict ``errors=`` handler.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDED_ROOTS = [REPO_ROOT / name for name in ("hermes_cli", "pm")]


def _iter_call_text_nodes(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for root in GUARDED_ROOTS
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "test" not in p.name
    ),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
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
                    f"{path.relative_to(REPO_ROOT)}:{kw.lineno}: subprocess text=True "
                    "without encoding= — on Windows this decodes with the ANSI code page "
                    "and can raise UnicodeDecodeError on non-ASCII child output (#55658)"
                )
                errors_kw = next((k for k in call.keywords if k.arg == "errors"), None)
                assert errors_kw is not None, (
                    f"{path.relative_to(REPO_ROOT)}:{kw.lineno}: subprocess text=True "
                    "without errors= — strict decoding (the default) kills "
                    "subprocess._readerthread, run() returns stdout=None, and the "
                    "call site's OSError/SubprocessError handlers never catch it (#122772)"
                )
                if isinstance(errors_kw.value, ast.Constant):
                    assert errors_kw.value.value not in (None, "strict"), (
                        f"{path.relative_to(REPO_ROOT)}:{kw.lineno}: subprocess text=True "
                        "with errors=None or 'strict' is the same silent-None crash as "
                        "omitting errors= (#122772)"
                    )
