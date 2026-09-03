#!/usr/bin/env python3
"""Blocking supply-chain contract check: every production call to
``fetch_with_fallback`` must explicitly declare ``content_class=``.

Rationale (#81883 audit): the security property of a fetch — whether a
third-party mirror may ever supply the bytes — is decided by
``content_class``. The default is ``"executed"`` (mirrors permanently
disabled), which is safe, but the whole point of the API is that the
*contract is visible at the call site*. A call that omits
``content_class`` silently relies on the default; the next person reading
the code can't tell whether executed content was meant. Requiring the
explicit keyword makes the supply-chain decision auditable in a grep.

Test files are exempt (they deliberately exercise defaults and both
classes); production code under hermes_cli/ is not.

Exit 0 when every production call declares content_class, 1 otherwise.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ("hermes_cli",)
EXEMPT = {"tests", "test_"}


def _iter_python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for base in SCAN_DIRS:
        base_dir = ROOT / base
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*.py")):
            parts = path.parts
            if any(part in EXEMPT for part in parts):
                continue
            files.append(path)
    return files


def _calls_without_content_class(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        # Let the normal lint/tests catch syntax problems; don't fail here.
        return []

    hits: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            self.generic_visit(node)
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "fetch_with_fallback"):
                return
            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            if "content_class" not in kw_names:
                hits.append((node.lineno, ast.unparse(node)[:120]))

    Visitor().visit(tree)
    return hits


def main() -> int:
    problems: list[tuple[str, int, str]] = []
    for path in _iter_python_files():
        for lineno, snippet in _calls_without_content_class(path):
            problems.append((str(path.relative_to(ROOT)), lineno, snippet))

    if not problems:
        print("fetch_with_fallback content_class contract: OK")
        return 0

    print("fetch_with_fallback calls missing explicit content_class= :")
    for path, lineno, snippet in problems:
        print(f"  {path}:{lineno}: {snippet}")
    print("Add content_class='executed' (or 'data' for non-executed payloads).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
