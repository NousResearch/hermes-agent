#!/usr/bin/env python3
"""Fail when a test file spawns a process at import time.

A ``@pytest.mark.skipif`` condition, a class-body constant, or any other module-
or class-scope expression is evaluated while pytest is *collecting*. An
exception there is not a test failure that pytest reports and moves past -- it
is a collection error, and collection errors abort the whole directory: one
file takes the entire tree down with ``Interrupted: N errors during
collection``, so thousands of unrelated tests never run.

Probing for a binary is the usual way this happens::

    @pytest.mark.skipif(
        subprocess.run(["which", "rg"], capture_output=True).returncode != 0,
        reason="ripgrep not installed",
    )

``which`` is not an executable on Windows, so ``CreateProcess`` raises
``FileNotFoundError: [WinError 2]`` before a single test in ``tests/tools`` is
collected. The probe is also wrong in the other direction: it reports "not
installed" on any host without ``which`` even when the binary is on ``PATH``.

``shutil.which("rg") is None`` is the total form -- it never raises, it honours
``PATHEXT`` so it finds ``rg.exe``, and it costs no subprocess.

Flags, in ``tests/**/test_*.py`` and ``tests/**/conftest.py``, any call to
``subprocess.{run,call,check_call,check_output,Popen,...}`` or ``os.{system,
popen,spawn*,exec*}`` with no enclosing ``def``/``lambda`` -- one that runs at
import rather than when a test or fixture body runs. A call inside a function,
a fixture, or a test is not flagged: by then collection has already succeeded
and an exception is an ordinary, contained failure.

Opt out of one call with ``# import-time-exec: ok -- <why>`` on the line that
opens it.

Run: python scripts/ci/check_import_time_spawn.py [tests_root]
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

OPT_OUT = "import-time-exec: ok"

_SPAWNS = {
    "subprocess": {
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "getoutput",
        "getstatusoutput",
    },
    "os": {
        "system",
        "popen",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
    },
}


def _spawn_name(node: ast.Call) -> str | None:
    """Return the ``"subprocess.run"``-style name if *node* spawns a process."""
    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return None
    module, attr = func.value.id, func.attr
    if attr in _SPAWNS.get(module, ()):
        return f"{module}.{attr}"
    return None


def find_import_time_spawns(path: Path) -> list[tuple[int, str, str]]:
    """Return ``[(lineno, dotted_name, source_line)]`` of import-time spawns.

    Only a function *body* defers evaluation. Everything else attached to a
    ``def`` -- its decorators, its default argument values, its annotations --
    is evaluated where the ``def`` sits, so a spawn there still runs at import.
    That distinction is the whole point of this check: the ``skipif`` case is a
    decorator on a test method, which a naive "is there a ``FunctionDef``
    ancestor?" walk would wrongly call lazy.

    A ``ClassDef`` never defers anything -- a class body runs on import.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return []
    raw_lines = text.splitlines()
    hits: list[tuple[int, str, str]] = []

    def record(node: ast.Call) -> None:
        name = _spawn_name(node)
        if name is None:
            return
        line = raw_lines[node.lineno - 1] if node.lineno <= len(raw_lines) else ""
        if OPT_OUT not in line:
            hits.append((node.lineno, name, line.strip()))

    def visit(node: ast.AST, *, lazy: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Evaluated where the `def` sits, at the enclosing scope's laziness.
            for sub in [*node.decorator_list, *_signature_nodes(node.args)]:
                visit(sub, lazy=lazy)
            if node.returns is not None:
                visit(node.returns, lazy=lazy)
            for stmt in node.body:
                visit(stmt, lazy=True)
            return
        if isinstance(node, ast.Lambda):
            for sub in _signature_nodes(node.args):
                visit(sub, lazy=lazy)
            visit(node.body, lazy=True)
            return
        if not lazy and isinstance(node, ast.Call):
            record(node)
        for child in ast.iter_child_nodes(node):
            visit(child, lazy=lazy)

    visit(tree, lazy=False)
    return sorted(hits)


def _signature_nodes(args: ast.arguments) -> list[ast.AST]:
    """Return the parts of *args* evaluated at ``def`` time: defaults, annotations."""
    out: list[ast.AST] = [d for d in (*args.defaults, *args.kw_defaults) if d is not None]
    every = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
    out.extend(a.annotation for a in every if a is not None and a.annotation is not None)
    return out


def _test_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname == "conftest.py" or (fname.startswith("test_") and fname.endswith(".py")):
                yield Path(dirpath) / fname


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    root = Path(argv[1]) if len(argv) > 1 else repo_root / "tests"
    if not root.is_dir():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 2

    total = 0
    for path in _test_files(root):
        hits = find_import_time_spawns(path)
        if not hits:
            continue
        resolved = path.resolve()
        base = repo_root if resolved.is_relative_to(repo_root) else root.resolve()
        rel = resolved.relative_to(base).as_posix()
        for lineno, name, line in hits:
            print(
                f"{rel}:{lineno}: {name}() runs at import, so a failure here aborts "
                f"collection: {line}"
            )
            total += 1

    if total:
        print(
            f"\n{total} import-time process spawn(s) in the test tree. A module- or class-scope "
            "call runs during collection, where an exception aborts every test in the directory "
            "instead of failing one test. Move the call into the test or fixture body, use a "
            "total probe such as `shutil.which(...) is None` for a skipif condition, or mark "
            f"`# {OPT_OUT} -- <why>` on the call.",
            file=sys.stderr,
        )
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
