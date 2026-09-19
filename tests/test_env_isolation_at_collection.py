"""A test module must not inject the developer's ~/.hermes/.env at import time.

`tests/run_agent/test_sequential_chats_live.py` is skipped unless HERMES_LIVE_TESTS=1,
but the skip only governs EXECUTION. Its `_load_user_env()` call sits at module scope,
so pytest runs it during COLLECTION — before any fixture, including the hermetic
`_hermetic_environment` autouse fixture that scrubs credential-shaped variables.

That made unrelated tests fail depending on what the developer happened to have in
`~/.hermes/.env`. With `SEARXNG_URL` present, `_get_backend()` resolved to "searxng"
and two web-backend tests failed with `assert 'searxng' == 'firecrawl'` — but only in a
run that COLLECTED the live module. Selecting the two victims out of the full tree
reproduced it with zero other tests executing, which is why an execution-order bisect
found nothing.

The env file also carries ~12 non-credential names (TELEGRAM_*, WHATSAPP_*,
HERMES_SPOTIFY_*, SEARXNG_URL) that the scrubber's credential-suffix heuristic does not
match, so this was never limited to one variable.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

# Names that write the process environment.
_ENV_MUTATORS = frozenset({"setdefault", "update", "pop", "clear", "setenv", "putenv"})

# Reading a file on the developer's machine and copying it into the environment is
# the dangerous shape: the leaked values differ per machine, so the suite passes for
# one developer and fails for another. A module-scope `os.environ["X"] = "constant"`
# is a different thing — it is deterministic, reviewable, and several modules use it
# deliberately to pin a mode (TERMINAL_ENV="local"). This test polices the former.
_FILESYSTEM_READERS = frozenset({"read_text", "readlines", "read", "load_dotenv", "open"})


def _mutates_environ(node: ast.AST) -> bool:
    """True if this node writes os.environ (or a bare `environ`)."""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and _is_environ(target.value):
                return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _ENV_MUTATORS:
            if _is_environ(func.value):
                return True
            # os.putenv / os.setenv
            if isinstance(func.value, ast.Name) and func.value.id == "os":
                return True
    return False


def _is_environ(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "environ":
        return True
    return isinstance(node, ast.Name) and node.id == "environ"


def _reads_filesystem(node: ast.AST) -> bool:
    """True if this node reads a file (the per-machine part of the hazard)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _FILESYSTEM_READERS:
        return True
    return isinstance(func, ast.Name) and func.id in _FILESYSTEM_READERS


def _module_level_env_writers(tree: ast.Module) -> set[str]:
    """Names of module-scope functions that copy file contents into the environment.

    Both halves are required: a function that only reads a file is harmless, and one
    that only sets a fixed value is deterministic. It is the combination that makes
    the suite's outcome depend on the machine it runs on.
    """
    writers = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(ast.walk(node))
        if any(_mutates_environ(sub) for sub in body) and any(
            _reads_filesystem(sub) for sub in body
        ):
            writers.add(node.name)
    return writers


def _called_at_module_scope(tree: ast.Module, names: set[str]) -> set[str]:
    """Subset of `names` invoked by a bare module-level expression."""
    called = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id in names:
            called.add(func.id)
    return called


def test_no_test_module_mutates_the_environment_at_import_time():
    offenders: list[str] = []

    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        if path == Path(__file__).resolve():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # not ours to police
            continue

        rel = path.relative_to(TESTS_ROOT)

        # (a) a bare module-level statement that copies a file into the environment
        for node in tree.body:
            if not isinstance(node, (ast.Expr, ast.Assign)):
                continue
            subtree = list(ast.walk(node))
            if any(_mutates_environ(s) for s in subtree) and any(
                _reads_filesystem(s) for s in subtree
            ):
                offenders.append(f"{rel}: module-level environment write sourced from a file")

        # (b) a helper that copies a file into the environment, invoked at module scope
        writers = _module_level_env_writers(tree)
        for name in sorted(_called_at_module_scope(tree, writers)):
            offenders.append(f"{rel}: calls {name}() at module scope, which copies file contents into os.environ")

    assert not offenders, (
        "Test modules must not copy environment values out of files at import time — "
        "pytest imports every collected module before any fixture runs, so the write "
        "escapes the hermetic environment fixture and leaks machine-specific values "
        "into unrelated tests. Read the file inside a fixture or behind the live-test "
        "gate, and import only the keys the module needs:\n  "
        + "\n  ".join(offenders)
    )
