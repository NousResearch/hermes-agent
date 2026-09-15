"""No test module may define the same name twice in one scope.

Python keeps only the last definition, so a duplicate silently deletes the
first one. When the shadowed name is a test, its coverage disappears with no
error and no skip; when it is an ``autouse`` fixture, the whole module quietly
switches to the newer one's isolation rules.

Both had already happened here:

* ``tests/agent/test_auxiliary_client.py`` grew a second ``_clean_env``
  ``autouse`` fixture alongside an NVIDIA feature. It replaced the original for
  all 158 tests in the file, dropping the ``ANTHROPIC_API_KEY`` /
  ``ANTHROPIC_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN`` env stripping and the
  ``_aux_unhealthy_*`` cache reset between tests. Individual tests had started
  clearing that cache by hand to work around it.
* ``tests/gateway/test_mattermost.py`` had two copies of
  ``test_progress_send_with_invalid_thread_root_never_falls_back_flat``; the
  surviving one omitted the recorded 400 ``invalid root_id`` state, so the
  case the name describes was never exercised.

This guard is cheap and catches the whole class at collection time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# Decorators that legitimately repeat a name in one scope.
_REDEFINING_DECORATORS = ("overload", "setter", "getter", "deleter", "register")

# Throwaway callbacks conventionally named `_` are not shadowing bugs.
_ALLOWED_REPEATS = {"_"}


def _decorator_names(node: ast.AST) -> list[str]:
    out = []
    for dec in getattr(node, "decorator_list", []):
        try:
            out.append(ast.unparse(dec))
        except Exception:  # pragma: no cover - defensive
            pass
    return out


def _duplicates_in(body, scope: str, rel: str) -> list[str]:
    seen: dict[str, int] = {}
    problems: list[str] = []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in _ALLOWED_REPEATS:
            continue
        if any(
            marker in dec
            for dec in _decorator_names(node)
            for marker in _REDEFINING_DECORATORS
        ):
            seen[node.name] = node.lineno
            continue
        if node.name in seen:
            problems.append(
                f"{rel}:{node.lineno} {scope}.{node.name}() shadows the "
                f"definition at line {seen[node.name]}"
            )
        seen[node.name] = node.lineno
    return problems


def _test_modules() -> list[Path]:
    return sorted(TESTS_ROOT.rglob("test_*.py"))


def test_no_shadowed_definitions_in_test_modules():
    problems: list[str] = []
    for path in _test_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        problems += _duplicates_in(tree.body, "<module>", rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                problems += _duplicates_in(node.body, node.name, rel)

    assert not problems, (
        "A duplicate definition silently deletes the earlier one. Rename or "
        "remove:\n  " + "\n  ".join(problems)
    )


def test_guard_detects_a_known_duplicate_shape():
    """The guard must actually fire, not vacuously pass."""
    src = (
        "def test_a():\n    pass\n\n"
        "def test_a():\n    pass\n"
    )
    tree = ast.parse(src)
    assert _duplicates_in(tree.body, "<module>", "fake.py")


@pytest.mark.parametrize("decorator", ["@property", "@x.setter", "@functools.singledispatch"])
def test_guard_allows_legitimate_redefinition(decorator):
    src = (
        "class C:\n"
        "    def f(self):\n        pass\n"
        f"    {decorator}\n"
        "    def f(self):\n        pass\n"
    )
    tree = ast.parse(src)
    cls = tree.body[0]
    problems = _duplicates_in(cls.body, "C", "fake.py")
    # Only the setter/getter/register family is exempt; a bare @property
    # repeat is still a real shadow.
    if any(m in decorator for m in _REDEFINING_DECORATORS):
        assert not problems
    else:
        assert problems
