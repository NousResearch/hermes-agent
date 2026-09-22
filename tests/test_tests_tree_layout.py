"""The tests/ tree mirrors the source tree.

``scripts/run_tests.sh tests/<dir>/`` is how a change gets its regression
coverage run, so a test filed under a directory that does not correspond to
the code it exercises is a test nobody runs when that code changes. Two
drifts had accumulated: parallel directories for one source package
(``tests/cli`` beside ``tests/hermes_cli``, ``tests/run_agent`` beside
``tests/agent``, ``tests/state`` beside ``tests/hermes_state``) and ~250
loose files at ``tests/`` root that belonged to a package.

Filenames also stopped carrying issue numbers: ``test_89315_x.py`` reads as
noise in a directory listing and the number belongs in the docstring, where
``git log -S`` and a reader can both find it with context.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# tests/<name>/ directories that do not mirror a source directory but are
# legitimate homes: cross-cutting suites, fixtures, and script-family tests.
_NON_MIRROR_DIRS = {
    "ci", "conformance", "dashboard", "desktop", "docker", "e2e", "evals",
    "fakes", "fixtures", "honcho_plugin", "install", "integration", "manual",
    "monitoring", "openviking_plugin", "perf_guards", "scripts", "secret_sources",
    "security", "skills", "verify", "website", "computer_use", "hermes_state",
}

# Root-level modules whose tests sit directly in tests/ (no package to mirror).
_ROOT_MODULE_STEMS = {p.stem for p in REPO_ROOT.glob("*.py")}

_ISSUE_NUMBER = re.compile(r"(^test_\d{4,6}_)|(_\d{4,6}\.py$)")


def _source_dirs() -> set[str]:
    return {
        p.name
        for p in REPO_ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    }


def test_every_test_directory_mirrors_a_source_directory_or_is_declared():
    """No sibling directory for a source package that already has one."""
    source_dirs = _source_dirs()
    offenders = sorted(
        d.name
        for d in TESTS_ROOT.iterdir()
        if d.is_dir()
        and d.name not in ("__pycache__",)
        and d.name not in source_dirs
        and d.name not in _NON_MIRROR_DIRS
    )
    assert not offenders, (
        "tests/ directories that mirror no source package: "
        f"{offenders}. Put the tests under tests/<source dir>/ (tests/agent, "
        "tests/hermes_cli, ...) or, for a genuinely cross-cutting suite, add the "
        "name to _NON_MIRROR_DIRS in this file with a reason."
    )


def test_no_issue_numbers_in_test_filenames():
    """Issue numbers live in docstrings, where the reader gets context."""
    offenders = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in TESTS_ROOT.rglob("test_*.py")
        if _ISSUE_NUMBER.search(p.name)
    )
    assert not offenders, (
        f"Issue numbers in test filenames: {offenders}. Drop the number from the "
        "name and cite the issue in the module docstring instead."
    )
