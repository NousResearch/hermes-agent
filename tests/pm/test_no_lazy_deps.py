"""Contract tests for scripts/ci/check_lazy_deps_imports.py.

``tools/lazy_deps.py`` was deleted by the pm migration; pm.extras
(available / ensure_import / ensure_and_bind) is the only lazy-install
surface. A ``tools.lazy_deps`` import left in production code is an
ImportError at call time.

Each test drives the real checker CLI (subprocess) against a fresh
per-test git fixture repo, so the tests are order-independent: the guard
must be red/green on both sides of the contract, and a repo it cannot
inventory (not a git repo, no tracked ``*.py`` files, unreadable or
unparseable tracked file) must FAIL rather than quietly scan nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_lazy_deps_imports.py"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A fresh minimal tracked git repo per test.

    ``git add`` is enough — the checker's inventory is ``git ls-files``,
    which reads the index, so no commit is made (or needed).
    """
    env = {**os.environ, **_GIT_ENV}

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True
        )

    _git("init", "-q")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    pass\n", encoding="utf-8"
    )
    _git("add", "-A")
    return tmp_path


def _run_checker(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _track(root: Path, relpath: str, body: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    subprocess.run(
        ["git", "add", relpath], cwd=root, check=True, capture_output=True
    )


def test_clean_repo_passes(fixture_repo):
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize(
    ("body", "lineno"),
    [
        ("from tools.lazy_deps import install_specs\n", 1),
        ("from tools import lazy_deps\n\nlazy_deps.ensure('x')\n", 1),
        ("from tools.lazy_deps.sub import x\n", 1),
        ("import tools.lazy_deps\n", 1),
        ("import tools.lazy_deps.submodule as s\n", 1),
        ("if True:\n    from tools.lazy_deps import ensure\n", 2),
    ],
    ids=[
        "from-module",
        "from-parent",
        "from-dotted-child",
        "plain-import",
        "dotted-import",
        "nested",
    ],
)
def test_each_import_form_is_caught_with_exact_site(fixture_repo, body, lineno):
    _track(fixture_repo, "app.py", body)
    result = _run_checker(fixture_repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"app.py:{lineno}" in result.stdout
    assert "lazy_deps" in result.stdout


@pytest.mark.parametrize(
    ("relpath", "body", "lineno"),
    [
        ("tools/helper.py", "from . import lazy_deps\n\nlazy_deps.ensure('x')\n", 1),
        ("tools/helper.py", "from .lazy_deps import ensure\n", 1),
        ("tools/sub/mod.py", "from .. import lazy_deps\n", 1),
        ("tools/__init__.py", "from . import lazy_deps\n", 1),
    ],
    ids=["from-dot", "from-dot-module", "from-dotdot-in-subpackage", "pkg-init"],
)
def test_relative_import_resolved_inside_tools_is_caught(
    fixture_repo, relpath, body, lineno
):
    """``from . import lazy_deps`` inside the tracked tools/ package IS an
    import of the deleted tools.lazy_deps module."""
    _track(fixture_repo, "tools/__init__.py", "")
    _track(fixture_repo, relpath, body)
    result = _run_checker(fixture_repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{relpath}:{lineno}" in result.stdout


def test_relative_import_outside_tools_is_not_flagged(fixture_repo):
    """A same-package ``lazy_deps`` elsewhere is not the deleted module."""
    _track(fixture_repo, "pkg/__init__.py", "")
    _track(fixture_repo, "pkg/other.py", "from . import lazy_deps\n")
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_comments_and_prose_are_not_imports(fixture_repo):
    _track(
        fixture_repo,
        "app.py",
        (
            "# tools.lazy_deps was deleted; see pm.extras\n"
            '\"\"\"Best-effort tools.lazy_deps.ensure: failures are swallowed.\"\"\"\n'
            "x = 1\n"
        ),
    )
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_untracked_files_are_not_scanned(fixture_repo):
    # Untracked debris (not in the tracked inventory) must not be scanned.
    (fixture_repo / "debris.py").write_text(
        "from tools.lazy_deps import ensure\n", encoding="utf-8"
    )
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_not_a_git_repo_fails_the_check(tmp_path):
    # The inventory cannot be built: the checker must fail loudly
    # (nonzero, inventory error) — never pass by scanning nothing.
    result = _run_checker(tmp_path)
    assert result.returncode == 2
    assert "inventory" in (result.stdout + result.stderr).lower()


def test_repo_without_tracked_python_fails_the_check(tmp_path):
    """A git repo whose tracked inventory contains no ``*.py`` files fails.

    The guard's policy is to fail rather than pass an empty inventory:
    for this repo a zero-candidate inventory means the scan is broken,
    not clean.
    """
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "inventory" in (result.stdout + result.stderr).lower()


def test_unparseable_tracked_file_fails_the_check(fixture_repo):
    # A tracked file that cannot be parsed is an error, not a skip: the
    # check must fail rather than silently scan a partial inventory.
    _track(fixture_repo, "broken.py", "def f(:\n")
    result = _run_checker(fixture_repo)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "broken.py" in (result.stdout + result.stderr)


def test_deleted_tracked_files_are_not_live_source(fixture_repo):
    _track(fixture_repo, "removed.py", "from tools import lazy_deps\n")
    (fixture_repo / "removed.py").unlink()
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_repo_invocation_stays_deterministic():
    """Drive the checker against THIS repo.

    Exit 0 = migration complete (stdout empty). Exit 1 = live offenders
    remain — output must be the deterministic exact-site list the
    lazy-deps-guard CI job blocks on (migration owned elsewhere; the
    fixture tests above, not this visibility check, are the red/green
    proof of the guard itself).
    """
    result = _run_checker(REPO_ROOT)
    assert result.returncode in (0, 1), result.stdout + result.stderr
    if result.returncode == 0:
        assert result.stdout.strip() == ""
        return
    for line in result.stdout.splitlines():
        path, lineno, rest = line.split(":", 2)
        assert path.endswith(".py") and lineno.isdigit()
        assert "lazy_deps" in rest
