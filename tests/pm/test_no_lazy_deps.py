"""Contract tests for scripts/ci/check_lazy_deps_imports.py.

``tools/lazy_deps.py`` survives only as an old-updater stub; pm.extras
(available / ensure_import / ensure_and_bind) is the only lazy-install
surface. A ``tools.lazy_deps`` import left in production code never gets
its dependencies.

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



@pytest.mark.parametrize("relpath, body, code, site", [
    ("app.py", "from tools.lazy_deps import install_specs\n", 1, "app.py:1"),
    ("app.py", "from tools import lazy_deps\n", 1, "app.py:1"),
    ("app.py", "from tools.lazy_deps.sub import x\n", 1, "app.py:1"),
    ("app.py", "import tools.lazy_deps\n", 1, "app.py:1"),
    ("app.py", "import tools.lazy_deps.submodule as s\n", 1, "app.py:1"),
    ("app.py", "if True:\n    from tools.lazy_deps import ensure\n", 1, "app.py:2"),
    ("tools/helper.py", "from . import lazy_deps\n", 1, "tools/helper.py:1"),
    ("tools/helper.py", "from .lazy_deps import ensure\n", 1, "tools/helper.py:1"),
    ("tools/sub/mod.py", "from .. import lazy_deps\n", 1, "tools/sub/mod.py:1"),
    ("tools/__init__.py", "from . import lazy_deps\n", 1, "tools/__init__.py:1"),
    ("pkg/other.py", "from . import lazy_deps\n", 0, ""),
    ("app.py", '# tools.lazy_deps\n"""tools.lazy_deps.ensure"""\nx=1\n', 0, ""),
    ("broken.py", "def f(:\n", 2, "broken.py"),
    ("app.py", "x = 1\n", 0, ""),
])
def test_checker_outcomes(fixture_repo, relpath, body, code, site):
    _track(fixture_repo, "tools/__init__.py", "")
    _track(fixture_repo, "pkg/__init__.py", "")
    _track(fixture_repo, relpath, body)
    result = _run_checker(fixture_repo)
    assert result.returncode == code, result.stdout + result.stderr
    assert site in result.stdout + result.stderr
    if code == 0:
        assert result.stdout.strip() == ""





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



def test_deleted_tracked_files_are_not_live_source(fixture_repo):
    _track(fixture_repo, "removed.py", "from tools import lazy_deps\n")
    (fixture_repo / "removed.py").unlink()
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr
