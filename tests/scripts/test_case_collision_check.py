"""Wrappers for scripts/check-case-collisions.py.

Same pattern as tests/scripts/test_windows_footguns_full_repo_scan.py: run
the real checker and assert its outcomes, so a normal pytest run catches a
regression — someone committing a case-colliding pair — without anyone
having to remember to run the script by hand.

The collision cases are built with ``git update-index --cacheinfo`` (index
only, never touching the working tree), so they exercise the same index the
checker reads and work even on a case-insensitive filesystem, where the two
spellings cannot coexist on disk.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-case-collisions.py"


def _git_blob_sha(data: bytes) -> str:
    """The git object hash for a blob with ``data`` as its content."""
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _run_check(*args, root=None):
    cmd = [sys.executable, str(SCRIPT)] + list(args)
    if root is not None:
        cmd.append(str(root))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        cwd=REPO_ROOT,
    )


def _git_init(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


def test_full_repo_has_no_case_colliding_paths():
    """The real checker against the whole tracked tree must exit clean."""
    result = _run_check()
    assert result.returncode == 0, (
        f"Case-collision check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_detects_case_colliding_paths(tmp_path):
    """Same-directory Foo.txt + foo.txt must fail, naming both paths."""
    repo = _git_init(tmp_path)
    subprocess.run(
        [
            "git", "update-index", "--add", "--cacheinfo",
            f"100644,{_git_blob_sha(b'a')},Foo.txt",
        ],
        cwd=repo, check=True,
    )
    subprocess.run(
        [
            "git", "update-index", "--add", "--cacheinfo",
            f"100644,{_git_blob_sha(b'b')},foo.txt",
        ],
        cwd=repo, check=True,
    )

    result = _run_check(root=repo)
    assert result.returncode == 1, f"expected failure, got:\n{result.stdout}"
    assert "Foo.txt" in result.stdout
    assert "foo.txt" in result.stdout


def test_detects_directory_case_collisions(tmp_path):
    """The comparison is on the FULL path — dir/Foo.txt vs DIR/foo.txt too."""
    repo = _git_init(tmp_path)
    subprocess.run(
        [
            "git", "update-index", "--add", "--cacheinfo",
            f"100644,{_git_blob_sha(b'a')},src/Helper.py",
        ],
        cwd=repo, check=True,
    )
    subprocess.run(
        [
            "git", "update-index", "--add", "--cacheinfo",
            f"100644,{_git_blob_sha(b'b')},SRC/helper.py",
        ],
        cwd=repo, check=True,
    )

    result = _run_check(root=repo)
    assert result.returncode == 1, f"expected failure, got:\n{result.stdout}"
    assert "src/Helper.py" in result.stdout
    assert "SRC/helper.py" in result.stdout


def test_same_name_in_different_dirs_is_not_a_collision(tmp_path):
    """a/Readme.txt and b/readme.txt share a basename but not a path."""
    repo = _git_init(tmp_path)
    (repo / "a").mkdir()
    (repo / "b").mkdir()
    (repo / "a" / "Readme.txt").write_text("a", encoding="utf-8")
    (repo / "b" / "readme.txt").write_text("b", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    result = _run_check(root=repo)
    assert result.returncode == 0, f"expected clean, got:\n{result.stdout}"
