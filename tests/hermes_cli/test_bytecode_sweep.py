"""Tests for the launch-time stale-bytecode sweep (checkout fingerprint guard).

Bug class: the checkout's ``.py`` files change (``hermes update``, manual
``git pull``, ZIP update) while ``__pycache__`` retains bytecode compiled
from the previous revision; the next process to import trusts the stale
``.pyc`` and dies with ``cannot import name ...`` (#6207, #60242).

The launch-time guard compares the current checkout fingerprint against the
last-validated stamp and sweeps ``__pycache__`` once when they diverge —
covering paths no update-time clear can reach (manual pulls, pre-hardening
updaters).
"""

from pathlib import Path

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import main_web_build


def _make_repo(tmp_path: Path, sha: str = "a" * 40) -> Path:
    """Minimal git checkout layout that _read_git_revision_fingerprint groks."""
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    return repo


def _make_pycache(repo: Path, subdir: str = "hermes_cli") -> Path:
    cache = repo / subdir / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "main.cpython-311.pyc").write_bytes(b"stale")
    return cache


def test_sweep_clears_pycache_when_checkout_changed(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, sha="b" * 40)
    cache = _make_pycache(repo)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    # Stamp records a different (older) fingerprint.
    (repo / main_web_build._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "a" * 40, encoding="utf-8"
    )

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not cache.exists()
    # Stamp updated to the current fingerprint.
    recorded = (repo / main_web_build._BYTECODE_FINGERPRINT_FILE).read_text(encoding="utf-8")
    assert recorded.strip().endswith("b" * 40)







# ---------------------------------------------------------------------------
# Plugin-update sibling site: __pycache__ under ~/.hermes/plugins/<name>
# ---------------------------------------------------------------------------

def test_clear_plugin_bytecode_removes_nested_caches(tmp_path):
    from hermes_cli import plugins_cmd

    plugin = tmp_path / "myplugin"
    top = plugin / "__pycache__"
    nested = plugin / "sub" / "__pycache__"
    top.mkdir(parents=True)
    nested.mkdir(parents=True)
    (top / "a.pyc").write_bytes(b"stale")
    (nested / "b.pyc").write_bytes(b"stale")

    removed = plugins_cmd._clear_plugin_bytecode(plugin)

    assert removed == 2
    assert not top.exists()
    assert not nested.exists()




def test_a_package_the_checkout_moved_leaves_no_directory_behind(monkeypatch, tmp_path):
    """Git removes a moved package's files but keeps its directory while ``__pycache__`` is in it;
    the sweep emptied it, and the empty directory imported as a namespace package that hid the
    package's new location. A directory that still holds a user's file is kept."""
    import subprocess

    repo = tmp_path / "repo"

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    (repo / "pkg" / "sub").mkdir(parents=True)
    (repo / "notes").mkdir()
    for rel in ("pkg/__init__.py", "pkg/sub/__init__.py", "notes/__init__.py", "kept.py"):
        (repo / rel).write_text("x = 1\n", encoding="utf-8")
    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("add", ".")
    git("commit", "-qm", "flat")
    flat = git("rev-parse", "HEAD")
    (repo / "src").mkdir()
    git("mv", "pkg", "src/pkg")
    git("rm", "-q", "notes/__init__.py")
    git("commit", "-qm", "move pkg under src")
    git("checkout", "-q", "--detach", flat)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    hermes_main._sweep_stale_bytecode_if_checkout_changed()
    for cache in ("pkg/__pycache__", "pkg/sub/__pycache__", "notes/__pycache__"):
        (repo / cache).mkdir()
        (repo / cache / "m.cpython-311.pyc").write_bytes(b"stale")
    (repo / "notes" / "mine.txt").write_text("user file\n", encoding="utf-8")
    git("checkout", "-q", "main")
    assert (repo / "pkg" / "sub").is_dir(), "sanity: git leaves the directory while __pycache__ is in it"

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not (repo / "pkg").exists()
    assert (repo / "notes" / "mine.txt").is_file()
    assert (repo / "src" / "pkg" / "sub" / "__init__.py").is_file()


def test_a_sweep_that_cannot_list_dropped_files_keeps_the_old_fingerprint(monkeypatch, tmp_path):
    """A failed ``git diff`` used to count as "nothing to prune", and the new fingerprint then hid the
    old commit from every later sweep. The old fingerprint stays, so the next launch retries."""
    import subprocess

    from hermes_cli import main_web_build as build

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@e.invalid", "-c", "user.name=T", "commit", "-q",
                    "--allow-empty", "-m", "c"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()
    recorded = f"git:refs/heads/main:{head}"
    (repo / build._BYTECODE_FINGERPRINT_FILE).write_text(recorded, encoding="utf-8")
    (repo / ".git" / "refs" / "heads" / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert (repo / build._BYTECODE_FINGERPRINT_FILE).read_text(encoding="utf-8") == recorded


@pytest.mark.platforms("linux")
def test_dropped_directory_keeps_exact_filename_bytes(monkeypatch, tmp_path):
    import os
    import subprocess

    repo = tmp_path / "repo"
    dropped = repo / os.fsdecode(b"old-\xff")
    dropped.mkdir(parents=True)
    (dropped / "__init__.py").write_text("", encoding="utf-8")

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, timeout=30)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("add", ".")
    git("commit", "-qm", "before")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    hermes_main._sweep_stale_bytecode_if_checkout_changed()
    _make_pycache(repo, dropped.name)
    git("rm", "--", f"{dropped.name}/__init__.py")
    git("commit", "-qm", "drop package")
    unrelated = repo / "old-\ufffd"
    unrelated.mkdir()

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert unrelated.is_dir(), "lossy decoding must not target a different directory"
    assert not dropped.exists()
