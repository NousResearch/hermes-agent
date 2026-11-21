"""Code-skew contracts through the real Git fingerprint reader (PR #137969)."""

import subprocess

import pytest

from gateway import code_skew


def _git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=Test",
         "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "module.py")
    _git(root, "commit", "-qm", "initial code")
    monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root)
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    return root


@pytest.mark.parametrize("transition", ["branch", "detach", "reattach"])
def test_same_commit_checkout_does_not_require_restart(checkout, transition):
    sha = _git(checkout, "rev-parse", "HEAD")
    if transition == "reattach":
        _git(checkout, "checkout", "-q", "--detach")
    boot_ref = "HEAD" if transition == "reattach" else "refs/heads/main"
    code_skew.record_boot_fingerprint()
    # A failed import/reader must not turn this into a vacuous no-skew success.
    assert code_skew._boot_fingerprint == f"git:{boot_ref}:{sha}"
    assert code_skew.current_code_sha() == sha
    assert code_skew.detect_code_skew() is None

    if transition == "branch":
        _git(checkout, "checkout", "-q", "-b", "topic")
        disk_ref = "refs/heads/topic"
    elif transition == "detach":
        _git(checkout, "checkout", "-q", "--detach")
        disk_ref = "HEAD"
    else:
        _git(checkout, "checkout", "-q", "main")
        disk_ref = "refs/heads/main"

    assert _git(checkout, "rev-parse", "HEAD") == sha
    assert code_skew._fingerprint() == f"git:{disk_ref}:{sha}"
    assert code_skew.current_code_sha() == sha
    assert code_skew.detect_code_skew() is None


@pytest.mark.parametrize("empty", [False, True], ids=["code-change", "empty-commit"])
def test_new_commit_still_requires_restart(checkout, empty):
    boot_sha = _git(checkout, "rev-parse", "HEAD")
    code_skew.record_boot_fingerprint()
    assert code_skew._boot_fingerprint == f"git:refs/heads/main:{boot_sha}"
    assert code_skew.current_code_sha() == boot_sha
    assert code_skew.detect_code_skew() is None

    if empty:
        _git(checkout, "commit", "-q", "--allow-empty", "-m", "new revision")
    else:
        (checkout / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        _git(checkout, "commit", "-qam", "new code")
    disk_sha = _git(checkout, "rev-parse", "HEAD")

    assert disk_sha != boot_sha
    assert code_skew._fingerprint() == f"git:refs/heads/main:{disk_sha}"
    # Cron compares this value with the gateway's commit SHA, never a tree ID.
    assert code_skew.current_code_sha() == disk_sha
    assert code_skew.detect_code_skew() == (boot_sha[:10], disk_sha[:10])
