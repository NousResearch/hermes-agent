"""Issue #101351 — the review-pane git helpers must never carry credentials.

Two shapes are covered:

1. A recorded ``origin`` URL that already embeds a PAT. ``git clone
   https://x-access-token:<token>@host/…`` writes the credential straight into
   ``.git/config``, and ``git remote -v`` / every later push echo it out. The
   review-pane helpers must scrub it before running any network verb.
2. Git's own stderr echoes the remote URL on a failed fetch/push. That text is
   what the helper raises, so it lands in the dashboard toast and in the
   gateway log — it must be scrubbed before it leaves the process.

Every test runs against a throwaway repo under ``tmp_path``; the credentialed
remote points at ``127.0.0.1:9`` (discard port) so a push fails instantly and
nothing touches the network or a real credential.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import git_credentials, plugins_cmd, web_git

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0"  # matches agent/redact.py's ghp_ PAT shape
CREDENTIALED = f"http://x-access-token:{TOKEN}@127.0.0.1:9/acme/private.git"
SCRUBBED = "http://127.0.0.1:9/acme/private.git"
CLEAN = "http://127.0.0.1:9/acme/private.git"


def _run(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")


def _committed_repo(tmp_path: Path, origin: str = CREDENTIALED) -> Path:
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _run(root, "remote", "add", "origin", origin)
    _run(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-qm", "init")
    return root


def _config(root: Path) -> str:
    return (root / ".git" / "config").read_text(encoding="utf-8")


# ── the shared helper ────────────────────────────────────────────────────────


def test_without_credentials_strips_userinfo_and_query() -> None:
    assert git_credentials.without_credentials(CREDENTIALED) == SCRUBBED
    assert git_credentials.without_credentials(
        "https://user:pw@github.com/o/r.git?token=abc#frag") == "https://github.com/o/r.git"
    # Non-HTTP transports and already-clean URLs are returned unchanged.
    assert git_credentials.without_credentials(CLEAN) == CLEAN
    assert git_credentials.without_credentials("git@github.com:o/r.git") == "git@github.com:o/r.git"
    assert git_credentials.without_credentials("") == ""


def test_plugins_cmd_shares_the_one_implementation() -> None:
    """One scrubber, not two: the plugin installer's helper delegates."""
    assert plugins_cmd._scrub_git_url(CREDENTIALED) == SCRUBBED
    assert plugins_cmd._scrub_git_url(CREDENTIALED) == git_credentials.without_credentials(CREDENTIALED)


# ── the review-pane helpers ──────────────────────────────────────────────────


def test_git_ok_scrubs_the_remote_url_out_of_raised_errors(monkeypatch) -> None:
    """git echoes the remote URL on failure; the raised message must not."""
    stderr = f"fatal: unable to access '{CREDENTIALED}/': Failed to connect to 127.0.0.1 port 9"
    monkeypatch.setattr(web_git, "_git", lambda cwd, args, **kw: (128, "", stderr))

    with pytest.raises(RuntimeError) as exc:
        web_git._git_ok("/tmp/repo", ["push"])

    message = str(exc.value)
    assert TOKEN not in message
    assert "x-access-token" not in message
    assert "Failed to connect" in message  # the diagnosis survives


def test_recorded_origin_credentials_are_scrubbed(tmp_path: Path) -> None:
    root = _committed_repo(tmp_path)
    assert TOKEN in _config(root)

    web_git._scrub_recorded_origin(str(root))

    assert TOKEN not in _config(root)
    assert "x-access-token" not in _config(root)
    assert web_git._git_line(str(root), ["remote", "get-url", "origin"]) == SCRUBBED

    # Idempotent: a second pass changes nothing.
    before = _config(root)
    web_git._scrub_recorded_origin(str(root))
    assert _config(root) == before


def test_recorded_origin_without_credentials_is_left_alone(tmp_path: Path) -> None:
    root = _committed_repo(tmp_path, origin=CLEAN)
    before = _config(root)

    web_git._scrub_recorded_origin(str(root))

    assert _config(root) == before


def test_review_push_never_persists_credentials(tmp_path: Path) -> None:
    """End to end: a failing push must not leave the PAT in .git/config nor in the error."""
    root = _committed_repo(tmp_path)

    with pytest.raises(RuntimeError) as exc:
        web_git.review_push(str(root))

    assert TOKEN not in str(exc.value)
    assert TOKEN not in _config(root)
    assert "x-access-token" not in _config(root)


def test_worktree_add_from_a_remote_branch_never_persists_credentials(tmp_path: Path) -> None:
    """The fetch path (worktree from ``origin/<branch>``) is the other network verb."""
    root = _committed_repo(tmp_path)
    _run(root, "update-ref", "refs/remotes/origin/feature", "HEAD")

    created = web_git.worktree_add(str(root), {"existingBranch": "origin/feature"})

    assert Path(created["path"]).is_dir()
    assert TOKEN not in _config(root)
    assert "x-access-token" not in _config(root)
