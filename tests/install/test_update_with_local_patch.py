"""The install path must name local work before an update overwrites it.

``stage_repository`` moves the tree twice: a ``checkout`` that refuses a
conflicting working copy, then ``reset --hard origin/<branch>`` for whatever
could not fast-forward. Between them, anything the operator wrote to
``~/.hermes/hermes-agent`` is gone, and there is no later point at which they
could still see it.

``git status --porcelain`` covers two of the three shapes that matter -- edited
and untracked files. A *committed* patch on a branch no remote knows about
reports an empty porcelain, so a guard that reads only porcelain is blind to
exactly the work an update would clobber. That is how the RUAAAAA-5212
compaction run-budget patch stayed invisible: clean tree, one local commit, no
signal anywhere. ``local_patch_inventory`` counts those commits explicitly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(repo: Path, name: str, message: str | None = None) -> str:
    (repo / name).write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message or f"add {name}")
    return _git(repo, "rev-parse", "HEAD")


def _managed_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A clean checkout of ``main`` with an upstream that has moved on."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "shared.txt")
    managed = tmp_path / "hermes-agent"
    _git(tmp_path, "clone", "-q", str(upstream), str(managed))
    return managed, upstream


def _run_repository_stage(managed: Path, tmp_path: Path, **extra) -> subprocess.CompletedProcess:
    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }
    return subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "repository", "--non-interactive", *extra],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )


def _output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


@pytest.mark.live_system_guard_bypass
@needs_git
def test_clean_checkout_prints_no_maintenance_step(tmp_path: Path) -> None:
    """No local work: the step must be invisible, not a new banner on every install."""
    managed, upstream = _managed_checkout(tmp_path)
    _commit(upstream, "upstream-only.txt")

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, _output(result)
    output = _output(result)
    assert "carries local work" not in output
    assert "Local patch maintenance" not in output
    assert _git(managed, "rev-parse", "HEAD") == _git(managed, "rev-parse", "origin/main")


@pytest.mark.live_system_guard_bypass
@needs_git
def test_local_modification_is_named_before_the_tree_moves(tmp_path: Path) -> None:
    """An edited file is named with its kind, and survives the update."""
    managed, upstream = _managed_checkout(tmp_path)
    _commit(upstream, "upstream-only.txt")
    (managed / "shared.txt").write_text("local edit\n", encoding="utf-8")

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, _output(result)
    output = _output(result)
    assert "carries local work" in output
    assert "modified shared.txt" in output
    # The disposition is in the install log, so the next operator can tell a
    # parked patch from a dropped one without re-deriving it from the tree.
    assert "Local patch maintenance: stashed as hermes-install-autostash-" in output
    stashed = _git(managed, "stash", "list")
    assert "hermes-install-autostash-" in stashed, stashed


@pytest.mark.live_system_guard_bypass
@needs_git
def test_untracked_file_is_named_before_the_tree_moves(tmp_path: Path) -> None:
    """An untracked file is named too -- and parked, not merely reported."""
    managed, upstream = _managed_checkout(tmp_path)
    _commit(upstream, "upstream-only.txt")
    (managed / "notes.tmp").write_text("scratch\n", encoding="utf-8")

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, _output(result)
    output = _output(result)
    assert "untracked notes.tmp" in output
    assert "hermes-install-autostash-" in _git(managed, "stash", "list")


@pytest.mark.live_system_guard_bypass
@needs_git
def test_committed_local_patch_is_named_though_porcelain_is_empty(tmp_path: Path) -> None:
    """The RUAAAAA-5212 shape: clean tree, one local commit, porcelain empty.

    This is the regression the step exists for. A porcelain-only guard sees an
    empty string here and walks straight past the patch into a ``reset --hard``.
    """
    managed, _ = _managed_checkout(tmp_path)
    _git(managed, "checkout", "-q", "-b", "fix/ruaaaaa-5212-compaction-run-budget-cap")
    patch_sha = _commit(managed, "agent_patch.py", "compaction: never outlive the turn")

    assert _git(managed, "status", "--porcelain") == "", "precondition: the tree is clean"

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, _output(result)
    output = _output(result)
    assert (
        "branch: fix/ruaaaaa-5212-compaction-run-budget-cap (1 local commit(s))" in output
    ), output
    # Named, and still reachable after the tree move.
    assert patch_sha in _git(managed, "log", "--format=%H", "--all")
    assert "no uncommitted changes" in output


@pytest.mark.live_system_guard_bypass
@needs_git
@pytest.mark.skipif(shutil.which("script") is None, reason="needs util-linux script(1)")
def test_abort_leaves_the_working_copy_untouched(tmp_path: Path) -> None:
    """With a terminal to answer, `abort` stops the run and parks nothing.

    Driven through ``script(1)`` so the installer finds a real terminal on
    /dev/tty -- under ``curl | bash`` stdin is the script itself, which is why
    the prompt reads the tty. Without that device the installer takes the
    stash-and-continue fallback instead of asking, so this path is only
    reachable on a terminal.
    """
    managed, _ = _managed_checkout(tmp_path)
    (managed / "shared.txt").write_text("local edit\n", encoding="utf-8")

    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }
    # script(1) allocates the pty and relays both streams to stdout. Its own
    # stdin is what reaches the pty master, and /dev/tty is what the installer
    # reads the answer from -- piping the answer into the installer's stdin
    # instead would be ignored (and is exactly why it reads the tty at all).
    proc = subprocess.run(
        ["script", "-qec", f"bash {INSTALL_SH} --stage repository", "/dev/null"],
        cwd=tmp_path, env=env, input="a\n", capture_output=True, text=True, check=False,
    )
    output = _output(proc)

    assert "carries local work" in output, output
    assert "modified shared.txt" in output
    assert "abort" in output.lower()
    # The run stopped rather than proceeding to the tree move.
    assert proc.returncode != 0, output
    # Nothing parked: the working copy is still exactly the operator's.
    assert "hermes-install-autostash-" not in _git(managed, "stash", "list")
    assert (managed / "shared.txt").read_text(encoding="utf-8") == "local edit\n"
    assert _git(managed, "rev-parse", "--abbrev-ref", "HEAD") == "main"