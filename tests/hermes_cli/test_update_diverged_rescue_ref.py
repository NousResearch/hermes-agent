"""`hermes update` must not drop local commits without leaving a named way back.

When the checkout sits on the update's target branch and its history has
diverged, the update resets hard to ``origin/<branch>``. Divergence there has
two indistinguishable causes: an upstream force-push (nothing local is lost)
and local commits on that branch (everything is). The update must park the old
HEAD under ``refs/hermes-update-backups/`` first and tell the user the ref name.
The installer update paths are covered in
``tests/scripts/install/test_install_diverged_rescue_ref.py``.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd


GIT = ["git"]


def _git(repo, *args, check=True):
    return subprocess.run(
        GIT + list(args), cwd=repo, capture_output=True, text=True, check=check)


def _commit(repo, name, text):
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", f"add {name}")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def diverged_checkout(tmp_path):
    """A checkout on ``main`` carrying a local commit its ``origin/main`` does not have."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "shared.txt", "shared\n")
    _commit(upstream, "upstream-only.txt", "upstream\n")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", str(upstream), str(checkout))
    _git(checkout, "reset", "-q", "--hard", "HEAD~1")          # back to the shared commit
    local_sha = _commit(checkout, "local-fix.txt", "local\n")  # diverges from origin/main
    return checkout, local_sha


def _rescue_refs(checkout):
    out = _git(checkout, "for-each-ref", "--format=%(refname) %(objectname)",
               "refs/hermes-update-backups/").stdout
    return dict(line.split() for line in out.splitlines() if line.strip())


def _assert_reset_kept_local_commit(checkout, local_sha, output):
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    origin = _git(checkout, "rev-parse", "origin/main").stdout.strip()
    assert head == origin, "the reset itself must still happen"
    refs = [ref for ref, sha in _rescue_refs(checkout).items() if sha == local_sha]
    assert len(refs) == 1, f"the dropped commit needs exactly one rescue ref: {_rescue_refs(checkout)}"
    assert refs[0].startswith("refs/hermes-update-backups/diverged-main-")
    assert refs[0] in output, "the user must be told where the commits went"
    dropped = _git(checkout, "log", "--format=%H", f"origin/main..{refs[0]}").stdout.split()
    assert dropped == [local_sha]


def test_hermes_update_keeps_local_commit_behind_a_rescue_ref(
        diverged_checkout, monkeypatch, capsys):
    """The real apply path: ff-only fails, the reconcile resets, the local commit stays reachable."""
    checkout, local_sha = diverged_checkout
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)

    update_cmd._pull_updates(
        GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
        discard_local_changes=False, keep_stash=False)

    out = capsys.readouterr().out
    _assert_reset_kept_local_commit(checkout, local_sha, out)
    assert "1 commit(s) not on origin/main" in out
