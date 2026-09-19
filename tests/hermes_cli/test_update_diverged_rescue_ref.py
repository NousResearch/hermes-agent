"""`hermes update` must not drop local commits without leaving a way back.

When the checkout sits on the update's target branch and its history has
diverged, the update resets hard to ``origin/<branch>``. Divergence there has
two indistinguishable causes: an upstream force-push (nothing local is lost)
and local commits on that branch (everything is). The rescue ref that makes
the second case recoverable was written only for orphan divergence — a
disjoint history — which is the rarer of the two.
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
def diverged_checkout(tmp_path, monkeypatch):
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

    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    return checkout, local_sha


def _rescue_refs(checkout):
    out = _git(checkout, "for-each-ref", "--format=%(refname) %(objectname)",
               "refs/hermes-update-backups/").stdout
    return dict(line.split() for line in out.splitlines() if line.strip())


def test_reset_leaves_a_rescue_ref_for_the_discarded_commits(diverged_checkout):
    """The reset is fine; losing the only pointer to the local work is not."""
    checkout, local_sha = diverged_checkout

    update_cmd._reconcile_diverged_checkout(GIT, "main", local_sha)

    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    origin = _git(checkout, "rev-parse", "origin/main").stdout.strip()
    assert head == origin, "the reset itself must still happen"

    refs = _rescue_refs(checkout)
    assert local_sha in refs.values(), (
        "the discarded commit must stay reachable through a rescue ref; without one "
        f"it is recoverable only from the reflog. refs found: {refs}")


def test_the_rescue_ref_carries_the_whole_discarded_history(diverged_checkout):
    """A ref on the tip is enough: every dropped commit is reachable from it."""
    checkout, local_sha = diverged_checkout

    update_cmd._reconcile_diverged_checkout(GIT, "main", local_sha)

    ref = next(r for r, sha in _rescue_refs(checkout).items() if sha == local_sha)
    listed = _git(checkout, "log", "--format=%H", ref).stdout.split()
    assert local_sha in listed
    assert len(listed) >= 2, "the shared base must remain reachable from the rescue ref too"
