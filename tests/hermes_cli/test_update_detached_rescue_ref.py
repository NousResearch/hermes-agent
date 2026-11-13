"""`hermes update` must not orphan commits made on a detached HEAD.

Both update shapes move HEAD off a detached commit: the branch update checks out ``main``, the
release update checks out the release commit. A commit made there is on no branch, so after the
move only the expiring reflog reaches it. The update must park it under
``refs/hermes-update-backups/detached-...`` first and print the ref name. The installed-product
proof is the e2e cell
``tests/e2e/core/upgrade/git/test_foreign_state.py::test_work_committed_on_a_detached_head_stays_reachable``.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd

GIT = ["git"]


def _git(repo, *args, check=True):
    return subprocess.run(GIT + list(args), cwd=repo, capture_output=True, text=True, check=check)


def _commit(repo, name, text):
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", f"add {name}")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def detached_work(tmp_path, monkeypatch):
    """A checkout detached at ``origin/main~1`` carrying a commit that no ref contains."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "shared.txt", "shared\n")
    release = _commit(upstream, "release.txt", "release\n")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", str(upstream), str(checkout))
    _git(checkout, "checkout", "-q", "--detach", "HEAD~1")
    work = _commit(checkout, "detached-work.txt", "work\n")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    return checkout, work, release


def _assert_parked(checkout, work, output):
    refs = _git(checkout, "for-each-ref", "--contains", work, "--format=%(refname)").stdout.split()
    assert len(refs) == 1 and refs[0].startswith("refs/hermes-update-backups/detached-main-"), \
        f"the detached commit must be kept by exactly one rescue ref, got {refs}\n{output}"
    assert refs[0] in output, f"the user must be told where the commit went:\n{output}"
    assert "1 commit(s) made on the detached HEAD" in output, output


def test_branch_update_from_detached_head_parks_the_work(detached_work, capsys):
    """The branch shape: detached HEAD -> checkout main, with dirty edits riding in the autostash
    (whose refs/stash contains HEAD until it is dropped, so it must not count as keeping it)."""
    checkout, work, _release = detached_work
    (checkout / "shared.txt").write_text("edited\n", encoding="utf-8")

    plan = update_cmd._prepare_checkout_for_update(
        GIT, "main", "HEAD", is_fork=False, assume_yes=True, gateway_mode=False, gw_input_fn=None,
        switch_branch=False, _windows_gateway_resume=None)
    if plan.auto_stash_ref is not None:
        update_cmd._restore_stashed_changes(GIT, checkout, plan.auto_stash_ref, prompt_user=False)

    assert _git(checkout, "branch", "--show-current").stdout.strip() == "main"
    _assert_parked(checkout, work, capsys.readouterr().out)


def test_release_update_from_detached_head_parks_only_unreachable_work(detached_work, capsys):
    """The release shape moves HEAD with checkout --detach; a detached HEAD that is already on a
    branch or tag (a pinned release) needs no backup and gets none."""
    checkout, work, release = detached_work

    update_cmd._pull_updates(GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False, target_ref=release)

    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == release
    _assert_parked(checkout, work, capsys.readouterr().out)

    _git(checkout, "checkout", "-q", "--detach", "origin/main~1")
    update_cmd._pull_updates(GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False, target_ref=release)
    backups = _git(checkout, "for-each-ref", "--format=%(refname)", "refs/hermes-update-backups/").stdout.split()
    assert len(backups) == 1, f"a detached HEAD already on a branch must not get a backup ref: {backups}"


# ---------------------------------------------------------------------------
# #148947: the release channel skips the parked-branch guard (branch policy is
# main-only), so a parked checkout's carried commits must at least be named —
# and commits a detached HEAD would orphan must get the same rescue ref the
# branch channel writes.
# ---------------------------------------------------------------------------

@pytest.fixture()
def parked_branch_work(tmp_path, monkeypatch):
    """A checkout parked on branch ``carried`` (one commit the release lacks), upstream main
    two commits ahead. Returns (checkout, carried_sha, release_sha)."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "shared.txt", "shared\n")
    release = _commit(upstream, "release.txt", "release\n")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", str(upstream), str(checkout))
    _git(checkout, "checkout", "-qb", "carried", "HEAD~1")
    carried = _commit(checkout, "carried.txt", "carried work\n")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    return checkout, carried, release


def _release_plan(checkout, current_branch, target_ref):
    return update_cmd._prepare_checkout_for_update(
        GIT, "main", current_branch, is_fork=False, assume_yes=True, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, target_ref=target_ref,
        _windows_gateway_resume=None)


def test_release_plan_warns_parked_branch_carried_commits(parked_branch_work, capsys):
    """FAIL-BEFORE: a release update from a parked branch carrying commits the release lacks
    printed success and moved the working tree past them with no word — the strategy key never
    applied on this channel and nothing said so (#148947). Now the landing names the branch and
    the count before anything moves."""
    checkout, _carried, release = parked_branch_work

    _release_plan(checkout, "carried", release)

    out = capsys.readouterr().out
    assert "branch 'carried' carries 1 commit(s) not in this release" in out, out
    assert "updates.parked_branch_strategy" in out, out
    assert "will NOT be in the working tree" in out, out
    # The branch and its commit are untouched — the warning names, never rewrites.
    assert _git(checkout, "rev-parse", "carried").stdout.strip() == _carried


def test_release_plan_parks_unreachable_detached_commits(detached_work, capsys):
    """The release channel skips the branch guard, so it must give a detached HEAD the same
    orphan rescue the branch channel does before landing on the release commit."""
    checkout, work, release = detached_work

    _release_plan(checkout, "HEAD", release)

    out = capsys.readouterr().out
    _assert_parked(checkout, work, out)
    assert "the detached HEAD carries 1 commit(s) not in this release" in out, out


def test_release_plan_stays_silent_for_pinned_release(detached_work, capsys):
    """The normal release-to-release shape (detached at an ancestor release, nothing carried)
    must not grow a spurious warning or a rescue ref."""
    checkout, _work, release = detached_work
    _git(checkout, "checkout", "-q", "--detach", "HEAD~1")  # back to the pinned release commit

    _release_plan(checkout, "HEAD", release)

    out = capsys.readouterr().out
    assert "carries" not in out, out
    backups = _git(checkout, "for-each-ref", "--format=%(refname)", "refs/hermes-update-backups/").stdout.split()
    assert backups == [], f"an ancestor release needs no rescue ref: {backups}"
