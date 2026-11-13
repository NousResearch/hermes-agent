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
