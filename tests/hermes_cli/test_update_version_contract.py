"""E2E contract for the release-update git plumbing, against real temporary repositories.

Contract under test (``hermes update --version``):
- only official CalVer release shapes (vYYYY.M.D[.N]) are accepted; refspecs, globs, branch
  names, zero-padded or partial dates are rejected locally
- the fetch pulls exactly ONE canonical tag with --no-tags (no auto-following) and a forced
  refspec, so a counterfeit local tag of the same name is replaced by the official one
- the tag must peel to a commit (``tag^{commit}``); tags of trees/blobs never reach checkout
- an attached branch sitting at the release commit still requires a (detached) checkout
- release mode NEVER autostashes: the clean-tree preflight is the contract, re-checked at
  the last moment before the checkout; a tree dirtied after the first preflight refuses
  rather than being absorbed, and a non-None autostash reaching the release path fails
  closed before the checkout
- ignored user files the target release would begin tracking block the update BEFORE the
  checkout (``git checkout`` silently overwrites ignored files by default); the colliding
  bytes survive byte-for-byte and the checkout never occurs. Paths are compared literally —
  no glob/pathspec magic — and unrelated ignored directories (venv/, node_modules/) never
  block. ``--no-overwrite-ignore`` backstops the probe: even a blind probe cannot let git
  clobber an ignored file
- the checkout is transactional through the post-checkout verification: a checkout that does
  not land detached exactly at the resolved release commit (e.g. the tag moved between
  resolve and checkout) rolls back, as does a critical-file syntax failure
- the transaction is exception-safe: an exception raised from the post-checkout verification
  or the critical-file syntax validation (e.g. an OSError from TemporaryDirectory) rolls back
  and re-raises — HEAD is never stranded detached at the release
- rollback restores ONLY the starting SHA and the attached/detached identity, verified; it
  never stashes, deletes, or cleans user paths (preflight guaranteed a clean tree)
- rollback NEVER rewrites the starting branch ref: if a concurrent git actor advanced,
  deleted, or corrupted ``refs/heads/<start_branch>`` after the (detached) release checkout,
  restoration fails closed with the branch — and the concurrent commits — untouched, and the
  caller reports failure with static guidance instead of claiming an exact restoration
- rollback never force-checkouts: immediately before the rollback checkout it re-probes
  index/worktree cleanliness (untracked included) and fails closed on any concurrent
  staged/tracked/untracked edit — or an unverifiable probe — leaving HEAD detached at the
  release with every concurrent byte untouched; a non-force checkout (with
  ``--no-overwrite-ignore``) backstops edits that land after the probe
- identity restoration accepts ONLY ``symbolic-ref`` return code 1 as proof of a detached
  HEAD; 128 or any other probe failure fails the restoration instead of claiming detached
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def _commit(repo: Path, message: str, content: str) -> str:
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    return repo


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "release;touch",
        "refs/tags/v2026.7.30",
        "backup/pre-release",
        "main",
        "v2026.07.30",  # zero-padded month
        "v2026.7.07",   # zero-padded day
        "v2026.7",      # partial date
        "2026.7.30",    # missing v prefix
        "v2026.7.30.0", # respin counter starts at 1
        "v2026.13.1",   # no thirteenth month
        "v2026.7.32",   # no 32nd day
        "v2026.2.29",   # 2026 is not a leap year
        "v2024.2.30",   # February never has 30 days
        "v2026.4.31",   # April has 30 days
        "v2026.2.29.2", # a respin cannot resurrect an impossible date
    ],
)
def test_official_release_tag_rejects_non_release_values(value):
    with pytest.raises(ValueError):
        update_cmd._official_release_tag(value)


@pytest.mark.parametrize(
    "value",
    ["v2026.7.30", "v2026.7.7.2", "v2030.12.31", "v2024.2.29", "v2024.2.29.3"],
)
def test_official_release_tag_accepts_release_shape(value):
    assert update_cmd._official_release_tag(value) == value


def test_fetch_official_release_is_exact_and_does_not_auto_follow_tags(monkeypatch, tmp_path):
    source = _repo(tmp_path)
    first = _commit(source, "first", "first")
    _git(source, "tag", "-a", "v2026.7.1", "-m", "first release", first)
    second = _commit(source, "second", "second")
    _git(source, "tag", "-a", "v2026.7.2", "-m", "second release", second)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    result = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.7.2")

    assert result.returncode == 0
    assert _git(checkout, "tag", "--list").stdout.splitlines() == ["v2026.7.2"]
    assert update_cmd._resolve_release_commit(["git"], checkout, "v2026.7.2") == second


def test_official_fetch_replaces_counterfeit_local_release_tag(monkeypatch, tmp_path):
    source = _repo(tmp_path)
    official_sha = _commit(source, "official", "official")
    _git(source, "tag", "v2026.8.1", official_sha)

    checkout = _repo(tmp_path, "checkout")
    counterfeit_sha = _commit(checkout, "counterfeit", "counterfeit")
    _git(checkout, "tag", "v2026.8.1", counterfeit_sha)
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    result = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.8.1")

    assert result.returncode == 0
    assert update_cmd._resolve_release_commit(["git"], checkout, "v2026.8.1") == official_sha


def test_non_commit_release_tag_is_rejected_before_checkout(monkeypatch, tmp_path):
    source = _repo(tmp_path)
    _commit(source, "first", "first")
    tree = _git(source, "rev-parse", "HEAD^{tree}").stdout.strip()
    _git(source, "tag", "-a", "v2026.7.30", "-m", "tree tag", tree)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    fetch = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.7.30")
    assert fetch.returncode == 0
    assert update_cmd._resolve_release_commit(["git"], checkout, "v2026.7.30") is None


def test_attached_checkout_at_release_commit_still_requires_detach():
    assert update_cmd._release_checkout_required(
        current_branch="main", head_sha="same", release_sha="same")
    assert not update_cmd._release_checkout_required(
        current_branch="HEAD", head_sha="same", release_sha="same")


@pytest.mark.parametrize("start_detached", [False, True])
def test_restore_checkout_identity_preserves_attached_or_detached_start(tmp_path, start_detached):
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "start")
    start_branch = "HEAD" if start_detached else "main"

    # The release checkout is always detached, so the starting branch never moves during
    # the transaction: park HEAD on a detached commit without touching main.
    _git(repo, "checkout", "--detach", start_sha)
    _commit(repo, "other", "other")

    assert update_cmd._restore_checkout_identity(["git"], repo, start_branch, start_sha)
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == start_sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == start_branch


def _detached_release_with_start_branch_at(
        tmp_path: Path) -> tuple[Path, str, str]:
    """A repo mid-rollback: HEAD detached at a release commit, main still at the start:
    returns ``(repo, start_sha, release_sha)``."""
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "start")
    _git(repo, "checkout", "--detach", start_sha)
    release_sha = _commit(repo, "release", "release")
    return repo, start_sha, release_sha


def _assert_left_detached_at(repo: Path, sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "HEAD"


def test_restore_checkout_identity_refuses_to_rewind_concurrently_advanced_branch(tmp_path):
    """Another git actor advanced the starting branch while HEAD was detached at the
    release. Rollback must fail closed: no checkout/reset of the branch, the concurrent
    commit stays at the branch tip, and the release checkout is left in place."""
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "start")
    concurrent_sha = _commit(repo, "concurrent", "concurrent work")
    _git(repo, "checkout", "--detach", start_sha)
    release_sha = _commit(repo, "release", "release")

    assert not update_cmd._restore_checkout_identity(["git"], repo, "main", start_sha)
    assert _git(repo, "rev-parse", "refs/heads/main").stdout.strip() == concurrent_sha
    assert _git(repo, "show", "refs/heads/main:tracked.txt").stdout == "concurrent work"
    _assert_left_detached_at(repo, release_sha)


def test_restore_checkout_identity_fails_closed_when_starting_branch_was_deleted(tmp_path):
    """The starting branch was deleted during the transaction: rollback must not recreate
    or update it, and must report failure instead of claiming restoration."""
    repo, start_sha, release_sha = _detached_release_with_start_branch_at(tmp_path)
    _git(repo, "branch", "-D", "main")

    assert not update_cmd._restore_checkout_identity(["git"], repo, "main", start_sha)
    assert _git(repo, "rev-parse", "--verify", "--quiet", "refs/heads/main",
                check=False).returncode != 0
    _assert_left_detached_at(repo, release_sha)


def test_restore_checkout_identity_fails_closed_on_unverifiable_branch_ref(tmp_path):
    """A starting-branch ref git itself cannot resolve (corrupt loose ref) means the
    captured start SHA cannot be verified: rollback must fail without rewriting the ref."""
    repo, start_sha, release_sha = _detached_release_with_start_branch_at(tmp_path)
    ref_file = repo / ".git" / "refs" / "heads" / "main"
    assert ref_file.is_file()
    ref_file.write_text("garbage-not-a-sha\n", encoding="utf-8")

    assert not update_cmd._restore_checkout_identity(["git"], repo, "main", start_sha)
    assert ref_file.read_text(encoding="utf-8") == "garbage-not-a-sha\n"
    _assert_left_detached_at(repo, release_sha)


def test_restore_checkout_identity_fails_closed_on_unverifiable_worktree_probe(
        monkeypatch, tmp_path):
    """A cleanliness re-probe git cannot answer means concurrent edits cannot be ruled out:
    restoration must fail closed WITHOUT running any rollback checkout."""
    repo, start_sha, release_sha = _detached_release_with_start_branch_at(tmp_path)

    real_git_run = update_cmd._git_run

    def failing_status(git_cmd, args, cwd=None, **kwargs):
        if args and args[0] == "status":
            return subprocess.CompletedProcess(
                git_cmd + args, 128, stdout="", stderr="fatal: unable to read the index")
        assert args[0] != "checkout", "no rollback checkout may run on an unverifiable probe"
        return real_git_run(git_cmd, args, cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", failing_status)

    assert not update_cmd._restore_checkout_identity(["git"], repo, "main", start_sha)
    _assert_left_detached_at(repo, release_sha)


def test_non_force_checkout_backstops_a_blind_rollback_cleanliness_probe(
        monkeypatch, tmp_path):
    """Race window: a concurrent edit landing between the cleanliness re-probe and the
    rollback checkout (simulated by a blind probe) must be stopped by the checkout itself —
    non-force, so git refuses the overwrite, the concurrent bytes survive, HEAD stays at
    the release, and no restoration is claimed."""
    repo, start_sha, release_sha = _detached_release_with_start_branch_at(tmp_path)
    payload = b"concurrent bytes written after the probe \xf0\x9f\x9a\xa7\n"
    (repo / "tracked.txt").write_bytes(payload)

    real_git_run = update_cmd._git_run

    def blind_status(git_cmd, args, cwd=None, **kwargs):
        if args and args[0] == "status":
            return subprocess.CompletedProcess(git_cmd + args, 0, stdout="", stderr="")
        return real_git_run(git_cmd, args, cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", blind_status)

    assert not update_cmd._restore_checkout_identity(["git"], repo, "main", start_sha)
    assert (repo / "tracked.txt").read_bytes() == payload
    _assert_left_detached_at(repo, release_sha)
    assert _git(repo, "rev-parse", "refs/heads/main").stdout.strip() == start_sha


@pytest.mark.parametrize("probe_rc", [128, 2])
def test_restore_checkout_identity_rejects_errored_detached_probe(monkeypatch, tmp_path, probe_rc):
    """Only ``symbolic-ref`` return code 1 proves a detached HEAD; 128 (repository error) or
    any other failure means the identity could not be verified — restoration must fail rather
    than claim the checkout ended detached."""
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "start")
    other_sha = _commit(repo, "other", "other")
    _git(repo, "checkout", "--detach", other_sha)

    real_git_run = update_cmd._git_run

    def failing_symbolic_ref(git_cmd, args, cwd=None, **kwargs):
        if args and args[0] == "symbolic-ref":
            return subprocess.CompletedProcess(
                git_cmd + args, probe_rc, stdout="", stderr="fatal: unable to read HEAD")
        return real_git_run(git_cmd, args, cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", failing_symbolic_ref)

    assert not update_cmd._restore_checkout_identity(["git"], repo, "HEAD", start_sha)


def _pull_release(repo_args) -> str:
    """Invoke the release transaction with the reduced clean-tree signature."""
    return update_cmd._pull_release_update(*repo_args)


def _release_repo_at_start(tmp_path: Path, start_detached: bool) -> tuple[Path, str, str, str]:
    """A repo with a valid local release tag and a clean checkout parked at the start:
    returns ``(repo, start_branch, start_sha, release_sha)``."""
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "clean")
    release_sha = _commit(repo, "release", "release")
    _git(repo, "tag", "v2026.9.1", release_sha)
    _git(repo, "reset", "--hard", start_sha)
    if start_detached:
        _git(repo, "checkout", "--detach", start_sha)
    return repo, ("HEAD" if start_detached else "main"), start_sha, release_sha


def _assert_untouched_start(repo: Path, start_branch: str, start_sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == start_sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == start_branch
    assert _git(repo, "stash", "list").stdout.strip() == ""


def _release_toctou_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo whose release tag moved between resolve and checkout: returns
    ``(repo, start_sha, resolved_sha)`` with HEAD parked at *start_sha* on main and the tag
    now pointing at a third commit, so a detached checkout of the tag can never land at
    *resolved_sha* and the post-checkout verification must fail."""
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "clean")
    release_sha = _commit(repo, "release", "release")
    _git(repo, "tag", "v2026.9.1", release_sha)
    resolved = update_cmd._resolve_release_commit(["git"], repo, "v2026.9.1")
    assert resolved == release_sha
    moved_sha = _commit(repo, "moved", "moved")
    _git(repo, "tag", "-f", "v2026.9.1", moved_sha)
    _git(repo, "reset", "--hard", start_sha)
    return repo, start_sha, resolved


def test_release_pull_succeeds_on_clean_tree_without_any_stash_operation(
        monkeypatch, tmp_path, capsys):
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, False)

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    returned = update_cmd._pull_release_update(
        ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    assert returned == start_sha
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == release_sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "HEAD"
    assert _git(repo, "stash", "list").stdout.strip() == ""


@pytest.mark.parametrize("start_detached", [False, True])
def test_release_pull_verification_failure_restores_clean_start(
        monkeypatch, tmp_path, capsys, start_detached):
    repo, start_sha, resolved_sha = _release_toctou_repo(tmp_path)
    if start_detached:
        _git(repo, "checkout", "--detach", start_sha)
    start_branch = "HEAD" if start_detached else "main"

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", resolved_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "not detached at the requested release commit" in out
    assert "Original checkout restored." in out
    _assert_untouched_start(repo, start_branch, start_sha)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "clean"


@pytest.mark.parametrize("start_detached", [False, True])
def test_release_pull_checkout_failure_restores_clean_start(
        monkeypatch, tmp_path, capsys, start_detached):
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "clean")
    release_sha = _commit(repo, "release", "release")
    _git(repo, "reset", "--hard", start_sha)
    if start_detached:
        _git(repo, "checkout", "--detach", start_sha)
    start_branch = "HEAD" if start_detached else "main"

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    # The release commit exists (the collision probe can answer) but its tag was never
    # fetched, so the checkout itself fails.
    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "Failed to checkout official release" in out
    assert "Original checkout restored." in out
    _assert_untouched_start(repo, start_branch, start_sha)


@pytest.mark.parametrize("start_detached", [False, True])
@pytest.mark.parametrize(
    ("fault", "expected_exc"),
    [("tempdir_oserror", OSError), ("verify_exception", RuntimeError)],
)
def test_release_pull_exception_rolls_back_clean_start(
        monkeypatch, tmp_path, capsys, start_detached, fault, expected_exc):
    """Fault injection: an exception raised inside the release transaction (an OSError from
    TemporaryDirectory under the syntax validator; any exception from the post-checkout
    identity verification) must roll back to the captured start — SHA and attached/detached
    identity — then re-raise. HEAD must never stay at the release."""
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, start_detached)

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    if fault == "tempdir_oserror":
        def broken_tempdir(*_args, **_kwargs):
            raise OSError("no space left for the syntax-check scratch dir")
        monkeypatch.setattr("tempfile.TemporaryDirectory", broken_tempdir)
    else:
        def broken_branch_probe(*_args, **_kwargs):
            raise RuntimeError("identity probe crashed")
        monkeypatch.setattr(update_cmd, "_current_branch_name", broken_branch_probe)

    with pytest.raises(expected_exc):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "Original checkout restored." in out
    _assert_untouched_start(repo, start_branch, start_sha)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "clean"


def test_release_rollback_never_rewinds_a_concurrently_advanced_branch(
        monkeypatch, tmp_path, capsys):
    """Data-loss guard: another git actor advances the starting branch between the release
    checkout and a failed post-checkout validation. The rollback must NOT move the branch
    back to the start SHA (that would orphan the concurrent commit); it reports failure with
    static guidance and leaves the branch — and the user's commit — untouched."""
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, False)
    _git(repo, "checkout", "--detach", start_sha)
    concurrent_sha = _commit(repo, "concurrent", "concurrent work")
    _git(repo, "checkout", "--force", "main")

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    def advance_branch_then_fail_validation(_root):
        _git(repo, "update-ref", f"refs/heads/{start_branch}", concurrent_sha)
        return False, "hermes_cli/config.py", "injected syntax error"

    monkeypatch.setattr(
        update_cmd, "_validate_critical_files_syntax", advance_branch_then_fail_validation)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "could not be restored automatically" in out
    assert "Original checkout restored." not in out
    assert _git(repo, "rev-parse", f"refs/heads/{start_branch}").stdout.strip() == concurrent_sha
    assert _git(repo, "show", f"refs/heads/{start_branch}:tracked.txt").stdout == "concurrent work"
    _assert_left_detached_at(repo, release_sha)


@pytest.mark.parametrize("start_detached", [False, True])
@pytest.mark.parametrize("edit_kind", ["staged", "tracked", "untracked"])
def test_release_rollback_fails_closed_on_concurrent_worktree_edits(
        monkeypatch, tmp_path, capsys, start_detached, edit_kind):
    """Concurrent-edit guard: another actor writes staged/tracked/untracked state while HEAD
    is detached at the release, and the post-checkout validation then fails. The rollback
    must NOT check out over that work (forced or otherwise): it fails closed with static
    guidance, leaves HEAD detached at the release, every concurrent byte survives
    byte-for-byte (staged entries still staged), and the branch refs are unchanged."""
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, start_detached)

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    # Mid-edit files are legitimately invalid Python; survival must be byte-exact anyway.
    payloads = {
        "staged": ("concurrent_staged.py", b"class Unfinished(:\n  # staged mid-edit \xf0\x9f\x9a\xa7\n"),
        "tracked": ("tracked.txt", b"def broken(:\n  # concurrent tracked edit\n"),
        "untracked": ("concurrent_note.txt", b"scratch notes another process wrote\n"),
    }
    name, payload = payloads[edit_kind]

    def inject_edit_then_fail_validation(_root):
        (repo / name).write_bytes(payload)
        if edit_kind == "staged":
            _git(repo, "add", name)
        return False, "hermes_cli/config.py", "injected syntax error"

    monkeypatch.setattr(
        update_cmd, "_validate_critical_files_syntax", inject_edit_then_fail_validation)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "could not be restored automatically" in out
    assert "Original checkout restored." not in out
    _assert_left_detached_at(repo, release_sha)
    assert (repo / name).read_bytes() == payload
    if edit_kind == "staged":
        assert name in _git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert _git(repo, "rev-parse", "refs/heads/main").stdout.strip() == start_sha
    assert _git(repo, "stash", "list").stdout.strip() == ""


def test_release_pull_fails_closed_on_unexpected_autostash(monkeypatch, tmp_path, capsys):
    """Release mode never autostashes. If a non-None autostash ever reaches the release
    transaction despite the clean-tree preflight, it must refuse BEFORE the checkout — the
    stash is not applied, dropped, or otherwise touched."""
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, False)
    (repo / "tracked.txt").write_text("stashed work", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "unexpected")
    stash_sha = _git(repo, "rev-parse", "refs/stash").stdout.strip()

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, stash_sha)

    out = capsys.readouterr().out
    assert "Refusing before checkout" in out
    # The checkout never happened and the stash entry is untouched.
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == start_sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == start_branch
    assert _git(repo, "rev-parse", "refs/stash").stdout.strip() == stash_sha


def test_release_pull_rechecks_clean_tree_right_before_checkout(monkeypatch, tmp_path, capsys):
    """The preflight runs before the receipt/backup/pause phases; a tree dirtied in that
    window must refuse at the last moment before the checkout instead of being absorbed."""
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, False)
    (repo / "tracked.txt").write_text("dirtied after preflight", encoding="utf-8")

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    out = capsys.readouterr().out
    assert "clean, settled Git state" in out
    _assert_untouched_start(repo, start_branch, start_sha)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirtied after preflight"


def _ignored_collision_repo(tmp_path: Path, added_path: str) -> tuple[Path, str, str]:
    """A repo whose tagged release adds *added_path* while the same path exists locally as
    an ignored user file: returns ``(repo, start_sha, release_sha)`` with HEAD clean on main."""
    repo = _repo(tmp_path)
    # Backslash-escape gitignore glob metacharacters so the PATTERN is literal too — the
    # point under test is the updater's literal path handling, not gitignore globbing.
    ignore_pattern = added_path.replace("[", "\\[").replace("]", "\\]")
    (repo / ".gitignore").write_text(f"/{ignore_pattern}\nvenv/\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "commit", "-m", "start")
    start_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / added_path).write_text("official release bytes\n", encoding="utf-8")
    _git(repo, "add", "-f", added_path)
    _git(repo, "commit", "-m", "release adds the path")
    release_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "tag", "v2026.9.1", release_sha)
    _git(repo, "reset", "--hard", start_sha)
    assert not (repo / added_path).exists()
    return repo, start_sha, release_sha


@pytest.mark.parametrize(
    "added_path",
    ["secrets.txt", "data[1].txt"],
    ids=["plain-name", "glob-metacharacters-taken-literally"],
)
def test_release_apply_refuses_ignored_file_collision_before_checkout(
        monkeypatch, tmp_path, capsys, added_path):
    """An ignored user file at a target-added path blocks the update before the checkout:
    the bytes survive byte-for-byte and the checkout never occurs. Paths with glob
    metacharacters are compared literally."""
    repo, start_sha, release_sha = _ignored_collision_repo(tmp_path, added_path)
    user_bytes = b"USER SECRET \xf0\x9f\x94\x91\n"
    (repo / added_path).write_bytes(user_bytes)
    # Preflight passes: the collision is invisible to `git status` (ignored).
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""
    assert update_cmd._release_apply_git_state_block_reason(["git"], repo) is None

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, "main", start_sha, None)

    out = capsys.readouterr().out
    assert added_path in out
    assert (repo / added_path).read_bytes() == user_bytes
    _assert_untouched_start(repo, "main", start_sha)


def test_release_apply_ignores_unrelated_ignored_directories(monkeypatch, tmp_path):
    """venv/, node_modules/ and other ignored trees the release does not add must never
    block, and must survive the checkout untouched — including when an ignored file merely
    MATCHES a glob-shaped target-added path without being it (literal comparison)."""
    repo, start_sha, release_sha = _ignored_collision_repo(tmp_path, "data[1].txt")
    (repo / "venv").mkdir()
    (repo / "venv" / "lib.py").write_text("site-packages\n", encoding="utf-8")
    # `data1.txt` matches the glob `data[1].txt` but is NOT the literal target-added path.
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("data1.txt\n", encoding="utf-8")
    (repo / "data1.txt").write_text("mine\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    returned = update_cmd._pull_release_update(
        ["git"], "v2026.9.1", release_sha, "main", start_sha, None)

    assert returned == start_sha
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == release_sha
    assert (repo / "venv" / "lib.py").read_text(encoding="utf-8") == "site-packages\n"
    assert (repo / "data1.txt").read_text(encoding="utf-8") == "mine\n"
    assert (repo / "data[1].txt").read_text(encoding="utf-8") == "official release bytes\n"


def test_release_apply_fails_closed_when_collision_probe_cannot_answer(
        monkeypatch, tmp_path, capsys):
    repo, start_branch, start_sha, release_sha = _release_repo_at_start(tmp_path, False)

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(
        update_cmd, "_release_overwrite_collision_paths", lambda *_a, **_k: None)

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, start_branch, start_sha, None)

    assert "could not verify" in capsys.readouterr().out
    _assert_untouched_start(repo, start_branch, start_sha)


def test_no_overwrite_ignore_backstops_a_blind_collision_probe(monkeypatch, tmp_path, capsys):
    """Final defense: even if the collision probe misses (simulated blind probe), the
    checkout itself must refuse to overwrite the ignored file and roll back — git never
    clobbers it."""
    repo, start_sha, release_sha = _ignored_collision_repo(tmp_path, "secrets.txt")
    user_bytes = b"USER SECRET\n"
    (repo / "secrets.txt").write_bytes(user_bytes)

    from hermes_cli import main as hermes_main
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(
        update_cmd, "_release_overwrite_collision_paths", lambda *_a, **_k: [])

    with pytest.raises(SystemExit, match="1"):
        update_cmd._pull_release_update(
            ["git"], "v2026.9.1", release_sha, "main", start_sha, None)

    out = capsys.readouterr().out
    assert "Failed to checkout official release" in out
    assert "Original checkout restored." in out
    assert (repo / "secrets.txt").read_bytes() == user_bytes
    _assert_untouched_start(repo, "main", start_sha)


def test_collision_probe_lists_only_existing_target_added_paths(tmp_path):
    """Unit contract for the probe itself: target-added paths that exist locally are
    returned; target-added paths absent locally and unrelated local files are not."""
    repo, start_sha, release_sha = _ignored_collision_repo(tmp_path, "secrets.txt")
    (repo / "secrets.txt").write_text("mine\n", encoding="utf-8")
    (repo / "venv").mkdir()
    (repo / "venv" / "lib.py").write_text("x\n", encoding="utf-8")

    collisions = update_cmd._release_overwrite_collision_paths(
        ["git"], repo, start_sha, release_sha)
    assert collisions == ["secrets.txt"]

    (repo / "secrets.txt").unlink()
    assert update_cmd._release_overwrite_collision_paths(
        ["git"], repo, start_sha, release_sha) == []
