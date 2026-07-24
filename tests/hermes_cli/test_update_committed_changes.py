import subprocess
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import main as hermes_main


_REAL_SUBPROCESS_RUN = subprocess.run


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _REAL_SUBPROCESS_RUN(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def _commit(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _diverged_repositories(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(seed))
    _git(seed, "config", "user.email", "tests@example.com")
    _git(seed, "config", "user.name", "Hermes Tests")
    _commit(seed, "base.txt", "base\n", "base")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", "-b", "main", str(remote), str(local))
    _git(local, "config", "user.email", "tests@example.com")
    _git(local, "config", "user.name", "Hermes Tests")
    return local, seed


def _advance_and_fetch(local: Path, seed: Path, path: str = "upstream.txt") -> str:
    remote_head = _commit(seed, path, "upstream\n", "upstream change")
    _git(seed, "push", "origin", "main")
    _git(local, "fetch", "origin", "main")
    return remote_head


def test_reconcile_reports_no_local_commits(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    _advance_and_fetch(local, seed)

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"],
        local,
        "main",
        "rebase",
        _git(local, "rev-parse", "HEAD").stdout.strip(),
    )

    assert outcome.status == "no_local_commits"
    assert outcome.backup_ref is None


def test_reconcile_defaults_to_refusal(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    _commit(local, "local.txt", "local\n", "local patch")
    _advance_and_fetch(local, seed)

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"],
        local,
        "main",
        "refuse",
        _git(local, "rev-parse", "HEAD").stdout.strip(),
    )

    assert outcome.status == "refused"
    assert outcome.ahead_count == 1
    assert outcome.backup_ref is None


def test_reconcile_successfully_rebases_linear_patches(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    original_head = _commit(local, "local.txt", "local\n", "local patch")
    remote_head = _advance_and_fetch(local, seed)

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )

    assert outcome.status == "rebased"
    assert outcome.backup_ref
    assert _git(local, "rev-parse", outcome.backup_ref).stdout.strip() == original_head
    assert (
        _git(local, "merge-base", "--is-ancestor", remote_head, "HEAD").returncode == 0
    )
    assert (local / "local.txt").read_text() == "local\n"


def test_reconcile_auto_drops_upstream_equivalent_patch(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    original_head = _commit(local, "equivalent.txt", "same\n", "local wording")
    remote_head = _commit(seed, "equivalent.txt", "same\n", "upstream wording")
    _git(seed, "push", "origin", "main")
    _git(local, "fetch", "origin", "main")

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )

    assert outcome.status == "rebased"
    assert _git(local, "rev-parse", "HEAD").stdout.strip() == remote_head
    assert _git(local, "rev-list", "--count", "origin/main..HEAD").stdout.strip() == "0"


def test_reconcile_conflict_aborts_and_restores_original_head(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    _commit(local, "conflict.txt", "base\n", "shared starting point")
    _git(local, "push", "origin", "main")
    _git(seed, "pull", "--ff-only", "origin", "main")
    original_head = _commit(local, "conflict.txt", "local\n", "local conflict")
    _commit(seed, "conflict.txt", "upstream\n", "upstream conflict")
    _git(seed, "push", "origin", "main")
    _git(local, "fetch", "origin", "main")

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )

    assert outcome.status == "failed"
    assert outcome.backup_ref
    assert _git(local, "rev-parse", "HEAD").stdout.strip() == original_head
    assert not (local / ".git" / "rebase-merge").exists()


def test_reconcile_conflict_reports_the_conflicting_files(tmp_path, capsys):
    """A failed rebase must name the failing commit and the conflicting paths.

    Regression: git puts ``CONFLICT (content): ...`` on stdout and
    ``Rebasing (n/m)`` first on stderr, so ``(stderr or stdout).splitlines()[0]``
    surfaced only the progress banner and the operator learned nothing.
    """
    local, seed = _diverged_repositories(tmp_path)
    _commit(local, "conflict.txt", "base\n", "shared starting point")
    _git(local, "push", "origin", "main")
    _git(seed, "pull", "--ff-only", "origin", "main")
    original_head = _commit(local, "conflict.txt", "local\n", "local conflict")
    _commit(seed, "conflict.txt", "upstream\n", "upstream conflict")
    _git(seed, "push", "origin", "main")
    _git(local, "fetch", "origin", "main")

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )
    output = capsys.readouterr().out

    assert outcome.status == "failed"
    # The unmerged path is read before --abort destroys it.
    assert "conflict.txt" in output
    # The progress banner must no longer be the whole story.
    assert "could not apply" in (outcome.detail or "").lower()
    assert not (outcome.detail or "").startswith("Rebasing (")
    # And the operator gets a copy-pasteable way to finish by hand.
    assert f"git -C {local} rebase --empty=drop --no-keep-empty" in output


def test_rebase_conflict_summary_falls_back_to_stdout_conflict_lines(tmp_path):
    """Rename/delete conflicts leave no unmerged index entry — scrape stdout."""
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    result = SimpleNamespace(
        stdout="Auto-merging a/b.py\nCONFLICT (content): Merge conflict in a/b.py\n",
        stderr="Rebasing (1/3)\nerror: could not apply deadbee... some patch\nhint: blah\n",
    )

    detail, files = hermes_main._rebase_conflict_summary(["git"], repo, result)

    assert detail == "error: could not apply deadbee... some patch"
    assert files == ["a/b.py"]


def test_reconcile_abort_failure_preserves_backup_without_reset(
    tmp_path, monkeypatch, capsys
):
    local, seed = _diverged_repositories(tmp_path)
    _commit(local, "conflict.txt", "base\n", "shared starting point")
    _git(local, "push", "origin", "main")
    _git(seed, "pull", "--ff-only", "origin", "main")
    original_head = _commit(local, "conflict.txt", "local\n", "local conflict")
    _commit(seed, "conflict.txt", "upstream\n", "upstream conflict")
    _git(seed, "push", "origin", "main")
    _git(local, "fetch", "origin", "main")
    calls = []

    def run_with_failed_abort(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:] == ["rebase", "--abort"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="abort failed")
        return _REAL_SUBPROCESS_RUN(cmd, **kwargs)

    monkeypatch.setattr(hermes_main.subprocess, "run", run_with_failed_abort)
    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )

    assert outcome.status == "failed"
    assert outcome.backup_ref
    assert _git(local, "rev-parse", outcome.backup_ref).stdout.strip() == original_head
    assert not [cmd for cmd in calls if cmd[1:3] == ["reset", "--hard"]]
    output = capsys.readouterr().out
    assert f"git -C {local} rebase --abort" in output
    assert f"git -C {local} reset --hard {outcome.backup_ref}" in output

    _REAL_SUBPROCESS_RUN(
        ["git", "rebase", "--abort"], cwd=local, capture_output=True, text=True
    )


def test_reconcile_refuses_local_merge_commits(tmp_path):
    local, seed = _diverged_repositories(tmp_path)
    _git(local, "checkout", "-b", "feature")
    _commit(local, "feature.txt", "feature\n", "feature patch")
    _git(local, "checkout", "main")
    _commit(local, "main-local.txt", "main\n", "main patch")
    _git(local, "merge", "--no-ff", "feature", "-m", "local merge")
    original_head = _git(local, "rev-parse", "HEAD").stdout.strip()
    _advance_and_fetch(local, seed)

    outcome = hermes_main._reconcile_committed_local_changes(
        ["git"], local, "main", "rebase", original_head
    )

    assert outcome.status == "refused"
    assert "merge commit" in (outcome.detail or "")
    refs = _git(
        local, "for-each-ref", "--format=%(refname)", "refs/hermes/update-backups"
    )
    assert refs.stdout == ""


def test_backup_refs_prune_by_reflog_creation_time_and_keep_unknown_metadata(
    tmp_path, monkeypatch
):
    local, _seed = _diverged_repositories(tmp_path)
    head = _git(local, "rev-parse", "HEAD").stdout.strip()
    prefix = "refs/hermes/update-backups/main/"
    refs = [f"{prefix}{name}" for name in ["z", "a", "y", "b", "x", "unknown"]]
    for ref in refs:
        _git(local, "update-ref", "--create-reflog", ref, head)
    creation_times = {
        refs[0]: 1,
        refs[1]: 5,
        refs[2]: 2,
        refs[3]: 4,
        refs[4]: 3,
        refs[5]: None,
    }
    monkeypatch.setattr(
        hermes_main,
        "_backup_ref_created_at",
        lambda _git_cmd, _cwd, ref: creation_times.get(ref, 100),
    )

    new_ref = hermes_main._create_committed_local_changes_backup(
        ["git"], local, "main", head, keep=5
    )

    remaining = set(
        _git(local, "for-each-ref", "--format=%(refname)", prefix).stdout.splitlines()
    )
    assert new_ref in remaining
    assert refs[0] not in remaining
    assert refs[5] in remaining
    assert remaining == {refs[1], refs[2], refs[3], refs[4], refs[5], new_ref}
