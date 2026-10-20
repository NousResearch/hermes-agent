"""The merge-base gate must evaluate the PR head, not the merge ref.

`actions/checkout` under a pull_request caller materializes refs/pull/N/merge,
whose first parent is always main's tip: `git merge-base origin/main HEAD`
against that commit resolves to main's tip no matter what history the PR
carries, so a check against HEAD can never reject; that dead guard let an
unrelated-histories PR pass the gate this workflow exists to enforce.

These tests run the workflow's own `merge-base-check` script verbatim against
fixture repositories exposing refs/pull/<n>/head, the way GitHub exposes a
pull request head (for forks included) on the base repository.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import hermes_yaml as yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/history-check.yml"


def _merge_base_step():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return next(
        step
        for step in workflow["jobs"]["check-common-ancestor"]["steps"]
        if step.get("id") == "merge-base-check"
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


@pytest.fixture
def remote(tmp_path):
    """A bare 'GitHub' remote: main history, an unrelated orphan head as
    refs/pull/1/head, and a related feature branch as refs/pull/2/head, each
    with a synthetic merge ref as GitHub publishes it."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", str(seed)], check=True)
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    _git(seed, "commit", "--allow-empty", "-qm", "root")
    _git(seed, "commit", "--allow-empty", "-qm", "main work")
    _git(seed, "branch", "-M", "main")

    _git(seed, "checkout", "-q", "--orphan", "orphan")
    _git(seed, "reset", "-q")
    _git(seed, "commit", "--allow-empty", "-qm", "orphan root")
    orphan = _git(seed, "rev-parse", "orphan")

    _git(seed, "checkout", "-q", "-b", "feature", "main")
    _git(seed, "commit", "--allow-empty", "-qm", "feature work")
    feature = _git(seed, "rev-parse", "feature")

    # A main-descended head that merged the orphan branch: merge-base finds
    # a shared ancestor, but the head carries a second root.
    _git(seed, "checkout", "-q", "-b", "graft", "main")
    _git(seed, "merge", "--no-ff", "--allow-unrelated-histories", "-qm",
         "merge unrelated work", "orphan")
    graft = _git(seed, "rev-parse", "graft")

    _git(seed, "push", "-q", str(bare), "main:main")
    # GitHub publishes the PR head on the base repo for fork PRs as well.
    _git(seed, "push", "-q", str(bare), f"{orphan}:refs/pull/1/head")
    _git(seed, "push", "-q", str(bare), f"{feature}:refs/pull/2/head")
    _git(seed, "push", "-q", str(bare), f"{graft}:refs/pull/3/head")

    # Synthetic merge refs, matching what refs/pull/<n>/merge carries.
    _git(seed, "checkout", "-q", "-b", "merge-1", "main")
    _git(seed, "merge", "--no-ff", "--allow-unrelated-histories", "-qm", "merge", "orphan")
    _git(seed, "push", "-q", str(bare), "merge-1:refs/pull/1/merge")
    _git(seed, "checkout", "-q", "-b", "merge-2", "main")
    _git(seed, "merge", "--no-ff", "-qm", "merge", "feature")
    _git(seed, "push", "-q", str(bare), "merge-2:refs/pull/2/merge")
    _git(seed, "checkout", "-q", "-b", "merge-3", "main")
    _git(seed, "merge", "--no-ff", "-qm", "merge", "graft")
    _git(seed, "push", "-q", str(bare), "merge-3:refs/pull/3/merge")

    _git(bare, "symbolic-ref", "HEAD", "refs/heads/main")
    return bare


def _run_step(remote: Path, tmp_path: Path, pr_number: int, env_extra=None):
    """Clone the remote, check out the merge ref the way actions/checkout does
    under a pull_request caller, then run the workflow step's script."""
    work = tmp_path / f"work-{pr_number}"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True)
    merged = subprocess.run(
        ["git", "-C", str(work), "fetch", "-q", "origin", f"pull/{pr_number}/merge"],
        stderr=subprocess.DEVNULL,
    )
    if merged.returncode == 0:
        _git(work, "checkout", "-q", "FETCH_HEAD")
    else:
        _git(work, "checkout", "-q", "origin/main")
    output = tmp_path / f"output-{pr_number}"
    output.touch()
    step = _merge_base_step()
    env = {
        **os.environ,
        **{key: str(value) for key, value in (step.get("env") or {}).items()},
        "PR_NUMBER": str(pr_number),
        "GITHUB_OUTPUT": str(output),
        **(env_extra or {}),
    }
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, work, output


@pytest.mark.platforms("posix")
def test_merge_ref_head_is_structurally_uncheckable(remote, tmp_path):
    """Pin the defect the fix addresses: merge-base against the merge ref is
    never empty, so a gate on HEAD passes even for an unrelated history."""
    work = tmp_path / "probe"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True)
    _git(work, "fetch", "-q", "origin", "pull/1/merge")
    assert _git(work, "merge-base", "origin/main", "FETCH_HEAD") == _git(
        work, "rev-parse", "origin/main"
    )


@pytest.mark.platforms("posix")
def test_unrelated_history_pr_is_rejected(remote, tmp_path):
    result, work, output = _run_step(remote, tmp_path, 1)

    assert result.returncode == 1, result.stderr or result.stdout
    status_line = next(
        line for line in output.read_text().splitlines() if line.startswith("review_status=")
    )
    status = json.loads(status_line.split("=", 1)[1])
    assert status[0]["results"][0]["kind"] == "action_required"
    assert json.loads((work / "review-status.json").read_text().split("=", 1)[1]) == status


@pytest.mark.platforms("posix")
def test_related_history_pr_passes(remote, tmp_path):
    result, work, output = _run_step(remote, tmp_path, 2)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "review_status=[]" in output.read_text()
    assert "review_status=[]" in (work / "review-status.json").read_text()


@pytest.mark.platforms("posix")
def test_grafted_unrelated_root_is_rejected(remote, tmp_path):
    """merge-base alone cannot see this: the head shares history with main
    yet drags the orphan root along, which is the same #25045 damage the
    check exists to prevent."""
    result, work, output = _run_step(remote, tmp_path, 3)

    assert result.returncode == 1, result.stderr or result.stdout
    assert "root commit" in result.stdout
    status_line = next(
        line for line in output.read_text().splitlines() if line.startswith("review_status=")
    )
    status = json.loads(status_line.split("=", 1)[1])
    assert status[0]["results"][0]["kind"] == "action_required"


@pytest.mark.platforms("posix")
def test_transient_fetch_failure_retries(remote, tmp_path):
    """A fetch that fails once then succeeds must still pass: transient
    transport errors should not take the check down. A PATH shim makes the
    first `git fetch` in the run block exit 1, then delegates to real git."""
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    counter = tmp_path / "fetch-count"
    real_git = shutil.which("git")
    (shim_dir / "git").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "fetch" ]; then\n'
        f'  n=$(( $(cat "{counter}" 2>/dev/null || echo 0) + 1 ))\n'
        f'  echo "$n" > "{counter}"\n'
        '  [ "$n" -eq 1 ] && exit 1\n'
        "fi\n"
        f'exec {real_git} "$@"\n'
    )
    os.chmod(shim_dir / "git", 0o755)

    result, _, output = _run_step(
        remote, tmp_path, 2,
        env_extra={"PATH": f"{shim_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "review_status=[]" in output.read_text()
    # Two fetch calls: the first failed, the retry succeeded.
    assert counter.read_text().strip() == "2"


@pytest.mark.platforms("posix")
def test_missing_pull_head_fails_closed(remote, tmp_path):
    """A fetch that cannot resolve refs/pull/<n>/head (closed PR, transient
    failure) must fail the step red rather than emit a passing verdict or a
    misleading unrelated-histories one: the job goes red and no review_status
    is produced."""
    result, work, output = _run_step(remote, tmp_path, 99)

    assert result.returncode != 0
    assert "review_status=" not in output.read_text()
    assert not (work / "review-status.json").exists()
