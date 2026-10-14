"""Regression checks for fork-safe GitHub Actions policy."""

from pathlib import Path
import re

import pytest
import hermes_yaml as yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_fork_owner_attribution_exception_is_narrow() -> None:
    workflow = _read(".github/workflows/contributor-check.yml")
    assert "PR_AUTHOR: ${{ github.event.pull_request.user.login || github.actor }}" in workflow
    assert '[ "$GITHUB_REPOSITORY" != "NousResearch/hermes-agent" ]' in workflow
    assert '[ "$PR_AUTHOR" = "$GITHUB_REPOSITORY_OWNER" ]' in workflow
    assert "Non-owner contributor PRs still pass through the full gate" in workflow
    assert "Check for unmapped contributor emails" in workflow


def test_private_runner_assignments_are_removed_from_the_fork() -> None:
    # Upstream can keep its larger runners; every such assignment must select
    # ordinary hosted capacity when the repository is a fork.
    relatives = ["js-tests.yml", "rust-tests.yml", "tests.yml", "tests-os.yml", "nix.yml"]
    conditional = re.compile(
        r"\$\{\{ github\.repository == 'NousResearch/hermes-agent' && '[^']+-core' "
        r"\|\| '(ubuntu-latest|windows-latest|windows-11-arm)' \}\}")
    for name in relatives:
        workflow = yaml.safe_load(_read(f".github/workflows/{name}"))
        runners = []
        for job in workflow["jobs"].values():
            runners.append(job.get("runs-on", ""))
            matrix = job.get("strategy", {}).get("matrix", {})
            if isinstance(matrix, dict):
                runners.extend(row.get("runner", "") for row in matrix.get("include", []))
        for runner in runners:
            if isinstance(runner, str) and "-core" in runner:
                assert conditional.fullmatch(runner), (name, runner)


def test_standard_python_runner_has_bounded_workers() -> None:
    workflow = _read(".github/workflows/tests.yml")
    assert "runs-on: ubuntu-latest" in workflow
    workers = re.findall(r"HERMES_TEST_WORKERS: (.+)", workflow)
    assert workers and all("github.repository == 'NousResearch/hermes-agent'" in value
                           and "|| '2'" in value for value in workers)
    assert "timeout-minutes: 60" in workflow


def _upstream_and_fork(expression: object) -> tuple[int, int]:
    match = re.fullmatch(
        r"\$\{\{ github\.repository == 'NousResearch/hermes-agent' && '?(\d+)'? \|\| '?(\d+)'? \}\}",
        str(expression))
    assert match, expression
    return int(match.group(1)), int(match.group(2))


def test_windows_update_budgets_nest_on_upstream_and_fork_runners() -> None:
    # Each worker runs its share of the journeys back to back, so the job must outlast that share
    # at the per-file limit plus setup (~10 min cold); otherwise a slow file cancels the whole job
    # instead of failing on its own timeout. Each journey's install must also fit inside its file.
    setup_seconds = 15 * 60
    workflow = yaml.safe_load(_read(".github/workflows/windows-install-update-e2e.yml"))
    job = workflow["jobs"]["install-update"]
    env = next(step for step in job["steps"] if step.get("name") == "Run Windows install + update E2E")["env"]
    journeys = len(list((ROOT / "tests/e2e/core/windows_update").glob("test_*.py")))
    budgets = {
        "job_minutes": _upstream_and_fork(job["timeout-minutes"]),
        "file_seconds": _upstream_and_fork(env["HERMES_TEST_FILE_TIMEOUT"]),
        "install_seconds": _upstream_and_fork(env["HERMES_E2E_INSTALL_TIMEOUT"]),
        "workers": _upstream_and_fork(env["HERMES_TEST_WORKERS"]),
    }
    assert journeys >= 1
    for side in (0, 1):
        job_minutes, file_seconds, install_seconds, workers = (b[side] for b in budgets.values())
        assert install_seconds < file_seconds
        assert -(-journeys // workers) * file_seconds + setup_seconds <= job_minutes * 60
    # The fork runs on standard hosted capacity, so none of its budgets may undercut upstream's.
    assert budgets["workers"][1] <= budgets["workers"][0]
    for name in ("job_minutes", "file_seconds", "install_seconds"):
        assert budgets[name][1] >= budgets[name][0], name


def test_standard_nix_runner_has_bounded_parallelism() -> None:
    workflow = _read(".github/workflows/nix.yml")
    assert "runs-on: ubuntu-latest" in workflow
    assert "--print-build-logs --max-jobs" in workflow
    assert "github.repository == 'NousResearch/hermes-agent' && '32' || '2'" in workflow
    assert "timeout-minutes: 60" in workflow


@pytest.mark.parametrize("relative", [
    ".github/workflows/ci.yaml",
    ".github/workflows/nix.yml",
])
def test_validation_workflows_accept_manual_runs(relative) -> None:
    workflow = yaml.safe_load(_read(relative))
    events = workflow.get("on", workflow.get(True))  # YAML 1.1-compatible loader
    assert {"pull_request", "push", "workflow_dispatch"} <= set(events)
    assert events["push"]["branches"] == ["main"]


def test_manual_nix_cache_is_main_only_without_delete_permissions() -> None:
    workflow = yaml.safe_load(_read(".github/workflows/nix.yml"))
    assert workflow["permissions"] == {"contents": "read"}
    cache_steps = [
        step for job in workflow["jobs"].values() for step in job.get("steps", [])
        if step.get("uses", "").startswith("nix-community/cache-nix-action@")
    ]
    assert len(cache_steps) == 1
    options = cache_steps[0]["with"]
    assert options["save"] == (
        "${{ github.event_name != 'pull_request' && github.ref == 'refs/heads/main' }}"
    )
    assert options["purge"] is False


def test_detect_action_only_receives_declared_inputs() -> None:
    action = yaml.safe_load(_read(".github/actions/detect-changes/action.yml"))
    workflow = yaml.safe_load(_read(".github/workflows/ci.yaml"))
    callers = [
        step for step in workflow["jobs"]["detect"]["steps"]
        if step.get("uses") == "./.github/actions/detect-changes"
    ]
    assert len(callers) == 1
    assert set(callers[0].get("with", {})) <= set(action["inputs"])
