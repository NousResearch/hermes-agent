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
