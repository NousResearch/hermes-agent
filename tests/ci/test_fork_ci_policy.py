"""Regression checks for fork-safe GitHub Actions policy."""

from pathlib import Path

import pytest


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
    assignments = {
        ".github/workflows/js-tests.yml": (
            "runs-on: ubuntu-latest-32-core",
            "runs-on: ubuntu-latest",
        ),
        ".github/workflows/rust-tests.yml": (
            "runs-on: ubuntu-latest-32-core",
            "runs-on: ubuntu-latest",
        ),
        ".github/workflows/tests.yml": (
            "runs-on: ubuntu-latest-96-core",
            "runs-on: ubuntu-latest",
        ),
        ".github/workflows/tests-os.yml": (
            "runner: windows-latest-32-core",
            "runner: windows-latest",
        ),
        ".github/workflows/nix.yml": (
            "runs-on: ubuntu-latest-32-core",
            "runs-on: ubuntu-latest",
        ),
    }
    for relative, (private, public) in assignments.items():
        workflow = _read(relative)
        assert private not in workflow
        assert public in workflow


def test_standard_python_runner_has_bounded_workers() -> None:
    workflow = _read(".github/workflows/tests.yml")
    assert "runs-on: ubuntu-latest" in workflow
    assert "HERMES_TEST_WORKERS: 4" in workflow
    assert "timeout-minutes: 60" in workflow


def test_standard_nix_runner_has_bounded_parallelism() -> None:
    workflow = _read(".github/workflows/nix.yml")
    assert "runs-on: ubuntu-latest" in workflow
    assert "nix flake check --print-build-logs --max-jobs 2" in workflow
    assert "timeout-minutes: 90" in workflow


@pytest.mark.parametrize("relative", [
    ".github/workflows/ci.yaml",
    ".github/workflows/nix.yml",
])
def test_validation_workflows_accept_manual_runs(relative) -> None:
    yaml = pytest.importorskip("yaml")
    # BaseLoader keeps the YAML key `on` as text instead of YAML 1.1's boolean.
    workflow = yaml.load(_read(relative), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert workflow["on"]["push"]["branches"] == ["main"]


def test_manual_nix_cache_is_main_only_without_delete_permissions() -> None:
    yaml = pytest.importorskip("yaml")
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
    yaml = pytest.importorskip("yaml")
    action = yaml.safe_load(_read(".github/actions/detect-changes/action.yml"))
    workflow = yaml.safe_load(_read(".github/workflows/ci.yaml"))
    callers = [
        step for step in workflow["jobs"]["detect"]["steps"]
        if step.get("uses") == "./.github/actions/detect-changes"
    ]
    assert len(callers) == 1
    assert set(callers[0].get("with", {})) <= set(action["inputs"])
