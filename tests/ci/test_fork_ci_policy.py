"""Regression checks for fork-safe GitHub Actions policy."""

from pathlib import Path


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
