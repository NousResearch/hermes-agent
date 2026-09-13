import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import TypeGuard

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "osv-scanner.yml"
SHA_PIN_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")
STAGED_LOCKFILES = {
    "nix/node-gyp-11-4-0-package-lock.json": (
        ".osv-lockfiles/node-gyp-11-4-0/package-lock.json"
    ),
}
# The scanner and reporter run the digest-pinned upstream container images
# directly (docker run) instead of through google/osv-scanner-action — the
# remote action.yml pins those images by mutable tag, which the supply-chain
# audit fails closed on.
SCANNER_IMAGE = (
    "ghcr.io/google/osv-scanner-action:v2.5.1@sha256:"
    "dcd947131d8d11b8d0964de6590661fb921a4ecbd7b90a7cb21083acfc3fd8cc"
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _scan_steps() -> list[dict]:
    scan = _workflow()["jobs"]["scan"]
    steps = scan.get("steps")
    assert isinstance(steps, list), "scan must be a normal job with local steps"
    return steps


def _uses_step(prefix: str) -> dict:
    matches = [
        step
        for step in _scan_steps()
        if isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and step["uses"].startswith(prefix)
    ]
    assert len(matches) == 1, f"expected exactly one {prefix} step"
    return matches[0]


def _docker_step(image: str) -> dict:
    matches = [
        step
        for step in _scan_steps()
        if isinstance(step, dict)
        and isinstance(step.get("run"), str)
        and image in step["run"]
        and "--entrypoint /root/osv-reporter" not in step["run"]
    ]
    assert len(matches) == 1, f"expected exactly one step invoking {image}"
    return matches[0]


def _named_step(name: str) -> dict:
    matches = [step for step in _scan_steps() if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one {name!r} step"
    return matches[0]


def _external_uses(reference: object) -> TypeGuard[str]:
    return isinstance(reference, str) and not reference.startswith("./")


def _scan_arg_lines(step: dict) -> list[str]:
    return [
        line.strip()
        for line in step["with"]["scan-args"].splitlines()
        if line.strip()
    ]


def test_osv_scan_covers_every_repository_lockfile() -> None:
    scanner = _docker_step(SCANNER_IMAGE)
    configured = {
        line.strip().removesuffix("\\").strip().removeprefix("--lockfile=")
        for line in scanner["run"].splitlines()
        if line.strip().startswith("--lockfile=")
    }
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    tracked_lockfiles = {
        path
        for path in tracked
        if path.endswith("package-lock.json")
        or PurePosixPath(path).name == "uv.lock"
    }
    expected = (tracked_lockfiles - STAGED_LOCKFILES.keys()) | set(
        STAGED_LOCKFILES.values()
    )

    assert configured == expected


def test_nonstandard_lockfile_names_are_staged_under_supported_basenames() -> None:
    stage_script = _named_step("Stage nonstandard lockfiles")["run"]

    for source, staged in STAGED_LOCKFILES.items():
        assert source in stage_script
        assert staged in stage_script
        assert PurePosixPath(staged).name == "package-lock.json"


def _run_results_validation(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _named_step("Validate scan results")["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_osv_results_fail_validation(tmp_path: Path) -> None:
    result = _run_results_validation(tmp_path)

    assert result.returncode != 0


def test_malformed_osv_results_fail_validation(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text("not json", encoding="utf-8")

    result = _run_results_validation(tmp_path)

    assert result.returncode != 0


def test_well_formed_osv_results_pass_validation(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text('{"results": []}', encoding="utf-8")

    result = _run_results_validation(tmp_path)

    assert result.returncode == 0, result.stderr


def test_every_external_action_reference_is_pinned_to_exact_sha() -> None:
    for job_name, job in _workflow()["jobs"].items():
        reusable = job.get("uses")
        if _external_uses(reusable):
            assert SHA_PIN_RE.fullmatch(reusable), (
                f"{job_name}: mutable reusable workflow ref {reusable!r}"
            )
        for step in job.get("steps", []):
            uses = step.get("uses") if isinstance(step, dict) else None
            if _external_uses(uses):
                assert SHA_PIN_RE.fullmatch(uses), f"{job_name}: mutable uses ref {uses!r}"


def test_osv_reporter_blocks_on_vulnerabilities() -> None:
    reporter = _reporter_step()
    fail_args = [
        line.strip()
        for line in reporter["run"].splitlines()
        if line.strip().startswith("--fail-on-vuln")
    ]
    assert fail_args == ["--fail-on-vuln=true"]


def _reporter_step() -> dict:
    matches = [
        step
        for step in _scan_steps()
        if isinstance(step, dict)
        and isinstance(step.get("run"), str)
        and "--entrypoint /root/osv-reporter" in step["run"]
    ]
    assert len(matches) == 1, "expected exactly one osv-reporter step"
    return matches[0]


def test_scan_is_bounded_normal_job_not_reusable_workflow() -> None:
    scan = _workflow()["jobs"]["scan"]
    assert "uses" not in scan
    assert scan.get("runs-on")
    assert isinstance(scan.get("steps"), list)
    assert isinstance(scan.get("timeout-minutes"), int)
    assert scan["timeout-minutes"] > 0


def test_every_job_has_a_positive_timeout() -> None:
    for job_name, job in _workflow()["jobs"].items():
        assert isinstance(job.get("timeout-minutes"), int), (
            f"{job_name}: missing integer timeout-minutes"
        )
        assert job["timeout-minutes"] > 0


def test_permissions_are_scoped_to_each_job() -> None:
    workflow = _workflow()
    assert workflow.get("permissions") == {}
    assert workflow["jobs"]["scan"].get("permissions") == {
        "contents": "read",
        "security-events": "write",
    }
    assert workflow["jobs"]["emit-status"].get("permissions") == {
        "actions": "read"
    }


def test_emit_status_does_not_checkout_repository_contents() -> None:
    emit_steps = _workflow()["jobs"]["emit-status"]["steps"]
    assert not any(
        isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and step["uses"].startswith("actions/checkout@")
        for step in emit_steps
    )


def test_emit_status_and_its_artifact_do_not_run_after_cancellation() -> None:
    emit = _workflow()["jobs"]["emit-status"]
    assert emit.get("needs") == "scan"
    assert emit.get("if") == "${{ !cancelled() }}"
    uploads = [
        step
        for step in emit["steps"]
        if isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and step["uses"].startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert uploads[0].get("if") == (
        "${{ !cancelled() && steps.emit.outcome != 'skipped' }}"
    )


def test_sarif_publication_runs_after_reporter_failure() -> None:
    steps = _scan_steps()
    reporter_index = steps.index(_reporter_step())
    sarif_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and (
            step["uses"].startswith("actions/upload-artifact@")
            or step["uses"].startswith("github/codeql-action/upload-sarif@")
        )
    ]
    assert len(sarif_steps) == 2
    for index, step in sarif_steps:
        assert index > reporter_index
        assert step.get("if") == "${{ !cancelled() }}"


def _shell_quote(script: str) -> str:
    return shlex.quote(script)


# These tests execute the real emit-status workflow script through bash.
# The conftest live-system guard's `hermes update` heuristic false-positives
# here: "update" appears in the embedded workflow prose ("Update the
# affected dependencies") and "hermes" in the CI pytest temp-root path
# (hermes-pytest-tmproot-*). The tests are hermetic (tmp_path cwd +
# /tmp/osv-results symlink with cleanup, no repo/system mutation), which is
# the sanctioned escape hatch for this marker.
@pytest.mark.live_system_guard_bypass
def test_emit_status_fails_closed_on_missing_sarif(tmp_path: Path) -> None:
    emit = _workflow()["jobs"]["emit-status"]
    download = next(
        step
        for step in emit["steps"]
        if isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and step["uses"].startswith("actions/download-artifact@")
    )
    assert "continue-on-error" not in download, (
        "SARIF artifact download must not be allowed to fail silently"
    )
    run_script = next(
        step
        for step in emit["steps"]
        if isinstance(step, dict) and step.get("name") == "Emit review_status"
    )["run"]

    # Simulate the post-download state: artifact missing/empty. The emit
    # script must exit nonzero rather than emit a clean "[]" status.
    # The script hardcodes /tmp/osv-results/osv-results.sarif; sandbox the
    # whole check by making /tmp/osv-results point at our empty temp dir.
    sarif_dir = tmp_path / "osv-results"
    sarif_dir.mkdir()
    real_tmp = Path("/tmp/osv-results")
    if real_tmp.exists():  # pragma: no cover - defensive on shared runners
        pytest.skip("unrelated /tmp/osv-results present; test not hermetic")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"mkdir -p /tmp && ln -s {sarif_dir} /tmp/osv-results && "
            f"GITHUB_OUTPUT={tmp_path / 'github_output'} bash -c {_shell_quote(run_script)}; rc=$?; "
            "rm -f /tmp/osv-results; exit $rc",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,  # script writes review-status.json relatively
    )
    assert result.returncode != 0, (
        "emit script must fail when the SARIF artifact is missing"
    )


@pytest.mark.live_system_guard_bypass
def test_emit_status_succeeds_on_clean_sarif(tmp_path: Path) -> None:
    emit = _workflow()["jobs"]["emit-status"]
    run_script = next(
        step
        for step in emit["steps"]
        if isinstance(step, dict) and step.get("name") == "Emit review_status"
    )["run"]

    sarif_dir = tmp_path / "osv-results"
    sarif_dir.mkdir()
    (sarif_dir / "osv-results.sarif").write_text(
        json.dumps({"runs": [{"results": []}]}), encoding="utf-8"
    )
    real_tmp = Path("/tmp/osv-results")
    if real_tmp.exists():  # pragma: no cover - defensive on shared runners
        pytest.skip("unrelated /tmp/osv-results present; test not hermetic")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"mkdir -p /tmp && ln -s {sarif_dir} /tmp/osv-results && "
            f"GITHUB_OUTPUT={tmp_path / 'github_output'} bash -c {_shell_quote(run_script)}; rc=$?; "
            "rm -f /tmp/osv-results; exit $rc",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,  # script writes review-status.json relatively
    )
    assert result.returncode == 0, result.stderr
    assert "review_status=[]" in (tmp_path / "github_output").read_text(
        encoding="utf-8"
    )


@pytest.mark.live_system_guard_bypass
def test_emit_status_fails_on_corrupt_sarif(tmp_path: Path) -> None:
    emit = _workflow()["jobs"]["emit-status"]
    run_script = next(
        step
        for step in emit["steps"]
        if isinstance(step, dict) and step.get("name") == "Emit review_status"
    )["run"]

    sarif_dir = tmp_path / "osv-results"
    sarif_dir.mkdir()
    (sarif_dir / "osv-results.sarif").write_text("not json", encoding="utf-8")
    real_tmp = Path("/tmp/osv-results")
    if real_tmp.exists():  # pragma: no cover - defensive on shared runners
        pytest.skip("unrelated /tmp/osv-results present; test not hermetic")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"mkdir -p /tmp && ln -s {sarif_dir} /tmp/osv-results && "
            f"GITHUB_OUTPUT={tmp_path / 'github_output'} bash -c {_shell_quote(run_script)}; rc=$?; "
            "rm -f /tmp/osv-results; exit $rc",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,  # script writes review-status.json relatively
    )
    assert result.returncode != 0, (
        "emit script must fail when the SARIF is corrupt, not count 0 findings"
    )


def _ci_evaluate_run() -> str:
    ci = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    return next(
        step
        for job in ci["jobs"].values()
        if "evaluate" in str(job.get("steps", ""))
        for step in job["steps"]
        if step.get("name") == "Evaluate job results"
    )["run"]


def _run_aggregate_gate(needs: dict) -> subprocess.CompletedProcess[str]:
    """Run the ci.yml evaluate step through the real shell.

    Executing via bash (not by extracting the python body) means shell
    quoting bugs in the embedded script surface as failures here.
    """
    ci = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    evaluate = next(
        step
        for job in ci["jobs"].values()
        if "evaluate" in str(job.get("steps", ""))
        for step in job["steps"]
        if step.get("name") == "Evaluate job results"
    )
    return subprocess.run(
        ["bash", "-c", evaluate["run"]],
        input=json.dumps(needs),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "NEEDS": json.dumps(needs),
            "GITHUB_OUTPUT": "/dev/null",
        },
    )


def test_aggregate_gate_rejects_skipped_osv_scanner() -> None:
    script = _ci_evaluate_run()
    # The gate must explicitly reject a skipped osv-scanner result.
    assert "osv-scanner" in script
    assert "skipped" in script

    result = _run_aggregate_gate(
        {
            "tests": {"result": "success"},
            "osv-scanner": {"result": "skipped"},
        }
    )
    assert result.returncode != 0, (
        "aggregate gate must fail when osv-scanner is skipped"
    )


def test_aggregate_gate_passes_when_osv_scanner_succeeds() -> None:
    result = _run_aggregate_gate(
        {
            "tests": {"result": "success"},
            "osv-scanner": {"result": "success"},
            "tests-os": {"result": "skipped"},
        }
    )
    assert result.returncode == 0, result.stderr


def test_aggregate_gate_rejects_failed_job_through_shell() -> None:
    result = _run_aggregate_gate(
        {
            "tests": {"result": "success"},
            "osv-scanner": {"result": "failure"},
        }
    )
    assert result.returncode != 0
    assert "did not pass" in result.stdout
