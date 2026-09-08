"""Advisory failures cannot soften mandatory checks or masquerade as a clean diff."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]


def _workflow(name):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(
        (_REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )


def _shell(command, tmp_path, **environment):
    bash = shutil.which("bash")
    assert bash, "These workflow command tests require Bash"
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", command],
        cwd=tmp_path,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("lint_result", ["success", "failure", "cancelled"])
def test_advisory_is_separate_and_required_lint_still_fails_closed(tmp_path, lint_result):
    jobs = _workflow("lint.yml")["jobs"]
    mandatory = jobs["windows-footguns"]
    advisory = jobs["public-surface-advisory"]
    for step in advisory["steps"]:
        if step.get("id") in {"checkout", "surface"}:
            assert step["continue-on-error"] is True
    assert not mandatory.get("continue-on-error", False)
    for checker in ("check-windows-footguns.py", "check_compat_pointers.py"):
        step = next(step for step in mandatory["steps"] if checker in step.get("run", ""))
        assert not step.get("continue-on-error", False)
        failed = _shell("python() { return 23; };\n" + step["run"], tmp_path)
        assert failed.returncode == 23
    assert all("check_public_surface.py" not in step.get("run", "") for step in mandatory["steps"])
    assert all(0 < step["timeout-minutes"] for step in advisory["steps"])
    assert sum(step["timeout-minutes"] for step in advisory["steps"]) < advisory["timeout-minutes"]

    orchestrator = _workflow("ci.yaml")["jobs"]
    assert not orchestrator["lint"].get("continue-on-error", False)
    gate = orchestrator["all-checks-pass"]
    assert "lint" in gate["needs"]
    evaluate = next(step for step in gate["steps"] if step["name"] == "Evaluate job results")
    source = evaluate["run"].split('python3 -c "', 1)[1].rsplit('"', 1)[0]
    source = source.replace(r'\"', '"').replace(
        "'$GITHUB_OUTPUT'", repr(str(tmp_path / "gate-output"))
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        input=json.dumps({"lint": {"result": lint_result}}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert (result.returncode == 0) is (lint_result == "success"), result.stdout + result.stderr


@pytest.mark.parametrize("scenario", ["success", "findings", "fetch-failure", "checker-failure", "timeout", "checkout-failure"])
def test_advisory_reports_incomplete_unless_analysis_finishes(tmp_path, scenario):
    job = _workflow("lint.yml")["jobs"]["public-surface-advisory"]
    analyze = next(step for step in job["steps"] if step.get("id") == "surface")
    report = next(step for step in job["steps"] if step["name"] == "Report advisory completeness")
    assert analyze["if"] == "steps.checkout.outcome == 'success'"
    assert report["if"] == "always()"
    assert "steps.surface.outputs.completed" in report["env"]["ADVISORY_COMPLETED"]
    output = tmp_path / "step-output"
    summary = tmp_path / "step-summary"
    # Command doubles keep the actual workflow shell and exit propagation under test.
    # Exit 124 models a terminated fetch; like runner timeout, it leaves no completion output.
    commands = """
git() {
  if [ "$1" = fetch ]; then
    case "$SCENARIO" in fetch-failure) return 1;; timeout) return 124;; esac
  fi
  return 0
}
python() {
  if [ "$SCENARIO" = checker-failure ]; then return 2; fi
  if [ "$SCENARIO" = findings ]; then echo 'public-surface: 1 public name dropped'; fi
}
"""
    run = None
    if scenario != "checkout-failure":
        run = _shell(
            commands + analyze["run"], tmp_path,
            SCENARIO=scenario, BASE_REF="main", GITHUB_OUTPUT=output.as_posix(),
        )
    completed = output.exists() and "completed=true" in output.read_text(encoding="utf-8")
    assert completed is (scenario in {"success", "findings"})
    if run is not None:
        assert (run.returncode == 0) is completed
    result = _shell(
        report["run"], tmp_path,
        ADVISORY_COMPLETED="true" if completed else "", GITHUB_STEP_SUMMARY=summary.as_posix(),
    )
    assert result.returncode == 0, result.stderr
    message = summary.read_text(encoding="utf-8")
    assert ("INCOMPLETE" in message) is (not completed)
    if completed:
        assert "completed" in message
        assert "clean" not in message.lower()
    else:
        assert "::warning::" in result.stdout
    if scenario == "findings":
        assert "1 public name dropped" in run.stdout
