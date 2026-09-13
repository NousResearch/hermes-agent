"""Behavior tests for the strict CI aggregate gate (scripts/ci/required_results.py)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "required_results.py"

_spec = importlib.util.spec_from_file_location("required_results", SCRIPT)
required_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(required_results)

evaluate_gate = required_results.evaluate_gate
PR_ONLY_JOBS = required_results.PR_ONLY_JOBS
DEFERRED_JOBS = required_results.DEFERRED_JOBS
EXCLUDED_JOBS = required_results.EXCLUDED_JOBS


def needs_of(entries):
    """Build the needs-context shape ci.yaml's all-checks-pass receives."""
    return {name: {"result": result} for name, result in entries}


def test_non_release_failure_fails_gate_skip_counts_as_success():
    base = [("detect", "success"), ("tests", "success"), ("lint", "success")]

    assert evaluate_gate(needs_of(base))["ok"] is True

    with_failure = evaluate_gate(needs_of([*base, ("js-tests", "failure")]))
    assert with_failure["ok"] is False
    assert with_failure["failed"] == ["js-tests"]

    # Skipped lanes are fine when not releasing (e.g. docs-only PR).
    with_skips = evaluate_gate(
        needs_of([("detect", "success"), ("tests", "skipped"), ("rust-tests", "skipped"), ("lint", "success")])
    )
    assert with_skips["ok"] is True
    assert with_skips["allowed_skips"] == ["rust-tests", "tests"]


def test_exclusion_sets_name_exactly_pr_only_and_deferred_jobs():
    assert PR_ONLY_JOBS == ("history-check", "lockfile-diff", "supply-chain", "review-labels")
    assert DEFERRED_JOBS == ("e2e-desktop",)
    assert EXCLUDED_JOBS == set(PR_ONLY_JOBS) | set(DEFERRED_JOBS)


def test_release_only_excluded_jobs_may_skip():
    needs = needs_of(
        [
            ("detect", "success"),
            ("tests", "success"),
            ("js-tests", "success"),
            ("osv-scanner", "success"),
            # PR-only: never ran — release runs are push/tag events.
            ("history-check", "skipped"),
            ("lockfile-diff", "skipped"),
            ("supply-chain", "skipped"),
            ("review-labels", "skipped"),
            # Deferred: Desktop E2E stays disabled.
            ("e2e-desktop", "skipped"),
        ]
    )
    verdict = evaluate_gate(needs, release=True)
    assert verdict["ok"] is True
    assert verdict["allowed_skips"] == ["e2e-desktop", "history-check", "lockfile-diff", "review-labels", "supply-chain"]


def test_release_osv_skipped_is_a_failure_not_advisory():
    # Advisory FINDINGS do not make the EXECUTION advisory: the OSV scan
    # must run to success on a release candidate.
    verdict = evaluate_gate(needs_of([("detect", "success"), ("osv-scanner", "skipped")]), release=True)
    assert verdict["ok"] is False
    assert verdict["failed"] == ["osv-scanner"]


def test_release_skipped_required_job_fails_gate():
    needs = needs_of(
        [
            ("detect", "success"),
            ("tests", "success"),
            # Everything tagged python (and python_prod/frontend) skipped on
            # the release run: the strict gate must not wave this through.
            ("tests-os", "skipped"),
            ("lint", "skipped"),
            ("js-tests", "skipped"),
            ("infographic-check", "success"),
        ]
    )
    verdict = evaluate_gate(needs, release=True)
    assert verdict["ok"] is False
    assert verdict["failed"] == ["js-tests", "lint", "tests-os"]


def test_every_non_success_result_fails_in_release_mode():
    # cancelled / with-status / action_required / unknown / missing all
    # fail; only `success` passes and only `skipped` on an excluded job
    # is tolerated.
    needs = needs_of(
        [
            ("a", "cancelled"),
            ("b", "with-status"),
            ("c", "action_required"),
            ("d", "unknown"),
        ]
    )
    verdict = evaluate_gate(needs, release=True)
    assert verdict["ok"] is False
    assert verdict["failed"] == ["a", "b", "c", "d"]

    # A needs entry with no result key at all (should never happen, but
    # must not silently pass).
    verdict = evaluate_gate({"x": {}}, release=True)
    assert verdict["ok"] is False
    assert verdict["failed"] == ["x"]


def test_release_failure_fails_gate_even_beside_allowed_skips():
    needs = needs_of(
        [
            ("detect", "success"),
            ("tests", "failure"),
            ("history-check", "skipped"),
            ("e2e-desktop", "skipped"),
        ]
    )
    verdict = evaluate_gate(needs, release=True)
    assert verdict["ok"] is False
    assert verdict["failed"] == ["tests"]


def test_empty_needs_fails_closed():
    assert evaluate_gate({}, release=True)["ok"] is False
    assert evaluate_gate({}, release=False)["ok"] is False
    assert evaluate_gate(None, release=True)["ok"] is False


def test_release_full_pipeline_all_green_passes():
    # What ci.yaml's all-checks-pass actually needs. A release run forces
    # the classifier lanes true, so a healthy candidate looks like this.
    all_green = [
        "detect", "tests", "tests-os", "lint", "js-tests", "installer-tests",
        "rust-tests", "bootstrap-installer", "e2e-desktop", "docs-site",
        "history-check", "contributor-check", "uv-lockfile", "infographic-check",
        "case-collision-check", "lazy-deps-guard", "lockfile-diff",
        "docker-lint", "profile-artifact-check", "icons-freshness-check",
        "supply-chain", "review-labels", "osv-scanner",
    ]
    needs = needs_of((name, "success") for name in all_green)
    assert evaluate_gate(needs, release=True)["ok"] is True
    assert evaluate_gate(needs, release=False)["ok"] is True


def test_compact_results_maps_every_job_to_its_result():
    needs = needs_of([("detect", "success"), ("tests", "skipped")])
    assert required_results.compact_results(needs) == {"detect": "success", "tests": "skipped"}
    assert required_results.compact_results(None) == {}


def test_render_report_reports_honest_skip_counts():
    needs = needs_of([("tests", "skipped"), ("lint", "failure"), ("detect", "success")])
    verdict = evaluate_gate(needs, release=True)
    lines = "\n".join(required_results.render_report(needs, verdict))
    assert "⏭️ tests: skipped" in lines
    assert "❌ lint: failure" in lines
    assert "::error::2 job(s) failed: lint, tests" in lines


def test_cli_stdin_exit_codes_and_output(tmp_path):
    gh_output = tmp_path / "github-output.txt"
    green = needs_of([("detect", "success"), ("tests", "success"), ("e2e-desktop", "skipped")])

    def run(needs, release, env_file):
        # Inherit the full environment: on Windows a stripped env breaks
        # subprocess startup (SystemRoot etc.).
        env = dict(os.environ)
        env.pop("GITHUB_OUTPUT", None)
        if env_file is not None:
            env["GITHUB_OUTPUT"] = str(env_file)
        return subprocess.run(
            [sys.executable, str(SCRIPT)] + (["--release"] if release else []),
            input=json.dumps(needs),
            capture_output=True,
            text=True,
            env=env,
        )

    ok = run(green, release=True, env_file=gh_output)
    assert ok.returncode == 0
    assert "All checks passed" in ok.stdout
    assert "needs-json=" in ok.stdout
    assert "needs-json=" in gh_output.read_text(encoding="utf-8")

    bad = run(needs_of([("tests", "skipped")]), release=True, env_file=None)
    assert bad.returncode == 1

    failed = run(needs_of([("tests", "failure")]), release=False, env_file=None)
    assert failed.returncode == 1
