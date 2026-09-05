"""Tests for scripts/ci/gate_results.py.

``all-checks-pass`` is the only check branch protection requires, so the
contract asserted here is what may reach ``main``: a required job passes
the gate only when it reported ``success`` or ``skipped``. ``skipped`` is
a pass because that is how a path-filtered lane (``if:
needs.detect.outputs.python == 'true'``) reports on a PR that does not
touch its area. Anything else — ``cancelled`` above all, which is what a
sub-workflow's concurrency group produces when two pushes land on ``main``
minutes apart — must block the merge.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "gate_results.py"
_spec = importlib.util.spec_from_file_location("gate_results", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load gate_results.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _needs(**results: str) -> dict[str, dict]:
    return {name: {"result": result} for name, result in results.items()}


def _run(needs: dict, monkeypatch, github_output: Path | None = None) -> int:
    monkeypatch.setattr(_mod.sys, "stdin", io.StringIO(json.dumps(needs)))
    if github_output is None:
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    else:
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    return _mod.main()


# ─── the gate contract ────────────────────────────────────────────────


def test_success_and_skipped_pass_the_gate():
    """Path-filtered lanes report ``skipped``; that must stay a pass."""
    results = _mod.compact_results(_needs(tests="success", docker_lint="skipped"))
    assert _mod.failing_jobs(results) == []


@pytest.mark.parametrize("result", ["cancelled", "failure", "timed_out", "neutral", ""])
def test_any_non_passing_result_fails_the_gate(result):
    results = _mod.compact_results(_needs(tests="success", lint=result))
    assert _mod.failing_jobs(results) == ["lint"]


def test_unknown_result_key_fails_the_gate():
    """The gate never guesses that a job it cannot read was green."""
    assert _mod.failing_jobs(_mod.compact_results({"tests": {}})) == ["tests"]


def test_failing_jobs_are_reported_sorted():
    results = _mod.compact_results(_needs(tests="cancelled", lint="failure", docs="success"))
    assert _mod.failing_jobs(results) == ["lint", "tests"]


def test_render_marks_only_passing_results_green():
    lines = _mod.render_lines(_mod.compact_results(_needs(a="skipped", b="cancelled")))
    assert lines == ["✅ a: skipped", "❌ b: cancelled"]


# ─── CLI ──────────────────────────────────────────────────────────────


def test_cli_exits_nonzero_on_a_cancelled_job(monkeypatch, capsys):
    exit_code = _run(_needs(tests="cancelled", lint="success"), monkeypatch)
    assert exit_code == 1
    assert "::error::1 job(s) did not pass: tests" in capsys.readouterr().out


def test_cli_exits_zero_when_every_job_passed_or_skipped(monkeypatch, capsys):
    exit_code = _run(_needs(tests="success", lint="skipped"), monkeypatch)
    assert exit_code == 0
    assert "All checks passed (or were skipped)" in capsys.readouterr().out


def test_cli_emits_compact_needs_json_for_the_comment_assembler(monkeypatch, tmp_path, capsys):
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    _run(_needs(tests="cancelled", lint="success"), monkeypatch, github_output=output)

    written = output.read_text(encoding="utf-8")
    assert written.endswith("\n")
    key, _, payload = written.strip().partition("=")
    assert key == "needs-json"
    assert json.loads(payload) == {"tests": "cancelled", "lint": "success"}
    assert written.strip() in capsys.readouterr().out


def test_cli_refuses_a_gate_it_was_handed_no_results_for(monkeypatch, capsys):
    """An empty ``needs`` context is an absence of evidence, not a pass.

    Same shape as the run-level defect in ``live_comment.py``: nothing
    reported must never settle as everything green.
    """
    exit_code = _run({}, monkeypatch)
    assert exit_code == 1
    assert "::error::the gate received no job results at all" in capsys.readouterr().out
