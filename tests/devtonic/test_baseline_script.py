"""Behaviour tests for scripts/devtonic/baseline.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "devtonic" / "baseline.py"


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location("devtonic_baseline", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_parse_summary_counts_and_failures(baseline):
    output = "\n".join(
        [
            "..F.E",
            "FAILED tests/x/test_a.py::test_one - AssertionError: boom",
            "ERROR tests/x/test_b.py::test_two - RuntimeError",
            "ERROR    hermes_state:repair.py:866 state.db repair skipped",
            "=== 1 failed, 3 passed, 2 skipped, 1 error in 4.20s ===",
        ]
    )
    counts, summary, failures = baseline.parse_summary(output)
    assert counts == {"failed": 1, "passed": 3, "skipped": 2, "errors": 1}
    assert summary.startswith("1 failed, 3 passed")
    assert failures == ["tests/x/test_a.py::test_one", "tests/x/test_b.py::test_two"]


def test_parse_summary_no_tests(baseline):
    counts, summary, failures = baseline.parse_summary("=== no tests ran in 0.01s ===")
    assert counts == {}
    assert "no tests ran" in summary
    assert failures == []


def test_status_mapping(baseline):
    def make(rc):
        return baseline.TargetResult("t", rc, 0.0, {}, "", [])

    assert [make(rc).status for rc in (0, 1, 5, -1)] == ["pass", "fail", "no tests", "timeout"]


def test_render_table_rows(baseline):
    res = baseline.TargetResult("tests/cron", 0, 12.4, {"passed": 7, "skipped": 1}, "", [])
    table = baseline.render_table([res])
    assert table.splitlines()[2] == "| `tests/cron` | pass | 7 | 0 | 0 | 1 | 12 |"


def test_expand_target_splits_directories_into_files(baseline):
    files = baseline.expand_target("tests/devtonic")
    assert "tests/devtonic/test_baseline_script.py" in files
    assert all(f.endswith(".py") and "/test_" in f for f in files)
    node = "tests/devtonic/test_baseline_script.py::test_status_mapping"
    assert baseline.expand_target(node) == [node]


def test_run_isolated_aggregates_per_file_results(baseline, monkeypatch):
    results = {
        "a.py": baseline.TargetResult("a.py", 0, 1.0, {"passed": 3}, "", []),
        "b.py": baseline.TargetResult("b.py", 1, 1.0, {"passed": 1, "failed": 1}, "", ["b.py::t"]),
        "c.py": baseline.TargetResult("c.py", 5, 1.0, {}, "no tests ran", []),
    }
    monkeypatch.setattr(baseline, "expand_target", lambda _t: list(results))
    monkeypatch.setattr(baseline, "run_target", lambda f, *_a: results[f])
    res = baseline.run_isolated("tests/x", 10, [], 2)
    assert res.status == "fail"
    assert res.counts == {"passed": 4, "failed": 1}
    assert res.failures == ["b.py::t"]

    del results["b.py"]
    assert baseline.run_isolated("tests/x", 10, [], 2).status == "pass"


def test_main_runs_real_target_and_writes_json(baseline, tmp_path):
    out = tmp_path / "out.json"
    md = tmp_path / "summary.md"
    md.write_text("existing\n")
    # A single passing node from this file, run through the real subprocess path.
    node = f"{Path(__file__).resolve().relative_to(baseline.REPO_ROOT)}::test_status_mapping"
    rc = baseline.main([node, "--json", str(out), "--markdown", str(md), "--timeout", "300"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data[0]["status"] == "pass"
    assert data[0]["counts"]["passed"] == 1
    # --markdown appends (GITHUB_STEP_SUMMARY semantics) rather than overwriting.
    text = md.read_text()
    assert text.startswith("existing\n")
    assert "| pass | 1 | 0 | 0 | 0 |" in text


def test_render_markdown_includes_host_and_failures(baseline):
    ok = baseline.TargetResult("tests/a", 0, 1.0, {"passed": 2}, "", [])
    bad = baseline.TargetResult("tests/b", 1, 1.0, {"failed": 1}, "", ["tests/b/test_x.py::t"])
    text = baseline.render_markdown([ok, bad])
    first = text.splitlines()[0]
    assert first.startswith("Measured on ") and "Python " in first and "SQLite " in first
    assert "| `tests/a` | pass |" in text
    assert "- `tests/b`: `tests/b/test_x.py::t`" in text
    assert "Failures:" not in baseline.render_markdown([ok])
