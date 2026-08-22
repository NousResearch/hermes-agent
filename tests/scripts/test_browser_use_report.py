"""The browser_use scorecard's ``vs base`` column must be paired.

``evals/browser_use/orchestrate.py`` runs a fully crossed battery — every
(arm, task, model, rep) cell — so base and each treatment arm are measured on
the same work. Averaging each arm over its own ok runs and subtracting throws
that pairing away: when the arms fail on different cells the two means cover
different task sets, and an arm that resolves an extra expensive task is
penalised for having resolved it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PY = REPO_ROOT / "evals" / "browser_use" / "report.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("bu_report", REPORT_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bu_report"] = mod
    spec.loader.exec_module(mod)
    return mod


report = _load_report()


def _row(model, arm, task, rep, ok, tokens):
    return {
        "model": model,
        "arm": arm,
        "task": task,
        "rep": rep,
        "ok": ok,
        "total_tokens": tokens,
        "tool_calls": 1,
        "wall_s": 1.0,
    }


def _ok_by_cell(rows):
    """Mirror of the index main() builds, so the helper can be tested alone."""
    from collections import defaultdict

    idx = defaultdict(dict)
    for r in rows:
        if r.get("ok"):
            idx[(r["model"], r["arm"])][(r["task"], r["rep"])] = r["total_tokens"]
    return idx


# ── the defect this test exists for ───────────────────────────────────


def test_unpaired_delta_reverses_sign_when_arms_fail_on_different_cells():
    """base fails the one expensive task; the arm resolves it and looks worse.

    base:  t1 100k ok, t2 100k ok, t3 FAILED     -> own-runs mean 100k
    arm:   t1  50k ok, t2  50k ok, t3 400k ok    -> own-runs mean 166.7k

    Unpaired that reads +67% (a regression). On the two cells both arms
    resolved the arm costs half as much, which is what the battery measured.
    """
    rows = [
        _row("m", "base", "t1", 0, True, 100_000),
        _row("m", "base", "t2", 0, True, 100_000),
        _row("m", "base", "t3", 0, False, 0),
        _row("m", "pr", "t1", 0, True, 50_000),
        _row("m", "pr", "t2", 0, True, 50_000),
        _row("m", "pr", "t3", 0, True, 400_000),
    ]
    idx = _ok_by_cell(rows)

    pct, n_paired = report.paired_token_delta(idx, "m", "pr")
    assert n_paired == 2
    assert pct == pytest.approx(-50.0)

    # The old statistic, kept here so the regression is explicit.
    unpaired = (166_666.67 - 100_000) / 100_000 * 100
    assert unpaired > 0 > pct


def test_paired_delta_uses_only_cells_both_arms_resolved():
    rows = [
        _row("m", "base", "t1", 0, True, 200),
        _row("m", "base", "t2", 0, True, 100),
        _row("m", "pr", "t1", 0, True, 100),
        _row("m", "pr", "t2", 0, False, 0),
    ]
    pct, n_paired = report.paired_token_delta(_ok_by_cell(rows), "m", "pr")
    assert n_paired == 1
    assert pct == pytest.approx(-50.0)  # 100 vs 200 on t1, t2 excluded entirely


def test_reps_are_separate_cells():
    """Two reps of one task are two cells, not one — pairing is (task, rep)."""
    rows = [
        _row("m", "base", "t1", 0, True, 100),
        _row("m", "base", "t1", 1, True, 100),
        _row("m", "pr", "t1", 0, True, 50),
        _row("m", "pr", "t1", 1, False, 0),
    ]
    pct, n_paired = report.paired_token_delta(_ok_by_cell(rows), "m", "pr")
    assert n_paired == 1
    assert pct == pytest.approx(-50.0)


def test_no_shared_ok_cell_reports_none_rather_than_a_number():
    rows = [
        _row("m", "base", "t1", 0, True, 100),
        _row("m", "pr", "t2", 0, True, 50),
    ]
    pct, n_paired = report.paired_token_delta(_ok_by_cell(rows), "m", "pr")
    assert pct is None
    assert n_paired == 0


def test_zero_paired_base_tokens_does_not_divide_by_zero():
    rows = [
        _row("m", "base", "t1", 0, True, 0),
        _row("m", "pr", "t1", 0, True, 50),
    ]
    pct, n_paired = report.paired_token_delta(_ok_by_cell(rows), "m", "pr")
    assert pct is None
    assert n_paired == 1


def test_arms_measured_on_identical_cells_are_unaffected():
    """The fix must not move a number when nothing failed — the common case."""
    rows = [
        _row("m", "base", "t1", 0, True, 100),
        _row("m", "base", "t2", 0, True, 300),
        _row("m", "pr", "t1", 0, True, 50),
        _row("m", "pr", "t2", 0, True, 150),
    ]
    pct, n_paired = report.paired_token_delta(_ok_by_cell(rows), "m", "pr")
    assert n_paired == 2
    assert pct == pytest.approx(-50.0)


# ── end to end through main() ─────────────────────────────────────────


def test_main_prints_pair_count_and_paired_delta(tmp_path, capsys):
    rows = [
        _row("qwen3-coder-30b", "base", "t1", 0, True, 100_000),
        _row("qwen3-coder-30b", "base", "t2", 0, True, 100_000),
        _row("qwen3-coder-30b", "base", "t3", 0, False, 0),
        _row("qwen3-coder-30b", "pr", "t1", 0, True, 50_000),
        _row("qwen3-coder-30b", "pr", "t2", 0, True, 50_000),
        _row("qwen3-coder-30b", "pr", "t3", 0, True, 400_000),
    ]
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report.main([str(p)])
    out = capsys.readouterr().out

    pr_line = next(ln for ln in out.splitlines() if " pr " in f" {ln} ")
    assert "-50%" in pr_line, pr_line
    assert "2/3" in pr_line, pr_line  # 2 paired cells out of the arm's 3 ok runs
    assert "+67%" not in out
