"""Invariant tests for the tool-perf A/B credit analysis (issues 111237, 110671).

The analysis stage of `evals/toolperf_abeval` is what decides whether a tweak is credited,
and none of it was covered before. These tests pin the parts a reviewer cannot re-derive by
eye: how a scheduled cell is accounted for when it crashed, lost its trace or lost its
oracle; that the arms are paired by run_id; that the interval is resampled over TASKS (so
extra reps buy precision, not significance); and that a verdict is fail-closed about an
incomplete battery, an unknown battery identity and an unattributable tree.

Everything runs on tmp_path fixtures: no model call, no network, no subprocess (except one
test that asserts `tree_identity` does not spawn one).
"""
import json
import subprocess
from pathlib import Path

import pytest

from evals.toolperf_abeval import credit

FINGERPRINT = "fp-battery-v1"
TASKS = [f"t{i}" for i in range(1, 7)]
RUN_IDS = [f"{t}-r0" for t in TASKS]


def _metrics(llm: float, tools: float = 2, errs: float = 0) -> dict:
    return {"llm": llm, "tools": tools, "errs": errs, "retries": 0, "kb": 1}


def _pair(task: str, rep: int, *, baseline_llm: float = 10.0, fixes_llm: float = 5.0,
          baseline_ok: bool = True, fixes_ok: bool = True):
    """Two RECORDED cells, one per arm, with controlled metric values."""
    return (
        credit.Cell(run_id=f"{task}-r{rep}", task=task, rep=rep, arm="baseline",
                    status=credit.RECORDED, oracle_ok=baseline_ok,
                    metrics=_metrics(baseline_llm), wall_s=10.0),
        credit.Cell(run_id=f"{task}-r{rep}", task=task, rep=rep, arm="fixes",
                    status=credit.RECORDED, oracle_ok=fixes_ok,
                    metrics=_metrics(fixes_llm), wall_s=10.0),
    )


def _trace(n_llm: int = 2, n_tools: int = 2, kb: int = 1) -> str:
    """A minimal NeMo Relay ATOF trace: llm scopes plus tool start/end pairs."""
    events = [{"kind": "scope", "category": "llm", "scope_category": "end"}] * n_llm
    for _ in range(n_tools):
        events.append({"kind": "scope", "category": "tool", "scope_category": "start",
                       "name": "terminal"})
        events.append({"kind": "scope", "category": "tool", "scope_category": "end",
                       "name": "terminal", "data": "x" * (kb * 1024),
                       "metadata": {"status": "ok"}})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _row(run_id: str, task: str, rep: int, arm: str, *, tail: str = "ok",
         battery="fp-battery-v1", wall_s: float = 1.0) -> dict:
    return {"run_id": run_id, "task": task, "rep": rep, "arm": arm, "model": "m",
            "wall_s": wall_s, "exit": 0, "tree": "deadbeef", "pythonpath": "/tree",
            "battery": battery, "tail": tail}


def _arm(tmp_path: Path, arm: str, scheduled, *, recorded=None, traced=None,
         oracle=credit.ORACLE_OK, n_llm: int = 2, tail: str = "ok",
         battery="fp-battery-v1", rows_extra=None) -> dict:
    """Write one arm's results dir, then classify it exactly the way `ab_eval credit` does.

    `recorded` = run_ids that have a meta row (the runner writes no row for a startup
    crash); `traced` = run_ids that have a trace file; `oracle` = one outcome for the arm
    or a mapping per run_id.
    """
    directory = tmp_path / arm
    directory.mkdir(parents=True, exist_ok=True)
    recorded = list(scheduled) if recorded is None else list(recorded)
    traced = recorded if traced is None else list(traced)
    lines = []
    for run_id in recorded:
        task, rep = credit.split_run_id(run_id)
        row = _row(run_id, task, rep, arm, tail=tail, battery=battery)
        if rows_extra:
            row.update(rows_extra.get(run_id, {}))
        lines.append(json.dumps(row))
    if lines:
        (directory / "meta.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for run_id in traced:
        (directory / f"{run_id}.atof.jsonl").write_text(_trace(n_llm=n_llm), encoding="utf-8")
    outcomes = {rid: oracle for rid in scheduled} if isinstance(oracle, str) else dict(oracle)
    return credit.classify_arm(arm, scheduled, credit.load_meta_rows(directory / "meta.jsonl"),
                               directory, outcomes, tree="deadbeef")


def _arms(pairs):
    """(baseline, fixes) keyed by run_id — the shape `classify_arm` returns."""
    return ({b.run_id: b for b, _ in pairs}, {f.run_id: f for _, f in pairs})


def _verdict(baseline, fixes, scheduled, **overrides) -> dict:
    """The same composition `ab_eval.credit` performs, with a small exact bootstrap."""
    pairs, unpaired_baseline, unpaired_fixes = credit.pair_cells(baseline, fixes)
    by_arm = {arm: credit.completeness(scheduled, cells)
              for arm, cells in (("baseline", baseline), ("fixes", fixes))}
    kwargs = dict(metric="ok", guardrails=(), margin=0.10, alpha=0.05, comparisons=1,
                  resamples=500, seed=11, min_pairs=1, min_tasks=1, battery="unchanged",
                  battery_fingerprint=FINGERPRINT, model="m")
    kwargs.update(overrides)
    return credit.credit_verdict(
        pairs, by_arm, unpaired={"baseline": unpaired_baseline, "fixes": unpaired_fixes},
        **kwargs)


def test_pairs_are_formed_by_run_id_not_by_order(tmp_path):
    shared = [f"{t}-r0" for t in TASKS[:5]]
    baseline = _arm(tmp_path, "baseline", shared + ["t6-r0", "t6-r1"])
    fixes = _arm(tmp_path, "fixes", list(reversed(shared)) + ["t5-r1"])

    pairs, unpaired_baseline, unpaired_fixes = credit.pair_cells(baseline, fixes)

    assert [b.run_id for b, _ in pairs] == shared
    assert unpaired_baseline == ["t6-r0", "t6-r1"]
    assert unpaired_fixes == ["t5-r1"]


def test_dropped_cell_is_unaccounted_and_withholds(tmp_path):
    scheduled = [f"{t}-r{rep}" for t in TASKS[:3] for rep in range(2)]
    baseline = _arm(tmp_path, "baseline", scheduled)
    # The runner prints INFRA-CRASH and writes NO row, so this cell has no evidence at all:
    # it must not be silently treated as a harmless flake.
    fixes = _arm(tmp_path, "fixes", scheduled, recorded=scheduled[:-1])

    assert fixes["t3-r1"].status == credit.UNACCOUNTED
    ledger = credit.completeness(scheduled, fixes)
    assert ledger["scheduled"] == 6
    assert ledger["by_status"][credit.RECORDED] == 5

    verdict = _verdict(baseline, fixes, scheduled)
    assert (verdict["verdict"], verdict["reason"]) == (credit.WITHHELD, credit.R_INCOMPLETE)
    assert verdict["counts"]["scheduled"] == 12
    assert verdict["arms"]["fixes"]["unaccounted"] == 1

    # A cell that DECLARES the crash is accounted for: kept in the ledger, and it still
    # cannot be scored as a capability result either way.
    declared = _arm(tmp_path, "declared", ["t4-r0"],
                    rows_extra={"t4-r0": {"status": credit.INFRA_CRASH}})
    assert declared["t4-r0"].status == credit.INFRA_CRASH
    closed = credit.completeness(["t4-r0"], declared)
    assert closed["unaccounted"] == 0
    assert closed["by_status"][credit.INFRA_CRASH] == 1


def test_missing_trace_is_not_scored_as_zero_waste(tmp_path):
    scheduled = ["t1-r0", "t1-r1", "t2-r0"]
    baseline = _arm(tmp_path, "baseline", scheduled)
    fixes = _arm(tmp_path, "fixes", scheduled, traced=["t1-r0", "t2-r0"])

    vanished = fixes["t1-r1"]
    assert vanished.status == credit.MISSING_TRACE
    assert vanished.metrics is None, "a vanished trace is no evidence, not zero waste"

    pairs, _, _ = credit.pair_cells(baseline, fixes)
    assert [b.run_id for b, _ in pairs if b.task == "t1"] == ["t1-r0"]
    # Both surviving traces are identical, so the paired delta is exactly zero. Had the
    # traceless run been scored as a zero-waste run, fixes would have looked 50% better.
    assert credit.per_task_delta(pairs, "llm")["t1"] == 0.0

    verdict = _verdict(baseline, fixes, scheduled)
    assert verdict["counts"]["missing_trace"] == 1
    assert verdict["counts"]["pairs"] == 2
    assert (verdict["verdict"], verdict["reason"]) == (credit.WITHHELD, credit.R_INCOMPLETE)


def test_oracle_exception_is_not_a_task_failure(tmp_path):
    scheduled = ["t1-r0", "t2-r0"]
    baseline = _arm(tmp_path, "baseline", scheduled)
    fixes = _arm(tmp_path, "fixes", scheduled,
                 oracle={"t1-r0": credit.ORACLE_RAISED, "t2-r0": credit.ORACLE_OK})

    broken = fixes["t1-r0"]
    assert broken.status == credit.ORACLE_ERROR
    assert broken.oracle_ok is None, "a raised oracle is unknown, not False"
    assert broken.metrics is None, "an unusable cell contributes no metric evidence either"

    ledger = credit.completeness(scheduled, fixes)
    assert ledger["by_status"][credit.ORACLE_ERROR] == 1
    assert ledger["unaccounted"] == 0

    pairs, unpaired_baseline, _ = credit.pair_cells(baseline, fixes)
    assert [b.run_id for b, _ in pairs] == ["t2-r0"]
    assert unpaired_baseline == ["t1-r0"], "it leaves the mean, it is not a scored failure"

    verdict = _verdict(baseline, fixes, scheduled)
    assert verdict["counts"]["oracle_error"] == 1
    assert (verdict["verdict"], verdict["reason"]) == (credit.WITHHELD, credit.R_INCOMPLETE)


def test_credit_is_withheld_below_min_pairs_and_min_tasks(tmp_path):
    scheduled = [f"{t}-r0" for t in TASKS]
    baseline = _arm(tmp_path, "baseline", scheduled, oracle=credit.ORACLE_FAIL)
    fixes = _arm(tmp_path, "fixes", scheduled, oracle=credit.ORACLE_OK)

    # Control: the same evidence does credit once the declared bar is met, so the two
    # withholds below are caused by the bar and not by a broken fixture.
    control = _verdict(baseline, fixes, scheduled, min_pairs=5, min_tasks=5)
    assert control["verdict"] == credit.CREDITS
    assert control["primary"]["lo"] > 0

    few_tasks = _verdict(baseline, fixes, scheduled, min_pairs=5, min_tasks=7)
    assert (few_tasks["verdict"], few_tasks["reason"]) == (credit.WITHHELD,
                                                           credit.R_INSUFFICIENT)

    subset = scheduled[:2]
    few_pairs = _verdict({k: v for k, v in baseline.items() if k in subset},
                         {k: v for k, v in fixes.items() if k in subset}, subset,
                         min_pairs=3)
    assert (few_pairs["verdict"], few_pairs["reason"]) == (credit.WITHHELD,
                                                           credit.R_INSUFFICIENT)


def test_bootstrap_is_deterministic_under_seed():
    per_task = {t: v for t, v in zip(TASKS, (0.2, -0.1, 0.3, 0.0, 0.1, 0.25))}

    first = credit.cluster_bootstrap_ci(per_task, 0.05, 500, 3)
    same_seed = credit.cluster_bootstrap_ci(per_task, 0.05, 500, 3)
    other_seed = credit.cluster_bootstrap_ci(per_task, 0.05, 500, 4)

    assert first == same_seed
    assert first[2] == pytest.approx(sum(per_task.values()) / len(per_task))
    assert abs(first[0] - other_seed[0]) < 0.15
    assert abs(first[1] - other_seed[1]) < 0.15


def test_cluster_resampling_ignores_reps():
    """Reps are not independent evidence: tripling them must not narrow the interval."""
    targets = {1: 0.0, 2: 0.5, 3: 0.0, 4: 0.25, 5: 0.75, 6: 0.0}
    spread = {0.0: (0.25, 0.0, -0.25), 0.5: (0.25, 0.5, 0.75),
              0.25: (0.0, 0.25, 0.5), 0.75: (0.5, 0.75, 1.0)}

    def pairs_for(reps: int):
        pairs = []
        for index, target in targets.items():
            values = spread[target][:reps] if reps > 1 else (target,)
            for rep, value in enumerate(values):
                pairs.append(_pair(f"t{index}", rep, fixes_llm=10.0 * (1.0 - value)))
        return pairs

    one = credit.per_task_delta(pairs_for(1), "llm")
    three = credit.per_task_delta(pairs_for(3), "llm")

    assert one == three == {f"t{i}": v for i, v in targets.items()}
    assert credit.cluster_bootstrap_ci(one, 0.05, 500, 5) == \
        credit.cluster_bootstrap_ci(three, 0.05, 500, 5)


def test_guardrail_breach_denies_a_positive_primary_delta():
    # The primary metric (ok) improves on every task, so only a guardrail can deny this.
    healthy = [_pair(t, 0, baseline_ok=False, fixes_ok=True,
                     baseline_llm=10.0, fixes_llm=9.0) for t in TASKS]
    wrecked = [*healthy[:5],
               _pair("t6", 0, baseline_ok=False, fixes_ok=True,
                     baseline_llm=10.0, fixes_llm=20.0)]

    credited = _verdict(*_arms(healthy), RUN_IDS, guardrails=["llm"], min_pairs=5,
                        min_tasks=5)
    assert credited["verdict"] == credit.CREDITS
    assert credited["guardrails"]["llm"]["lo"] > -0.10

    breached = _verdict(*_arms(wrecked), RUN_IDS, guardrails=["llm"], min_pairs=5,
                        min_tasks=5)
    assert breached["primary"]["lo"] > 0, "the primary signal is positive"
    assert breached["primary"]["task_deltas"]["t6"] == 1.0
    assert (breached["verdict"], breached["reason"]) == (credit.DENIES,
                                                         credit.R_GUARDRAIL)
    assert breached["guardrails"]["llm"]["breached_tasks"] == ["t6"]


def test_denied_and_withheld_candidates_are_retained(tmp_path):
    path = tmp_path / "results" / "verdicts.jsonl"
    improved = _arms([_pair(t, 0, baseline_ok=False, fixes_ok=True) for t in TASKS])
    tied = _arms([_pair(t, 0) for t in TASKS])

    credited = _verdict(*improved, RUN_IDS, min_pairs=5, min_tasks=5)
    denied = _verdict(*tied, RUN_IDS)
    withheld = _verdict(*tied, RUN_IDS, battery="unverified")

    assert [row["verdict"] for row in (credited, denied, withheld)] == [
        credit.CREDITS, credit.DENIES, credit.WITHHELD]

    credit.write_verdicts(path, [credited, denied, withheld])
    credit.write_verdicts(path, [denied])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert [row["verdict"] for row in rows] == [
        credit.CREDITS, credit.DENIES, credit.WITHHELD, credit.DENIES]
    assert rows[1]["reason"] == credit.R_NO_EFFECT
    assert rows[2]["reason"] == credit.R_BATTERY_UNVERIFIED


def test_battery_fingerprint_is_a_contract_not_a_snapshot():
    base = credit.battery_fingerprint(["t1", "t2"], ["oracle_a", "oracle_b"])
    # oracle identity is a set: reordering it is not a battery change
    assert base == credit.battery_fingerprint(["t1", "t2"], ["oracle_b", "oracle_a"])
    # task order, task set, oracle set and battery version are all identity-bearing, so a
    # silent edit to the exam cannot pass as "the candidate got better"
    assert base != credit.battery_fingerprint(["t2", "t1"], ["oracle_a", "oracle_b"])
    assert base != credit.battery_fingerprint(["t1", "t2"], ["oracle_a"])
    assert base != credit.battery_fingerprint(
        ["t1", "t2"], ["oracle_a", "oracle_b"], version="2")
    assert base == credit.battery_fingerprint(["t1", "t2"], ["oracle_a", "oracle_b"])


def test_verdict_artifact_carries_no_trace_free_text(tmp_path):
    marker = "TRACE-TAIL-MARKER-9f31c"
    scheduled = [f"{t}-r0" for t in TASKS]
    baseline = _arm(tmp_path, "baseline", scheduled, tail=f"agent output {marker}", n_llm=3)
    fixes = _arm(tmp_path, "fixes", scheduled, tail=f"agent output {marker}", n_llm=1)

    raw = (tmp_path / "baseline" / "meta.jsonl").read_text(encoding="utf-8")
    assert marker in raw, "the fixture must really carry trace text"

    path = tmp_path / "verdicts.jsonl"
    credit.write_verdicts(path, [_verdict(baseline, fixes, scheduled, min_pairs=5,
                                          min_tasks=5)])
    artifact = path.read_text(encoding="utf-8")

    row = json.loads(artifact.strip())
    assert not credit.contains_free_text(row, marker)
    assert marker not in artifact
    assert row["verdict"] in (credit.CREDITS, credit.DENIES, credit.WITHHELD)
    assert row["primary"]["metric"] == "ok"
    assert "tail" not in json.dumps(row)
    assert {"metric", "alpha", "comparisons", "seed", "min_pairs", "min_tasks"} <= set(
        row["declared"])


def test_tree_identity_reads_head_without_spawning_a_process(tmp_path, monkeypatch):
    sha = "a" * 40
    other = "b" * 40

    checkout = tmp_path / "plain"
    (checkout / ".git" / "refs" / "heads").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (checkout / ".git" / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")

    # a linked worktree: .git is a file, HEAD is in the worktree dir, the ref in commondir
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text("gitdir: ../plain/.git/worktrees/linked\n",
                                 encoding="utf-8")
    wt = checkout / ".git" / "worktrees" / "linked"
    (wt / "refs" / "heads").mkdir(parents=True)
    (wt / "HEAD").write_text("ref: refs/heads/branch\n", encoding="utf-8")
    (wt / "commondir").write_text("../..\n", encoding="utf-8")
    (checkout / ".git" / "refs" / "heads" / "branch").write_text(other + "\n",
                                                                encoding="utf-8")

    packed = tmp_path / "packed"
    (packed / ".git").mkdir(parents=True)
    (packed / ".git" / "HEAD").write_text("ref: refs/heads/old\n", encoding="utf-8")
    (packed / ".git" / "packed-refs").write_text(
        f"# pack-refs with: peeled\n{sha} refs/heads/old\n", encoding="utf-8")

    detached = tmp_path / "detached"
    (detached / ".git").mkdir(parents=True)
    (detached / ".git" / "HEAD").write_text(other + "\n", encoding="utf-8")

    def _no_process(*args, **kwargs):
        raise AssertionError("tree_identity must not spawn a process")

    monkeypatch.setattr(subprocess, "run", _no_process)
    monkeypatch.setattr(subprocess, "Popen", _no_process)

    assert credit.tree_identity(checkout) == sha
    assert credit.tree_identity(linked) == other
    assert credit.tree_identity(packed) == sha
    assert credit.tree_identity(detached) == other
    assert credit.tree_identity(tmp_path / "not-a-repo") is None


def test_battery_identity_is_fail_closed(tmp_path):
    rows = {"t1-r0": _row("t1-r0", "t1", 0, "baseline")}
    assert credit.battery_status(rows, FINGERPRINT) == "unchanged"
    assert credit.battery_status({"t1-r0": _row("t1-r0", "t1", 0, "baseline",
                                                battery="old-fp")}, FINGERPRINT) == "changed"
    # rows written before the fingerprint existed cannot support a verdict at all
    legacy = _arm(tmp_path, "legacy", ["t1-r0"], battery=None)
    assert credit.battery_status(credit.load_meta_rows(tmp_path / "legacy" / "meta.jsonl"),
                                 FINGERPRINT) == "unverified"

    changed = _verdict(*_arms([_pair(t, 0) for t in TASKS]), RUN_IDS, battery="changed")
    assert (changed["verdict"], changed["reason"]) == (credit.WITHHELD,
                                                       credit.R_BATTERY_CHANGED)

    unverified = _verdict(*_arms([_pair(t, 0) for t in TASKS]), RUN_IDS,
                          battery="unverified")
    assert (unverified["verdict"], unverified["reason"]) == (credit.WITHHELD,
                                                             credit.R_BATTERY_UNVERIFIED)
    assert legacy["t1-r0"].status == credit.RECORDED
