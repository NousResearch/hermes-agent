#!/usr/bin/env python3
"""Paired, completeness-accounted, fail-closed credit analysis for the toolperf A/B battery.

Pure analysis over an already-run local results tree. No network, no model client, no
subprocess, no import-time environment read, no module-level mutable state, stdlib only:
paths and rows go in, data comes out, so a test can import this module hermetically.

The contract (issue NousResearch/hermes-agent#111237, design ``receipts/issue-111237-design.md``):

- **Every scheduled cell is accounted for.** A cell is exactly one of ``recorded``,
  ``infra_crash``, ``missing_trace``, ``oracle_error``, ``unaccounted``. The runner drops
  startup crashes from ``meta.jsonl``, so "did the battery complete?" has to be answered
  from the schedule, not from whichever rows happen to exist afterwards, and a scheduled
  cell with no evidence at all is ``unaccounted`` — never assumed to be a harmless flake.
  Infrastructure failures stay in the denominator.
- **Arms are paired by run_id**, never by list order or index. Unpaired runs are counted
  and reported; they are never averaged into a cell mean.
- **The resampling unit is the task, not the run** (cluster bootstrap). The reps of one
  task share a sandbox, a prompt and a failure mode, so they are one cluster: raising
  ``--reps`` buys precision inside a task, not significance across tasks.
- **A verdict is ``credits`` / ``denies`` / ``withheld``**, and a ``withheld`` verdict
  names the precondition that failed. ``NO UPDATE`` is a first-class outcome; a
  measurement problem is never resolved in favour of the candidate. In particular the
  battery must be *complete as evidence*: every scheduled cell has to carry both a metric
  trace and a success signal, so "did the recorded cases behave correctly?" can never be
  mistaken for "did the experiment complete all scheduled cases?".
- **The verdict artifact carries no free text from traces** (no ``tail``, no sandbox
  paths). Rows are assembled from named pieces only, so a verdict can be shared while the
  raw traces stay local.

Deltas are relative improvements so that one margin covers metrics in different units
(turns vs KB): positive always means "better", and the promotion test is the lower bound
of the task-clustered bootstrap interval being above zero.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA = "toolperf_abeval.credit.v1"

#: Bump on any change to the semantics of a battery task or an oracle. It is part of the
#: fingerprint, so a battery edit cannot pass silently.
BATTERY_VERSION = "1"

ARMS = ("baseline", "fixes")

# ── cell statuses ───────────────────────────────────────────────────────────
RECORDED = "recorded"
INFRA_CRASH = "infra_crash"
MISSING_TRACE = "missing_trace"
ORACLE_ERROR = "oracle_error"
UNACCOUNTED = "unaccounted"
STATUSES = (RECORDED, INFRA_CRASH, MISSING_TRACE, ORACLE_ERROR, UNACCOUNTED)

# ── oracle outcomes, supplied per run_id by the caller ──────────────────────
ORACLE_OK = "ok"
ORACLE_FAIL = "fail"
ORACLE_RAISED = "error"
ORACLE_ABSENT = "absent"

# ── verdicts and machine reasons ────────────────────────────────────────────
CREDITS = "credits"
DENIES = "denies"
WITHHELD = "withheld"

R_INCOMPLETE = "incomplete"
R_INSUFFICIENT = "insufficient"
R_BATTERY_CHANGED = "battery_changed"
R_BATTERY_UNVERIFIED = "battery_unverified"
R_NO_EFFECT = "no_effect"
R_GUARDRAIL = "guardrail_breach"
R_CREDITED = "credited"

HIGHER_IS_BETTER = frozenset({"ok"})
LOWER_IS_BETTER = frozenset({"llm", "tools", "errs", "retries", "kb", "wall"})
DEFAULT_GUARDRAILS = ("llm", "tools", "errs")
DEFAULT_MARGIN = 0.10

_SHA_RE = re.compile(r"[0-9a-f]{40}")


# ── battery identity ────────────────────────────────────────────────────────
def battery_fingerprint(
    tasks: Sequence[str], oracle_names: Sequence[str], version: str = BATTERY_VERSION
) -> str:
    """Digest of the ordered task ids, the oracle identities and the battery version.

    Oracle *identity* is the oracle name (function objects have no stable digest); the
    oracle bodies still have to be reviewed by hand when they change.
    """
    payload = json.dumps(
        {"version": version, "tasks": list(tasks), "oracles": sorted(oracle_names)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """SHA-256 of one input file, or ``"absent"`` so a missing input is visible."""
    if not path.exists():
        return "absent"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_identity(path: str | Path) -> str | None:
    """Best-effort git revision of a checkout, read from its ``.git`` — no subprocess.

    Handles a normal checkout (``.git`` directory), a linked worktree (``.git`` file with a
    ``gitdir:`` pointer, refs resolved through ``commondir``), packed refs and a detached
    HEAD. Returns ``None`` when the path is not a git tree, so a verdict records
    "unattributed" instead of guessing which tree produced the numbers.
    """
    try:
        dot = Path(path) / ".git"
        if dot.is_file():
            pointer = dot.read_text(encoding="utf-8", errors="replace").strip()
            if not pointer.startswith("gitdir:"):
                return None
            gitdir = Path(pointer.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (Path(path) / gitdir).resolve()
        elif dot.is_dir():
            gitdir = dot
        else:
            return None

        common = gitdir
        commondir = gitdir / "commondir"
        if commondir.is_file():
            raw = commondir.read_text(encoding="utf-8", errors="replace").strip()
            common = Path(raw) if Path(raw).is_absolute() else (gitdir / raw).resolve()

        head = (gitdir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head.startswith("ref:"):
            return head if _SHA_RE.fullmatch(head) else None
        ref = head.split(":", 1)[1].strip()
        for base in (common, gitdir):
            loose = base / ref
            if loose.is_file():
                sha = loose.read_text(encoding="utf-8", errors="replace").strip()
                return sha if _SHA_RE.fullmatch(sha) else None
        for base in (common, gitdir):
            sha = _packed_ref(base, ref)
            if sha:
                return sha
        return None
    except OSError:
        return None


def _packed_ref(gitdir: Path, ref: str) -> str | None:
    packed = gitdir / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref and _SHA_RE.fullmatch(sha):
            return sha
    return None


# ── trace scoring (shared with the runner's report path) ────────────────────
def score_trace(atof: Path) -> dict | None:
    """Metrics from one NeMo Relay ATOF trace, or ``None`` when it cannot be scored.

    ``None`` means "no evidence", not "perfect run": the trace is absent, unreadable, or
    carries no parseable event. Callers must never substitute zeros for it — a traceless
    run scored as zero waste makes an inefficient arm look like the better one.
    """
    llm = tools = errs = retries = 0
    result_bytes = 0
    last_err_tool = None
    seen = False
    try:
        text = atof.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        seen = True
        k, c, sc = ev.get("kind"), ev.get("category"), ev.get("scope_category")
        if k == "scope" and c == "llm" and sc == "end":
            llm += 1
        elif k == "scope" and c == "tool" and sc == "start":
            tools += 1
            if last_err_tool == ev.get("name"):
                retries += 1
        elif k == "scope" and c == "tool" and sc == "end":
            d = ev.get("data")
            ds = d if isinstance(d, str) else json.dumps(d or "")
            result_bytes += len(ds)
            is_err = ev.get("metadata", {}).get("status") not in (None, "ok")
            if not is_err:
                if re.search(r'"error":\s*"(?!null)', ds[:1500]) or \
                        re.search(r'"exit_code":\s*[1-9-]', ds[:200]):
                    is_err = True
            if is_err:
                errs += 1
                last_err_tool = ev.get("name")
            else:
                last_err_tool = None
    if not seen:
        return None
    return {"llm": llm, "tools": tools, "errs": errs,
            "retries": retries, "kb": result_bytes // 1024}


def load_meta_rows(path: Path) -> dict[str, dict]:
    """``run_id -> row`` for one arm's ``meta.jsonl``; unparsable lines are skipped."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("run_id"):
            rows[row["run_id"]] = row
    return rows


def battery_status(rows: Mapping[str, Mapping], current: str) -> str:
    """``unchanged`` | ``changed`` | ``unverified`` against what the runs recorded.

    A row written before the fingerprint existed cannot support a verdict: fail closed
    rather than credit against an exam nobody can name.
    """
    seen = {r.get("battery") for r in rows.values() if isinstance(r, Mapping)}
    seen = {s for s in seen if s}
    if not seen:
        return "unverified"
    return "unchanged" if seen == {current} else "changed"


# ── per-cell accounting ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Cell:
    """One scheduled (task, rep, arm) cell and how it was accounted for.

    Only whitelisted fields are carried, so nothing from a run's free text can reach the
    verdict artifact.
    """

    run_id: str
    task: str
    rep: int
    arm: str
    status: str
    oracle_ok: bool | None = None
    #: Populated exactly when ``status == RECORDED``: a cell is either fully usable
    #: (trace + success signal) or contributes no evidence of its own.
    metrics: Mapping[str, float] | None = None
    wall_s: float | None = None
    tree: str = "unknown"

    @property
    def usable(self) -> bool:
        """True when the cell carries both a metric trace and a success signal."""
        return self.status == RECORDED


def split_run_id(run_id: str) -> tuple[str, int]:
    """``"err_big_output-r2"`` -> ``("err_big_output", 2)``; unknown shapes get rep 0."""
    task, _, rep = run_id.rpartition("-r")
    if not task:
        return run_id, 0
    try:
        return task, int(rep)
    except ValueError:
        return run_id, 0


def classify_cell(
    *,
    arm: str,
    run_id: str,
    meta_row: Mapping | None,
    trace_path: Path,
    oracle: str = ORACLE_ABSENT,
    tree: str = "unknown",
) -> Cell:
    """Account for one scheduled cell.

    Priority, highest first — a cell gets exactly one status:

    1. a row that declares itself a crash (``status: infra_crash``) -> ``infra_crash``;
       accounted for, kept in the ledger, never dropped from the denominator
    2. no ``meta.jsonl`` row -> ``unaccounted``. The runner's documented behaviour is to
       *drop* a startup crash from ``meta.jsonl``, which makes it indistinguishable from a
       run that never happened. Assuming "harmless flake" here is exactly how
       infrastructure failures leave the denominator, so the cell gaps the battery
       instead, and the verdict is withheld until the run is repeated.
    3. no oracle for the task -> ``unaccounted``: success is unknowable, so the cell must
       not be counted as a failure either
    4. oracle raised -> ``oracle_error``: the sandbox was damaged. The cell is not a
       capability failure and is excluded from the success denominator.
    5. trace unscoreable -> ``missing_trace``: no metric evidence at all
    6. otherwise -> ``recorded``
    """
    task, rep = split_run_id(run_id)
    if meta_row is not None:
        task = str(meta_row.get("task") or task)
        try:
            rep = int(meta_row.get("rep", rep))
        except (TypeError, ValueError):
            pass

    if meta_row is None:
        return Cell(run_id=run_id, task=task, rep=rep, arm=arm, status=UNACCOUNTED, tree=tree)
    if meta_row.get("status") == INFRA_CRASH:
        return Cell(run_id=run_id, task=task, rep=rep, arm=arm, status=INFRA_CRASH, tree=tree)
    if oracle == ORACLE_ABSENT:
        return Cell(run_id=run_id, task=task, rep=rep, arm=arm, status=UNACCOUNTED, tree=tree)
    if oracle == ORACLE_RAISED:
        return Cell(run_id=run_id, task=task, rep=rep, arm=arm, status=ORACLE_ERROR, tree=tree)
    metrics = score_trace(trace_path)
    if metrics is None:
        return Cell(run_id=run_id, task=task, rep=rep, arm=arm, status=MISSING_TRACE, tree=tree)
    wall = meta_row.get("wall_s")
    try:
        wall = float(wall) if wall is not None else None
    except (TypeError, ValueError):
        wall = None
    return Cell(
        run_id=run_id,
        task=task,
        rep=rep,
        arm=arm,
        status=RECORDED,
        tree=tree,
        oracle_ok=(oracle == ORACLE_OK),
        metrics=metrics,
        wall_s=wall,
    )


def classify_arm(
    arm: str,
    scheduled_ids: Sequence[str],
    meta_rows: Mapping[str, Mapping],
    trace_dir: Path,
    oracle_outcomes: Mapping[str, str] | None = None,
    tree: str = "unknown",
) -> dict[str, Cell]:
    """One :class:`Cell` per scheduled run_id for one arm — nothing is left unaccounted."""
    oracles = oracle_outcomes or {}
    return {
        run_id: classify_cell(
            arm=arm,
            run_id=run_id,
            meta_row=meta_rows.get(run_id),
            trace_path=Path(trace_dir) / f"{run_id}.atof.jsonl",
            oracle=oracles.get(run_id, ORACLE_ABSENT),
            tree=tree,
        )
        for run_id in scheduled_ids
    }


def scheduled_run_ids(tasks: Sequence[str], reps: int) -> list[str]:
    """The battery's schedule: every (task, rep) cell the run was asked to fill."""
    return [f"{task}-r{rep}" for rep in range(max(0, reps)) for task in tasks]


def completeness(scheduled_ids: Sequence[str], cells: Mapping[str, Cell]) -> dict:
    """Per-arm ledger: scheduled vs accounted, by status."""
    counts = Counter(c.status for c in cells.values())
    by_status = {s: int(counts.get(s, 0)) for s in STATUSES}
    return {
        "scheduled": len(scheduled_ids),
        "by_status": by_status,
        "unaccounted": by_status[UNACCOUNTED],
        "off_schedule": sorted(set(cells) - set(scheduled_ids)),
    }


def pair_cells(
    baseline: Mapping[str, Cell], fixes: Mapping[str, Cell]
) -> tuple[list[tuple[Cell, Cell]], list[str], list[str]]:
    """Pair the arms by run_id and name the runs that have no healthy twin.

    Only cells carrying both a metric trace and a success signal in *both* arms can be
    paired: a cell that crashed, lost its trace, or lost its oracle on one arm is counted
    as unpaired rather than compared against a healthier twin.
    """
    b_ok = {rid: c for rid, c in baseline.items() if c.usable}
    f_ok = {rid: c for rid, c in fixes.items() if c.usable}
    pairs = [(b_ok[rid], f_ok[rid]) for rid in sorted(set(b_ok) & set(f_ok))]
    return pairs, sorted(set(b_ok) - set(f_ok)), sorted(set(f_ok) - set(b_ok))


# ── statistics ──────────────────────────────────────────────────────────────
def metric_value(cell: Cell, metric: str) -> float | None:
    """A cell's value for one metric, or ``None`` when the cell cannot supply it."""
    if metric == "ok":
        return None if cell.oracle_ok is None else (1.0 if cell.oracle_ok else 0.0)
    if metric == "wall":
        return None if cell.wall_s is None else float(cell.wall_s)
    if not cell.metrics:
        return None
    value = cell.metrics.get(metric)
    return None if value is None else float(value)


def improvement(baseline: Cell, fixes: Cell, metric: str) -> float | None:
    """Relative improvement of one paired cell on one metric (positive == better).

    Divided by ``max(|baseline|, 1.0)`` so the promotion threshold is scale-free and one
    margin covers every metric regardless of its unit.
    """
    b, f = metric_value(baseline, metric), metric_value(fixes, metric)
    if b is None or f is None:
        return None
    delta = (f - b) if metric in HIGHER_IS_BETTER else (b - f)
    return delta / max(abs(b), 1.0)


def per_task_delta(pairs: Sequence[tuple[Cell, Cell]], metric: str) -> dict[str, float]:
    """task id -> mean paired relative improvement on ``metric``.

    The task is the resampling unit, so this is the per-cluster summary: pairs that cannot
    supply the metric on both sides are dropped (and therefore never look like zero).
    """
    by_task: dict[str, list[float]] = {}
    for baseline, fixes in pairs:
        value = improvement(baseline, fixes, metric)
        if value is None:
            continue
        by_task.setdefault(baseline.task, []).append(value)
    return {task: sum(vals) / len(vals) for task, vals in by_task.items()}


def cluster_bootstrap_ci(
    per_task: Mapping[str, float], alpha: float, resamples: int, seed: int
) -> tuple[float, float, float]:
    """``(lo, hi, observed)`` for the mean per-task improvement, resampling TASKS.

    Deterministic: same values + same seed -> identical bounds. Duplicating the reps inside
    a task cannot narrow the interval, because each task contributes one mean.
    """
    tasks = sorted(per_task)
    values = [per_task[t] for t in tasks]
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    observed = sum(values) / n
    if resamples <= 0:
        return (observed, observed, observed)
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return (_quantile(means, alpha), _quantile(means, 1.0 - alpha), observed)


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    pos = q * (len(ordered) - 1)
    idx = int(pos)
    frac = pos - idx
    if idx + 1 >= len(ordered):
        return ordered[-1]
    return ordered[idx] + frac * (ordered[idx + 1] - ordered[idx])


def incomplete_arms(by_arm: Mapping[str, Mapping]) -> list[str]:
    """Arms whose battery did not complete with usable evidence.

    "Did the recorded cases behave correctly?" and "did the experiment complete all
    scheduled cases?" are separate checks. This is the second one: a crash, a vanished
    trace, a damaged sandbox and a cell with no row at all are all *reported* in the
    ledger (never dropped from the denominator) and all mean the battery is incomplete,
    so the verdict is withheld until the cell carries real evidence. A candidate whose own
    diff crashed cells the baseline ran cleanly must not be credited on the survivors.
    """
    return [
        arm
        for arm, ledger in by_arm.items()
        if ledger["by_status"][RECORDED] != ledger["scheduled"] or ledger["off_schedule"]
    ]


# ── the verdict ─────────────────────────────────────────────────────────────
def credit_verdict(
    pairs: Sequence[tuple[Cell, Cell]],
    by_arm: Mapping[str, Mapping],
    *,
    metric: str = "ok",
    guardrails: Sequence[str] = DEFAULT_GUARDRAILS,
    margin: float = DEFAULT_MARGIN,
    alpha: float = 0.05,
    comparisons: int = 1,
    resamples: int = 10000,
    seed: int = 0,
    min_pairs: int = 5,
    min_tasks: int = 5,
    battery: str = "unverified",
    battery_fingerprint: str | None = None,
    trees: Mapping[str, str] | None = None,
    input_digests: Mapping[str, str] | None = None,
    unpaired: Mapping[str, Sequence[str]] | None = None,
    model: str | None = None,
) -> dict:
    """Decide ``credits`` / ``denies`` / ``withheld`` for one candidate, fail-closed.

    The preconditions are checked first and any failure withholds the verdict: an
    incomplete battery, an unattributable or edited battery, and too little paired
    evidence are measurement problems, not weak candidates, and a measurement problem
    must never be resolved in the candidate's favour.
    """
    alpha_adjusted = alpha / max(1, comparisons)
    per_task = {m: per_task_delta(pairs, m) for m in (metric, *guardrails)}
    primary_delta = per_task[metric]
    primary_ci = cluster_bootstrap_ci(primary_delta, alpha_adjusted, resamples, seed)
    guardrail_detail = {}
    for name in guardrails:
        lo, hi, mean = cluster_bootstrap_ci(per_task[name], alpha_adjusted, resamples, seed)
        guardrail_detail[name] = {
            "mean": _round(mean),
            "lo": _round(lo),
            "hi": _round(hi),
            "tasks": len(per_task[name]),
            "floor": _round(-margin),
            "breached_tasks": sorted(
                t for t, v in per_task[name].items() if v < -margin
            ),
        }

    broken_arms = incomplete_arms(by_arm)
    reasons: list[str] = []
    if broken_arms:
        reasons.append(R_INCOMPLETE)
    if battery == "changed":
        reasons.append(R_BATTERY_CHANGED)
    elif battery != "unchanged":
        reasons.append(R_BATTERY_UNVERIFIED)
    if len(pairs) < min_pairs or len(primary_delta) < min_tasks:
        reasons.append(R_INSUFFICIENT)
    if not reasons:
        if primary_ci[0] <= 0:
            reasons.append(R_NO_EFFECT)
        elif any(
            guardrail_detail[name]["lo"] < -margin for name in guardrails
        ):
            reasons.append(R_GUARDRAIL)

    if reasons:
        verdict = WITHHELD if reasons[0] in (
            R_INCOMPLETE, R_BATTERY_CHANGED, R_BATTERY_UNVERIFIED, R_INSUFFICIENT
        ) else DENIES
        reason = reasons[0]
    else:
        verdict, reason = CREDITS, R_CREDITED

    counts = {s: 0 for s in STATUSES}
    for ledger in by_arm.values():
        for status, value in ledger["by_status"].items():
            counts[status] += value
    unpaired = unpaired or {}
    counts.update(
        scheduled=sum(led["scheduled"] for led in by_arm.values()),
        pairs=len(pairs),
        tasks_paired=len(primary_delta),
        unpaired_baseline=len(unpaired.get("baseline", ())),
        unpaired_fixes=len(unpaired.get("fixes", ())),
    )

    return {
        "schema": SCHEMA,
        "model": model,
        "verdict": verdict,
        "reason": reason,
        "reasons": reasons,
        "declared": {
            "metric": metric,
            "guardrails": list(guardrails),
            "margin": margin,
            "alpha": alpha,
            "alpha_adjusted": alpha_adjusted,
            "comparisons": comparisons,
            "resamples": resamples,
            "seed": seed,
            "min_pairs": min_pairs,
            "min_tasks": min_tasks,
        },
        "battery": {
            "version": BATTERY_VERSION,
            "fingerprint": battery_fingerprint,
            "status": battery,
        },
        "trees": dict(trees or {}),
        "input_digests": dict(input_digests or {}),
        "counts": counts,
        "evidence_complete": not broken_arms,
        "arms": {
            arm: dict(ledger["by_status"], scheduled=ledger["scheduled"],
                      off_schedule=list(ledger["off_schedule"]))
            for arm, ledger in by_arm.items()
        },
        "unpaired": {
            "baseline": list(unpaired.get("baseline", ())),
            "fixes": list(unpaired.get("fixes", ())),
        },
        "primary": {
            "metric": metric,
            "mean": _round(primary_ci[2]),
            "lo": _round(primary_ci[0]),
            "hi": _round(primary_ci[1]),
            "tasks": len(primary_delta),
            "task_deltas": {t: _round(v) for t, v in sorted(primary_delta.items())},
        },
        "guardrails": guardrail_detail,
    }


def _round(value: float) -> float:
    return round(float(value), 6)


def write_verdicts(path: Path, rows: Sequence[Mapping]) -> None:
    """Append verdict rows as JSONL — credited, denied and withheld alike.

    Rejected candidates are retained as negative data: a search loop's refusals are the
    only record that it looked, and a later reader can see which precondition it missed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def contains_free_text(row: Mapping, marker: str) -> bool:
    """True when ``marker`` survives into a verdict row — the artifact is JSON-only."""
    return marker in json.dumps(row, sort_keys=True)
