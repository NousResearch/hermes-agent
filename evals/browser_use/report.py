"""Aggregate results.jsonl into the scorecard tables.

Usage:
    python3 report.py [results/results.jsonl ...]

Groups by (model, arm): ok-rate, token mean/median, tool calls, wall clock,
and token delta vs the ``base`` arm of the same model when present.

The battery is fully crossed — ``orchestrate.py`` runs every
(arm, task, model, rep) cell — so ``vs base`` is a *paired* comparison and is
computed only over the (task, rep) cells where this arm and ``base`` both
produced an ok run. ``pair`` reports how many cells that was. Averaging each
arm over its own ok runs and subtracting compares two different task sets
whenever the arms fail on different cells, which makes an arm that succeeds on
an extra expensive task look more costly for having succeeded.
"""

import json
import statistics
import sys
from collections import defaultdict

BASE_ARM = "base"


def _cell_key(r):
    return (r.get("task", "?"), r.get("rep"))


def _cell_means(ok_by_cell):
    """Collapse each cell's list of ok runs to its mean.

    A cell is one (task, rep) of one arm and normally holds exactly one ok run.
    It can hold more when several results.jsonl files are passed at once, or
    when a resumed battery re-ran a cell. Overwriting on collision would make a
    duplicated record silently decide the delta; averaging keeps the same
    behaviour the per-arm columns already have, where every ok row counts.
    """
    return {
        key: {cell: sum(vals) / len(vals) for cell, vals in cells.items()}
        for key, cells in ok_by_cell.items()
    }


def paired_token_delta(ok_by_cell, model, arm, base_arm=BASE_ARM):
    """Return (pct, n_paired) for ``arm`` against ``base_arm`` on one model.

    Only cells both arms resolved contribute. ``pct`` is None when the arms
    share no ok cell, or when the paired base total is zero.
    """
    arm_cells = ok_by_cell.get((model, arm), {})
    base_cells = ok_by_cell.get((model, base_arm), {})
    # Sorted for a stable order: means do not care, but the pair count and any
    # future per-cell output should not shuffle between runs.
    shared = sorted(set(arm_cells) & set(base_cells), key=lambda k: (str(k[0]), str(k[1])))
    if not shared:
        return None, 0
    arm_tok = statistics.mean(arm_cells[k] for k in shared)
    base_tok = statistics.mean(base_cells[k] for k in shared)
    if not base_tok:
        return None, len(shared)
    return (arm_tok - base_tok) / base_tok * 100, len(shared)


def main(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print("no rows")
        return

    cells = defaultdict(list)
    for r in rows:
        cells[(r.get("model", "?"), r.get("arm", "?"))].append(r)

    # (model, arm) -> {(task, rep): [total_tokens, ...]} over ok runs only.
    # Keyed by cell so the vs-base delta can be taken on the intersection; a
    # list rather than a scalar so a duplicated record cannot silently replace
    # the value the delta is computed from.
    raw_by_cell = defaultdict(lambda: defaultdict(list))
    for (model, arm), rs in cells.items():
        for r in rs:
            if r.get("ok"):
                raw_by_cell[(model, arm)][_cell_key(r)].append(r.get("total_tokens", 0))
    dupes = sum(
        len(v) - 1 for cells_ in raw_by_cell.values() for v in cells_.values() if len(v) > 1
    )
    ok_by_cell = _cell_means(raw_by_cell)

    hdr = (
        f"{'model':<34} {'arm':<16} {'ok':>7} {'tok_mean':>9} {'tok_med':>8} "
        f"{'calls':>6} {'wall_s':>7} {'vs base':>8} {'pair':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model, arm in sorted(cells):
        rs = cells[(model, arm)]
        oks = [r for r in rs if r.get("ok")]
        n_ok, n = len(oks), len(rs)
        toks = [r.get("total_tokens", 0) for r in oks]
        calls = [r.get("tool_calls", 0) for r in oks]
        walls = [r.get("wall_s", 0) for r in oks]
        tok_mean = statistics.mean(toks) if toks else 0
        delta, pair = "", ""
        if arm != BASE_ARM and (model, BASE_ARM) in ok_by_cell:
            pct, n_paired = paired_token_delta(ok_by_cell, model, arm)
            if pct is None:
                delta, pair = "n/a", f"0/{n_ok}"
            else:
                delta, pair = f"{pct:+.0f}%", f"{n_paired}/{n_ok}"
        print(
            f"{model:<34} {arm:<16} {n_ok:>3}/{n:<3} {tok_mean:>9.0f} "
            f"{statistics.median(toks) if toks else 0:>8.0f} "
            f"{statistics.mean(calls) if calls else 0:>6.1f} "
            f"{statistics.mean(walls) if walls else 0:>7.1f} {delta:>8} {pair:>6}"
        )

    if dupes:
        print(
            f"\nNOTE: {dupes} duplicate (task, rep) record(s) across the inputs; "
            "each cell's paired value is the mean of its ok runs."
        )
    print(
        "\nNote: tok_mean/tok_med/calls/wall_s are per-arm means over that arm's\n"
        "own ok runs. 'vs base' is paired — it uses only the (task, rep) cells\n"
        "this arm and base both resolved, and 'pair' shows how many of the arm's\n"
        "ok runs that was. A pair count below the ok count means the unpaired\n"
        "columns above are averaging different task sets across arms."
    )

    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\nerrors: {len(errs)}")
        for r in errs[:10]:
            print(
                f"  {r.get('model')}/{r.get('arm')}/{r.get('task')}/rep{r.get('rep')}: {r['error'][:120]}"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/results.jsonl"])
