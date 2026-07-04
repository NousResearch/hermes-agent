#!/usr/bin/env python3
"""Offline retro probe for answerability vs objective failures in the nbq arm."""

import argparse
import json
import math
import sys


DEFAULT_INPUT = "/Users/dadleet/.hermes/outcome_eval_32_iter2probe.json"
PRIMARY_FRAMINGS = (
    ("top1_unans", "top1_unans"),
    ("any_unans", "any_unans"),
)


def is_unanswerable(qa_entry):
    answer = str(qa_entry.get("answer", ""))
    return qa_entry.get("revealed") is False or "doesn't say" in answer.lower()


def derive_task_booleans(row):
    unanswerable = [is_unanswerable(entry) for entry in row["qa"]]
    n_unans = sum(1 for flag in unanswerable if flag)
    return {
        "task": row.get("task"),
        "top1_unans": bool(unanswerable[0]) if unanswerable else False,
        "any_unans": any(unanswerable),
        "n_unans": n_unans,
        "fail": float(row["frac"]) < 1.0,
        "frac": float(row["frac"]),
    }


def contingency_2x2(predictor_bools, fail_bools):
    if len(predictor_bools) != len(fail_bools):
        raise ValueError("predictor and outcome lengths differ")
    cells = {
        "pred_true_fail_true": 0,
        "pred_true_fail_false": 0,
        "pred_false_fail_true": 0,
        "pred_false_fail_false": 0,
    }
    for pred, fail in zip(predictor_bools, fail_bools):
        if pred and fail:
            cells["pred_true_fail_true"] += 1
        elif pred and not fail:
            cells["pred_true_fail_false"] += 1
        elif not pred and fail:
            cells["pred_false_fail_true"] += 1
        else:
            cells["pred_false_fail_false"] += 1
    return cells


def point_biserial(x_bools_or_floats, y_floats):
    if len(x_bools_or_floats) != len(y_floats):
        raise ValueError("x and y lengths differ")
    n = len(x_bools_or_floats)
    if n < 2:
        return 0.0
    xs = [1.0 if isinstance(x, bool) and x else 0.0 if isinstance(x, bool) else float(x)
          for x in x_bools_or_floats]
    ys = [float(y) for y in y_floats]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    ss_x = sum(v * v for v in dx)
    ss_y = sum(v * v for v in dy)
    if ss_x == 0.0 or ss_y == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(ss_x * ss_y)


def se_for_r(r, n):
    if n <= 2:
        return float("inf")
    return math.sqrt((1.0 - r ** 2) / (n - 2))


def base_rate(bools):
    if not bools:
        return 0.0
    return sum(1 for value in bools if value) / len(bools)


def is_degenerate(bools, threshold=0.15):
    rate = base_rate(bools)
    return rate < threshold or rate > (1.0 - threshold)


def _row_label(row, index):
    task = row.get("task")
    if task is not None:
        return f"row {index} task {task!r}"
    for key in ("task_id", "id"):
        if row.get(key) is not None:
            return f"row {index} {key} {row[key]!r}"
    return f"row {index}"


def validate_nbq_row(row, index):
    q_len = len(row.get("questions", []))
    qa_len = len(row.get("qa", []))
    qv_len = len(row.get("meta", {}).get("q_values", []))
    if not (q_len == qa_len == qv_len):
        label = _row_label(row, index)
        raise ValueError(
            f"{label}: len(questions)={q_len}, len(qa)={qa_len}, "
            f"len(meta.q_values)={qv_len}"
        )


def framing_failure_reasons(r, se, degenerate):
    reasons = []
    if degenerate:
        reasons.append("degenerate")
    if r < 0.0:
        reasons.append("wrong-direction")
    elif r == 0.0 or abs(r) <= se:
        reasons.append("no-association")
    return reasons


def framing_proceeds(r, se, degenerate):
    return r > 0.0 and abs(r) > se and not degenerate


def assemble_stats(data):
    nbq_rows = [row for row in data.get("rows", []) if row.get("arm") == "nbq"]
    per_task = []
    for index, row in enumerate(nbq_rows):
        validate_nbq_row(row, index)
        task_stats = derive_task_booleans(row)
        task_stats["index"] = index
        per_task.append(task_stats)

    fail_bools = [task["fail"] for task in per_task]
    fail_floats = [1.0 if fail else 0.0 for fail in fail_bools]

    framings = {}
    for key, label in PRIMARY_FRAMINGS:
        predictor = [task[key] for task in per_task]
        r = point_biserial(predictor, fail_floats)
        se = se_for_r(r, len(per_task))
        degenerate = is_degenerate(predictor)
        proceeds = framing_proceeds(r, se, degenerate)
        reasons = [] if proceeds else framing_failure_reasons(r, se, degenerate)
        framings[key] = {
            "label": label,
            "contingency": contingency_2x2(predictor, fail_bools),
            "r": r,
            "se": se,
            "clears_se": abs(r) > se,
            "base_rate": base_rate(predictor),
            "degenerate": degenerate,
            "proceeds": proceeds,
            "failure_reasons": reasons,
        }

    n_unans_frac_r = point_biserial(
        [task["n_unans"] for task in per_task],
        [task["frac"] for task in per_task],
    )
    proceed = any(stats["proceeds"] for stats in framings.values())
    if proceed:
        winners = [key for key, stats in framings.items() if stats["proceeds"]]
        verdict = "PROCEED"
        reason = "at least one primary framing has positive r, |r| > SE, and non-degenerate predictor: " + ", ".join(winners)
    else:
        verdict = "PARK"
        parts = []
        for key, stats in framings.items():
            parts.append(f"{key}={'+'.join(stats['failure_reasons'])}")
        reason = "; ".join(parts)

    return {
        "n": len(per_task),
        "framings": framings,
        "n_unans_frac_r": n_unans_frac_r,
        "verdict": verdict,
        "reason": reason,
        "per_task": per_task,
    }


def _fmt_float(value):
    return f"{value:.6f}"


def format_stats(stats):
    lines = [f"n={stats['n']} nbq tasks"]
    for key, framing in stats["framings"].items():
        cells = framing["contingency"]
        lines.append("")
        lines.append(f"FRAMING: {key} x fail")
        lines.append("2x2 contingency:")
        lines.append(f"  pred_true_fail_true={cells['pred_true_fail_true']}")
        lines.append(f"  pred_true_fail_false={cells['pred_true_fail_false']}")
        lines.append(f"  pred_false_fail_true={cells['pred_false_fail_true']}")
        lines.append(f"  pred_false_fail_false={cells['pred_false_fail_false']}")
        lines.append(
            "effect: "
            f"r={_fmt_float(framing['r'])} "
            f"SE={_fmt_float(framing['se'])} "
            f"clears_se={str(framing['clears_se'])}"
        )
        lines.append(
            "predictor: "
            f"base_rate={_fmt_float(framing['base_rate'])} "
            f"DEGENERATE={str(framing['degenerate'])}"
        )
    lines.append("")
    lines.append(f"TERTIARY: n_unans x frac r={_fmt_float(stats['n_unans_frac_r'])}")
    lines.append(f"VERDICT: {stats['verdict']} {stats['reason']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stats = assemble_stats(data)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, sort_keys=True)
                fh.write("\n")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_stats(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
