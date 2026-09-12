#!/usr/bin/env python3
"""Append-only decision log for the hillclimb optimization loop.

The log is `.hillclimb/decision.tsv` (override with --dir or HILLCLIMB_DIR).
One row per attempt. The 11 columns, in order:

    id  timestamp  hypothesis  change  before  after  delta  tests  verdict  harness_id  note

`delta` is always raw `after - before` and is computed here, never accepted
from the caller. `direction` (minimize|maximize) is NOT a column; it lives in
baseline.json and is applied by `stats`.

Subcommands:
  append --hypothesis S --change S --before F --after F --tests {pass,fail,none}
         --verdict {kept,reverted} [--note S] [--harness-id S]
      Add one attempt. Ids are assigned here and are monotonic. --before and
      --after accept a float or the literal `na` (delta is then `na`).
      Prints the row as JSON. Exit 0.
  list [--limit N] [--verdict kept|reverted] [--json]
      Rows oldest-first. Exit 0.
  stats [--window N] [--threshold F] [--json]
      Direction-aware best/worst, mean improvement over the last window, and a
      plateau verdict: True when EVERY one of the last `window` attempts
      (default 3) improved by less than `threshold` (default 0.02) relative to
      its own `before`. Plateau needs at least `window` attempts. Exit 0.
      `best_improvement` is positive when the metric got better (in either
      direction); `worst_regression` is negative when it got worse.
  verify
      Validate the whole trail against the schema and the frozen baseline:
      column counts, ids, delta arithmetic, enums, unparseable metric fields,
      and rows whose harness no longer matches the frozen baseline. Prints
      JSON. Exit 0 when clean or when no log exists yet (`no-log`); exit 1 when
      any problem is found.

Exit codes:
  0 - success (including `verify` on a clean log and on a missing log)
  1 - `verify` found problems
  2 - bad invocation or unusable data (invalid enum, non-numeric value,
      --dir exists but is not a directory)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

COLUMNS = [
    "id",
    "timestamp",
    "hypothesis",
    "change",
    "before",
    "after",
    "delta",
    "tests",
    "verdict",
    "harness_id",
    "note",
]
HEADER = "\t".join(COLUMNS)
TESTS_VALUES = ("pass", "fail", "none")
VERDICT_VALUES = ("kept", "reverted")
DELTA_TOLERANCE = 1e-9


def die(message: str, code: int = 2) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def resolve_dir(value: str | None) -> Path:
    return Path(value or os.environ.get("HILLCLIMB_DIR") or ".hillclimb")


def ensure_dir(path: Path) -> None:
    """Create the log directory. Only an unusable path is an error."""
    if path.exists() and not path.is_dir():
        die(f"{path} exists but is not a directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        die(f"cannot create {path}: {exc}")
    if not path.is_dir():
        die(f"{path} could not be created as a directory")


def check_dir(path: Path) -> None:
    """Read paths: absent state is empty state, but a non-directory is an error."""
    if path.exists() and not path.is_dir():
        die(f"{path} exists but is not a directory")


def sanitize(value: str | None) -> str:
    return (value or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_rows(path: Path) -> list[list[str]]:
    """Return the data rows (header excluded). Rows are not validated here."""
    tsv = path / "decision.tsv"
    if not tsv.is_file():
        return []
    rows: list[list[str]] = []
    with open(tsv, "r", encoding="utf-8", errors="replace") as handle:
        handle.readline()  # header, validated by `verify`
        for line in handle:
            line = line.rstrip("\r\n")
            if line:
                rows.append(line.split("\t"))
    return rows


def load_baseline(path: Path) -> dict:
    baseline = path / "baseline.json"
    if not baseline.is_file():
        return {}
    try:
        return json.loads(baseline.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_metric(value: str | None) -> float | None:
    """`na` and empty both mean 'no measurement'. Garbage is a hard error."""
    if value is None or value == "na" or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        die(f"not a number and not 'na': {value!r}")


def parse_field(value: str) -> float | None:
    """Non-fatal variant used by `verify`: unparseable text yields None."""
    if value in ("na", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_metric(value: float | None) -> str:
    """Human-readable, and stable enough for `verify`'s numeric comparison."""
    if value is None:
        return "na"
    return f"{float(value):.10g}"


def compute_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def next_id(rows: list[list[str]]) -> int:
    highest = 0
    for row in rows:
        try:
            highest = max(highest, int(row[0]))
        except (IndexError, ValueError):
            continue
    return highest + 1


def cmd_append(args: argparse.Namespace) -> int:
    if args.tests not in TESTS_VALUES:
        die(f"--tests must be one of {', '.join(TESTS_VALUES)}")
    if args.verdict not in VERDICT_VALUES:
        die(f"--verdict must be one of {', '.join(VERDICT_VALUES)}")

    before = parse_metric(args.before)
    after = parse_metric(args.after)
    delta = compute_delta(before, after)

    path = resolve_dir(args.dir)
    ensure_dir(path)
    rows = load_rows(path)
    harness_id = args.harness_id or load_baseline(path).get("harness_id") or ""

    row = [
        str(next_id(rows)),
        now_iso(),
        sanitize(args.hypothesis),
        sanitize(args.change),
        format_metric(before),
        format_metric(after),
        format_metric(delta),
        args.tests,
        args.verdict,
        sanitize(harness_id),
        sanitize(args.note),
    ]

    tsv = path / "decision.tsv"
    new_file = not tsv.is_file()
    with open(tsv, "a", encoding="utf-8", newline="") as handle:
        if new_file:
            handle.write(HEADER + "\n")
        handle.write("\t".join(row) + "\n")

    print(json.dumps(dict(zip(COLUMNS, row)), indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = resolve_dir(args.dir)
    check_dir(path)
    raw = load_rows(path)
    rows = [r for r in raw if len(r) == len(COLUMNS)]
    malformed = len(raw) - len(rows)
    if malformed:
        # Never hide corruption in a display command; `verify` names it precisely.
        print(
            f"warning: {malformed} malformed row(s) skipped; run `verify` for details",
            file=sys.stderr,
        )
    if args.verdict:
        rows = [r for r in rows if len(r) > 8 and r[8] == args.verdict]
    if args.limit is not None:
        rows = rows[-args.limit :]
    if args.json:
        print(json.dumps([dict(zip(COLUMNS, r)) for r in rows], indent=2))
        return 0
    if not rows:
        print("(no attempts recorded)")
        return 0
    widths = [max(len(COLUMNS[i]), *(len(r[i]) for r in rows)) for i in range(len(COLUMNS))]
    print("  ".join(COLUMNS[i].ljust(widths[i]) for i in range(len(COLUMNS))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(COLUMNS))))
    return 0


def relative_improvement(row: list[str], direction: str) -> float | None:
    """Direction-aware improvement divided by |before|. None when unmeasurable."""
    try:
        before = float(row[4])
        after = float(row[5])
    except (IndexError, ValueError):
        return None
    if before == 0:
        return None
    gain = before - after if direction == "minimize" else after - before
    return gain / abs(before)


def cmd_stats(args: argparse.Namespace) -> int:
    path = resolve_dir(args.dir)
    check_dir(path)
    rows = [r for r in load_rows(path) if len(r) == len(COLUMNS)]
    baseline = load_baseline(path)
    direction = baseline.get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        direction = "minimize"

    window = args.window if args.window is not None else 3
    if window < 1:
        die("--window must be at least 1")
    threshold = args.threshold if args.threshold is not None else 0.02

    kept = sum(1 for r in rows if r[8] == "kept")
    reverted = sum(1 for r in rows if r[8] == "reverted")
    deltas = []
    for row in rows:
        try:
            deltas.append(float(row[6]))
        except (IndexError, ValueError):
            continue

    improvements = [-d if direction == "minimize" else d for d in deltas]
    recent = rows[-window:]
    window_improvements: list[float] = []
    for row in recent:
        try:
            value = float(row[6])
        except (IndexError, ValueError):
            continue
        window_improvements.append(-value if direction == "minimize" else value)

    # An attempt that cannot be measured (na metric, or a zero baseline) is not
    # evidence that the metric moved, so it counts as below threshold.
    reasons: list[str] = []
    for row in recent:
        rel = relative_improvement(row, direction)
        if rel is not None and rel >= threshold:
            continue
        shown = "unmeasurable" if rel is None else f"{rel:.4f}"
        reasons.append(f"attempt {row[0]}: relative improvement {shown} < {threshold:.2f}")
    plateau = len(recent) >= window and len(reasons) == len(recent)
    if not plateau:
        reasons = []

    result = {
        "attempts": len(rows),
        "kept": kept,
        "reverted": reverted,
        "direction": direction,
        "best_improvement": max([i for i in improvements if i > 0], default=None),
        "worst_regression": min([i for i in improvements if i < 0], default=None),
        "mean_improvement_last_window": (
            sum(window_improvements) / len(window_improvements) if window_improvements else None
        ),
        "window": window,
        "threshold": threshold,
        "plateau": plateau,
        "plateau_reasons": reasons,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Attempts: {result['attempts']} (kept {kept}, reverted {reverted})")
    print(f"Direction: {direction}")
    print(f"Best improvement: {result['best_improvement']}")
    print(f"Worst regression: {result['worst_regression']}")
    print(f"Mean improvement (last {window}): {result['mean_improvement_last_window']}")
    print(f"Plateau: {plateau}")
    for reason in reasons:
        print(f"  - {reason}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    path = resolve_dir(args.dir)
    check_dir(path)
    tsv = path / "decision.tsv"
    if not tsv.is_file():
        print(json.dumps({"problems": [{"type": "no-log"}]}, indent=2))
        return 0

    problems: list[dict] = []
    with open(tsv, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\r\n")
        if header != HEADER:
            problems.append({"type": "wrong-header"})
        raw = [line.rstrip("\r\n") for line in handle if line.strip()]
    baseline_id = load_baseline(path).get("harness_id")

    ids: list[int] = []
    for index, line in enumerate(raw, start=1):
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            problems.append({"type": "row-wrong-column-count", "row": index})
            continue
        try:
            ids.append(int(fields[0]))
        except ValueError:
            problems.append({"type": "non-numeric-id", "row": index})

        for name, raw_value in (("before", fields[4]), ("after", fields[5])):
            if raw_value not in ("na", "") and parse_field(raw_value) is None:
                problems.append(
                    {"type": "invalid-metric", "row": index, "field": name, "value": raw_value}
                )
        before = parse_field(fields[4])
        after = parse_field(fields[5])
        expected = compute_delta(before, after)
        stored = fields[6]
        if expected is None:
            if stored != "na":
                problems.append({"type": "delta-mismatch", "row": index, "expected": "na", "got": stored})
        else:
            try:
                if abs(float(stored) - expected) > DELTA_TOLERANCE:
                    problems.append(
                        {"type": "delta-mismatch", "row": index, "expected": repr(expected), "got": stored}
                    )
            except ValueError:
                problems.append({"type": "delta-mismatch", "row": index, "expected": repr(expected), "got": stored})

        if fields[7] not in TESTS_VALUES:
            problems.append({"type": "invalid-tests", "row": index, "value": fields[7]})
        if fields[8] not in VERDICT_VALUES:
            problems.append({"type": "invalid-verdict", "row": index, "value": fields[8]})

        if baseline_id and fields[9] != baseline_id:
            problems.append(
                {"type": "stale-harness", "row": index, "log_id": fields[9], "baseline_id": baseline_id}
            )

    if ids != sorted(ids):
        problems.append({"type": "non-monotonic-ids"})
    if len(ids) != len(set(ids)):
        problems.append({"type": "duplicate-ids"})

    print(json.dumps({"problems": problems}, indent=2))
    return 1 if problems else 0


def _add_dir(sub: argparse.ArgumentParser) -> None:
    # SUPPRESS keeps the root-level value when --dir is given before the
    # subcommand, while still accepting it after the subcommand.
    sub.add_argument("--dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hillclimb decision log manager")
    parser.add_argument(
        "--dir",
        help="Decision log directory (default: HILLCLIMB_DIR env var, else .hillclimb)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    append = subparsers.add_parser("append", help="Append one attempt")
    append.add_argument("--hypothesis", required=True)
    append.add_argument("--change", required=True)
    append.add_argument("--before", required=True, help="float or 'na'")
    append.add_argument("--after", required=True, help="float or 'na'")
    append.add_argument("--tests", required=True, choices=list(TESTS_VALUES))
    append.add_argument("--verdict", required=True, choices=list(VERDICT_VALUES))
    append.add_argument("--note", default="")
    append.add_argument("--harness-id", dest="harness_id", default=None)
    _add_dir(append)
    append.set_defaults(func=cmd_append)

    list_cmd = subparsers.add_parser("list", help="List attempts, oldest first")
    list_cmd.add_argument("--limit", type=int, default=None)
    list_cmd.add_argument("--verdict", choices=list(VERDICT_VALUES), default=None)
    list_cmd.add_argument("--json", action="store_true")
    _add_dir(list_cmd)
    list_cmd.set_defaults(func=cmd_list)

    stats = subparsers.add_parser("stats", help="Attempt statistics and plateau verdict")
    stats.add_argument("--window", type=int, default=None)
    stats.add_argument("--threshold", type=float, default=None)
    stats.add_argument("--json", action="store_true")
    _add_dir(stats)
    stats.set_defaults(func=cmd_stats)

    verify = subparsers.add_parser("verify", help="Validate the log")
    _add_dir(verify)
    verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
