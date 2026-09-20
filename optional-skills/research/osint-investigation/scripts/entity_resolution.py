#!/usr/bin/env python3
"""Cross-source entity resolution (stdlib-only).

Given two CSV files with name columns, find candidate matches using three
tiers of normalization:

  1. exact          — normalized strings equal
  2. fuzzy          — sorted-token (word-bag) match
  3. token_overlap  — >=60% Jaccard overlap on >=4-char tokens, >=2 shared

Adapted from ShinMegamiBoson/OpenPlanter (MIT) but generalized: no Boston-
specific record types, no contribution-code filters, no fixed schemas.

Output CSV columns:
    match_type, confidence, left_name, right_name,
    left_normalized, right_normalized, left_row, right_row,
    overlap_ratio, shared_tokens
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running directly or as a module.
sys.path.insert(0, str(Path(__file__).parent))
from _normalize import (  # noqa: E402
    normalize_name,
    normalize_aggressive,
    token_overlap_ratio,
)

CONFIDENCE = {
    "exact": "high",
    "fuzzy": "medium",
    "token_overlap": "low",
}


def _read_csv(path: str, name_col: str) -> list[dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if name_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column {name_col!r} not in {path}. "
                f"Available: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            row["__row__"] = str(i)
            rows.append(row)
    return rows


def _build_index(rows: list[dict[str, str]], name_col: str):
    """Index by exact-normalized and aggressive (sorted-token) form."""
    exact: dict[str, list[dict[str, str]]] = {}
    aggressive: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        raw = row.get(name_col, "")
        n = normalize_name(raw)
        if n:
            exact.setdefault(n, []).append(row)
        a = normalize_aggressive(raw)
        if a:
            aggressive.setdefault(a, []).append(row)
    return exact, aggressive


def _emit(
    out_rows: list[dict[str, str]],
    seen: set[tuple],
    match_type: str,
    left_row: dict[str, str],
    right_row: dict[str, str],
    left_col: str,
    right_col: str,
    ratio: float = 0.0,
    shared: int = 0,
):
    left_raw = left_row.get(left_col, "")
    right_raw = right_row.get(right_col, "")
    key = (
        left_row["__row__"],
        right_row["__row__"],
        match_type,
    )
    if key in seen:
        return
    seen.add(key)
    out_rows.append(
        {
            "match_type": match_type,
            "confidence": CONFIDENCE[match_type],
            "left_name": left_raw,
            "right_name": right_raw,
            "left_normalized": normalize_name(left_raw),
            "right_normalized": normalize_name(right_raw),
            "left_row": left_row["__row__"],
            "right_row": right_row["__row__"],
            "overlap_ratio": f"{ratio:.3f}" if ratio else "",
            "shared_tokens": str(shared) if shared else "",
        }
    )


def resolve(
    left_path: str,
    left_col: str,
    right_path: str,
    right_col: str,
    out_path: str,
    overlap_threshold: float = 0.60,
    min_shared: int = 2,
    skip_overlap: bool = False,
) -> int:
    left_rows = _read_csv(left_path, left_col)
    right_rows = _read_csv(right_path, right_col)

    right_exact, right_aggressive = _build_index(right_rows, right_col)

    out_rows: list[dict[str, str]] = []
    seen: set[tuple] = set()

    # Pass 1+2: exact / fuzzy via index lookup.
    for lrow in left_rows:
        raw = lrow.get(left_col, "")
        n = normalize_name(raw)
        if not n:
            continue
        for rrow in right_exact.get(n, []):
            _emit(out_rows, seen, "exact", lrow, rrow, left_col, right_col)
        a = normalize_aggressive(raw)
        if a:
            for rrow in right_aggressive.get(a, []):
                _emit(out_rows, seen, "fuzzy", lrow, rrow, left_col, right_col)

    if not skip_overlap:
        # Pass 3: token overlap (O(N*M) — expensive; allow opt-out).
        for lrow in left_rows:
            l_raw = lrow.get(left_col, "")
            if not normalize_name(l_raw):
                continue
            for rrow in right_rows:
                ratio, shared = token_overlap_ratio(
                    l_raw, rrow.get(right_col, "")
                )
                if ratio >= overlap_threshold and shared >= min_shared:
                    _emit(
                        out_rows,
                        seen,
                        "token_overlap",
                        lrow,
                        rrow,
                        left_col,
                        right_col,
                        ratio=ratio,
                        shared=shared,
                    )

    fieldnames = [
        "match_type",
        "confidence",
        "left_name",
        "right_name",
        "left_normalized",
        "right_normalized",
        "left_row",
        "right_row",
        "overlap_ratio",
        "shared_tokens",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", required=True, help="Left CSV path")
    p.add_argument(
        "--left-name-col", required=True, help="Name column in left CSV"
    )
    p.add_argument("--right", required=True, help="Right CSV path")
    p.add_argument(
        "--right-name-col",
        required=True,
        help="Name column in right CSV",
    )
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.60,
        help="Jaccard overlap threshold for token_overlap tier (default 0.60)",
    )
    p.add_argument(
        "--min-shared",
        type=int,
        default=2,
        help="Minimum shared tokens for token_overlap tier (default 2)",
    )
    p.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip the O(N*M) token_overlap pass (much faster on large CSVs)",
    )
    args = p.parse_args()

    count = resolve(
        left_path=args.left,
        left_col=args.left_name_col,
        right_path=args.right,
        right_col=args.right_name_col,
        out_path=args.out,
        overlap_threshold=args.overlap_threshold,
        min_shared=args.min_shared,
        skip_overlap=args.skip_overlap,
    )
    print(f"Wrote {count} match rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
