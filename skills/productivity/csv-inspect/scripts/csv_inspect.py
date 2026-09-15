#!/usr/bin/env python3
"""Profile a CSV/TSV file: schema inference, nulls, distincts, numerics.

Stdlib-only so it runs on any host without installs. JSON on stdout;
errors on stderr with nonzero exit.

Usage:
  csv_inspect.py data.csv
  csv_inspect.py data.csv --head 3 --pretty
  csv_inspect.py export.csv --delimiter ";" --encoding cp1252
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter

_SNIFF_CANDIDATES = [",", "\t", ";", "|"]
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$"
    r"|^\d{1,2}/\d{1,2}/\d{4}$"
    r"|^\d{1,2}\.\d{1,2}\.\d{4}$"
)
_BOOL_VALUES = {"true", "false", "yes", "no", "t", "f", "y", "n"}


def _parse_number(raw):
    s = raw.strip().replace("$", "").replace("%", "")
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _classify(value):
    s = value.strip()
    if _DATE_RE.match(s):
        return "date"
    low = s.lower()
    if low in _BOOL_VALUES:
        return "bool"
    clean = s.lstrip("+-").replace(",", "")
    if clean.isdigit():
        return "int"
    try:
        float(s.replace(",", ""))
        return "float"
    except ValueError:
        return "text"


def _infer_type(non_null):
    counts = Counter(_classify(v) for v in non_null)
    if not counts:
        return "empty"
    return counts.most_common(1)[0][0]


def _sniff(sample, forced):
    if forced:
        return forced
    for cand in _SNIFF_CANDIDATES:
        if cand in sample:
            counts = {c: sample.count(c) for c in _SNIFF_CANDIDATES if c in sample}
            return max(counts, key=counts.get)
    return ","


def profile(path, delimiter=None, encoding=None, no_header=False,
            head=0, na_values=(), distinct_scan=10000):
    raw = open(path, "rb").read()
    enc_used = encoding
    if not enc_used:
        if raw.startswith(b"\xef\xbb\xbf"):
            enc_used = "utf-8-sig"
        else:
            try:
                raw.decode("utf-8")
                enc_used = "utf-8"
            except UnicodeDecodeError:
                enc_used = "cp1252"
    text = raw.decode(enc_used, errors="replace")
    delim = _sniff(text[:65536], delimiter)

    reader = csv.reader(text.splitlines(), delimiter=delim)
    rows = list(reader)
    if not rows:
        raise SystemExit(f"csv_inspect: {path}: no rows parsed")
    if no_header:
        header, data = [], rows
        width = max(len(r) for r in rows)
        names = [f"col_{i + 1}" for i in range(width)]
    else:
        header, data = rows[0], rows[1:]
        names = [h.strip() or f"col_{i + 1}" for i, h in enumerate(header)]

    na = set(na_values) | {""}
    blank_rows = sum(1 for r in data if not any(c.strip() for c in r))
    columns = []
    width = len(names)
    for i, name in enumerate(names[:width]):
        vals = [(r[i].strip() if i < len(r) and r[i] is not None else "")
                for r in data]
        non_null = [v for v in vals if v not in na]
        col = {
            "name": name,
            "type": _infer_type(non_null),
            "null_count": len(vals) - len(non_null),
            "distinct": len(set(non_null[:distinct_scan])),
        }
        if non_null:
            col["null_pct"] = round(100.0 * col["null_count"] / len(vals), 1)
            col["sample_values"] = non_null[:5]
        nums = [n for n in (_parse_number(v) for v in non_null) if n is not None]
        if nums and col["type"] in ("int", "float"):
            col["min"] = min(nums)
            col["max"] = max(nums)
            col["mean"] = round(sum(nums) / len(nums), 4)
        dates = sorted(v for v in non_null if _DATE_RE.match(v))
        if dates and col["type"] == "date":
            col["min_value"] = dates[0]
            col["max_value"] = dates[-1]
        columns.append(col)

    out = {
        "file": path,
        "encoding": enc_used,
        "delimiter": delim,
        "rows": len(data),
        "columns": len(names),
        "blank_rows": blank_rows,
        "header_row_present": not no_header,
    }
    if head:
        out["head"] = data[:head]
    out["column_profiles"] = columns
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--delimiter", default=None)
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--na-values", nargs="*", default=[])
    ap.add_argument("--distinct-scan", type=int, default=10000)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    result = profile(
        args.path,
        delimiter=args.delimiter,
        encoding=args.encoding,
        no_header=args.no_header,
        head=args.head,
        na_values=args.na_values,
        distinct_scan=args.distinct_scan,
    )
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    main()
