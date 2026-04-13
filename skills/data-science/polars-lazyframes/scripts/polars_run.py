#!/usr/bin/env python3
"""CLI helper for Polars lazy scans and small tabular jobs. Emits JSON on stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=str))


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")


def _scan(path: Path):
    import polars as pl

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pl.scan_parquet(path)
    if suf in (".csv", ".tsv"):
        sep = "\t" if suf == ".tsv" else ","
        return pl.scan_csv(path, separator=sep)
    if suf in (".ndjson", ".jsonl"):
        return pl.scan_ndjson(path)
    if suf == ".json":
        return pl.scan_ndjson(path)
    raise ValueError(f"unsupported extension for scan: {suf}")


def _read_eager(path: Path, n_rows: int | None = None):
    import polars as pl

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pl.read_parquet(path, n_rows=n_rows)
    if suf in (".csv", ".tsv"):
        sep = "\t" if suf == ".tsv" else ","
        return pl.read_csv(path, n_rows=n_rows, separator=sep)
    if suf in (".ndjson", ".jsonl", ".json"):
        return pl.read_ndjson(path, n_rows=n_rows)
    raise ValueError(f"unsupported extension for read: {suf}")


def cmd_inspect(path: Path) -> dict[str, Any]:
    _require_file(path)
    lf = _scan(path)
    sch = lf.collect_schema()
    pairs = zip(sch.names(), map(str, sch.dtypes()))
    return {"ok": True, "path": path.name, "schema": dict(pairs)}


def cmd_head(path: Path, n: int) -> dict[str, Any]:
    _require_file(path)
    lf = _scan(path)
    df = lf.head(n).collect()
    return {"ok": True, "path": path.name, "n": len(df), "rows": df.to_dicts()}


def cmd_convert(src: Path, dst: Path) -> dict[str, Any]:
    _require_file(src)
    df = _read_eager(src, n_rows=None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    suf = dst.suffix.lower()
    if suf == ".parquet":
        df.write_parquet(dst)
    elif suf == ".csv":
        df.write_csv(dst)
    elif suf in (".ndjson", ".jsonl"):
        df.write_ndjson(dst)
    else:
        raise ValueError(f"unsupported output extension: {suf}")
    return {"ok": True, "from": src.name, "to": dst.name, "rows": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Polars tabular helper (JSON stdout).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Lazy schema for parquet/csv/tsv/json lines.")
    p_ins.add_argument("path", type=Path)

    p_hd = sub.add_parser("head", help="First N rows (collect).")
    p_hd.add_argument("path", type=Path)
    p_hd.add_argument("-n", type=int, default=10)

    p_cv = sub.add_parser("convert", help="Eager read/write format conversion.")
    p_cv.add_argument("src", type=Path)
    p_cv.add_argument("dst", type=Path)

    args = parser.parse_args()
    try:
        if args.cmd == "inspect":
            _emit(cmd_inspect(args.path.resolve()))
        elif args.cmd == "head":
            _emit(cmd_head(args.path.resolve(), max(1, args.n)))
        else:
            _emit(cmd_convert(args.src.resolve(), args.dst.resolve()))
    except Exception as e:
        _emit({"ok": False, "error": type(e).__name__, "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
