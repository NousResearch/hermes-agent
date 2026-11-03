#!/usr/bin/env python3
"""Inventory, pseudonymize, scan and record a file before it leaves the team.

Every subcommand prints counts and metadata as JSON, never a value from the
file: whatever this prints enters the agent session and reaches the model
provider. Stdlib only.

  inventory FILE                       per-column shape counts
  pseudonymize IN OUT --columns a,b --key-file PATH [--drop c,d] [--lowercase]
  scan FILE [--originals SRC --columns a,b] [--quasi x,y --k 5]
  record FILE --destination NAME       metadata-only share record

CSV/TSV are read by header, .jsonl/.ndjson by top-level keys; anything else is
scanned as one free-text blob.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{6,}\d(?![\w.])")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
SECRET_PATTERNS = [re.compile(p) for p in (
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    r"\bsk-[A-Za-z0-9_-]{20,}\b",
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    r"\bAIza[0-9A-Za-z_-]{35}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*",
    r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S{6,}",
    r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@",
)]
TEXT_FREE_THRESHOLD = 40  # mean length above which a column is reported as free text
MIN_KEY_BYTES = 16  # generated keys are 32 bytes; anything shorter is a truncated or hand-made file


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _phones(text: str) -> int:
    n = 0
    for m in PHONE.finditer(text):
        cand = m.group(0)
        digits = re.sub(r"\D", "", cand)
        if ISO_DATE.match(cand.strip()) or not 8 <= len(digits) <= 15:
            continue
        if len(digits) < 10 and not cand.startswith("+"):
            continue
        n += 1
    return n


def _ipv6(text: str) -> int:
    n = 0
    for m in IPV6.finditer(text):
        try:
            ipaddress.IPv6Address(m.group(0))
        except ValueError:
            continue
        n += 1
    return n


def _ipv4(text: str) -> int:
    return sum(1 for m in IPV4.finditer(text) if all(int(p) <= 255 for p in m.group(0).split(".")))


def _secrets(text: str) -> int:
    """Count secret-shaped spans; overlapping pattern hits (``api_key=sk-...``) count once."""
    spans = sorted(m.span() for p in SECRET_PATTERNS for m in p.finditer(text))
    n, end = 0, -1
    for start, stop in spans:
        if start >= end:
            n += 1
        end = max(end, stop)
    return n


def shape_counts(text: str) -> Counter:
    """Count identifier and secret shapes in one string."""
    return Counter({
        "email": len(EMAIL.findall(text)),
        "phone": _phones(text),
        "card_luhn": sum(1 for m in CARD.finditer(text) if _luhn(re.sub(r"\D", "", m.group(0)))),
        "ip_address": _ipv4(text) + _ipv6(text),
        "secret": _secrets(text),
    })


def _delimiter(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return {".csv": ",", ".tsv": "\t"}.get(suffix)


def _unique_pairs(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise SystemExit("a JSON object repeats a key; the later value would hide the earlier one from the scan")
    return dict(pairs)


def read_rows(path: Path):
    """Return (columns, rows) for tabular files, or (None, [text]) for free text.

    Every cell must land under its own column: a repeated header or an overlong
    row would hide a cell from the per-column scan, so both are refused.
    """
    delim = _delimiter(path)
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        if delim:
            reader = csv.DictReader(fh, delimiter=delim)
            cols = list(reader.fieldnames or [])
            if len(set(cols)) != len(cols):
                raise SystemExit("the header repeats a column name; rename it so every column is scanned")
            rows = []
            for row in reader:
                if None in row:
                    raise SystemExit(f"line {reader.line_num} has more cells than the header")
                rows.append({k: v or "" for k, v in row.items()})
            return cols, rows
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows = []
            for line in fh:
                if line.strip():
                    obj = json.loads(line, object_pairs_hook=_unique_pairs)
                    rows.append({str(k): v if isinstance(v, str) else json.dumps(v) for k, v in obj.items()})
            cols = sorted({k for r in rows for k in r})
            return cols, rows
        return None, [fh.read()]


def _labels(cols: list[str]) -> dict[str, str]:
    """Printable name per column; a header that is itself identifier- or secret-shaped (a pivoted
    export headed by an address) prints as its position so the report never echoes it."""
    return {c: f"#{i} (name redacted)" if any(shape_counts(c).values()) else c for i, c in enumerate(cols, 1)}


def _header_hits(cols: list[str]) -> dict:
    total = Counter()
    for col in cols:
        total.update(shape_counts(col))
    return {k: v for k, v in total.items() if v}


def cmd_inventory(args) -> dict:
    cols, rows = read_rows(Path(args.file))
    if cols is None:
        return {"file": args.file, "format": "text", "shapes": dict(shape_counts(rows[0]))}
    out, labels = {}, _labels(cols)
    for col in cols:
        values = [str(r.get(col, "")) for r in rows]
        filled = [v for v in values if v.strip()]
        shapes = Counter()
        for v in filled:
            shapes.update(shape_counts(v))
        mean_len = round(sum(map(len, filled)) / len(filled), 1) if filled else 0
        out[labels[col]] = {
            "non_empty": len(filled),
            "distinct": len(set(filled)),
            "mean_length": mean_len,
            "looks_free_text": mean_len >= TEXT_FREE_THRESHOLD,
            "shapes": {k: v for k, v in shapes.items() if v},
        }
    return {"file": args.file, "rows": len(rows), "header_shapes": _header_hits(cols), "columns": out}


def _load_key(key_file: Path, out_path: Path) -> bytes:
    out_dir = out_path.resolve().parent
    if out_dir == key_file.resolve().parent or out_dir in key_file.resolve().parents:
        raise SystemExit("refused: keep --key-file outside the shipped file's folder")
    if key_file.exists():
        key = bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
        # An empty file (an interrupted first run) would make b"" the HMAC key: pseudonyms anyone can recompute.
        if len(key) < MIN_KEY_BYTES:
            raise SystemExit(f"refused: --key-file holds fewer than {MIN_KEY_BYTES} key bytes; "
                             "delete it to generate a new key")
        return key
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key.hex())
    return key


def _csv_list(value: str | None) -> list[str]:
    return [c.strip() for c in (value or "").split(",") if c.strip()]


def cmd_pseudonymize(args) -> dict:
    src, dst = Path(args.input), Path(args.output)
    if _delimiter(src) is None or _delimiter(dst) is None:
        raise SystemExit("pseudonymize reads and writes .csv/.tsv only")
    cols, rows = read_rows(src)
    pseudo, drop = _csv_list(args.columns), set(_csv_list(args.drop))
    missing = [c for c in [*pseudo, *drop] if c not in cols]
    if missing:
        raise SystemExit(f"unknown columns: {missing}")
    key = _load_key(Path(args.key_file), dst)
    kept = [c for c in cols if c not in drop]
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=kept, delimiter=_delimiter(dst), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for col in pseudo:
                value = row.get(col, "").strip()
                if args.lowercase:
                    value = value.lower()
                if value:
                    row[col] = "p_" + hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:20]
            writer.writerow(row)
    return {"output": str(dst), "rows": len(rows), "pseudonymized": pseudo, "dropped": sorted(drop),
            "columns_out": [_labels(cols)[c] for c in kept], "key_file": str(Path(args.key_file))}


def _originals_count(text_cells: list[str], src: Path, columns: list[str]) -> dict:
    cols, rows = read_rows(src)
    if cols is None:
        raise SystemExit("--originals must be a .csv/.tsv/.jsonl file with a header")
    result = {}
    blob = "\n".join(text_cells)
    for col in columns:
        if col not in cols:
            raise SystemExit(f"unknown --columns entry in originals: {col!r}")
        values = sorted({str(r.get(col, "")).strip() for r in rows} - {""}, key=len, reverse=True)
        hits = 0
        for i in range(0, len(values), 500):
            chunk = "|".join(re.escape(v) for v in values[i:i + 500])
            hits += len(re.findall(rf"(?<![\w@.-])(?:{chunk})(?![\w@-])", blob, re.IGNORECASE))
        result[col] = hits
    return result


def _quasi_groups(rows: list[dict], quasi: list[str], k: int) -> dict:
    groups = Counter(tuple(str(r.get(c, "")) for c in quasi) for r in rows)
    small = {g: n for g, n in groups.items() if n < k}
    return {"columns": quasi, "k": k, "groups_below_k": len(small), "rows_in_groups_below_k": sum(small.values())}


def _scan_requests(args, cols: list[str] | None) -> tuple[list[str], list[str]]:
    """Return (originals columns, quasi columns); a check that cannot run is refused, never reported clean."""
    columns, quasi = _csv_list(args.columns), _csv_list(args.quasi)
    if bool(args.originals) != bool(columns):
        raise SystemExit("--originals and --columns go together: name the source file and the columns to look for")
    if quasi:
        if cols is None:
            raise SystemExit("--quasi needs a tabular file")
        unknown = [c for c in quasi if c not in cols]
        if unknown:
            raise SystemExit(f"unknown --quasi columns: {unknown}")
        if args.k < 2:
            raise SystemExit("--k must be at least 2")
    return columns, quasi


def cmd_scan(args) -> dict:
    path = Path(args.file)
    cols, rows = read_rows(path)
    columns, quasi = _scan_requests(args, cols)
    cells = [rows[0]] if cols is None else [*cols, *(str(v) for r in rows for v in r.values())]
    totals = Counter()
    per_column = {}
    header = {} if cols is None else _header_hits(cols)
    totals.update(header)
    if cols is None:
        totals.update(shape_counts(rows[0]))
    else:
        labels = _labels(cols)
        for col in cols:
            c = Counter()
            for r in rows:
                c.update(shape_counts(str(r.get(col, ""))))
            totals.update(c)
            hits = {k: v for k, v in c.items() if v}
            if hits:
                per_column[labels[col]] = hits
    report = {"file": str(path), "sha256": _sha256(path), "rows": len(rows) if cols else None,
              "totals": {k: totals.get(k, 0) for k in ("email", "phone", "card_luhn", "ip_address", "secret")},
              "header": header, "per_column": per_column}
    if columns:
        report["original_values"] = _originals_count(cells, Path(args.originals), columns)
    if quasi:
        report["quasi_identifier"] = _quasi_groups(rows, quasi, args.k)
    counts = list(report["totals"].values()) + list(report.get("original_values", {}).values())
    counts.append(report.get("quasi_identifier", {}).get("rows_in_groups_below_k", 0))
    report["all_zero"] = not any(counts)
    return report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_record(args) -> dict:
    path = Path(args.file)
    cols, rows = read_rows(path)
    fields = None if cols is None else list(_labels(cols).values())
    return {"destination": args.destination, "date": _dt.date.today().isoformat(), "file": path.name,
            "fields": fields, "rows": len(rows) if cols is not None else None, "sha256": _sha256(path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("file")
    ps = sub.add_parser("pseudonymize")
    ps.add_argument("input")
    ps.add_argument("output")
    ps.add_argument("--columns", default="")
    ps.add_argument("--drop", default="")
    ps.add_argument("--key-file", required=True)
    ps.add_argument("--lowercase", action="store_true")
    sc = sub.add_parser("scan")
    sc.add_argument("file")
    sc.add_argument("--originals")
    sc.add_argument("--columns", default="")
    sc.add_argument("--quasi", default="")
    sc.add_argument("--k", type=int, default=5)
    rec = sub.add_parser("record")
    rec.add_argument("file")
    rec.add_argument("--destination", required=True)
    return p


HANDLERS = {"inventory": cmd_inventory, "pseudonymize": cmd_pseudonymize, "scan": cmd_scan, "record": cmd_record}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = HANDLERS[args.cmd](args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(json.dumps({"error": exc.code}))
            return 2
        raise
    # A parser error message can quote the offending line; report only its type.
    except (ValueError, OSError, csv.Error) as exc:
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
