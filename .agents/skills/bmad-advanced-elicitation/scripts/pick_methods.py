#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Serve the elicitation method catalog without loading it all into context.

The catalog is a CSV (num, category, method_name, description, output_pattern).
`description` is a one-line gist — enough to run the method; `output_pattern` is
a flexible flow guide (e.g. "assumptions → truths → new approach").

Commands:
  categories                      list category names + counts (the cheap entry point)
  list --category C [...]         the index (num/category/name/gist) for those categories
  list --all                      the whole catalog at once — deliberate; large, avoid interactively
  show NAME_OR_NUM [...]          full row for each method, matched by name or num
  random [-n N] [--category C ...] [--exclude NAME ...] [--spread]
                                  draw N at random; --spread forces category diversity
                                  (at most one per category until categories run out) —
                                  the reshuffle draw; --exclude skips already-shown methods

`list` refuses to run with neither --category nor --all: dumping the full catalog
into context must always be an explicit, deliberate choice.

`--extra SPEC` merges additional methods (customize.toml's `additional_methods`)
into every command. SPEC is either a JSON array literal (starts with `[`) or a
path to a JSON file; each item is {code, category, method_name, description,
output_pattern}. An extra whose method_name matches a catalog row
(case-insensitive) REPLACES it and keeps that row's num — retune a shipped
method; others append and get the next free nums, so new methods and whole new
categories are first-class and number-addressable everywhere.

Default output is lean tab-separated text for an LLM to read; --json for structured.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "assets" / "methods.csv"
FIELDS = ("num", "category", "method_name", "description", "output_pattern")


def load(file: Path) -> list[dict]:
    # utf-8-sig: tolerate BOM-prefixed catalogs (Excel "CSV UTF-8", Notepad)
    with open(file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            r.setdefault(k, "")
            r[k] = (r.get(k) or "").strip()
    return rows


def load_extra(spec: str) -> list[dict]:
    """Parse the --extra overlay: a JSON array literal or a path to a JSON file."""
    text = spec if spec.lstrip().startswith("[") else Path(spec).read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("--extra must be a JSON array of objects")
    rows = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"each --extra entry must be a JSON object, got: {item!r}")
        row = {k: str(item.get(k) or "").strip() for k in FIELDS}
        row["code"] = str(item.get("code") or "").strip()  # kept for traceability
        rows.append(row)
    return rows


def merge_extra(rows: list[dict], extras: list[dict]) -> list[dict]:
    """Extras replace a catalog row with the same method_name (case-insensitive),
    otherwise append — so overrides can retune shipped methods or grow the catalog.
    A replacement inherits the shipped row's num; appended extras get the next
    free nums, so every merged method stays addressable by number."""
    merged = list(rows)
    index = {r["method_name"].lower(): i for i, r in enumerate(merged)}
    for e in extras:
        key = e["method_name"].lower()
        if key in index:
            e = dict(e)
            e["num"] = e["num"] or merged[index[key]]["num"]
            merged[index[key]] = e
        else:
            index[key] = len(merged)
            merged.append(dict(e))
    next_num = max((int(r["num"]) for r in merged if r["num"].isdigit()), default=0) + 1
    for r in merged:
        if not r["num"]:
            r["num"] = str(next_num)
            next_num += 1
    return merged


def categories(rows: list[dict]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return sorted(counts.items())


def filter_cats(rows: list[dict], cats: list[str] | None) -> list[dict]:
    if not cats:
        return rows
    wanted = {c.lower() for c in cats}
    return [r for r in rows if r["category"].lower() in wanted]


def find(rows: list[dict], names: list[str]) -> tuple[list[dict], list[str]]:
    """Match each query by method_name or by num, case-insensitively."""
    by_key: dict[str, dict] = {}
    for r in rows:
        by_key[r["method_name"].lower()] = r
        if r["num"]:
            by_key.setdefault(r["num"], r)
    found, missing = [], []
    for n in names:
        r = by_key.get(n.strip().lower())
        (found if r else missing).append(r if r else n)
    return found, missing


def exclude(rows: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return rows
    skip = {n.strip().lower() for n in names}
    return [r for r in rows if r["method_name"].lower() not in skip]


def spread_sample(rows: list[dict], n: int, rng: random.Random | None = None) -> list[dict]:
    """Draw n methods with maximum category diversity: shuffle the categories,
    take one random method per category round-robin, wrapping only when there
    are fewer categories than picks."""
    rng = rng or random
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    buckets = list(by_cat.values())
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(b)
    out: list[dict] = []
    while buckets and len(out) < n:
        exhausted = []
        for b in buckets:
            if len(out) >= n:
                break
            out.append(b.pop())
            if not b:
                exhausted.append(b)
        buckets = [b for b in buckets if b not in exhausted]
    return out


def fmt_categories(cats: list[tuple[str, int]], as_json: bool) -> str:
    if as_json:
        return json.dumps([{"category": c, "count": n} for c, n in cats])
    return "\n".join(f"{c}\t{n}" for c, n in cats)


def fmt_rows(rows: list[dict], as_json: bool) -> str:
    if as_json:
        return json.dumps([{k: r[k] for k in FIELDS} for r in rows])
    return "\n".join(
        f"{r['num']}\t{r['category']}\t{r['method_name']}\t{r['description']}\t{r['output_pattern']}"
        for r in rows
    )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # catalog rows contain →; don't die on locale code pages
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=DEFAULT_FILE, help="method CSV (default: sibling assets/methods.csv)")
    p.add_argument("--extra", help="additional methods: a JSON array literal or a path to a JSON file")
    p.add_argument("--json", action="store_true", help="emit structured JSON instead of lean text")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("categories", help="list category names + counts")
    pl = sub.add_parser("list", help="the index for chosen categories (needs --category or --all)")
    pl.add_argument("--category", action="append", help="filter to a category (repeatable)")
    pl.add_argument("--all", action="store_true", help="dump the entire catalog (deliberate; large)")
    ps = sub.add_parser("show", help="full row for each named method")
    ps.add_argument("names", nargs="+", help="method names or nums")
    pr = sub.add_parser("random", help="draw methods at random")
    pr.add_argument("-n", type=int, default=1, help="how many (default 1)")
    pr.add_argument("--category", action="append", help="restrict to a category (repeatable)")
    pr.add_argument("--exclude", action="append", help="method name to skip (repeatable) — e.g. already shown")
    pr.add_argument("--spread", action="store_true", help="force category diversity across the draw")
    args = p.parse_args(argv)

    if not args.file.is_file():
        print(f"error: method file not found: {args.file}", file=sys.stderr)
        return 2
    rows = load(args.file)
    if args.extra:
        try:
            rows = merge_extra(rows, load_extra(args.extra))
        except (OSError, ValueError) as e:
            print(f"error: could not read --extra: {e}", file=sys.stderr)
            return 2

    if args.cmd == "categories":
        print(fmt_categories(categories(rows), args.json))
    elif args.cmd == "list":
        if not args.category and not args.all:
            print(
                "error: `list` needs --category (one or more) — or --all to dump the whole "
                "catalog on purpose. Use `categories` for the cheap map, or `random` to draw blind.",
                file=sys.stderr,
            )
            return 2
        print(fmt_rows(filter_cats(rows, args.category), args.json))
    elif args.cmd == "show":
        found, missing = find(rows, args.names)
        for m in missing:
            print(f"# not found: {m}", file=sys.stderr)
        if not found:
            return 1
        print(fmt_rows(found, args.json))
    elif args.cmd == "random":
        pool = exclude(filter_cats(rows, args.category), args.exclude)
        if not pool:
            print("# no methods match", file=sys.stderr)
            return 1
        n = max(0, min(args.n, len(pool)))  # clamp: never crash on a negative or oversized -n
        picks = spread_sample(pool, n) if args.spread else random.sample(pool, n)
        print(fmt_rows(picks, args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
