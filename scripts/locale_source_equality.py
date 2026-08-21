#!/usr/bin/env python3
"""Report locale strings that still exactly match their source catalog.

Source equality is review evidence, not an automatic defect: commands,
placeholders, brands, and technical tokens may be intentionally identical.
The command exits 0 by default; ``--fail-on-equal`` opts into a CI gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def flatten_catalog(node: Any, prefix: str = "") -> dict[str, str]:
    """Flatten string leaves from a nested catalog to dotted keys."""
    flat: dict[str, str] = {}
    if not isinstance(node, dict):
        return flat
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_catalog(value, dotted))
        elif isinstance(value, str):
            flat[dotted] = value
    return flat


def find_source_equal_keys(
    source: dict[str, str], target: dict[str, str]
) -> set[str]:
    """Return shared keys whose source and target strings are exactly equal."""
    return {
        key
        for key, source_value in source.items()
        if key in target and target[key] == source_value
    }


def group_by_namespace(keys: set[str]) -> dict[str, list[str]]:
    """Group keys by their dotted parent namespace, deterministically."""
    groups: dict[str, list[str]] = {}
    for key in sorted(keys):
        namespace, separator, _leaf = key.rpartition(".")
        groups.setdefault(namespace if separator else "(root)", []).append(key)
    return dict(sorted(groups.items()))


def filter_by_namespace(keys: set[str], namespace: str | None) -> set[str]:
    """Keep a namespace and all descendants, or all keys when unset."""
    if not namespace:
        return keys
    prefix = f"{namespace}."
    return {key for key in keys if key == namespace or key.startswith(prefix)}


def load_catalog(path: Path) -> dict[str, str]:
    """Load one YAML catalog and flatten its string leaves."""
    with path.open("r", encoding="utf-8") as handle:
        return flatten_catalog(yaml.safe_load(handle) or {})


def format_report(
    grouped: dict[str, list[str]], source_lang: str, target_lang: str
) -> str:
    """Render a stable, human-readable source-equality report."""
    lines = [f"Source-equal review: {source_lang} -> {target_lang}"]
    for namespace, keys in grouped.items():
        lines.append(f"[{namespace}] ({len(keys)})")
        lines.extend(f"  {key}" for key in keys)
    total = sum(len(keys) for keys in grouped.values())
    lines.append(f"Total: {total}")
    lines.append(
        "Note: source equality is review evidence, not an automatic defect."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--namespace", help="Limit review to a namespace and its descendants"
    )
    parser.add_argument(
        "--fail-on-equal",
        action="store_true",
        help="Exit 1 when the filtered review contains matches",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = load_catalog(args.source)
        target = load_catalog(args.target)
    except (OSError, yaml.YAMLError) as exc:
        print(f"Locale audit error: {exc}", file=sys.stderr)
        return 2

    matches = filter_by_namespace(
        find_source_equal_keys(source, target), args.namespace
    )
    print(format_report(group_by_namespace(matches), args.source.stem, args.target.stem))
    return 1 if args.fail_on_equal and matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
