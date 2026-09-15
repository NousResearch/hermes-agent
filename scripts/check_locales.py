#!/usr/bin/env python3
"""Check locale catalogs for missing translation keys.

Compares every locale against its English baseline and reports keys that
exist in English but not in the locale:

  * CLI/gateway catalogs:  locales/*.yaml         (baseline: locales/en.yaml)
  * Desktop catalogs:      apps/desktop/src/i18n/*.ts  (baseline: en.ts)
  * Desktop JSON catalogs: apps/desktop/src/locales/*.json  (baseline: en.json;
    checked only when the directory exists)

Missing keys never break the build — YAML lookups and `defineLocale()` both
fall back to English — so gaps accumulate silently until users see mixed
English/localized UI. This script makes the drift visible.

Usage:
  python scripts/check_locales.py               # summary table
  python scripts/check_locales.py --list zh-hant  # list missing keys
  python scripts/check_locales.py --strict      # exit 1 if anything missing

Stdlib only, so it can run in CI without installing the project.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Desktop files that are i18n machinery rather than locale catalogs.
TS_NON_CATALOG = {
    "catalog.ts",
    "context.tsx",
    "define-locale.ts",
    "index.ts",
    "languages.ts",
    "runtime.ts",
    "types.ts",
}

_YAML_KEY = re.compile(r"^(\s*)([A-Za-z0-9_.-]+):(.*)$")
_TS_KEY = re.compile(r"^(\s*)([A-Za-z0-9_$]+|'[^']+'):\s*(.*)$")


def yaml_leaf_keys(path: Path) -> set[str]:
    """Dotted paths of every key with a scalar value in a simple YAML mapping.

    Handles the subset of YAML these catalogs use: nested mappings with
    scalar leaves (including quoted and block scalars). Lists of mappings
    would need a real parser, but no locale file uses them.
    """
    keys: set[str] = set()
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _YAML_KEY.match(raw)
        if not m:
            continue
        indent, key, value = len(m.group(1)), m.group(2), m.group(3).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        if value and not value.startswith("#"):
            keys.add(dotted)
    return keys


_INLINE_KEY = re.compile(r"(?:^|[,{]\s*)([A-Za-z0-9_$]+|'[^']+')\s*:")


def _inline_object_keys(prefix: str, value: str, keys: set[str]) -> None:
    """Add dotted keys for a single-line object literal like `{ label: 'x', help: 'y' }`.

    Only scans depth-1 of the inline literal; string contents are stripped
    first so colons inside translations don't produce phantom keys.
    """
    body = value.strip().rstrip(",").strip()
    if not (body.startswith("{") and body.endswith("}")):
        return
    # Drop quoted strings and template literals, keeping quoted keys intact
    # (quoted keys are immediately followed by a colon).
    scrubbed = re.sub(r"('[^']*'|\"[^\"]*\"|`[^`]*`)(?!\s*:)", "''", body[1:-1])
    depth = 0
    top_level = []
    for ch in scrubbed:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        top_level.append(ch if depth == 0 else " ")
    for m in _INLINE_KEY.finditer("{" + "".join(top_level)):
        keys.add(f"{prefix}.{m.group(1).strip(chr(39))}")


def ts_leaf_keys(path: Path) -> set[str]:
    """Dotted paths of every leaf entry in a desktop i18n catalog.

    The catalogs are plain nested object literals (string, template-function,
    or array values). Indentation-based like the YAML walker: a line whose
    value is exactly `{` opens a nested scope, a single-line `{ ... }` object
    contributes its inner keys, and anything else is a leaf.
    """
    keys: set[str] = set()
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = _TS_KEY.match(raw)
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2).strip("'")
        value = m.group(3).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        if value == "{":
            continue
        if value.startswith("{"):
            _inline_object_keys(dotted, value, keys)
        else:
            # Includes `key:` with the value continued on the next line —
            # in an object literal a bare key line is still a leaf.
            keys.add(dotted)
    return keys


def json_leaf_keys(path: Path) -> set[str]:
    """Dotted paths of every leaf value in a JSON catalog.

    Works for both flat dot-notation catalogs (`"a.b.c": "..."`) and nested
    objects; the two spellings of the same key produce the same dotted path.
    """
    keys: set[str] = set()

    def walk(prefix: str, node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        else:
            keys.add(prefix)

    walk("", json.loads(path.read_text(encoding="utf-8")))
    return keys


def collect(root: Path = REPO_ROOT) -> dict[str, tuple[set[str], set[str]]]:
    """Map of catalog label -> (baseline keys, locale keys)."""
    results: dict[str, tuple[set[str], set[str]]] = {}

    yaml_dir = root / "locales"
    en_yaml = yaml_leaf_keys(yaml_dir / "en.yaml")
    for path in sorted(yaml_dir.glob("*.yaml")):
        if path.name == "en.yaml":
            continue
        results[f"locales/{path.name}"] = (en_yaml, yaml_leaf_keys(path))

    ts_dir = root / "apps" / "desktop" / "src" / "i18n"
    en_ts = ts_leaf_keys(ts_dir / "en.ts")
    for path in sorted(ts_dir.glob("*.ts")):
        if path.name in TS_NON_CATALOG or path.name == "en.ts" or path.name.endswith(".test.ts"):
            continue
        results[f"desktop/{path.name}"] = (en_ts, ts_leaf_keys(path))

    # Hybrid JSON catalogs (proposed in #38846); skipped until that lands.
    json_dir = root / "apps" / "desktop" / "src" / "locales"
    if (json_dir / "en.json").is_file():
        en_json = json_leaf_keys(json_dir / "en.json")
        for path in sorted(json_dir.glob("*.json")):
            if path.name == "en.json":
                continue
            results[f"desktop-json/{path.name}"] = (en_json, json_leaf_keys(path))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 if any locale is missing keys")
    parser.add_argument("--list", metavar="LOCALE", help="list missing keys for one locale (e.g. zh-hant, ja)")
    args = parser.parse_args()

    results = collect()

    if args.list:
        wanted = args.list.lower()
        matched = False
        for label, (baseline, keys) in results.items():
            stem = label.split("/")[-1].rsplit(".", 1)[0].lower()
            if stem != wanted:
                continue
            matched = True
            missing = sorted(baseline - keys)
            print(f"{label}: {len(missing)} missing")
            for key in missing:
                print(f"  {key}")
        if not matched:
            print(f"no catalog named {args.list!r}", file=sys.stderr)
            return 2
        return 0

    width = max(len(label) for label in results)
    any_missing = False
    for label, (baseline, keys) in results.items():
        missing = len(baseline - keys)
        extra = len(keys - baseline)
        status = "OK" if missing == 0 else f"{missing} missing"
        if extra:
            status += f", {extra} extra"
        if missing:
            any_missing = True
        print(f"{label:<{width}}  {status}")

    if any_missing:
        print("\nRun with --list <locale> to see the missing keys.")
    return 1 if (args.strict and any_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
