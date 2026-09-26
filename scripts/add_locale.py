#!/usr/bin/env python3
"""Scaffold a new locale file for Hermes Agent's i18n catalog.

Usage:
    python scripts/add_locale.py <code> <display_name> [--aliases a1,a2,...]

Example:
    python scripts/add_locale.py id Indonesian --aliases bahasa,indonesian

This script:
  1. Reads locales/en.yaml as the baseline.
  2. Creates locales/<code>.yaml with English values as placeholders.
  3. Checks that agent/i18n.py has the code in SUPPORTED_LANGUAGES.
  4. Verifies key + placeholder parity against en.yaml.
  5. Prints next steps (register in SUPPORTED_LANGUAGES, add aliases).

The new file uses English values so the contributor can translate them
in-place.  The test suite (tests/agent/test_i18n.py) will fail until the
code is registered in SUPPORTED_LANGUAGES — that's intentional: it forces
the contributor to complete the registration step.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
LOCALES_DIR = ROOT / "locales"
EN_PATH = LOCALES_DIR / "en.yaml"
I18N_PATH = ROOT / "agent" / "i18n.py"

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into dotted-key dict."""
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        else:
            flat[key] = v
    return flat


def build_nested(flat: dict) -> dict:
    """Rebuild nested dict from dotted-key flat dict."""
    nested: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        node = nested
        for p in parts[:-1]:
            if p not in node:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = v
    return nested


def check_i18n_registered(code: str) -> bool:
    """Check if <code> is already in SUPPORTED_LANGUAGES."""
    text = I18N_PATH.read_text(encoding="utf-8")
    # Look for the code in the SUPPORTED_LANGUAGES tuple.
    # Matches "id" as a quoted string element in the tuple.
    pattern = re.compile(
        r'SUPPORTED_LANGUAGES.*?=.*?\(([^)]*)\)',
        re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        codes = [c.strip().strip('"').strip("'") for c in m.group(1).split(",")]
        return code in codes
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new locale file for Hermes Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "code",
        help='BCP-47 language code, e.g. "id", "ko", "pt".',
    )
    parser.add_argument(
        "display_name",
        help='Human-readable name, e.g. "Indonesian", "Korean".',
    )
    parser.add_argument(
        "--aliases",
        default="",
        help='Comma-separated aliases for _LANGUAGE_ALIASES, e.g. "bahasa,indonesian".',
    )
    args = parser.parse_args()

    code = args.code.strip().lower()
    display_name = args.display_name.strip()
    aliases = [a.strip().lower() for a in args.aliases.split(",") if a.strip()]

    # Validate inputs.
    if not re.match(r"^[a-z]{2,3}(-[a-z0-9]+)?$", code):
        print(f'Error: code "{code}" is not a valid BCP-47 language code.', file=sys.stderr)
        return 1

    target_path = LOCALES_DIR / f"{code}.yaml"
    if target_path.exists():
        print(f'Error: {target_path} already exists.', file=sys.stderr)
        return 1

    if not EN_PATH.exists():
        print(f"Error: {EN_PATH} not found. Run from the repo root.", file=sys.stderr)
        return 1

    # Load English baseline.
    with EN_PATH.open("r", encoding="utf-8") as f:
        en_data = yaml.safe_load(f) or {}
    en_flat = flatten(en_data)
    en_keys = set(en_flat.keys())

    # Build the new locale file with English values as placeholders.
    # Use the same nested structure as en.yaml for readability.
    new_data = build_nested(en_flat)

    header = (
        f"# Hermes static-message catalog -- {display_name}\n"
        f"#\n"
        f"# Only user-facing static messages from the CLI approval prompt and a handful\n"
        f"# of gateway slash-command replies live here.  Agent-generated output, log\n"
        f"# lines, error tracebacks, tool outputs, and slash-command descriptions stay\n"
        f"# in English and are NOT translated -- see agent/i18n.py for scope rationale.\n"
        f"#\n"
        f"# Keys are dotted paths; nesting below is purely for readability.  Values may\n"
        f"# contain {{placeholder}} tokens for str.format substitution.  When adding a\n"
        f"# new key, add it to EVERY locale file (en/zh/ja/de/es/fr/tr/uk/id) in the same commit --\n"
        f"# tests/agent/test_i18n.py asserts catalog parity.\n"
        f"#\n"
        f"# TODO: Translate the values below from English to {display_name}.\n"
        f"#       Preserve all {{{{placeholder}}}} tokens exactly as they appear.\n\n"
    )

    body = yaml.safe_dump(
        new_data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=10000,
    )
    target_path.write_text(header + body, encoding="utf-8")

    # Verify parity.
    with target_path.open("r", encoding="utf-8") as f:
        new_data_check = yaml.safe_load(f) or {}
    new_flat = flatten(new_data_check)
    new_keys = set(new_flat.keys())

    missing = en_keys - new_keys
    extra = new_keys - en_keys

    # Placeholder parity.
    mismatches = []
    for key, en_value in en_flat.items():
        en_p = set(PLACEHOLDER_RE.findall(en_value))
        new_p = set(PLACEHOLDER_RE.findall(new_flat.get(key, "")))
        if en_p != new_p:
            mismatches.append((key, en_p, new_p))

    print(f"\n✅ Created {target_path}")
    print(f"   Keys: {len(new_keys)} (parity with en.yaml: {'OK' if not missing and not extra else 'MISMATCH'})")
    if missing:
        print(f"   Missing: {sorted(missing)}")
    if extra:
        print(f"   Extra: {sorted(extra)}")
    if mismatches:
        print(f"   Placeholder mismatches: {len(mismatches)}")
        for k, en_p, new_p in mismatches[:5]:
            print(f"     {k}: en={en_p} new={new_p}")
    else:
        print(f"   Placeholders: all match ✓")

    # Check registration.
    registered = check_i18n_registered(code)
    print(f"\n📋 Registration status:")
    if registered:
        print(f"   ✅ '{code}' is already in SUPPORTED_LANGUAGES in agent/i18n.py")
    else:
        print(f"   ⚠️  '{code}' is NOT in SUPPORTED_LANGUAGES yet.")
        print(f"      Add it to the tuple in agent/i18n.py:")
        print(f'          "en", "zh", ..., "hu", "{code}",')
        if aliases:
            print(f"\n      Add these aliases to _LANGUAGE_ALIASES:")
            for a in aliases:
                print(f'          "{a}": "{code}",')
            print(f"      Plus standard variants:")
            print(f'          "{code}-{code}": "{code}",')

    # Run tests.
    print(f"\n🧪 To verify, run:")
    print(f"   python -m pytest tests/agent/test_i18n.py -v")
    print(f"\n📝 Next steps:")
    print(f"   1. Translate all values in {target_path}")
    print(f"   2. Register '{code}' in SUPPORTED_LANGUAGES (agent/i18n.py)")
    if aliases:
        print(f"   3. Add aliases: {', '.join(aliases)}")
    print(f"   4. Run: python -m pytest tests/agent/test_i18n.py -v")
    print(f"   5. Commit all changes in one commit (locale file + i18n.py)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
