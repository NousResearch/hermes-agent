#!/usr/bin/env python3
"""Fail when a retired updater shim is missing from the tested manifest.

Every function in ``hermes_cli/``, ``tools/`` and ``gateway/`` that can start the
old-updater takeover (a bare ``stop_for_relaunch()`` or the shared
``stop_historical_updater(...)`` gate) must be listed in
``tests/fixtures/retired_updater_shims.json``.
``tests/hermes_cli/test_retired_shims_runtime.py`` parametrizes over that
manifest and calls every shim for real (#122656); tests never read source, so
this AST scan is what keeps a new shim from skipping those contracts.

Exit codes: 0 = manifest matches, 1 = drift (listed), 2 = the scan found nothing.

Usage:
    python scripts/ci/check_retired_shims_manifest.py [--write] [repo_root]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

MANIFEST = Path("tests/fixtures/retired_updater_shims.json")
_TREES = ("hermes_cli", "tools", "gateway")
_GATES = {"stop_for_relaunch", "stop_historical_updater"}
# hermes_cli.main's lazy frozen export, resolved through main.__getattr__ (no def in main.py).
_LAZY_EXPORTS = (("hermes_cli.main", "_write_update_incomplete_marker"),)


def _calls_gate(fn: ast.FunctionDef) -> bool:
    return any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
               and call.func.id in _GATES and not call.keywords for call in ast.walk(fn))


def scan(root: Path) -> list[list[str]]:
    found = []
    for path in sorted(p for d in _TREES for p in (root / d).rglob("*.py")):
        if path.name == "_old_updater.py":
            continue
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        found += [[module, fn.name] for fn in ast.walk(tree)
                  if isinstance(fn, ast.FunctionDef) and _calls_gate(fn)]
    return sorted(found + [list(item) for item in _LAZY_EXPORTS] if found else [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the retired updater shim manifest.")
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[2], type=Path)
    args = parser.parse_args(argv)
    found = scan(args.root)
    if not found:
        print("retired-shims: the scan found no shims; refusing to pass on an empty scan")
        return 2
    manifest = args.root / MANIFEST
    if args.write:
        manifest.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
        print(f"retired-shims: wrote {len(found)} shims to {MANIFEST.as_posix()}")
        return 0
    listed = json.loads(manifest.read_text(encoding="utf-8-sig")) if manifest.is_file() else []
    missing = [item for item in found if item not in listed]
    stale = [item for item in listed if item not in found]
    for module, name in missing:
        print(f"retired-shims: {module}.{name} is not in {MANIFEST.as_posix()}")
    for module, name in stale:
        print(f"retired-shims: {module}.{name} is listed but no longer a shim")
    if missing or stale:
        print("Regenerate with: python scripts/ci/check_retired_shims_manifest.py --write")
        return 1
    print(f"retired-shims: {len(found)} shims, manifest matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
