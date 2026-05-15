#!/usr/bin/env python3
"""Validate npm package-lock.json files for basic supply-chain invariants.

This check is intentionally read-only and does not run npm. It rejects registry
package entries that point outside the expected npm registry or lack integrity
hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_ALLOWED_PREFIXES = ("https://registry.npmjs.org/",)


def iter_default_lockfiles(root: Path) -> list[Path]:
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".worktrees",
        "__pycache__",
        "node_modules",
        "venv",
    }
    lockfiles: list[Path] = []
    for path in root.rglob("package-lock.json"):
        rel_parts = path.relative_to(root).parts
        if any(part in ignored_parts for part in rel_parts):
            continue
        lockfiles.append(path)
    return sorted(lockfiles)


def validate_lockfile(lockfile: Path, allowed_prefixes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    data = json.loads(lockfile.read_text(encoding="utf-8"))
    packages = data.get("packages", {})
    if not isinstance(packages, dict):
        return [f"{lockfile}: packages must be an object"]

    for package_path, metadata in packages.items():
        if not isinstance(metadata, dict):
            continue
        resolved = metadata.get("resolved")
        integrity = metadata.get("integrity")
        if not isinstance(resolved, str) or not resolved.startswith("http"):
            continue

        if not any(resolved.startswith(prefix) for prefix in allowed_prefixes):
            parsed = urlparse(resolved)
            failures.append(
                f"{lockfile}: {package_path}: unexpected resolved URL host "
                f"{parsed.netloc!r}: {resolved}"
            )
            continue

        if not integrity:
            failures.append(f"{lockfile}: {package_path}: registry package missing integrity hash")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lockfiles",
        nargs="*",
        type=Path,
        help="package-lock.json files to validate. Defaults to all repo lockfiles outside node_modules.",
    )
    parser.add_argument(
        "--allowed-prefix",
        action="append",
        default=[],
        help="Allowed resolved URL prefix. May be passed multiple times. Defaults to npmjs registry.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    lockfiles = args.lockfiles or iter_default_lockfiles(root)
    allowed_prefixes = tuple(args.allowed_prefix or DEFAULT_ALLOWED_PREFIXES)

    failures: list[str] = []
    for lockfile in lockfiles:
        failures.extend(validate_lockfile(lockfile, allowed_prefixes))

    if failures:
        print("npm lockfile validation failed:")
        print("\n".join(failures))
        return 1

    print(f"validated {len(lockfiles)} npm lockfile(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
