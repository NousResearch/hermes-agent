#!/usr/bin/env python3
"""Reject exact resurrection of tests removed by the July 2026 prune.

The manifest is a set of SHA-256 records over ``path::qualname`` plus the
normalized AST.  It therefore does not cap suite growth and does not reject a
new or materially changed test.  It only rejects a function that is byte-for-
semantic-byte the pruned test at its original identity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import warnings
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_ROOT = REPO_ROOT / "tests"
DEFAULT_MANIFEST = DEFAULT_TESTS_ROOT / "fixtures" / "pruned_test_fingerprints.sha256bin"


def encode_fingerprint(key: str, node: ast.AST) -> bytes:
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(f"{key}\0{normalized}".encode()).digest()


def read_manifest(path: Path) -> set[bytes]:
    payload = path.read_bytes()
    if len(payload) % 32:
        raise ValueError(f"manifest size {len(payload)} is not a multiple of 32 bytes")
    return {payload[offset : offset + 32] for offset in range(0, len(payload), 32)}


def _functions(body: list[ast.stmt], parents: tuple[str, ...] = ()) -> Iterator[tuple[str, ast.AST]]:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join((*parents, node.name))
            if node.name.startswith("test_"):
                yield qualname, node
            yield from _functions(node.body, (*parents, node.name))
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node.body, (*parents, node.name))


def find_resurrections(tests_root: Path, banned: set[bytes]) -> list[str]:
    problems: list[str] = []
    for path in sorted(tests_root.rglob("test*.py")):
        rel = path.relative_to(tests_root).as_posix()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for qualname, node in _functions(tree.body):
            key = f"{rel}::{qualname}"
            if encode_fingerprint(key, node) in banned:
                problems.append(key)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=DEFAULT_TESTS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    try:
        banned = read_manifest(args.manifest)
        problems = find_resurrections(args.tests_root, banned)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        print(f"::error::test-prune resurrection guard could not run: {exc}")
        return 2

    if problems:
        print(
            "::error::exact low-value tests removed by the July 2026 prune "
            "were resurrected:"
        )
        for key in problems:
            print(f"  {key}")
        return 1

    print(
        f"::notice::{len(banned)} dormant pruned-test fingerprints checked; "
        "no exact resurrection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
