"""Command-line interface for Feature Parity ledger validation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .core import REGISTRY_PATH, LedgerValidationError
from .repository import load_json_document, validate_path, validate_repository

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledgers", nargs="*", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used for registry discovery and receipt verification",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="explicit contract registry for validating named ledger files",
    )
    args = parser.parse_args(argv)

    failed = False
    if not args.ledgers:
        errors = validate_repository(args.repository_root)
        if errors:
            failed = True
            print(f"{args.repository_root}: INVALID", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"{args.repository_root}: VALID")
        return 1 if failed else 0

    registry: Mapping[str, Any] | None = None
    registry_path = args.registry or (args.repository_root / REGISTRY_PATH)
    if registry_path.exists():
        try:
            registry = load_json_document(registry_path)
        except LedgerValidationError as exc:
            failed = True
            for error in exc.errors:
                print(error, file=sys.stderr)

    for path in args.ledgers:
        errors = validate_path(path, contract_registry=registry)
        if errors:
            failed = True
            print(f"{path}: INVALID", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"{path}: VALID")
    return 1 if failed else 0
