#!/usr/bin/env python3
"""Validate B9 coding-task decomposition packets against schema contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"jsonschema dependency unavailable: {exc}")


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "docs" / "ops" / "schemas" / "b9_coding_task_decomposition_contract.schema.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_packets(payload: Any):
    if isinstance(payload, list):
        for idx, row in enumerate(payload):
            yield f"[{idx}]", row
    else:
        yield "", payload


def validate_file(packet_path: Path, schema: dict[str, Any]) -> tuple[bool, str]:
    payload = _load_json(packet_path)
    validator = Draft202012Validator(schema)

    failures: list[str] = []
    for prefix, packet in _iter_packets(payload):
        errors = sorted(
            validator.iter_errors(packet),
            key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
        )
        if not errors:
            continue
        first = errors[0]
        data_path = "/".join(str(p) for p in first.absolute_path) or "$"
        failures.append(f"{packet_path}{prefix}: {data_path}: {first.message}")

    if failures:
        return False, "\n".join(failures)
    return True, f"{packet_path}: OK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", nargs="+", help="Path(s) to packet JSON files (object or array of objects).")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Schema path.")
    args = parser.parse_args()

    schema_path = Path(args.schema).resolve()
    if not schema_path.exists():
        print(f"schema not found: {schema_path}", file=sys.stderr)
        return 2

    schema_payload = _load_json(schema_path)
    ok = True
    for raw_packet in args.packet:
        packet_path = Path(raw_packet).resolve()
        if not packet_path.exists():
            print(f"packet not found: {packet_path}", file=sys.stderr)
            ok = False
            continue
        passed, detail = validate_file(packet_path, schema_payload)
        print(detail)
        ok = ok and passed

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
