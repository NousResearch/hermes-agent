#!/usr/bin/env python3
"""Merge an RSI model response into the audit-owned report scaffold."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsi_interview import merge_interview


STORE = Path.home() / ".hermes" / "rsi"
AUDIT = STORE / "audit" / "latest.json"
GRILL_PROMPT = STORE / "grill-prompt.txt"
VALIDATOR = Path(__file__).with_name("rsi-validate-interview.py")


def _load_validator():
    spec = importlib.util.spec_from_file_location("rsi_validate_interview", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load validator: {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"interview unavailable or malformed: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("model_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--grill-prompt", type=Path, default=GRILL_PROMPT)
    parser.add_argument("--grill-output", type=Path)
    args = parser.parse_args(argv)

    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ValueError("top level must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"valid": False, "profile": args.profile, "errors": [f"malformed audit JSON: {exc}"]}, sort_keys=True))
        return 2

    model_report, input_error = _read_json(args.model_report)
    merged = merge_interview(args.profile, model_report, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged.report, sort_keys=True), encoding="utf-8")

    validator = _load_validator()
    result = validator.validate_interview(
        args.profile,
        merged.report,
        audit,
        merge_conflicts=merged.conflicts,
    )
    if input_error:
        result["errors"].append(input_error)
        result["valid"] = False
    if not result["valid"] and args.grill_output is not None:
        validator._write_grill(args.grill_output, args.grill_prompt, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
