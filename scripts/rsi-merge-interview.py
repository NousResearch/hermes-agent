#!/usr/bin/env python3
"""Merge an RSI model response into the audit-owned report scaffold.

Accepts either the full interview-report schema or the documented
grill-admission schema (``{profile, admissions, reporting_sentence,
suggested_fix}``). Admissions require ``--prior-report`` (the previously
validated report) to merge into; the model response may be bare JSON or
wrapped in a markdown code fence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsi_interview import (
    apply_grill_admissions,
    looks_like_grill_admissions,
    merge_interview,
    parse_model_json,
)


STORE = Path.home() / ".hermes" / "rsi"
AUDIT = STORE / "audit" / "latest.json"
GRILL_PROMPT = STORE / "grill-prompt.txt"
VALIDATOR = Path(__file__).with_name("rsi-validate-interview.py")

FULL_SCHEMA_LINE = (
    "REPORT_JSON_SCHEMA:\n"
    '{"profile":"<name>","autonomous_failures":[{"id":"<exact audited id>",'
    '"summary":"...","evidence":"...","suggested_fix":"..."}],"incomplete_tasks":'
    '[{"id":"<exact audited id>","title":"...","summary":"...","why_incomplete":"..."}],'
    '"incidents":[],"correction_feedback":[],"accounted_session_ids":[]}'
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("rsi_validate_interview", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load validator: {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return parse_model_json(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"interview unavailable or malformed: {exc}"


def _write_full_schema_grill(
    path: Path,
    base_path: Path,
    profile: str,
    errors: list[str],
    required_ids: list[str],
) -> None:
    """A grill that asks for the FULL report schema: the admission schema can
    never satisfy per-row qualitative validation on its own."""
    try:
        base = base_path.read_text(encoding="utf-8").rstrip()
    except OSError:
        base = "RSI audited your chats since the last tick. Your interview did not match that evidence."
    mismatch = json.dumps(
        {
            "profile": profile,
            "validation_errors": errors,
            "required_ids": required_ids,
        },
        sort_keys=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{base}\n\nDETERMINISTIC VALIDATION MISMATCHES:\n{mismatch}\n\n"
        f"{FULL_SCHEMA_LINE}\n\n"
        "MANDATORY_REPORT_SCAFFOLD (runner-owned; enrich but do not alter IDs/categories):\n"
        f'{{"profile":"{profile}","required_ids":{json.dumps(required_ids)}}}\n',
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("model_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--grill-prompt", type=Path, default=GRILL_PROMPT)
    parser.add_argument("--grill-output", type=Path)
    parser.add_argument(
        "--prior-report",
        type=Path,
        help="previously validated report to merge grill admissions into",
    )
    args = parser.parse_args(argv)

    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ValueError("top level must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"valid": False, "profile": args.profile, "errors": [f"malformed audit JSON: {exc}"]}, sort_keys=True))
        return 2

    model_report, input_error = _read_json(args.model_report)
    prior_report: dict[str, Any] | None = None
    if args.prior_report is not None:
        try:
            prior_report = parse_model_json(args.prior_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if input_error is None:
                input_error = f"prior report unavailable or malformed: {exc}"
            prior_report = None
        if not isinstance(prior_report, dict):
            if input_error is None:
                input_error = "prior report must be a JSON object"
            prior_report = None

    if (
        input_error is None
        and isinstance(model_report, dict)
        and model_report.get("profile") not in (None, args.profile)
    ):
        input_error = f"profile mismatch: response names {model_report.get('profile')!r}, merge requested for {args.profile!r}"
        model_report = None

    if (
        input_error is None
        and prior_report is not None
        and isinstance(model_report, dict)
        and looks_like_grill_admissions(model_report)
    ):
        merged = apply_grill_admissions(prior_report, model_report, audit)
    else:
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
    result["omitted_admission_ids"] = merged.omitted_admission_ids
    if input_error:
        result["errors"].append(input_error)
        result["valid"] = False
    if not result["valid"] and args.grill_output is not None:
        if (
            prior_report is not None
            and isinstance(model_report, dict)
            and looks_like_grill_admissions(model_report)
        ):
            # The admission path itself failed (unknown ids, unfilled rows, or
            # a malformed wrapper): the retry must produce the full schema.
            _write_full_schema_grill(
                args.grill_output,
                args.grill_prompt,
                args.profile,
                result["errors"],
                result["required_ids"],
            )
        else:
            validator._write_grill(args.grill_output, args.grill_prompt, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
