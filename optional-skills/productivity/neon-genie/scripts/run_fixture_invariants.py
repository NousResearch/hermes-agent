#!/usr/bin/env python3
"""Lightweight structural checks for evals/cases fixtures (stdlib only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402


def _cases_dir() -> Path:
    try:
        return ng_paths.evals_dir() / "cases"
    except FileNotFoundError:
        return SKILL_ROOT / "evals" / "cases"


CASES = _cases_dir()

REQUIRED_CASES = {
    "zero-option.json",
    "x402-misfit.json",
    "wayfinder-change-control.json",
    "memetic-cannot-promote.json",
    "offline-no-fabricated-observed.json",
    "buyer-beneficiary-conflation.json",
    "authority-leakage.json",
    "fictional-resource.json",
    "scorecard-cannot-override-gate.json",
    "public-gap-must-attempt-research.json",
    "public-gap-research-attempted.json",
    "private-gap-must-request.json",
    "private-gap-request-open.json",
    "private-gap-silent-invent.json",
    "completion-proof-required.json",
    "completion-proof-present.json",
    "privacy-egress-local-only.json",
    "privacy-secret-blocks-egress.json",
    "privacy-consent-purpose-bound.json",
}


def main() -> int:
    errors: list[str] = []

    if not CASES.is_dir():
        print("FAIL: evals/cases missing", file=sys.stderr)
        return 1

    found = {p.name for p in CASES.glob("*.json")}
    missing = sorted(REQUIRED_CASES - found)
    if missing:
        errors.append(f"missing required fixtures: {', '.join(missing)}")

    for path in sorted(CASES.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path.name}: root must be object")
            continue
        if "expected" not in data:
            errors.append(f"{path.name}: missing expected")
        # Known invariant samples
        if path.name == "zero-option.json":
            exp = data.get("expected") or {}
            if exp.get("status") != "NOT_COMPUTABLE":
                errors.append("zero-option.json expected.status must be NOT_COMPUTABLE")
        if path.name == "x402-misfit.json":
            exp = data.get("expected") or {}
            if exp.get("x402_fit") != "REJECT":
                errors.append("x402-misfit.json expected.x402_fit must be REJECT")
        if path.name == "authority-leakage.json":
            exp = data.get("expected") or {}
            if exp.get("grants_execution") is not False:
                errors.append("authority-leakage.json expected.grants_execution must be false")
        if path.name == "memetic-cannot-promote.json":
            exp = data.get("expected") or {}
            if exp.get("memetic_may_raise_promotion") is not False:
                errors.append("memetic-cannot-promote.json must forbid promotion raise")
        if path.name == "fictional-resource.json":
            exp = data.get("expected") or {}
            if exp.get("gate") != "G":
                errors.append("fictional-resource.json expected.gate must be G")
        if path.name == "scorecard-cannot-override-gate.json":
            exp = data.get("expected") or {}
            if exp.get("promotion_blocked") is not True:
                errors.append("scorecard-cannot-override-gate.json must block promotion")
        if path.name == "privacy-egress-local-only.json":
            exp = data.get("expected") or {}
            if exp.get("egress_decision") != "BLOCK":
                errors.append("privacy-egress-local-only.json expected.egress_decision must be BLOCK")
        if path.name == "privacy-secret-blocks-egress.json":
            exp = data.get("expected") or {}
            if exp.get("egress_decision") != "BLOCK":
                errors.append("privacy-secret-blocks-egress.json expected.egress_decision must be BLOCK")
            if exp.get("raw_secret_in_findings") is not False:
                errors.append("privacy-secret-blocks-egress.json must forbid raw secrets in findings")
        if path.name == "privacy-consent-purpose-bound.json":
            exp = data.get("expected") or {}
            if exp.get("without_consent") != "REQUEST_CONSENT":
                errors.append("privacy-consent-purpose-bound.json without_consent must be REQUEST_CONSENT")
            if exp.get("with_purpose_bound_consent") != "REDACT_THEN_ALLOW":
                errors.append(
                    "privacy-consent-purpose-bound.json with_purpose_bound_consent must be REDACT_THEN_ALLOW"
                )
            if exp.get("global_disable_consent") != "rejected":
                errors.append("privacy-consent-purpose-bound.json must reject global disable")

    if errors:
        print("FAIL: fixture invariants", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"PASS: fixture invariants ({len(found)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
