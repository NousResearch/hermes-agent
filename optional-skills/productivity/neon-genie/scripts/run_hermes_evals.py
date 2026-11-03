#!/usr/bin/env python3
"""Run Neon Genie golden eval fixtures against deterministic gate logic.

Packaging-level invariants only — not full product invention. Each case under
`evals/cases/` is evaluated from `input` (when present) and compared to `expected`.

Usage:
  python scripts/run_hermes_evals.py
  python scripts/run_hermes_evals.py --case zero-option
  python scripts/run_hermes_evals.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402
import privacy_runtime  # noqa: E402

EvalFn = Callable[[dict[str, Any]], dict[str, Any]]


def eval_zero_option(inp: dict[str, Any]) -> dict[str, Any]:
    skills = inp.get("skills") or []
    access = inp.get("access") or []
    constraints = [str(c).lower() for c in (inp.get("constraints") or [])]
    zero_capital = any("no capital" in c or "zero capital" in c for c in constraints)
    if not skills and not access and (zero_capital or inp.get("goal") == "first_cash"):
        return {
            "status": "NOT_COMPUTABLE",
            "reason": "No executable capabilities or access supplied",
        }
    return {
        "status": "PROPOSED",
        "reason": "Capabilities or access present; full Zero Option loop is prose-runtime",
    }


def eval_x402_misfit(inp: dict[str, Any]) -> dict[str, Any]:
    service = str(inp.get("service") or "").lower()
    persistent = bool(inp.get("persistent_relationship"))
    # Ornamental x402: long-lived subjective relationship beats machine micropayment
    if persistent and any(
        k in service for k in ("consulting", "retainer", "relationship", "subjective")
    ):
        return {"x402_fit": "REJECT"}
    if inp.get("conventional_billing_superior"):
        return {"x402_fit": "REJECT"}
    return {"x402_fit": "CANDIDATE"}


def eval_wayfinder_change_control(inp: dict[str, Any]) -> dict[str, Any]:
    # Skill-level constant; handoff packets must never rewrite product intent.
    return {"product_intent_changes_require_neon_genie_review": True}


def eval_memetic_cannot_promote(inp: dict[str, Any]) -> dict[str, Any]:
    promotion = str(inp.get("promotion_state") or "RAW_SIGNAL")
    evidence_fail = str(inp.get("evidence_gate") or "").upper() == "FAIL"
    memetic = float(inp.get("memetic_score") or 0.0)
    may_raise = not evidence_fail and memetic >= 0.0
    # Gate D: memetic never raises promotion when evidence fails
    if evidence_fail:
        may_raise = False
    return {
        "promotion_state_max": promotion if evidence_fail else promotion,
        "memetic_may_raise_promotion": may_raise,
        "gate": "D",
    }


def eval_offline_no_fabricated_observed(inp: dict[str, Any]) -> dict[str, Any]:
    research = inp.get("research") or {}
    offline = bool(research.get("offline") or research.get("enabled") is False)
    source = str(inp.get("claim_source") or "")
    if offline and source in {"model_prior_only", "model_prior"}:
        return {
            "max_label": "SPECULATIVE",
            "forbidden_label": "OBSERVED",
            "gate": "N",
        }
    return {
        "max_label": "OBSERVED",
        "forbidden_label": None,
        "gate": "N",
    }


def eval_buyer_beneficiary_conflation(inp: dict[str, Any]) -> dict[str, Any]:
    roles = inp.get("roles") or {}
    values = [roles.get(k) for k in ("user", "beneficiary", "buyer", "payer", "authorizer")]
    values = [v for v in values if v is not None]
    separated = bool(inp.get("evidence_of_separation"))
    if values and len(set(values)) == 1 and not separated:
        return {
            "status": "GATE_FAIL",
            "gate": "C",
            "reason": "Buyer and beneficiary roles conflated without evidence of identity",
        }
    return {"status": "PASS", "gate": "C", "reason": "roles separated or evidenced"}


def eval_authority_leakage(inp: dict[str, Any]) -> dict[str, Any]:
    implies = set(inp.get("packet_implies") or [])
    auth = inp.get("authority") or {}
    forbidden = {"submit_application", "spend_budget", "publish_campaign", "modify_repo", "execute"}
    leak = bool(implies & forbidden)
    exec_allowed = bool(auth.get("execution"))
    spend_allowed = bool(auth.get("spending"))
    pub_allowed = bool(auth.get("publishing"))
    if leak and not (exec_allowed or spend_allowed or pub_allowed):
        return {
            "status": "GATE_FAIL",
            "gate": "E",
            "grants_execution": False,
        }
    if leak and (exec_allowed or spend_allowed or pub_allowed):
        # Still advisory skill: packaging never grants execution
        return {
            "status": "GATE_FAIL",
            "gate": "E",
            "grants_execution": False,
        }
    return {"status": "PASS", "gate": "E", "grants_execution": False}


def eval_fictional_resource(inp: dict[str, Any]) -> dict[str, Any]:
    constraints = [str(c).lower() for c in (inp.get("constraints") or [])]
    no_fiction = any("no fictional" in c or "zero fiction" in c for c in constraints)
    invented = inp.get("invented_resources") or []
    if no_fiction and invented:
        return {
            "status": "GATE_FAIL",
            "gate": "G",
            "reason": "Fictional resources declared under zero-fiction constraints",
        }
    return {"status": "PASS", "gate": "G", "reason": "no fictional resource violation"}


def eval_scorecard_cannot_override_gate(inp: dict[str, Any]) -> dict[str, Any]:
    gate_fail = str(inp.get("mandatory_gate") or "").upper() == "FAIL"
    score = float(inp.get("composite_score") or 0.0)
    if gate_fail:
        return {
            "status": "GATE_FAIL",
            "promotion_blocked": True,
            "reason": "Composite score cannot override mandatory gate failure",
            "composite_score_ignored": score,
        }
    return {
        "status": "PASS",
        "promotion_blocked": False,
        "reason": "no mandatory gate failure",
        "composite_score_ignored": None,
    }


def eval_public_gap_must_attempt_research(inp: dict[str, Any]) -> dict[str, Any]:
    sens = str(inp.get("sensitivity") or "")
    tools = bool(inp.get("host_tools_available"))
    attempted = bool(inp.get("research_attempted"))
    label = str(inp.get("claim_label_emitted") or "")
    if sens == "public" and tools and not attempted and label in {
        "OBSERVED",
        "NOT_COMPUTABLE",
        "INFERRED",
    }:
        # NOT_COMPUTABLE without attempt also fails when tools available
        return {
            "status": "GATE_FAIL",
            "gate": "P",
            "reason": "Public gap requires research attempt before OBSERVED or NOT_COMPUTABLE",
        }
    if sens == "public" and tools and not attempted:
        return {
            "status": "GATE_FAIL",
            "gate": "P",
            "reason": "Public gap requires research attempt before OBSERVED or NOT_COMPUTABLE",
        }
    return {"status": "PASS", "gate": "P"}


def eval_private_gap_must_request(inp: dict[str, Any]) -> dict[str, Any]:
    sens = str(inp.get("sensitivity") or "")
    emitted = bool(inp.get("data_request_emitted"))
    blocks = bool(inp.get("blocks_decision") or inp.get("blocks_promotion"))
    if sens == "private" and blocks and not emitted:
        return {
            "status": "GATE_FAIL",
            "gate": "Q",
            "reason": "Private decision-critical gap requires DataRequest",
        }
    promotion_capped = bool(inp.get("blocks_promotion")) and str(
        inp.get("data_request_status") or ""
    ) == "open"
    return {
        "status": "PASS",
        "gate": "Q",
        "promotion_capped": promotion_capped,
    }


def eval_private_gap_silent_invent(inp: dict[str, Any]) -> dict[str, Any]:
    sens = str(inp.get("sensitivity") or "")
    emitted = bool(inp.get("data_request_emitted"))
    label = str(inp.get("claim_label_emitted") or "")
    source = str(inp.get("claim_source") or "")
    if sens == "private" and not emitted and label == "OBSERVED":
        return {
            "status": "GATE_FAIL",
            "gate": "R",
            "reason": "Silent invent of private facts as OBSERVED is forbidden",
        }
    if sens == "private" and not emitted and source == "model_prior_only" and label == "OBSERVED":
        return {
            "status": "GATE_FAIL",
            "gate": "R",
            "reason": "Silent invent of private facts as OBSERVED is forbidden",
        }
    return {"status": "PASS", "gate": "R"}


def eval_completion_proof(inp: dict[str, Any]) -> dict[str, Any]:
    has_proof = bool(inp.get("has_completion_proof") or inp.get("completion_proof"))
    promo = str(inp.get("promotion_state") or "RAW_SIGNAL")
    high = promo in {
        "TESTABLE",
        "SERVICE_FIRST",
        "SERVICE_PROVEN",
        "SPEC_COMPLETE",
        "WAYFINDER_READY",
        "BUILD_READY",
        "CANON_CANDIDATE",
    }
    if high and not has_proof:
        return {
            "status": "GATE_FAIL",
            "gate": "PROOF",
            "reason": "completion_proof required before TESTABLE or higher promotion",
        }
    return {"status": "PASS", "gate": "PROOF"}


def eval_privacy_egress_local_only(inp: dict[str, Any]) -> dict[str, Any]:
    mode = str(inp.get("privacy_mode") or "local_only")
    ctx = privacy_runtime.default_privacy_context(mode=mode)
    decision = privacy_runtime.privacy_egress_check(
        str(inp.get("payload") or "public market size"),
        inp.get("destination"),
        inp.get("purpose"),
        ctx,
    )
    return {
        "egress_decision": decision,
        "external_actions": list(ctx.get("external_actions") or []),
        "reason": "local_only repository boundary blocks Neon-Genie-initiated egress",
    }


def eval_privacy_secret_blocks_egress(inp: dict[str, Any]) -> dict[str, Any]:
    mode = str(inp.get("privacy_mode") or "external_research_allowed")
    ctx = privacy_runtime.default_privacy_context(mode=mode)
    secret = "sk_abcdefghijklmnopqrstuvwxyz123456"
    payload_class = str(inp.get("payload_class") or "api_key")
    if payload_class == "api_key":
        payload = secret
    else:
        payload = secret
    decision = privacy_runtime.privacy_egress_check(
        payload,
        inp.get("destination") or "example.com",
        inp.get("purpose") or "search",
        ctx,
    )
    findings_blob = str(privacy_runtime.privacy_findings(payload))
    receipt_blob = json.dumps({"note": "blocked", "findings": privacy_runtime.privacy_findings(payload)})
    return {
        "egress_decision": decision,
        "raw_secret_in_findings": secret in findings_blob,
        "raw_secret_in_receipt": secret in receipt_blob,
        "reason": "credential-like material never egresses and never appears in findings/receipts",
    }


def eval_privacy_consent_purpose_bound(inp: dict[str, Any]) -> dict[str, Any]:
    mode = str(inp.get("privacy_mode") or "external_research_allowed")
    payload = (
        "customer list: alice@example.com bob@example.com "
        "carol@example.com dana@example.com"
    )
    destination = inp.get("destination") or "crm.example.com"
    purpose = inp.get("purpose") or "crm enrichment research"
    bare = privacy_runtime.default_privacy_context(mode=mode)
    without = privacy_runtime.privacy_egress_check(payload, destination, purpose, bare)

    consent_in = inp.get("consent") or {}
    consent = privacy_runtime.build_consent_record(
        purpose=str(consent_in.get("purpose") or purpose),
        categories_allowed=list(
            consent_in.get("categories_allowed")
            or ["private_email_list", "private_customer_list"]
        ),
        destinations=[destination],
        issued_at="2026-01-01T00:00:00Z",
    )
    with_ctx = privacy_runtime.default_privacy_context(mode=mode, consents=[consent])
    with_consent = privacy_runtime.privacy_egress_check(payload, destination, purpose, with_ctx)
    prep = privacy_runtime.prepare_egress(
        payload,
        destination,
        purpose,
        with_ctx,
        recorded_at="2026-07-31T12:00:00Z",
    )
    global_rejected = "rejected"
    try:
        privacy_runtime.validate_consent_record(
            {
                "consent_id": "bad",
                "scope": "global_disable",
                "purpose": "anything",
                "categories_allowed": ["all"],
                "source_class": "operator",
                "issued_at": "2026-01-01T00:00:00Z",
            }
        )
        global_rejected = "accepted"
    except ValueError:
        global_rejected = "rejected"

    return {
        "without_consent": without,
        "with_purpose_bound_consent": with_consent,
        "global_disable_consent": global_rejected,
        "private_source_persisted": bool(
            (prep.get("redaction") or {}).get("private_source_persisted")
        ),
    }


EVALUATORS: dict[str, EvalFn] = {
    "zero-option.json": eval_zero_option,
    "x402-misfit.json": eval_x402_misfit,
    "wayfinder-change-control.json": eval_wayfinder_change_control,
    "memetic-cannot-promote.json": eval_memetic_cannot_promote,
    "offline-no-fabricated-observed.json": eval_offline_no_fabricated_observed,
    "buyer-beneficiary-conflation.json": eval_buyer_beneficiary_conflation,
    "authority-leakage.json": eval_authority_leakage,
    "fictional-resource.json": eval_fictional_resource,
    "scorecard-cannot-override-gate.json": eval_scorecard_cannot_override_gate,
    "public-gap-must-attempt-research.json": eval_public_gap_must_attempt_research,
    "public-gap-research-attempted.json": eval_public_gap_must_attempt_research,
    "private-gap-must-request.json": eval_private_gap_must_request,
    "private-gap-request-open.json": eval_private_gap_must_request,
    "private-gap-silent-invent.json": eval_private_gap_silent_invent,
    "completion-proof-required.json": eval_completion_proof,
    "completion-proof-present.json": eval_completion_proof,
    "privacy-egress-local-only.json": eval_privacy_egress_local_only,
    "privacy-secret-blocks-egress.json": eval_privacy_secret_blocks_egress,
    "privacy-consent-purpose-bound.json": eval_privacy_consent_purpose_bound,
}


def subset_match(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Every expected key must equal actual (expected is a subset contract)."""
    errs: list[str] = []
    for key, want in expected.items():
        if key not in actual:
            errs.append(f"missing key {key!r} (want {want!r})")
        elif actual[key] != want:
            errs.append(f"{key}: got {actual[key]!r}, want {want!r}")
    return errs


def run_case(path: Path) -> tuple[bool, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = data.get("expected")
    if not isinstance(expected, dict):
        return False, {"case": path.name, "error": "expected must be object"}

    fn = EVALUATORS.get(path.name)
    if fn is None:
        return False, {
            "case": path.name,
            "error": "no evaluator registered for this fixture",
        }

    inp = data.get("input")
    if inp is None:
        inp = {}
    if not isinstance(inp, dict):
        return False, {"case": path.name, "error": "input must be object when present"}

    actual = fn(inp)
    errs = subset_match(actual, expected)
    ok = not errs
    return ok, {
        "case": path.name,
        "pass": ok,
        "expected": expected,
        "actual": actual,
        "errors": errs,
        "gate": actual.get("gate") or expected.get("gate"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Neon Genie golden eval fixtures")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case stem or filename (repeatable). Default: all cases with evaluators.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        cases_dir = ng_paths.evals_dir() / "cases"
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if not cases_dir.is_dir():
        print("FAIL: evals/cases missing (also tried examples/evals/cases)", file=sys.stderr)
        return 1

    if args.case:
        paths: list[Path] = []
        for c in args.case:
            name = c if c.endswith(".json") else f"{c}.json"
            p = cases_dir / name
            if not p.is_file():
                print(f"FAIL: case not found: {name}", file=sys.stderr)
                return 1
            paths.append(p)
    else:
        paths = sorted(cases_dir.glob("*.json"))

    results: list[dict[str, Any]] = []
    failed = 0
    for path in paths:
        ok, report = run_case(path)
        results.append(report)
        if not ok:
            failed += 1

    summary = {
        "skill": "neon-genie",
        "runner": "scripts/run_hermes_evals.py",
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            status = "PASS" if r.get("pass") else "FAIL"
            print(f"{status}: {r.get('case')}")
            for e in r.get("errors") or []:
                print(f"  - {e}")
            if r.get("error"):
                print(f"  - {r['error']}")
        print(
            f"{'PASS' if failed == 0 else 'FAIL'}: hermes evals "
            f"{summary['passed']}/{summary['total']}"
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
