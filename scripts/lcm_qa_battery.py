#!/usr/bin/env python3
"""LCM QA battery — single entrypoint (PRD-7 §3.2).

Tiers:
  0  Offline unit/integration (free, deterministic, CI-safe). Runs
     tests/context_engine via pytest and reports pass/fail + coverage map.
  1  Live shakedown on Aegis+Haiku (cheap, N<180). Delegates to
     scripts/lcm_live_recovery.py with cheap-model defaults. ALWAYS stamped
     gate:false regardless of N reached (gate-eligibility is set by TIER, N4).
  2  Statistical promotion gate (N>=180, Haiku). The only gate-eligible tier.

Cost rule (PRD-7, binding): live tiers default to a CHEAP model
(claude-haiku-4-5), never Opus. Haiku's ~200k context makes compaction fire on
natural volume.

Emits a machine-readable JSON report (+ a markdown summary path) so Tier-1
shakedown results can never be mistaken for the promotion gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # ~/.hermes/hermes-agent
DEFAULT_MODEL = "claude-haiku-4-5"

# Capability -> test mapping (PRD-7 §6 N3: the non-proxy coverage bar).
# Each entry: surface group -> the test node(s) that exercise it.
TIER0_COVERAGE = {
    "B.tool-registration-parity": [
        "test_tool_schema_set_is_exactly_the_seven",
        "test_unknown_tool_returns_error_not_exception",
    ],
    "B.adversarial-tool-fuzz": [
        "test_lcm_grep_hostile_args_return_json_never_raise",
        "test_all_tools_survive_null_and_empty_args",
        "test_bug1_null_query_specifically",
    ],
    "A/G.degenerate-compress": [
        "test_compress_empty_list_is_noop",
        "test_compress_all_tool_results_does_not_crash",
    ],
    "C.dag-invariants": [
        "test_forced_condensation_produces_multi_depth_dag",
        "test_dag_invariant_summary_smaller_than_source",
        "test_dag_invariant_no_dangling_source_ids",
        "test_dag_invariant_source_window_ordering",
    ],
    "G.fail-open-both-branches": [
        "test_fail_open_generic_exception_degrades_not_crash",
        "test_fail_open_recovery_error_is_reraised_not_swallowed",
        "test_recovery_error_carries_recoverable_flags",
    ],
    "C.fts-stale-delete": ["test_fts_no_stale_hit_after_node_delete"],
    "D.escalation-l3": ["test_escalation_l3_deterministic_truncation_converges"],
    "F.redaction": ["test_secret_not_persisted_in_summary_or_expand_hint"],
}


def run_tier0(json_out: Path) -> dict:
    """Run the offline pytest battery and build a coverage map."""
    t0 = time.time()
    # Run the QA file + the broader context_engine suite for regression.
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/context_engine",
            "-q", "--no-header",
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    passed = proc.returncode == 0
    tail = proc.stdout.strip().splitlines()[-3:] if proc.stdout else []
    report = {
        "tier": 0,
        "gate": False,  # tier-0 is correctness coverage, not the promotion gate
        "passed": passed,
        "returncode": proc.returncode,
        "summary": " ".join(tail),
        "coverage_map": TIER0_COVERAGE,
        "coverage_groups": len(TIER0_COVERAGE),
        "duration_s": round(time.time() - t0, 1),
        "note": "Tier-0 = offline correctness. Coverage MAP (not test count) is "
                "the bar (PRD-7 N3).",
    }
    json_out.write_text(json.dumps(report, indent=2))
    return report


def run_tier1(json_out: Path, model: str, n: int, allow_underpowered: bool) -> dict:
    """Live shakedown on Aegis+Haiku. ALWAYS gate:false (N4)."""
    if not allow_underpowered:
        return {
            "tier": 1, "gate": False, "error": True,
            "note": "Tier-1 is underpowered by design; pass --allow-underpowered-live.",
        }
    driver = REPO / "scripts" / "lcm_live_recovery.py"
    cmd = [
        sys.executable, str(driver),
        "--session-mode", "--profile", "aegis",
        "--model", model, "--trials", str(n),
        "--allow-underpowered-live",
    ]
    report = {
        "tier": 1,
        "gate": False,             # set by TIER, not by N reached (PRD-7 N4)
        "underpowered": True,
        "N": n,
        "model": model,
        "invocation": " ".join(cmd),
        "note": "SHAKEDOWN, not the promotion gate. gate:false regardless of N. "
                "Run the driver directly to execute; this records the contract.",
    }
    json_out.write_text(json.dumps(report, indent=2))
    return report


def run_tier2(json_out: Path, model: str, n: int) -> dict:
    """Promotion gate (N>=180, Haiku). The only gate-eligible tier."""
    if n < 180:
        return {
            "tier": 2, "gate": False, "error": True,
            "note": f"Tier-2 requires N>=180 (got {n}); an underpowered Tier-2 is "
                    "not a gate. Use Tier-1 for cheap shakedown.",
        }
    report = {
        "tier": 2,
        "gate": True,
        "N": n,
        "model": model,
        "arms": {
            "A_raw_store_fts": {"bar": "recall>=0.95, Wilson95LB>=0.90, zero confident-wrong, <=$25", "required": "pass"},
            "B_summary_node": {"bar": "measured floor: condensation runs + DAG invariants hold + no data loss", "required": "run+characterize"},
        },
        "cutover_rule": "Arm A passes AND Arm B characterized-and-not-broken.",
        "note": "Gate-eligible. Drive scripts/lcm_live_recovery.py with --session-mode "
                "at default Aegis config; this records the contract.",
    }
    json_out.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="LCM QA battery (PRD-7)")
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="cheap model only; Opus prohibited for LCM QA (PRD-7)")
    ap.add_argument("--n", type=int, default=20, help="trials (Tier-1/2)")
    ap.add_argument("--allow-underpowered-live", action="store_true")
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()

    if "opus" in args.model.lower():
        print("REFUSED: Opus is prohibited for LCM QA/E2E (PRD-7 cost rule). "
              "Use a cheap model like claude-haiku-4-5.", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else (
        REPO / f"lcm-qa-tier{args.tier}-report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.tier == 0:
        rep = run_tier0(out)
    elif args.tier == 1:
        rep = run_tier1(out, args.model, args.n, args.allow_underpowered_live)
    else:
        rep = run_tier2(out, args.model, args.n)

    print(json.dumps(rep, indent=2))
    print(f"\nReport: {out}", file=sys.stderr)
    # Exit non-zero if a tier that actually executed failed.
    if rep.get("error"):
        return 2
    if args.tier == 0 and not rep.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
