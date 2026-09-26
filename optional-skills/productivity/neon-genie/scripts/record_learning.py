#!/usr/bin/env python3
"""Append a PROPOSED learning-ledger observation (stdlib only).

Never auto-applies to skill corpus or canon.

Usage:
  python scripts/record_learning.py --class proof_failed --summary "No cash in 7 days" \\
    --ledger out/neon-genie/learning-ledger.jsonl --run-id ng_run_abc

  python scripts/record_learning.py --class buyer_failure --summary "…" \\
    --envelope out/neon-genie/demo/run-envelope.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import privacy_runtime  # noqa: E402

CLASSES = {
    "failed_opportunity",
    "brittle_integration",
    "buyer_failure",
    "distribution_failure",
    "anti_capture_failure",
    "proof_obtained",
    "proof_failed",
    "data_request_satisfied",
    "false_route",
    "false_gate",
    "other",
}


def load_envelope(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL: NG-LEARN-001: cannot read envelope {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append learning ledger entry")
    parser.add_argument(
        "--class",
        dest="obs_class",
        required=True,
        choices=sorted(CLASSES),
        help="Observation class",
    )
    parser.add_argument("--summary", required=True, help="What was observed")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("out/neon-genie/learning-ledger.jsonl"),
        help="Append-only JSONL path (default out/neon-genie/learning-ledger.jsonl)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Canonical run_id from run-envelope.json (preferred linkage)",
    )
    parser.add_argument(
        "--envelope",
        type=Path,
        default=None,
        help="Path to run-envelope.json — extracts run_id, profiles, recipe",
    )
    parser.add_argument("--source-run", default="", help="Legacy: receipt path or free text")
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated related profiles",
    )
    parser.add_argument(
        "--gates",
        default="",
        help="Comma-separated related gate ids",
    )
    parser.add_argument(
        "--routing-quality",
        default="",
        choices=["", "correct", "over_routed", "under_routed", "wrong_profile", "n/a"],
        help="Optional routing quality feedback",
    )
    parser.add_argument(
        "--gate-quality",
        default="",
        choices=["", "correct_fail_closed", "false_positive_block", "false_negative_allowed", "n/a"],
        help="Optional gate quality feedback",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id.strip()
    source_run = args.source_run.strip()
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    recipe = None
    envelope_path = None
    privacy_ctx: dict[str, Any] | None = None

    if args.envelope is not None:
        ep = args.envelope if args.envelope.is_absolute() else Path.cwd() / args.envelope
        if not ep.is_file():
            # try skill-root relative
            alt = SKILL_ROOT / args.envelope
            ep = alt if alt.is_file() else ep
        if not ep.is_file():
            print(f"FAIL: NG-LEARN-002: envelope not found: {args.envelope}", file=sys.stderr)
            return 1
        env = load_envelope(ep)
        envelope_path = str(ep)
        run_id = run_id or str(env.get("run_id") or "")
        if not profiles:
            profiles = list(env.get("resolved_profiles") or env.get("selected_profiles") or [])
        recipe = (env.get("request") or {}).get("recipe")
        if not source_run:
            source_run = run_id or envelope_path
        if isinstance(env.get("privacy"), dict):
            privacy_ctx = env["privacy"]

    if not source_run:
        source_run = run_id or "unspecified"

    # Never put raw secrets into ledger summaries.
    safe_summary = privacy_runtime.sanitize_error_message(args.summary)

    entry: dict[str, Any] = {
        "entry_id": f"ll-{uuid.uuid4().hex[:12]}",
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": source_run,
        "run_id": run_id or None,
        "observation_class": args.obs_class,
        "summary": safe_summary,
        "related_profiles": profiles,
        "related_gates": [g.strip() for g in args.gates.split(",") if g.strip()],
        "evidence_refs": [],
        "canon_status": "PROPOSED",
        "authority": "advisory_only",
        "auto_apply_forbidden": True,
        "skill": "neon-genie",
        **privacy_runtime.learning_ledger_privacy_fields(privacy_ctx),
    }
    if recipe:
        entry["recipe"] = recipe
    if envelope_path:
        entry["envelope_path"] = envelope_path
    if args.routing_quality:
        entry["routing_quality"] = args.routing_quality
    if args.gate_quality:
        entry["gate_quality"] = args.gate_quality
    # Drop null run_id for cleaner JSONL when missing
    if not entry.get("run_id"):
        entry.pop("run_id", None)

    for key in (
        "entry_id",
        "observed_at",
        "source_run",
        "observation_class",
        "summary",
        "canon_status",
        "authority",
    ):
        if not entry.get(key):
            print(f"FAIL: NG-LEARN-003: missing {key}", file=sys.stderr)
            return 1

    if entry.get("canon_status") not in {"PROPOSED", "OBSERVATION"}:
        print("FAIL: NG-LEARN-004: canon_status must be PROPOSED or OBSERVATION", file=sys.stderr)
        return 1
    if entry.get("auto_apply_forbidden") is not True:
        print("FAIL: NG-LEARN-005: auto_apply_forbidden must be true", file=sys.stderr)
        return 1

    try:
        privacy_runtime.assert_no_raw_secrets(
            json.dumps(entry, sort_keys=True), context="learning-ledger-entry"
        )
    except ValueError as exc:
        print(f"FAIL: {privacy_runtime.sanitize_error_message(str(exc))}", file=sys.stderr)
        return 1

    ledger: Path = args.ledger
    if not ledger.is_absolute():
        ledger = Path.cwd() / ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")

    print("PASS: learning ledger entry appended")
    print(f"  ledger: {ledger}")
    print(f"  entry_id: {entry['entry_id']}")
    if entry.get("run_id"):
        print(f"  run_id: {entry['run_id']}")
    print(f"  class: {entry['observation_class']}")
    print(f"  privacy_mode: {entry.get('privacy_mode')}")
    print("  canon_status: PROPOSED (never auto-applied)")
    print("  learning: local operator artifact only; auto-apply forbidden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
