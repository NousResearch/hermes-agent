"""``hermes sessions repair-profiles`` — the CLI face of :mod:`hermes_cli.sessions_repair_profiles`.

Runs pre-DB (``_PRE_DB_HANDLERS``): it opens every profile's store itself rather than the one
ambient ``SessionDB()``. Report-only unless ``--apply``; ``--json`` for automation.
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

from hermes_cli.sessions_repair_profiles import (
    Finding, default_snapshot, enumerate_stores, live_gateway_homes, scan_stores,
)

_KIND_LABELS = {
    "mislabelled": "profile_name disagrees with the row's own session key",
    "wrong_store": "rows in another profile's store",
    "legacy_main": "legacy agent:main rows inside a named profile's store",
    "unclaimed_namespace": "rows keyed to a profile that does not exist",
    "crossed_parent": "parent_session_id crossing profile namespaces",
    "routing_stray": "routing rows outside the default store",
    "routing_unclaimed": "routing rows for a profile that does not exist",
    "topic_profile_less": "Telegram topic bindings without their bot's profile",
    "voice_profile_less": "voice-mode entries without their bot's profile",
    "sessions_json_unclaimed": "sessions.json entries for a profile that does not exist",
}


def _group(findings: List[Finding]) -> Dict[str, List[Finding]]:
    grouped: Dict[str, List[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.kind, []).append(finding)
    return grouped


def _print_report(findings: List[Finding]) -> None:
    for kind, rows in _group(findings).items():
        print(f"\n{_KIND_LABELS.get(kind, kind)} ({len(rows)}):")
        for finding in rows:
            print(f"  [{finding.store}] {finding.subject}\n      {finding.detail}")
            if finding.action:
                print(f"      → {finding.action}")
            else:
                print(f"      ✗ not repaired — {finding.reason}")


def cmd_repair_profiles(args) -> int:
    from state_store_maintenance import require_state_store_maintenance

    require_state_store_maintenance("sessions-repair-profiles")
    stores = enumerate_stores()
    apply = bool(getattr(args, "apply", False))
    as_json = bool(getattr(args, "json", False))
    legacy_main = getattr(args, "legacy_main", None) or "report"

    if apply:
        # Before any store is opened for write: a writable open runs schema init, and a live
        # gateway holds the routing index in memory and would write it back over the repair.
        live = live_gateway_homes(stores)
        if live:
            names = ", ".join(f"{profile} (pid {pid})" for profile, pid in live)
            print(f"A gateway is running for: {names}. It holds the routing index in memory and would "
                  "write it back over this repair. Stop it (`hermes gateway stop`), then re-run --apply.",
                  file=sys.stderr)
            return 1

    plan, session = scan_stores(stores, legacy_main=legacy_main, read_only=not apply)
    try:
        findings = plan.findings
        repairable = plan.repairable
        if as_json and not apply:
            print(json.dumps({"stores": [s.profile for s in stores], "findings": [f.as_dict() for f in findings],
                              "repairable": len(repairable)}, indent=2))
            return 0
        if not as_json:
            print(f"Scanned {len(stores)} profile store(s): {', '.join(s.profile for s in stores)}")
            if not findings:
                print("✓ No crossed-profile state found.")
                return 0
            _print_report(findings)
        if not apply:
            print(f"\n{len(repairable)} of {len(findings)} finding(s) can be repaired. "
                  "Re-run with --apply to perform them.")
            return 0
        if not repairable:
            if as_json:
                print(json.dumps({"findings": [f.as_dict() for f in findings], "applied": {}}, indent=2))
            else:
                print("\nNothing to repair.")
            return 0

        if not getattr(args, "yes", False) and not as_json:
            from hermes_cli.sessions_cmd import _confirm_prompt
            if not _confirm_prompt(f"\nApply {len(repairable)} repair(s)? A snapshot of every store is taken first. [y/N] "):
                print("Aborted — nothing was changed.")
                return 0
        result = plan.apply(session, snapshot=default_snapshot)
    finally:
        session.close()

    if as_json:
        print(json.dumps({"findings": [f.as_dict() for f in findings], "applied": result}, indent=2))
    else:
        for profile, snap in result["snapshots"].items():
            print(f"  snapshot [{profile}]: {snap or 'FAILED'}")
        applied = ", ".join(f"{k}={v}" for k, v in sorted(result["totals"].items()) if v)
        print(f"\nApplied: {applied or 'nothing'}")
        for failure in result["failures"]:
            print(f"  ✗ {failure['kind']} {failure['subject']}: {failure['error']}")
        print("Re-run without --apply to confirm the stores are clean.")
    return 1 if result["failures"] else 0
