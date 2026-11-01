"""``hermes skills doctor`` — registry↔disk reconciliation for the skills usage registry.

Ports the ORCA-06 host check (~/.hermes/scripts/skills-health-check.py) as a first-class
command. Exits non-zero when a CRITICAL desync exists — an active/stale registry entry
that is pinned or was used/viewed within the lookback window resolves to no on-disk
skill dir and no ``.archive`` copy — or when the registry itself is missing/unreadable
(nothing can be reconciled). The int return follows the hermes CLI convention:
``hermes_cli/main.py`` turns a handler's non-zero int into the process exit code.
"""

from __future__ import annotations

import json
from typing import Optional

from tools.skills_health import default_skills_root, doctor_report


def skills_doctor_command(args) -> int:
    """Reconcile ``skills/.usage.json`` against on-disk skill dirs; 1 on critical desync."""
    root = default_skills_root()
    report = doctor_report(root)
    usage_path = f"{root / '.usage.json'}"

    if getattr(args, "json", False):
        print(json.dumps(report, indent=1))
    elif not report["registry_present"]:
        print(f"skills doctor: NO REGISTRY ({usage_path}) — cannot reconcile")
    elif report.get("registry_error"):
        print(f"skills doctor: REGISTRY UNREADABLE ({usage_path}) — {report['registry_error']}")
    elif report["critical"]:
        crit = report["critical"]
        print(f"skills doctor: CRITICAL — {len(crit)} active/pinned registry "
              f"entr{'y' if len(crit) == 1 else 'ies'} with NO on-disk skill dir (registry/disk desync):")
        for c in crit:
            print(f"  - {c['key']}  state={c['state']} pinned={c['pinned']} "
                  f"use_count={c['use_count']} last_used={c['last_used_at']}")
        print("  (registry keys are name-based; hub-viewed-only and archived entries are excluded)")
    else:
        print(f"skills doctor: OK — {report['disk_skill_dirs']} disk skill dirs reconciled "
              f"against {report['registry_entries']} registry entries; 0 critical desyncs.")

    if report["info_unregistered_dirs"] and not getattr(args, "json", False):
        print(f"  info: {len(report['info_unregistered_dirs'])} on-disk skill dirs not yet in "
              f"registry (normal for freshly created skills)")

    failed = bool(report["critical"]) or not report["registry_present"] or bool(report.get("registry_error"))
    return 1 if failed else 0
