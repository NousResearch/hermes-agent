"""``hermes gateway preflight``: the boot-mode verdict for a host that is NOT booting yet.

``resolve_multiplex_mode`` answers "what mode did this boot settle into" — and it only runs inside
the gateway process, after which the verdict is persisted to runtime status. An operator
provisioning boxes could not ask "what mode WILL this box boot in" before starting it, so two
identical-looking boxes ended up in different modes.

This verb runs the SAME decision (:func:`gateway_multiplex_mode.compute_multiplex_decision`, pure)
plus the SAME blocker list the migration preflight builds — every blocker names its remedy, and the
convergence path is ``hermes gateway migrate --multiplex``. It writes NOTHING: no config change, no
runtime-status record.

Exit codes: 0 when the host will multiplex (or is single-profile, nothing to multiplex), 1 when it
will come up standalone-by-blocker, so provisioning scripts can gate on it.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

STANDALONE_MODE_LINE = "mode: standalone (default only)"


def _served_roster(decision, plan) -> list[str]:
    """The profiles the boot will serve: every profile when multiplexing, the default only when a
    blocker (or a single-profile install) keeps the host standalone."""
    if decision.enabled:
        return [p.name for p in plan.profiles]
    return ["default"]


def _blocker_texts(decision, plan) -> list[str]:
    """Every blocker one per line, deduplicated, order preserved. Reuses the migration preflight's
    own lists (plan.blockers + the auto-migration boundary guards) so this verb never hand-writes a
    second list that can drift from what the multiplexer actually refuses at boot."""
    from hermes_cli.gateway_migrate_guards import auto_migration_blockers
    from hermes_cli.gateway_multiplex_mode import SINGLE_PROFILE_REASON
    if decision.enabled or decision.reason == SINGLE_PROFILE_REASON:
        return []  # nothing to multiplex is not a blocker: there is no fold to converge on
    texts = [decision.reason, *plan.blockers, *auto_migration_blockers(plan)]
    return list(dict.fromkeys(t for t in texts if t))


def _blocker_records(texts: list[str]) -> list[dict]:
    # Each blocker's remedy is named inside its own text; ``fix`` carries the convergence command
    # that finishes the fold once the specific remedy has been applied.
    return [{"reason": t, "fix": "run `hermes gateway migrate --multiplex` once fixed"} for t in texts]


def preflight_report() -> dict:
    """The full preflight report: mode, reason, every blocker, and the served roster. Pure."""
    from gateway.config import load_gateway_config
    from hermes_cli.gateway_migrate import build_migration_plan
    from hermes_cli.gateway_multiplex_mode import (
        SINGLE_PROFILE_REASON,
        compute_multiplex_decision,
    )
    decision = compute_multiplex_decision(load_gateway_config())
    plan = build_migration_plan()
    if decision.enabled:
        mode_name = "multiplex"
    elif decision.reason == SINGLE_PROFILE_REASON:
        mode_name = "single"
    else:
        mode_name = "standalone"
    reason = None if decision.enabled else decision.reason
    return {
        "mode": mode_name,
        "reason": reason,
        "blockers": _blocker_records(_blocker_texts(decision, plan)),
        "profiles": _served_roster(decision, plan),
    }


def _text_lines(report: dict) -> list[str]:
    if report["mode"] == "multiplex":
        head = "mode: multiplex"
    elif report["mode"] == "single":
        head = "mode: single"
    else:
        head = STANDALONE_MODE_LINE
    lines = [head]
    if report["mode"] != "multiplex" and report["reason"]:
        lines.append(f"reason: {report['reason']}")
    lines.extend(f"blocker: {b['reason']}" for b in report["blockers"])
    lines.append(f"profiles: {', '.join(report['profiles'])}")
    return lines


def cmd_preflight(args) -> Optional[int]:
    """``hermes gateway preflight [--json]``: exit 0 when the host will multiplex (or has nothing to
    multiplex), 1 when a blocker keeps it standalone."""
    report = preflight_report()
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print("\n".join(_text_lines(report)))
    sys.exit(0 if report["mode"] != "standalone" else 1)
