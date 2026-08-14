#!/usr/bin/env python3
"""Denji canary-observe loop (closes the P2-3 self-modification loop).

For each profile edit Denji applied (recorded as a canary by the
blast-radius guard), this driver:

  1. Runs the P2-2 eval harness against the edited profile.
  2. Feeds the pass-rate to ``EditGuard.observe_canary`` which, after
     ``canary_observe_runs`` observations, promotes a healthy edit or
     marks a regressed one REVERTED.
  3. Git-reverts a REVERTED canary via ``profile_editor.py --rollback``.
  4. Refreshes ``fleet_health.json`` via ``check_fleet_health`` so the
     blast-radius tripwire reflects current aggregate quality (a fleet-wide
     drop pauses ALL further Denji edits until an operator resumes).

Cadence: intended as a base-store cron (every few hours). Each tick
observes a canary once; a decision lands after ``canary_observe_runs``
(default 3) ticks. The per-tick fleet-health tripwire is the immediate
aggregate backstop while individual canaries are still under observation.

Only profiles with a golden task set (mapped via
``governance.eval_domains`` in config.yaml, resolved to
``~/.hermes/kensei/eval/golden-tasks-<domain>-*.yaml``) are evaluated;
unmapped profiles are left for manual review and logged, never
auto-promoted blind.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Resolve hermes_cli from the canonical repo checkout(s). The deployed copy
# under ~/.hermes/scripts/ has no hermes_cli on sys.path, so bootstrap it from
# the repo that owns this script (or the canonical KenseiAgent checkout).
for _root in (
    Path(__file__).resolve().parents[1],  # repo checkout layout
    Path.home() / "repos" / "KenseiAgent",  # canonical repo
):
    if (_root / "hermes_cli").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("denji-canary-observe")

# P13 isolation: HERMES_HOME parameterises the hermes root.
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
EVAL_DIR = os.path.join(_HERMES_HOME, "kensei/eval")
PROFILE_EDITOR = os.path.join(_HERMES_HOME, "scripts/profile_editor.py")

# P13 dry-run: --dry-run prevents git operations. The canary observation
# loop (eval + observe_canary) still runs so pass-rate is computed, but
# no REVERTED canary is git-reverted (no subprocess call to
# profile_editor.py --rollback) and no fleet_health mutation.
_DRY_RUN = "--dry-run" in sys.argv


def _load_eval_domains() -> dict:
    """profile -> domain map (shared single source with the edit gate)."""
    from hermes_cli.blast_radius import load_eval_domains
    return load_eval_domains()


def _golden_set_for_domain(domain: str) -> str | None:
    """Newest golden-tasks-<domain>-*.yaml for a domain, or None."""
    matches = sorted(glob.glob(os.path.join(EVAL_DIR, f"golden-tasks-{domain}-*.yaml")))
    return matches[-1] if matches else None


def _revert_commit(commit: str) -> bool:
    """Revert an edit's commit via profile_editor.py --rollback."""
    if _DRY_RUN:
        logger.info("DRY-RUN: would revert commit %s (git operation skipped)", commit)
        return False  # dry-run: never run the git rollback
    try:
        r = subprocess.run(
            [sys.executable, PROFILE_EDITOR, "--rollback", commit],
            capture_output=True, text=True, timeout=60,
        )
        ok = r.returncode == 0 and json.loads(r.stdout or "{}").get("ok")
        if not ok:
            logger.error("rollback of %s failed: %s", commit, r.stdout or r.stderr)
        return bool(ok)
    except Exception as exc:
        logger.error("rollback of %s errored: %s", commit, exc)
        return False


def main() -> int:
    from hermes_cli.blast_radius import EditGuard, CanaryStage
    from hermes_cli.eval_harness import GoldenTask, run_eval

    guard = EditGuard()
    canaries = guard.active_canaries()
    if not canaries:
        logger.info("no active canaries; nothing to observe")
        # No eval ran this tick, so leave the fleet-health tripwire as-is.
        return 0

    domains = _load_eval_domains()
    pass_rates: list[float] = []
    regressed_profiles: list[str] = []

    for canary in canaries:
        domain = domains.get(canary.profile)
        if not domain:
            logger.warning(
                "canary %s: profile %s has no governance.eval_domains mapping; "
                "skipping (manual review)", canary.edit_id, canary.profile,
            )
            continue
        golden = _golden_set_for_domain(domain)
        if not golden:
            logger.warning(
                "canary %s: no golden set for domain %s; skipping",
                canary.edit_id, domain,
            )
            continue
        try:
            tasks = GoldenTask.load_set(golden)
        except Exception as exc:
            logger.error("could not load golden set %s: %s", golden, exc)
            continue
        if not tasks:
            # An empty/corrupt set would make run_eval report 0.0 and
            # trigger a FALSE revert of a healthy profile. Skip instead.
            logger.warning(
                "golden set %s has no tasks; skipping canary %s",
                golden, canary.edit_id,
            )
            continue
        try:
            eval_run = run_eval(tasks, profile=canary.profile)
        except Exception as exc:
            logger.error("eval failed for %s: %s", canary.profile, exc)
            continue

        pass_rates.append(eval_run.pass_rate)
        updated = guard.observe_canary(canary.edit_id, eval_run)
        if updated is None:
            continue
        logger.info(
            "canary %s (%s): stage=%s pass_rate=%.2f",
            canary.edit_id, canary.profile, updated.stage.value, eval_run.pass_rate,
        )
        if updated.stage == CanaryStage.REVERTED:
            regressed_profiles.append(canary.profile)
            if updated.commit:
                if _revert_commit(updated.commit):
                    logger.info(
                        "reverted regressed canary %s (commit %s)",
                        updated.edit_id, updated.commit[:12],
                    )
            else:
                logger.error(
                    "canary %s marked REVERTED but has no commit to roll back",
                    updated.edit_id,
                )

    # Refresh the fleet-health tripwire from this tick's evals.
    # Under --dry-run we skip this: check_fleet_health persists the
    # tripwire state (a write), so a disposable run must not call it.
    if pass_rates and not _DRY_RUN:
        mean_pass = sum(pass_rates) / len(pass_rates)
        health = guard.check_fleet_health(
            eval_pass_rate=mean_pass,
            active_profiles=len(pass_rates),
            profiles_with_regressions=regressed_profiles,
        )
        logger.info(
            "fleet health: eval_pass_rate=%.2f tripwire=%s",
            mean_pass, health.tripwire.value,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
