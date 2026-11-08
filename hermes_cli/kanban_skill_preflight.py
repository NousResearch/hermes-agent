"""Dispatch preflight: two questions, asked before a worker is spawned.

**Q1 — can the profile that will run this card resolve the skills its worker preloads?**

A managed worker dies before its first turn when *none* of its preloaded identifiers resolve
(``ValueError: kanban_skills_unavailable`` in ``gateway/session_kanban.py``). The worker now parks
the card ``blocked``/``capability`` with a reason naming the missing skills, so a stranded gate is
visible on the first attempt instead of after the breaker's last retry. This module carries the two
pieces that make that answer checkable from outside a worker:

* :func:`resolve_for_profile` — the same resolution the worker performs, run against the
  assignee's own profile home (``HERMES_HOME`` override, exactly like the gateway).
* :func:`sweep_assignees` — every assignee on a board, per lane, so an under-provisioned profile
  is found before a gate is routed to it::

      python -m hermes_cli.kanban_skill_preflight --board default

**Q2 — is this the right agent for this class of work?**

Q1 alone let the review lane be filled by whichever profile looked free. :func:`routing_refusal`
adds the policy lookup against the fleet routing rule and returns one refusal carrying both
answers, so dispatch can refuse before it claims. Deny is the default: a profile absent from the
rule cannot be routed.

The identifier list is not guessed here: :func:`hermes_cli.kanban_db.worker_skill_identifiers` is
the single definition the dispatcher and the gateway also use (task-pinned skills, plus
``sdlc-review`` on the review lane), and the refusal predicate is the gateway's own
``missing and not loaded``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterable, Optional, Sequence

_log = logging.getLogger(__name__)

PREFLIGHT_FAILURE_MARKER = "worker skill preload unsatisfiable"


def preload_failure_reason(profile: Optional[str], missing: Iterable[str]) -> str:
    """The durable reason written on a card whose preload cannot resolve.

    Names the profile and every missing identifier, so the board alone says what to install and
    who to reassign away from. One string for the worker's block and the sweep report.
    """
    names = ", ".join(str(name) for name in missing) or "(none listed)"
    return (
        f"{PREFLIGHT_FAILURE_MARKER} for profile '{profile or 'unknown'}': {names} did not resolve, "
        "so no worker turn can start. Install the skill(s) on that profile, or reassign the card "
        "to a provisioned reviewer."
    )


def resolve_for_profile(
    profile: str, identifiers: Sequence[str], *, task_id: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """``(loaded, missing)`` for *identifiers* resolved against *profile*'s own home.

    Raises when the profile cannot be resolved to a Hermes home (a missing/renamed profile is a
    different defect than an unprovisioned one); the caller decides whether to care.
    """
    from agent.skill_commands import build_preloaded_skills_prompt
    from hermes_cli.profiles import resolve_profile_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    ids = list(dict.fromkeys(str(skill).strip() for skill in identifiers if str(skill).strip()))
    if not ids:
        return [], []
    token = set_hermes_home_override(resolve_profile_env(profile))
    try:
        _prompt, loaded, missing = build_preloaded_skills_prompt(ids, task_id=task_id)
    finally:
        reset_hermes_home_override(token)
    return list(loaded), list(missing)


def unsatisfiable_preload(
    profile: Optional[str], identifiers: Sequence[str], *, task_id: Optional[str] = None,
) -> Optional[list[str]]:
    """The missing identifiers when NOTHING resolves, else ``None``.

    Mirrors the worker's own fail-closed rule (``missing and not loaded``): one resolved skill is
    enough for a turn to start, so a partial miss is a warning, never a refusal. Returns ``None``
    when the profile home cannot be resolved at all — an environment that cannot be read is not
    evidence that a preload will fail, and the worker's own guard remains the backstop.
    """
    try:
        loaded, missing = resolve_for_profile(profile or "", identifiers, task_id=task_id)
    except Exception:
        return None
    if not identifiers or loaded:
        return None
    return missing


def routing_refusal(
    profile: Optional[str],
    task_class: Optional[str],
    *,
    lane: str,
    identifiers: Sequence[str],
):
    """Both dispatch questions for one assignment: the refusal, or ``None``.

    Q1 (can it start) is :func:`unsatisfiable_preload` against *identifiers* — the exact list
    :func:`hermes_cli.kanban_db.worker_skill_identifiers` hands the worker. Q2 (is it the right
    agent) is the fleet routing rule, a policy record: see
    :mod:`hermes_cli.kanban_fleet_routing`, which denies any profile absent from that record.

    The returned object names the profile, the class, the lane, the missing identifiers, the
    decision, the rule path and the config version, so the caller can park a card with a reason
    that is actually diagnosable.

    A rule that cannot be read does not stop dispatch: an unreadable policy file is an operational
    fault, and the worker's own Q1 guard plus the next tick remain the backstop. The failure is
    logged, never swallowed silently.
    """
    from hermes_cli.kanban_fleet_routing import assignment_refusal

    try:
        return assignment_refusal(profile or "", task_class, lane=lane, identifiers=list(identifiers))
    except Exception:
        _log.warning(
            "kanban preflight: fleet routing rule unreadable; Q2 not enforced for %r on the %s lane",
            profile, lane, exc_info=True,
        )
        return None


def sweep_assignees(conn, *, lanes: Sequence[str] = ("review", "ready")) -> list[dict]:
    """Resolve every live assignee's preload, per lane, against their own profile.

    A card's pinned ``skills`` and the review lane's force-loaded reviewer skill are the two
    sources of ``context['skills']``, so both are evaluated. An assignee whose own home cannot be
    resolved is reported rather than skipped: the dispatcher would refuse to spawn for it anyway.
    """
    from hermes_cli.kanban_db import worker_skill_identifiers

    rows = conn.execute(
        "SELECT DISTINCT assignee, skills FROM tasks "
        "WHERE assignee IS NOT NULL AND assignee != '' "
        "AND status NOT IN ('done', 'archived') ORDER BY assignee"
    ).fetchall()
    per_profile: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    for row in rows:
        assignee = row["assignee"]
        raw = row["skills"]
        pinned: list[str] = []
        if raw:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                pinned = list(parsed or []) if isinstance(parsed, (list, tuple)) else []
            except (TypeError, ValueError):
                pinned = []
        for lane in lanes:
            identifiers = worker_skill_identifiers(pinned, review_lane=(lane == "review"))
            if identifiers:
                per_profile.setdefault(assignee, set()).add((lane, tuple(identifiers)))

    report: list[dict] = []
    for assignee in sorted(per_profile):
        for lane, identifiers in sorted(per_profile[assignee]):
            entry = {"profile": assignee, "lane": lane, "identifiers": list(identifiers)}
            try:
                loaded, missing = resolve_for_profile(assignee, list(identifiers))
            except Exception as exc:  # profile home unresolvable: a different, louder defect
                entry.update(state="unknown", loaded=[], missing=list(identifiers), error=str(exc))
            else:
                entry.update(
                    state="ok" if loaded else "unsatisfiable",
                    loaded=loaded, missing=missing,
                    reason=None if loaded else preload_failure_reason(assignee, missing),
                )
            report.append(entry)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.kanban_skill_preflight",
        description="Resolve the skill preload every board assignee's worker would receive.",
    )
    parser.add_argument("--board", default=None, help="Board slug (default: the active board)")
    parser.add_argument("--lane", default="both", choices=("review", "ready", "both"),
                        help="Which lane's identifier list to resolve (default: both)")
    parser.add_argument("--json", action="store_true", help="Machine-readable report")
    args = parser.parse_args(argv)

    from hermes_cli.kanban_db_connect import connect_closing

    lanes = ("review", "ready") if args.lane == "both" else (args.lane,)
    with connect_closing(board=args.board) as conn:
        report = sweep_assignees(conn, lanes=lanes)

    bad = [entry for entry in report if entry["state"] != "ok"]
    if args.json:
        print(json.dumps(report, indent=2))
    elif not report:
        print("no assignee on this board carries a skill preload")
    else:
        for entry in report:
            print(f"{entry['profile']:<22} {entry['lane']:<7} {entry['state']:<14} "
                  f"{', '.join(entry['identifiers'])}")
            if entry["state"] != "ok":
                print(f"    {entry.get('error') or entry.get('reason')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
