#!/usr/bin/env python3
"""Deterministic generator for ~/.hermes/kanban/boards-manifest.json.

Card: jarvis-os t_e7bd85a5 (P8-R7). Replaces the hand-edited manifest with a
script-derived one so future drift is a diff against this file's config, not
a silent JSON edit.

Board SET + active/alias status is derived programmatically from the
canonical portfolio registry (source of truth per the second-brain contract):
    /home/frank/obsidian-fleet-vault/Projects/Portfolio/registry.yaml
  -> projects[].execution_adapter == "hermes-kanban"
  -> projects[].board_or_tracker (PRIMARY slug), projects[].tracker_aliases (alias slugs)

registry.yaml has no concept of PM-per-board, reviewer lane, or lifecycle
flags (dispatch/gc/triage/sweep) -- those are fleet-operational policy, not
project metadata. They come from two places, cited per board below:
  1. Architecture/Jarvis-Fleet-Architecture-Canonical.md (council-ratified
     2026-08-29) -- L2 PM roster (line 116-120) and reviewer lane definitions
     (line 131-132: os-reviewer = meta/OS/fleet-safety only; platform-reviewer
     = product/platform class, absorbs upero-design-reviewer +
     yorkstone-supplies-reviewer).
  2. Prior manifest continuity, where Architecture is silent and no card has
     asked for a change (upero/sycode-trading/jarvis-os reviewers untouched
     here; changing those is out of this card's scope).

Non-registry live boards (ecohome, orchestrator-sync) are
declared explicitly below because they are NOT hermes-kanban projects in the
registry (ecohome-venture's execution_adapter is jarvis-goal, incubating) yet
demonstrably exist as live kanban boards with real owners (board.json checked
2026-08-30; Architecture line 120 names ecohome-pm). This mismatch is a
registry/board-inventory contradiction the P8 verification flagged
(Governance/2026-08-30-fleet-post-refactor-p8-verification.md, evidence #2)
and is not resolved here -- only reflected honestly.

Run: python3 generate_boards_manifest.py [--write] [--check-only]
  (no args)     print the generated manifest to stdout
  --write       write it to $HERMES_BOARDS_MANIFEST (default live path)
  --check-only  diff against the live file, exit 1 if different
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import yaml

REGISTRY_PATH = Path("/home/frank/obsidian-fleet-vault/Projects/Portfolio/registry.yaml")
MANIFEST_PATH = Path(
    os.environ.get("HERMES_BOARDS_MANIFEST", "/home/frank/.hermes/kanban/boards-manifest.json")
)


class ManifestConflictError(RuntimeError):
    """Raised when the registry cannot be resolved into one manifest deterministically."""


class DispatchFlipGuardError(RuntimeError):
    """Raised by --write when the generated manifest would flip a board's
    `dispatch` flag (either direction) vs. the live manifest, without the
    caller passing --ack-dispatch-flip. See Bug 3, kanban t_aee179af."""

DOC = (
    "Single source of truth for which kanban boards each fleet lifecycle loop touches. "
    "Loops MUST read this via scripts/fleet_boards.py instead of hardcoding board slugs. "
    "Flags: dispatch (fleet-dispatch.sh spawns workers), gc (GC/hygiene bundle), "
    "triage (PM triage / needs-input / review routing), sweep (read-only staleness & "
    "health reporting). 'owner' is the PM/triage profile responsible for the board; "
    "null means no owner and MUST be paired with state != active."
)

STATES = {
    "active": "Board is live; flags apply as written.",
    "dormant": (
        "Deliberately not worked. All lifecycle flags must be false. Requires "
        "review_date and reason — silence is intentional, not accidental."
    ),
    "denied": "Board must NEVER be dispatched or GC'd by fleet loops (infrastructure/coordination buses). All flags false, permanently.",
}

# Architecture L2 PM roster (Jarvis-Fleet-Architecture-Canonical.md:116-120).
# ai-restaurant: jarvis-os-pm owns it "until its volume justifies a PM" (line 116,
# scaling rule 3) -- no ai-restaurant-pm profile exists yet.
PM_BY_PRIMARY_SLUG = {
    "jarvis-os": "jarvis-os-pm",
    "sycode-trading": "sycode-trading-pm",
    "upero": "upero-pm",
    "yorkstone-supplies": "yorkstone-supplies-pm",
    "ai-restaurant": "jarvis-os-pm",
}

# Reviewer lane per board. Architecture line 131-132: os-reviewer = meta/OS/
# fleet-safety card class only; platform-reviewer = product/platform card
# class (explicitly absorbs yorkstone-supplies-reviewer). Where Architecture
# is silent and no card requires a change, the prior manifest value is kept
# (continuity, not re-derivation) -- upero and sycode-trading are NOT in this
# card's scope.
REVIEWER_BY_SLUG = {
    "jarvis-os": "os-reviewer",  # meta/OS/fleet-safety class (Architecture:131)
    "sycode-trading": "trading-risk-reviewer",  # unchanged, out of card scope
    "upero": "guardian",  # unchanged, out of card scope (prior manifest continuity)
    "yorkstone-supplies": "platform-reviewer",  # CORRECTED by t_e7bd85a5: Architecture:132
    # explicitly absorbs yorkstone-supplies-reviewer into platform-reviewer.
    "ai-restaurant": "platform-reviewer",  # product/platform class, same lane as ecohome
    "ecohome": "platform-reviewer",  # product/platform class (Architecture:132)
}

# Dispatch quiesce overrides (Bug 2, kanban t_aee179af): boards that stay
# `state: active` (owner/reviewer/gc/triage/sweep unchanged) but have
# `dispatch` forced off by explicit Frank-level policy, independent of the
# registry's `state` field. This is DATA, not a permanent code special-case:
# adding/removing a quiesce is a one-line edit here, reviewed like any other
# change, and every entry must cite the decision it encodes -- never blank.
#
# Current entries mirror the live manifest's `_manual_override` block
# (2026-09-12, jarvis: "Frank: only yorkstone-supplies worked; all other
# boards dispatch-gated off. Regeneration from registry.yaml sets
# dispatch=True for every active board -- re-apply after any
# generate_boards_manifest.py run."). This map makes that override
# regeneration-safe instead of a load-bearing comment that silently drops
# the next time someone runs `--write`. Un-quiescing a board is done by
# deleting its entry here, not by --write overwriting it.
DISPATCH_QUIESCE: dict[str, str] = {
    "jarvis-os": "Frank token-cost quiesce 2026-09-12: only yorkstone-supplies actively worked.",
    "sycode-trading": "Frank token-cost quiesce 2026-09-12: only yorkstone-supplies actively worked.",
    "upero": "Frank token-cost quiesce 2026-09-12: only yorkstone-supplies actively worked.",
    "ai-restaurant": "Frank token-cost quiesce 2026-09-12: only yorkstone-supplies actively worked.",
    "ecohome": "Frank token-cost quiesce 2026-09-12: only yorkstone-supplies actively worked.",
}

# Live kanban boards that are NOT hermes-kanban projects in the registry (see
# module docstring). Declared explicitly, not derived.
NON_REGISTRY_BOARDS = {
    "default": {
        "state": "dormant",
        "owner": None,
        "dispatch": False,
        "gc": False,
        "triage": False,
        "sweep": True,
        "reason": (
            "CORRECTION (initial pass wrongly checked boards/default/kanban.db, "
            "which IS an empty 0-task fixture): the 'default' slug actually "
            "resolves to the ROOT ~/.hermes/kanban.db (board name 'LiveFixture', "
            "no manifest-consuming code distinguishes the two paths). That root "
            "DB has REAL open work -- 6 todo + 7 blocked, all sycode-trading "
            "calibration/quant-gate incident cards (assignees "
            "trading-strategy-dev, fleet-analyst, sycode-trading-pm, "
            "trading-devops, db-architect, jarvis), stalled since 2026-08-15 "
            "with a P1-labelled card ('quant gate paralysis'). This is NOT safe "
            "to permanently deny -- it looks like orphaned, un-dispatched "
            "trading-incident work sitting silent for 3.5+ weeks because no "
            "manifest entry ever routed it to a dispatcher. Dormant (not "
            "denied) pending urgent sycode-trading-pm triage: escalated as "
            "child task t_cc79ea25 -> see that card for disposition (migrate "
            "into sycode-trading board vs. reactivate this board as active)."
        ),
        "review_date": "2026-09-11",
        "declared_by": "os-architect/t_cc79ea25",
    },
    "quicknote": {
        "state": "dormant",
        "owner": None,
        "dispatch": False,
        "gc": False,
        "triage": False,
        "sweep": True,
        "reason": (
            "registry project 'quicknote' execution_adapter='direct-git', not "
            "'hermes-kanban' -- canonical tracker is git; this kanban board is "
            "historical only (14 archived + 1 done, last real work 2026-07-11; "
            "recent adds are validation/reflection probe cards). Registry "
            "next_bottleneck: 'decide whether QuickNote is maintained or "
            "retired' -- revisit at that decision, not sooner."
        ),
        "review_date": "2026-09-01",
        "declared_by": "os-architect/t_cc79ea25",
    },
    # trading-data-oracle and zz-metaflag-test-1789066340 were deleted 2026-09-10
    # by fleet-engineer/t_d9698b71 (re-verified 0 open work immediately before
    # deletion; board dirs moved to boards/_archived/). Entries removed here so
    # the manifest doesn't declare lifecycle policy for boards that no longer
    # exist -- fleet_boards.py --check would otherwise be silently unaffected
    # (it only flags UNDECLARED boards with open work), but keeping dead-board
    # entries around is exactly the drift this generator exists to prevent.
    "orchestrator-sync": {
        "state": "denied",
        "owner": None,
        "dispatch": False,
        "gc": False,
        "triage": False,
        "sweep": False,
        "reason": (
            "Coordination bus, not a work board. Dispatching it spawns phantom "
            "workers on protocol cards. Permanent deny — do not flip to active."
        ),
        "declared_by": "devops/t_911a916c",
    },
    # legacy-yss: REMOVED 2026-09-13 (jarvis). The `boards/legacy-yss/` dir is an empty
    # shell (0 tasks; the 118KB file is freed pages) and the canonical portfolio registry
    # records the legacy-yss PROJECT with board_or_tracker="yorkstone-supplies"
    # (verified 2026-09-11 in Projects/Portfolio/registry.yaml) -- i.e. there is no
    # legacy-yss BOARD. Declaring lifecycle policy for a board that does not exist is
    # exactly the drift this generator exists to prevent (same treatment as
    # trading-data-oracle / zz-metaflag-test, removed 2026-09-10).
    "ecohome": {
        "state": "active",
        "owner": "ecohome-pm",
        "reviewer": "platform-reviewer",
        "dispatch": True,
        "gc": True,
        "triage": True,
        "sweep": True,
    },
}

_KEY_ORDER = (
    "state", "owner", "reviewer", "alias_of", "dispatch", "gc", "triage",
    "sweep", "reason", "review_date", "declared_by", "quiesce_reason",
)


def _reorder(entry: dict) -> dict:
    return {k: entry[k] for k in _KEY_ORDER if k in entry}


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def hermes_kanban_projects(registry: dict) -> list[dict]:
    return [p for p in registry["projects"] if p.get("execution_adapter") == "hermes-kanban"]


def build_boards(registry: dict) -> dict:
    """Build the {slug: entry} board map, deterministic under project order.

    Tie-break (Bug 1, kanban t_aee179af): when two+ registry projects declare
    the same PRIMARY board_or_tracker (e.g. 'yorkstone-supplies' active +
    'legacy-yss' maintenance both pointing at the yorkstone-supplies board),
    the ACTIVE project always wins the slot. A non-active project must never
    downgrade an already-active board -- so this groups primary-slug entries
    BEFORE writing to `boards`, rather than clobbering in a single pass.
    Multiple ACTIVE projects sharing one slug is treated as a genuine
    registry conflict and raises loudly (silently picking one would hide a
    real data-modelling problem the registry needs to resolve).
    """
    boards: dict[str, dict] = {}
    projects = hermes_kanban_projects(registry)

    # Group by primary slug, preserving registry order for determinism.
    by_primary: dict[str, list[dict]] = {}
    for proj in projects:
        by_primary.setdefault(proj["board_or_tracker"], []).append(proj)

    for primary, projs in by_primary.items():
        active_projs = [p for p in projs if p.get("state") == "active"]
        is_active = bool(active_projs)
        if is_active:
            # Multiple ACTIVE projects legitimately sharing one board slug is
            # real fleet data (e.g. 'grok-quant-trader' is a specialist seat
            # deliberately working the 'sycode-trading' board alongside the
            # primary sycode-trading project) -- not the dual-slug bug. The
            # entry is fully determined by primary-slug lookup tables
            # (PM_BY_PRIMARY_SLUG / REVIEWER_BY_SLUG), so it's identical no
            # matter which active project "wins". Only raise when the owner
            # is NOT resolvable from those tables AND the active projects
            # disagree on primary_owner -- a genuine unresolved conflict.
            owner = PM_BY_PRIMARY_SLUG.get(primary)
            if owner is None:
                primary_owners = {p.get("primary_owner") for p in active_projs}
                if len(primary_owners) > 1:
                    ids = ", ".join(p["id"] for p in active_projs)
                    raise ManifestConflictError(
                        f"board slug {primary!r} is claimed by {len(active_projs)} ACTIVE "
                        f"registry projects ({ids}) with disagreeing primary_owner and no "
                        "PM_BY_PRIMARY_SLUG override -- registry.yaml or "
                        "PM_BY_PRIMARY_SLUG must resolve this before the manifest can be "
                        "generated; the generator refuses to silently pick a winner."
                    )
                owner = next(iter(primary_owners))
            entry = {
                "state": "active",
                "owner": owner,
                "reviewer": REVIEWER_BY_SLUG.get(primary),
                "dispatch": True,
                "gc": True,
                "triage": True,
                "sweep": True,
            }
            if primary in DISPATCH_QUIESCE:
                entry["dispatch"] = False
                entry["quiesce_reason"] = DISPATCH_QUIESCE[primary]
        else:
            winner = projs[0]
            entry = {
                "state": "dormant",
                "owner": None,
                "dispatch": False,
                "gc": False,
                "triage": False,
                "sweep": True,
                "reason": f"registry project '{winner['id']}' state={winner.get('state')!r}, not active.",
                "review_date": _default_review_date(),
            }
        boards[primary] = entry

        # Aliases: union across every project sharing this primary slug, not
        # just the winner's -- a losing project's aliases are still real
        # slugs that must not double-dispatch the same work.
        seen_aliases: set[str] = set()
        for proj in projs:
            for alias in proj.get("tracker_aliases") or []:
                if alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                boards[alias] = {
                    "state": "dormant",
                    "owner": None,
                    "alias_of": primary,
                    "dispatch": False,
                    "gc": False,
                    "triage": False,
                    "sweep": True,
                    "reason": (
                        f"Alias of '{primary}' per registry project '{proj['id']}'.tracker_aliases; "
                        "not a live independent board. Dormant so lifecycle loops do not "
                        "double-dispatch the same work under two slugs."
                    ),
                    "review_date": _default_review_date(),
                }

    for slug, entry in NON_REGISTRY_BOARDS.items():
        entry = dict(entry)
        if entry.get("reviewer") is None and slug in REVIEWER_BY_SLUG:
            entry["reviewer"] = REVIEWER_BY_SLUG[slug]
        if entry.get("state") == "active" and slug in DISPATCH_QUIESCE:
            entry["dispatch"] = False
            entry["quiesce_reason"] = DISPATCH_QUIESCE[slug]
        boards[slug] = entry

    return {slug: _reorder(entry) for slug, entry in boards.items()}


def _default_review_date() -> str:
    """Stable governance input, changed only by an explicit manifest review.

    This must not depend on the wall clock: unchanged registry/policy inputs
    must generate byte-equivalent board semantics on every calendar day.
    """
    return "2026-09-01"


# Deterministic key ordering: registry hermes-kanban PRIMARY boards first (in
# registry.yaml project order), then their aliases, then non-registry boards.
def ordered_boards(registry: dict, boards: dict) -> dict:
    order: list[str] = []
    for proj in hermes_kanban_projects(registry):
        primary = proj["board_or_tracker"]
        if primary not in order:
            order.append(primary)
        for alias in proj.get("tracker_aliases") or []:
            if alias not in order:
                order.append(alias)
    for slug in NON_REGISTRY_BOARDS:
        if slug not in order:
            order.append(slug)
    for slug in boards:  # safety net for anything unexpected
        if slug not in order:
            order.append(slug)
    return {slug: boards[slug] for slug in order}


def generate() -> dict:
    registry = load_registry()
    boards = build_boards(registry)
    boards = ordered_boards(registry, boards)
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "version": 1,
        "updated": now.date().isoformat(),
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/generate_boards_manifest.py",
        "generated_from": str(REGISTRY_PATH),
        "doc": DOC,
        "states": STATES,
        "boards": boards,
    }


def dispatch_flips(new_boards: dict, old_boards: dict) -> list[tuple[str, bool | None, bool]]:
    """Boards whose `dispatch` flag differs between old (live) and new
    (generated) manifests. A board present only in `new_boards` counts as a
    flip from None (no prior policy) if it comes up dispatch=True -- a brand
    new board must not silently start dispatching either.
    """
    flips = []
    slugs = set(new_boards) | set(old_boards)
    for slug in sorted(slugs):
        old_val = old_boards.get(slug, {}).get("dispatch")
        new_val = new_boards.get(slug, {}).get("dispatch", False)
        if old_val != new_val:
            flips.append((slug, old_val, new_val))
    return flips


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write to the live manifest path")
    ap.add_argument("--check-only", action="store_true", help="diff against live file; rc=1 if different")
    ap.add_argument("--manifest", default=None, help="override manifest path")
    ap.add_argument(
        "--ack-dispatch-flip",
        action="store_true",
        help=(
            "required alongside --write when the generated manifest would flip any "
            "board's dispatch flag vs. the on-disk file (Bug 3 guard, t_aee179af); "
            "absent this flag --write refuses rather than silently re-enabling or "
            "disabling a lifecycle loop"
        ),
    )
    args = ap.parse_args(argv)
    path = Path(args.manifest) if args.manifest else MANIFEST_PATH

    try:
        manifest = generate()
    except ManifestConflictError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if args.check_only:
        live = path.read_text(encoding="utf-8") if path.exists() else ""
        # ignore volatile timestamp fields for the diff
        import re

        def strip_volatile(s: str) -> str:
            s = re.sub(r'"updated": "[^"]*"', '"updated": ""', s)
            s = re.sub(r'"generated_at": "[^"]*"', '"generated_at": ""', s)
            return s

        if strip_volatile(text) != strip_volatile(live):
            print(f"DIFFERS from {path}")
            return 1
        print(f"MATCHES {path} (ignoring updated/generated_at)")
        return 0

    if args.write:
        old_boards: dict = {}
        if path.exists():
            try:
                old_boards = json.loads(path.read_text(encoding="utf-8")).get("boards", {})
            except json.JSONDecodeError as exc:
                print(f"ERROR cannot parse existing manifest {path}: {exc}", file=sys.stderr)
                return 2
        flips = dispatch_flips(manifest["boards"], old_boards)
        if flips and not args.ack_dispatch_flip:
            print(
                f"REFUSED: {len(flips)} board(s) would flip 'dispatch' vs {path}. "
                "Re-run with --ack-dispatch-flip once you have confirmed every flip "
                "below is intended (a quiesced board must stay in DISPATCH_QUIESCE, "
                "a dual-slug conflict must resolve to the active project):",
                file=sys.stderr,
            )
            for slug, old_val, new_val in flips:
                print(f"  {slug}: dispatch {old_val!r} -> {new_val!r}", file=sys.stderr)
            return 3
        if flips:
            print(f"ACKNOWLEDGED {len(flips)} dispatch flip(s):")
            for slug, old_val, new_val in flips:
                print(f"  {slug}: dispatch {old_val!r} -> {new_val!r}")
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
