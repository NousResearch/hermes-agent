#!/usr/bin/env python3
"""Generate the initial kanban lifecycle registry (P0-G-B1 migration).

One-time migration tool. Reads the already-gathered, ground-truth board
inventory (``board-inventory.json`` from the P0-G-B1 evidence bundle) and
deterministically produces the initial ``boards.json`` lifecycle registry:

    POSSIBLE_PRODUCTION  -> LEGACY_ACTIVE  (purpose=production)
    VALIDATION_EVIDENCE  -> INACTIVE       (purpose=validation)
    DISPOSABLE_FIXTURE   -> INACTIVE       (purpose=test)
    FORENSIC_KEEP        -> QUARANTINED    (purpose=forensic)
    CORRUPT              -> QUARANTINED    (purpose=forensic, reason="integrity failure")

Hard validation gate: the output MUST have exactly counts matching the
inventory's own classification tally (in this rollout: 43 LEGACY_ACTIVE +
21 INACTIVE + 2 QUARANTINED = 66 total), and every inventory slug must
appear exactly once, with no extra slugs. Any mismatch aborts with a
non-zero exit and NO output file is written — this script must never
"force" a workaround for a counting/classification discrepancy.

Zero board-DB writes: this script only (a) reads ``board-inventory.json``
(a plain JSON evidence file, not a board DB) and (b) optionally re-hashes
each board's ``kanban.db`` file via plain read-only file I/O (never a
read-write SQLite connection, never any SQLite connection at all) to cross
-check the fingerprint already recorded in the inventory. It writes only
the single external registry file — never anything under a board
directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli import kanban_lifecycle as lifecycle  # noqa: E402

_CLASS_TO_STATE = {
    "POSSIBLE_PRODUCTION": ("LEGACY_ACTIVE", "production", ""),
    "VALIDATION_EVIDENCE": ("INACTIVE", "validation", ""),
    "DISPOSABLE_FIXTURE": ("INACTIVE", "test", ""),
    "FORENSIC_KEEP": ("QUARANTINED", "forensic", ""),
    "CORRUPT": ("QUARANTINED", "forensic", "integrity failure"),
}

EXPECTED_COUNTS = {
    "LEGACY_ACTIVE": 43,
    "INACTIVE": 21,
    "QUARANTINED": 2,
}
EXPECTED_TOTAL = 66


class MigrationValidationError(Exception):
    """Raised when the generated registry does not match the inventory
    exactly. Callers must treat this as BLOCKING — do not force a
    workaround, stop and report."""


def _load_inventory(inventory_path: Path) -> dict:
    with open(inventory_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_registry_boards(
    inventory: dict, *, rehash: bool = True, strict_hash_match: bool = False,
) -> dict:
    """Build the ``boards`` mapping deterministically from the inventory.

    ``rehash``: recompute each board's fingerprint via read-only file
    hashing, rather than trusting the inventory's recorded ``db_sha256``
    verbatim. Cross-checked against the inventory's own value; a mismatch
    is logged (and, if ``strict_hash_match``, raises) rather than silently
    substituted — the inventory is ground truth for classification, and a
    hash drift between inventory-gathering time and migration time (e.g. a
    real board received new work in between) is expected and not itself an
    error for POSSIBLE_PRODUCTION boards.
    """
    boards: dict[str, dict] = {}
    mismatches: list[str] = []
    for slug, entry in inventory.items():
        classification = entry.get("classification")
        if classification not in _CLASS_TO_STATE:
            raise MigrationValidationError(
                f"board {slug!r} has unknown classification {classification!r}"
            )
        state, purpose, reason = _CLASS_TO_STATE[classification]
        recorded_hash = entry.get("db_sha256", "")
        fingerprint = f"sha256:{recorded_hash}" if recorded_hash else ""
        if rehash:
            db_path = Path(entry["db_path"])
            if db_path.exists():
                try:
                    computed = lifecycle.compute_db_fingerprint(db_path)
                except lifecycle.LifecycleRegistryError as exc:
                    mismatches.append(f"{slug}: cannot rehash ({exc})")
                    computed = fingerprint
                computed_hex = computed.split(":", 1)[-1]
                if recorded_hash and computed_hex != recorded_hash:
                    mismatches.append(
                        f"{slug}: inventory sha256={recorded_hash} != "
                        f"recomputed sha256={computed_hex}"
                    )
                fingerprint = computed
        boards[slug] = {
            "state": state,
            "purpose": purpose,
            "actor": "migration",
            "reason": reason or f"migration: classification={classification}",
            "updated_at": lifecycle._now_iso(),
            "db_fingerprint": fingerprint,
        }
    if mismatches and strict_hash_match:
        raise MigrationValidationError(
            "fingerprint mismatches during migration (strict mode): "
            + "; ".join(mismatches)
        )
    if mismatches:
        for m in mismatches:
            print(f"WARNING: {m}", file=sys.stderr)
    return boards


def validate_counts(boards: dict, inventory: dict) -> None:
    """Hard validation gate. Raises MigrationValidationError on ANY mismatch."""
    inventory_slugs = set(inventory.keys())
    boards_slugs = set(boards.keys())
    missing = inventory_slugs - boards_slugs
    extra = boards_slugs - inventory_slugs
    if missing or extra:
        raise MigrationValidationError(
            f"slug set mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    if len(boards) != len(inventory):
        raise MigrationValidationError(
            f"duplicate slug detected: {len(boards)} unique boards from "
            f"{len(inventory)} inventory entries"
        )
    counts = {"LEGACY_ACTIVE": 0, "ACTIVE": 0, "INACTIVE": 0, "QUARANTINED": 0, "ARCHIVED": 0}
    for entry in boards.values():
        counts[entry["state"]] += 1
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise MigrationValidationError(
            f"total board count {total} != expected {EXPECTED_TOTAL}"
        )
    for state, expected in EXPECTED_COUNTS.items():
        if counts[state] != expected:
            raise MigrationValidationError(
                f"count mismatch for {state}: got {counts[state]}, expected {expected} "
                f"(full counts: {counts})"
            )


def generate(
    inventory_path: Path, *, rehash: bool = True, strict_hash_match: bool = False,
) -> dict:
    inventory = _load_inventory(inventory_path)
    boards = build_registry_boards(inventory, rehash=rehash, strict_hash_match=strict_hash_match)
    validate_counts(boards, inventory)
    return {
        "schema_version": lifecycle.SCHEMA_VERSION,
        "generation": 1,
        "boards": boards,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "/home/curioctylab/.claude/deployment-evidence/"
            "kanban-controlled-reset-20260722/board-inventory.json"
        ),
        help="Path to board-inventory.json",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the registry. Default: hermes_cli.kanban_lifecycle.registry_path()",
    )
    parser.add_argument(
        "--no-rehash", action="store_true",
        help="Skip read-only re-hashing; trust the inventory's recorded db_sha256 verbatim.",
    )
    parser.add_argument(
        "--strict-hash-match", action="store_true",
        help="Abort on any fingerprint mismatch instead of warning (use for CORRUPT/"
             "quarantined boards where drift is unexpected).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the summary; do not write any file.",
    )
    args = parser.parse_args(argv)

    try:
        registry = generate(
            args.inventory, rehash=not args.no_rehash, strict_hash_match=args.strict_hash_match,
        )
    except MigrationValidationError as exc:
        print(f"BLOCKED: migration validation gate failed: {exc}", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for entry in registry["boards"].values():
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    print(f"Validation gate: PASS. counts={counts} total={len(registry['boards'])}")

    if args.dry_run:
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0

    out_path = args.out or lifecycle.registry_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote registry to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
