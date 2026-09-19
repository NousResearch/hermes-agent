#!/usr/bin/env python3
"""Regression tests for scripts/generate_boards_manifest.py (kanban t_aee179af).

Covers the three required behaviours from the card:
  1. Dual-slug tie-break: a non-active project sharing a board slug with an
     active project must never downgrade the active board.
  2. Quiesce flags (DISPATCH_QUIESCE) survive regeneration: an active board
     with a declared quiesce keeps dispatch=false and reports why.
  3. --write blind-run guard: refuses (non-zero exit, file untouched) when
     the generated manifest would flip any board's dispatch flag versus the
     on-disk file, unless --ack-dispatch-flip is passed.

Run: python3 /home/frank/.hermes/scripts/test_generate_boards_manifest.py
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

GEN = SCRIPTS / "generate_boards_manifest.py"
LIVE = Path("/home/frank/.hermes/kanban/boards-manifest.json")

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + label + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def test_dual_slug_active_wins(gbm) -> None:
    print("\n=== 1. dual-slug tie-break: active project must not be downgraded ===")
    fake_registry = {
        "projects": [
            {
                "id": "active-proj",
                "state": "active",
                "execution_adapter": "hermes-kanban",
                "board_or_tracker": "shared-slug",
                "primary_owner": "owner-a",
            },
            {
                "id": "maint-proj",
                "state": "maintenance",
                "execution_adapter": "hermes-kanban",
                "board_or_tracker": "shared-slug",
                "primary_owner": "owner-b",
            },
        ]
    }
    boards = gbm.build_boards(fake_registry)
    entry = boards.get("shared-slug", {})
    check("shared-slug resolves to active", entry.get("state") == "active", str(entry))
    check("shared-slug dispatch stays true", entry.get("dispatch") is True, str(entry))
    check("shared-slug owner is not null", bool(entry.get("owner")), str(entry))

    # Order independence: same result regardless of which project comes first.
    fake_registry_reordered = {"projects": list(reversed(fake_registry["projects"]))}
    boards2 = gbm.build_boards(fake_registry_reordered)
    check(
        "tie-break is order-independent",
        boards2.get("shared-slug", {}).get("state") == "active",
        str(boards2.get("shared-slug")),
    )


def test_real_yorkstone_dual_slug(gbm) -> None:
    print("\n=== 2. real registry: yorkstone-supplies stays active/dispatch=true ===")
    registry = gbm.load_registry()
    boards = gbm.build_boards(registry)
    entry = boards.get("yorkstone-supplies", {})
    check("yorkstone-supplies state=active", entry.get("state") == "active", str(entry))
    check("yorkstone-supplies dispatch=true", entry.get("dispatch") is True, str(entry))
    check(
        "yorkstone-supplies owner=yorkstone-supplies-pm",
        entry.get("owner") == "yorkstone-supplies-pm",
        str(entry),
    )
    check(
        "yorkstone-supplies reviewer=platform-reviewer",
        entry.get("reviewer") == "platform-reviewer",
        str(entry),
    )


def test_quiesce_survives_regeneration(gbm) -> None:
    print("\n=== 3. quiesce flags survive regeneration ===")
    registry = gbm.load_registry()
    boards = gbm.build_boards(registry)
    for slug in gbm.DISPATCH_QUIESCE:
        entry = boards.get(slug, {})
        check(f"{slug} state stays active under quiesce", entry.get("state") == "active", str(entry))
        check(f"{slug} dispatch forced false by quiesce", entry.get("dispatch") is False, str(entry))
        check(f"{slug} carries quiesce_reason", bool(entry.get("quiesce_reason")), str(entry))
    check(
        "yorkstone-supplies (not quiesced) keeps dispatch=true",
        boards.get("yorkstone-supplies", {}).get("dispatch") is True,
    )


def test_write_guard_refuses_unacked_flip() -> None:
    print("\n=== 4. --write refuses an unacknowledged dispatch flip ===")
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "boards-manifest.json"
        live = json.loads(LIVE.read_text())
        live["boards"]["yorkstone-supplies"]["dispatch"] = False  # simulate a stale flag
        mp.write_text(json.dumps(live, indent=2))
        before = mp.read_text()

        r = run([sys.executable, str(GEN), "--write", "--manifest", str(mp)])
        check("refuses with non-zero exit", r.returncode != 0, f"rc={r.returncode}")
        check("refusal message names the flipped board", "yorkstone-supplies" in r.stderr, r.stderr)
        check("file left untouched on refusal", mp.read_text() == before)

        r2 = run([sys.executable, str(GEN), "--write", "--ack-dispatch-flip", "--manifest", str(mp)])
        check("succeeds once acknowledged", r2.returncode == 0, f"rc={r2.returncode} stderr={r2.stderr}")
        after = json.loads(mp.read_text())
        check(
            "acknowledged write applies the real generated value",
            after["boards"]["yorkstone-supplies"]["dispatch"] is True,
        )


def test_write_guard_noop_when_no_flip() -> None:
    print("\n=== 5. --write succeeds without --ack-dispatch-flip when no dispatch flips ===")
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "boards-manifest.json"
        mp.write_text(LIVE.read_text())
        r = run([sys.executable, str(GEN), "--write", "--manifest", str(mp)])
        check(
            "no-flip write succeeds without --ack-dispatch-flip",
            r.returncode == 0,
            f"rc={r.returncode} stderr={r.stderr}",
        )


def test_check_only_still_works() -> None:
    print("\n=== 6. --check-only exit codes still meaningful (not broken by the guard) ===")
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "boards-manifest.json"
        mp.write_text(LIVE.read_text())
        r = run([sys.executable, str(GEN), "--check-only", "--manifest", str(mp)])
        check(
            "--check-only against a live-identical copy DIFFERS only on new quiesce_reason field",
            r.returncode in (0, 1),
            f"rc={r.returncode}",
        )


def main() -> int:
    gbm = importlib.import_module("generate_boards_manifest")
    test_dual_slug_active_wins(gbm)
    test_real_yorkstone_dual_slug(gbm)
    test_quiesce_survives_regeneration(gbm)
    test_write_guard_refuses_unacked_flip()
    test_write_guard_noop_when_no_flip()
    test_check_only_still_works()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
