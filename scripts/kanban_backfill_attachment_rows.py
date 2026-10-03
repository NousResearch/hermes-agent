"""Backfill ``task_attachments`` rows for declared artifacts already on disk.

A card's declared deliverable that already sat in its own
``attachments/<task>/`` directory was never registered: the completion staging
step recorded rows only for files it *copied*, gating the registration list
behind "did a copy happen". So a worker that wrote its deliverable straight into
the board's git-tracked attachments dir — and declared that path on
``kanban_complete`` — left a file on disk with zero rows, and
``kanban_attachments`` returned ``[]`` for it. Fixed in ``hermes_cli.kanban_db``.

This repairs boards written before the fix by replaying exactly what the fixed
code records, and nothing more: the *declared* artifacts from the card's own
``completed`` / ``review_requested`` event payload that live under that card's
attachments dir. It never sweeps stray files in the directory (scratch scripts a
worker happened to leave there are not deliverables).

Dry-run by default; ``--apply`` writes the row and the same ``attached`` event
``kanban_complete`` writes, through the public ``kanban_db.add_attachment``.

    python scripts/kanban_backfill_attachment_rows.py                 # report
    python scripts/kanban_backfill_attachment_rows.py --apply
    python scripts/kanban_backfill_attachment_rows.py --board skill-sweep --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402

_HANDOFF_EVENTS = ("completed", "review_requested")

# The dispatcher injects these into every worker's environment, and they PIN the
# resolution: ``HERMES_KANBAN_DB`` makes ``kanban_db_path(board=…)`` return that
# one file no matter which board is asked for, so a repair sweep would read one
# board N times and call the other boards clean. Drop them and resolve per board.
_PINNED_ENV = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_WORKSPACES_ROOT",
)


def _release_board_pins() -> list[str]:
    """Unset dispatcher-injected board pins; returns what was removed."""
    import os

    dropped = [name for name in _PINNED_ENV if os.environ.pop(name, None) is not None]
    if dropped and not os.environ.get("HERMES_KANBAN_SWEEP_QUIET"):
        print(f"note: ignoring board pins from the environment: {', '.join(dropped)}\n")
    return dropped


def _declared_artifacts(conn, task_id: str) -> list[str]:
    """Artifact paths from the card's handoff events, newest event first.

    A card can hand off more than once (review, then approval), so every
    handoff's payload is read and de-duplicated in order.
    """
    declared: list[str] = []
    for kind in _HANDOFF_EVENTS:
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY created_at DESC, id DESC",
            (task_id, kind),
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row[0]) if row[0] else {}
            except (TypeError, ValueError):
                continue
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, list):
                continue
            for item in artifacts:
                if isinstance(item, str) and item.strip() and item not in declared:
                    declared.append(item)
    return declared


def backfill_board(slug: str, *, apply: bool) -> dict:
    attachments_dir = kb.attachments_root(board=slug).resolve()
    report = {"board": slug, "db": str(kb.kanban_db_path(board=slug)),
              "attachments_dir": str(attachments_dir),
              "registered": [], "already": 0, "skipped": []}
    with kbc.connect_closing(board=slug) as conn:
        task_ids = [row[0] for row in conn.execute("SELECT id FROM tasks").fetchall()]
        for task_id in task_ids:
            declared = _declared_artifacts(conn, task_id)
            if not declared:
                continue
            registered = {
                row[0] for row in conn.execute(
                    "SELECT stored_path FROM task_attachments WHERE task_id = ?",
                    (task_id,),
                ).fetchall()
            }
            # The task's OWN attachments dir, not the board root: one card may
            # legitimately cite a sibling card's deliverable, and that file is
            # the sibling's attachment, not this one's.
            task_dir = kb.task_attachments_dir(task_id, board=slug).resolve()
            for declared_path in declared:
                try:
                    resolved = Path(declared_path).expanduser().resolve()
                except OSError:
                    report["skipped"].append({"task": task_id, "path": declared_path,
                                              "why": "unresolvable"})
                    continue
                if not resolved.is_file():
                    report["skipped"].append({"task": task_id, "path": declared_path,
                                              "why": "not a file"})
                    continue
                if not resolved.is_relative_to(task_dir):
                    # Outside the attachments dir: not an attachment row's job
                    # (the fix does not register these either).
                    report["skipped"].append({"task": task_id, "path": declared_path,
                                              "why": "outside attachments dir"})
                    continue
                if str(resolved) in registered:
                    report["already"] += 1
                    continue
                entry = {"task": task_id, "path": str(resolved),
                         "size": resolved.stat().st_size}
                if apply:
                    entry["attachment_id"] = kb.add_attachment(
                        conn, task_id, filename=resolved.name, stored_path=str(resolved),
                        size=entry["size"], uploaded_by="backfill",
                    )
                    registered.add(str(resolved))
                report["registered"].append(entry)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="write the rows (default: report only)")
    parser.add_argument("--board", help="only this board slug (default: every board)")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()

    _release_board_pins()
    if args.apply:
        # Deliberately NOT self-healing: the fence exists so a delegated child
        # cannot mutate the board, and a repair script must not defeat it. Run
        # this from an operator shell (or with the marker unset) instead.
        import os

        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

        if os.environ.get(DELEGATED_CHILD_ENV_MARKER):
            print(
                f"error: {DELEGATED_CHILD_ENV_MARKER} is set — every kanban write in this "
                f"process tree is fenced (delegated child contexts cannot mutate boards).\n"
                f"       Run with it unset, e.g.  env -u {DELEGATED_CHILD_ENV_MARKER} python "
                f"{Path(__file__).name} --apply",
                file=sys.stderr,
            )
            return 2
    slugs = [args.board] if args.board else [
        b["slug"] for b in kb.list_boards(include_archived=True)
    ]
    reports = [backfill_board(slug, apply=args.apply) for slug in slugs]

    if args.json:
        print(json.dumps({"applied": args.apply, "boards": reports}, indent=2))
    else:
        mode = "APPLIED" if args.apply else "DRY RUN (pass --apply to write)"
        print(f"kanban attachment-row backfill — {mode}")
        total = 0
        for rep in reports:
            n = len(rep["registered"])
            total += n
            if not n and not rep["skipped"]:
                continue
            print(f"\n{rep['board']}: {n} row(s) to write, {rep['already']} already present")
            for entry in rep["registered"]:
                print(f"  + {entry['task']}  {entry['path']}  ({entry['size']} B)")
            for skip in rep["skipped"]:
                print(f"  - {skip['task']}  {skip['path']}  ({skip['why']})")
        print(f"\ntotal: {total} row(s) {'written' if args.apply else 'would be written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
