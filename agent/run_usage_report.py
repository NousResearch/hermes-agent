"""Deterministic reporting across explicitly selected profile-local ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hermes_constants import get_default_hermes_root
from agent.run_usage_ledger import UsageLedger, default_ledger_path


def _run_fingerprint(row: dict) -> str:
    """Fingerprint accounting data, excluding ledger-local attribution."""
    comparable = {
        key: value for key, value in row.items()
        if key not in {"identity", "source_profile", "source_profiles", "process_id", "updated_at"}
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"), default=str)


def _deduplicate_global_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_id"]), []).append(row)
    deduplicated: list[dict] = []
    for run_id, candidates in grouped.items():
        candidates.sort(key=lambda row: (str(row.get("source_profile") or ""), str(row.get("process_id") or "")))
        fingerprints = {_run_fingerprint(row) for row in candidates}
        if len(fingerprints) > 1:
            profiles = ", ".join(str(row.get("source_profile") or "unknown") for row in candidates)
            raise ValueError(f"conflicting duplicate run_id {run_id!r} across profiles: {profiles}")
        winner = dict(candidates[0])
        winner["source_profiles"] = sorted({str(row.get("source_profile") or "unknown") for row in candidates})
        deduplicated.append(winner)
    return deduplicated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report Hermes model/tool usage by run, process, session, profile, or card.")
    parser.add_argument("--db", type=Path, default=None, help="one explicit state database path")
    parser.add_argument("--profile", action="append", help="named profile to include; repeatable")
    parser.add_argument("--all-profiles", action="store_true", help="include default and every installed profile ledger")
    parser.add_argument("--board", help="explicit board slug; required for card queries")
    parser.add_argument("--kanban-db", type=Path, default=None, help="explicit Kanban DB for exact task-run joins")
    parser.add_argument("--card-id", "--task-id", dest="task_id")
    parser.add_argument("--task-run-id", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--process-id")
    parser.add_argument("--session-id")
    parser.add_argument("--include-unassigned", action="store_true", help="include direct runs with no board metadata")
    return parser


def _selected_ledgers(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.db is not None:
        return [(args.profile[0] if args.profile else "default", args.db)]
    from hermes_cli import profiles
    root = Path(get_default_hermes_root()).resolve()
    installed = {item.name: Path(item.path) for item in profiles.list_profiles() if item.name != "default"}
    if args.all_profiles:
        # The default ledger is always the root ledger, not the active profile.
        selected = [("default", root / "state.db")]
        selected.extend((name, installed[name] / "state.db") for name in sorted(installed))
        return selected
    if args.profile:
        unknown = [name for name in args.profile if name != "default" and name not in installed]
        if unknown:
            raise ValueError(f"profile ledger not found: {', '.join(sorted(set(unknown)))}")
        return [(name, root / "state.db") if name == "default" else (name, installed[name] / "state.db") for name in args.profile]
    return [("default", root / "state.db")]


def _linked_usage_run_ids(args: argparse.Namespace) -> list[str] | None:
    """Resolve a card to exact task_runs usage links when a board DB is supplied."""
    if args.kanban_db is None or (args.task_id is None and args.task_run_id is None):
        return None
    from hermes_cli import kanban_db
    if not args.kanban_db.is_file():
        raise ValueError(f"Kanban DB not found: {args.kanban_db}")
    with kanban_db.connect_closing(args.kanban_db) as connection:
        if args.task_run_id is not None:
            rows = connection.execute(
                "SELECT usage_run_id FROM task_run_usage WHERE task_run_id = ? ORDER BY task_run_id",
                (args.task_run_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT u.usage_run_id FROM task_run_usage u
                   JOIN task_runs r ON r.id = u.task_run_id
                   WHERE r.task_id = ? ORDER BY u.task_run_id""",
                (args.task_id,),
            ).fetchall()
    return [str(row[0]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    card_query = args.task_id is not None or args.task_run_id is not None or args.include_unassigned
    if card_query and not args.board:
        build_parser().error("--board is required for card/task-run queries")
    if not any((args.board, args.all_profiles, args.profile, args.task_id, args.task_run_id, args.run_id, args.process_id, args.session_id, args.include_unassigned)):
        build_parser().error("select --board, --run-id, --process-id, --session-id, or --all-profiles")
    try:
        ledgers = _selected_ledgers(args)
        linked_run_ids = _linked_usage_run_ids(args)
        rows: list[dict] = []
        for source_profile, path in ledgers:
            if not path.is_file():
                if args.all_profiles:
                    continue
                raise ValueError(f"usage ledger not found for profile {source_profile}: {path}")
            ledger = UsageLedger(path)
            if linked_run_ids:
                for linked_run_id in linked_run_ids:
                    rows.extend(ledger.report(
                        board=args.board, run_id=linked_run_id,
                        source_profile=source_profile,
                        include_unassigned=args.include_unassigned,
                    ))
            else:
                rows.extend(ledger.report(
                    board=args.board,
                    task_id=args.task_id,
                    task_run_id=args.task_run_id,
                    run_id=args.run_id,
                    process_id=args.process_id,
                    session_id=args.session_id,
                    source_profile=source_profile,
                    include_unassigned=args.include_unassigned,
                ))
        rows = _deduplicate_global_runs(rows)
        for row in rows:
            row["identity"] = [row["source_profile"], row["run_id"]]
        rows.sort(key=lambda row: (row.get("started_at") or 0, row["source_profile"], row["run_id"]))
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"run usage report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
