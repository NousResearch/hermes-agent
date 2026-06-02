#!/usr/bin/env python3
"""Render deterministic Self/Lineage status lines.

This is a lightweight copy-paste helper for profile/Matrix status reporting. It
uses the canonical live vocabulary (working/waiting/blocked/dormant) and accepts
legacy Kanban aliases only at the input boundary for migration compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.status_lines import (  # noqa: E402
    LIVE_STATUSES,
    StatusLineError,
    render_status_pair,
    render_status_with_lineage_aggregate,
)


def _statuses_from_counts(count_args: list[list[str]] | None) -> list[str]:
    statuses: list[str] = []
    for status, raw_count in count_args or []:
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise StatusLineError(f"invalid count {raw_count!r} for lineage status {status!r}") from exc
        if count < 0:
            raise StatusLineError(f"negative count {count} for lineage status {status!r}")
        statuses.extend([status] * count)
    return statuses


def _statuses_from_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("descendants", "profiles", "tasks"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise StatusLineError("lineage JSON must be a list or an object with descendants/profiles/tasks")

    statuses: list[str] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            statuses.append(item)
            continue
        if isinstance(item, dict):
            for key in ("live_status", "status", "state", "kanban_status"):
                if key in item:
                    statuses.append(str(item[key]))
                    break
            else:
                raise StatusLineError(f"lineage JSON item {idx} lacks live_status/status/state/kanban_status")
            continue
        raise StatusLineError(f"lineage JSON item {idx} must be a string or object")
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render deterministic Self/Lineage status lines.")
    parser.add_argument(
        "--self",
        nargs=2,
        metavar=("STATUS", "DETAIL"),
        required=True,
        help="self status and specific detail text",
    )
    lineage = parser.add_mutually_exclusive_group(required=True)
    lineage.add_argument(
        "--lineage",
        nargs=2,
        metavar=("STATUS", "DETAIL"),
        help="explicit lineage status and detail text",
    )
    lineage.add_argument(
        "--lineage-status",
        action="append",
        metavar="STATUS",
        help="one structural descendant status; repeat to aggregate",
    )
    lineage.add_argument(
        "--lineage-count",
        action="append",
        nargs=2,
        metavar=("STATUS", "COUNT"),
        help="aggregate descendant status count, e.g. --lineage-count working 2",
    )
    lineage.add_argument(
        "--lineage-json",
        type=Path,
        metavar="PATH",
        help="JSON list/object containing descendant statuses",
    )
    parser.add_argument(
        "--list-statuses",
        action="store_true",
        help="print the canonical live status vocabulary before the rendered block",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.lineage is not None:
            rendered = render_status_pair(args.self[0], args.self[1], args.lineage[0], args.lineage[1])
        elif args.lineage_status is not None:
            rendered = render_status_with_lineage_aggregate(args.self[0], args.self[1], args.lineage_status)
        elif args.lineage_count is not None:
            rendered = render_status_with_lineage_aggregate(args.self[0], args.self[1], _statuses_from_counts(args.lineage_count))
        else:
            rendered = render_status_with_lineage_aggregate(args.self[0], args.self[1], _statuses_from_json(args.lineage_json))
    except (OSError, json.JSONDecodeError, StatusLineError) as exc:
        parser.error(str(exc))

    if args.list_statuses:
        print("Canonical live statuses: " + ", ".join(LIVE_STATUSES))
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
