#!/usr/bin/env python3
"""Manage Hermes typed knowledge lanes."""

from __future__ import annotations

import argparse
import json

from agent.knowledge_lanes import KnowledgeLaneStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Hermes typed knowledge lanes")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show validation report and counts")

    add = sub.add_parser("add-draft", help="Add a draft knowledge record")
    add.add_argument("--title", required=True)
    add.add_argument("--body", required=True)
    add.add_argument("--source", required=True)
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--confidence", choices=["low", "medium", "high"], default="medium")
    add.add_argument("--provenance", default="{}", help="JSON object")

    find = sub.add_parser("find", help="Find relevant knowledge records")
    find.add_argument("--query", default="")
    find.add_argument("--lane", choices=["draft", "promoted", "all"], default="promoted")
    find.add_argument("--tag", action="append", default=[])
    find.add_argument("--limit", type=int, default=5)

    promote = sub.add_parser("promote", help="Promote a draft record")
    promote.add_argument("--id", required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--evidence", action="append", default=[])

    args = parser.parse_args()
    store = KnowledgeLaneStore()

    if args.command == "status":
        payload = store.validation_report()
    elif args.command == "add-draft":
        payload = store.add_draft(
            title=args.title,
            body=args.body,
            source=args.source,
            provenance=json.loads(args.provenance),
            tags=args.tag,
            confidence=args.confidence,
        )
    elif args.command == "find":
        payload = {
            "items": store.find_relevant_items(
                args.query,
                lane=args.lane,
                tags=args.tag,
                limit=args.limit,
            ),
            "lane": args.lane,
            "query": args.query,
            "tags": args.tag,
            "limit": args.limit,
        }
    else:
        payload = store.promote_draft(args.id, promotion_reason=args.reason, evidence=args.evidence)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
