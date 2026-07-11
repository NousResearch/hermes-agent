"""Machine-readable CLI used by the local-video-asset-indexer Skill."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .batch import library_status, list_libraries, scan_library, search_library


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capabilities.video_library.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("libraries")
    scan = commands.add_parser("scan")
    scan.add_argument("--library", required=True)
    scan.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--library", required=True)
    search = commands.add_parser("search")
    search.add_argument("--library", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--tag", default="")
    search.add_argument("--limit", type=int, default=50)
    return parser


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "libraries":
            payload = list_libraries()
        elif args.command == "scan":
            payload = scan_library(args.library, dry_run=bool(args.dry_run))
        elif args.command == "status":
            payload = library_status(args.library)
        else:
            payload = search_library(args.library, args.query, tag=args.tag, limit=args.limit)
        _print(payload)
        return 0
    except Exception as exc:
        _print({"error": {"message": str(exc), "type": type(exc).__name__}, "ok": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
