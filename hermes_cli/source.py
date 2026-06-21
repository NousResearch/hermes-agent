from __future__ import annotations

import json
import sys
from pathlib import Path

_ACCELERATOR_ROOT = Path(__file__).resolve().parents[2] / "workspace" / "scripts" / "source_accelerator"
if str(_ACCELERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACCELERATOR_ROOT))


def cmd_source(args):
    if args.source_command == "search":
        from source_accelerator.query import search
        print(json.dumps(search(args.query, args.scope, args.mode, args.limit), indent=2))
        return
    if args.source_command == "open":
        from source_accelerator.query import open_result
        print(json.dumps(open_result(args.result_id, args.context), indent=2))
        return
    if args.source_command == "status":
        from source_accelerator.status import status
        print(json.dumps(status(), indent=2))
        return
    if args.source_command == "refresh":
        from source_accelerator.indexer import refresh_fast
        print(json.dumps(refresh_fast(kind="fast" if args.fast else "manual", force=args.force), indent=2))
        return
    if args.source_command == "benchmark":
        from source_accelerator.benchmark import benchmark
        print(json.dumps(benchmark(args.iterations, args.queries, args.output_json), indent=2))
        return
    raise SystemExit("missing source subcommand")


def register_source_parser(subparsers):
    parser = subparsers.add_parser(
        "source",
        help="Fast deterministic Hermes source/workspace lookup",
        description="Hermes Source Accelerator: SQLite FTS5/trigram hot path, no LLM/Graphify/GitNexus rebuild in lookup path.",
    )
    sub = parser.add_subparsers(dest="source_command", required=True)
    search_p = sub.add_parser("search", help="Search Hermes source/workspace index")
    search_p.add_argument("query")
    search_p.add_argument("--scope", default="auto")
    search_p.add_argument("--mode", default="auto")
    search_p.add_argument("--limit", type=int, default=20)
    open_p = sub.add_parser("open", help="Open a result by reading the current source file from disk")
    open_p.add_argument("result_id")
    open_p.add_argument("--context", type=int, default=80)
    sub.add_parser("status", help="Show source accelerator index health")
    refresh_p = sub.add_parser("refresh", help="Refresh deterministic local index")
    refresh_p.add_argument("--fast", action="store_true", default=True)
    refresh_p.add_argument("--force", action="store_true")
    bench_p = sub.add_parser("benchmark", help="Run benchmark query set against the accelerator")
    bench_p.add_argument("--iterations", type=int, default=5)
    bench_p.add_argument("--queries")
    bench_p.add_argument("--output-json")
    parser.set_defaults(func=cmd_source)
    return parser
