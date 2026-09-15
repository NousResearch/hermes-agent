"""``hermes runs`` parser."""

from __future__ import annotations

import os


def build_runs_parser(subparsers, *, cmd_runs) -> None:
    parser = subparsers.add_parser("runs", help="Watch and control API-server runs")
    actions = parser.add_subparsers(dest="runs_action")
    watch = actions.add_parser("watch", help="Stream one run and answer approval prompts")
    watch.add_argument("run_id", help="Run ID returned by POST /v1/runs")
    watch.add_argument("--url", default="http://127.0.0.1:8642", help="API server base URL")
    watch.add_argument("--api-key", default=os.getenv("HERMES_API_KEY", ""),
                       help="API key (defaults to HERMES_API_KEY)")
    watch.set_defaults(func=cmd_runs)
