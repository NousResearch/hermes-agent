"""``hermes engineering-memory`` parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_engineering_memory_parser(subparsers, *, cmd_engineering_memory: Callable) -> None:
    parser = subparsers.add_parser(
        "engineering-memory",
        help="Review and search the shared engineering-memory index",
        description="Manage the explicit local engineering-memory ledger and its rebuildable index.",
    )
    parser.add_argument("--vault", default=None, help="Shared engineering-memory vault directory")
    parser.add_argument("--index", dest="index_path", default=None, help="Materialized SQLite index path")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    commands = parser.add_subparsers(dest="engineering_memory_command")

    propose = commands.add_parser("propose", help="Validate and stage a Markdown record")
    propose.add_argument("record", help="Path to a structured Markdown record")
    add_json_flag(propose, "Emit a machine-readable result")

    review = commands.add_parser("review", help="Apply an explicit human review action")
    review.add_argument("record_id")
    review.add_argument("action", choices=["approve", "reject", "supersede", "withdraw"])
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--evidence-ref", action="append", default=[])
    add_json_flag(review, "Emit a machine-readable result")

    search = commands.add_parser("search", help="Search approved engineering records")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--repository")
    search.add_argument("--workspace")
    search.add_argument("--component")
    search.add_argument("--task-type")
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--as-of-head")
    search.add_argument("--include-unverified", action="store_true")
    add_json_flag(search, "Emit a machine-readable result")

    rebuild = commands.add_parser("rebuild", help="Rebuild the SQLite index from approved records")
    add_json_flag(rebuild, "Emit a machine-readable result")

    verify = commands.add_parser("verify", help="Check ledger and index health")
    add_json_flag(verify, "Emit a machine-readable result")

    laya = commands.add_parser("laya", help="Inspect or explicitly prepare the local LAYA sidecar")
    laya_commands = laya.add_subparsers(dest="laya_command")
    doctor = laya_commands.add_parser("doctor", help="Check LAYA configuration without loading a model")
    doctor.add_argument("--repo", default="convaiinnovations/laya")
    doctor.add_argument("--revision", default="")
    doctor.add_argument("--local-path", default="")
    add_json_flag(doctor, "Emit a machine-readable result")
    download = laya_commands.add_parser("download", help="Download a pinned LAYA revision")
    download.add_argument("--repo", default="convaiinnovations/laya")
    download.add_argument("--revision", required=True)
    download.add_argument("--dir", dest="destination", required=True)
    add_json_flag(download, "Emit a machine-readable result")
    calibrate = laya_commands.add_parser("calibrate", help="Create a local held-out calibration artifact")
    calibrate.add_argument("--fixtures", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--revision", required=True)
    calibrate.add_argument("--runtime-command", nargs="+", required=True)
    add_json_flag(calibrate, "Emit a machine-readable result")

    parser.set_defaults(func=cmd_engineering_memory)
