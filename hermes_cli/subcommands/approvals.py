"""``hermes approvals`` subcommand parser."""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_approvals_parser(subparsers, *, cmd_approvals: Callable) -> None:
    """Attach the ``approvals`` subcommand to ``subparsers``."""
    approvals_parser = subparsers.add_parser(
        "approvals", help="Approval-prompt tools (mine history into allowlist proposals)",
        description="Tools for the dangerous-command approval system. "
            "`hermes approvals suggest` mines past approval decisions from "
            "the session database and proposes command_allowlist entries so "
            "repeatedly-approved commands stop prompting.")
    approvals_subparsers = approvals_parser.add_subparsers(
        dest="approvals_command", metavar="<subcommand>")

    suggest_parser = approvals_subparsers.add_parser(
        "suggest", help="Propose command_allowlist entries from past approvals",
        description="Scan the session database for dangerous-classified commands "
            "that ran with user approval, rank the recurring patterns, and "
            "print a numbered allowlist proposal. Nothing is written unless "
            "--apply is given. Destructive classes (recursive delete, sudo, "
            "disk writes, credential edits, ...) are never proposed.")
    suggest_parser.add_argument(
        "--apply", dest="apply_indices", metavar="N[,M...]",
        help="Merge the numbered proposals (from a prior run) into "
        "command_allowlist in config.yaml")
    add_json_flag(suggest_parser, "Emit machine-readable JSON instead of human-readable text")
    suggest_parser.add_argument(
        "--days", type=int, default=90,
        help="How far back to scan session history (default: 90; 0 = all)")
    suggest_parser.add_argument(
        "--min-count", dest="min_count", type=int, default=2,
        help="Minimum approval count for a pattern to be proposed (default: 2)")
    suggest_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of proposals to show (default: 20)")
    suggest_parser.add_argument(
        "--db", help="Path to an alternate session database (default: ~/.hermes/state.db)")
    suggest_parser.set_defaults(func=cmd_approvals)

    test_parser = approvals_subparsers.add_parser(
        "test", help="Dry-run the approval verdict for a command (never executes it)",
        description="Evaluate a command against the REAL runtime approval guards — "
            "hardline blocklist, user approvals.deny rules, dangerous-pattern "
            "detection, allowlist, yolo/off bypass — and print the verdict, "
            "the matching rule, and the normalized-command trace, without "
            "executing the command, prompting anyone, or persisting anything. "
            "Exit codes: 0 allow, 2 ask-approval, 3 deny (hardline or user "
            "deny rule). Tip: use `--` before the command so its own flags "
            "aren't parsed: hermes approvals test -- rm -rf ./build")
    test_parser.add_argument(
        "--env-type", dest="env_type", default="local",
        help="Terminal backend type to evaluate against (default: local; "
        "isolated container backends like docker skip the guards)")
    add_json_flag(test_parser, "Emit machine-readable JSON instead of human-readable text")
    test_parser.add_argument(
        "command_words",
        nargs=argparse.REMAINDER,
        metavar="command",
        # NOTE: dest must NOT be "command" — main.py's startup path reads
        # args.command as the top-level subcommand name ("approvals").
        help="The command to evaluate (prefix with -- to protect its flags)")
    test_parser.set_defaults(func=cmd_approvals)

    audit_parser = approvals_subparsers.add_parser(
        "audit", help="Read the append-only approval / guarded-command audit log",
        description="List approval decisions and guarded-command verdicts recorded by "
            "the append-only audit sink (one SQLite partition per UTC day under "
            "<store>/audit/, HMAC-keyed target digests, no raw target text). "
            "Filter by day range, decision, guard surface, trace id or "
            "classification. `--protected` is the pre-built filter for attempts "
            "that targeted a protected path (Hermes home/config/env, SSH and "
            "shell-rc, cloud instance credentials).")
    audit_parser.add_argument(
        "--days", type=int, default=7,
        help="Only partitions from the last N days (default: 7; 0 = every partition)")
    audit_parser.add_argument(
        "--decision", choices=("allow", "deny", "prompt", "ask"), default="",
        help="Only this decision (ask is an alias for prompt)")
    audit_parser.add_argument(
        "--surface", default="",
        help="Only this guard surface (terminal, execute_code, file_write, "
             "computer_use, plugin_rule, ...)")
    audit_parser.add_argument(
        "--trace-id", dest="trace_id", default="",
        help="Only this trace id (the delegation/execution chain that produced the row)")
    audit_parser.add_argument(
        "--class-like", dest="class_like", default="",
        help="Only classifications matching this SQL LIKE pattern "
             "(e.g. '%%secrets%%')")
    audit_parser.add_argument(
        "--protected", action="store_true",
        help="Only attempts that targeted a protected path")
    audit_parser.add_argument(
        "--limit", type=int, default=50,
        help="Maximum rows to print, newest first (default: 50; 0 = no limit)")
    audit_parser.add_argument(
        "--summary", action="store_true",
        help="Also print aggregate counts over the selected range")
    audit_parser.add_argument(
        "--verify", action="store_true",
        help="Recompute every row's HMAC chain instead of listing; exit 1 on a broken "
             "partition")
    add_json_flag(audit_parser, "Emit machine-readable JSON instead of human-readable text")
    audit_parser.set_defaults(func=cmd_approvals)

    approvals_parser.set_defaults(func=cmd_approvals)
