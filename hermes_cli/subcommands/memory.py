"""``hermes memory`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be positive")
    return parsed


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory",
        help="Configure external memory provider",
        description=(
            "Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration"
    )
    _setup_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker",
    )
    memory_sub.add_parser("status", help="Show current memory provider config")
    wiki_parser = memory_sub.add_parser(
        "wiki-index",
        help="Export a read-only structured index over built-in memory files",
    )
    wiki_parser.add_argument(
        "--out",
        default=None,
        help="Write JSON to this path instead of stdout; relative paths go under ~/.hermes/memory-wiki/",
    )
    wiki_parser.add_argument(
        "--query",
        default=None,
        help="Select relevant memory context for this query instead of dumping the full index",
    )
    wiki_parser.add_argument(
        "--max-chars",
        type=_positive_int,
        default=1200,
        help="Character budget for --query context selection (default: 1200)",
    )
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    lint_parser = memory_sub.add_parser("lint", help="Lint built-in memory for duplicates, stale facts, and skill-like entries")
    lint_parser.add_argument("--json", action="store_true")
    bench_parser = memory_sub.add_parser("bench", help="Run a tiny JSON retrieval benchmark against built-in memory")
    bench_parser.add_argument("cases", help="Path to JSON list of {query, expected} cases")
    bench_parser.add_argument("--k", type=_positive_int, default=3)
    bench_parser.add_argument("--json", action="store_true")
    _reset_parser = memory_sub.add_parser(
        "reset",
        help="Erase all built-in memory (MEMORY.md and USER.md)",
    )
    _reset_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    _reset_parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'",
    )
    memory_parser.set_defaults(func=cmd_memory)
