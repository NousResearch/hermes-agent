"""Argument layer for the legacy ``hermes-agent`` runner (``run_agent.main``).

A console script calls its target with no arguments, so pointing ``hermes-agent``
at ``run_agent.main`` ignored argv entirely: ``--help``, ``--version`` and a bare
invocation all ran a real model turn with ``main()``'s built-in demo query, and
``--query`` was silently dropped (#54648). ``python run_agent.py`` routes through
here too, so the installer's PATH launcher behaves the same way.
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Optional


def _build_parser() -> argparse.ArgumentParser:
    from hermes_cli import __release_date__, __version__

    parser = argparse.ArgumentParser(
        prog="hermes-agent",
        description="Legacy single-query Hermes Agent runner. For the full CLI use `hermes`.",
    )
    parser.add_argument("--version", action="version", version=f"Hermes Agent v{__version__} ({__release_date__})")
    parser.add_argument("prompt", nargs="*", help="query to run (same as --query)")
    parser.add_argument("--query", "-q", help="natural-language query to run")
    parser.add_argument("--model", default="", help="model id (provider/model)")
    parser.add_argument("--api-key", "--api_key", dest="api_key", help="API key for the model endpoint")
    parser.add_argument("--base-url", "--base_url", dest="base_url", default="", help="model API base URL")
    parser.add_argument("--max-turns", "--max_turns", dest="max_turns", type=int, default=10,
                        help="maximum API call iterations (default: 10)")
    parser.add_argument("--enabled-toolsets", "--enabled_toolsets", dest="enabled_toolsets",
                        help="comma-separated toolsets to enable")
    parser.add_argument("--disabled-toolsets", "--disabled_toolsets", dest="disabled_toolsets",
                        help="comma-separated toolsets to disable")
    parser.add_argument("--list-tools", "--list_tools", dest="list_tools", action="store_true",
                        help="list available tools and exit")
    parser.add_argument("--save-trajectories", "--save_trajectories", dest="save_trajectories",
                        action="store_true", help="append the conversation to trajectory JSONL files")
    parser.add_argument("--save-sample", "--save_sample", dest="save_sample", action="store_true",
                        help="save one trajectory sample to a UUID-named file")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--log-prefix-chars", "--log_prefix_chars", dest="log_prefix_chars", type=int,
                        default=20, help="characters shown in tool-call log previews (default: 20)")
    return parser


def main(argv: Optional[List[str]] = None, *, run: Optional[Callable[..., object]] = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``) and run one query through ``run_agent.main``.

    Metadata flags and a bare invocation never reach the runner; ``run`` lets
    ``python run_agent.py`` pass its own ``main`` instead of importing the module twice.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    positional = " ".join(args.prompt).strip()
    if args.query and positional:
        parser.error("pass the query either positionally or via --query, not both")
    query = args.query or positional or None
    if query is None and not args.list_tools:
        parser.print_help()
        print("\nNo query given: pass one with --query (or run `hermes` for the interactive CLI).")
        return 0

    if run is None:
        from run_agent import main as run

    run(
        query=query,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_turns=args.max_turns,
        enabled_toolsets=args.enabled_toolsets,
        disabled_toolsets=args.disabled_toolsets,
        list_tools=args.list_tools,
        save_trajectories=args.save_trajectories,
        save_sample=args.save_sample,
        verbose=args.verbose,
        log_prefix_chars=args.log_prefix_chars,
    )
    return 0
