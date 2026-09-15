"""``hermes cleanse`` and ``hermes check-fix`` — project-checker repair loop.

Detect issues via VCS + manifests + linters, normalize diagnostics, apply
automated fixes in parallel file-sticky workers, and verify. Success = empty
remaining diagnostics + written verification_evidence.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_cleanse_command(args) -> int:
    """Run the cleanse command."""
    from agent.cleanse import run_cleanse

    root = Path(getattr(args, "path", None) or ".").resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    max_workers = getattr(args, "workers", 2)
    run_tests = getattr(args, "tests", False)
    json_output = getattr(args, "json", False)

    try:
        result = run_cleanse(root, max_workers=max_workers, run_tests=run_tests)
    except Exception as exc:
        if json_output:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1

    # Human-readable output
    print(f"\n{'='*60}")
    print(f"Cleanse Report: {root}")
    print(f"{'='*60}")
    print(f"Initial issues:   {result.initial_count}")
    print(f"Fixed:            {result.fixed_count}")
    print(f"Remaining:        {len(result.remaining_diagnostics)}")

    if result.skipped_tools:
        print(f"Skipped tools:    {', '.join(result.skipped_tools)}")

    if result.remaining_diagnostics:
        print(f"\n{'─'*60}")
        print("Remaining Issues:")
        print(f"{'─'*60}")
        for diag in result.remaining_diagnostics[:10]:  # Show first 10
            print(f"  {diag.file}:{diag.line}:{diag.col or 1} [{diag.code}] {diag.message}")
        if len(result.remaining_diagnostics) > 10:
            print(f"  ... and {len(result.remaining_diagnostics) - 10} more")

    test_status = result.verification_evidence.get("test_status")
    if test_status:
        print(f"\nTests: {test_status}")

    print(f"\nStatus: {'✓ Clean' if result.ok else '⚠ Incomplete'}")
    print(f"Evidence written to: {root / 'verification_evidence.json'}")
    print(f"{'='*60}\n")

    return 0 if result.ok else 1


def register_cleanse_command(subparsers) -> None:
    """Register cleanse/check-fix commands with argparse."""
    # Main command: hermes cleanse
    cleanse_parser = subparsers.add_parser(
        "cleanse",
        help="Run project-checker repair loop",
        description="Detect issues, apply automated fixes, and verify",
    )
    cleanse_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    cleanse_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2)",
    )
    cleanse_parser.add_argument(
        "--tests",
        action="store_true",
        help="Run project tests after fixes",
    )
    cleanse_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    cleanse_parser.set_defaults(func=run_cleanse_command)

    # Alias: hermes check-fix
    checkfix_parser = subparsers.add_parser(
        "check-fix",
        help="Alias for cleanse",
        description="Detect issues, apply automated fixes, and verify (alias for cleanse)",
    )
    checkfix_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    checkfix_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2)",
    )
    checkfix_parser.add_argument(
        "--tests",
        action="store_true",
        help="Run project tests after fixes",
    )
    checkfix_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    checkfix_parser.set_defaults(func=run_cleanse_command)
