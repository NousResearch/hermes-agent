"""Slash-command entry point for `/evolve-cc`."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from tools.claude_code_evolve import (
    CLAUDE_PROJECTS_ROOT,
    CLAUDE_SKILLS_ROOT,
    analyze_claude_code_history,
    apply_write_plan,
    format_candidates_report,
    plan_candidate_writes,
)


def run_evolve_cc(args: argparse.Namespace, confirm_fn: Callable[[str], str | None] | None = None) -> int:
    """Run Claude Code history analysis from parsed arguments."""

    anchor_path = _resolve_anchor_path(args.path)
    scope = args.scope or _default_scope(anchor_path)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    try:
        result = analyze_claude_code_history(
            anchor_path=anchor_path,
            since=since,
            scope=scope,
            projects_root=CLAUDE_PROJECTS_ROOT,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    selected_memory = tuple(result.memory_candidates[: args.limit_memory])
    selected_skills = tuple(result.skill_candidates[: args.limit_skills])
    write_plan = plan_candidate_writes(
        memory_candidates=selected_memory,
        skill_candidates=selected_skills,
        projects_root=CLAUDE_PROJECTS_ROOT,
        skills_root=CLAUDE_SKILLS_ROOT,
    )

    print(
        format_candidates_report(
            result=result,
            memory_candidates=selected_memory,
            skill_candidates=selected_skills,
            write_plan=write_plan,
        )
    )

    if not args.apply:
        print("\nDry run only. Re-run with /evolve-cc --apply to write the planned files.")
        return 0

    if not write_plan:
        print("\nNothing to write.")
        return 0
    if all(write.already_exists for write in write_plan):
        print("\nAll planned files already exist. Nothing new to write.")
        return 0

    print("\nTarget paths:")
    for write in write_plan:
        suffix = " (exists)" if write.already_exists else ""
        print(f"  - {write.path}{suffix}")

    if confirm_fn is None:
        if not sys.stdin.isatty():
            print("\n--apply requires an interactive TTY for confirmation.")
            return 1
        answer = input("\nApply these writes? [y/N]: ").strip().lower()
    else:
        answer = (confirm_fn("\nApply these writes? [y/N]: ") or "").strip().lower()

    if answer not in {"y", "yes"}:
        print("Aborted.")
        return 1

    written, skipped = apply_write_plan(write_plan)
    print(f"Wrote {written} file(s); skipped {skipped} existing file(s).")
    return 0


def run_evolve_cc_slash(
    cmd: str,
    confirm_fn: Callable[[str], str | None] | None = None,
) -> int:
    """Parse and execute `/evolve-cc ...` from the chat UI."""

    parser = _build_parser()
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()

    if parts and parts[0].lower() == "/evolve-cc":
        parts = parts[1:]

    try:
        args = parser.parse_args(parts)
    except SystemExit:
        print(f"Usage: {parser.format_usage().strip()}")
        return 2

    return run_evolve_cc(args, confirm_fn=confirm_fn)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/evolve-cc", add_help=False)
    parser.add_argument("--days", type=_positive_int, default=30)
    parser.add_argument("--scope", choices=("cwd", "repo", "all"), default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit-memory", type=_positive_int, default=10)
    parser.add_argument("--limit-skills", type=_positive_int, default=5)
    return parser


def _default_scope(anchor_path: Path) -> str:
    from tools.claude_code_evolve import _git_repo_root  # local import to avoid polluting public surface

    return "repo" if _git_repo_root(anchor_path) is not None else "cwd"


def _resolve_anchor_path(raw_path: str | None) -> Path:
    base = Path(os.getcwd())
    if not raw_path:
        return base.resolve(strict=False)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed
