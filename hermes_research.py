#!/usr/bin/env python3
"""Hermes Research Agent CLI preset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _print_unread_inbox(cwd: Path) -> None:
    try:
        from research.state_store import ResearchStateStore
    except Exception:
        return

    store = ResearchStateStore(cwd=cwd)
    project_id = store.latest_project_id()
    if not project_id:
        return
    unread = store.unread_inbox(project_id)
    if not unread:
        return
    print("Unread research inbox items:")
    for path in unread[:5]:
        preview = path.read_text(encoding="utf-8").splitlines()
        title = preview[0].lstrip("# ").strip() if preview else path.name
        print(f"  - {title} ({path})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hermesresearch",
        description="Launch Hermes Research Agent with the research toolset and autonomous research defaults.",
    )
    parser.add_argument("-q", "--query", help="Single query mode")
    parser.add_argument("-m", "--model", help="Model override")
    parser.add_argument("--provider", help="Inference provider override")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--resume", help="Resume a specific session")
    parser.add_argument("--worktree", "-w", action="store_true", help="Run in an isolated worktree")
    parser.add_argument("--research-mode", choices=["approval", "auto"], help="Override the research execution mode")
    parser.add_argument("--toolsets", help="Additional comma-separated toolsets to append to the research preset")
    args = parser.parse_args()

    os.environ["HERMES_RESEARCH_ENABLED"] = "1"
    if args.research_mode:
        os.environ["HERMES_RESEARCH_MODE"] = args.research_mode

    research_prompt = (
        "Operate as Hermes Research Agent. Prioritize autonomous end-to-end research completion, "
        "durable project state, and resumable long-running loops."
    )
    existing_prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "").strip()
    os.environ["HERMES_EPHEMERAL_SYSTEM_PROMPT"] = (
        f"{existing_prompt}\n\n{research_prompt}".strip() if existing_prompt else research_prompt
    )

    try:
        from tools.skills_sync import sync_skills

        sync_skills(quiet=True)
    except Exception:
        pass

    if not os.getenv("TINKER_API_KEY"):
        print("Warning: TINKER_API_KEY is not set. Research planning works, but Tinker training actions will fail until the key is configured.\n")

    _print_unread_inbox(Path.cwd())

    from cli import main as cli_main

    toolsets = ["research"]
    if args.toolsets:
        toolsets.extend(part.strip() for part in args.toolsets.split(",") if part.strip())

    cli_main(
        query=args.query,
        toolsets=",".join(dict.fromkeys(toolsets)),
        model=args.model,
        provider=args.provider,
        verbose=args.verbose,
        resume=args.resume,
        worktree=args.worktree,
    )


if __name__ == "__main__":
    main()
