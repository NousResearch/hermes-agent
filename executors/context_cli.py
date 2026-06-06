#!/usr/bin/env python3
"""
CLI subcommands for project context management.

Usage (via executors.cli):
    python -m executors.cli context show
    python -m executors.cli context edit --field overview --value "..."
    python -m executors.cli context edit --field sprint --value "..."
    python -m executors.cli context adr add ADR-001 "Use Redis" "Use Redis for caching"
    python -m executors.cli context adr remove ADR-001
    python -m executors.cli context cmd add build "npm run build"
    python -m executors.cli context cmd remove build
    python -m executors.cli context forbidden add "secrets/"
    python -m executors.cli context injection on|off
    python -m executors.cli context preview --executor codex-cli --goal "refactor auth"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from executors.context import WorkspaceContextManager
from executors.prompt_builder import PromptBuilder
from executors.types import ProjectContext


# ---------------------------------------------------------------------------
# Field mapping: CLI field name → context attribute + type
# ---------------------------------------------------------------------------

_TEXT_FIELDS = {
    "overview": ("project_overview", "Project overview"),
    "architecture": ("architecture_notes", "Architecture notes"),
    "sprint": ("current_sprint", "Current sprint"),
    "conventions": ("coding_conventions", "Coding conventions"),
}


async def cmd_show(mgr: WorkspaceContextManager, json_output: bool = False) -> None:
    """Display the full project context."""
    ctx = mgr.get_context()

    if json_output:
        data = {
            "project_overview": ctx.project_overview,
            "architecture_notes": ctx.architecture_notes,
            "current_sprint": ctx.current_sprint,
            "adr_summaries": [
                {"id": a.id, "title": a.title, "decision": a.decision}
                for a in ctx.adr_summaries
            ],
            "common_commands": [
                {"label": c.label, "command": c.command}
                for c in ctx.common_commands
            ],
            "test_commands": [
                {"label": c.label, "command": c.command}
                for c in ctx.test_commands
            ],
            "forbidden_areas": ctx.forbidden_areas,
            "coding_conventions": ctx.coding_conventions,
            "recent_tasks": [
                {"thread_id": t.thread_id, "title": t.title, "executor": t.executor,
                 "status": t.status, "completed_at": t.completed_at}
                for t in ctx.recent_tasks
            ],
            "context_injection_enabled": ctx.context_injection_enabled,
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print("=== Project Context ===")
    print(f"Injection: {'enabled' if ctx.context_injection_enabled else 'DISABLED'}")

    def _section(title: str, content: str, max_len: int = 200):
        if not content:
            return
        print(f"\n--- {title} ---")
        if len(content) > max_len:
            print(content[:max_len] + "…")
        else:
            print(content)

    _section("Overview", ctx.project_overview)
    _section("Architecture", ctx.architecture_notes, 500)
    _section("Sprint", ctx.current_sprint)

    if ctx.adr_summaries:
        print(f"\n--- ADRs ({len(ctx.adr_summaries)}) ---")
        for a in ctx.adr_summaries:
            print(f"  {a.id}: {a.title}")
            print(f"    → {a.decision}")

    if ctx.common_commands:
        print(f"\n--- Common Commands ---")
        for c in ctx.common_commands:
            print(f"  {c.label}: {c.command}")

    if ctx.test_commands:
        print(f"\n--- Test Commands ---")
        for c in ctx.test_commands:
            print(f"  {c.label}: {c.command}")

    if ctx.forbidden_areas:
        print(f"\n--- Forbidden Areas ---")
        for p in ctx.forbidden_areas:
            print(f"  ✗ {p}")

    _section("Conventions", ctx.coding_conventions)

    if ctx.recent_tasks:
        print(f"\n--- Recent Tasks ({len(ctx.recent_tasks)}) ---")
        for t in ctx.recent_tasks[-5:]:
            icon = "✓" if t.status == "done" else "✗"
            print(f"  {icon} {t.title} ({t.executor})")


async def cmd_edit(mgr: WorkspaceContextManager, field: str, value: str) -> None:
    """Edit a text field of the project context."""
    if field not in _TEXT_FIELDS:
        valid = ", ".join(_TEXT_FIELDS.keys())
        print(f"Invalid field: {field}. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    attr_name, label = _TEXT_FIELDS[field]
    setattr(mgr.get_context(), attr_name, value)
    mgr.save()
    print(f"Updated {label}")


async def cmd_adr(mgr: WorkspaceContextManager, action: str, *args: str) -> None:
    """Add or remove an ADR."""
    if action == "add" and len(args) >= 3:
        adr_id, title, decision = args[0], args[1], " ".join(args[2:])
        mgr.add_adr(adr_id, title, decision)
        print(f"Added ADR: {adr_id}")
    elif action == "remove" and len(args) >= 1:
        mgr.remove_adr(args[0])
        print(f"Removed ADR: {args[0]}")
    else:
        print("Usage: context adr add <id> <title> <decision>", file=sys.stderr)
        print("       context adr remove <id>", file=sys.stderr)
        sys.exit(1)


async def cmd_command(mgr: WorkspaceContextManager, action: str, *args: str) -> None:
    """Add or remove a common command."""
    if action == "add" and len(args) >= 2:
        label, command = args[0], " ".join(args[1:])
        mgr.add_common_command(label, command)
        print(f"Added command: {label}")
    elif action == "remove" and len(args) >= 1:
        mgr.remove_common_command(args[0])
        print(f"Removed command: {args[0]}")
    else:
        print("Usage: context cmd add <label> <command>", file=sys.stderr)
        print("       context cmd remove <label>", file=sys.stderr)
        sys.exit(1)


async def cmd_forbidden(mgr: WorkspaceContextManager, action: str, *args: str) -> None:
    """Add or remove a forbidden area."""
    if action == "add" and len(args) >= 1:
        mgr.add_forbidden_area(args[0])
        print(f"Added forbidden area: {args[0]}")
    elif action == "remove" and len(args) >= 1:
        mgr.remove_forbidden_area(args[0])
        print(f"Removed forbidden area: {args[0]}")
    else:
        print("Usage: context forbidden add <path>", file=sys.stderr)
        print("       context forbidden remove <path>", file=sys.stderr)
        sys.exit(1)


async def cmd_injection(mgr: WorkspaceContextManager, action: str) -> None:
    """Enable or disable context injection."""
    if action in ("on", "enable", "true"):
        mgr.set_injection_enabled(True)
        print("Context injection: ENABLED")
    elif action in ("off", "disable", "false"):
        mgr.set_injection_enabled(False)
        print("Context injection: DISABLED")
    else:
        print("Usage: context injection on|off", file=sys.stderr)
        sys.exit(1)


async def cmd_preview(
    mgr: WorkspaceContextManager,
    executor_id: str,
    goal: str = "",
) -> None:
    """Preview the injected context for a specific executor."""
    ctx = mgr.get_context()
    builder = PromptBuilder()

    if not ctx.context_injection_enabled:
        print("Context injection is DISABLED. No context will be injected.")
        print(f"\nPrompt would be:\n  {goal or '(no goal)'}")
        return

    preview = builder.build_injection_preview(ctx, executor_id)
    snapshot = builder.build(user_prompt=goal, context=ctx, executor_id=executor_id)

    print(f"=== Context Injection Preview ===")
    print(f"Executor:      {executor_id}")
    print(f"Est. tokens:   {snapshot.estimated_tokens}")
    print(f"Hash:          {mgr.context_hash()}")
    print()
    print("--- Context to be injected ---")
    print(preview)
    if goal:
        print(f"\n--- User prompt ---")
        print(goal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def handle_context_command(
    project_root: Path,
    args,  # argparse.Namespace
) -> None:
    """Dispatch context subcommands."""
    mgr = WorkspaceContextManager(project_root)

    sub = args.context_subcommand
    if sub is None:
        print("Usage: python -m executors.cli context <command> [...]")
        print("Commands: show, edit, adr, cmd, forbidden, injection, preview")
        sys.exit(1)

    try:
        if sub == "show":
            await cmd_show(mgr, getattr(args, "json", False))
        elif sub == "edit":
            await cmd_edit(mgr, args.field, args.value)
        elif sub == "adr":
            await cmd_adr(mgr, args.adr_action, *getattr(args, "adr_args", []))
        elif sub == "cmd":
            await cmd_command(mgr, args.cmd_action, *getattr(args, "cmd_args", []))
        elif sub == "forbidden":
            await cmd_forbidden(mgr, args.forbidden_action, *getattr(args, "forbidden_args", []))
        elif sub == "injection":
            await cmd_injection(mgr, args.injection_action)
        elif sub == "preview":
            await cmd_preview(mgr, args.executor_id, getattr(args, "goal", ""))
        else:
            print(f"Unknown context command: {sub}", file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
