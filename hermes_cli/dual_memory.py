"""CLI helpers for the dual memory framework."""

from __future__ import annotations

import sys
from pathlib import Path

from agent.dual_memory import (
    PARA_BUCKETS,
    PersonalWorkspace,
    ProceduralMemory,
    SkillDraft,
    WorkspaceItem,
    default_procedural_skills_root,
    default_workspace_root,
    filter_workspace_candidate,
    format_retrieval_results,
)


def _read_content(args) -> str:
    file_value = getattr(args, "file", None)
    if file_value:
        return Path(file_value).expanduser().read_text(encoding="utf-8")
    parts = getattr(args, "content", None) or []
    if parts:
        return " ".join(parts)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def cmd_workspace(args) -> None:
    """Handle ``hermes memory workspace ...``."""
    action = getattr(args, "workspace_command", None)
    workspace = PersonalWorkspace()

    if action == "init":
        workspace.initialize()
        print(f"\n  Personal workspace initialized: {default_workspace_root()}\n")
        return

    if action == "add":
        content = _read_content(args).strip()
        if not filter_workspace_candidate(content):
            print("\n  Skipped: content is too short or too transient for workspace memory.\n")
            return
        bucket = getattr(args, "bucket", None) or None
        if bucket and bucket not in PARA_BUCKETS:
            print(f"\n  Unknown PARA bucket: {bucket}\n")
            return
        item = WorkspaceItem(
            title=getattr(args, "title", "").strip(),
            content=content,
            bucket=bucket,
            summary=(getattr(args, "summary", "") or "").strip(),
            tags=list(getattr(args, "tag", []) or []),
            backlinks=list(getattr(args, "backlink", []) or []),
            status_hint=(getattr(args, "status_hint", "") or "").strip(),
        )
        path = workspace.write_item(item, mode=getattr(args, "mode", "new"))
        print(f"\n  Wrote workspace item: {path}\n")
        return

    if action == "search":
        query = getattr(args, "query", "").strip()
        results = workspace.retrieve(query, top_k=getattr(args, "top_k", 3))
        if not results:
            print("\n  No workspace matches.\n")
            return
        print()
        print(format_retrieval_results(results))
        print()
        return

    print("\n  Usage: hermes memory workspace {init,add,search}\n")


def cmd_procedural(args) -> None:
    """Handle ``hermes memory procedural ...``."""
    action = getattr(args, "procedural_command", None)
    procedural = ProceduralMemory()

    if action == "distill":
        steps = list(getattr(args, "step", []) or [])
        triggers = list(getattr(args, "trigger", []) or [])
        constraints = list(getattr(args, "constraint", []) or [])
        recovery = list(getattr(args, "recovery", []) or [])
        source = _read_content(args).strip()
        draft = SkillDraft(
            name=getattr(args, "name", "").strip(),
            description=getattr(args, "description", "").strip(),
            triggers=triggers,
            steps=steps,
            constraints=constraints,
            recovery=recovery,
            source=source,
        )
        try:
            path = procedural.write_skill(draft, overwrite=getattr(args, "overwrite", False))
        except FileExistsError as exc:
            print(f"\n  {exc}\n  Re-run with --overwrite to replace it.\n")
            return
        print(f"\n  Wrote procedural skill: {path}")
        print(f"  Skills root: {default_procedural_skills_root()}\n")
        return

    print("\n  Usage: hermes memory procedural distill ...\n")
