"""
meta_router_context.py — Pre-execution context gathering for the meta-router.

Runs context_gatherer.py BEFORE Hermes executes a task and returns a compact
preamble string to inject into the task message. This gives Hermes a head-start
with relevant file paths, wiki notes, and past outcomes rather than having it
discover everything from scratch through tool calls.

Only runs for task types that meaningfully benefit from pre-gathered context:
  research   — discovery/inventory tasks need to know WHERE to look
  audit      — security tasks need relevant file paths and past findings
  production — incident/deploy tasks benefit from known service locations

Skipped for: code, config, integration, design (narrow scope, context overhead
not worth the latency for tasks that just need to write/change specific things).

Timeout: 15 seconds hard cap — context gathering must not block Hermes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

_CONTEXT_GATHERER = Path(
    "/home/samade10/.openclaw/workspace/rql/scripts/context_gatherer.py"
)
_WORKSPACE_ROOT = Path("/home/samade10/.openclaw/workspace")

# Task types that get pre-execution context gathering
_CONTEXT_TYPES = {"research", "audit", "production"}

# Max seconds to wait for context_gatherer.py before giving up
_TIMEOUT = 15


def _format_brief(brief: dict) -> Optional[str]:
    """
    Convert context_brief.json into a compact preamble Hermes can use.
    Returns None if the brief has no useful content.
    """
    summary = brief.get("summary", {})
    if not summary.get("has_context"):
        return None

    lines: list[str] = []

    # Relevant local files
    files = summary.get("relevant_files", [])
    if files:
        lines.append("Relevant files found:")
        for f in files[:5]:
            lines.append(f"  - {f}")

    # Obsidian wiki / notes
    notes = summary.get("relevant_notes", [])
    if notes:
        lines.append("Relevant wiki/notes:")
        for n in notes[:4]:
            lines.append(f"  - {n}")

    # Chat knowledge hits (past decisions/errors from Telegram exports)
    chat_hits = summary.get("chat_knowledge_hits", [])
    if chat_hits:
        lines.append("Related past context:")
        for hit in chat_hits[:3]:
            snippet = str(hit.get("snippet") or hit.get("text") or hit)[:120]
            lines.append(f"  - {snippet}")

    # Recent REF outcomes for similar tasks
    avg_score = summary.get("avg_recent_score")
    recent_types = summary.get("recent_task_types", [])
    if avg_score is not None:
        lines.append(
            f"Recent similar task outcomes: avg score {avg_score:.0f}/100"
            + (f" (types: {', '.join(recent_types)})" if recent_types else "")
        )

    # Recent memory files (Hermes memory: USER.md, MEMORY.md)
    sources = brief.get("sources", {})
    mem = sources.get("recent_memory", {})
    mem_days = mem.get("days", [])
    if mem_days:
        # Just note how many days of memory are available
        lines.append(f"Hermes memory: {len(mem_days)} recent memory day(s) available")

    if not lines:
        return None

    return "\n".join(lines)


def gather_pre_execution_context(
    task_text: str,
    task_type: str,
    state_dir: Optional[Path],
    memory_need: Optional[str] = None,
    required_tools: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Run context_gatherer.py before Hermes executes the task.

    Returns a formatted context preamble string to inject into the task
    message, or None if context gathering was skipped, timed out, or
    produced nothing useful.

    Args:
        task_text:  Raw task text (the original user message).
        task_type:  Classified task type from meta_router.
        state_dir:  SoM state directory (used as output dir for context_brief.json).
                    If None, gathering is skipped.
    """
    required = set(required_tools or [])
    needs_retrieval = bool(required.intersection({"memory_search", "qmd__query"})) or (memory_need in {"history", "docs", "history+docs", "wiki+history"})
    if task_type not in _CONTEXT_TYPES and not needs_retrieval:
        return None
    if not _CONTEXT_GATHERER.exists():
        return None
    if state_dir is None:
        return None

    output_dir = str(state_dir)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_CONTEXT_GATHERER),
                "--task", task_text[:2000],
                "--task-type", task_type if task_type in {
                    "code", "research", "design", "config", "documentation"
                } else "research",
                "--output-dir", output_dir,
                "--workspace", str(_WORKSPACE_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    context_path = Path(output_dir) / "context_brief.json"
    if not context_path.exists():
        return None

    try:
        brief = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return _format_brief(brief)
