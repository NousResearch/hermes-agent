"""
Continuum Plugin — DAG + Daily Journal for Hermes Agent.

This plugin is NOT a memory backend. It is a continuity layer that:

1. Captures session_shards at session-end via on_session_end hook
2. Provides tools for DAG provenance lookup (dag_trace, dag_describe)
3. Detects model switches
4. Injects continuity packet into user messages via pre_llm_call hook
5. Provides promotion candidate identification

Compression (daily/weekly/monthly) is handled by external cron scripts.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_plugin_ctx = None
_initialized = False
_last_model_used: Optional[str] = None
_injected_session_ids: Set[str] = set()


def register(ctx) -> None:
    """Plugin entry point — called by Hermes plugin system."""
    global _plugin_ctx, _initialized

    _plugin_ctx = ctx
    manifest_name = ctx.manifest.name
    logger.info("Continuity plugin registering: %s v%s", manifest_name, ctx.manifest.version)

    # ---- Initialize DB ----
    try:
        from plugins.continuity.db import migrate, get_db_path
        migrate()
        logger.info("Continuity DB ready at %s", get_db_path())
    except Exception as exc:
        logger.warning("Continuity DB init failed (non-fatal): %s", exc)

    # ---- Register slash commands ----
    ctx.register_command(
        name="continuity",
        handler=_cmd_continuity,
        description="Continuity DAG commands: status, trace <node_id>, promote, sweep",
        args_hint="status | trace <node_id> | promote | sweep",
    )

    # ---- Register tools ----
    ctx.register_tool(
        name="dag_trace",
        toolset="hermes",
        schema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node ID to trace provenance for",
                },
            },
            "required": ["node_id"],
        },
        handler=_handle_dag_trace,
        description="Trace a DAG node down to its source session_shards, showing the full provenance chain.",
        is_async=False,
    )

    ctx.register_tool(
        name="dag_describe",
        toolset="hermes",
        schema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node ID to describe",
                },
            },
            "required": ["node_id"],
        },
        handler=_handle_dag_describe,
        description="Describe a DAG node: its metadata, parents, children, and token accounting.",
        is_async=False,
    )

    ctx.register_tool(
        name="dag_promote",
        toolset="hermes",
        schema={
            "type": "object",
            "properties": {
                "session_scope": {
                    "type": "string",
                    "description": "Scope: 'recent' (last 7 days) or 'all'",
                    "default": "recent",
                },
            },
            "required": [],
        },
        handler=_handle_dag_promote,
        description="Identify promotion candidates (relational fragments corroborated across ≥3 sessions).",
        is_async=False,
    )

    # ---- Register hooks ----
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _pre_llm_call)

    _initialized = True
    logger.info("Continuity plugin registered successfully")


# ---------------------------------------------------------------------------
# Hook: on_session_end — capture session_shard
# ---------------------------------------------------------------------------

def _on_session_end(**kwargs: Any) -> None:
    """Capture session metadata as a session_shard node."""
    try:
        from plugins.continuity.db import upsert_node, migrate, get_db_path

        session_id = kwargs.get("session_id", "")
        model = kwargs.get("model", "")
        platform = kwargs.get("platform", "")
        completed = kwargs.get("completed", True)
        interrupted = kwargs.get("interrupted", False)

        if not session_id:
            return

        today = date.today().isoformat()
        provider = model.split("/")[0] if "/" in model else "unknown"

        upsert_node(
            node_type="session_shard",
            date_key=today,
            title=f"Session {session_id[:12]}...",
            token_count=0,
            compression_depth=0,
            provider=provider,
            model=model,
            source_session_id=session_id,
            author_mode="system",
        )
        logger.debug("Continuity: captured session_shard for %s", session_id)

    except Exception as exc:
        logger.debug("Continuity on_session_end hook: %s", exc)


# ---------------------------------------------------------------------------
# Hook: on_session_start — track injection state per session
# ---------------------------------------------------------------------------

def _on_session_start(**kwargs: Any) -> None:
    """Track model info and mark session for potential injection."""
    global _last_model_used
    session_id = kwargs.get("session_id", "")
    model = kwargs.get("model", "")

    if session_id:
        # Clear injection tracking for fresh session
        _injected_session_ids.discard(session_id)
        # Track model
        if model:
            _last_model_used = model


# ---------------------------------------------------------------------------
# Hook: pre_llm_call — inject continuity packet
# ---------------------------------------------------------------------------

def _pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Inject continuity packet at the start of a session.

    Uses pre_llm_call to inject context into the user message.
    Only injects once per session (first turn).
    """
    session_id = kwargs.get("session_id", "")

    if not session_id:
        return None

    # Only inject once per session
    if session_id in _injected_session_ids:
        return None
    _injected_session_ids.add(session_id)

    try:
        packet = _build_continuity_packet(session_id)
        if not packet:
            return None

        logger.debug("Continuity: injecting packet for session %s", session_id[:12])
        return {"context": packet}
    except Exception as exc:
        logger.debug("Continuity pre_llm_call: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Continuity packet assembly
# ---------------------------------------------------------------------------

TOKEN_BUDGET = {
    "self_model": 1500,
    "memory_relation": 2000,
    "memory_factual": 1000,
    "daily_journals": 3000,
    "weekly": 1500,
    "monthly": 800,
    "open_threads": 1200,
    "handoff": 1000,
    "total": 12000,
}

JOURNAL_ROOT_NAME = "journal"

# ---------------------------------------------------------------------------
# Path validation — security
# ---------------------------------------------------------------------------

_CONTINUITY_ROOT_CACHE = None


def _get_continuity_root() -> Path:
    """Get the resolved continuity journal root path."""
    global _CONTINUITY_ROOT_CACHE
    if _CONTINUITY_ROOT_CACHE is None:
        from plugins.continuity.db import get_continuity_home
        root = get_continuity_home() / JOURNAL_ROOT_NAME
        _CONTINUITY_ROOT_CACHE = root.resolve()
    return _CONTINUITY_ROOT_CACHE


def _is_path_safe(path_str: str) -> bool:
    """Validate that a file path is within the continuity journal root.

    Resolves symlinks and '..' traversal before checking.
    Returns True if the path is safe to read, False otherwise.
    """
    if not path_str:
        return False
    try:
        target = Path(path_str).resolve()
        root = _get_continuity_root().resolve()
        return root in target.parents or root == target
    except (OSError, ValueError, RuntimeError):
        return False


def _validate_and_read(path_str: str) -> Optional[str]:
    """Safely read a file if it's within the continuity journal root.

    Returns None if the path is unsafe or the file doesn't exist.
    """
    if not _is_path_safe(path_str):
        logger.warning("Blocked read of unsafe path: %s", path_str)
        return None
    try:
        return Path(path_str).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError, FileNotFoundError):
        return None


DAILY_TOKEN_CAP = 1200
WEEKLY_TOKEN_CAP = 650
MONTHLY_TOKEN_CAP = 400


def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for CJK/English mix."""
    return max(1, len(text) // 4)


def _truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget."""
    max_chars = budget * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated to fit budget]"


def _get_hermes_home() -> Path:
    """Get the Hermes home directory."""
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _find_memory_file() -> Optional[Path]:
    """Find MEMORY.md by checking multiple candidate paths.

    Priority:
    1. HERMES_HOME/MEMORY.md
    2. HERMES_HOME/memories/MEMORY.md
    3. HERMES_HOME/memory/MEMORY.md
    """
    hermes_home = _get_hermes_home()
    candidates = [
        hermes_home / "MEMORY.md",
        hermes_home / "memories" / "MEMORY.md",
        hermes_home / "memory" / "MEMORY.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _build_continuity_packet(session_id: str) -> Optional[str]:
    """Build the continuity packet for session-start injection.

    Returns a formatted string ≤ ~12k tokens, or None if nothing to inject.
    """
    from plugins.continuity.db import (
        get_latest_node,
        get_recent_daily_journals,
        get_current_weekly,
        get_current_monthly,
    )

    sections = []
    total_est_tokens = 0

    # 1. SELF_MODEL excerpt (~1,500 tokens)
    self_model_path = _get_hermes_home() / "SELF_MODEL.md"
    if self_model_path.exists():
        text = _truncate_to_budget(self_model_path.read_text(encoding="utf-8", errors="replace"), 1500)
        sections.append(("## Identity Context (SELF_MODEL)", text))
        total_est_tokens += _count_tokens(text)
    else:
        sections.append(("## Identity Context (SELF_MODEL)", "(No SELF_MODEL.md found)"))

    # 2. MEMORY relation layer excerpt (~2,000 tokens)
    memory_file = _find_memory_file()
    memory_relation_text = ""
    if memory_file is not None:
        content = memory_file.read_text(encoding="utf-8", errors="replace")
        # Extract the relationship layer section
        rel_section = _extract_relation_layer(content)
        memory_relation_text = _truncate_to_budget(rel_section, 2000)
    if memory_relation_text:
        sections.append(("## MEMORY Relation Layer", memory_relation_text))
        total_est_tokens += _count_tokens(memory_relation_text)

    # 3. Recent daily journals (1-3, ~3,000 tokens)
    journals = get_recent_daily_journals(days=7)
    journal_texts = []
    journal_tokens = 0
    for j in journals[:3]:
        path_str = j.get("markdown_path", "")
        content = _validate_and_read(path_str)
        if content:
            capped = _truncate_to_budget(content, DAILY_TOKEN_CAP)
            journal_texts.append(f"**{j.get('title', j['date_key'])}**:\n{capped}")
            journal_tokens += _count_tokens(capped)
    if journal_texts:
        sections.append(("## Recent Daily Journals", "\n---\n".join(journal_texts[:3])))
        total_est_tokens += journal_tokens

    # 4. Weekly summary (~1,500 tokens)
    wk = get_current_weekly()
    if wk:
        wk_content = _validate_and_read(wk.get("markdown_path", ""))
        if wk_content:
            text = _truncate_to_budget(wk_content, 1500)
            sections.append(("## Weekly Summary", text))
            total_est_tokens += _count_tokens(text)

    # 5. Monthly summary (~800 tokens)
    mo = get_current_monthly()
    if mo:
        mo_content = _validate_and_read(mo.get("markdown_path", ""))
        if mo_content:
            text = _truncate_to_budget(mo_content, 800)
            sections.append(("## Monthly Summary", text))
            total_est_tokens += _count_tokens(text)

    # 6. Model-switch handoff
    global _last_model_used
    current_model = os.environ.get("HERMES_DEFAULT_MODEL", "")
    if _last_model_used and current_model and _last_model_used != current_model:
        handoff = (
            f"**Model Transition Note:** Underlying model changed from "
            f"`{_last_model_used}` to `{current_model}`. "
            f"Identity continuity is anchored to SOUL.md > SELF_MODEL.md > MEMORY relation layer. "
            f"Journal continuity below provides recent temporal context. "
            f"Do not reinterpret identity based on model change."
        )
        sections.append(("## Model Transition", handoff))
        total_est_tokens += _count_tokens(handoff)
        _last_model_used = current_model

    # 7. Open threads (~1,200 tokens)
    from plugins.continuity.db import get_open_threads
    threads = get_open_threads()
    if threads:
        thread_lines = []
        for t in threads[:5]:
            thread_lines.append(f"- {t.get('title', t['date_key'])} (node: `{t['node_id'][:12]}...`)")
        thread_text = "\n".join(thread_lines)
        thread_text = _truncate_to_budget(thread_text, 1200)
        sections.append(("## Open Threads", thread_text))
        total_est_tokens += _count_tokens(thread_text)

    # Budget enforcement: crop from bottom (daily → weekly → monthly) if over
    budget_remaining = TOKEN_BUDGET["total"] - total_est_tokens
    if budget_remaining < 0:
        # Crop in reverse order: remove recent sections from the end
        while sections and budget_remaining < 0:
            removed = sections.pop()
            budget_remaining += _count_tokens(removed[1])
        logger.info("Continuity packet cropped: %d tokens over budget", -budget_remaining)

    if not sections:
        return None

    # Assemble final packet
    lines = [
        "## Continuity Packet",
        f"_Injected by continuity plugin | Token budget: ~{total_est_tokens}/{TOKEN_BUDGET['total']}_",
        "",
    ]
    for heading, content in sections:
        lines.append(heading)
        lines.append("")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def _extract_relation_layer(content: str) -> str:
    """Extract the relationship layer (relational memory) from MEMORY.md.

    Looks for 【關係層】 or [Relational] section markers.
    """
    import re
    # Try to find relationship sections
    patterns = [
        r"【關係層.*?】(.*?)(?=【|§|---|$)",
        r"\[Relational.*?\](.*?)(?=\[|---|$)",
        r"relation.*?layer.*?\n(.*?)(?=\n##|\n---|$)",
    ]
    for pat in patterns:
        match = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: return the full content but capped
    return _truncate_to_budget(content, 2000)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_dag_trace(args: Dict[str, Any]) -> str:
    """Trace a DAG node to its source session_shards."""
    from plugins.continuity.db import trace_provenance, get_node

    node_id = args.get("node_id", "")
    if not node_id:
        return json.dumps({"error": "node_id is required"})

    node = get_node(node_id)
    if not node:
        return json.dumps({"error": f"Node not found: {node_id}"})

    sources = trace_provenance(node_id)

    return json.dumps({
        "node": {
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "date_key": node["date_key"],
            "title": node.get("title", ""),
            "token_count": node.get("token_count", 0),
            "compression_depth": node.get("compression_depth", 0),
        },
        "source_count": len(sources),
        "sources": [
            {
                "node_id": s["node_id"],
                "date_key": s["date_key"],
                "model": s.get("model", ""),
                "source_session_id": s.get("source_session_id", ""),
            }
            for s in sources
        ],
    }, ensure_ascii=False, indent=2)


def _handle_dag_describe(args: Dict[str, Any]) -> str:
    """Describe a DAG node with its metadata, parents, and children."""
    from plugins.continuity.db import get_node, get_children, get_parents, trace_provenance

    node_id = args.get("node_id", "")
    if not node_id:
        return json.dumps({"error": "node_id is required"})

    node = get_node(node_id)
    if not node:
        return json.dumps({"error": f"Node not found: {node_id}"})

    parents = get_parents(node_id)
    children = get_children(node_id)

    result = {
        "node": {
            k: node[k] for k in [
                "node_id", "node_type", "date_key", "title", "token_count",
                "compression_depth", "provider", "model", "source_session_id",
                "author_mode", "operational_tokens", "relational_tokens",
            ] if k in node
        },
        "created_at": node.get("created_at", ""),
        "parents": [
            {"node_id": p["node_id"], "type": p["node_type"], "date_key": p["date_key"]}
            for p in parents
        ],
        "children": [
            {"node_id": c["node_id"], "type": c["node_type"], "date_key": c["date_key"]}
            for c in children
        ],
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


def _handle_dag_promote(args: Dict[str, Any]) -> str:
    """Identify promotion candidates: relational fragments corroborated ≥3 sessions."""
    from plugins.continuity.db import get_nodes_by_type

    candidates = _find_promotion_candidates()
    return json.dumps({
        "candidates": candidates,
        "count": len(candidates),
        "note": "Manual review required before promotion to MEMORY.md relation layer.",
    }, ensure_ascii=False, indent=2)


def _find_promotion_candidates() -> List[Dict[str, Any]]:
    """Scan recent daily journals for patterns that appear across ≥3 sessions.

    This is a heuristic tool — it flags potential candidates for human review,
    not automatic promotion.
    """
    from plugins.continuity.db import get_nodes_by_type, get_latest_node

    journals = get_nodes_by_type("daily", limit=14)
    if len(journals) < 3:
        return []

    observed_patterns: Dict[str, List[str]] = {}

    for j in journals:
        # Read the journal markdown content for relational section
        content = _validate_and_read(j.get("markdown_path", ""))
        if not content:
            continue

        # Extract relational continuity section
        rel_section = _extract_relational_from_journal(content)
        if not rel_section:
            continue

        # Simple heuristic: flag if same topic mentioned across ≥3 sessions
        lines = rel_section.lower().split("\n")
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            # Use first 40 chars as pattern key
            key = line[:60].strip()
            if key not in observed_patterns:
                observed_patterns[key] = []
            if j["node_id"] not in observed_patterns[key]:
                observed_patterns[key].append(j["date_key"])

    results = []
    for pattern, session_dates in observed_patterns.items():
        if len(session_dates) >= 3:
            results.append({
                "pattern_preview": pattern[:80],
                "session_count": len(session_dates),
                "session_dates": sorted(session_dates),
            })

    results.sort(key=lambda x: x["session_count"], reverse=True)
    return results[:10]


def _extract_relational_from_journal(content: str) -> str:
    """Extract the Relational Continuity section from a daily journal."""
    import re
    for marker in ["## Relational Continuity", "### Relational", "[Relational]"]:
        if marker in content:
            parts = content.split(marker, 1)
            if len(parts) > 1:
                rest = parts[1]
                # Take up to next section heading
                end = re.search(r"\n## ", rest)
                if end:
                    return rest[:end.start()]
                return rest.strip()
    return ""


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

def _cmd_continuity(raw_args: str) -> str:
    """Handle /continuity slash command."""
    from plugins.continuity.db import (
        integrity_sweep, get_latest_node, get_total_token_count,
        trace_provenance, get_node,
    )

    args = raw_args.strip().split()
    if not args:
        return (
            "Continuity plugin commands:\n"
            "  `/continuity status` — DB overview\n"
            "  `/continuity trace <node_id>` — provenance trace\n"
            "  `/continuity describe <node_id>` — node details\n"
            "  `/continuity promote` — find promotion candidates\n"
            "  `/continuity sweep` — integrity check"
        )

    cmd = args[0]

    if cmd == "status":
        total_nodes = get_total_token_count()
        latest_daily = get_latest_node("daily")
        latest_weekly = get_latest_node("weekly")
        latest_monthly = get_latest_node("monthly")
        return (
            f"**Continuity Status**\n"
            f"- DB: `{get_db_path() if 'get_db_path' in dir() else 'continuity.db'}`\n"
            f"- Total tokens stored: {total_nodes}\n"
            f"- Latest daily: {latest_daily['date_key'] if latest_daily else 'none'}\n"
            f"- Latest weekly: {latest_weekly['date_key'] if latest_weekly else 'none'}\n"
            f"- Latest monthly: {latest_monthly['date_key'] if latest_monthly else 'none'}\n"
        )

    elif cmd == "trace" and len(args) >= 2:
        node_id = args[1]
        node = get_node(node_id)
        if not node:
            return f"Node not found: {node_id}"
        sources = trace_provenance(node_id)
        return (
            f"**Provenance trace for {node_id[:16]}...**\n"
            f"Type: {node['node_type']} | Date: {node['date_key']}\n"
            f"Depth: {node['compression_depth']} | Tokens: {node['token_count']}\n"
            f"Sources found: {len(sources)}\n" +
            "\n".join(f"  - `{s['node_id'][:12]}...` ({s['date_key']}, {s.get('model', '?')})" for s in sources)
        )

    elif cmd == "describe" and len(args) >= 2:
        return _handle_dag_describe({"node_id": args[1]})

    elif cmd == "promote":
        candidates = _find_promotion_candidates()
        if not candidates:
            return "No promotion candidates found (need ≥3 sessions with recurring relational patterns)."
        lines = ["**Promotion Candidates** (manual review required):", ""]
        for c in candidates:
            lines.append(f"- \"{c['pattern_preview']}\"")
            lines.append(f"  Appeared in {c['session_count']} sessions: {', '.join(c['session_dates'])}")
        return "\n".join(lines)

    elif cmd == "sweep":
        findings = integrity_sweep()
        return (
            f"**Integrity Sweep**\n"
            f"- Total nodes: {findings['total_nodes']}\n"
            f"- Total edges: {findings['total_edges']}\n"
            f"- Orphaned nodes: {len(findings['orphaned_nodes'])}\n"
            f"- Broken provenance: {len(findings['broken_provenance'])}\n"
            f"- Token drift: {len(findings['token_drift'])}\n"
            + ("\nOrphaned:\n" + "\n".join(
                f"  - `{o['node_id'][:12]}...` ({o['node_type']}, {o['date_key']})"
                for o in findings['orphaned_nodes']
            ) if findings['orphaned_nodes'] else "")
            + ("\nBroken provenance:\n" + "\n".join(
                f"  - `{b['node_id'][:12]}...` ({b['node_type']}, {b['date_key']})"
                for b in findings['broken_provenance']
            ) if findings['broken_provenance'] else "")
        )

    return "Unknown command. Use `/continuity` for help."


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    """Expose DB path for external scripts."""
    try:
        from plugins.continuity.db import get_db_path as _db_path
        return _db_path()
    except Exception:
        return Path.home() / ".hermes" / "continuum" / "continuity.db"


def get_continuity_home() -> Path:
    """Expose continuity home for external scripts."""
    try:
        from plugins.continuity.db import get_continuity_home as _ch
        return _ch()
    except Exception:
        return Path.home() / ".hermes" / "continuum"
