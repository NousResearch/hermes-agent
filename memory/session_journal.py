"""Lightweight session journal — JSONL turns → tagged markdown.

Zero LLM calls. Rule-based tag extraction from tool names and content patterns.
Writes to ~/wiki/session-recordings/{year}-W{week}/ for easy browsing.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from memory.config import SESSIONS_DIR

logger = logging.getLogger(__name__)

# ── Tag rules: (pattern_source, pattern, tag) ───────────────────────────────
# Patterns are matched against tool_name (if present) or lowercased content.
_TOOL_TAG_RULES: List[Tuple[str, str, str]] = [
    # Git / GitHub
    ("tool", r"github|git_", "#github"),
    ("tool", r"git_", "#git"),
    # Build / Test
    ("tool", r"terminal", "#terminal"),
    ("content", r"\bpytest\b|\bnpm test\b|\bjest\b|\bgo test\b|\brusttest\b", "#test"),
    ("content", r"\bbuild\b|\bcompile\b|\bmake\b|\bnpm run\b|\bdocker build\b", "#build"),
    # Web / Research
    ("tool", r"web_search|web_extract", "#research"),
    ("tool", r"browser", "#browser"),
    # Files / Code
    ("tool", r"write_file|patch|read_file", "#code"),
    ("tool", r"execute_code", "#python"),
    # Media / Creative
    ("tool", r"image_generate|text_to_speech", "#media"),
    # Infrastructure / DevOps
    ("tool", r"cronjob|terminal", "#devops"),
    # Communication
    ("tool", r"send_message|telegram|discord|slack", "#comms"),
    # Memory / Search
    ("tool", r"memory_|session_search", "#memory"),
    # Delegation
    ("tool", r"delegate", "#delegation"),
    # Docs / Productivity
    ("tool", r"google_workspace|gws_|notion|linear", "#docs"),
]

_CONTENT_TAG_RULES: List[Tuple[str, str]] = [
    (r"\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\bfailure\b", "#debug"),
    (r"\bfix\b|\bbug\b|\bresolve\b|\bworkaround\b|\bpatch\b", "#debug"),
    (r"\bdeploy\b|\brelease\b|\bship\b|\blaunch\b", "#deploy"),
    (r"\bplan\b|\broa(d|d)map\b|\bstrategy\b|\barchitecture\b", "#planning"),
    (r"\breview\b|\bcode review\b|\bpr\b|\bpull request\b", "#review"),
    (r"\btest\b|\btesting\b|\bspec\b", "#test"),
    (r"\bdocument\b|\bdoc\b|\breadme\b|\bwiki\b", "#docs"),
    (r"\bmeeting\b|\bcall\b|\bdiscuss\b|\bstandup\b", "#meeting"),
]

# ── Output directory ────────────────────────────────────────────────────────
JOURNAL_DIR = Path.home() / "wiki" / "session-recordings"


def _iso_week_folder(dt: datetime) -> Path:
    """Return {JOURNAL_DIR}/{year}-W{week}/ for a given datetime."""
    year, week, _ = dt.isocalendar()
    return JOURNAL_DIR / f"{year}-W{week:02d}"


def _extract_tags(turns: List[dict]) -> Set[str]:
    """Extract tags from tool names and content patterns."""
    tags: Set[str] = set()
    tool_counts: Dict[str, int] = {}

    for turn in turns:
        tool_name = (turn.get("tool_name") or "").lower()
        tool_calls_list = turn.get("tool_calls") or []
        content = (turn.get("content") or "").lower()
        role = turn.get("role", "")

        # Collect all tool names from this turn
        turn_tool_names = []
        if tool_name:
            turn_tool_names.append(tool_name)
        for tc in tool_calls_list:
            if isinstance(tc, dict):
                tc_name = (tc.get("name") or tc.get("function", {}).get("name") or "").lower()
                if tc_name:
                    turn_tool_names.append(tc_name)

        # Count tools
        for name in turn_tool_names:
            tool_counts[name] = tool_counts.get(name, 0) + 1

        # Tool-based tags
        for source, pattern, tag in _TOOL_TAG_RULES:
            if source == "tool":
                for name in turn_tool_names:
                    if re.search(pattern, name):
                        tags.add(tag)
            elif source == "content" and content:
                if re.search(pattern, content):
                    tags.add(tag)

        # Content-based tags (assistant turns only, to avoid tagging user chitchat)
        if role == "assistant" and content:
            for pattern, tag in _CONTENT_TAG_RULES:
                if re.search(pattern, content):
                    tags.add(tag)

    # Heavy session heuristic
    total_tools = sum(tool_counts.values())
    if total_tools >= 10:
        tags.add("#heavy-session")
    elif total_tools >= 5:
        tags.add("#active-session")

    return tags


def _summarize_turns(turns: List[dict]) -> Tuple[str, List[dict]]:
    """Build a condensed timeline of turns for the markdown body.

    Returns (summary_paragraph, tool_call_log).
    """
    user_turns = 0
    asst_turns = 0
    tool_calls: List[dict] = []
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    for turn in turns:
        role = turn.get("role", "")
        ts = turn.get("timestamp")
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if role == "user":
            user_turns += 1
        elif role == "assistant":
            asst_turns += 1

        tool_name = turn.get("tool_name")
        tool_calls_list = turn.get("tool_calls") or []

        if tool_name:
            tool_calls.append({
                "tool": tool_name,
                "role": role,
                "time": ts,
                "preview": (turn.get("content") or "")[:80].replace("\n", " "),
            })
        for tc in tool_calls_list:
            if isinstance(tc, dict):
                tc_name = tc.get("name") or tc.get("function", {}).get("name") or "unknown"
                tool_calls.append({
                    "tool": tc_name,
                    "role": role,
                    "time": ts,
                    "preview": (turn.get("content") or "")[:80].replace("\n", " "),
                })

    duration_mins = 0
    if first_ts is not None and last_ts is not None:
        duration_mins = int((last_ts - first_ts) / 60)

    summary = (
        f"{user_turns} user turns, {asst_turns} assistant turns, "
        f"{len(tool_calls)} tool calls"
    )
    if duration_mins > 0:
        summary += f", ~{duration_mins} min duration"

    return summary, tool_calls


def _build_markdown(
    session_id: str,
    turns: List[dict],
    tags: Set[str],
    summary: str,
    tool_calls: List[dict],
    platform: str = "",
) -> str:
    """Assemble the markdown journal page."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M UTC")
    tag_line = " ".join(sorted(tags)) if tags else "#untagged"

    lines = [
        f"# Session: {session_id}",
        "",
        f"**Date:** {date_str}",
    ]
    if platform:
        lines.append(f"**Platform:** {platform}")
    lines.extend([
        f"**Tags:** {tag_line}",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Tool Calls",
        "",
    ])

    if tool_calls:
        seen_tools: Dict[str, int] = {}
        for tc in tool_calls:
            name = tc["tool"]
            seen_tools[name] = seen_tools.get(name, 0) + 1

        for name, count in sorted(seen_tools.items(), key=lambda x: -x[1]):
            lines.append(f"- `{name}` × {count}")
    else:
        lines.append("_No tool calls in this session._")

    lines.extend([
        "",
        "## Turn Log",
        "",
    ])

    # Condensed log: first line of each user message + tool call markers
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "") or ""
        tool_name = turn.get("tool_name")
        tool_calls_list = turn.get("tool_calls") or []

        # Show tool calls from assistant turns
        shown_tools = []
        if tool_name:
            shown_tools.append(tool_name)
        for tc in tool_calls_list:
            if isinstance(tc, dict):
                tc_name = tc.get("name") or tc.get("function", {}).get("name")
                if tc_name:
                    shown_tools.append(tc_name)

        if role == "user":
            preview = content.split("\n")[0][:100]
            if len(content) > 100:
                preview += "…"
            lines.append(f"**U:** {preview}")
        elif role == "assistant" and shown_tools:
            tools_str = ", ".join(f"`{t}`" for t in shown_tools)
            lines.append(f"**A → {tools_str}:** {(content or '')[:60].replace(chr(10), ' ')}")
        elif role == "assistant":
            preview = content.split("\n")[0][:100]
            if len(content) > 100:
                preview += "…"
            lines.append(f"**A:** {preview}")
        elif role == "tool":
            lines.append(f"**T → `{tool_name}`:** {(content or '')[:60].replace(chr(10), ' ')}")

    lines.append("")
    return "\n".join(lines)


def write_session_journal(
    session_id: str,
    turns: Optional[List[dict]] = None,
    platform: str = "",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write a tagged markdown journal for a session.

    Args:
        session_id: The session identifier.
        turns: Optional pre-loaded turn list. If None, reads from JSONL.
        platform: Source platform (telegram, discord, etc.).
        output_dir: Override output directory. Defaults to JOURNAL_DIR.

    Returns:
        Path to the written file, or None if skipped (empty session).
    """
    output_dir = output_dir or JOURNAL_DIR

    # Load turns from JSONL if not provided
    if turns is None:
        jsonl_path = SESSIONS_DIR / f"{session_id}.jsonl"
        if not jsonl_path.exists():
            logger.debug("No JSONL found for session %s, skipping journal", session_id)
            return None
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                turns = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error("Failed to read JSONL for session %s: %s", session_id, e)
            return None

    # Skip empty or nearly-empty sessions
    if not turns:
        return None

    # Count meaningful turns (exclude skill-review auto-prompts)
    meaningful = 0
    for turn in turns:
        content = turn.get("content", "") or ""
        if "Review the conversation above and consider saving or updating a skill" not in content:
            meaningful += 1
    if meaningful < 2:
        logger.debug("Session %s has <2 meaningful turns, skipping journal", session_id)
        return None

    tags = _extract_tags(turns)
    summary, tool_calls = _summarize_turns(turns)

    now = datetime.now(timezone.utc)
    folder = _iso_week_folder(now)
    folder.mkdir(parents=True, exist_ok=True)

    safe_sid = session_id.replace("/", "-").replace(":", "-")[:40]
    filename = f"{now.strftime('%Y-%m-%d')}_{safe_sid}.md"
    filepath = folder / filename

    md = _build_markdown(session_id, turns, tags, summary, tool_calls, platform=platform)

    try:
        filepath.write_text(md, encoding="utf-8")
        logger.info("Session journal written: %s (%d turns, tags: %s)", filepath.name, len(turns), " ".join(sorted(tags)) or "none")
        return filepath
    except Exception as e:
        logger.error("Failed to write session journal: %s", e)
        return None


def write_journal_for_jsonl(jsonl_path: Path, platform: str = "", output_dir: Optional[Path] = None) -> Optional[Path]:
    """Convenience entry point: pass a JSONL file path directly.

    Useful for cron jobs or batch reprocessing.
    """
    session_id = jsonl_path.stem
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            turns = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.error("Failed to read %s: %s", jsonl_path, e)
        return None
    return write_session_journal(session_id, turns=turns, platform=platform, output_dir=output_dir)
