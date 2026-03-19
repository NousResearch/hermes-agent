"""Context Anchors - Persistent project memory that survives compression.

Context anchors are user-defined markdown files that:
1. Auto-inject into the conversation after every compression (read path)
2. Auto-update when the agent works on the associated project (write path)

Configuration in config.yaml:

    context_anchors:
      - path: ~/.hermes/context/my-project.md
        keywords: [myproject, /var/www/myproject]
        max_chars: 5000

    context_anchors_max_total_chars: 20000
    context_anchors_auto_save: true
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MAX_CHARS_PER_ANCHOR = 5000
DEFAULT_MAX_TOTAL_CHARS = 20000
ANCHOR_INJECTION_PREFIX = (
    "[ANCHORED PROJECT CONTEXT - These files contain persistent project state "
    "that survives compression. They reflect the current ground truth. Do NOT "
    "repeat work described here. Do NOT re-debug issues marked as fixed.]\n"
)


def parse_anchor_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse context_anchors from config.yaml into a normalized list.

    Each anchor dict has:
      - path: str (expanded, absolute)
      - keywords: list[str] (lowercased)
      - max_chars: int
    """
    raw = config.get("context_anchors")
    if not raw or not isinstance(raw, list):
        return []

    anchors = []
    for entry in raw:
        if isinstance(entry, str):
            # Simple form: just a path, no keywords
            entry = {"path": entry}
        if not isinstance(entry, dict):
            continue

        path_str = entry.get("path", "")
        if not path_str:
            continue

        # Expand ~ and env vars
        expanded = os.path.expanduser(os.path.expandvars(path_str))
        resolved = str(Path(expanded).resolve())

        keywords = entry.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()]

        # Auto-derive keywords from filename if none provided
        if not keywords:
            stem = Path(resolved).stem.lower()
            keywords = [stem]

        max_chars = int(entry.get("max_chars", DEFAULT_MAX_CHARS_PER_ANCHOR))

        anchors.append({
            "path": resolved,
            "keywords": keywords,
            "max_chars": max_chars,
        })

    return anchors


def get_max_total_chars(config: Dict[str, Any]) -> int:
    """Get the global max total chars for all anchors combined."""
    return int(config.get("context_anchors_max_total_chars", DEFAULT_MAX_TOTAL_CHARS))


def is_auto_save_enabled(config: Dict[str, Any]) -> bool:
    """Check if auto-save to anchor files is enabled."""
    return bool(config.get("context_anchors_auto_save", True))


def _truncate_content(content: str, max_chars: int) -> str:
    """Truncate content with head/tail preservation (70/20 split, 10% marker)."""
    if len(content) <= max_chars:
        return content

    head_chars = int(max_chars * 0.70)
    tail_chars = int(max_chars * 0.20)

    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars > 0 else ""
    marker = (
        f"\n\n[...truncated: kept {head_chars}+{tail_chars} "
        f"of {len(content)} chars. Use file tools to read the full file.]\n\n"
    )
    return head + marker + tail


def load_anchor_file(anchor: Dict[str, Any]) -> Optional[str]:
    """Load and truncate a single anchor file. Returns None if missing/empty."""
    path = anchor["path"]
    try:
        content = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, IOError) as e:
        logger.debug("Could not read anchor file %s: %s", path, e)
        return None

    if not content:
        return None

    return _truncate_content(content, anchor["max_chars"])


def load_all_anchors(anchors: List[Dict[str, Any]], max_total: int) -> Optional[str]:
    """Load all anchor files and format for injection.

    Returns a formatted string ready to inject as a user message,
    or None if no anchors have content.
    """
    if not anchors:
        return None

    sections = []
    total_chars = 0

    for anchor in anchors:
        content = load_anchor_file(anchor)
        if not content:
            continue

        # Check total budget
        if total_chars + len(content) > max_total:
            remaining = max_total - total_chars
            if remaining > 200:  # Only include if meaningful content remains
                content = _truncate_content(content, remaining)
            else:
                logger.info(
                    "Skipping anchor %s: would exceed total char limit (%d/%d)",
                    anchor["path"], total_chars, max_total,
                )
                continue

        rel_path = anchor["path"]
        # Try to make path relative to home for readability
        try:
            rel_path = "~/" + str(Path(anchor["path"]).relative_to(Path.home()))
        except ValueError:
            pass

        sections.append(f"## {rel_path}\n\n{content}")
        total_chars += len(content)

    if not sections:
        return None

    return ANCHOR_INJECTION_PREFIX + "\n\n".join(sections)


def get_anchor_paths_for_summary(anchors: List[Dict[str, Any]]) -> List[str]:
    """Get list of anchor file paths for the compression summary prompt."""
    return [a["path"] for a in anchors if Path(a["path"]).exists()]


def detect_active_anchors(
    anchors: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    lookback: int = 20,
) -> List[Dict[str, Any]]:
    """Detect which project anchors are relevant to recent conversation.

    Scans the last `lookback` messages for keyword matches in:
    - User messages (text content)
    - Tool call arguments (file paths, URLs, commands)
    - Tool results (file contents, terminal output)

    Returns list of matching anchor configs.
    """
    if not anchors:
        return []

    # Build searchable text from recent messages
    recent = messages[-lookback:] if len(messages) > lookback else messages
    search_text_parts = []

    for msg in recent:
        content = msg.get("content", "")
        if isinstance(content, str):
            search_text_parts.append(content)
        elif isinstance(content, list):
            # Handle structured content (e.g. Anthropic format)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    search_text_parts.append(block.get("text", ""))

        # Tool calls: check function arguments
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
            else:
                args = getattr(getattr(tc, "function", None), "arguments", "")
            if isinstance(args, str):
                search_text_parts.append(args)

    search_text = "\n".join(search_text_parts).lower()

    matched = []
    for anchor in anchors:
        for keyword in anchor["keywords"]:
            if keyword in search_text:
                matched.append(anchor)
                break

    return matched


def build_anchor_save_prompt(anchor: Dict[str, Any]) -> str:
    """Build the flush prompt for auto-saving project state to an anchor file.

    Returns a system-style instruction for the model to update the anchor.
    """
    path = anchor["path"]
    keywords = ", ".join(anchor["keywords"])

    return (
        f"[System: You've been working on a project associated with {path} "
        f"(keywords: {keywords}). Before context is compressed, update that file "
        f"to reflect the current state of the project.\n"
        f"\n"
        f"RULES for updating the anchor file:\n"
        f"1. Read the file first with read_file to understand its structure.\n"
        f"2. REPLACE outdated info in-place -- do NOT append new lines about "
        f"something that already has a section. Use patch with the old text.\n"
        f"3. REMOVE references to fixed bugs, resolved issues, or completed "
        f"one-time tasks. If something is done, delete it, don't mark it done.\n"
        f"4. Keep the file organized: clear sections, no duplicates, no stale state.\n"
        f"5. The file should read as CURRENT TRUTH, not a changelog. "
        f"A reader should understand the project state without knowing its history.\n"
        f"6. Be concise and factual. No narrative, no timestamps for resolved work.]"
    )
