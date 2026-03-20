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

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MAX_CHARS_PER_ANCHOR = 5000
DEFAULT_MAX_TOTAL_CHARS = 20000
PRE_FLUSH_THRESHOLD = 0.70  # trigger pre-flush nudge at 70% of compression threshold
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
        try:
            expanded = os.path.expanduser(os.path.expandvars(path_str))
            resolved = str(Path(expanded).resolve())
        except (OSError, ValueError):
            continue

        keywords = entry.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()]

        # Auto-derive keywords from filename if none provided
        if not keywords:
            stem = Path(resolved).stem.lower()
            keywords = [stem]

        try:
            max_chars = int(entry.get("max_chars", DEFAULT_MAX_CHARS_PER_ANCHOR))
        except (ValueError, TypeError):
            max_chars = DEFAULT_MAX_CHARS_PER_ANCHOR

        anchors.append({
            "path": resolved,
            "keywords": keywords,
            "max_chars": max_chars,
        })

    return anchors


def get_max_total_chars(config: Dict[str, Any]) -> int:
    """Get the global max total chars for all anchors combined."""
    try:
        return int(config.get("context_anchors_max_total_chars", DEFAULT_MAX_TOTAL_CHARS))
    except (ValueError, TypeError):
        return DEFAULT_MAX_TOTAL_CHARS


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
    except (OSError, IOError, UnicodeDecodeError) as e:
        logger.debug("Could not read anchor file %s: %s", path, e)
        return None

    if not content:
        return None

    return _truncate_content(content, anchor["max_chars"])


def load_all_anchors(
    anchors: List[Dict[str, Any]],
    max_total: int,
    only_anchors: List[Dict[str, Any]] = None,
) -> Optional[str]:
    """Load anchor files and format for injection.

    Args:
        anchors: All configured anchors.
        max_total: Max total chars across all loaded anchors.
        only_anchors: If provided, only load these specific anchors
                      (selective injection for active projects only).
                      If None, loads all anchors (backward compat).

    Returns a formatted string ready to inject as a user message,
    or None if no anchors have content.
    """
    target = only_anchors if only_anchors is not None else anchors
    if not target:
        return None

    sections = []
    total_chars = 0

    for anchor in target:
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
    """Build the flush prompt for a single anchor (legacy, used by tests)."""
    return build_batch_anchor_save_prompt([anchor])


def build_batch_anchor_save_prompt(anchors: List[Dict[str, Any]]) -> str:
    """Build a single flush prompt for ALL active anchors.

    One LLM call reads and patches all relevant anchor files.
    The model should emit read_file calls first, then patch calls
    in subsequent rounds.
    """
    file_list = "\n".join(
        f"  - {a['path']} (keywords: {', '.join(a['keywords'])})"
        for a in anchors
    )

    return (
        f"[System: Before context is compressed, update these project context files "
        f"to reflect current state:\n"
        f"{file_list}\n"
        f"\n"
        f"PROCEDURE:\n"
        f"1. Call read_file for EACH file above (emit all read_file calls at once).\n"
        f"2. After seeing the contents, use patch to update ONLY what changed.\n"
        f"\n"
        f"RULES for updating:\n"
        f"- Each file must have a '## Progress' section with TWO subsections:\n"
        f"  '### Done' = concrete actions completed this session (specific: what was built, how, key details)\n"
        f"  '### Next' = the immediate next task(s) to do, with any blockers\n"
        f"- When something from '### Next' has been accomplished, move it to '### Done' with concrete details of what was done, then write the NEW next task(s).\n"
        f"- '### Done' is a rolling summary of work, NOT a changelog. Replace old items when they become irrelevant. Keep only what matters to understand current state.\n"
        f"- '### Next' must always exist and never be empty. It tells future-you exactly what to do when the session resumes.\n"
        f"- REPLACE outdated info in-place elsewhere in the file. Do NOT append to the end.\n"
        f"- REMOVE resolved issues, completed one-time tasks from other sections.\n"
        f"- Keep files organized: clear sections, no duplicates, no stale state.\n"
        f"- Be SPECIFIC: name concrete actions (e.g. 'Deployed 18 tables via psql including RLS policies and triggers for auto user creation'), NOT vague ('Schema OK', 'Done').\n"
        f"- Do NOT duplicate info already in other files (e.g. don't copy passwords or creds that live in .env files).\n"
        f"- If nothing changed for a file, skip it entirely.]"
    )


def should_pre_flush(
    threshold_tokens: int,
    current_tokens: int,
    pre_flush_ratio: float = PRE_FLUSH_THRESHOLD,
) -> bool:
    """Check if we should nudge the model to save anchors proactively.

    Returns True when token usage crosses pre_flush_ratio * threshold_tokens
    (e.g. 70% of the compression threshold), giving the model time to save
    state in its normal flow before compression fires.
    """
    if threshold_tokens <= 0 or current_tokens <= 0:
        return False
    return current_tokens >= int(threshold_tokens * pre_flush_ratio)


def build_pre_flush_nudge(anchors: List[Dict[str, Any]]) -> str:
    """Build a nudge message telling the model to save state NOW.

    This is injected as a system message in the normal conversation flow
    so the model saves anchors without needing a separate LLM call.
    """
    file_list = "\n".join(f"  - {a['path']}" for a in anchors)

    return (
        f"[System: Context compression is approaching. Before continuing, "
        f"update these project context files with any changes from this session. "
        f"Use read_file then patch.\n"
        f"KEY RULE: Each file must have a '## Progress' section with "
        f"'### Done' (concrete actions completed) and '### Next' (immediate next tasks). "
        f"Move accomplished Next items into Done with details, then write new Next items. "
        f"Be specific, not vague. No duplicated info from .env files.\n"
        f"{file_list}\n"
        f"Do this now, then continue with the user's request.]"
    )


def snapshot_anchor_hashes(anchors: List[Dict[str, Any]]) -> Dict[str, str]:
    """Take a hash snapshot of all anchor files.

    Returns {path: md5_hex} for files that exist.
    Used to detect if files were modified between pre-flush nudge and compression.
    """
    hashes = {}
    for anchor in anchors:
        path = anchor["path"]
        try:
            content = Path(path).read_bytes()
            hashes[path] = hashlib.md5(content).hexdigest()
        except (OSError, IOError):
            pass
    return hashes


def anchors_changed_since(
    anchors: List[Dict[str, Any]],
    old_hashes: Dict[str, str],
) -> bool:
    """Check if any anchor files changed since the snapshot.

    Returns True if at least one file was modified (meaning the model
    already saved state via pre-flush, so we can skip the expensive flush).
    """
    for anchor in anchors:
        path = anchor["path"]
        try:
            content = Path(path).read_bytes()
            current_hash = hashlib.md5(content).hexdigest()
        except (OSError, IOError):
            continue
        old_hash = old_hashes.get(path)
        if old_hash and current_hash != old_hash:
            return True
    return False
