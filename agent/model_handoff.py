"""Model handoff compressor for routing transitions.

When the smart router selects a different model for a turn (e.g. escalating
from glm-5.1 to opus-4.6 or descending back), the conversation context
needs to be condensed into a handoff message so the new model continues
seamlessly without repeating everything.

This module produces a structured handoff summary that:
1. Captures what was accomplished in previous turns
2. Preserves active task state (files modified, decisions made)
3. Includes the last N messages verbatim for continuity
4. Marks the handoff clearly so the new model treats it as reference, not instruction

The handoff is injected as a system-level message prefix before the new model
receives the conversation, ensuring zero loss of critical context while keeping
token usage lean.

Public API:
    build_handoff(prev_model, new_model, conversation, task_state, **kwargs) -> str
    should_generate_handoff(prev_model, new_model, task_state) -> bool
    HANDOFF_PREFIX (constant)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

logger = __import__("logging").getLogger(__name__)

HANDOFF_PREFIX = (
    "[MODEL HANDOFF — REFERENCE ONLY] The assistant model has changed for this turn. "
    "The summary below describes what the previous model accomplished. "
    "Treat it as background context, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; they were already addressed. "
    "Your task is to continue from the '## Active Task' section. "
    "Respond ONLY to the latest user message after this handoff."
)

# Maximum number of recent messages to include verbatim in the handoff
_MAX_VERBATIM_MESSAGES = 4

# Maximum character length for the handoff before truncation
_MAX_HANDOFF_CHARS = 4000


def should_generate_handoff(
    prev_model: str,
    new_model: str,
    task_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Determine whether a model transition warrants a handoff summary.

    Skip handoff when:
    - Same model (no transition)
    - Task state indicates a trivial continuation (short "ok", "sigue", etc.)
    - Both models are in the same tier group
    """
    if not prev_model or not new_model:
        return False
    if prev_model == new_model:
        return False

    # If task state says this is a continuation marker, we still want a handoff
    # because the model changed — the new model needs to know what it's continuing.
    # But if the task is idle (no active_task), skip lightweight transitions.

    if task_state and not task_state.get("active_task", False):
        # No active task — only handoff if going UP a tier (escalation)
        prev_tier = _tier_of(prev_model)
        new_tier = _tier_of(new_model)
        if new_tier <= prev_tier:
            # Descending or same tier with no active task — skip handoff
            return prev_tier != new_tier  # only on actual tier change if descending

    return True


def build_handoff(
    prev_model: str,
    new_model: str,
    conversation: List[Dict[str, Any]],
    task_state: Optional[Dict[str, Any]] = None,
    *,
    focus_topic: Optional[str] = None,
    recent_only: int = _MAX_VERBATIM_MESSAGES,
    max_chars: int = _MAX_HANDOFF_CHARS,
) -> str:
    """Build a handoff summary string for a model transition.

    Args:
        prev_model: Model that was handling the conversation.
        new_model: Model that will handle the next turn.
        conversation: List of message dicts (role, content) from the session.
        task_state: Current routing task state (active_task, last_category, etc).
        focus_topic: Optional explicit focus topic for the summary.
        recent_only: Number of recent messages to include verbatim.
        max_chars: Maximum character length for the handoff before truncation.

    Returns:
        A formatted handoff string to inject as a system message prefix.
    """
    if not should_generate_handoff(prev_model, new_model, task_state):
        return ""

    parts: List[str] = []

    # --- Header ---
    parts.append(HANDOFF_PREFIX)
    parts.append("")
    parts.append(f"**Previous model:** {prev_model}")
    parts.append(f"**Current model:**  {new_model}")
    parts.append(f"**Transition at:**  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")

    # --- Active Task ---
    parts.append("## Active Task")
    if task_state and task_state.get("active_task"):
        category = task_state.get("last_category", "unknown")
        tier = task_state.get("last_tier", "?")
        turns = task_state.get("turns_in_task", 0)
        parts.append(f"Category: {category} | Tier: {tier} | Turns in task: {turns}")
    else:
        parts.append("No active task — this is a fresh or idle session.")
    parts.append("")

    # --- Recent Context (verbatim last N messages) ---
    if conversation:
        recent = conversation[-recent_only:] if len(conversation) > recent_only else conversation
        parts.append("## Recent Messages (verbatim)")
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part messages (e.g. images + text)
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            content = str(content)
            # Truncate very long individual messages
            if len(content) > 800:
                content = content[:777] + "... [truncated]"
            parts.append(f"[{role}]: {content}")
        parts.append("")

    # --- Focus Topic ---
    if focus_topic:
        parts.append("## Focus")
        parts.append(focus_topic)
        parts.append("")

    # --- Accomplishments (derived from conversation if no explicit summary) ---
    accomplishments = _extract_accomplishments(conversation)
    if accomplishments:
        parts.append("## What Was Done")
        for item in accomplishments:
            parts.append(f"- {item}")
        parts.append("")

    # --- Remaining Work (if identifiable) ---
    remaining = _extract_remaining(conversation)
    if remaining:
        parts.append("## Remaining Work")
        for item in remaining:
            parts.append(f"- {item}")
        parts.append("")

    handoff = "\n".join(parts)

    # Truncate if too long
    if len(handoff) > max_chars:
        handoff = handoff[: max_chars - 3] + "..."

    return handoff


def _tier_of(model: str) -> int:
    """Map a model name to its tier number using routing_v2 DEFAULT_TIERS."""
    try:
        from agent.routing_v2 import DEFAULT_TIERS
        for idx, group in enumerate(DEFAULT_TIERS, start=1):
            if model in group:
                return idx
    except Exception:
        pass
    # Fallback heuristic based on model name patterns
    model_lower = (model or "").lower()
    if any(x in model_lower for x in ["glm", "nano", "mini", "haiku"]):
        return 1
    if any(x in model_lower for x in ["sonnet", "coder", "kimi"]):
        return 3
    if any(x in model_lower for x in ["opus", "qwen3.5", "mistral-large"]):
        return 5
    return 2  # default intermediate


def _extract_accomplishments(conversation: List[Dict[str, Any]]) -> List[str]:
    """Heuristic extraction of completed items from recent assistant messages."""
    accomplishments = []
    _DONE_MARKERS = ["done", "completed", "fixed", "created", "written", "pushed", "merged", "passed", "verified"]

    for msg in conversation:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        if len(content) > 400:
            content = content[:400]
        # Look for action summaries (tool call results, completion statements)
        for line in content.split("\n"):
            line_lower = line.lower().strip()
            if any(m in line_lower for m in _DONE_MARKERS) and len(line.strip()) > 10:
                accomplishments.append(line.strip()[:120])
                if len(accomplishments) >= 5:
                    return accomplishments
    return accomplishments


def _extract_remaining(conversation: List[Dict[str, Any]]) -> List[str]:
    """Heuristic extraction of pending/remaining items from the latest messages."""
    remaining = []
    _PENDING_MARKERS = ["pending", "remaining", "next:", "todo:", "still need", "needs to", "not yet"]

    # Only check the last few messages
    for msg in conversation[-6:]:
        content = str(msg.get("content", ""))
        if len(content) > 400:
            content = content[:400]
        for line in content.split("\n"):
            line_lower = line.lower().strip()
            if any(m in line_lower for m in _PENDING_MARKERS):
                remaining.append(line.strip()[:120])
                if len(remaining) >= 3:
                    return remaining
    return remaining


# ─────────────────────────────────────────────────────────────
# Handoff file persistence (optional)
# ─────────────────────────────────────────────────────────────

_HANDOFF_DIR = os.path.expanduser("~/.hermes/router/handoffs")


def save_handoff(handoff_text: str, prev_model: str, new_model: str) -> str:
    """Persist a handoff summary to disk for debugging/auditing.

    Returns the file path of the saved handoff.
    """
    os.makedirs(_HANDOFF_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"handoff_{prev_model}_to_{new_model}_{ts}.md"
    # Sanitize filename
    filename = "".join(c if c.isalnum() or c in "_-." else "_" for c in filename)
    filepath = os.path.join(_HANDOFF_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(handoff_text)
        return filepath
    except Exception:
        logger.debug("Failed to save handoff to %s", filepath, exc_info=True)
        return ""


def load_recent_handoffs(limit: int = 5) -> List[Dict[str, Any]]:
    """Load recent handoff summaries from disk for debugging."""
    if not os.path.exists(_HANDOFF_DIR):
        return []
    files = sorted(
        (f for f in os.listdir(_HANDOFF_DIR) if f.endswith(".md")),
        key=lambda f: os.path.getmtime(os.path.join(_HANDOFF_DIR, f)),
        reverse=True,
    )
    result = []
    for fname in files[:limit]:
        fpath = os.path.join(_HANDOFF_DIR, fname)
        try:
            text = open(fpath, encoding="utf-8").read()
            result.append({"file": fname, "path": fpath, "text": text[:500]})
        except Exception:
            pass
    return result