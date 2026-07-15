"""Adaptation layer — injects learned preferences into agent context.

Reads the preference model from the fleet graph and produces:
1. Compact preference summary block for system prompt injection
2. [context: preference] memory entries for the scored pipeline
3. Tool selection guidance
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.collaboration.graph_client import get_client

logger = logging.getLogger(__name__)

# Confidence threshold for adaptation
_ADAPT_THRESHOLD = 0.6


def load_preference_summary() -> str:
    """Load high-confidence preferences from the graph and format as a prompt block.

    Returns an empty string if no preferences have sufficient confidence.
    """
    try:
        client = get_client()
        if not client.is_available():
            return ""

        # Load preference memories from graph
        prefs = client.search_memories("category:preference", limit=20)
        if not prefs:
            return ""

        # Parse and filter by confidence
        lines: List[str] = []
        for p in prefs:
            confidence = p.get("confidence", 0.0)
            content = p.get("content", "")
            if confidence >= _ADAPT_THRESHOLD and content:
                # Content format: "dimension: value (confidence=X, ...)"
                lines.append(f"  - {content.split(' (')[0]}")

        if not lines:
            return ""

        return (
            "[Learned collaboration preferences — derived from past interactions]\n"
            "The following preferences were learned from observing how you work. "
            "They adapt automatically as patterns change.\n"
            + "\n".join(lines)
        )

    except Exception as e:
        logger.debug("Failed to load preference summary: %s", e)
        return ""


def format_preference_entries() -> List[Dict[str, Any]]:
    """Format high-confidence preferences as memory tool operations.

    Returns a list of operation dicts suitable for memory(action='batch', ...).
    Each preference becomes a [context: preference] [signal: X] entry.
    """
    try:
        client = get_client()
        if not client.is_available():
            return []

        prefs = client.search_memories("category:preference", limit=20)
        operations = []

        for p in prefs:
            confidence = p.get("confidence", 0.0)
            content = p.get("content", "")
            if confidence >= _ADAPT_THRESHOLD and content:
                # Parse "dimension: value"
                dimension = content.split(":")[0] if ":" in content else "general"
                entry_text = content.split(" (")[0] if " (" in content else content

                operations.append({
                    "action": "add",
                    "target": "memory",
                    "context": "preference",
                    "signal": min(confidence, 0.95),
                    "content": entry_text,
                })

        return operations

    except Exception as e:
        logger.debug("Failed to format preference entries: %s", e)
        return []


def get_tool_guidance() -> Dict[str, List[str]]:
    """Return tool preference mappings per task type.

    Returns a dict like: {"deploy": ["terminal", "read_file"], ...}
    """
    try:
        client = get_client()
        if not client.is_available():
            return {}

        prefs = client.search_memories(
            query="category:preference tool_preference",
            limit=5,
        )
        if not prefs:
            return {}

        for p in prefs:
            content = p.get("content", "")
            if "tool_preference" not in content:
                continue
            # Extract JSON from content
            import re
            m = re.search(r'\{.*\}', content)
            if m:
                import json
                try:
                    return json.loads(m.group(0))
                except (json.JSONDecodeError, Exception):
                    pass

        return {}

    except Exception as e:
        logger.debug("Failed to load tool guidance: %s", e)
        return {}
