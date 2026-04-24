"""Context injection — assemble relevant episodic memory for system prompt injection.

Gathers top episodes + entities within a token budget, prioritizing
recency and relevance. Replaces the ad-hoc search in the provider
with a structured approach.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from memory.config import MAX_MEMORY_INJECTION_TOKENS, TOP_EPISODES, TOP_ENTITIES
from memory.episodic_store import EpisodicStore

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English."""
    return len(text) // 4


def _format_episode(ep: Dict[str, Any]) -> str:
    """Format an episode for injection."""
    parts = [f"[Episode: {ep.get('topic', 'untitled')}]"]
    parts.append(ep.get("summary", ""))
    if ep.get("key_decisions"):
        kd = ep["key_decisions"]
        if isinstance(kd, str) and kd.startswith("["):
            try:
                kd_list = json.loads(kd)
                if kd_list:
                    parts.append(f"Decisions: {'; '.join(str(d) for d in kd_list[:3])}")
            except (json.JSONDecodeError, TypeError):
                parts.append(f"Decisions: {kd}")
        elif kd:
            parts.append(f"Decisions: {kd}")
    if ep.get("unresolved"):
        unr = ep["unresolved"]
        if isinstance(unr, str) and unr.startswith("["):
            try:
                unr_list = json.loads(unr)
                if unr_list:
                    parts.append(f"Open: {'; '.join(str(u) for u in unr_list[:3])}")
            except (json.JSONDecodeError, TypeError):
                parts.append(f"Open: {unr}")
    return " | ".join(parts)


def _format_entity(ent: Dict[str, Any]) -> str:
    """Format an entity for injection."""
    profile = ent.get("profile_json", {})
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except (json.JSONDecodeError, TypeError):
            profile = {}

    # Pick the most informative fields
    preview_parts = []
    for key in ("role", "description", "tool_type", "language", "status"):
        if key in profile:
            preview_parts.append(f"{key}={profile[key]}")

    preview = ", ".join(preview_parts[:3]) if preview_parts else ""
    if preview:
        return f"[{ent.get('type', '?')}] {ent.get('name', '?')} ({preview})"
    return f"[{ent.get('type', '?')}] {ent.get('name', '?')}"


def assemble_context(
    store: EpisodicStore,
    query: str,
    *,
    max_tokens: int = 0,
    top_episodes: int = 0,
    top_entities: int = 0,
) -> str:
    """Assemble episodic memory context for injection into the system prompt.

    Searches for relevant episodes and entities, formats them, and trims
    to fit within the token budget.

    Args:
        store: EpisodicStore instance.
        query: Search query (typically the user's message or recent context).
        max_tokens: Token budget (0 = use config default).
        top_episodes: Max episodes to include (0 = use config default).
        top_entities: Max entities to include (0 = use config default).

    Returns:
        Formatted context string, or empty string if nothing relevant.
    """
    budget = max_tokens or MAX_MEMORY_INJECTION_TOKENS
    ep_limit = top_episodes or TOP_EPISODES
    ent_limit = top_entities or TOP_ENTITIES

    parts = []
    used_tokens = 0

    # 1. Search episodes
    try:
        episodes = store.search_episodes(query, limit=ep_limit)
        if episodes:
            parts.append("Relevant past episodes:")
            for ep in episodes:
                line = _format_episode(ep)
                line_tokens = _estimate_tokens(line)
                if used_tokens + line_tokens > budget:
                    break
                parts.append(f"  - {line}")
                used_tokens += line_tokens
    except Exception as e:
        logger.debug("Episode search failed during context assembly: %s", e)

    # 2. Search entities
    try:
        entities = store.search_entities(query, limit=ent_limit)
        if entities:
            parts.append("Relevant entities:")
            for ent in entities:
                line = _format_entity(ent)
                line_tokens = _estimate_tokens(line)
                if used_tokens + line_tokens > budget:
                    break
                parts.append(f"  - {line}")
                used_tokens += line_tokens
    except Exception as e:
        logger.debug("Entity search failed during context assembly: %s", e)

    if len(parts) <= 1:
        # Only header, no actual content
        return ""

    result = "\n".join(parts)

    # Final budget check
    if _estimate_tokens(result) > budget:
        # Truncate to budget
        max_chars = budget * 4
        result = result[:max_chars] + "\n...[truncated]"

    return f"\n[Episodic Memory Context]\n{result}\n"


def assemble_recent_context(
    store: EpisodicStore,
    *,
    max_tokens: int = 0,
    max_episodes: int = 3,
    max_entities: int = 3,
) -> str:
    """Assemble context from recent episodes and entities (no query needed).

    Used for cold-start or when there's no search query yet.
    Prioritizes recency.
    """
    budget = max_tokens or MAX_MEMORY_INJECTION_TOKENS
    parts = []
    used_tokens = 0

    try:
        episodes = store.get_recent_episodes(limit=max_episodes)
        if episodes:
            parts.append("Recent episodes:")
            for ep in episodes:
                line = _format_episode(ep)
                if used_tokens + _estimate_tokens(line) > budget:
                    break
                parts.append(f"  - {line}")
                used_tokens += _estimate_tokens(line)
    except Exception:
        pass

    try:
        entities = store.get_recent_entities(limit=max_entities)
        if entities:
            parts.append("Active entities:")
            for ent in entities:
                line = _format_entity(ent)
                if used_tokens + _estimate_tokens(line) > budget:
                    break
                parts.append(f"  - {line}")
                used_tokens += _estimate_tokens(line)
    except Exception:
        pass

    if len(parts) <= 1:
        return ""

    return f"\n[Episodic Memory Context]\n{chr(10).join(parts)}\n"
