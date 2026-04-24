"""Structured fact extraction from conversation turns.

Uses GPT-5.4-mini (via Codex OAuth) to pull entities, facts, and
preferences from raw conversation blocks. Called every EXTRACT_BATCH_SIZE
turns by the EpisodicMemoryProvider.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from memory.config import EXTRACT_TIMEOUT, get_memory_model_settings

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """You are a knowledge extraction system. Given a block of conversation turns, extract structured information as JSON.

Extract these categories:
1. entities: People, projects, tools, concepts mentioned — with type and attributes
2. facts: Discrete, verifiable statements about the world or user preferences
3. events: Things that happened, were planned, or decided

Rules:
- Only extract EXPLICITLY stated information, do not infer or speculate
- Use consistent, normalized entity names (e.g. "Hermes" not "the bot")
- Mark confidence as "high" (direct statement), "medium" (implied), or "low" (uncertain)
- For facts, include: subject, predicate, object, confidence
- For entities, include: name, type (person/project/tool/concept/location), attributes dict
- For events, include: description, participants (list of entity names), timestamp_hint
- Omit empty arrays
- Output ONLY valid JSON, no markdown fences or explanation"""

EXTRACT_USER_TEMPLATE = """CONVERSATION TURNS:
{turns_json}

Extract structured knowledge from this conversation block."""


def _format_turns(turns: List[Dict[str, Any]]) -> str:
    """Format turns into a readable string for the extraction prompt."""
    lines = []
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        tool_name = t.get("tool_name", "")
        if role == "tool" and tool_name:
            lines.append(f"[Tool: {tool_name}] {content[:500]}")
        elif role == "assistant" and tool_name:
            lines.append(f"[Assistant uses {tool_name}] {content[:500]}")
        else:
            lines.append(f"[{role}] {content[:1000]}")
    return "\n".join(lines)


def extract_from_turns(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract structured facts from a block of conversation turns.

    Args:
        turns: List of turn dicts with 'role', 'content', optional 'tool_name'.

    Returns:
        Dict with keys: entities, facts, events (each a list).
        Returns empty extraction on failure (never raises).
    """
    if not turns:
        return {"entities": [], "facts": [], "events": []}

    turns_text = _format_turns(turns)
    user_msg = EXTRACT_USER_TEMPLATE.format(turns_json=turns_text)

    provider, model = get_memory_model_settings("extract")

    try:
        response = call_llm(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=2000,
            timeout=EXTRACT_TIMEOUT,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines)

        result = json.loads(content)

        # Validate structure — ensure expected keys exist with lists
        for key in ("entities", "facts", "events"):
            if key not in result:
                result[key] = []
            if not isinstance(result[key], list):
                result[key] = []

        logger.info(
            "Extraction complete: %d entities, %d facts, %d events from %d turns",
            len(result.get("entities", [])),
            len(result.get("facts", [])),
            len(result.get("events", [])),
            len(turns),
        )
        return result

    except json.JSONDecodeError as e:
        logger.error("Extraction returned invalid JSON: %s", e)
        return {"entities": [], "facts": [], "events": []}
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return {"entities": [], "facts": [], "events": []}


def _looks_like_url(value: str) -> bool:
    text = (value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")



def _slugify(value: str) -> str:
    normalized = value.lower().strip().replace(" ", "-")
    normalized = "".join(c for c in normalized if c.isalnum() or c == "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-") or "unknown"



def canonicalize_entity(
    name: str,
    entity_type: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize an extracted entity before merge/storage.

    Returns a dict with canonical name/type plus normalized attributes.
    """
    attrs = dict(attributes or {})
    raw_name = (name or "").strip()
    canonical_type = (entity_type or "concept").strip().lower() or "concept"
    role = str(attrs.get("role", "") or "").lower()

    if _looks_like_url(raw_name) or "url" in role:
        canonical_type = "resource"

    canonical_name = raw_name
    if canonical_type == "person":
        canonical_name = (
            str(attrs.get("full_name") or "").strip()
            or str(attrs.get("preferred_name") or "").strip()
            or raw_name
        )
    elif canonical_type == "resource":
        canonical_name = (
            str(attrs.get("title") or "").strip()
            or str(attrs.get("document_title") or "").strip()
            or raw_name
        )

    aliases = attrs.get("aliases")
    if not isinstance(aliases, list):
        aliases = [] if aliases in (None, "") else [str(aliases)]
    if raw_name and raw_name != canonical_name and raw_name not in aliases:
        aliases.append(raw_name)
    if aliases:
        attrs["aliases"] = aliases

    return {
        "name": canonical_name or raw_name,
        "type": canonical_type,
        "attributes": attrs,
    }



def entity_id_from_name(name: str, entity_type: str, attributes: Optional[Dict[str, Any]] = None) -> str:
    """Generate a stable entity ID from name and type.

    Examples:
        ("Aaron", "person") -> "person-aaron"
        ("Hermes", "project") -> "project-hermes"
    """
    canonical = canonicalize_entity(name, entity_type, attributes)
    normalized = _slugify(canonical["name"])
    return f"{canonical['type']}-{normalized}"


def turns_to_extraction_input(
    turns: List[Dict[str, Any]],
    start_idx: int = 0,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """Slice turns into an extraction batch.

    Args:
        turns: Full turn list.
        start_idx: Start index in the turn list.
        batch_size: Number of turns to extract from.

    Returns:
        Sliced turn list.
    """
    end_idx = min(start_idx + batch_size, len(turns))
    return turns[start_idx:end_idx]
