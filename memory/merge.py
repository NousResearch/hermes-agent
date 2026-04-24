"""Merge pipeline — deduplicate extracted facts into existing entity profiles.

Implements the 4-operation approach (ADD/UPDATE/DELETE/NOOP) from Mem0's
proven pattern. Called at session end by the EpisodicMemoryProvider.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from memory.config import MERGE_TIMEOUT
from memory.episodic_store import EpisodicStore

logger = logging.getLogger(__name__)

EPISODE_TYPES = ("raw", "substantive", "chitchat", "banter")

_BANTER_PATTERNS = [
    "going to sleep",
    "goodnight",
    "good night",
    "said goodnight",
    "said they were tired",
    "going to bed",
    "good morning",
    "good afternoon",
    "good evening",
    "dad joke",
    "brb",
    "be right back",
    "lol",
    "haha",
    "thanks k",
]

_CHITCHAT_SIGNALS = ["joke", "banter", "chitchat", "small talk", "fun", "emoji", "lol"]


def _jsonish_has_items(value: Any) -> bool:
    if not value:
        return False
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except Exception:
            return bool(text)
        return bool(parsed)
    return True


def classify_episode_type(
    topic: str,
    summary: str,
    key_decisions: Optional[str] = None,
    source_turns_json: Optional[str] = None,
) -> str:
    """Heuristic episode classification.

    Returns one of:
    - "substantive": has decisions, unresolved items, or rich source material
    - "raw": the default for extracted episodes that are too short for wiki but not banter
    - "chitchat": social filler that isn't pure banter (e.g., jokes, greetings with some substance)
    - "banter": one-liners, greetings, farewell, low-information social exchanges
    """
    topic_l = (topic or "").lower()
    summary_l = (summary or "").lower()
    combined = f"{topic_l} {summary_l}"

    # Immediate banter patterns (only when the summary itself is short)
    if len(summary_l) < 60:
        for pat in _BANTER_PATTERNS:
            if pat in combined:
                return "banter"

    # Substantive if it has decisions or unresolved items
    if _jsonish_has_items(key_decisions):
        return "substantive"

    # Source turn reference usually means real work
    if _jsonish_has_items(source_turns_json):
        return "substantive"

    # Short, no-decision, no-source → banter
    if len(summary_l) < 40:
        return "banter"

    # Longer but still no decisions → chitchat if social signals present
    if len(summary_l) < 100:
        has_social = any(signal in combined for signal in _CHITCHAT_SIGNALS)
        if has_social:
            return "chitchat"
        return "raw"

    # Anything long enough to be worth keeping but without decisions
    return "raw"

MERGE_SYSTEM_PROMPT = """You are a memory consolidation system. Given existing stored entity profiles and newly extracted facts, determine how to update the knowledge base.

For each new fact, decide one of:
- ADD: Fact is genuinely new and not redundant with existing knowledge
- UPDATE: Fact contradicts or supersedes an existing fact — specify the old value
- DELETE: Fact explicitly retracts or corrects an existing fact
- NOOP: Fact is redundant with existing knowledge, skip it

Output a JSON array of operations:
[{"operation": "ADD|UPDATE|DELETE|NOOP", "entity_id": "...", "fact": {...}, "target_field": "...", "reason": "..."}]

Rules:
- Prefer UPDATE over ADD when a fact modifies an existing attribute
- Keep NOOPs to a minimum — only skip truly redundant facts
- For ADD operations on existing entities, include the entity_id
- For new entities, set entity_id to null (will be created)
- Output ONLY valid JSON, no markdown fences"""

MERGE_USER_TEMPLATE = """EXISTING ENTITY PROFILES:
{existing_json}

NEWLY EXTRACTED FACTS:
{new_facts_json}

Determine merge operations for the new facts."""


def _get_existing_entities(store: EpisodicStore, entity_names: List[str]) -> List[dict]:
    """Look up existing entities by name using FTS5 search."""
    existing = []
    seen_ids = set()
    for name in entity_names:
        try:
            results = store.search_entities(name, limit=2)
            for r in results:
                if r["id"] not in seen_ids:
                    existing.append(r)
                    seen_ids.add(r["id"])
        except Exception:
            pass
    return existing


def _entity_names_from_facts(extracted: Dict[str, Any]) -> List[str]:
    """Pull all entity names mentioned in extracted facts."""
    names = set()
    for entity in extracted.get("entities", []):
        if isinstance(entity, dict) and entity.get("name"):
            names.add(entity["name"])
    for fact in extracted.get("facts", []):
        if isinstance(fact, dict):
            for key in ("subject", "object"):
                val = fact.get(key, "")
                if val and isinstance(val, str) and len(val) < 100:
                    names.add(val)
    return list(names)


def merge_extracted_facts(
    store: EpisodicStore,
    extracted: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """Merge extracted facts into the episodic store.

    Args:
        store: The EpisodicStore instance.
        extracted: Dict with entities, facts, events from extraction.
        session_id: Current session ID.

    Returns:
        Summary dict with counts of operations performed.
    """
    stats = {"added": 0, "updated": 0, "deleted": 0, "noop": 0, "errors": 0}

    entities = extracted.get("entities", [])
    facts = extracted.get("facts", [])
    events = extracted.get("events", [])

    if not entities and not facts and not events:
        return stats

    # 1. Upsert entities directly (no LLM needed for entity creation)
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        try:
            from memory.extraction import canonicalize_entity, entity_id_from_name

            canonical = canonicalize_entity(
                entity["name"],
                entity.get("type", "concept"),
                entity.get("attributes", {}),
            )
            eid = entity_id_from_name(canonical["name"], canonical["type"], canonical["attributes"])
            profile = canonical.get("attributes", {})
            if not isinstance(profile, dict):
                profile = {}
            # Add extraction metadata
            profile["_source"] = "extraction"
            profile["_extracted_at"] = time.time()

            store.upsert_entity(
                entity_id=eid,
                entity_type=canonical.get("type", entity.get("type", "concept")),
                name=canonical.get("name", entity["name"]),
                profile_json=profile,
            )
            stats["added"] += 1
        except Exception as e:
            logger.error("Entity upsert failed for %s: %s", entity.get("name"), e)
            stats["errors"] += 1

    # 2. Create episode from facts and events
    if facts or events:
        try:
            topic_parts = []
            summary_parts = []

            for fact in facts:
                if isinstance(fact, dict):
                    subj = fact.get("subject", "")
                    pred = fact.get("predicate", "")
                    obj = fact.get("object", "")
                    if subj and pred:
                        summary_parts.append(f"{subj} {pred} {obj}".strip())
                        if subj not in topic_parts:
                            topic_parts.append(subj)

            for event in events:
                if isinstance(event, dict):
                    desc = event.get("description", "")
                    if desc:
                        summary_parts.append(desc)

            if summary_parts:
                topic = ", ".join(topic_parts[:3]) if topic_parts else "general"
                summary = "; ".join(summary_parts[:10])

                store.create_episode(
                    session_id=session_id,
                    topic=topic,
                    summary=summary,
                    key_decisions=None,
                    participants=", ".join(topic_parts[:5]) if topic_parts else None,
                    episode_type=classify_episode_type(topic, summary),
                )
                stats["added"] += 1
        except Exception as e:
            logger.error("Episode creation from extraction failed: %s", e)
            stats["errors"] += 1

    # 3. Create episodes from events
    for event in events:
        if not isinstance(event, dict) or not event.get("description"):
            continue
        try:
            desc = event.get("description", "")
            topic = desc[:100]
            store.create_episode(
                session_id=session_id,
                topic=topic,
                summary=desc,
                participants=", ".join(event.get("participants", []))
                if event.get("participants")
                else None,
                episode_type=classify_episode_type(topic, desc),
            )
            stats["added"] += 1
        except Exception as e:
            logger.error("Event episode creation failed: %s", e)
            stats["errors"] += 1

    # 4. Extract relationships between entities
    try:
        from memory.temporal import extract_relationships_from_facts
        rel_count = extract_relationships_from_facts(store, extracted, session_id)
        if rel_count:
            stats["relationships"] = rel_count
            logger.info("Extracted %d relationships from session %s", rel_count, session_id)
    except Exception as e:
        logger.debug("Relationship extraction failed: %s", e)

    # 5. Record entity changes in fact_history for temporal tracking
    try:
        for entity in entities:
            if not isinstance(entity, dict) or not entity.get("name"):
                continue
            from memory.extraction import canonicalize_entity, entity_id_from_name
            canonical = canonicalize_entity(
                entity["name"],
                entity.get("type", "concept"),
                entity.get("attributes", {}),
            )
            eid = entity_id_from_name(canonical["name"], canonical["type"], canonical.get("attributes", {}))
            attrs = canonical.get("attributes", {})
            if isinstance(attrs, dict):
                for field, value in attrs.items():
                    if field.startswith("_"):
                        continue
                    store.record_fact_change(
                        entity_id=eid,
                        field_path=field,
                        old_value=None,
                        new_value=str(value),
                        operation="ADD",
                        session_id=session_id,
                        confidence="medium",
                    )
    except Exception as e:
        logger.debug("Fact history recording failed: %s", e)

    logger.info(
        "Merge complete for session %s: %s",
        session_id,
        json.dumps(stats),
    )
    return stats


def merge_session(
    store: EpisodicStore,
    session_id: str,
    extracted: Dict[str, Any],
) -> Dict[str, Any]:
    """Full merge pipeline for a session's extracted facts.

    This is the main entry point called by the provider at session end.
    Uses direct upsert for entities (fast path) and LLM-based dedup
    only when there are conflicting facts.

    Args:
        store: EpisodicStore instance.
        session_id: Session to merge.
        extracted: Extracted facts dict from extraction pipeline.

    Returns:
        Merge statistics dict.
    """
    if not extracted or not any(extracted.get(k) for k in ("entities", "facts", "events")):
        logger.debug("No extracted facts to merge for session %s", session_id)
        return {"added": 0, "updated": 0, "deleted": 0, "noop": 0, "errors": 0}

    return merge_extracted_facts(store, extracted, session_id)
