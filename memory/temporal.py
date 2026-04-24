"""Temporal quality analysis — staleness detection, contradiction resolution, relationship extraction.

Phase 4 of the episodic memory system. Provides the logic layer on top of
the episodic store's temporal queries.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.config import (
    CONTRADICTION_CONFIDENCE,
    MAX_CONTRADICTIONS_REPORT,
    STALENESS_CHECK_SESSIONS,
    STALENESS_THRESHOLD_DAYS,
)
from memory.episodic_store import EpisodicStore

logger = logging.getLogger(__name__)

# Seconds in a day
_DAY = 86400


# ── Staleness Detection ─────────────────────────────────────────────────

def detect_stale_entities(
    store: EpisodicStore,
    threshold_days: int = 0,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Find entities not confirmed in N days.

    Args:
        store: EpisodicStore instance.
        threshold_days: Days since last confirmation (0 = use config default).
        limit: Max results.

    Returns:
        List of stale entity dicts with staleness metadata.
    """
    days = threshold_days or STALENESS_THRESHOLD_DAYS
    threshold_seconds = days * _DAY
    cutoff = time.time() - threshold_seconds

    stale = store.get_stale_entities(threshold_seconds, limit=limit)

    results = []
    for entity in stale:
        last_confirmed = entity.get("last_confirmed_at")
        if last_confirmed:
            days_stale = (time.time() - last_confirmed) / _DAY
        else:
            # Never confirmed — use updated_at
            days_stale = (time.time() - entity.get("updated_at", 0)) / _DAY

        results.append({
            "entity_id": entity["id"],
            "name": entity["name"],
            "type": entity["type"],
            "days_stale": round(days_stale, 1),
            "last_confirmed": last_confirmed,
            "last_updated": entity.get("updated_at"),
            "severity": "high" if days_stale > days * 2 else "medium" if days_stale > days else "low",
        })

    return results


def detect_stale_facts(
    store: EpisodicStore,
    entity_id: str,
    threshold_days: int = 0,
) -> List[Dict[str, Any]]:
    """Find facts for an entity that haven't been re-observed recently.

    Args:
        store: EpisodicStore instance.
        entity_id: Entity to check.
        threshold_days: Days since last observation (0 = use config default).

    Returns:
        List of stale fact dicts.
    """
    days = threshold_days or STALENESS_THRESHOLD_DAYS
    threshold_seconds = days * _DAY

    stale = store.get_stale_facts(entity_id, threshold_seconds)

    results = []
    for fact in stale:
        last_observed = fact.get("last_observed", 0)
        days_stale = (time.time() - last_observed) / _DAY if last_observed else 999
        results.append({
            "field_path": fact["field_path"],
            "current_value": fact.get("new_value"),
            "last_observed": last_observed,
            "days_stale": round(days_stale, 1),
            "severity": "high" if days_stale > days * 2 else "medium",
        })

    return results


# ── Contradiction Resolution ────────────────────────────────────────────

def detect_contradictions(
    store: EpisodicStore,
    entity_id: Optional[str] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Find potential contradictions in entity facts.

    A contradiction is when the same field has been updated multiple times
    with different values, suggesting conflicting information.

    Args:
        store: EpisodicStore instance.
        entity_id: Optional entity to check (None = all entities).
        limit: Max contradictions to return (0 = use config default).

    Returns:
        List of contradiction dicts with resolution suggestions.
    """
    max_results = limit or MAX_CONTRADICTIONS_REPORT
    raw = store.get_potential_contradictions(entity_id, limit=max_results)

    results = []
    for item in raw:
        values = item.get("values_seen", [])
        if len(values) < 2:
            continue

        # Simple resolution: prefer the most recent value
        latest_value = values[-1] if values else None

        results.append({
            "entity_id": item["entity_id"],
            "entity_name": item.get("entity_name", item["entity_id"]),
            "field_path": item["field_path"],
            "values_seen": values,
            "change_count": item["change_count"],
            "suggested_resolution": {
                "action": "prefer_latest",
                "value": latest_value,
                "reason": f"Most recent of {item['change_count']} changes",
            },
            "severity": "high" if item["change_count"] >= 3 else "medium",
        })

    return results


def resolve_contradiction(
    store: EpisodicStore,
    entity_id: str,
    field_path: str,
    resolved_value: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a contradiction by setting the definitive value.

    Records the resolution in fact_history and updates the entity profile.

    Args:
        store: EpisodicStore instance.
        entity_id: Entity with the contradiction.
        field_path: The field that had conflicting values.
        resolved_value: The definitive value to keep.
        session_id: Current session ID.

    Returns:
        Resolution summary dict.
    """
    # Record the resolution
    store.record_fact_change(
        entity_id=entity_id,
        field_path=field_path,
        old_value=None,  # Resolved from multiple
        new_value=resolved_value,
        operation="RESOLVE",
        session_id=session_id,
        confidence="high",
    )

    # Update the entity profile with the resolved value
    entity = store.get_entity(entity_id)
    if entity:
        profile = entity.get("profile_json", {})
        if isinstance(profile, str):
            profile = json.loads(profile)

        # Navigate field path (supports dotted paths like "address.city")
        parts = field_path.split(".")
        target = profile
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = resolved_value

        store.upsert_entity(
            entity_id=entity_id,
            entity_type=entity["type"],
            name=entity["name"],
            profile_json=profile,
        )

    return {
        "entity_id": entity_id,
        "field_path": field_path,
        "resolved_value": resolved_value,
        "status": "resolved",
    }


# ── Relationship Extraction ─────────────────────────────────────────────

def extract_relationships_from_facts(
    store: EpisodicStore,
    extracted: Dict[str, Any],
    session_id: str,
) -> int:
    """Extract relationships between entities from extracted facts.

    Looks for facts that connect two entities (e.g., "Aaron" "works_with" "Jefe")
    and creates relationship records.

    Args:
        store: EpisodicStore instance.
        extracted: Extracted facts dict with entities, facts, events.
        session_id: Current session ID.

    Returns:
        Number of relationships created.
    """
    from memory.extraction import entity_id_from_name

    entities = extracted.get("entities", [])
    facts = extracted.get("facts", [])

    # Build entity name → ID map from the extracted entities
    name_to_id = {}
    for entity in entities:
        if isinstance(entity, dict) and entity.get("name"):
            eid = entity_id_from_name(entity["name"], entity.get("type", "concept"))
            name_to_id[entity["name"].lower()] = eid

    # Also check existing entities in the store
    for name in list(name_to_id.keys()):
        try:
            results = store.search_entities(name, limit=1)
            if results:
                name_to_id[name] = results[0]["id"]
        except Exception:
            pass

    created = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        subject = (fact.get("subject") or "").lower()
        predicate = fact.get("predicate") or ""
        obj = (fact.get("object") or "").lower()

        if not subject or not predicate or not obj:
            continue

        # Check if both subject and object are known entities
        source_id = name_to_id.get(subject)
        target_id = name_to_id.get(obj)

        if source_id and target_id and source_id != target_id:
            try:
                store.add_relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=predicate,
                    attributes={"confidence": fact.get("confidence", "medium")},
                    session_id=session_id,
                )
                created += 1
            except Exception as e:
                logger.debug("Relationship creation failed: %s", e)

    return created


def get_entity_context(
    store: EpisodicStore,
    entity_id: str,
    depth: int = 1,
) -> Dict[str, Any]:
    """Get rich context for an entity — relationships, history, staleness.

    This is the "full picture" tool — combines multiple temporal queries
    into a single comprehensive view.

    Args:
        store: EpisodicStore instance.
        entity_id: Entity to describe.
        depth: Relationship graph depth (1 = direct only).

    Returns:
        Comprehensive entity context dict.
    """
    entity = store.get_entity(entity_id)
    if not entity:
        return {"error": f"Entity not found: {entity_id}"}

    # Get relationship graph
    graph = store.get_entity_relationships_graph(entity_id, depth=depth)

    # Get fact history (recent changes)
    history = store.get_fact_history(entity_id, limit=10)

    # Check staleness
    stale_facts = detect_stale_facts(store, entity_id)

    # Check contradictions
    contradictions = detect_contradictions(store, entity_id=entity_id, limit=5)

    return {
        "entity": {
            "id": entity["id"],
            "name": entity["name"],
            "type": entity["type"],
            "profile": entity.get("profile_json", {}),
            "created_at": entity.get("created_at"),
            "updated_at": entity.get("updated_at"),
            "last_confirmed_at": entity.get("last_confirmed_at"),
        },
        "relationships": graph,
        "recent_changes": history,
        "stale_facts": stale_facts,
        "contradictions": contradictions,
    }
