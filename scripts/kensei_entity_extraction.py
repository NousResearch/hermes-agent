#!/usr/bin/env python3
"""
Kensei Entity Extraction & Consolidation Module
================================================
LLM-based entity extraction and consolidation for the Kensei Super-Memory Stack.

Provides:
  - extract_facts(text)        — LLM-based fact/entity extraction from conversation turns
  - consolidate_facts()        — Compare new facts against existing triples using
                                 vector similarity + LLM judge for ADD/UPDATE/DELETE/NOOP
  - extract_and_consolidate()  — Full pipeline: extract → consolidate → store
  - process_conversation_turn()— Integration point for conversation pipeline
  - ExtractionCache            — LRU cache for frequent extractions

All functions integrate with Mnemosyne's TripleStore and embedding system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kensei.entity_extraction")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_MNEMOSYNE_HOME = Path(os.environ.get("MNEMOSYNE_HOME", str(_HERMES_HOME / "mnemosyne")))
_DEFAULT_BANK = os.environ.get("KENSEI_MEMORY_BANK", "default")

# Cache defaults
_DEFAULT_CACHE_SIZE = int(os.environ.get("KENSEI_EXTRACTION_CACHE_SIZE", "256"))
_DEFAULT_CACHE_TTL = int(os.environ.get("KENSEI_EXTRACTION_CACHE_TTL", "3600"))  # 1 hour

# Similarity threshold for vector-based dedup (cosine similarity, 0-1)
_SIMILARITY_THRESHOLD = float(os.environ.get("KENSEI_CONSOLIDATION_SIMILARITY", "0.85"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFact:
    """A single extracted fact triple with metadata."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source: str = "extraction"
    valid_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "valid_from": self.valid_from,
        }

    def key(self) -> str:
        """Unique key for dedup: (subject, predicate, object)."""
        return f"{self.subject}|{self.predicate}|{self.object}"


@dataclass
class ConsolidationDecision:
    """Result of comparing a new fact against existing triples."""
    action: str  # "ADD", "UPDATE", "DELETE", "NOOP"
    reason: str
    existing_triple: Optional[Dict[str, Any]] = None
    new_fact: Optional[ExtractedFact] = None


# ---------------------------------------------------------------------------
# Extraction Cache (LRU with TTL)
# ---------------------------------------------------------------------------


class ExtractionCache:
    """LRU cache for extraction results with TTL expiry.

    Caches the full extraction result (entities + triples) keyed by
    a hash of the input text. Reduces redundant LLM calls when the
    same or very similar text is processed multiple times.
    """

    def __init__(self, maxsize: int = _DEFAULT_CACHE_SIZE, ttl: int = _DEFAULT_CACHE_TTL):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        """Generate a deterministic cache key from input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached extraction result. Returns None on miss or expiry."""
        key = self._make_key(text)
        if key not in self._cache:
            self._misses += 1
            return None

        timestamp, result = self._cache[key]
        if time.monotonic() - timestamp > self._ttl:
            # Expired — remove and treat as miss
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return result

    def put(self, text: str, result: Dict[str, Any]) -> None:
        """Store extraction result in cache."""
        key = self._make_key(text)
        now = time.monotonic()

        # Evict oldest if at capacity
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[key] = (now, result)

    def invalidate(self, text: str) -> None:
        """Remove a specific entry from cache."""
        key = self._make_key(text)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
        }


# Global cache instance
_extraction_cache = ExtractionCache()


# ---------------------------------------------------------------------------
# LLM helper (reuses the MCP server's _call_llm)
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, max_tokens: int = 2000, timeout: int = 120) -> Optional[str]:
    """Call the configured auxiliary LLM.

    Tries the Hermes auxiliary client first, falls back to direct Ollama call.
    """
    # Try the Hermes auxiliary client first
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task="compression",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Auxiliary LLM call failed: %s", exc)

    # Fallback: direct Ollama call
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("KENSEI_EXTRACTION_MODEL", "tinyllama:1.1b-chat")
    try:
        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Ollama fallback LLM call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Embedding helper (reuses Mnemosyne's embedding system)
# ---------------------------------------------------------------------------


def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector for a text string.

    Uses Mnemosyne's embedding system (fastembed local or API-based).
    Returns None if embeddings are unavailable.
    """
    try:
        from mnemosyne.core.embeddings import embed_query
        import numpy as np

        vec = embed_query(text)
        if vec is not None and np is not None:
            return vec.tolist()
        return None
    except ImportError:
        logger.debug("mnemosyne.core.embeddings not available")
        return None
    except Exception as exc:
        logger.debug("Embedding failed: %s", exc)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Triple store helpers
# ---------------------------------------------------------------------------


def _get_triple_store(bank: str = _DEFAULT_BANK):
    """Get a TripleStore instance pointing at the given bank."""
    try:
        from mnemosyne.core.triples import TripleStore
    except ImportError:
        logger.error("mnemosyne.core.triples not available")
        return None

    # Resolve the bank's DB path
    bank_dir = _MNEMOSYNE_HOME / "banks" / bank
    db_path = bank_dir / "mnemosyne.db"
    if not db_path.exists():
        # Fall back to default triples DB
        db_path = _MNEMOSYNE_HOME / "data" / "triples.db"
    return TripleStore(db_path=db_path)


def _get_existing_triples(bank: str = _DEFAULT_BANK) -> List[Dict[str, Any]]:
    """Get all current (non-expired) triples from the triple store."""
    store = _get_triple_store(bank)
    if store is None:
        return []
    try:
        today = __import__("datetime").datetime.now().isoformat()[:10]
        # Use the provider-supported query(as_of=) which returns all
        # current (non-expired) facts — semantically identical to the
        # previously-called get_facts_valid_at(today) that was never
        # shipped in the installed mnemosyne package.
        return store.query(as_of=today)
    except Exception as exc:
        logger.debug("Failed to query existing triples: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a fact extraction system. Given a text passage, extract all factual
statements as structured triples (subject, predicate, object). Also extract
named entities (people, places, organizations, concepts).

Rules:
- Each triple must be a verifiable fact stated in the text, not inferred.
- Subject and object should be specific entities or concepts.
- Predicate should be a concise relationship verb (e.g. "works_at", "lives_in",
  "prefers", "created", "is_a", "has_property", "assigned_to", "role").
- Assign a confidence score (0.0-1.0) to each triple based on how directly
  the fact is stated. Direct statements = 0.9-1.0, implied = 0.6-0.8.
- Return ONLY valid JSON with no extra text.

Output format:
{
    "entities": ["entity1", "entity2", ...],
    "triples": [
        {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}
    ]
}

Text:
"""


def _parse_llm_json(llm_response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown-wrapped output."""
    if not llm_response:
        return None
    json_str = llm_response
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Public API: extract_facts
# ---------------------------------------------------------------------------


def extract_facts(
    text: str,
    use_cache: bool = True,
    max_tokens: int = 2000,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Extract entities and fact triples from text using LLM.

    Args:
        text: The conversation turn text to extract from.
        use_cache: If True, check cache before calling LLM.
        max_tokens: Max tokens for LLM response.
        timeout: Timeout in seconds for LLM call.

    Returns:
        Dict with keys:
            - status: "ok" or "error"
            - entities: list of entity strings
            - triples: list of ExtractedFact objects (as dicts)
            - message: error message if status is "error"
            - cached: True if result was from cache
    """
    if not text or not text.strip():
        return {
            "status": "error",
            "message": "text is required",
            "entities": [],
            "triples": [],
            "cached": False,
        }

    # Check cache
    if use_cache:
        cached = _extraction_cache.get(text)
        if cached is not None:
            logger.debug("Extraction cache HIT for text of length %d", len(text))
            return {**cached, "cached": True}

    # Call LLM
    prompt = _EXTRACTION_PROMPT + text
    llm_response = _call_llm(prompt, max_tokens=max_tokens, timeout=timeout)

    if not llm_response:
        result = {
            "status": "error",
            "message": "LLM extraction failed — no response from model",
            "entities": [],
            "triples": [],
            "cached": False,
        }
        return result

    # Parse JSON
    parsed = _parse_llm_json(llm_response)
    if parsed is None:
        result = {
            "status": "error",
            "message": f"LLM returned unparseable JSON: {llm_response[:200]}",
            "entities": [],
            "triples": [],
            "cached": False,
        }
        return result

    entities = parsed.get("entities", [])
    raw_triples = parsed.get("triples", [])

    # Convert to ExtractedFact objects
    triples = []
    for t in raw_triples:
        subj = t.get("subject", "").strip()
        pred = t.get("predicate", "").strip()
        obj = t.get("object", "").strip()
        conf = t.get("confidence", 0.8)
        if all([subj, pred, obj]):
            triples.append(ExtractedFact(
                subject=subj,
                predicate=pred,
                object=obj,
                confidence=min(max(float(conf), 0.0), 1.0),
                source="extraction",
            ))

    result = {
        "status": "ok",
        "entities": entities,
        "triples": [t.to_dict() for t in triples],
        "message": None,
        "cached": False,
    }

    # Store in cache
    if use_cache:
        _extraction_cache.put(text, result)

    return result


# ---------------------------------------------------------------------------
# Consolidation prompt
# ---------------------------------------------------------------------------

_CONSOLIDATION_PROMPT = """\
You are a fact consolidation judge. Given a NEW fact (as a subject-predicate-object
triple) and a list of EXISTING facts from a knowledge graph, decide what action
to take.

Possible actions:
- ADD: The new fact is novel and should be added to the knowledge graph.
- UPDATE: The new fact contradicts or supersedes an existing fact with the same
  subject and predicate. The existing fact should be invalidated and the new one
  added.
- DELETE: The new fact explicitly negates or retracts an existing fact. The
  existing fact should be invalidated.
- NOOP: The new fact is already represented (same or equivalent triple exists),
  or is not sufficiently novel to warrant storage.

Consider:
1. If the same (subject, predicate, object) already exists → NOOP
2. If the same (subject, predicate) has a DIFFERENT object → UPDATE
3. If the new fact negates an existing one (e.g. "no longer works_at") → DELETE
4. If the new fact is a minor variation of an existing one → NOOP
5. If the new fact adds genuinely new information → ADD

Return ONLY valid JSON with no extra text.

Output format:
{{"action": "ADD|UPDATE|DELETE|NOOP", "reason": "Brief explanation of the decision", "confidence": 0.95}}

Existing facts:
{existing_facts}

New fact:
Subject: {subject}
Predicate: {predicate}
Object: {object}
"""


def _format_existing_for_prompt(triples: List[Dict[str, Any]]) -> str:
    """Format existing triples for the consolidation prompt."""
    lines = []
    for i, t in enumerate(triples, 1):
        lines.append(
            f"{i}. ({t.get('subject', '?')}, {t.get('predicate', '?')}, "
            f"{t.get('object', '?')}) — confidence: {t.get('confidence', 1.0)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API: consolidate_facts
# ---------------------------------------------------------------------------


def consolidate_facts(
    new_facts: List[ExtractedFact],
    bank: str = _DEFAULT_BANK,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    use_llm_judge: bool = True,
) -> List[ConsolidationDecision]:
    """Compare new facts against existing triples and decide actions.

    Two-stage consolidation:
    1. Vector similarity pass — find semantically similar existing triples
    2. LLM judge pass — decide ADD/UPDATE/DELETE/NOOP for each new fact

    Args:
        new_facts: List of ExtractedFact objects to consolidate.
        bank: Memory bank to query existing triples from.
        similarity_threshold: Cosine similarity threshold for vector dedup.
        use_llm_judge: If True, use LLM for nuanced decisions. If False,
                       use rule-based logic only.

    Returns:
        List of ConsolidationDecision objects.
    """
    if not new_facts:
        return []

    # Get existing triples
    existing = _get_existing_triples(bank)
    if not existing:
        # No existing facts — everything is ADD
        return [
            ConsolidationDecision(action="ADD", reason="No existing facts in store", new_fact=f)
            for f in new_facts
        ]

    # Build embedding cache for existing triples
    existing_embeddings: Dict[int, Optional[List[float]]] = {}
    for i, t in enumerate(existing):
        text = f"{t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}"
        existing_embeddings[i] = _get_embedding(text)

    decisions: List[ConsolidationDecision] = []

    for fact in new_facts:
        fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
        fact_embedding = _get_embedding(fact_text)

        # Stage 1: Vector similarity — find most similar existing triple
        most_similar_idx = -1
        highest_similarity = 0.0

        if fact_embedding is not None:
            for i, emb in existing_embeddings.items():
                if emb is not None:
                    sim = _cosine_similarity(fact_embedding, emb)
                    if sim > highest_similarity:
                        highest_similarity = sim
                        most_similar_idx = i

        # Stage 2: Decision
        if use_llm_judge and most_similar_idx >= 0 and highest_similarity >= similarity_threshold:
            # Use LLM judge for nuanced decision
            decision = _llm_consolidation_judge(fact, existing[most_similar_idx])
            decisions.append(decision)
        else:
            # Rule-based fallback
            decision = _rule_based_consolidation(fact, existing, most_similar_idx, highest_similarity, similarity_threshold)
            decisions.append(decision)

    return decisions


def _llm_consolidation_judge(
    new_fact: ExtractedFact,
    similar_existing: Dict[str, Any],
) -> ConsolidationDecision:
    """Use LLM to decide the consolidation action for a new fact."""
    # Format existing facts — include the most similar one plus any with same subject
    existing_for_prompt = [similar_existing]

    prompt = _CONSOLIDATION_PROMPT.format(
        existing_facts=_format_existing_for_prompt(existing_for_prompt),
        subject=new_fact.subject,
        predicate=new_fact.predicate,
        object=new_fact.object,
    )

    llm_response = _call_llm(prompt, max_tokens=500, timeout=60)
    if not llm_response:
        # Fall back to rule-based
        return ConsolidationDecision(
            action="ADD",
            reason="LLM judge unavailable; defaulting to ADD",
            existing_triple=similar_existing,
            new_fact=new_fact,
        )

    parsed = _parse_llm_json(llm_response)
    if parsed is None:
        return ConsolidationDecision(
            action="ADD",
            reason=f"LLM judge returned unparseable response; defaulting to ADD",
            existing_triple=similar_existing,
            new_fact=new_fact,
        )

    action = parsed.get("action", "ADD").upper()
    reason = parsed.get("reason", "LLM judge decision")

    # Validate action
    if action not in ("ADD", "UPDATE", "DELETE", "NOOP"):
        action = "ADD"
        reason = f"LLM returned invalid action '{action}'; defaulting to ADD"

    return ConsolidationDecision(
        action=action,
        reason=reason,
        existing_triple=similar_existing,
        new_fact=new_fact,
    )


def _rule_based_consolidation(
    new_fact: ExtractedFact,
    existing: List[Dict[str, Any]],
    most_similar_idx: int,
    highest_similarity: float,
    similarity_threshold: float,
) -> ConsolidationDecision:
    """Rule-based consolidation decision (fallback when LLM is unavailable)."""
    # Check for exact match
    for t in existing:
        if (t.get("subject", "").lower() == new_fact.subject.lower()
                and t.get("predicate", "").lower() == new_fact.predicate.lower()
                and t.get("object", "").lower() == new_fact.object.lower()):
            return ConsolidationDecision(
                action="NOOP",
                reason="Exact triple already exists",
                existing_triple=t,
                new_fact=new_fact,
            )

    # Check for same (subject, predicate) with different object → UPDATE
    for t in existing:
        if (t.get("subject", "").lower() == new_fact.subject.lower()
                and t.get("predicate", "").lower() == new_fact.predicate.lower()
                and t.get("object", "").lower() != new_fact.object.lower()):
            return ConsolidationDecision(
                action="UPDATE",
                reason=f"Same (subject, predicate) with different object: "
                       f"'{t.get('object')}' → '{new_fact.object}'",
                existing_triple=t,
                new_fact=new_fact,
            )

    # Check for high vector similarity → NOOP (semantic duplicate)
    if most_similar_idx >= 0 and highest_similarity >= similarity_threshold:
        return ConsolidationDecision(
            action="NOOP",
            reason=f"Semantically similar to existing triple "
                   f"(cosine similarity: {highest_similarity:.3f})",
            existing_triple=existing[most_similar_idx],
            new_fact=new_fact,
        )

    # Default: ADD
    return ConsolidationDecision(
        action="ADD",
        reason="Novel fact not found in existing triples",
        new_fact=new_fact,
    )


# ---------------------------------------------------------------------------
# Public API: extract_and_consolidate (full pipeline)
# ---------------------------------------------------------------------------


def extract_and_consolidate(
    text: str,
    bank: str = _DEFAULT_BANK,
    use_cache: bool = True,
    use_llm_judge: bool = True,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    store_results: bool = True,
) -> Dict[str, Any]:
    """Full pipeline: extract facts → consolidate against existing → store.

    Args:
        text: Conversation turn text to process.
        bank: Memory bank to use.
        use_cache: If True, use extraction cache.
        use_llm_judge: If True, use LLM for consolidation decisions.
        similarity_threshold: Cosine similarity threshold for vector dedup.
        store_results: If True, store ADD/UPDATE decisions in triple store.

    Returns:
        Dict with extraction and consolidation results.
    """
    # Step 1: Extract
    extraction = extract_facts(text, use_cache=use_cache)
    if extraction["status"] == "error":
        return {
            "status": "error",
            "message": extraction["message"],
            "extraction": extraction,
            "consolidation": [],
            "stored": 0,
            "cached": extraction.get("cached", False),
        }

    # Convert dict triples back to ExtractedFact
    new_facts = []
    for t in extraction["triples"]:
        new_facts.append(ExtractedFact(
            subject=t["subject"],
            predicate=t["predicate"],
            object=t["object"],
            confidence=t.get("confidence", 0.8),
            source="extraction",
        ))

    if not new_facts:
        return {
            "status": "ok",
            "message": "No facts extracted from text",
            "extraction": extraction,
            "consolidation": [],
            "stored": 0,
            "cached": extraction.get("cached", False),
        }

    # Step 2: Consolidate
    decisions = consolidate_facts(
        new_facts=new_facts,
        bank=bank,
        similarity_threshold=similarity_threshold,
        use_llm_judge=use_llm_judge,
    )

    # Step 3: Store
    stored_count = 0
    stored_facts = []
    errors = []

    if store_results:
        store = _get_triple_store(bank)
        if store is not None:
            for decision in decisions:
                if decision.action == "ADD":
                    try:
                        store.add(
                            subject=decision.new_fact.subject,
                            predicate=decision.new_fact.predicate,
                            object=decision.new_fact.object,
                            source=decision.new_fact.source,
                            confidence=decision.new_fact.confidence,
                        )
                        stored_count += 1
                        stored_facts.append(decision.new_fact.to_dict())
                    except Exception as exc:
                        errors.append(f"Failed to store ({decision.new_fact.subject}, "
                                      f"{decision.new_fact.predicate}, "
                                      f"{decision.new_fact.object}): {exc}")

                elif decision.action == "UPDATE":
                    try:
                        # Add new fact (auto-invalidates old)
                        store.add(
                            subject=decision.new_fact.subject,
                            predicate=decision.new_fact.predicate,
                            object=decision.new_fact.object,
                            source=decision.new_fact.source,
                            confidence=decision.new_fact.confidence,
                        )
                        stored_count += 1
                        stored_facts.append(decision.new_fact.to_dict())
                    except Exception as exc:
                        errors.append(f"Failed to update ({decision.new_fact.subject}, "
                                      f"{decision.new_fact.predicate}): {exc}")

                elif decision.action == "DELETE":
                    try:
                        # Invalidate the existing triple by setting valid_until
                        if decision.existing_triple:
                            existing_id = decision.existing_triple.get("id")
                            if existing_id:
                                store.conn.execute(
                                    "UPDATE triples SET valid_until = ? WHERE id = ?",
                                    (
                                        __import__("datetime").datetime.now().isoformat()[:10],
                                        existing_id,
                                    ),
                                )
                                store.conn.commit()
                            else:
                                # Fallback: invalidate by (subject, predicate, object)
                                store.conn.execute(
                                    "UPDATE triples SET valid_until = ? "
                                    "WHERE subject = ? AND predicate = ? AND object = ? "
                                    "AND valid_until IS NULL",
                                    (
                                        __import__("datetime").datetime.now().isoformat()[:10],
                                        decision.new_fact.subject,
                                        decision.new_fact.predicate,
                                        decision.existing_triple.get("object"),
                                    ),
                                )
                                store.conn.commit()
                    except Exception as exc:
                        errors.append(f"Failed to delete ({decision.new_fact.subject}, "
                                      f"{decision.new_fact.predicate}): {exc}")

    # Format decisions for output
    decision_output = []
    for d in decisions:
        decision_output.append({
            "action": d.action,
            "reason": d.reason,
            "new_fact": d.new_fact.to_dict() if d.new_fact else None,
            "existing_triple": d.existing_triple,
        })

    return {
        "status": "ok",
        "extraction": {
            "entities": extraction["entities"],
            "triples_extracted": len(extraction["triples"]),
        },
        "consolidation": decision_output,
        "stored": stored_count,
        "errors": errors if errors else None,
        "cached": extraction.get("cached", False),
    }


# ---------------------------------------------------------------------------
# Conversation pipeline integration
# ---------------------------------------------------------------------------


def process_conversation_turn(
    user_message: str,
    assistant_response: str,
    bank: str = _DEFAULT_BANK,
    use_cache: bool = True,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Process a full conversation turn (user + assistant) for fact extraction.

    This is the primary integration point for the conversation pipeline.
    It extracts facts from both the user message and assistant response,
    consolidates them, and stores new facts in the triple store.

    Args:
        user_message: The user's message text.
        assistant_response: The assistant's response text.
        bank: Memory bank to use.
        use_cache: If True, use extraction cache.
        use_llm_judge: If True, use LLM for consolidation decisions.

    Returns:
        Dict with combined extraction and consolidation results.
    """
    combined_text = f"User: {user_message}\nAssistant: {assistant_response}"

    result = extract_and_consolidate(
        text=combined_text,
        bank=bank,
        use_cache=use_cache,
        use_llm_judge=use_llm_judge,
        store_results=True,
    )

    return result


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def get_cache_stats() -> Dict[str, Any]:
    """Get extraction cache statistics."""
    return _extraction_cache.stats


def clear_cache() -> None:
    """Clear the extraction cache."""
    _extraction_cache.clear()


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run extraction and consolidation from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kensei Entity Extraction & Consolidation"
    )
    parser.add_argument("text", nargs="?", help="Text to extract facts from")
    parser.add_argument("--file", "-f", help="Read text from file")
    parser.add_argument("--bank", default=_DEFAULT_BANK, help="Memory bank")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Disable LLM judge (use rule-based only)")
    parser.add_argument("--similarity", type=float, default=_SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold (default: {_SIMILARITY_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and consolidate without storing")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Show cache statistics and exit")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear extraction cache and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.cache_stats:
        print(json.dumps(get_cache_stats(), indent=2))
        return

    if args.clear_cache:
        clear_cache()
        print("Cache cleared")
        return

    # Get text
    text = args.text
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()

    if not text:
        parser.print_help()
        return

    # Run pipeline
    result = extract_and_consolidate(
        text=text,
        bank=args.bank,
        use_cache=not args.no_cache,
        use_llm_judge=not args.no_llm_judge,
        similarity_threshold=args.similarity,
        store_results=not args.dry_run,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
