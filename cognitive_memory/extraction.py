"""
Fact Extraction and Forgetting System for cognitive memory.

Responsible for:
  - Extracting memorable facts from conversation turns
  - Determining what's worth remembering vs. ephemeral
  - Automated forgetting (decay + pruning) lifecycle
  - Consolidating duplicate/overlapping memories
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_memory.encoding import encode, EncodingResult
from cognitive_memory.recall import RecallEngine
from cognitive_memory.store import CognitiveStore, Memory, ScoredMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Patterns that signal memorable content in conversation
_MEMORABLE_PATTERNS = [
    # User sharing information about themselves
    re.compile(r"\b(my name is|i'm called|call me)\b", re.I),
    re.compile(r"\b(i work at|i'm a|my role is|my job is)\b", re.I),
    re.compile(r"\b(i live in|my timezone|i'm in|i'm from)\b", re.I),
    re.compile(r"\b(i prefer|i like|i hate|i always|i never)\b", re.I),
    # Explicit remember requests
    re.compile(r"\b(remember that|keep in mind|note that|don't forget)\b", re.I),
    re.compile(r"\b(for future reference|going forward|from now on)\b", re.I),
    # Environment/project facts
    re.compile(r"\b(the project uses|we use|our stack|the codebase)\b", re.I),
    re.compile(r"\b(the api|the endpoint|the database|the server)\b", re.I),
    # Corrections
    re.compile(r"\b(actually|no,|wrong|that's incorrect|correction)\b", re.I),
    # Conventions
    re.compile(r"\b(we always|we never|the convention is|the standard is)\b", re.I),
    re.compile(r"\b(the rule is|coding style|naming convention)\b", re.I),
]

# Patterns that signal ephemeral/not-worth-remembering content
_EPHEMERAL_PATTERNS = [
    re.compile(r"^(ok|okay|sure|yes|no|thanks|thank you|got it|understood)\.?$", re.I),
    re.compile(r"^(hi|hello|hey|good morning|good evening)\.?$", re.I),
    re.compile(r"\b(what time is it|what's the weather)\b", re.I),
    re.compile(r"^.{0,15}$"),  # Very short messages
]

# Minimum content length to consider for extraction
MIN_CONTENT_LENGTH = 20


@dataclass
class ExtractedFact:
    """A fact extracted from conversation text."""
    content: str
    source_text: str
    encoding: EncodingResult
    scope: str = "/"


@dataclass
class ForgettingResult:
    """Result of a forgetting cycle."""
    decayed_count: int
    pruned_count: int
    consolidated_count: int
    total_active: int


def is_memorable(text: str) -> bool:
    """
    Determine if text contains information worth remembering.

    Uses pattern matching to distinguish memorable content from
    ephemeral chatter.
    """
    text = text.strip()

    if len(text) < MIN_CONTENT_LENGTH:
        return False

    # Check ephemeral patterns first (fast rejection)
    for pattern in _EPHEMERAL_PATTERNS:
        if pattern.match(text):
            return False

    # Check for memorable signals
    for pattern in _MEMORABLE_PATTERNS:
        if pattern.search(text):
            return True

    # Longer messages with substance are worth considering
    word_count = len(text.split())
    if word_count >= 8:
        return True

    return False


def extract_facts(
    text: str,
    scope: str = "/",
    role: str = "user",
) -> List[ExtractedFact]:
    """
    Extract memorable facts from a conversation turn.

    Splits text into sentences and evaluates each for memorability.
    Returns a list of ExtractedFact with encoding metadata.

    Args:
        text: The conversation text to extract from
        scope: Scope prefix for categorization
        role: Message role ('user' or 'assistant')
    """
    if not text or not text.strip():
        return []

    # For assistant messages, only extract explicit observations
    if role == "assistant":
        # Assistants don't generate memorable facts about themselves
        return []

    # Split into sentences
    sentences = _split_sentences(text)

    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not is_memorable(sentence):
            continue

        encoding = encode(sentence)
        facts.append(ExtractedFact(
            content=sentence,
            source_text=text,
            encoding=encoding,
            scope=scope,
        ))

    # If no individual sentences were memorable but the full text is
    if not facts and is_memorable(text):
        encoding = encode(text)
        facts.append(ExtractedFact(
            content=text,
            source_text=text,
            encoding=encoding,
            scope=scope,
        ))

    return facts


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Also split on newlines
    result = []
    for part in parts:
        result.extend(part.split('\n'))
    return [s.strip() for s in result if s.strip()]


class ForgettingManager:
    """
    Manages the forgetting lifecycle for cognitive memory.

    Operations:
      - decay: Reduce importance of memories over time
      - prune: Remove memories that have decayed below threshold
      - consolidate: Merge highly similar memories into one
    """

    def __init__(
        self,
        store: CognitiveStore,
        decay_half_life_days: float = 30.0,
        prune_threshold: float = 0.05,
        consolidation_similarity: float = 0.92,
    ):
        self._store = store
        self._decay_half_life = decay_half_life_days
        self._prune_threshold = prune_threshold
        self._consolidation_similarity = consolidation_similarity
        self._last_cycle_run: Optional[float] = None

    def run_forgetting_cycle(
        self,
        exempt_scopes: Optional[List[str]] = None,
    ) -> ForgettingResult:
        """
        Run a complete forgetting cycle: decay -> consolidate -> prune.

        Args:
            exempt_scopes: Scope prefixes exempt from decay (e.g., ["/user"])

        Returns:
            ForgettingResult with counts of affected memories
        """
        # Step 1: Decay importance based on time since last access
        decayed = self._store.decay_importance(
            half_life_days=self._decay_half_life,
            exempt_scopes=exempt_scopes,
        )

        # Step 2: Consolidate similar memories
        consolidated = self._consolidate_memories()

        # Step 3: Prune memories below threshold
        pruned = self._store.prune(threshold=self._prune_threshold)

        total_active = self._store.count()

        return ForgettingResult(
            decayed_count=decayed,
            pruned_count=pruned,
            consolidated_count=consolidated,
            total_active=total_active,
        )

    def _consolidate_memories(self) -> int:
        """
        Find and merge highly similar active memories.

        When two memories are almost identical (similarity > threshold),
        keep the one with higher importance and soft-delete the other.

        Returns the number of memories consolidated.
        """
        from cognitive_memory.embeddings import cosine_similarity

        active = self._store.get_all_active()
        # Only check memories that have embeddings
        with_embeddings = [m for m in active if m.embedding is not None]

        if len(with_embeddings) < 2:
            return 0

        consolidated = 0
        deleted_ids = set()

        for i in range(len(with_embeddings)):
            if with_embeddings[i].id in deleted_ids:
                continue
            for j in range(i + 1, len(with_embeddings)):
                if with_embeddings[j].id in deleted_ids:
                    continue

                sim = cosine_similarity(
                    with_embeddings[i].embedding,
                    with_embeddings[j].embedding,
                )

                if sim >= self._consolidation_similarity:
                    # Keep the one with higher importance
                    keep, remove = (
                        (with_embeddings[i], with_embeddings[j])
                        if with_embeddings[i].importance >= with_embeddings[j].importance
                        else (with_embeddings[j], with_embeddings[i])
                    )

                    # Boost kept memory's importance slightly
                    new_importance = min(keep.importance + 0.05, 1.0)
                    self._store.update_memory(
                        keep.id, importance=new_importance
                    )

                    # Soft-delete the duplicate
                    self._store.soft_delete(remove.id)
                    deleted_ids.add(remove.id)
                    consolidated += 1

                    logger.debug(
                        "Consolidated memory #%d into #%d (sim=%.3f)",
                        remove.id, keep.id, sim,
                    )

        return consolidated

    def should_run_cycle(self, last_run_time: Optional[float] = None) -> bool:
        """
        Determine if a forgetting cycle should run.

        Runs at most once every 6 hours and only if there are enough memories.
        """
        min_interval = 6 * 3600  # 6 hours

        if last_run_time is not None:
            elapsed = time.time() - last_run_time
            if elapsed < min_interval:
                return False

        return self._store.count() >= 5

    def maybe_run_cycle(
        self, exempt_scopes: Optional[List[str]] = None,
    ) -> Optional[ForgettingResult]:
        """
        Run a forgetting cycle if enough time has passed and enough memories exist.

        Tracks last run time internally.
        """
        if not self.should_run_cycle(self._last_cycle_run):
            return None

        result = self.run_forgetting_cycle(exempt_scopes)
        self._last_cycle_run = time.time()
        logger.info(
            "Forgetting cycle: decayed=%d, consolidated=%d, pruned=%d, active=%d",
            result.decayed_count, result.consolidated_count,
            result.pruned_count, result.total_active,
        )
        return result
