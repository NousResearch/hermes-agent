"""
Cognitive Encoding: auto-classification and contradiction detection.

Responsible for:
  - Categorizing incoming memories by content type (fact, preference,
    procedure, observation, etc.)
  - Estimating importance based on content signals
  - Detecting contradictions between a new memory and existing ones

Uses simple heuristic classifiers (no LLM call required) for speed.
LLM-based refinement can be layered on top for higher accuracy.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_memory.embeddings import cosine_similarity
from cognitive_memory.store import CognitiveStore, Memory, ScoredMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "fact": "Objective factual information",
    "preference": "User preference or opinion",
    "procedure": "How-to, step-by-step process",
    "observation": "Subjective observation or note",
    "convention": "Project convention or coding standard",
    "environment": "Environment or setup detail",
    "correction": "Correction of prior information",
    "skill": "Learned capability or technique",
}

# Heuristic patterns for category classification
_CATEGORY_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    ("preference", [
        re.compile(r"\b(prefer|like|dislike|hate|love|want|always use|never use|favorite)\b", re.I),
        re.compile(r"\b(style|choice|rather|instead of)\b", re.I),
    ]),
    ("procedure", [
        re.compile(r"\b(step \d|first|then|after that|finally|how to|to do this)\b", re.I),
        re.compile(r"\b(run|execute|install|configure|setup|deploy)\b", re.I),
    ]),
    ("convention", [
        re.compile(r"\b(convention|standard|pattern|rule|always|never|must|should)\b", re.I),
        re.compile(r"\b(naming|format|style guide|linting|indentation)\b", re.I),
    ]),
    ("environment", [
        re.compile(r"\b(os|operating system|macos|linux|windows|python|node|version)\b", re.I),
        re.compile(r"\b(installed|configured|path|directory|env|environment)\b", re.I),
    ]),
    ("correction", [
        re.compile(r"\b(actually|correction|wrong|incorrect|not true|mistake|update)\b", re.I),
        re.compile(r"\b(instead|rather|should be|corrected|fix)\b", re.I),
    ]),
    ("skill", [
        re.compile(r"\b(learned|discovered|figured out|trick|technique|approach)\b", re.I),
        re.compile(r"\b(solved|workaround|solution|method)\b", re.I),
    ]),
    ("observation", [
        re.compile(r"\b(noticed|seems|appears|looks like|observed|found that)\b", re.I),
        re.compile(r"\b(interesting|note|curious|surprisingly)\b", re.I),
    ]),
    ("fact", [
        re.compile(r"\b(is|are|was|were|has|have|uses|runs|supports|contains)\b", re.I),
    ]),
]

# Importance signals
_HIGH_IMPORTANCE_PATTERNS = [
    re.compile(r"\b(critical|important|essential|must|never|always|key)\b", re.I),
    re.compile(r"\b(security|password|secret|credential|api.?key)\b", re.I),
    re.compile(r"\b(breaking|dangerous|careful|warning|caution)\b", re.I),
]

_LOW_IMPORTANCE_PATTERNS = [
    re.compile(r"\b(minor|trivial|maybe|possibly|might|could)\b", re.I),
    re.compile(r"\b(temporary|for now|quick|hack)\b", re.I),
]


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

# Negation and opposition patterns for contradiction detection
_NEGATION_WORDS = {
    "not", "no", "never", "don't", "doesn't", "didn't", "won't",
    "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
    "shouldn't", "wouldn't", "couldn't", "nor", "neither",
    "disable", "disabled", "remove", "removed", "stop", "stopped",
    "false", "off",
}

_AFFIRMATION_WORDS = {
    "is", "are", "was", "were", "does", "do", "did", "will",
    "can", "should", "would", "could", "has", "have", "had",
    "enable", "enabled", "add", "added", "start", "started",
    "true", "on", "use", "using", "uses",
}


@dataclass
class ContradictionResult:
    """Result of contradiction detection between two memories."""
    is_contradiction: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    existing_memory: Optional[Memory] = None


@dataclass
class EncodingResult:
    """Result of encoding a piece of content."""
    categories: List[str]
    importance: float
    contradictions: List[ContradictionResult] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)


def classify_content(text: str) -> Tuple[List[str], Dict[str, float]]:
    """
    Classify content into categories using heuristic pattern matching.

    Returns:
        Tuple of (categories list, category_scores dict)
    """
    scores: Dict[str, float] = {}
    words = text.lower().split()
    word_count = max(len(words), 1)

    for category, patterns in _CATEGORY_PATTERNS:
        match_count = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            match_count += len(matches)

        if match_count > 0:
            # Normalize by word count so short texts aren't penalized
            score = min(match_count / (word_count * 0.3), 1.0)
            scores[category] = score

    # If no patterns matched, default to "observation"
    if not scores:
        scores["observation"] = 0.3

    # Return categories above a minimum threshold
    threshold = 0.15
    categories = [cat for cat, score in sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    ) if score >= threshold]

    # Limit to top 3 categories
    categories = categories[:3]

    return categories, scores


def estimate_importance(text: str, categories: List[str]) -> float:
    """
    Estimate the importance of content based on signals in the text.

    Returns a value between 0.0 and 1.0.
    """
    base = 0.5

    # Boost for high-importance signals
    high_matches = sum(
        len(p.findall(text)) for p in _HIGH_IMPORTANCE_PATTERNS
    )
    if high_matches > 0:
        base += min(high_matches * 0.1, 0.3)

    # Reduce for low-importance signals
    low_matches = sum(
        len(p.findall(text)) for p in _LOW_IMPORTANCE_PATTERNS
    )
    if low_matches > 0:
        base -= min(low_matches * 0.08, 0.2)

    # Category-based adjustments
    if "correction" in categories:
        base += 0.15  # Corrections are important - they fix prior knowledge
    if "preference" in categories:
        base += 0.1
    if "convention" in categories:
        base += 0.05

    return max(0.1, min(1.0, base))


def _extract_key_terms(text: str) -> set:
    """Extract meaningful terms from text for comparison."""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "as", "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
        "our", "their", "me", "him", "us", "them", "and", "or", "but",
        "if", "so", "be", "been", "being", "am",
    }
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set(words) - stop_words


def detect_contradiction(
    new_text: str,
    existing: Memory,
    similarity: float = 0.0,
) -> ContradictionResult:
    """
    Detect if new text contradicts an existing memory.

    Uses a combination of:
      1. High semantic similarity (content is about the same topic)
      2. Negation pattern mismatch (one affirms, other negates)
      3. Key term overlap with opposing signals

    Returns ContradictionResult with confidence score.
    """
    new_lower = new_text.lower()
    existing_lower = existing.content.lower()

    # Need minimum similarity to even consider contradiction
    if similarity < 0.5:
        return ContradictionResult(
            is_contradiction=False,
            confidence=0.0,
            reason="low_similarity",
            existing_memory=existing,
        )

    new_words = set(new_lower.split())
    existing_words = set(existing_lower.split())

    new_negations = new_words & _NEGATION_WORDS
    existing_negations = existing_words & _NEGATION_WORDS
    new_affirmations = new_words & _AFFIRMATION_WORDS
    existing_affirmations = existing_words & _AFFIRMATION_WORDS

    # Check for negation asymmetry
    negation_conflict = False
    if (new_negations and existing_affirmations and not existing_negations):
        negation_conflict = True
    elif (existing_negations and new_affirmations and not new_negations):
        negation_conflict = True

    # Check key term overlap
    new_terms = _extract_key_terms(new_text)
    existing_terms = _extract_key_terms(existing.content)
    overlap = new_terms & existing_terms
    overlap_ratio = len(overlap) / max(len(new_terms | existing_terms), 1)

    # Compute contradiction confidence
    confidence = 0.0

    if negation_conflict:
        confidence += 0.4

    if overlap_ratio > 0.3:
        confidence += 0.2 * overlap_ratio

    # High similarity with negation conflict is strong signal
    if similarity > 0.7 and negation_conflict:
        confidence += 0.2

    # Explicit correction phrases
    correction_phrases = [
        "actually", "not true", "wrong", "incorrect",
        "correction", "should be", "instead",
    ]
    if any(phrase in new_lower for phrase in correction_phrases):
        if overlap_ratio > 0.2 or similarity > 0.6:
            confidence += 0.3
        # High similarity + correction phrase is very strong signal
        if similarity > 0.7:
            confidence += 0.2

    confidence = min(confidence, 1.0)
    is_contradiction = confidence >= 0.5

    if is_contradiction:
        reason = "negation_conflict" if negation_conflict else "correction_detected"
    else:
        reason = "no_contradiction"

    return ContradictionResult(
        is_contradiction=is_contradiction,
        confidence=confidence,
        reason=reason,
        existing_memory=existing,
    )


def detect_contradictions(
    new_text: str,
    candidates: List[ScoredMemory],
    min_similarity: float = 0.5,
) -> List[ContradictionResult]:
    """
    Check new text against a list of candidate memories for contradictions.

    Args:
        new_text: The new content to check
        candidates: Semantically similar existing memories
        min_similarity: Minimum similarity to consider for contradiction

    Returns:
        List of ContradictionResults (only those detected as contradictions)
    """
    contradictions = []
    for sm in candidates:
        if sm.similarity < min_similarity:
            continue
        result = detect_contradiction(new_text, sm.memory, sm.similarity)
        if result.is_contradiction:
            contradictions.append(result)

    # Sort by confidence descending
    contradictions.sort(key=lambda c: c.confidence, reverse=True)
    return contradictions


def encode(
    text: str,
    candidates: Optional[List[ScoredMemory]] = None,
) -> EncodingResult:
    """
    Full encoding pipeline: classify, estimate importance, detect contradictions.

    Args:
        text: Content to encode
        candidates: Optional list of similar existing memories for contradiction check

    Returns:
        EncodingResult with categories, importance, and contradictions
    """
    categories, category_scores = classify_content(text)
    importance = estimate_importance(text, categories)

    contradictions = []
    if candidates:
        contradictions = detect_contradictions(text, candidates)

        # Boost importance if contradictions found (corrections are valuable)
        if contradictions:
            importance = min(importance + 0.15, 1.0)

    return EncodingResult(
        categories=categories,
        importance=importance,
        contradictions=contradictions,
        category_scores=category_scores,
    )
