"""ScoredMemory dataclass and hybrid scoring primitives.

ScoredMemory carries multi-dimensional relevance metadata so the fusion
layer can rank and compare entries across providers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metadata parsing from tag-annotated entries
# ---------------------------------------------------------------------------

_CONTEXT_TAG_RE = re.compile(r"^\[context:\s*(\w[\w-]*)\]\s*", re.DOTALL)
_SIGNAL_TAG_RE = re.compile(r"\[signal:\s*([0-9]*\.?[0-9]+)\]\s*", re.DOTALL)
_CONTEXT_DESCRIPTIONS: Dict[str, str] = {
    "deploy": "infrastructure, deployments, configuration changes",
    "feature": "building features (branch+PR, TDD, testing)",
    "debug": "root cause analysis, bug fixing, proving causation",
    "review": "code review, PR review, security audit",
    "research": "investigating options, competitor analysis, exploration",
    "planning": "strategy, architecture decisions, roadmap",
    "content": "writing, documentation, social media, changelogs",
    "maintenance": "repo org, cleanup, dep updates, caching",
    "kanban": "kanban board operations, task routing, orchestrator",
    "design": "UI/UX, mockups, visual design, component architecture",
}

DEFAULT_SIGNAL = 0.5
HYBRID_WEIGHTS = {
    "fts": 0.30,          # FTS5 keyword match
    "embedding": 0.40,    # Semantic similarity (future phase)
    "context_tag": 0.20,  # [context: X] tag match with task type
    "signal": 0.10,       # [signal: N] user-set importance
}


@dataclass
class ScoredMemory:
    """A memory entry with multi-dimensional relevance score.

    Each scoring dimension lives as a separate field so the fusion layer
    can inspect and re-weight across providers.
    """

    content: str                       # The memory entry text (markup stripped)
    provider: str                      # "builtin", "holographic", etc.
    score: float = 0.0                 # Composite relevance 0.0-1.0

    # Provenance
    source_file: Optional[str] = None  # "MEMORY.md" / "USER.md"
    source_path: Optional[str] = None
    entry_index: Optional[int] = None  # Position within provider's store

    # Extracted metadata
    context_tag: Optional[str] = None  # "deploy", "feature", ... or None
    signal_strength: float = DEFAULT_SIGNAL
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    access_count: int = 0

    # Scoring dimension breakdown
    score_semantic: float = 0.0    # Embedding similarity
    score_keyword: float = 0.0     # Keyword / FTS match
    score_context: float = 0.0     # Context-tag match score
    score_entity: float = 0.0      # Named-entity overlap
    score_temporal: float = 1.0    # Time-decay factor (1.0 = fresh)
    score_signal: float = DEFAULT_SIGNAL  # Normalised to 0-1

    # Calibration metadata
    calibrated: bool = False
    calibration_bias: float = 0.0

    @classmethod
    def from_raw_entry(
        cls,
        content: str,
        provider: str = "builtin",
        source_file: Optional[str] = None,
        source_path: Optional[str] = None,
        entry_index: Optional[int] = None,
    ) -> "ScoredMemory":
        """Parse a raw memory entry string with [context:] / [signal:] tags.

        Strips the markup tags from content and populates the structured fields.
        """
        ctx, content = parse_context_tag(content)
        signal, content = parse_signal_tag(content)

        return cls(
            content=content.strip(),
            provider=provider,
            context_tag=ctx,
            signal_strength=signal,
            score_signal=signal,
            source_file=source_file,
            source_path=source_path,
            entry_index=entry_index,
        )

    def compute_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute composite score from dimension scores and weights."""
        w = weights or HYBRID_WEIGHTS
        self.score = (
            w["fts"] * self.score_keyword
            + w["embedding"] * self.score_semantic
            + w["context_tag"] * self.score_context
            + w["signal"] * self.score_signal
        )
        self.score = min(max(self.score, 0.0), 1.0)
        return self.score

    def breakdown(self) -> str:
        """Human-readable score breakdown for diagnostics."""
        return (
            f"score={self.score:.2f}  "
            f"sem:{self.score_semantic:.2f} "
            f"key:{self.score_keyword:.2f} "
            f"ctx:{self.score_context:.2f} "
            f"ent:{self.score_entity:.2f} "
            f"tmp:{self.score_temporal:.2f} "
            f"sig:{self.score_signal:.2f}  "
            f"[{self.provider}]"
        )


# ---------------------------------------------------------------------------
# Tag parsing helpers
# ---------------------------------------------------------------------------


def parse_context_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract [context: <type>] from the start of a string.

    Returns (context_type or None, remaining_text_with_tag_stripped).
    """
    m = _CONTEXT_TAG_RE.match(text)
    if m:
        ctx = m.group(1).lower()
        remaining = text[m.end():].strip()
        return ctx, remaining
    return None, text


def parse_signal_tag(text: str) -> Tuple[float, str]:
    """Extract [signal: X.X] from anywhere in a string.

    Returns (signal_value, remaining_text_with_tag_stripped).
    Defaults to DEFAULT_SIGNAL if no tag found.
    """
    m = _SIGNAL_TAG_RE.search(text)
    if m:
        cleaned = (text[:m.start()] + text[m.end():]).strip()
        return float(m.group(1)), cleaned
    return DEFAULT_SIGNAL, text


def strip_memory_markup(text: str) -> str:
    """Remove [context:] and [signal:] tags from entry text.

    Returns clean content suitable for injection into the system prompt.
    """
    _, text = parse_context_tag(text)
    _, text = parse_signal_tag(text)
    return text.strip()


def extract_signal_strength(entry: str) -> float:
    """Extract just the signal strength, ignoring the rest."""
    return parse_signal_tag(entry)[0]


def extract_context_tag(entry: str) -> Optional[str]:
    """Extract just the context tag, ignoring the rest."""
    return parse_context_tag(entry)[0]


def context_tag_match(context_tag: Optional[str], task_type: Optional[str]) -> float:
    """Score the match between a memory entry's context tag and current task type.

    Returns 1.0 for exact match, 0.3 for general entries, 0.0 otherwise.
    """
    if context_tag is None or task_type is None:
        # Untagged entries are somewhat useful (catch-all)
        return 0.3
    if context_tag == task_type:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Entity / keyword extraction helpers (lightweight)
# ---------------------------------------------------------------------------

# Env vars, paths, tool names, project names — keywords that help classify intent
_PROJECT_KEYWORDS: Dict[str, float] = {
    # Infrastructure keywords
    "deploy": 0.8, "gke": 0.9, "k8s": 0.9, "kubernetes": 0.9,
    "docker": 0.7, "container": 0.6, "terraform": 0.8, "gcloud": 0.8,
    "firebase": 0.6, "fly.io": 0.6, "vercel": 0.6, "artifact": 0.5,
    "registry": 0.5, "cluster": 0.7, "namespace": 0.6,

    # Feature / dev keywords
    "feature": 0.7, "branch": 0.6, "pr": 0.7, "pull request": 0.7,
    "implement": 0.6, "write": 0.5, "add": 0.4, "build": 0.5,
    "refactor": 0.6, "migrate": 0.6, "api": 0.5, "endpoint": 0.5,
    "tdd": 0.7, "test": 0.5,

    # Debug keywords
    "bug": 0.9, "fix": 0.7, "error": 0.8, "crash": 0.9, "fail": 0.7,
    "debug": 0.9, "traceback": 0.9, "exception": 0.8, "broken": 0.7,
    "root cause": 0.9, "why": 0.4, "not working": 0.7,

    # Review keywords
    "review": 0.8, "audit": 0.7, "approve": 0.6, "lgtm": 0.6,
    "code review": 0.9, "security": 0.6, "check": 0.4,

    # Research keywords
    "research": 0.8, "investigate": 0.7, "compare": 0.6, "options": 0.5,
    "alternatives": 0.6, "vs": 0.5, "survey": 0.6, "explore": 0.5,

    # Planning keywords
    "plan": 0.8, "strategy": 0.7, "roadmap": 0.8, "architecture": 0.7,
    "design": 0.6, "timeline": 0.6, "phase": 0.5, "milestone": 0.6,
    "decision": 0.6,

    # Content keywords
    "write": 0.5, "draft": 0.6, "post": 0.5, "document": 0.5,
    "blog": 0.7, "changelog": 0.7, "social": 0.5, "thread": 0.5,

    # Maintenance keywords
    "cleanup": 0.7, "clean up": 0.7, "update": 0.5, "upgrade": 0.5,
    "deprecate": 0.6, "remove": 0.4, "cache": 0.5,

    # Kanban keywords
    "kanban": 0.9, "task": 0.6, "ticket": 0.6, "board": 0.6,
    "orchestrat": 0.7, "assignee": 0.7, "sprint": 0.7,
}

# Mapping from top-weighted keyword to task type
_KEYWORD_TO_TASK = {
    # deploy
    "deploy": "deploy", "gke": "deploy", "k8s": "deploy", "kubernetes": "deploy",
    "docker": "deploy", "terraform": "deploy", "gcloud": "deploy",
    "cluster": "deploy", "namespace": "deploy",
    # feature
    "feature": "feature", "branch": "feature", "pull request": "feature",
    "implement": "feature", "refactor": "feature", "migrate": "feature",
    "tdd": "feature", "test": "feature",
    # debug
    "bug": "debug", "crash": "debug", "traceback": "debug",
    "exception": "debug", "not working": "debug",
    # review
    "review": "review", "audit": "review", "code review": "review",
    # research
    "research": "research", "investigate": "research",
    "alternatives": "research", "options": "research", "explore": "research",
    # planning
    "plan": "planning", "roadmap": "planning", "architecture": "planning",
    "strategy": "planning", "timeline": "planning", "milestone": "planning",
    # content
    "blog": "content", "changelog": "content", "draft": "content",
    "document": "content", "post": "content",
    # maintenance
    "cleanup": "maintenance", "deprecate": "maintenance", "clean up": "maintenance",
    # kanban
    "kanban": "kanban", "ticket": "kanban", "sprint": "kanban",
    "orchestrat": "kanban",
}


def classify_task_type(message: str) -> Tuple[Optional[str], float]:
    """Classify the task type from a user message using keyword heuristics.

    Returns (task_type or None, confidence 0.0-1.0).
    Runs in <1ms — no ML model, no LLM call.
    """
    if not message:
        return None, 0.0

    msg_lower = message.lower()

    # Score each task type by keyword hits
    type_scores: Dict[str, float] = {}
    for keyword, weight in _PROJECT_KEYWORDS.items():
        if keyword in msg_lower:
            task = _KEYWORD_TO_TASK.get(keyword)
            if task:
                type_scores[task] = type_scores.get(task, 0.0) + weight

    if not type_scores:
        return None, 0.0

    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]

    # Normalise confidence: cap at 1.0, floor meaningful at 0.15
    confidence = min(best_score / 3.0, 1.0)
    confidence = max(confidence, 0.15)

    return best_type, confidence


def extract_entities(message: str) -> List[Tuple[str, float]]:
    """Extract named entities from a user message.

    Uses simple pattern heuristics for project-specific terms.
    Returns list of (entity_name, salience).

    Future: swap to spacy NER when spacy model is available.
    """
    entities: List[Tuple[str, float]] = []
    msg_lower = message.lower()

    # Known project entities (could be extracted to config)
    KNOWN_ENTITIES = {
        "gke", "k8s", "kubernetes", "docker", "terraform", "gcloud",
        "firebase", "vercel", "fly.io", "supabase", "redis", "postgres",
        "neo4j", "sqlite", "nginx", "node", "python", "go", "typescript",
        "hermes", "xentropy", "pi", "langchain", "langgraph",
        "telegram", "discord", "slack", "whatsapp",
        "expo", "next.js", "nestjs", "graphql", "rest",
    }

    for entity in KNOWN_ENTITIES:
        if entity in msg_lower:
            # Salience = how specific the entity is to the task
            salience = 0.6 if len(entity) > 3 else 0.4
            entities.append((entity, salience))

    return entities
