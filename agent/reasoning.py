"""Structured Reasoning Engine — parse and track reasoning chains.

Converts raw extended thinking text (Claude, OpenAI, DeepSeek, etc.) into
structured ReasoningStep objects with titles, actions, confidence scores,
and next-action decisions. Stores reasoning chains in the session DB for
post-hoc analysis and self-evolution.

Inspired by agno's ReasoningManager with ReasoningStep dataclass.

Usage:
    chain = ReasoningChain(session_id="abc")
    chain.ingest_text(raw_thinking_text)
    for step in chain.steps:
        print(f"Step {step.step_number}: {step.title} (confidence: {step.confidence})")

    # Store for self-evolution
    chain.persist(session_db)
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ReasoningStep — single structured step
# ============================================================================

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int = 0
    title: str = ""
    action: str = ""       # "I will..."
    result: str = ""       # "I found..."
    reasoning: str = ""    # Full reasoning text for this step
    confidence: float = 0.0  # 0.0-1.0
    next_action: str = "continue"  # continue, validate, final_answer
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# ReasoningChain — collector for a full reasoning sequence
# ============================================================================

@dataclass
class ReasoningChain:
    """Collects and structures a reasoning sequence for one agent turn."""
    session_id: str = ""
    turn_number: int = 0
    model: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    raw_text: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    reasoning_tokens: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def duration_ms(self) -> float:
        if self.completed_at > 0:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    @property
    def avg_confidence(self) -> float:
        if not self.steps:
            return 0.0
        scored = [s for s in self.steps if s.confidence > 0]
        if not scored:
            return 0.0
        return sum(s.confidence for s in scored) / len(scored)

    def ingest_text(self, raw_text: str) -> List[ReasoningStep]:
        """Parse raw reasoning/thinking text into structured steps.

        Handles multiple formats:
        - Numbered steps ("1. ", "Step 1:", etc.)
        - Header-delimited sections ("## Analysis", "### Step 1")
        - Paragraph-delimited (fallback: each paragraph = one step)
        - XML-tagged (<step>, <thinking>, etc.)

        Returns the list of new steps added.
        """
        if not raw_text or not raw_text.strip():
            return []

        self.raw_text = raw_text
        new_steps = []

        # Try structured parsing in order of specificity
        segments = _try_numbered_steps(raw_text)
        if not segments:
            segments = _try_header_sections(raw_text)
        if not segments:
            segments = _try_xml_sections(raw_text)
        if not segments:
            segments = _try_paragraph_split(raw_text)

        for i, (title, content) in enumerate(segments, start=1):
            confidence = _estimate_confidence(content)
            is_last = (i == len(segments))
            next_action = "final_answer" if is_last else "continue"

            step = ReasoningStep(
                step_number=len(self.steps) + 1,
                title=title,
                action=_extract_action(content),
                result=_extract_result(content),
                reasoning=content.strip(),
                confidence=confidence,
                next_action=next_action,
            )
            self.steps.append(step)
            new_steps.append(step)

        return new_steps

    def complete(self, reasoning_tokens: int = 0) -> None:
        """Mark the chain as completed."""
        self.completed_at = time.time()
        self.reasoning_tokens = reasoning_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "avg_confidence": round(self.avg_confidence, 3),
            "reasoning_tokens": self.reasoning_tokens,
            "duration_ms": round(self.duration_ms, 1),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def persist(self, session_db: Any) -> None:
        """Store the reasoning chain in the session DB as a message annotation.

        Writes to the session_state so the self-evolution system can analyze
        reasoning patterns across sessions.
        """
        if not session_db or not self.session_id:
            return
        try:
            state = session_db.get_session_state(self.session_id)
            chains = state.get("_reasoning_chains", [])
            chains.append(self.to_dict())
            # Keep last 10 chains per session to avoid unbounded growth
            if len(chains) > 10:
                chains = chains[-10:]
            session_db.update_session_state(self.session_id, {"_reasoning_chains": chains})
        except Exception as e:
            logger.debug("Failed to persist reasoning chain: %s", e)

    def format_summary(self) -> str:
        """Format a human-readable summary of the reasoning chain."""
        if not self.steps:
            return ""
        lines = [f"Reasoning ({self.total_steps} steps, avg confidence: {self.avg_confidence:.0%}):"]
        for step in self.steps:
            conf = f" [{step.confidence:.0%}]" if step.confidence > 0 else ""
            lines.append(f"  {step.step_number}. {step.title}{conf}")
        return "\n".join(lines)


# ============================================================================
# Parsing helpers
# ============================================================================

_NUMBERED_RE = re.compile(
    r'^(?:(?:Step\s+)?(\d+)[.):\s]+)',
    re.MULTILINE | re.IGNORECASE,
)

_HEADER_RE = re.compile(
    r'^#{1,4}\s+(.+?)$',
    re.MULTILINE,
)

_XML_STEP_RE = re.compile(
    r'<(?:step|thinking|analysis|approach)[^>]*>(.*?)</(?:step|thinking|analysis|approach)>',
    re.DOTALL | re.IGNORECASE,
)


def _try_numbered_steps(text: str) -> List[tuple]:
    """Parse numbered steps like '1. First...' or 'Step 1: ...'."""
    matches = list(_NUMBERED_RE.finditer(text))
    if len(matches) < 2:
        return []

    segments = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        # Extract title from first line
        first_line = content.split('\n')[0].strip()
        title = first_line[:80] if first_line else f"Step {match.group(1)}"
        segments.append((title, content))

    return segments


def _try_header_sections(text: str) -> List[tuple]:
    """Parse markdown-style headers as section delimiters."""
    matches = list(_HEADER_RE.finditer(text))
    if len(matches) < 2:
        return []

    segments = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        segments.append((title, content))

    return segments


def _try_xml_sections(text: str) -> List[tuple]:
    """Parse XML-tagged reasoning sections."""
    matches = list(_XML_STEP_RE.finditer(text))
    if not matches:
        return []

    segments = []
    for i, match in enumerate(matches):
        content = match.group(1).strip()
        first_line = content.split('\n')[0].strip()
        title = first_line[:80] if first_line else f"Step {i + 1}"
        segments.append((title, content))

    return segments


def _try_paragraph_split(text: str) -> List[tuple]:
    """Fallback: split on double newlines, each paragraph is a step."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Don't split if there's only one paragraph or too many (> 15)
    if len(paragraphs) <= 1:
        title = text.strip().split('\n')[0][:80]
        return [(title, text.strip())]
    if len(paragraphs) > 15:
        paragraphs = paragraphs[:15]

    segments = []
    for i, para in enumerate(paragraphs, start=1):
        first_line = para.split('\n')[0].strip()
        title = first_line[:80] if first_line else f"Thought {i}"
        segments.append((title, para))

    return segments


# ============================================================================
# Confidence estimation
# ============================================================================

_HIGH_CONFIDENCE_MARKERS = [
    r'\bclearly\b', r'\bdefinitely\b', r'\bcertain\b', r'\bconfident\b',
    r'\bthe answer is\b', r'\bthis means\b', r'\btherefore\b',
    r'\bconfirm\b', r'\bverified\b', r'\bcorrect\b',
]

_LOW_CONFIDENCE_MARKERS = [
    r'\bmaybe\b', r'\bperhaps\b', r'\bnot sure\b', r'\buncertain\b',
    r'\bmight\b', r'\bcould be\b', r'\bpossibly\b', r'\bunclear\b',
    r'\bassume\b', r'\bguess\b', r'\bwait\b', r'\bhmm\b',
    r'\blet me reconsider\b', r'\bon the other hand\b',
]


def _estimate_confidence(text: str) -> float:
    """Heuristic confidence score based on language markers. Returns 0.0-1.0."""
    if not text:
        return 0.5

    text_lower = text.lower()
    high = sum(1 for p in _HIGH_CONFIDENCE_MARKERS if re.search(p, text_lower))
    low = sum(1 for p in _LOW_CONFIDENCE_MARKERS if re.search(p, text_lower))

    total = high + low
    if total == 0:
        return 0.5  # neutral
    ratio = high / total
    # Map 0..1 ratio to 0.3..0.9 range (never fully 0 or 1 from heuristics)
    return 0.3 + (ratio * 0.6)


def _extract_action(text: str) -> str:
    """Extract 'I will...' action statement from reasoning text."""
    patterns = [
        r"(?:I(?:'ll| will| need to| should))\s+(.{10,80}?)(?:\.|$)",
        r"(?:Let me|Let's)\s+(.{10,80}?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def _extract_result(text: str) -> str:
    """Extract 'I found...' result statement from reasoning text."""
    patterns = [
        r"(?:I (?:found|see|notice|observe|got|determined))\s+(.{10,80}?)(?:\.|$)",
        r"(?:The (?:result|answer|output|value) (?:is|was|shows))\s+(.{10,80}?)(?:\.|$)",
        r"(?:This (?:means|shows|indicates|confirms))\s+(.{10,80}?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""
