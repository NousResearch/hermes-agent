"""Intent Router (Phase 2 minimal stub implementation).

Classifies user intent via deterministic keyword-based heuristics.
NO LLM invocation in Phase 2. NO workers. NO delegate_task.
Safe-by-default: returns chat_only when intent is ambiguous or low-confidence.

This is a minimal stub for Phase 2. Phase 3 may replace this with a real
LLM-based classifier (via auxiliary_client).

Cardinal rules:
- NO LLM invocation
- NO auxiliary_client calls
- NO workers
- NO delegate_task
- NO Kanban DB mutation
- NO GBrain CLI invocation
- NO R7 file modification
- NO hermes artifact modification
- NO config.yaml modification
- NO gateway restart
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


INTENT_TYPES = (
    "chat",
    "delegate",
    "research",
    "code",
    "kanban",
    "lookup",
    "approval",
    "composite",
    "unknown",
)

ROUTING_STRATEGIES = ("chat_only", "orchestrate", "approval_required")

CONFIDENCE_THRESHOLD = 0.5


@dataclass
class SubIntent:
    sub_intent_type: str
    description: str
    confidence: float
    required_tools: list[str] = field(default_factory=list)


@dataclass
class SafetyFlag:
    flag_type: str  # r7_protected|self_improvement|write_to_hermes|requires_approval
    severity: str  # low|medium|high|critical
    description: str
    blocked: bool = False


@dataclass
class IntentClassification:
    intent_type: str
    confidence: float
    required_tools: list[str] = field(default_factory=list)
    required_profiles: list[str] = field(default_factory=list)
    sub_intents: list[SubIntent] = field(default_factory=list)
    safety_flags: list[SafetyFlag] = field(default_factory=list)
    classifier_model: str = "phase2_deterministic_stub"
    classified_at_utc: str = ""
    routing_strategy: str = "chat_only"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "confidence": self.confidence,
            "required_tools": list(self.required_tools),
            "required_profiles": list(self.required_profiles),
            "sub_intents": [
                {"sub_intent_type": s.sub_intent_type, "description": s.description,
                 "confidence": s.confidence, "required_tools": list(s.required_tools)}
                for s in self.sub_intents
            ],
            "safety_flags": [
                {"flag_type": f.flag_type, "severity": f.severity,
                 "description": f.description, "blocked": f.blocked}
                for f in self.safety_flags
            ],
            "classifier_model": self.classifier_model,
            "classified_at_utc": self.classified_at_utc,
            "routing_strategy": self.routing_strategy,
            "reason": self.reason,
        }


# Deterministic keyword rules for Phase 2 stub classification
# Each rule: (pattern, intent_type, confidence, required_profiles)
_KEYWORD_RULES: list[tuple[re.Pattern, str, float, list[str]]] = [
    (re.compile(r"\b(research|find out|look up about)\b", re.I), "research", 0.75, ["researcher"]),
    (re.compile(r"\b(code|implement|write (a )?(function|class|script))\b", re.I), "code", 0.75, ["coder"]),
    (re.compile(r"\b(kanban|board task|task)\b", re.I), "kanban", 0.70, ["orchestrator"]),
    (re.compile(r"\b(delegate|dispatch|run in parallel)\b", re.I), "delegate", 0.70, ["orchestrator"]),
    (re.compile(r"\b(what is|who is|when did|where is|define|explain)\b", re.I), "lookup", 0.65, []),
    (re.compile(r"\b(approve|approval)\b", re.I), "approval", 0.60, []),
    (re.compile(r"\b(chat|hello|hi|hey|thanks)\b", re.I), "chat", 0.95, []),
]

# Self-improvement / R7 protection patterns (Phase 2 critical safety)
_PROTECTED_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"self_improvement_guard\.py", re.I), "r7_protected", "critical"),
    (re.compile(r"\bself[-_ ]improvement\b", re.I), "self_improvement", "critical"),
    (re.compile(r"\bwrite to hermes\b", re.I), "write_to_hermes", "high"),
    (re.compile(r"\bmodify (config|allowlist|schema|sidecar)\b", re.I), "write_to_hermes", "high"),
]


def _now_iso_utc() -> str:
    """Return current UTC as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_safety_flags(message: str) -> list[SafetyFlag]:
    """Inspect message and return safety flags.

    Phase 2: deterministic regex-based.
    Returns empty list when safe.
    Returns critical flag when R7 protection would be violated.
    """
    flags: list[SafetyFlag] = []
    for pattern, flag_type, severity in _PROTECTED_PATTERNS:
        if pattern.search(message or ""):
            flags.append(SafetyFlag(
                flag_type=flag_type,
                severity=severity,
                description=f"matches pattern: {pattern.pattern}",
                blocked=(severity == "critical"),
            ))
    return flags


def determine_routing_strategy(
    intent_type: str,
    confidence: float,
    safety_flags: list[SafetyFlag],
) -> str:
    """Decide routing strategy (safe-by-default).

    Returns "approval_required" if any critical safety flag blocks.
    Returns "chat_only" if confidence < 0.5 or intent_type is non-orchestration.
    Returns "orchestrate" for delegate/research/code/kanban/composite with confidence >= 0.5.
    """
    if any(flag.severity == "critical" and flag.blocked for flag in safety_flags):
        return "approval_required"

    if confidence < CONFIDENCE_THRESHOLD:
        return "chat_only"

    orchestration_intents = {"delegate", "research", "code", "kanban", "composite"}
    if intent_type in orchestration_intents:
        return "orchestrate"

    return "chat_only"


class IntentRouter:
    """Deterministic intent classifier for Phase 2 (no LLM).

    Safe-by-default:
    - Default to chat_only when ambiguous
    - Validate against R7 safety rules (deterministic regex)
    - Log all classifications via the optional persistence (no real LLM call)
    """

    def __init__(self, self_improvement_guard=None, persistence=None):
        """Initialize router.

        Args:
            self_improvement_guard: optional R7 gate (for safety check)
            persistence: optional ConversationPersistence for logging
        """
        self._guard = self_improvement_guard
        self._persistence = persistence

    def route(
        self,
        message: str,
        conversation_context: dict | None = None,
        available_tools: list[dict] | None = None,
        available_profiles: list[str] | None = None,
        gbrain_context: dict | None = None,
    ) -> IntentClassification:
        """Classify user intent via deterministic regex rules.

        Phase 2 stub: NO LLM invocation. NO auxiliary_client.
        Returns IntentClassification with intent_type, confidence,
        required_tools, required_profiles, safety_flags,
        classifier_model, classified_at_utc, routing_strategy.
        """
        text = (message or "").strip()

        # 1. Compute safety flags FIRST (deterministic regex)
        safety_flags = classify_safety_flags(text)

        # 2. If any critical safety flag, return approval_required
        if any(flag.severity == "critical" and flag.blocked for flag in safety_flags):
            return IntentClassification(
                intent_type="approval",
                confidence=1.0,
                required_tools=[],
                required_profiles=[],
                sub_intents=[],
                safety_flags=safety_flags,
                classified_at_utc=_now_iso_utc(),
                routing_strategy="approval_required",
                reason="critical_safety_flag",
            )

        # 3. Apply keyword rules (deterministic)
        intent_type = "unknown"
        confidence = 0.0
        required_profiles: list[str] = []
        reason = "no_keyword_match"

        for pattern, itype, conf, profiles in _KEYWORD_RULES:
            if pattern.search(text):
                intent_type = itype
                confidence = conf
                required_profiles = list(profiles)
                reason = f"keyword_match:{pattern.pattern}"
                break

        # 4. Determine routing strategy (safe-by-default)
        routing_strategy = determine_routing_strategy(intent_type, confidence, safety_flags)

        classification = IntentClassification(
            intent_type=intent_type,
            confidence=confidence,
            required_tools=[],
            required_profiles=required_profiles,
            sub_intents=[],
            safety_flags=safety_flags,
            classified_at_utc=_now_iso_utc(),
            routing_strategy=routing_strategy,
            reason=reason,
        )

        # 5. Optional: log to persistence (no-op if persistence is None)
        if self._persistence is not None and conversation_context:
            cid = conversation_context.get("conversation_id")
            if cid:
                try:
                    from agent.conversation_persistence import make_event
                    self._persistence.save_event(
                        cid,
                        make_event(
                            "intent_classified",
                            cid,
                            {"intent_classification": classification.to_dict()},
                        ),
                    )
                except Exception:
                    pass  # Persistence failure does not block classification

        return classification

    def validate_intent(self, intent: IntentClassification) -> bool:
        """Validate intent against R7 safety rules.

        Returns False if any critical safety flag is set.
        """
        return not any(flag.severity == "critical" and flag.blocked for flag in intent.safety_flags)

    def requires_orchestration(self, intent: IntentClassification) -> bool:
        """Decide if intent requires orchestration.

        Returns True only when routing_strategy == "orchestrate".
        """
        return intent.routing_strategy == "orchestrate"

    def get_safe_fallback_intent(
        self,
        original_message: str = "",
        reason: str = "ambiguous",
    ) -> IntentClassification:
        """Return safe fallback intent (chat_only).

        Used when:
        - classification fails
        - R7 gate blocks
        - Intent cannot be classified
        - Confidence below threshold
        """
        return IntentClassification(
            intent_type="chat",
            confidence=1.0,
            required_tools=[],
            required_profiles=[],
            sub_intents=[],
            safety_flags=[],
            classified_at_utc=_now_iso_utc(),
            routing_strategy="chat_only",
            reason=reason,
        )
