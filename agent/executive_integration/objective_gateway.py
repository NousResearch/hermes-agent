"""ObjectiveGateway — read-only decision layer.

Decides whether to escalate a user message to Executive v2.
Produces an ``ObjectiveGatewayDecision`` (immutable).

Deterministic. No LLM. No provider. No network. No subprocess.
No DB write. No new state.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ExecutiveIntegrationMetrics,
    ObjectiveGatewayDecision,
    RouteKind,
    compute_decision_fingerprint,
    _now_iso8601,
)


# ──────────────────────────────────────────────────────────────────────
# Keyword lists (deterministic, immutable)
# ──────────────────────────────────────────────────────────────────────


# EXECUTIVE keywords (case-insensitive substring match).
EXECUTIVE_KEYWORDS: Tuple[str, ...] = (
    "consigue",
    "logra",
    "haz que",
    "construye",
    "organiza",
    "analiza y ejecuta",
    "prepara y valida",
    "resuelve",
    "automatiza",
    "planifica",
    "orquesta",
    "despliega",
    "implementa",
)

# REJECT keywords (case-insensitive substring match).
REJECT_KEYWORDS: Tuple[str, ...] = (
    "leak",
    "exfiltrate",
    "credentials",
    "secrets",
    "private key",
    "ssh key",
    "token",
    "password",
    "exploit",
    "malware",
)

# Tool-like patterns: very simple read-only / single-shot patterns.
TOOL_PATTERNS: Tuple[str, ...] = (
    r"^leer\s+(?:el\s+)?archivo\b",
    r"^mostrar\s+(?:el\s+)?estado\b",
    r"^run\s+the\s+test\s+suite\b",
    r"^find\s+the\s+line\b",
)

# Default creator.
DEFAULT_CREATED_BY = "ExecutiveIntegrationRouter"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _looks_like_exec_keyword(message_lc: str) -> Tuple[str, ...]:
    """Return matched executive keywords (case-insensitive substring).

    Uses whole-word boundary for short keywords (<=5 chars) and
    substring match for long keywords (e.g. 'analiza y ejecuta').
    """
    matched: List[str] = []
    for kw in EXECUTIVE_KEYWORDS:
        if len(kw) <= 5:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, message_lc):
                matched.append(kw)
        else:
            if kw in message_lc:
                matched.append(kw)
    return tuple(matched)


def _looks_like_reject(message_lc: str) -> bool:
    """Return True if any REJECT keyword is present."""
    return any(kw in message_lc for kw in REJECT_KEYWORDS)


def _looks_like_tool(message_lc: str) -> bool:
    """Return True if any TOOL regex matches."""
    return any(re.search(p, message_lc) for p in TOOL_PATTERNS)


def _word_count(message: str) -> int:
    return len([w for w in re.split(r"\s+", message.strip()) if w])


def _flags_enabled() -> Dict[str, bool]:
    """Read EIL flags from env. Default-off."""
    return {
        "integration_enabled": os.environ.get("HERMES_EXECUTIVE_INTEGRATION_ENABLED", "0") == "1",
        "gateway_enabled": os.environ.get("HERMES_OBJECTIVE_GATEWAY_ENABLED", "0") == "1",
        "autolaunch_enabled": os.environ.get("HERMES_EXECUTIVE_AUTOLAUNCH_ENABLED", "0") == "1",
    }


# ──────────────────────────────────────────────────────────────────────
# ObjectiveGateway
# ──────────────────────────────────────────────────────────────────────


class ObjectiveGateway:
    """Read-only decision layer. Does NOT execute anything.

    Cardinal rules:
      * No LLM. No provider. No network. No subprocess.
      * No DB write. No new state. No commit.
      * Deterministic: same inputs → same outputs.
    """

    SCHEMA_VERSION = "eil.v1"

    def __init__(
        self,
        *,
        intent_router: Any = None,
        policy_engine: Any = None,
    ) -> None:
        """Inject the existing intent_router and policy_engine (read-only)."""
        self._intent_router = intent_router
        self._policy_engine = policy_engine

    # ── public ────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """Return True iff the EIL is enabled (HERMES_EXECUTIVE_INTEGRATION_ENABLED=1)."""
        return _flags_enabled()["integration_enabled"]

    def route(
        self,
        user_message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveGatewayDecision:
        """Decide which route a user message should take.

        Returns an immutable ObjectiveGatewayDecision.
        """
        t0 = time.monotonic()
        flags = _flags_enabled()
        matched_intent = None
        intent_strategy: Optional[str] = None

        # Disabled → CHAT (passthrough).
        if not flags["integration_enabled"]:
            return self._make_decision(
                user_message=user_message,
                route_kind=RouteKind.CHAT,
                confidence=1.0,
                rationale="EIL disabled (HERMES_EXECUTIVE_INTEGRATION_ENABLED=0); passthrough to CHAT.",
                matched_keywords=(),
                matched_intent=None,
                intent_routing_strategy=None,
                fallback_used=True,
            )

        # 1. Get intent from intent_router (read-only).
        intent_obj = None
        if self._intent_router is not None:
            try:
                intent_obj = self._intent_router.classify_intent(user_message)
            except Exception:
                intent_obj = None
            if intent_obj is not None:
                matched_intent = getattr(intent_obj, "intent_type", None) or getattr(intent_obj, "type", None)
                intent_strategy = getattr(intent_obj, "routing_strategy", None)
        intent_lc = (intent_strategy or "").lower()
        message = (user_message or "").strip()
        message_lc = message.lower()

        # 2. REJECT first (highest priority).
        policy_reject = False
        policy_decision_obj = None
        if self._policy_engine is not None:
            try:
                policy_decision_obj = self._policy_engine.evaluate(message)
            except Exception:
                policy_decision_obj = None
            if policy_decision_obj is not None:
                decision = getattr(policy_decision_obj, "decision", None) or getattr(policy_decision_obj, "result", None)
                if str(decision).lower() == "reject":
                    policy_reject = True
        if policy_reject or _looks_like_reject(message_lc):
            return self._make_decision(
                user_message=user_message,
                route_kind=RouteKind.REJECT,
                confidence=0.99,
                rationale="Message contains rejected content (policy REJECT or known REJECT keyword).",
                matched_keywords=(),
                matched_intent=matched_intent,
                intent_routing_strategy=intent_strategy,
                fallback_used=False,
            )

        # 3. CLARIFY (low confidence or missing input).
        word_count = _word_count(message)
        exec_keywords_matched = _looks_like_exec_keyword(message_lc)
        # Only ask for clarification if the intent is unknown and there are no
        # executive keywords. Chat-only intents with short messages are still
        # CHAT (they're just brief questions).
        if (
            not exec_keywords_matched
            and (
                (intent_obj is None and word_count < 4)
                or matched_intent in (None, "unknown")
            )
        ):
            return self._make_decision(
                user_message=user_message,
                route_kind=RouteKind.CLARIFY,
                confidence=0.4,
                rationale="Message too short or ambiguous; please provide more detail.",
                matched_keywords=(),
                matched_intent=matched_intent,
                intent_routing_strategy=intent_strategy,
                fallback_used=True,
            )

        # 4. EXECUTIVE (priority over TOOL).
        gateway_enabled = flags["gateway_enabled"]  # noqa: F841 (used inside `if` below)
        if (exec_keywords_matched and gateway_enabled) or (
            intent_strategy in ("orchestrate", "approval_required")
        ):
            return self._make_decision(
                user_message=user_message,
                route_kind=RouteKind.EXECUTIVE,
                confidence=0.85,
                rationale=(
                    f"Message matched EXECUTIVE keywords: {list(exec_keywords_matched)}"
                    if exec_keywords_matched
                    else f"intent routing strategy '{intent_strategy}' triggers EXECUTIVE"
                ),
                matched_keywords=exec_keywords_matched,
                matched_intent=matched_intent,
                intent_routing_strategy=intent_strategy,
                fallback_used=False,
            )

        # 5. TOOL (single instrumental action).
        if _looks_like_tool(message_lc) or (matched_intent in ("lookup", "code") and word_count < 15):
            return self._make_decision(
                user_message=user_message,
                route_kind=RouteKind.TOOL,
                confidence=0.7,
                rationale="Message matches a TOOL pattern (single instrumental action).",
                matched_keywords=(),
                matched_intent=matched_intent,
                intent_routing_strategy=intent_strategy,
                fallback_used=False,
            )

        # 6. CHAT (default).
        return self._make_decision(
            user_message=user_message,
            route_kind=RouteKind.CHAT,
            confidence=0.6,
            rationale="Default CHAT route (no EXECUTIVE, TOOL, CLARIFY, or REJECT trigger).",
            matched_keywords=(),
            matched_intent=matched_intent,
            intent_routing_strategy=intent_strategy,
            fallback_used=True,
        )

    # ── private ───────────────────────────────────────────────

    def _make_decision(
        self,
        *,
        user_message: str,
        route_kind: RouteKind,
        confidence: float,
        rationale: str,
        matched_keywords: Tuple[str, ...],
        matched_intent: Optional[str],
        intent_routing_strategy: Optional[str],
        fallback_used: bool,
    ) -> ObjectiveGatewayDecision:
        fingerprint = compute_decision_fingerprint(
            message=user_message,
            route_kind=route_kind,
            matched_keywords=matched_keywords,
            matched_intent=matched_intent,
            intent_routing_strategy=intent_routing_strategy,
        )
        return ObjectiveGatewayDecision(
            route_kind=route_kind,
            objective_id=None,
            confidence=confidence,
            rationale=rationale,
            matched_keywords=matched_keywords,
            matched_intent=matched_intent,
            intent_routing_strategy=intent_routing_strategy,
            fallback_used=fallback_used,
            fingerprint=fingerprint,
            created_at=_now_iso8601(),
            created_by=DEFAULT_CREATED_BY,
        )


# ──────────────────────────────────────────────────────────────────────
# Re-export for convenience
# ──────────────────────────────────────────────────────────────────────


__all__ = [
    "ObjectiveGateway",
    "EXECUTIVE_KEYWORDS",
    "REJECT_KEYWORDS",
    "TOOL_PATTERNS",
]
