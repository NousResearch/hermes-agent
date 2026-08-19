"""Shared topic-shift detection for skill-graph injection and session topics.

Two independent features need to answer the same question — "is this user
message continuing the same topic as the previous one, or is it a new topic?":

1.  **skill-graph** (plugin, ``pre_llm_call``): decides whether to inject a
    full candidate list (new topic) or only the delta (same topic).
2.  **session-topics** (core, ``system_prompt`` / ``run_agent``): provides a
    pre-LLM hint so the model emits a more reliable ``TOPIC:`` signal.

This module is a **pure-function library** — no hermes-agent imports, no
global state, no I/O.  Callers manage their own per-session state and pass
it in.  This keeps the module importable from both core and plugins (via
``importlib``), and trivially unit-testable.

Detection strategy: **AND gate** (calibrated on 18 real session transitions,
94.4% accuracy vs 83% for LLM-only or S6-only):

    topic_continuation = (
        llm_topic_continuation is True
        AND s6_cosine >= S6_THRESHOLD
    )

Both signals must agree on "same topic" to classify as continuation.  If
either says "shift", we treat it as a shift.  This is deliberately biased
toward false-shift (over-injection) over false-same (missing a new topic's
skills).

The primary signal is an LLM judgment piggy-backed onto the existing
intent-split call (zero extra API cost).  The secondary signal is the
cosine similarity between the first 100 characters of consecutive messages
(S6 — first-100-char intent embedding).  S6 alone achieves 88.9% accuracy;
combined with the LLM signal via AND gate, accuracy rises to 94.4%.

When either signal is unavailable (LLM failed, no embedding backend), the
available signal is used alone.  When both are unavailable, the function
defaults to ``True`` (continuation) so the caller falls back to
conservative full injection rather than blocking all injection.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger("topic_detection")

# Calibrated on 18 real session transitions (20260806_005634_165184).
# S6 cosine at this threshold catches 2/3 LLM false-sames (8→9, 14→15)
# while passing all 12 genuine same-topic transitions.
S6_THRESHOLD: float = 0.47

# Number of leading characters to use for the S6 embedding comparison.
# Long messages dilute the core intent with context, quotes, and numbered
# lists; the first ~100 chars capture what the user actually wants.
INTENT_PREFIX_LEN: int = 100


def compute_s6_cosine(
    current_msg: str,
    prev_msg: str,
    embed_fn: Optional[Any] = None,
) -> Optional[float]:
    """Compute S6 cosine similarity between two messages.

    Takes the first ``INTENT_PREFIX_LEN`` characters of each message,
    embeds them via ``embed_fn``, and returns cosine similarity.

    Args:
        current_msg: Current user message text.
        prev_msg: Previous user message text.
        embed_fn: Callable that takes a string and returns a list[float]
            embedding vector.  If ``None``, returns ``None``.

    Returns:
        Cosine similarity in [-1, 1], or ``None`` if embedding fails.
    """
    if not embed_fn:
        return None
    if not current_msg or not prev_msg:
        return None

    try:
        intent_a = current_msg[:INTENT_PREFIX_LEN].strip()
        intent_b = prev_msg[:INTENT_PREFIX_LEN].strip()
        if not intent_a or not intent_b:
            return None

        emb_a = embed_fn(intent_a)
        emb_b = embed_fn(intent_b)
        if emb_a is None or emb_b is None:
            return None

        return _cosine(emb_a, emb_b)
    except Exception as exc:
        logger.debug("S6 cosine computation failed: %s", exc)
        return None


def detect_topic_shift(
    current_msg: str,
    prev_msg: Optional[str] = None,
    llm_topic_continuation: Optional[bool] = None,
    embed_fn: Optional[Any] = None,
    *,
    s6_threshold: float = S6_THRESHOLD,
) -> dict[str, Any]:
    """Determine whether ``current_msg`` continues the previous topic.

    Uses the AND-gate strategy: both the LLM signal and S6 cosine must
    agree on "same topic" for the result to be continuation.

    Args:
        current_msg: Current user message text.
        prev_msg: Previous user message text, or ``None`` if this is the
            first message in the session.
        llm_topic_continuation: LLM judgment from the intent-split call
            (``True`` = same topic, ``False`` = shift, ``None`` = LLM
            unavailable).
        embed_fn: Embedding callable for S6 cosine.  ``None`` to skip S6.
        s6_threshold: Override the default S6 cosine threshold.

    Returns:
        Dict with keys:

        - ``topic_continuation`` (bool): ``True`` if same topic, ``False``
          if shift.  Defaults to ``True`` when no signals are available
          (caller should fall back to full injection).
        - ``confidence`` (float): 0.0–1.0, rough confidence based on how
          many signals agreed.
        - ``method`` (str): Which signal(s) were used — ``"and_gate"``,
          ``"llm_only"``, ``"s6_only"``, or ``"fallback"``.
        - ``s6_cosine`` (float|None): Raw S6 cosine value if computed.
        - ``llm_signal`` (bool|None): Raw LLM signal if provided.
    """
    # First turn or no previous message → always "new topic" (full injection)
    if not prev_msg:
        return {
            "topic_continuation": False,
            "confidence": 1.0,
            "method": "first_turn",
            "s6_cosine": None,
            "llm_signal": llm_topic_continuation,
        }

    # Compute S6 cosine (secondary signal)
    s6_cosine = compute_s6_cosine(current_msg, prev_msg, embed_fn)

    # ── AND gate: both signals must agree on "same" ──

    has_llm = llm_topic_continuation is not None
    has_s6 = s6_cosine is not None
    s6_same = s6_cosine is not None and s6_cosine >= s6_threshold
    llm_same = llm_topic_continuation is True

    if has_llm and has_s6:
        # Both available → AND gate
        is_same = llm_same and s6_same
        method = "and_gate"
        # Confidence: both agree = high; disagree = the "no" wins but
        # with lower confidence
        if llm_same == s6_same:
            confidence = 0.95
        else:
            confidence = 0.70
    elif has_llm:
        # LLM only
        is_same = llm_same
        method = "llm_only"
        confidence = 0.83  # measured LLM-only accuracy
    elif has_s6:
        # S6 only
        is_same = s6_same
        method = "s6_only"
        confidence = 0.89  # measured S6-only accuracy
    else:
        # Neither signal available → conservative: treat as continuation
        # so the caller injects full (safer than silently suppressing)
        # Actually — "continuation" means delta, which could suppress
        # needed candidates. Better to say "shift" → full injection.
        # But if prev_msg exists and we have NO signals at all, full
        # injection is the right call.
        return {
            "topic_continuation": False,
            "confidence": 0.0,
            "method": "fallback",
            "s6_cosine": None,
            "llm_signal": None,
        }

    return {
        "topic_continuation": is_same,
        "confidence": confidence,
        "method": method,
        "s6_cosine": s6_cosine,
        "llm_signal": llm_topic_continuation,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
