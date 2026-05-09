"""Proactive Communication Loop — Hermes initiates when it has something worth saying.

This module implements the synthesis-and-initiative pass that lets Hermes send
the user an unprompted message when it detects something genuinely worth surfacing.

Background
----------
Requested by @charlesmcdowell (2.2K views, May 8 2026). Teknium replied:
"This is a good idea 🤔"

The name matters: this is the **Proactive Communication Loop** — not a "nightly summary"
or "scheduled report". The communication is:

  - Proactive: initiated by the agent, not the user
  - Communicative: a natural message, not a data dump
  - Looping: runs on a schedule, synthesizes and decides each time

Two synthesis modes
-------------------

**Recency-only** (always available):
  Reviews the last N hours of conversation history. Good for: completed task
  notifications, unresolved threads that now have answers, inline "tell me when X"
  instructions the user gave earlier.

**BartokGraph-augmented** (when BartokGraph plugin is installed):
  Traverses a local knowledge graph built from the user's files and conversation
  history. Detects cross-temporal and cross-domain connections the user cannot
  see themselves — because they can't hold months of context in their head.

  Three new message types only BartokGraph enables:
    TEMPORAL_BRIDGE   — "You worked on this exact problem 3 weeks ago."
    CROSS_DOMAIN      — "Your X work and your Y work share the same structure."
    PERSON_KNOWLEDGE  — "Alice mentioned X. You asked about Y. These converge."

Design invariants
-----------------
- NEVER modifies session state, memory, or system prompt.
- NEVER sends more than max_per_day messages per user.
- ALWAYS prefers silence: threshold defaults to conservative (0.75).
- Falls back to no-send on any error (fail-open, err toward quiet).
- Uses cheap/fast local model for synthesis judge when available.
- Fully opt-in: proactive_communication.enabled defaults to False.
- BartokGraph is optional: if not installed, falls back to recency-only.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = "conservative"
DEFAULT_MAX_PER_DAY = 1
DEFAULT_HISTORY_WINDOW_HOURS = 16
DEFAULT_SYNTHESIS_BUDGET_TOKENS = 2000
_HISTORY_SNIPPET_CHARS = 8000

# Threshold scores. Only messages scoring above these are sent.
# Calibrated conservatively — users prefer silence over noise.
THRESHOLD_SCORES: Dict[str, float] = {
    "conservative": 0.75,  # Default. High bar. Better to stay quiet.
    "balanced": 0.55,      # Moderate. Useful insights + completed tasks.
    "eager": 0.35,         # Low bar. For users who want maximum initiative.
}


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """Outcome of one Proactive Communication Loop synthesis pass.

    ``should_send`` is the gate. Callers check this before dispatching.
    ``connection_type`` indicates which BartokGraph insight triggered the
    message (or "none" for recency-only synthesis).
    """

    should_send: bool
    message: Optional[str]
    reasoning: str             # written to audit log — why send or not
    novelty_score: float       # 0–1: how new vs what was already said
    relevance_score: float     # 0–1: how useful to recent work
    combined_score: float      # weighted combination for threshold check
    connection_type: str = "none"  # none | temporal_bridge | cross_domain | person_knowledge
    candidates: List[str] = field(default_factory=list)
    synthesis_ms: int = 0


@dataclass
class BartokGraphConnection:
    """A connection detected between today's work and past knowledge.

    These are the 'how did it know that?' moments — non-obvious links
    across time and domain that the user cannot see on their own.
    """

    node_a_content: str    # today's concept
    node_b_content: str    # past concept (from BartokGraph)
    connection_type: str   # temporal_bridge | cross_domain | person_knowledge
    strength: float        # 0–1 semantic overlap
    days_apart: int        # how long since node_b was last active
    explanation: str       # human-readable bridge explanation


@dataclass
class BartokGraphContext:
    """Graph-augmented context gathered for one synthesis pass.

    When available, this is added to the synthesis prompt as a
    BARTOKGRAPH CONNECTIONS section — giving the judge model access to
    cross-temporal knowledge it couldn't otherwise see.
    """

    connections: List[BartokGraphConnection]
    provider_name: str  # which BartokGraph backend was used
    traversal_ms: int = 0


# ──────────────────────────────────────────────────────────────────────
# Pluggable threshold protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class ProactiveThreshold(Protocol):
    """Custom threshold implementation.

    Third-party plugins can register a custom threshold::

        from hermes_cli.proactive_communication_loop import register_threshold

        @register_threshold("my_threshold")
        class MyThreshold:
            def should_send(self, result: SynthesisResult) -> bool:
                return result.combined_score > 0.6 and result.novelty_score > 0.5
    """

    def should_send(self, result: SynthesisResult) -> bool: ...


_registered_thresholds: Dict[str, ProactiveThreshold] = {}


def register_threshold(name: str):
    """Decorator to register a custom threshold by name."""

    def _decorator(cls):
        _registered_thresholds[name] = cls()
        return cls

    return _decorator


# ──────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────


class ProactiveCommunicationLoop:
    """Synthesis-and-initiative engine for the Proactive Communication Loop.

    Typical usage (from gateway cron)::

        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)
        result = await loop.run_synthesis(session_id)
        if result.should_send and result.message:
            await deliver_message(result.message)
            await loop.record_sent(session_id, result)

    BartokGraph augmentation is automatic when the plugin is installed.
    Falls back gracefully to recency-only synthesis if unavailable.
    """

    def __init__(
        self,
        session_db: Any,  # SessionDB from run_agent.py
        config: Any,      # HermesConfig
    ) -> None:
        self._db = session_db
        self._cfg = config
        self._bartokgraph: Optional[Any] = self._try_load_bartokgraph()

    def _try_load_bartokgraph(self) -> Optional[Any]:
        """Attempt to load BartokGraph adapter. Never raises."""
        if not self._cfg.get("proactive_communication.bartokgraph.enabled", True):
            return None
        try:
            from hermes_cli.bartokgraph_adapter import BartokGraphAdapter
            return BartokGraphAdapter(config=self._cfg)
        except ImportError:
            logger.debug("PCL: BartokGraph plugin not installed — using recency-only synthesis")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_synthesis(
        self,
        session_id: str,
        history_window_hours: int = DEFAULT_HISTORY_WINDOW_HOURS,
    ) -> SynthesisResult:
        """Run one synthesis pass for ``session_id``.

        Returns SynthesisResult. Never raises — on any error, returns a
        no-send result with reasoning logged.
        """
        try:
            return await self._run_synthesis_inner(session_id, history_window_hours)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PCL: synthesis failed for %s: %s", session_id, exc)
            return SynthesisResult(
                should_send=False,
                message=None,
                reasoning=f"synthesis error: {exc}",
                novelty_score=0.0,
                relevance_score=0.0,
                combined_score=0.0,
            )

    async def record_sent(self, session_id: str, result: SynthesisResult) -> None:
        """Record that a proactive message was sent (for dedup/rate-limiting)."""
        try:
            summary = (result.message or "")[:200]
            self._db.record_proactive_sent(session_id, {
                "summary": summary,
                "connection_type": result.connection_type,
                "score": result.combined_score,
                "ts": int(time.time()),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("PCL: failed to record sent: %s", exc)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_synthesis_inner(
        self,
        session_id: str,
        history_window_hours: int,
    ) -> SynthesisResult:
        t0 = time.monotonic()

        # 1. Load recent history
        history = self._load_recent_history(session_id, history_window_hours)
        if not history:
            return SynthesisResult(
                should_send=False,
                message=None,
                reasoning="no conversation history in synthesis window",
                novelty_score=0.0,
                relevance_score=0.0,
                combined_score=0.0,
            )

        # 2. Check rate limit
        if self._over_daily_limit(session_id):
            return SynthesisResult(
                should_send=False,
                message=None,
                reasoning="daily message limit reached",
                novelty_score=0.0,
                relevance_score=0.0,
                combined_score=0.0,
            )

        # 3. BartokGraph traversal (optional — graceful degradation if unavailable)
        graph_ctx: Optional[BartokGraphContext] = None
        if self._bartokgraph:
            try:
                active_topics = await self._extract_topics_from_history(history)
                graph_ctx = await self._bartokgraph.get_connections(
                    active_topics=active_topics,
                    top_k=10,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("PCL: BartokGraph traversal failed: %s — continuing without graph", exc)

        # 4. Build prompt and call judge
        already_sent = self._load_sent_summaries(session_id)
        prompt = _build_synthesis_prompt(history, already_sent, graph_ctx)
        raw = await self._call_synthesis_model(prompt)
        parsed = _parse_synthesis_response(raw)

        # 5. Score and threshold check
        novelty = _clamp_unit_interval(parsed.get("novelty", 0.0))
        relevance = _clamp_unit_interval(parsed.get("relevance", 0.0))
        combined = 0.6 * novelty + 0.4 * relevance
        threshold_name = self._cfg.get(
            "proactive_communication.threshold", DEFAULT_THRESHOLD
        )
        threshold_score = _get_threshold_score(threshold_name, combined, parsed)

        model_wants_send = bool(parsed.get("should_send", True))
        should_send = (
            model_wants_send
            and combined >= threshold_score
            and bool(parsed.get("message"))
        )

        return SynthesisResult(
            should_send=should_send,
            message=parsed.get("message") if should_send else None,
            reasoning=parsed.get("reasoning", ""),
            novelty_score=novelty,
            relevance_score=relevance,
            combined_score=combined,
            connection_type=parsed.get("connection_type", "none"),
            candidates=parsed.get("candidates", []),
            synthesis_ms=int((time.monotonic() - t0) * 1000),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_recent_history(self, session_id: str, window_hours: int) -> str:
        cutoff = time.time() - window_hours * 3600
        try:
            messages = self._db.get_messages_since(session_id, cutoff)
            lines = [
                f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:500]}"
                for m in messages
            ]
            full = "\n".join(lines)
            return full[-_HISTORY_SNIPPET_CHARS:] if len(full) > _HISTORY_SNIPPET_CHARS else full
        except Exception as exc:  # noqa: BLE001
            logger.debug("PCL: history load failed: %s", exc)
            return ""

    def _load_sent_summaries(self, session_id: str) -> str:
        try:
            sent = self._db.get_proactive_sent(session_id, since_hours=24)
            return "; ".join(s.get("summary", "") for s in sent[:5]) or "(none sent today)"
        except Exception:  # noqa: BLE001
            return "(unknown)"

    def _over_daily_limit(self, session_id: str) -> bool:
        try:
            limit = int(self._cfg.get("proactive_communication.max_per_day", DEFAULT_MAX_PER_DAY))
            sent = self._db.get_proactive_sent(session_id, since_hours=24)
            return len(sent) >= limit
        except Exception:  # noqa: BLE001
            return False

    async def _extract_topics_from_history(self, history: str) -> List[str]:
        """Extract top topics from recent history for BartokGraph traversal.

        Uses word-frequency over non-stopword tokens (length > 3). Good enough
        for an initial implementation; replace with NER or phrase extraction when
        graph quality needs improvement.
        """
        words = history.lower().split()
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
                     "or", "is", "was", "are", "were", "i", "you", "me", "my", "your"}
        # Frequency count of non-stop tokens
        freq: Dict[str, int] = {}
        for word in words:
            clean = word.strip(".,!?;:\"'()")
            if len(clean) > 3 and clean not in stopwords:
                freq[clean] = freq.get(clean, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in top[:10]]

    async def _call_synthesis_model(self, prompt: str) -> str:
        """Call a cheap/fast model for the synthesis judge.

        Implementation: wire to the session's configured LLM provider,
        using the same lightweight judge pattern as GoalManager._judge_goal()
        in goals.py. Prefer a local model (Ollama) when available.
        """
        raise NotImplementedError(
            "ProactiveCommunicationLoop._call_synthesis_model must be wired "
            "to the session's LLM provider. See goals.py GoalManager._judge_goal "
            "for the pattern — use the cheapest/fastest model configured."
        )


# ──────────────────────────────────────────────────────────────────────
# Threshold resolution
# ──────────────────────────────────────────────────────────────────────


def _clamp_unit_interval(value: Any) -> float:
    """Clamp model-provided scores to [0, 1] with safe float coercion."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _get_threshold_score(
    threshold_name: str,
    combined_score: float,
    parsed: Dict[str, Any],
) -> float:
    """Resolve the effective threshold score for the given threshold name.

    Custom thresholds registered via @register_threshold are checked first.
    Built-in thresholds fall back to THRESHOLD_SCORES.
    """
    if threshold_name in _registered_thresholds:
        # Custom threshold: create a SynthesisResult stub to pass to it
        stub = SynthesisResult(
            should_send=True,
            message=parsed.get("message"),
            reasoning=parsed.get("reasoning", ""),
            novelty_score=_clamp_unit_interval(parsed.get("novelty", 0.0)),
            relevance_score=_clamp_unit_interval(parsed.get("relevance", 0.0)),
            combined_score=combined_score,
        )
        # If custom threshold says no, return 1.1 (impossible to reach)
        return 0.0 if _registered_thresholds[threshold_name].should_send(stub) else 1.1
    return THRESHOLD_SCORES.get(threshold_name, THRESHOLD_SCORES[DEFAULT_THRESHOLD])


# ──────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────


def _build_synthesis_prompt(
    history: str,
    already_sent: str,
    graph_ctx: Optional[BartokGraphContext],
) -> str:
    """Build the synthesis prompt for the judge model.

    When BartokGraph context is available, adds a BARTOKGRAPH CONNECTIONS
    section that gives the judge access to cross-temporal knowledge.
    When not available, produces a clean recency-only prompt.
    """
    base = f"""You are deciding whether to send the user an unprompted message.
Your only job: find something genuinely worth saying. If nothing is, stay quiet.

RECENT CONVERSATION HISTORY:
{history}

ALREADY SENT TODAY (do not repeat):
{already_sent}
"""

    graph_section = ""
    if graph_ctx and graph_ctx.connections:
        lines = []
        for conn in graph_ctx.connections[:3]:
            lines.append(
                f"  [{conn.connection_type.upper()}] "
                f"'{conn.node_a_content}' ↔ '{conn.node_b_content}' "
                f"(strength {conn.strength:.2f}, {conn.days_apart}d ago) — "
                f"{conn.explanation}"
            )
        graph_section = f"""
BARTOKGRAPH CONNECTIONS — cross-temporal knowledge from past conversations:
These are non-obvious connections between today's work and things discussed weeks ago.
If one of these represents a genuinely surprising insight the user hasn't seen, surface it.

{chr(10).join(lines)}

Connection types:
  TEMPORAL_BRIDGE:   same concept appeared weeks ago — user may have forgotten the solution
  CROSS_DOMAIN:      concept from one domain structurally connects to a different domain
  PERSON_KNOWLEDGE:  connects today's work to something a specific person mentioned
"""

    instructions = """
WHAT'S WORTH SENDING:
1. A completed background task with a result the user cares about
2. An unresolved question from earlier that now has a clear answer
3. A TEMPORAL_BRIDGE — something from today echoes an older thread the user forgot
4. A CROSS_DOMAIN connection — "your X work and Y work share the same structure"
5. Something the user explicitly asked you to "let me know about" earlier

THE BAR IS HIGH. Prefer silence. If you're uncertain, set should_send=false.

COMPOSE THE MESSAGE naturally — as if continuing a conversation.
  Good: "Hey — just connected something. Your regime detection work and your 
         soil carbon research are solving the same problem: detecting state 
         transitions in noisy signals."
  Bad:  "BARTOKGRAPH ANALYSIS: cross-domain connection detected between..."

Lead with the surprise. Never mention BartokGraph or the mechanism to the user.
Keep it to 2-4 sentences.

JSON response:
{
  "should_send": true/false,
  "message": "natural message to send, or null",
  "novelty": 0.0-1.0,
  "relevance": 0.0-1.0,
  "connection_type": "temporal_bridge|cross_domain|person_knowledge|none",
  "reasoning": "1-2 sentences on why you send or don't",
  "candidates": ["things considered"]
}"""

    return base + graph_section + instructions


def _parse_synthesis_response(raw: str) -> Dict[str, Any]:
    """Parse synthesis model JSON response safely."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        return json.loads(text, strict=False)
    except Exception:  # noqa: BLE001
        logger.debug("PCL: failed to parse synthesis response: %r", raw[:200])
        return {
            "should_send": False, "message": None,
            "novelty": 0.0, "relevance": 0.0,
            "connection_type": "none",
            "reasoning": "parse failure", "candidates": [],
        }
