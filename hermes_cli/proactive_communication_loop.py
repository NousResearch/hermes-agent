"""Proactive Communication Loop — Hermes reaches out when it sees something the user can't.

The goal is magic.

Not notifications. Not task completion alerts. Not a nightly summary.
Those exist already — every task runner and reminder app does that.

This is different: Hermes traverses a weighted knowledge graph built from the
user's entire conversation history and surfaces connections that no human could
hold in their head. The user worked on something six weeks ago. Today's work
echoes it in a way they can't see — because they can't hold six weeks of context
simultaneously. Hermes can. It reaches out unprompted.

"Hey — just noticed something. Your work on X and what you were building three
weeks ago with Y are solving the same problem. The approach you found then applies
here directly."

That's the experience. That's what makes the agent feel alive.

Background
----------
Requested by @charlesmcdowell (2.2K views, May 8 2026). Teknium: "This is a good idea 🤔"

Architecture
------------
BartokGraph is NOT optional for this feature. It IS the feature.

Without BartokGraph connections, the loop stays silent. Deliberately.
This is not a notification system. The bar is: "would this genuinely surprise
the user in a way that changes how they think about their work right now?"
If the answer isn't clearly yes, nothing is sent.

Three connection types that trigger a message:

  TEMPORAL_BRIDGE   — same concept, separated by weeks.
    "You solved this before. You've forgotten. Here it is again."

  CROSS_DOMAIN      — structurally identical problem in different contexts.
    "Your trading bot work and your soil monitoring share the same math."

  PERSON_KNOWLEDGE  — something a person in your life said that connects to now.
    "Sarah mentioned the Kenya project last week. It connects to what you're
    building today in a way neither of you saw."

Design invariants
-----------------
- NEVER sends without a BartokGraph connection. Silence is the default.
- NEVER sends more than max_per_day messages (default: 1).
- NEVER mentions BartokGraph, the graph, or the mechanism. Lead with the insight.
- NEVER modifies session state, memory, or system prompt.
- Fails silent on any error — prefer quiet over noise.
- Fully opt-in: proactive_communication.enabled defaults to False.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = "conservative"
DEFAULT_MAX_PER_DAY = 1
DEFAULT_HISTORY_WINDOW_HOURS = 72   # look back 3 days for topic extraction
DEFAULT_SYNTHESIS_BUDGET_TOKENS = 2000
_HISTORY_SNIPPET_CHARS = 8000
_JUDGE_TIMEOUT = 30.0

# Threshold scores for the graph-connection quality gate.
# Only connections scoring above these are surfaced.
THRESHOLD_SCORES: Dict[str, float] = {
    "conservative": 0.75,  # Default. High bar. Silence is fine.
    "balanced": 0.55,      # Moderate. Solid connections get through.
    "eager": 0.35,         # Low bar. More magic, some noise.
}


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """Outcome of one Proactive Communication Loop synthesis pass."""

    should_send: bool
    message: Optional[str]
    reasoning: str             # written to audit log
    novelty_score: float       # 0–1: how surprising this connection is
    relevance_score: float     # 0–1: how useful to current work
    combined_score: float      # weighted combination for threshold check
    connection_type: str = "none"  # temporal_bridge | cross_domain | person_knowledge | none
    candidates: List[str] = field(default_factory=list)
    synthesis_ms: int = 0


@dataclass
class BartokGraphConnection:
    """A connection the knowledge graph found between now and the past.

    These are the moments that make the agent feel alive —
    non-obvious links across time and domain the user cannot see themselves.
    """

    node_a_content: str    # today's concept
    node_b_content: str    # past concept
    connection_type: str   # temporal_bridge | cross_domain | person_knowledge
    strength: float        # 0–1 semantic overlap
    days_apart: int        # how long since node_b was active
    explanation: str       # human-readable bridge


@dataclass
class BartokGraphContext:
    """Graph-augmented context for one synthesis pass."""

    connections: List[BartokGraphConnection]
    provider_name: str
    traversal_ms: int = 0


# ──────────────────────────────────────────────────────────────────────
# Pluggable threshold protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class ProactiveThreshold(Protocol):
    """Custom threshold — register via @register_threshold("name")."""

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
    """The engine that makes Hermes feel alive.

    Traverses the user's knowledge graph, finds connections they can't see,
    and reaches out unprompted when it finds something genuinely surprising.

    Usage (from gateway cron)::

        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)
        result = await loop.run_synthesis(session_id)
        if result.should_send and result.message:
            await deliver_message(result.message)
            await loop.record_sent(session_id, result)

    Returns a no-send result silently if BartokGraph is unavailable or finds
    no connections worth surfacing. Never raises.
    """

    def __init__(self, session_db: Any, config: Any) -> None:
        self._db = session_db
        self._cfg = config
        self._bartokgraph: Optional[Any] = self._try_load_bartokgraph()

    def _try_load_bartokgraph(self) -> Optional[Any]:
        """Load BartokGraph adapter. Never raises."""
        if not self._cfg.get("proactive_communication.bartokgraph.enabled", True):
            return None
        try:
            from hermes_cli.bartokgraph_adapter import BartokGraphAdapter
            return BartokGraphAdapter(config=self._cfg)
        except ImportError:
            logger.debug("PCL: BartokGraph not installed — loop will stay silent")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_synthesis(
        self,
        session_id: str,
        history_window_hours: int = DEFAULT_HISTORY_WINDOW_HOURS,
    ) -> SynthesisResult:
        """Run one synthesis pass. Never raises."""
        try:
            return await self._run_synthesis_inner(session_id, history_window_hours)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PCL: synthesis error for %s: %s", session_id, exc)
            return SynthesisResult(
                should_send=False, message=None,
                reasoning=f"synthesis error: {exc}",
                novelty_score=0.0, relevance_score=0.0, combined_score=0.0,
            )

    async def record_sent(self, session_id: str, result: SynthesisResult) -> None:
        """Record that a proactive message was sent (stored in state_meta)."""
        try:
            import datetime as _dt
            import json as _json
            today = _dt.date.today().isoformat()
            key = f"proactive_sent:{session_id}:{today}"
            existing_raw = self._db.get_meta(key)
            existing = _json.loads(existing_raw) if existing_raw else []
            if not isinstance(existing, list):
                existing = []
            existing.append({
                "summary": (result.message or "")[:200],
                "connection_type": result.connection_type,
                "score": result.combined_score,
                "ts": int(time.time()),
            })
            self._db.set_meta(key, _json.dumps(existing))
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

        # 1. BartokGraph is required. Without it, stay silent.
        if not self._bartokgraph:
            return SynthesisResult(
                should_send=False, message=None,
                reasoning="BartokGraph not available — loop requires graph connections to send",
                novelty_score=0.0, relevance_score=0.0, combined_score=0.0,
            )

        # 2. Rate limit
        if self._over_daily_limit(session_id):
            return SynthesisResult(
                should_send=False, message=None,
                reasoning="daily message limit reached",
                novelty_score=0.0, relevance_score=0.0, combined_score=0.0,
            )

        # 3. Load history for topic extraction
        history = self._load_recent_history(session_id, history_window_hours)
        if not history:
            return SynthesisResult(
                should_send=False, message=None,
                reasoning="no conversation history to extract topics from",
                novelty_score=0.0, relevance_score=0.0, combined_score=0.0,
            )

        # 4. Traverse the knowledge graph
        graph_ctx: Optional[BartokGraphContext] = None
        try:
            active_topics = await self._extract_topics_from_history(history)
            graph_ctx = await self._bartokgraph.get_connections(
                active_topics=active_topics,
                top_k=10,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("PCL: graph traversal failed: %s", exc)

        # 5. No connections = stay silent. This is the heart of the design.
        if not graph_ctx or not graph_ctx.connections:
            return SynthesisResult(
                should_send=False, message=None,
                reasoning="graph traversal found no connections worth surfacing",
                novelty_score=0.0, relevance_score=0.0, combined_score=0.0,
            )

        # 6. Ask the judge model: is any of this worth saying?
        already_sent = self._load_sent_summaries(session_id)
        prompt = _build_synthesis_prompt(history, already_sent, graph_ctx)
        raw = await self._call_synthesis_model(prompt)
        parsed = _parse_synthesis_response(raw)

        # 7. Score and threshold gate
        novelty = _clamp_unit_interval(parsed.get("novelty", 0.0))
        relevance = _clamp_unit_interval(parsed.get("relevance", 0.0))
        combined = 0.6 * novelty + 0.4 * relevance
        threshold_name = self._cfg.get("proactive_communication.threshold", DEFAULT_THRESHOLD)
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
            # SessionDB.get_messages() returns all messages; filter by timestamp
            all_messages = self._db.get_messages(session_id)
            messages = [
                m for m in all_messages
                if float(m.get("timestamp", 0)) >= cutoff
            ]
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
        """Load summaries of proactive messages sent today for deduplication."""
        try:
            # Stored in state_meta keyed by proactive:<session_id>:<date>
            import datetime as _dt
            today = _dt.date.today().isoformat()
            key = f"proactive_sent:{session_id}:{today}"
            raw = self._db.get_meta(key)
            if not raw:
                return "(none sent today)"
            import json as _json
            sent = _json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(sent, list):
                return "; ".join(s.get("summary", "") for s in sent[:5])
            return "(none sent today)"
        except Exception:  # noqa: BLE001
            return "(none sent today)"

    def _over_daily_limit(self, session_id: str) -> bool:
        try:
            limit = int(self._cfg.get("proactive_communication.max_per_day", DEFAULT_MAX_PER_DAY))
            import datetime as _dt
            today = _dt.date.today().isoformat()
            key = f"proactive_sent:{session_id}:{today}"
            raw = self._db.get_meta(key)
            if not raw:
                return False
            import json as _json
            sent = _json.loads(raw) if isinstance(raw, str) else raw
            return isinstance(sent, list) and len(sent) >= limit
        except Exception:  # noqa: BLE001
            return False

    async def _extract_topics_from_history(self, history: str) -> List[str]:
        """Extract the top topics from session history for graph traversal."""
        words = history.lower().split()
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
            "or", "is", "was", "are", "were", "i", "you", "me", "my", "your",
            "this", "that", "with", "from", "have", "had", "not", "but",
        }
        freq: Dict[str, int] = {}
        for word in words:
            clean = word.strip(".,!?;:\"'()")
            if len(clean) > 3 and clean not in stopwords:
                freq[clean] = freq.get(clean, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in top[:10]]

    async def _call_synthesis_model(self, prompt: str) -> str:
        """Call the auxiliary judge model via Hermes's configured provider.

        Uses the same auxiliary client pattern as GoalManager (goals.py).
        Prefers the cheapest/fastest model — this is a lightweight judge call,
        not a reasoning task.
        """
        try:
            from agent.auxiliary_client import get_text_auxiliary_client
        except ImportError as exc:
            raise RuntimeError("auxiliary client not available") from exc

        client, model = get_text_auxiliary_client("proactive_loop_judge")
        if client is None or not model:
            raise RuntimeError("no auxiliary client configured for proactive_loop_judge")

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a synthesis judge for a proactive communication system. "
                        "Your job is to evaluate whether a knowledge graph connection is "
                        "surprising and useful enough to message the user about unprompted. "
                        "Be strict. Silence is the right answer most of the time."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=DEFAULT_SYNTHESIS_BUDGET_TOKENS,
            timeout=_JUDGE_TIMEOUT,
        )
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────────────────────────────
# Threshold resolution
# ──────────────────────────────────────────────────────────────────────


def _clamp_unit_interval(value: Any) -> float:
    """Clamp model scores to [0, 1] safely."""
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
    if threshold_name in _registered_thresholds:
        stub = SynthesisResult(
            should_send=True,
            message=parsed.get("message"),
            reasoning=parsed.get("reasoning", ""),
            novelty_score=_clamp_unit_interval(parsed.get("novelty", 0.0)),
            relevance_score=_clamp_unit_interval(parsed.get("relevance", 0.0)),
            combined_score=combined_score,
        )
        return 0.0 if _registered_thresholds[threshold_name].should_send(stub) else 1.1
    return THRESHOLD_SCORES.get(threshold_name, THRESHOLD_SCORES[DEFAULT_THRESHOLD])


# ──────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────


def _build_synthesis_prompt(
    history: str,
    already_sent: str,
    graph_ctx: BartokGraphContext,
) -> str:
    """Build the judge prompt. graph_ctx is always present here — it's required."""
    lines = []
    for conn in graph_ctx.connections[:5]:
        lines.append(
            f"  [{conn.connection_type.upper()}] "
            f"'{conn.node_a_content}' ↔ '{conn.node_b_content}' "
            f"(strength {conn.strength:.2f}, {conn.days_apart}d ago) — "
            f"{conn.explanation}"
        )

    return f"""You are deciding whether to send the user an unprompted message.

The only valid reason to send: a connection in the knowledge graph that would
genuinely surprise the user and change how they think about their current work.
If that bar isn't clearly met, return should_send=false. Silence is correct.

RECENT CONVERSATION HISTORY (for context on what they're working on now):
{history}

KNOWLEDGE GRAPH CONNECTIONS (cross-temporal, from past conversations):
{chr(10).join(lines)}

Connection types:
  TEMPORAL_BRIDGE:   same concept appeared weeks ago — they may have forgotten the solution
  CROSS_DOMAIN:      structurally identical problem in a different context they're not seeing
  PERSON_KNOWLEDGE:  something a specific person mentioned that connects to their current work

ALREADY SENT TODAY (do not repeat):
{already_sent}

THE BAR: Would this genuinely surprise the user? Would it change how they approach
their work right now? If not clearly yes, set should_send=false.

COMPOSE THE MESSAGE as a natural, brief note — as if you just noticed something and
wanted to share it. 2-4 sentences maximum.
  Right: "Hey — just noticed something. Three weeks ago you were working on X, and
          what you're building now is the same problem from a different angle."
  Wrong: "GRAPH CONNECTION DETECTED: cross-domain link between..."

Never mention the graph, the mechanism, or how you found it. Lead with the insight.

JSON response:
{{
  "should_send": true/false,
  "message": "the message, or null",
  "novelty": 0.0-1.0,
  "relevance": 0.0-1.0,
  "connection_type": "temporal_bridge|cross_domain|person_knowledge|none",
  "reasoning": "1-2 sentences on why send or not",
  "candidates": ["connections considered"]
}}"""


def _parse_synthesis_response(raw: str) -> Dict[str, Any]:
    """Parse synthesis response safely."""
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
