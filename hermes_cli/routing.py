"""
Routing intelligence (P2-9 — task-to-specialist matching).

Replaces keyword-only triage routing with a multi-signal scoring
engine that ranks specialists for any given task.

Signal model
------------
1. **Keyword signal** — specialist's `routing_keywords` config vs task
   title/body.  Weight: 0.50.
2. **Has-capability signal** — specialist's `capabilities` tags vs
   task domain indicators.  Weight: 0.30.
3. **Historical signal** — past routing decisions, weighted by Sahil's
   feedback (approve = +1, reject = -1, correct = reroute).  Weight: 0.20.
4. **Load signal** — (future) active task count per specialist.
   Weight: 0.10 (reduced from keyword when enabled).

The engine returns a ranked list of (specialist, score, confidence)
tuples.  The triage processor picks the top candidate.  Confidence
below 0.30 triggers "needs triage review" — routes to KENSEI.

Integration
-----------
- Denji triage processor calls `route()` before assigning
- Feedback loop (P2-10) feeds historical signal via `record_decision()`
- Specialist profiles define `routing_keywords` and `capabilities`
- Config: `kanban.routing` section with specialist definitions

Usage
-----
    from hermes_cli.routing import RoutingEngine

    engine = RoutingEngine(config_path="~/.hermes/config.yaml")
    results = engine.route(
        task_title="Fix memory leak in gateway",
        task_body="The dispatcher daemon leaks ~50MB/hr under load...",
    )
    # => [("wesker", 0.87, 0.92), ("octacon-frontend", 0.65, 0.70), ...]
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Default specialist definitions
# ---------------------------------------------------------------------------

# Each specialist has routing keywords, capabilities, and a domain.
# In production these come from config.yaml: `kanban.routing.specialists`.
DEFAULT_SPECIALISTS: dict[str, dict] = {
    "wesker": {
        "keywords": [
            "ops", "infra", "security", "gateway", "cron", "backup",
            "deploy", "systemd", "vps", "server", "container", "docker",
            "memory leak", "crash", "restart", "health", "monitor",
            "log", "error", "permission", "firewall", "ssh",
        ],
        "capabilities": ["devops", "security", "infrastructure"],
        "domain": "infrastructure & security",
    },
    "octacon-frontend": {
        "keywords": [
            "code", "bug", "fix", "feature", "implement", "refactor",
            "test", "build", "compile", "typeerror", "syntax",
            "component", "react", "api", "endpoint", "database",
            "schema", "migration", "pr", "merge", "commit",
            "node", "python", "typescript", "sql",
        ],
        "capabilities": ["coding", "debugging", "frontend", "backend"],
        "domain": "software development",
    },
    "remii-deep": {
        "keywords": [
            "research", "analysis", "compare", "evaluate", "survey",
            "market", "competitor", "trend", "signal", "scan",
            "deep dive", "report", "findings", "recommendation",
            "paper", "arxiv", "study", "benchmark",
        ],
        "capabilities": ["research", "analysis", "market-intel"],
        "domain": "research & intelligence",
    },
    "ceecee": {
        "keywords": [
            "content", "post", "tweet", "social", "brand", "copy",
            "draft", "publish", "schedule", "thread", "linkedin",
            "twitter", "blog", "article", "write", "edit",
            "tone", "voice", "messaging",
        ],
        "capabilities": ["content", "brand", "social-media"],
        "domain": "content & brand",
    },
    "gojo": {
        "keywords": [
            "admin", "calendar", "schedule", "meeting", "mail",
            "email", "invoice", "booking", "appointment", "travel",
            "flight", "hotel", "reminder", "todo", "task",
            "organize", "logistics",
        ],
        "capabilities": ["admin", "calendar", "logistics"],
        "domain": "admin & logistics",
    },
    "quan-code": {
        "keywords": [
            "qa", "test", "quality", "release", "verify", "validate",
            "regression", "coverage", "e2e", "integration test",
            "bug report", "reproduce", "steps", "checklist", "sign-off",
        ],
        "capabilities": ["qa", "testing", "quality-assurance"],
        "domain": "quality assurance",
    },
    "light-archivist": {
        "keywords": [
            "document", "runbook", "decision", "record", "archive",
            "knowledge", "wiki", "note", "log", "history",
            "post-mortem", "retro", "learnings",
        ],
        "capabilities": ["documentation", "knowledge-management"],
        "domain": "documentation & knowledge",
    },
}

# Weights for composite score
DEFAULT_KEYWORD_WEIGHT = 0.50
DEFAULT_CAPABILITY_WEIGHT = 0.30
DEFAULT_HISTORICAL_WEIGHT = 0.20
DEFAULT_LOAD_WEIGHT = 0.10  # future; reduces keyword weight when enabled

# Minimum confidence for auto-routing
MIN_CONFIDENCE_AUTO_ROUTE = 0.30

# Routing history database
ROUTING_HISTORY_DB = "~/.hermes/kensei/routing_history.db"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """A single routing recommendation."""
    specialist: str
    score: float       # 0.0–1.0 composite score
    confidence: float  # 0.0–1.0 confidence in this routing
    signals: dict[str, float]  # individual signal breakdown
    domain: str = ""
    reason: str = ""


@dataclass
class RouteDecision:
    """A historical routing decision for feedback learning."""
    task_id: str
    specialist: str
    score: float
    outcome: str = ""  # "accepted", "rejected", "reassigned_to"
    reassigned_to: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Tokenize text for keyword matching."""
    import re as _re
    return set(w.lower() for w in _re.findall(r'\b\w{3,}\b', text))


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Score how well keywords match the text.

    Uses cosine-style overlap: matched keywords / total keywords,
    weighted by how specific the match is (longer keywords > single-word).
    """
    text_tokens = _tokenize(text)
    if not keywords or not text_tokens:
        return 0.0

    matched = 0
    weighted = 0.0
    total_weight = 0.0

    for kw in keywords:
        kw_tokens = _tokenize(kw)
        kw_weight = len(kw_tokens)  # multi-word keywords are more specific
        total_weight += kw_weight

        # Check if all tokens of the keyword appear near each other in text
        if kw_tokens.issubset(text_tokens):
            # Bonus for exact phrase match (tokens adjacent)
            if kw.lower() in text.lower():
                weighted += kw_weight * 1.5
            else:
                weighted += kw_weight
            matched += 1

    if total_weight == 0:
        return 0.0

    # Normalise: weighted match ratio
    base_score = weighted / total_weight

    # Bonus for high match density (many keywords matched)
    density = matched / len(keywords) if keywords else 0
    return min(1.0, base_score * 0.7 + density * 0.3)


def _capability_score(text: str, capabilities: list[str], domain: str) -> float:
    """Score how a task's domain indicators overlap with specialist capabilities.

    Uses domain-indicator pattern matching: tasks that mention specific
    tech stacks, problem types, or outcome domains get higher scores.
    """
    text_lower = text.lower()

    # Domain indicator patterns
    domain_patterns: dict[str, list[str]] = {
        "devops": [
            r"\b(?:server|deploy|infra|ops|gateway|systemd|docker|container|cron)\b",
            r"\b(?:crash|restart|oom|memory|cpu|disk|health)\b",
        ],
        "coding": [
            r"\b(?:code|bug|fix|feature|implement|refactor|pr|merge|commit)\b",
            r"\b(?:function|class|module|import|export|type|interface)\b",
            r"\b(?:python|typescript|javascript|react|node\.?js|sql|rust)\b",
        ],
        "research": [
            r"\b(?:research|analysis|compare|evaluate|findings|recommend)\b",
            r"\b(?:study|benchmark|survey|scan|deep dive|report)\b",
        ],
        "content": [
            r"\b(?:content|post|tweet|thread|blog|article|social)\b",
            r"\b(?:brand|voice|tone|copy|draft|publish|schedule)\b",
        ],
        "admin": [
            r"\b(?:admin|calendar|schedule|meeting|booking|travel)\b",
            r"\b(?:email|invoice|reminder|organize|logistics)\b",
        ],
        "qa": [
            r"\b(?:qa|test|quality|release|verify|validate|regression)\b",
            r"\b(?:coverage|e2e|checklist|sign-off|bug report)\b",
        ],
        "documentation": [
            r"\b(?:document|runbook|decision|record|wiki|note|archive)\b",
            r"\b(?:knowledge|post-mortem|retro|learnings|history)\b",
        ],
        "security": [
            r"\b(?:security|auth|permission|firewall|ssh|secret|vault)\b",
            r"\b(?:vulnerability|exploit|inject|csrf|xss|sql injection)\b",
        ],
    }

    # Score each capability against the text
    cap_scores = []
    for cap in capabilities:
        patterns = domain_patterns.get(cap, [])
        if not patterns:
            # Fallback: check if cap name appears in text
            if cap.lower() in text_lower:
                cap_scores.append(0.5)
            continue

        hits = sum(1 for p in patterns if re.search(p, text_lower))
        if patterns:
            cap_scores.append(min(1.0, hits / len(patterns)))

    if not cap_scores:
        return 0.0

    return sum(cap_scores) / len(cap_scores)


# ---------------------------------------------------------------------------
# Routing engine
# ---------------------------------------------------------------------------


class RoutingEngine:
    """Multi-signal task-to-specialist routing engine.

    Usage:
        engine = RoutingEngine()
        results = engine.route("Fix cron job failure", "...")
        best = results[0]  # => RouteResult(specialist="wesker", score=0.85)
    """

    def __init__(
        self,
        *,
        specialists: dict[str, dict] | None = None,
        config_path: str = "",
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        capability_weight: float = DEFAULT_CAPABILITY_WEIGHT,
        historical_weight: float = DEFAULT_HISTORICAL_WEIGHT,
        history_db: str = ROUTING_HISTORY_DB,
    ):
        self.specialists = specialists or DEFAULT_SPECIALISTS
        self.keyword_weight = keyword_weight
        self.capability_weight = capability_weight
        self.historical_weight = historical_weight
        self.history_db = os.path.expanduser(history_db)

        # Normalise weights to sum to 1.0
        total = keyword_weight + capability_weight + historical_weight
        if total > 0:
            self.keyword_weight /= total
            self.capability_weight /= total
            self.historical_weight /= total

        self._init_history_db()

    def _init_history_db(self) -> None:
        os.makedirs(os.path.dirname(self.history_db), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_history (
                    task_id TEXT,
                    specialist TEXT,
                    score REAL,
                    outcome TEXT DEFAULT '',
                    reassigned_to TEXT DEFAULT '',
                    timestamp TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (task_id, specialist)
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.history_db)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        task_title: str,
        task_body: str = "",
        *,
        task_id: str = "",
        top_n: int = 5,
    ) -> list[RouteResult]:
        """Route a task to the best-matching specialists.

        Args:
            task_title: Task title (high signal weight).
            task_body: Task body for deeper matching.
            task_id: Task ID for historical signal lookup.
            top_n: Return top N results.

        Returns:
            Ranked list of RouteResult, best first.
        """
        combined_text = f"{task_title}\n{task_body}"

        results = []
        for name, spec in self.specialists.items():
            # 1. Keyword signal
            kw = _keyword_score(combined_text, spec.get("keywords", []))

            # 2. Capability signal
            cap = _capability_score(
                combined_text,
                spec.get("capabilities", []),
                spec.get("domain", ""),
            )

            # 3. Historical signal
            hist = self._historical_signal(name, task_id) if task_id else 0.5

            # Composite
            score = (
                kw * self.keyword_weight
                + cap * self.capability_weight
                + hist * self.historical_weight
            )

            # Confidence: weighted by signal agreement
            signals = [kw, cap, hist]
            signal_spread = max(signals) - min(signals)
            confidence = score * (1.0 - signal_spread * 0.5)

            # Build reason
            signal_parts = []
            if kw >= 0.5:
                signal_parts.append(f"keywords={kw:.0%}")
            if cap >= 0.3:
                signal_parts.append(f"capabilities={cap:.0%}")
            if hist != 0.5:
                signal_parts.append(f"history={hist:.0%}")

            reason = f"{spec.get('domain', name)}: {', '.join(signal_parts)}" if signal_parts else f"{spec.get('domain', name)}"

            results.append(RouteResult(
                specialist=name,
                score=score,
                confidence=confidence,
                signals={"keyword": kw, "capability": cap, "historical": hist},
                domain=spec.get("domain", ""),
                reason=reason,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def route_best(self, task_title: str, task_body: str = "", task_id: str = "") -> RouteResult | None:
        """Return the single best specialist or None if confidence too low."""
        results = self.route(task_title, task_body, task_id=task_id, top_n=1)
        if not results:
            return None
        if results[0].confidence < MIN_CONFIDENCE_AUTO_ROUTE:
            return None
        return results[0]

    # ------------------------------------------------------------------
    # Historical signal
    # ------------------------------------------------------------------

    def _historical_signal(self, specialist: str, task_id: str) -> float:
        """Compute historical routing signal for a specialist from past decisions.

        Each past assignment to this specialist that was 'accepted' adds
        +0.15 to the score; each 'rejected' subtracts 0.15.  Reassignments
        (outcome='reassigned_to') are neutral for the original and +0.10
        for the reassigned target.

        Returns 0.5 (neutral) when no history exists.
        """
        with self._connect() as conn:
            # Direct assignments
            rows = conn.execute(
                """
                SELECT outcome, reassigned_to
                FROM routing_history
                WHERE specialist = ?
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                (specialist,),
            ).fetchall()

        if not rows:
            return 0.5

        signals = []
        for r in rows:
            if r["outcome"] == "accepted":
                signals.append(0.15)
            elif r["outcome"] == "rejected":
                signals.append(-0.15)
            # reassigned_to is neutral for original specialist
            else:
                signals.append(0.0)

        # Sum dampened by count to prevent overfitting
        if not signals:
            return 0.5

        avg = sum(signals) / max(1, len(signals) ** 0.5)
        return max(0.0, min(1.0, 0.5 + avg))

    # ------------------------------------------------------------------
    # Feedback integration
    # ------------------------------------------------------------------

    def record_decision(self, decision: RouteDecision) -> None:
        """Record a routing decision for historical learning."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO routing_history
                (task_id, specialist, score, outcome, reassigned_to, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.task_id, decision.specialist, decision.score,
                    decision.outcome, decision.reassigned_to,
                    decision.timestamp or datetime.now(timezone.utc).isoformat(),
                ),
            )

    def record_feedback(
        self,
        task_id: str,
        specialist: str,
        outcome: str,
        *,
        score: float = 0.0,
        reassigned_to: str = "",
    ) -> None:
        """Record feedback (approved/rejected/reassigned) on a routing decision.

        This is the primary integration point with the FeedbackLoop (P2-10).
        When Sahil approves or rejects a task assignment, the triage processor
        calls this method to update the routing history.
        """
        self.record_decision(RouteDecision(
            task_id=task_id,
            specialist=specialist,
            score=score,
            outcome=outcome,
            reassigned_to=reassigned_to,
        ))

    # ------------------------------------------------------------------
    # Bulk specialist re-scoring (for periodic rebalancing)
    # ------------------------------------------------------------------

    def specialist_effectiveness(self, *, days: int = 30) -> dict[str, float]:
        """Compute specialist effectiveness scores from history.

        Effectiveness = accepted / (accepted + rejected) over the period.
        Returns 0.5 for specialists with no history (neutral).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT specialist, outcome, COUNT(*) as cnt
                FROM routing_history
                WHERE timestamp >= ?
                GROUP BY specialist, outcome
                """,
                (cutoff,),
            ).fetchall()

        # Aggregate per specialist
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})
        for r in rows:
            stats[r["specialist"]][r["outcome"]] = r["cnt"]

        effectiveness = {}
        for name in self.specialists:
            s = stats.get(name, {"accepted": 0, "rejected": 0})
            total = s["accepted"] + s["rejected"]
            if total == 0:
                effectiveness[name] = 0.5
            else:
                effectiveness[name] = s["accepted"] / total

        return effectiveness
