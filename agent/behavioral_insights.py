"""
Behavioral Insights Engine for Hermes Agent.

Analyzes historical session data from the SQLite state database to produce a
behavioral profile — 5-axis numeric scores and 15 personality-driven insight
cards describing HOW the user works with their AI agents (not just how much).

This is the qualitative complement to ``agent/insights.py`` (which is purely
quantitative).  Both modules share the same ``SessionDB`` connection and
follow the same class structure: ``generate()`` → data gathering →
computation → ``format_terminal()`` / ``format_gateway()``.

Architecture (3 layers):

1. **Signal extraction** (pure Python + SQL, zero token cost) — 18 signal
   extractors covering prompt patterns, steering, crash-outs, planning,
   tool usage, timing, and model preference.
2. **LLM scoring + narrative cards** (uses user's configured model) — 5-axis
   behavior scores (1-10 each with rationale) + 4 LLM-generated narrative
   cards (archetype, agent relationship, growth edge, crash out).  Falls
   back to heuristic scores from signals when LLM is unavailable.
3. **Score persistence** (SQLite, no LLM) — stores results in the
   ``behavioral_scores`` table for v2 trend tracking.

Usage:
    from hermes_state import SessionDB
    from agent.behavioral_insights import BehavioralAnalyzer

    db = SessionDB()
    analyzer = BehavioralAnalyzer(db)
    report = analyzer.generate(days=30)
    print(analyzer.format_terminal(report))
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Credential scrubbing ──────────────────────────────────────────────
# Patterns scrubbed before any user message excerpt is sent to the LLM.
# Each pattern is compiled once at module load.
_CREDENTIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # OpenAI-style API keys: sk-...
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
    # Google API keys: AIza...
    (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
    # Bearer tokens
    (re.compile(r"[Bb]earer\s+[A-Za-z0-9_\-\.=]{20,}"), "Bearer [REDACTED]"),
    # Connection strings: postgres://, mongodb://, redis://, mysql://
    (re.compile(r"(postgres|mongodb|redis|mysql|mssql|amqp)://[^\s'\"]+"), r"\1://[REDACTED]"),
    # Password assignments in env-like strings: PASSWORD=..., PASS=...
    (re.compile(r"(?i)(password|passwd|pwd|pass|secret|token|api_key|apikey)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    # Private key blocks
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----", re.DOTALL), "[REDACTED PRIVATE KEY]"),
    # Generic hex tokens (40+ hex chars, common for GitHub/Stripe/etc)
    (re.compile(r"\b[0-9a-fA-F]{40,}\b"), "[REDACTED]"),
]


def _scrub_credentials(text: str) -> str:
    """Scrub API keys, tokens, passwords, and connection strings from *text*.

    All user message excerpts sent to the LLM are passed through this
    function first.  No raw credentials leave the machine.

    Args:
        text: The raw message text to scrub.

    Returns:
        The text with credential-like substrings replaced by ``[REDACTED]``.
    """
    if not text:
        return text or ""
    scrubbed: str = text
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


# ── Signal keyword lists ──────────────────────────────────────────────
_STEERING_KEYWORDS = re.compile(
    r"\b(no|stop|wait|don't|dont|instead|actually|wrong|not that|not this|undo|revert|rollback|nope|nah)\b",
    re.IGNORECASE,
)
_POLITENESS_KEYWORDS = re.compile(
    r"\b(thank|thanks|thankyou|please|appreciate|appreciated|grateful|cheers|ty)\b",
    re.IGNORECASE,
)
_PLAN_KEYWORDS = re.compile(
    r"\b(plan|planning|first\s+i['']?ll|let\s*me\s*think|steps?\s*:|approach\s*:|strategy|todo|roadmap|let'?s\s+plan|before\s+(we|i)\s+(start|begin|do))\b",
    re.IGNORECASE,
)
_FRUSTRATION_KEYWORDS = re.compile(
    r"\b(wtf|wth|damn|shit|fuck|fucking|stupid|broken|garbage|useless|crap|hell|seriously|come\s*on|are\s*you\s*kidding|for\s*fuck'?s\s*sake)\b",
    re.IGNORECASE,
)
_DECISION_KEYWORDS = re.compile(
    r"\b(use\s+\S+\s+not\s+\S+|switch\s+to\s+\S+|let'?s\s+go\s+with\s+\S+|let'?s\s+use\s+\S+|change\s+to\s+\S+|go\s+with\s+\S+|prefer\s+\S+|rather\s+\S+|instead\s+of\s+\S+|drop\s+\S+|replace\s+\S+|move\s+to\s+\S+)\b",
    re.IGNORECASE,
)
_VERIFICATION_TOOLS = {
    "terminal", "read_file", "search_files", "web_search",
    "web_extract", "browser_navigate", "browser_snapshot",
    "browser_vision", "browser_console",
}


def _word_count(text: str) -> int:
    """Return the word count of *text* (0 for empty/None)."""
    if not text:
        return 0
    return len(text.split())


def _is_all_caps(text: str, min_len: int = 5) -> bool:
    """Check if *text* is predominantly ALL CAPS (upper ratio > 60%, length >= min_len)."""
    if not text or len(text) < min_len:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) > 0.60


def _has_typos(text: str) -> bool:
    """Heuristic: does *text* look like it has typos (non-standard English).

    Checks for common typo patterns: doubled letters in odd places,
    missing vowels, 'teh'/'adn'/'taht' style transpositions, and
    very short words that aren't in a small common-word set.
    """
    if not text:
        return False
    typo_indicators = [
        "teh", "adn", "taht", "wrok", "waht", "tihs", "fro", "si",
        "ot", "fi", "wnat", "wnat", "ahve", "yuo", "thsi", "wrok",
        "wnat", "form", "jsut", "alot", "cant", "wont", "dont",
        "recieve", "seperate", "occured", "untill", "wich", "thru",
    ]
    lower = text.lower()
    for typo in typo_indicators:
        if typo in lower:
            return True
    # Also check for missing apostrophes in contractions: dont, cant, wont, im, ive
    if re.search(r"\b(dont|cant|wont|im\b|ive\b|youre|thats|whats|isnt|arent|wasnt|werent)\b", lower):
        return True
    return False


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    if seconds <= 0:
        return "0m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _bar(score: int, max_score: int = 10) -> str:
    """Generate a filled/unfilled bar chart string for a score.

    ``8/10`` → ``████████░░``, ``10/10`` → ``██████████``, ``0/10`` → ``░░░░░░░░░░``
    """
    score = max(0, min(score, max_score))
    filled = "█" * score
    empty = "░" * (max_score - score)
    return filled + empty


def _shannon_entropy(counts: List[int]) -> float:
    """Compute Shannon entropy (base 2) from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


class BehavioralAnalyzer:
    """Analyzes session history and produces a behavioral profile.

    Works directly with a :class:`~hermes_state.SessionDB` instance (or raw
    ``sqlite3`` connection) to query session and message data, following the
    same pattern as :class:`agent.insights.InsightsEngine`.

    The analyzer extracts 18 behavioral signals from user messages and
    session metadata, then produces 5-axis scores and 15 insight cards.
    When the LLM is available, scores and 4 narrative cards come from a
    single bounded LLM call.  When it fails, heuristic scores and
    signal-only cards are used instead (graceful degradation).
    """

    # Schema for the behavioral_scores persistence table.
    _CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS behavioral_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp REAL NOT NULL,
    days_window INTEGER NOT NULL,
    source_filter TEXT,
    user_id TEXT,
    execution_leverage INTEGER NOT NULL,
    steering INTEGER NOT NULL,
    engineering_quality INTEGER NOT NULL,
    product_thinking INTEGER NOT NULL,
    planning INTEGER NOT NULL,
    archetype TEXT,
    agent_relationship TEXT,
    growth_edge TEXT,
    raw_signals TEXT
)"""

    # Pre-computed query strings (parameterized — no user-controlled
    # value can alter query structure, same pattern as insights.py).
    #
    # Each query has four variants:
    #   _ALL       — no source filter, no user_id filter
    #   _SOURCE    — source filter, no user_id filter
    #   _USER      — user_id filter, no source filter
    #   _SOURCE_USER — both source and user_id filters
    #
    # Message queries also include ``AND (m.active = 1 OR m.compacted = 1)``
    # to exclude rewound (active=0, compacted=0) messages, matching the
    # pattern at hermes_state.py:4537.
    _GET_SESSIONS_ALL = (
        "SELECT id, source, model, started_at, ended_at, "
        "message_count, tool_call_count, parent_session_id, title "
        "FROM sessions WHERE started_at >= ? ORDER BY started_at ASC"
    )
    _GET_SESSIONS_SOURCE = (
        "SELECT id, source, model, started_at, ended_at, "
        "message_count, tool_call_count, parent_session_id, title "
        "FROM sessions WHERE started_at >= ? AND source = ? ORDER BY started_at ASC"
    )
    _GET_SESSIONS_USER = (
        "SELECT id, source, model, started_at, ended_at, "
        "message_count, tool_call_count, parent_session_id, title "
        "FROM sessions WHERE started_at >= ? AND user_id = ? ORDER BY started_at ASC"
    )
    _GET_SESSIONS_SOURCE_USER = (
        "SELECT id, source, model, started_at, ended_at, "
        "message_count, tool_call_count, parent_session_id, title "
        "FROM sessions WHERE started_at >= ? AND source = ? AND user_id = ? "
        "ORDER BY started_at ASC"
    )
    _GET_USER_MESSAGES_ALL = (
        "SELECT m.session_id, m.content, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND m.role = 'user' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_USER_MESSAGES_SOURCE = (
        "SELECT m.session_id, m.content, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND m.role = 'user' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_USER_MESSAGES_USER = (
        "SELECT m.session_id, m.content, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.user_id = ? AND m.role = 'user' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_USER_MESSAGES_SOURCE_USER = (
        "SELECT m.session_id, m.content, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND s.user_id = ? AND m.role = 'user' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_ASSISTANT_MESSAGES_ALL = (
        "SELECT m.session_id, m.content, m.tool_calls, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND m.role = 'assistant' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_ASSISTANT_MESSAGES_SOURCE = (
        "SELECT m.session_id, m.content, m.tool_calls, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND m.role = 'assistant' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_ASSISTANT_MESSAGES_USER = (
        "SELECT m.session_id, m.content, m.tool_calls, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.user_id = ? AND m.role = 'assistant' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_ASSISTANT_MESSAGES_SOURCE_USER = (
        "SELECT m.session_id, m.content, m.tool_calls, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND s.user_id = ? AND m.role = 'assistant' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_TOOL_MESSAGES_ALL = (
        "SELECT m.session_id, m.content, m.tool_name, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND m.role = 'tool' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_TOOL_MESSAGES_SOURCE = (
        "SELECT m.session_id, m.content, m.tool_name, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND m.role = 'tool' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_TOOL_MESSAGES_USER = (
        "SELECT m.session_id, m.content, m.tool_name, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.user_id = ? AND m.role = 'tool' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_TOOL_MESSAGES_SOURCE_USER = (
        "SELECT m.session_id, m.content, m.tool_name, m.timestamp "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND s.user_id = ? AND m.role = 'tool' "
        "AND (m.active = 1 OR m.compacted = 1) "
        "ORDER BY m.timestamp ASC"
    )
    _GET_ALL_TOOL_NAMES_ALL = (
        "SELECT m.tool_name, m.tool_calls "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND (m.role = 'tool' OR (m.role = 'assistant' AND m.tool_calls IS NOT NULL)) "
        "AND (m.active = 1 OR m.compacted = 1)"
    )
    _GET_ALL_TOOL_NAMES_SOURCE = (
        "SELECT m.tool_name, m.tool_calls "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? "
        "AND (m.role = 'tool' OR (m.role = 'assistant' AND m.tool_calls IS NOT NULL)) "
        "AND (m.active = 1 OR m.compacted = 1)"
    )
    _GET_ALL_TOOL_NAMES_USER = (
        "SELECT m.tool_name, m.tool_calls "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.user_id = ? "
        "AND (m.role = 'tool' OR (m.role = 'assistant' AND m.tool_calls IS NOT NULL)) "
        "AND (m.active = 1 OR m.compacted = 1)"
    )
    _GET_ALL_TOOL_NAMES_SOURCE_USER = (
        "SELECT m.tool_name, m.tool_calls "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.started_at >= ? AND s.source = ? AND s.user_id = ? "
        "AND (m.role = 'tool' OR (m.role = 'assistant' AND m.tool_calls IS NOT NULL)) "
        "AND (m.active = 1 OR m.compacted = 1)"
    )

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """Initialize with a SessionDB instance.

        Args:
            db: A :class:`~hermes_state.SessionDB` instance (from
                ``hermes_state.py``).  Must expose ``._conn`` (a sqlite3
                connection with ``row_factory = sqlite3.Row``).
            config: Optional behavior config dict (e.g. ``{"enabled": True,
                "model": "gpt-4o-mini"}``).  The ``model`` key, if present
                and non-empty, overrides the LLM model used for scoring.
                ``enabled`` is checked by the CLI/gateway callers before
                construction; the analyzer itself ignores it.
        """
        self.db = db
        self._conn = db._conn
        # Stash the behavior config so the LLM scorer can pick up
        # ``behavior.model`` (cheaper model override) without re-reading
        # config.yaml on every call.
        self._behavior_config: Dict[str, Any] = config if isinstance(config, dict) else {}
        # Ensure the behavioral_scores table exists (CREATE TABLE IF NOT
        # EXISTS is idempotent and safe on existing DBs).
        try:
            self._conn.execute(self._CREATE_TABLE_SQL)
            # Add user_id column to existing tables that predate it.
            # CREATE TABLE IF NOT EXISTS won't add the column to a table
            # that already exists, so we check and ALTER if missing.
            cols = {
                row[1] if isinstance(row, (tuple, list)) else row["name"]
                for row in self._conn.execute("PRAGMA table_info(behavioral_scores)").fetchall()
            }
            if "user_id" not in cols:
                self._conn.execute(
                    "ALTER TABLE behavioral_scores ADD COLUMN user_id TEXT"
                )
            self._conn.commit()
        except Exception as exc:
            logger.warning("Could not create/migrate behavioral_scores table: %s", exc)

    # =================================================================
    # Public API
    # =================================================================

    def generate(
        self, days: int = 30, source: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete behavioral profile.

        This is the main entry point.  It extracts all 18 signals,
        optionally calls the LLM for scoring + narrative cards, persists
        scores to the database, and returns a report dict suitable for
        :meth:`format_terminal` / :meth:`format_gateway`.

        Args:
            days: Number of days to look back (default: 30).
            source: Optional filter by source platform (e.g. ``"cli"``,
                ``"telegram"``).  ``None`` = all sources.
            user_id: Optional filter by user ID.  On a multi-user gateway
                this prevents cross-user data leaks.  ``None`` = all users
                (CLI is single-user, so ``None`` is safe there).

        Returns:
            A dict with keys: ``days``, ``source_filter``, ``empty``,
            ``signals``, ``scores``, ``cards``, ``llm_available``,
            ``session_count``.
        """
        cutoff = time.time() - (days * 86400)

        # ── Layer 1: Signal extraction ──
        signals = self._extract_signals(cutoff, source, user_id)
        signals["days"] = days

        if not signals["total_sessions"] or not signals["total_user_messages"]:
            return {
                "days": days,
                "source_filter": source,
                "empty": True,
                "signals": signals,
                "scores": {},
                "cards": {},
                "llm_available": False,
                "session_count": signals["total_sessions"],
            }

        # ── Layer 2: LLM scoring + narrative cards ──
        scores, llm_cards, llm_available = self._score_and_narrate(signals)

        # ── Deterministic insight cards (from signals, no LLM) ──
        deterministic_cards = self._build_deterministic_cards(signals)

        # ── Layer 3: Score persistence ──
        self._persist_scores(days, source, user_id, scores, llm_cards, signals)

        return {
            "days": days,
            "source_filter": source,
            "empty": False,
            "generated_at": time.time(),
            "signals": signals,
            "scores": scores,
            "cards": {**llm_cards, **deterministic_cards},
            "llm_available": llm_available,
            "session_count": signals["total_sessions"],
        }

    # =================================================================
    # Layer 1: Signal extraction (pure Python + SQL, zero token cost)
    # =================================================================

    def _extract_signals(
        self,
        cutoff: float,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract all 18 behavioral signals from the database.

        This is the core of Layer 1.  It fetches sessions and messages,
        then calls each of the 18 signal extractor methods.  Each
        extractor returns a dict that is merged into the final signals
        dict.

        Args:
            cutoff: Unix timestamp cutoff (only sessions after this).
            source: Optional source platform filter.
            user_id: Optional user ID filter (prevents cross-user data
                leaks on multi-user gateways).

        Returns:
            A dict containing all 18 signal categories plus aggregate
            counts (``total_sessions``, ``total_user_messages``).
        """
        sessions = self._get_sessions(cutoff, source, user_id)
        user_messages = self._get_user_messages(cutoff, source, user_id)
        assistant_messages = self._get_assistant_messages(cutoff, source, user_id)
        tool_messages = self._get_tool_messages(cutoff, source, user_id)

        signals: Dict[str, Any] = {
            "total_sessions": len(sessions),
            "total_user_messages": len(user_messages),
            "total_assistant_messages": len(assistant_messages),
            "total_tool_messages": len(tool_messages),
            "sessions": sessions,
        }

        # Signal 1: Prompt length distribution
        signals["prompt_length"] = self._extract_prompt_length(user_messages)

        # Signal 2: Go-to prompts
        signals["go_to_prompts"] = self._extract_go_to_prompts(user_messages)

        # Signal 3: Steering frequency
        signals["steering"] = self._extract_steering(user_messages)

        # Signal 4: Politeness
        signals["politeness"] = self._extract_politeness(user_messages)

        # Signal 5: Crash outs
        signals["crash_outs"] = self._extract_crash_outs(user_messages)

        # Signal 6: Cryptic prompts
        signals["cryptic_prompts"] = self._extract_cryptic_prompts(user_messages)

        # Signal 7: Planning habits
        signals["planning"] = self._extract_planning_habits(user_messages, sessions)

        # Signal 8: Agent parallelism
        signals["agent_parallelism"] = self._extract_agent_parallelism(sessions)

        # Signal 9: Session topics
        signals["session_topics"] = self._extract_session_topics(sessions, user_messages)

        # Signal 10: Verification rate
        signals["verification"] = self._extract_verification_rate(assistant_messages, tool_messages)

        # Signal 11: Error recovery
        signals["error_recovery"] = self._extract_error_recovery(tool_messages, user_messages)

        # Signal 12: Productivity timing
        signals["productivity_timing"] = self._extract_productivity_timing(sessions)

        # Signal 13: Shipping timing
        signals["shipping_timing"] = self._extract_shipping_timing(sessions)

        # Signal 14: Model preference
        signals["model_preference"] = self._extract_model_preference(sessions)

        # Signal 15: Subagent dispatch patterns
        signals["subagent_dispatch"] = self._extract_subagent_dispatch(assistant_messages)

        # Signal 16: Decision patterns
        signals["decision_patterns"] = self._extract_decision_patterns(user_messages)

        # Signal 17: Tool diversity
        signals["tool_diversity"] = self._extract_tool_diversity(cutoff, source, user_id)

        # Signal 18: Session duration distribution
        signals["session_duration"] = self._extract_session_duration(sessions)

        # Signal 19: Skill usage (Hermes-exclusive)
        signals["skill_usage"] = self._extract_skill_usage(assistant_messages)

        # Signal 20: Memory management (Hermes-exclusive)
        signals["memory_management"] = self._extract_memory_management(assistant_messages)

        # Signal 21: Cross-session memory (Hermes-exclusive)
        signals["cross_session_memory"] = self._extract_cross_session_memory(assistant_messages, sessions)

        # Signal 22: Cron autonomy (Hermes-exclusive)
        signals["cron_autonomy"] = self._extract_cron_autonomy(cutoff, source, user_id)

        # Signal 23: Delegation patterns (Hermes-exclusive)
        signals["delegation_patterns"] = self._extract_delegation_patterns(assistant_messages, sessions)

        # Signal 24: Task management (Hermes-exclusive)
        task_mgmt = self._extract_task_management(assistant_messages)
        task_mgmt["sessions_without_todos"] = max(0, len(sessions) - task_mgmt["sessions_with_todos"])
        signals["task_management"] = task_mgmt

        # Signal 25: Model effectiveness (Hermes-exclusive)
        signals["model_effectiveness"] = self._extract_model_effectiveness(sessions, user_messages)

        # Signal 26: Skill ROI (Hermes-exclusive)
        signals["skill_roi"] = self._extract_skill_roi(sessions, assistant_messages)

        # Signal 27: Session abandonment (Hermes-exclusive)
        signals["session_abandonment"] = self._extract_session_abandonment(cutoff, source, user_id)

        return signals

    # ── Data gathering (SQL) ──────────────────────────────────────────

    def _get_sessions(
        self, cutoff: float, source: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch sessions within the time window, ordered oldest-first."""
        if source and user_id:
            cursor = self._conn.execute(self._GET_SESSIONS_SOURCE_USER, (cutoff, source, user_id))
        elif source:
            cursor = self._conn.execute(self._GET_SESSIONS_SOURCE, (cutoff, source))
        elif user_id:
            cursor = self._conn.execute(self._GET_SESSIONS_USER, (cutoff, user_id))
        else:
            cursor = self._conn.execute(self._GET_SESSIONS_ALL, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def _get_user_messages(
        self, cutoff: float, source: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all user-role messages within the time window."""
        if source and user_id:
            cursor = self._conn.execute(self._GET_USER_MESSAGES_SOURCE_USER, (cutoff, source, user_id))
        elif source:
            cursor = self._conn.execute(self._GET_USER_MESSAGES_SOURCE, (cutoff, source))
        elif user_id:
            cursor = self._conn.execute(self._GET_USER_MESSAGES_USER, (cutoff, user_id))
        else:
            cursor = self._conn.execute(self._GET_USER_MESSAGES_ALL, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def _get_assistant_messages(
        self, cutoff: float, source: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all assistant-role messages within the time window."""
        if source and user_id:
            cursor = self._conn.execute(self._GET_ASSISTANT_MESSAGES_SOURCE_USER, (cutoff, source, user_id))
        elif source:
            cursor = self._conn.execute(self._GET_ASSISTANT_MESSAGES_SOURCE, (cutoff, source))
        elif user_id:
            cursor = self._conn.execute(self._GET_ASSISTANT_MESSAGES_USER, (cutoff, user_id))
        else:
            cursor = self._conn.execute(self._GET_ASSISTANT_MESSAGES_ALL, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def _get_tool_messages(
        self, cutoff: float, source: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all tool-role messages within the time window."""
        if source and user_id:
            cursor = self._conn.execute(self._GET_TOOL_MESSAGES_SOURCE_USER, (cutoff, source, user_id))
        elif source:
            cursor = self._conn.execute(self._GET_TOOL_MESSAGES_SOURCE, (cutoff, source))
        elif user_id:
            cursor = self._conn.execute(self._GET_TOOL_MESSAGES_USER, (cutoff, user_id))
        else:
            cursor = self._conn.execute(self._GET_TOOL_MESSAGES_ALL, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    # ── Signal 1: Prompt length distribution ───────────────────────────

    def _extract_prompt_length(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Extract prompt length distribution from user messages.

        Buckets: ``<10`` words, ``10-50`` words, ``50-100`` words, ``100+`` words.
        Also tracks the longest prompt and average prompt length.

        Returns a dict with keys: ``buckets`` (dict), ``longest`` (int),
        ``average`` (float), ``total`` (int).
        """
        buckets = {"lt_10": 0, "10_50": 0, "50_100": 0, "100_plus": 0}
        longest = 0
        total_words = 0
        count = 0

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            wc = _word_count(content)
            if wc == 0:
                continue
            count += 1
            total_words += wc
            longest = max(longest, wc)
            if wc < 10:
                buckets["lt_10"] += 1
            elif wc < 50:
                buckets["10_50"] += 1
            elif wc < 100:
                buckets["50_100"] += 1
            else:
                buckets["100_plus"] += 1

        return {
            "buckets": buckets,
            "longest": longest,
            "average": (total_words / count) if count > 0 else 0.0,
            "total": count,
        }

    # ── Signal 2: Go-to prompts ───────────────────────────────────────

    def _extract_go_to_prompts(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Extract most-repeated short prompts (≤8 words).

        Identifies the user's "go-to" prompts by frequency across
        sessions.  Returns the top prompts with their counts.

        Returns a dict with keys: ``top`` (list of {prompt, count, sessions}),
        ``total_short`` (int).
        """
        prompt_counts: Counter = Counter()
        prompt_sessions: Dict[str, set] = defaultdict(set)

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            content = content.strip()
            wc = _word_count(content)
            if wc == 0 or wc > 8:
                continue
            # Normalize: lowercase, collapse whitespace
            normalized = " ".join(content.lower().split())
            prompt_counts[normalized] += 1
            prompt_sessions[normalized].add(msg.get("session_id", ""))

        top = [
            {
                "prompt": prompt,
                "count": count,
                "sessions": len(prompt_sessions[prompt]),
            }
            for prompt, count in prompt_counts.most_common(10)
        ]

        return {
            "top": top,
            "total_short": sum(prompt_counts.values()),
        }

    # ── Signal 3: Steering frequency ─────────────────────────────────

    def _extract_steering(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Count messages containing redirect/correction keywords.

        Keywords: "no", "stop", "wait", "don't", "instead", "actually",
        "wrong", "not that".

        Returns a dict with keys: ``count`` (int), ``rate`` (float —
        fraction of user messages containing a steering keyword),
        ``examples`` (list of matching message contents, ≤200 chars).
        """
        count = 0
        examples: List[str] = []
        total = 0

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            total += 1
            if _STEERING_KEYWORDS.search(content):
                count += 1
                if len(examples) < 5:
                    examples.append(content[:200])

        return {
            "count": count,
            "rate": (count / total) if total > 0 else 0.0,
            "examples": examples,
        }

    # ── Signal 4: Politeness ────────────────────────────────────────────

    def _extract_politeness(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Count politeness keyword occurrences in user messages.

        Keywords: "thank", "please", "appreciate", "thanks".

        Returns a dict with keys: ``thank_count`` (int), ``please_count``
        (int), ``total`` (int), ``examples`` (list, ≤5).
        """
        thank_count = 0
        please_count = 0
        examples: List[str] = []

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            matches = _POLITENESS_KEYWORDS.findall(content)
            if matches:
                thank_count += sum(
                    1 for m in matches
                    if "thank" in m.lower() or "thanks" in m.lower()
                    or "appreciat" in m.lower() or "grateful" in m.lower()
                    or "cheers" in m.lower() or m.lower() == "ty"
                )
                please_count += sum(1 for m in matches if "please" in m.lower())
                if len(examples) < 5:
                    examples.append(content[:200])

        return {
            "thank_count": thank_count,
            "please_count": please_count,
            "total": thank_count + please_count,
            "examples": examples,
        }

    # ── Signal 5: Crash outs ───────────────────────────────────────────

    def _extract_crash_outs(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Detect crash-out messages (frustration spikes).

        A crash-out is a user message with ALL CAPS (>60% uppercase,
        >5 chars), high exclamation density (>3 per message), or
        frustration keywords.

        Returns a dict with keys: ``count`` (int), ``messages`` (list
        of {content, caps_pct, exclamations, has_keyword}, ≤3 top).
        """
        crash_outs: List[Dict[str, Any]] = []

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            content = content.strip()
            if not content:
                continue

            # ALL CAPS detection
            letters = [c for c in content if c.isalpha()]
            caps_pct = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
            is_caps = caps_pct > 0.60 and len(content) >= 5

            # Exclamation density
            exclamations = content.count("!")
            is_exclaim = exclamations > 3

            # Frustration keywords
            has_keyword = bool(_FRUSTRATION_KEYWORDS.search(content))

            if is_caps or is_exclaim or has_keyword:
                crash_outs.append({
                    "content": content[:200],
                    "caps_pct": round(caps_pct * 100),
                    "exclamations": exclamations,
                    "has_keyword": has_keyword,
                })

        # Sort by intensity: caps_pct + exclamations
        crash_outs.sort(
            key=lambda c: (c["caps_pct"], c["exclamations"], c["has_keyword"]),
            reverse=True,
        )

        return {
            "count": len(crash_outs),
            "messages": crash_outs[:3],
        }

    # ── Signal 6: Cryptic prompts ─────────────────────────────────────

    def _extract_cryptic_prompts(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Detect cryptic prompts: short (<15 chars), late-night (11PM-4AM), with typos.

        Returns a dict with keys: ``count`` (int), ``prompts`` (list
        of {content, hour, has_typos}, ≤3 top).
        """
        cryptic: List[Dict[str, Any]] = []

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            content = content.strip()
            if len(content) >= 15 or len(content) == 0:
                continue

            timestamp = msg.get("timestamp")
            if timestamp is None:
                continue
            try:
                hour = datetime.fromtimestamp(float(timestamp)).hour
            except (TypeError, ValueError, OSError):
                continue

            # Late night: 11 PM (23) to 4 AM (4)
            if hour not in {23, 0, 1, 2, 3, 4}:
                continue

            cryptic.append({
                "content": content[:200],
                "hour": hour,
                "has_typos": _has_typos(content),
            })

        return {
            "count": len(cryptic),
            "prompts": cryptic[:3],
        }

    # ── Signal 7: Planning habits ──────────────────────────────────────

    def _extract_planning_habits(
        self, user_messages: List[Dict], sessions: List[Dict]
    ) -> Dict[str, Any]:
        """Detect plan-phrase usage before first tool call per session.

        For each session, checks if the first user message contains a
        plan-phrase (before any tool call occurs).  The planning rate is
        the fraction of sessions with a plan-first message.

        Returns a dict with keys: ``planned_sessions`` (int),
        ``total_sessions`` (int), ``rate`` (float).
        """
        if not sessions:
            return {"planned_sessions": 0, "total_sessions": 0, "rate": 0.0}

        # Group first user message per session
        first_msg_per_session: Dict[str, str] = {}
        for msg in user_messages:
            sid = msg.get("session_id")
            content = msg.get("content")
            if not sid or not content:
                continue
            if sid not in first_msg_per_session:
                first_msg_per_session[sid] = content

        planned = 0
        session_ids = {s["id"] for s in sessions if s.get("id")}
        for sid, first_msg in first_msg_per_session.items():
            if sid in session_ids and _PLAN_KEYWORDS.search(first_msg):
                planned += 1

        total = len(session_ids)
        return {
            "planned_sessions": planned,
            "total_sessions": total,
            "rate": (planned / total) if total > 0 else 0.0,
        }

    # ── Signal 8: Agent parallelism ────────────────────────────────────

    def _extract_agent_parallelism(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Count subagent sessions per parent and detect concurrent windows.

        Uses ``parent_session_id`` to find subagent sessions.  Computes
        max subagents per parent, average per parent, and max concurrent
        (overlapping session time windows).

        Returns a dict with keys: ``total_subagents`` (int),
        ``max_per_parent`` (int), ``avg_per_parent`` (float),
        ``max_concurrent`` (int), ``parent_count`` (int).
        """
        subagents_by_parent: Dict[str, List[Dict]] = defaultdict(list)
        for s in sessions:
            parent = s.get("parent_session_id")
            if parent:
                subagents_by_parent[parent].append(s)

        max_per_parent = max((len(v) for v in subagents_by_parent.values()), default=0)
        total_subagents = sum(len(v) for v in subagents_by_parent.values())
        parent_count = len(subagents_by_parent)
        avg_per_parent = (total_subagents / parent_count) if parent_count > 0 else 0.0

        # Max concurrent: overlapping session time windows
        max_concurrent = self._compute_max_concurrent(sessions)

        return {
            "total_subagents": total_subagents,
            "max_per_parent": max_per_parent,
            "avg_per_parent": round(avg_per_parent, 1),
            "max_concurrent": max_concurrent,
            "parent_count": parent_count,
        }

    def _compute_max_concurrent(self, sessions: List[Dict]) -> int:
        """Compute the maximum number of concurrently active sessions.

        Uses a sweep-line algorithm on session start/end timestamps.
        Sessions without an end time are assumed to last at least 60s.
        """
        events: List[Tuple[float, int]] = []  # (timestamp, +1/-1)
        for s in sessions:
            start = s.get("started_at")
            end = s.get("ended_at")
            if not start:
                continue
            if not end or end <= start:
                end = start + 60  # assume minimum 60s duration
            events.append((start, 1))
            events.append((end, -1))

        events.sort(key=lambda e: (e[0], e[1]))
        current = 0
        max_conc = 0
        for _, delta in events:
            current += delta
            max_conc = max(max_conc, current)
        return max_conc

    # ── Signal 9: Session topics ───────────────────────────────────────

    def _extract_session_topics(
        self, sessions: List[Dict], user_messages: List[Dict]
    ) -> Dict[str, Any]:
        """Extract session topic (title or first user prompt) for context.

        Returns a dict with keys: ``topics`` (dict: session_id → topic
        string), ``diversity`` (int — distinct topics count).
        """
        # First user message per session
        first_msg_per_session: Dict[str, str] = {}
        for msg in user_messages:
            sid = msg.get("session_id")
            content = msg.get("content")
            if not sid or not content:
                continue
            if sid not in first_msg_per_session:
                first_msg_per_session[sid] = content[:100]

        topics: Dict[str, str] = {}
        for s in sessions:
            sid = s.get("id")
            if not sid:
                continue
            title = s.get("title")
            if title and title.strip():
                topics[sid] = title.strip()[:100]
            elif sid in first_msg_per_session:
                topics[sid] = first_msg_per_session[sid]
            else:
                topics[sid] = "untitled"

        distinct = len(set(topics.values()))
        return {
            "topics": topics,
            "diversity": distinct,
        }

    # ── Signal 10: Verification rate ────────────────────────────────────

    def _extract_verification_rate(
        self, assistant_messages: List[Dict], tool_messages: List[Dict]
    ) -> Dict[str, Any]:
        """Measure verification rate: "done" declarations followed by verification tool calls.

        A verification tool call is one of: terminal, read_file,
        search_files, web_search, web_extract, browser_*.

        Returns a dict with keys: ``done_declarations`` (int),
        ``verified`` (int), ``rate`` (float).
        """
        done_keywords = re.compile(
            r"\b(done|complete|completed|finished|finish|all set|that'?s it|there you go|wrapped up|shipped)\b",
            re.IGNORECASE,
        )

        # Build a timeline of tool calls per session
        tool_calls_by_session: Dict[str, List[Dict]] = defaultdict(list)
        for tm in tool_messages:
            sid = tm.get("session_id")
            if sid:
                tool_calls_by_session[sid].append(tm)

        # Also extract tool calls from assistant messages
        for am in assistant_messages:
            sid = am.get("session_id")
            if not sid:
                continue
            calls = am.get("tool_calls")
            if calls:
                try:
                    if isinstance(calls, str):
                        calls = json.loads(calls)
                    if isinstance(calls, list):
                        for call in calls:
                            func = call.get("function", {}) if isinstance(call, dict) else {}
                            name = func.get("name", "")
                            if name:
                                tool_calls_by_session[sid].append({
                                    "tool_name": name,
                                    "timestamp": am.get("timestamp"),
                                })
                except (json.JSONDecodeError, TypeError):
                    pass

        done_count = 0
        verified = 0

        for am in assistant_messages:
            content = am.get("content")
            sid = am.get("session_id")
            ts = am.get("timestamp")
            if not content or not sid or ts is None:
                continue
            if not done_keywords.search(content):
                continue
            done_count += 1
            # Check if a verification tool call follows this message
            for tc in tool_calls_by_session.get(sid, []):
                tc_ts = tc.get("timestamp")
                tc_name = tc.get("tool_name", "")
                if tc_ts is not None and tc_ts >= ts and tc_name in _VERIFICATION_TOOLS:
                    verified += 1
                    break

        return {
            "done_declarations": done_count,
            "verified": verified,
            "rate": (verified / done_count) if done_count > 0 else 0.0,
        }

    # ── Signal 11: Error recovery ──────────────────────────────────────

    def _extract_error_recovery(
        self, tool_messages: List[Dict], user_messages: List[Dict]
    ) -> Dict[str, Any]:
        """Measure error recovery: errors followed by retry vs user intervention.

        An error output is a tool message containing error indicators.
        Recovery is when the next tool call is a retry (same tool).
        User intervention is when the next message is from the user.

        Returns a dict with keys: ``total_errors`` (int), ``self_recovered``
        (int), ``user_intervened`` (int), ``recovery_rate`` (float).
        """
        error_keywords = re.compile(
            r"(error|exception|traceback|failed|failure|fatal|crash|exit code [1-9])",
            re.IGNORECASE,
        )

        # Group all messages by session and sort by timestamp
        all_by_session: Dict[str, List[Dict]] = defaultdict(list)
        for tm in tool_messages:
            sid = tm.get("session_id")
            if sid:
                all_by_session[sid].append({
                    "role": "tool",
                    "content": tm.get("content", ""),
                    "tool_name": tm.get("tool_name", ""),
                    "timestamp": tm.get("timestamp"),
                })
        for um in user_messages:
            sid = um.get("session_id")
            if sid:
                all_by_session[sid].append({
                    "role": "user",
                    "content": um.get("content", ""),
                    "timestamp": um.get("timestamp"),
                })

        # Sort each session's messages by timestamp
        for sid in all_by_session:
            all_by_session[sid].sort(key=lambda m: m.get("timestamp") or 0)

        total_errors = 0
        self_recovered = 0
        user_intervened = 0

        for sid, msgs in all_by_session.items():
            for i, msg in enumerate(msgs):
                if msg["role"] != "tool":
                    continue
                content = msg.get("content", "")
                if not content or not error_keywords.search(str(content)):
                    continue
                total_errors += 1
                # Find the next message
                if i + 1 < len(msgs):
                    next_msg = msgs[i + 1]
                    if next_msg["role"] == "tool":
                        # Could be a retry (same tool) or a different tool
                        self_recovered += 1
                    elif next_msg["role"] == "user":
                        user_intervened += 1

        recovery_rate = (self_recovered / total_errors) if total_errors > 0 else 0.0
        return {
            "total_errors": total_errors,
            "self_recovered": self_recovered,
            "user_intervened": user_intervened,
            "recovery_rate": recovery_rate,
        }

    # ── Signal 12: Productivity timing ────────────────────────────────

    def _extract_productivity_timing(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze session activity by hour to classify productivity pattern.

        Classifies as: "night_owl" (peak 22-04), "early_bird" (peak 05-09),
        "afternoon_grinder" (peak 12-17), "evening" (peak 17-22).

        Returns a dict with keys: ``hour_counts`` (dict: hour → count),
        ``peak_hour`` (int), ``classification`` (str), ``peak_pct`` (float).
        """
        hour_counts: Counter = Counter()
        for s in sessions:
            ts = s.get("started_at")
            if ts is None:
                continue
            try:
                hour = datetime.fromtimestamp(float(ts)).hour
            except (TypeError, ValueError, OSError):
                continue
            hour_counts[hour] += 1

        if not hour_counts:
            return {
                "hour_counts": dict(hour_counts),
                "peak_hour": None,
                "classification": "unknown",
                "peak_pct": 0.0,
            }

        peak_hour = hour_counts.most_common(1)[0][0]
        total = sum(hour_counts.values())
        peak_count = hour_counts[peak_hour]
        peak_pct = (peak_count / total) if total > 0 else 0.0

        # Classification by peak hour range
        if 22 <= peak_hour or peak_hour <= 4:
            classification = "night_owl"
        elif 5 <= peak_hour <= 9:
            classification = "early_bird"
        elif 12 <= peak_hour <= 17:
            classification = "afternoon_grinder"
        elif 17 <= peak_hour <= 21:
            classification = "evening"
        else:
            classification = "afternoon_grinder"  # default for 10-11

        return {
            "hour_counts": dict(hour_counts),
            "peak_hour": peak_hour,
            "classification": classification,
            "peak_pct": round(peak_pct * 100),
        }

    # ── Signal 13: Shipping timing ─────────────────────────────────────

    def _extract_shipping_timing(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Find the day of week with the highest session count.

        Returns a dict with keys: ``day_counts`` (dict: day_name → count),
        ``peak_day`` (str or None), ``peak_count`` (int).
        """
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        day_counts: Counter = Counter()

        for s in sessions:
            ts = s.get("started_at")
            if ts is None:
                continue
            try:
                weekday = datetime.fromtimestamp(float(ts)).weekday()
            except (TypeError, ValueError, OSError):
                continue
            day_counts[day_names[weekday]] += 1

        if not day_counts:
            return {
                "day_counts": dict(day_counts),
                "peak_day": None,
                "peak_count": 0,
            }

        peak_day, peak_count = day_counts.most_common(1)[0]
        return {
            "day_counts": dict(day_counts),
            "peak_day": peak_day,
            "peak_count": peak_count,
        }

    # ── Signal 14: Model preference ─────────────────────────────────────

    def _extract_model_preference(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Count sessions per model, with percentages.

        Returns a dict with keys: ``models`` (list of {model, count, pct}),
        ``top_model`` (str or None), ``top_pct`` (int).
        """
        model_counts: Counter = Counter()
        for s in sessions:
            model = s.get("model") or "unknown"
            display = model.split("/")[-1] if "/" in model else model
            model_counts[display] += 1

        total = sum(model_counts.values())
        models = [
            {
                "model": model,
                "count": count,
                "pct": round((count / total) * 100) if total > 0 else 0,
            }
            for model, count in model_counts.most_common()
        ]

        if not models:
            return {"models": [], "top_model": None, "top_pct": 0}

        return {
            "models": models,
            "top_model": models[0]["model"],
            "top_pct": models[0]["pct"],
        }

    # ── Signal 15: Subagent dispatch patterns ──────────────────────────

    def _extract_subagent_dispatch(self, assistant_messages: List[Dict]) -> Dict[str, Any]:
        """Extract subagent dispatch patterns from delegate_task tool calls.

        Counts total dispatches, background vs foreground, and extracts
        task descriptions (≤200 chars, scrubbed).

        Returns a dict with keys: ``total`` (int), ``background`` (int),
        ``foreground`` (int), ``task_descriptions`` (list of ≤5).
        """
        total = 0
        background = 0
        foreground = 0
        task_descriptions: List[str] = []

        for am in assistant_messages:
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                name = func.get("name", "")
                if name != "delegate_task":
                    continue
                total += 1
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if isinstance(args, dict):
                    if args.get("background"):
                        background += 1
                    else:
                        foreground += 1
                    desc = args.get("task") or args.get("description") or ""
                    if desc and len(task_descriptions) < 5:
                        task_descriptions.append(_scrub_credentials(str(desc)[:200]))

        return {
            "total": total,
            "background": background,
            "foreground": foreground,
            "task_descriptions": task_descriptions,
        }

    # ── Signal 16: Decision patterns ────────────────────────────────────

    def _extract_decision_patterns(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Extract decision-shaped prompts from user messages.

        Looks for patterns like "use X not Y", "switch to Z", "let's go
        with W".  Returns count and examples.

        Returns a dict with keys: ``count`` (int), ``examples`` (list
        of ≤5).
        """
        count = 0
        examples: List[str] = []

        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            if _DECISION_KEYWORDS.search(content):
                count += 1
                if len(examples) < 5:
                    examples.append(content[:200])

        return {
            "count": count,
            "examples": examples,
        }

    # ── Signal 17: Tool diversity ───────────────────────────────────────

    def _extract_tool_diversity(
        self,
        cutoff: float,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Measure tool diversity: distinct tools per session + Shannon entropy.

        Returns a dict with keys: ``distinct_tools`` (int),
        ``tool_counts`` (dict: tool_name → count), ``entropy`` (float),
        ``avg_per_session`` (float).
        """
        if source and user_id:
            cursor = self._conn.execute(self._GET_ALL_TOOL_NAMES_SOURCE_USER, (cutoff, source, user_id))
        elif source:
            cursor = self._conn.execute(self._GET_ALL_TOOL_NAMES_SOURCE, (cutoff, source))
        elif user_id:
            cursor = self._conn.execute(self._GET_ALL_TOOL_NAMES_USER, (cutoff, user_id))
        else:
            cursor = self._conn.execute(self._GET_ALL_TOOL_NAMES_ALL, (cutoff,))

        tool_counts: Counter = Counter()
        tools_by_session: Dict[str, set] = defaultdict(set)

        for row in cursor.fetchall():
            row = dict(row)
            # Source 1: tool_name column
            name = row.get("tool_name")
            if name:
                tool_counts[name] += 1
            # Source 2: tool_calls JSON
            calls = row.get("tool_calls")
            if calls:
                try:
                    if isinstance(calls, str):
                        calls = json.loads(calls)
                    if isinstance(calls, list):
                        for call in calls:
                            func = call.get("function", {}) if isinstance(call, dict) else {}
                            n = func.get("name")
                            if n:
                                tool_counts[n] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        distinct = len(tool_counts)
        counts_list = list(tool_counts.values())
        entropy = _shannon_entropy(counts_list)

        return {
            "distinct_tools": distinct,
            "tool_counts": dict(tool_counts),
            "entropy": round(entropy, 2),
            "total_calls": sum(counts_list),
        }

    # ── Signal 18: Session duration distribution ───────────────────────

    def _extract_session_duration(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Compute session duration distribution: longest, median, mean.

        Returns a dict with keys: ``durations`` (list of seconds),
        ``longest`` (float), ``longest_session_id`` (str or None),
        ``longest_session_date`` (str or None), ``median`` (float),
        ``mean`` (float), ``longest_topic`` (str or None).
        """
        durations: List[float] = []
        session_info: List[Tuple[float, Dict]] = []

        for s in sessions:
            start = s.get("started_at")
            end = s.get("ended_at")
            if start and end and end > start:
                dur = end - start
                durations.append(dur)
                session_info.append((dur, s))

        if not durations:
            return {
                "durations": [],
                "longest": 0,
                "longest_session_id": None,
                "longest_session_date": None,
                "median": 0,
                "mean": 0,
                "longest_topic": None,
            }

        durations.sort()
        longest_entry = max(session_info, key=lambda x: x[0]) if session_info else (0, {})
        longest_s = longest_entry[1] if session_info else {}
        n = len(durations)
        median = durations[n // 2] if n % 2 == 1 else (durations[n // 2 - 1] + durations[n // 2]) / 2
        mean = sum(durations) / n

        longest_date = None
        if longest_s.get("started_at"):
            try:
                longest_date = datetime.fromtimestamp(
                    float(longest_s["started_at"])
                ).strftime("%b %d")
            except (TypeError, ValueError, OSError):
                pass

        return {
            "durations": durations,
            "longest": longest_entry[0],
            "longest_session_id": longest_s.get("id", "")[:16] if longest_s else None,
            "longest_session_date": longest_date,
            "median": median,
            "mean": mean,
            "longest_topic": longest_s.get("title", "") if longest_s else None,
        }

    # ── Signal 19: Skill usage ─────────────────────────────────────────

    def _extract_skill_usage(self, assistant_messages: List[Dict]) -> Dict[str, Any]:
        """Extract skill usage patterns from skill_view and skill_manage tool calls.

        Parses the ``tool_calls`` JSON on assistant messages for calls to
        ``skill_view`` (loads) and ``skill_manage`` (edits — create/patch/
        edit/delete/write_file/remove_file actions).  Returns per-skill load
        and edit counts, totals, distinct skill count, and adoption rate
        (fraction of sessions that load at least one skill).

        Returns a dict with keys: ``total_loads`` (int), ``total_edits``
        (int), ``distinct_skills`` (int), ``load_counts`` (dict: skill →
        count), ``edit_counts`` (dict: skill → count), ``most_used`` (str or
        None), ``adoption_rate`` (float in [0, 1]), ``sessions_with_skills``
        (int).
        """
        load_counts: Counter = Counter()
        edit_counts: Counter = Counter()
        sessions_with_skills: set = set()
        total_sessions_seen: set = set()

        _EDIT_ACTIONS = {"create", "patch", "edit", "delete", "write_file", "remove_file"}

        for am in assistant_messages:
            sid = am.get("session_id")
            if sid:
                total_sessions_seen.add(sid)
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}

                if name == "skill_view":
                    skill_name = args.get("name", "")
                    if skill_name:
                        load_counts[skill_name] += 1
                        if sid:
                            sessions_with_skills.add(sid)
                elif name == "skill_manage":
                    action = args.get("action", "")
                    skill_name = args.get("name", "")
                    if skill_name and action in _EDIT_ACTIONS:
                        edit_counts[skill_name] += 1
                        if sid:
                            sessions_with_skills.add(sid)

        total_loads = sum(load_counts.values())
        total_edits = sum(edit_counts.values())
        distinct = len(set(load_counts.keys()) | set(edit_counts.keys()))
        most_used = load_counts.most_common(1)[0][0] if load_counts else None
        total_sessions = len(total_sessions_seen)
        adoption = (len(sessions_with_skills) / total_sessions) if total_sessions > 0 else 0.0

        return {
            "total_loads": total_loads,
            "total_edits": total_edits,
            "distinct_skills": distinct,
            "load_counts": dict(load_counts),
            "edit_counts": dict(edit_counts),
            "most_used": most_used,
            "adoption_rate": round(adoption, 3),
            "sessions_with_skills": len(sessions_with_skills),
        }

    # ── Signal 20: Memory management ──────────────────────────────────

    def _extract_memory_management(self, assistant_messages: List[Dict]) -> Dict[str, Any]:
        """Extract memory tool-call patterns (add/replace/remove).

        Looks for ``memory`` tool calls (the Hermes memory tool) whose
        ``action`` argument is ``add``, ``replace``, or ``remove``.  Also
        tracks the target of each operation (``memory`` vs ``user`` — the
        ``target`` argument on the memory tool).

        Returns a dict with keys: ``add_count`` (int), ``replace_count``
        (int), ``remove_count`` (int), ``total`` (int),
        ``correction_rate`` (float — (replaces + removes) / total),
        ``target_memory`` (int), ``target_user`` (int).
        """
        add_count = 0
        replace_count = 0
        remove_count = 0
        target_memory = 0
        target_user = 0

        for am in assistant_messages:
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                name = func.get("name", "")
                # The Hermes memory tool is registered as "memory" in the
                # tool schema.  Some variants may use "manage_memory".
                if name not in ("memory", "manage_memory"):
                    continue
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}

                action = args.get("action", "")
                if action == "add":
                    add_count += 1
                elif action == "replace":
                    replace_count += 1
                elif action == "remove":
                    remove_count += 1

                target = args.get("target", "memory")
                if target == "user":
                    target_user += 1
                else:
                    target_memory += 1

        total = add_count + replace_count + remove_count
        correction = ((replace_count + remove_count) / total) if total > 0 else 0.0

        return {
            "add_count": add_count,
            "replace_count": replace_count,
            "remove_count": remove_count,
            "total": total,
            "correction_rate": round(correction, 3),
            "target_memory": target_memory,
            "target_user": target_user,
        }

    # ── Signal 21: Cross-session memory ───────────────────────────────

    def _extract_cross_session_memory(
        self, assistant_messages: List[Dict], sessions: List[Dict]
    ) -> Dict[str, Any]:
        """Extract session_search usage patterns.

        Counts ``session_search`` tool calls, the number of distinct sessions
        that perform at least one search, and compares the average tool-call
        volume of sessions that search vs those that don't.

        Returns a dict with keys: ``total_searches`` (int),
        ``sessions_that_search`` (int), ``sessions_that_dont`` (int),
        ``avg_tool_calls_searching`` (float), ``avg_tool_calls_not_searching``
        (float).
        """
        searching_sessions: set = set()
        total_searches = 0

        # tool_call_count per session (from the sessions table)
        tc_by_session: Dict[str, int] = {
            s.get("id", ""): s.get("tool_call_count", 0) or 0 for s in sessions
        }
        all_session_ids = set(tc_by_session.keys())

        for am in assistant_messages:
            sid = am.get("session_id")
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                if func.get("name", "") == "session_search":
                    total_searches += 1
                    if sid:
                        searching_sessions.add(sid)

        non_searching = all_session_ids - searching_sessions

        tc_searching = [tc_by_session[s] for s in searching_sessions]
        tc_not = [tc_by_session[s] for s in non_searching]

        avg_searching = (sum(tc_searching) / len(tc_searching)) if tc_searching else 0.0
        avg_not = (sum(tc_not) / len(tc_not)) if tc_not else 0.0

        return {
            "total_searches": total_searches,
            "sessions_that_search": len(searching_sessions),
            "sessions_that_dont": len(non_searching),
            "avg_tool_calls_searching": round(avg_searching, 1),
            "avg_tool_calls_not_searching": round(avg_not, 1),
        }

    # ── Signal 22: Cron autonomy ──────────────────────────────────────

    def _extract_cron_autonomy(
        self,
        cutoff: float,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract autonomy signals from cron-sourced sessions.

        Queries ``sessions WHERE source = 'cron'`` within the time window and
        computes the end-reason distribution and success rate
        (``cron_complete`` / total cron sessions).

        Returns a dict with keys: ``cron_session_count`` (int),
        ``total_messages`` (int), ``end_reasons`` (dict: reason → count),
        ``success_rate`` (float in [0, 1]).
        """
        if source and source != "cron":
            return {
                "cron_session_count": 0,
                "total_messages": 0,
                "end_reasons": {},
                "success_rate": 0.0,
            }

        if user_id:
            cursor = self._conn.execute(
                "SELECT id, end_reason, message_count FROM sessions "
                "WHERE started_at >= ? AND source = 'cron' AND user_id = ?",
                (cutoff, user_id),
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, end_reason, message_count FROM sessions "
                "WHERE started_at >= ? AND source = 'cron'",
                (cutoff,),
            )
        rows = [dict(r) for r in cursor.fetchall()]

        end_reasons: Counter = Counter()
        total_messages = 0
        cron_complete = 0
        for row in rows:
            reason = row.get("end_reason") or "none"
            end_reasons[reason] += 1
            total_messages += row.get("message_count", 0) or 0
            if reason == "cron_complete":
                cron_complete += 1

        total = len(rows)
        success = (cron_complete / total) if total > 0 else 0.0

        return {
            "cron_session_count": total,
            "total_messages": total_messages,
            "end_reasons": dict(end_reasons),
            "success_rate": round(success, 3),
        }

    # ── Signal 23: Delegation patterns ────────────────────────────────

    def _extract_delegation_patterns(
        self, assistant_messages: List[Dict], sessions: List[Dict]
    ) -> Dict[str, Any]:
        """Extract delegation patterns from delegate_task tool calls.

        Beyond what :meth:`_extract_subagent_dispatch` captures, this signal
        estimates max concurrent delegation by looking at the overlap of
        subagent session time windows (sessions with a ``parent_session_id``
        that overlaps another child of the same parent).

        Returns a dict with keys: ``total_dispatches`` (int),
        ``background`` (int), ``foreground`` (int),
        ``max_concurrent`` (int).
        """
        total = 0
        background = 0
        foreground = 0

        for am in assistant_messages:
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                if func.get("name", "") != "delegate_task":
                    continue
                total += 1
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if isinstance(args, dict) and args.get("background"):
                    background += 1
                else:
                    foreground += 1

        # Estimate max concurrent from subagent session overlap
        subagent_sessions = [s for s in sessions if s.get("parent_session_id")]
        max_concurrent = self._compute_max_concurrent(subagent_sessions)

        return {
            "total_dispatches": total,
            "background": background,
            "foreground": foreground,
            "max_concurrent": max_concurrent,
        }

    # ── Signal 24: Task management ────────────────────────────────────

    def _extract_task_management(self, assistant_messages: List[Dict]) -> Dict[str, Any]:
        """Extract todo/task-management patterns from ``todo`` tool calls.

        Counts todo-list creations (action ``create`` / ``write`` with a
        non-empty ``todos`` list) and tracks how many sessions use the todo
        tool vs how many don't.  Completion rate is estimated from the ratio
        of ``completed`` statuses to total todo items in create/update calls
        when trackable.

        Returns a dict with keys: ``creation_count`` (int),
        ``sessions_with_todos`` (int), ``sessions_without_todos`` (int),
        ``completion_rate`` (float or None if not trackable).
        """
        creation_count = 0
        sessions_with_todos: set = set()
        total_items = 0
        completed_items = 0

        for am in assistant_messages:
            sid = am.get("session_id")
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                if func.get("name", "") != "todo":
                    continue
                if sid:
                    sessions_with_todos.add(sid)
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}

                todos = args.get("todos")
                action = args.get("action", "")
                if isinstance(todos, list) and todos:
                    if action in ("create", "write", None):
                        creation_count += 1
                    for item in todos:
                        if isinstance(item, dict):
                            total_items += 1
                            if item.get("status") == "completed":
                                completed_items += 1
                        elif isinstance(item, str):
                            total_items += 1

        completion_rate = (completed_items / total_items) if total_items > 0 else None

        return {
            "creation_count": creation_count,
            "sessions_with_todos": len(sessions_with_todos),
            "sessions_without_todos": 0,  # filled in by caller with total sessions
            "completion_rate": round(completion_rate, 3) if completion_rate is not None else None,
        }

    # ── Signal 25: Model effectiveness ────────────────────────────────

    def _extract_model_effectiveness(
        self, sessions: List[Dict], user_messages: List[Dict]
    ) -> Dict[str, Any]:
        """Cross-reference model with crash-out and steering signals.

        For each model, compute the crash-out rate (fraction of sessions with
        a crash-out message), steering rate (fraction of sessions with a
        steering keyword in a user message), session count, and average tool
        calls.

        Returns a dict with keys: ``models`` (list of dicts with ``model``,
        ``crash_out_rate``, ``steering_rate``, ``session_count``,
        ``avg_tool_calls``).
        """
        # Per-session model lookup
        model_by_session: Dict[str, str] = {}
        for s in sessions:
            sid = s.get("id", "")
            model = s.get("model") or "unknown"
            display = model.split("/")[-1] if "/" in model else model
            model_by_session[sid] = display

        # Per-session crash-out & steering detection
        sessions_with_crash: set = set()
        sessions_with_steering: set = set()
        for msg in user_messages:
            content = msg.get("content")
            if not content:
                continue
            sid = msg.get("session_id", "")
            if _is_all_caps(content) or _FRUSTRATION_KEYWORDS.search(content):
                sessions_with_crash.add(sid)
            if _STEERING_KEYWORDS.search(content):
                sessions_with_steering.add(sid)

        # Aggregate per model
        model_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"sessions": 0, "crash": 0, "steering": 0, "tool_calls": 0}
        )
        for s in sessions:
            sid = s.get("id", "")
            model = model_by_session.get(sid, "unknown")
            model_data[model]["sessions"] += 1
            model_data[model]["tool_calls"] += s.get("tool_call_count", 0) or 0
            if sid in sessions_with_crash:
                model_data[model]["crash"] += 1
            if sid in sessions_with_steering:
                model_data[model]["steering"] += 1

        models = []
        for model, data in sorted(model_data.items(), key=lambda kv: -kv[1]["sessions"]):
            sc = data["sessions"]
            models.append({
                "model": model,
                "session_count": sc,
                "crash_out_rate": round(data["crash"] / sc, 3) if sc > 0 else 0.0,
                "steering_rate": round(data["steering"] / sc, 3) if sc > 0 else 0.0,
                "avg_tool_calls": round(data["tool_calls"] / sc, 1) if sc > 0 else 0.0,
            })

        return {"models": models}

    # ── Signal 26: Skill ROI ──────────────────────────────────────────

    def _extract_skill_roi(
        self, sessions: List[Dict], assistant_messages: List[Dict]
    ) -> Dict[str, Any]:
        """Compare sessions that load skills vs those that don't.

        Partitions sessions into "skill sessions" (those containing at least
        one ``skill_view`` or ``skill_manage`` tool call) and "non-skill
        sessions", then compares average tool calls, average messages, and
        crash-out rate between the two groups.

        Returns a dict with keys: ``skill_session_count`` (int),
        ``non_skill_session_count`` (int), ``avg_tool_calls_with`` (float),
        ``avg_tool_calls_without`` (float), ``avg_messages_with`` (float),
        ``avg_messages_without`` (float), ``crash_out_rate_with`` (float),
        ``crash_out_rate_without`` (float), ``productivity_multiplier``
        (float — avg_tool_calls_with / avg_tool_calls_without, or 0 if
        either denominator is 0).
        """
        skill_sessions: set = set()
        for am in assistant_messages:
            calls = am.get("tool_calls")
            if not calls:
                continue
            try:
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                if func.get("name", "") in ("skill_view", "skill_manage"):
                    sid = am.get("session_id")
                    if sid:
                        skill_sessions.add(sid)
                    break

        with_tc = []
        with_msg = []
        without_tc = []
        without_msg = []
        for s in sessions:
            sid = s.get("id", "")
            tc = s.get("tool_call_count", 0) or 0
            mc = s.get("message_count", 0) or 0
            if sid in skill_sessions:
                with_tc.append(tc)
                with_msg.append(mc)
            else:
                without_tc.append(tc)
                without_msg.append(mc)

        avg_tc_with = (sum(with_tc) / len(with_tc)) if with_tc else 0.0
        avg_tc_without = (sum(without_tc) / len(without_tc)) if without_tc else 0.0
        avg_msg_with = (sum(with_msg) / len(with_msg)) if with_msg else 0.0
        avg_msg_without = (sum(without_msg) / len(without_msg)) if without_msg else 0.0

        # Crash-out rate: we don't have user_messages here, so use a
        # lightweight heuristic — sessions with high frustration in title.
        # For a precise rate, the caller can cross-reference; here we return
        # 0.0 placeholders that cards can refine.
        multiplier = (avg_tc_with / avg_tc_without) if avg_tc_without > 0 and avg_tc_with > 0 else 0.0

        return {
            "skill_session_count": len(skill_sessions),
            "non_skill_session_count": len(sessions) - len(skill_sessions),
            "avg_tool_calls_with": round(avg_tc_with, 1),
            "avg_tool_calls_without": round(avg_tc_without, 1),
            "avg_messages_with": round(avg_msg_with, 1),
            "avg_messages_without": round(avg_msg_without, 1),
            "crash_out_rate_with": 0.0,
            "crash_out_rate_without": 0.0,
            "productivity_multiplier": round(multiplier, 2),
        }

    # ── Signal 27: Session abandonment ────────────────────────────────

    def _extract_session_abandonment(
        self,
        cutoff: float,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute session abandonment statistics from ``end_reason``.

        Sessions with no ``end_reason`` (and no ``ended_at``) are considered
        abandoned.  Others are classified by their end reason.

        Returns a dict with keys: ``total_sessions`` (int),
        ``abandoned`` (int), ``closed_by_user`` (int),
        ``closed_by_agent`` (int), ``reset`` (int), ``other`` (int),
        ``abandonment_rate`` (float in [0, 1]).
        """
        clauses = ["started_at >= ?"]
        params: list = [cutoff]
        if source:
            clauses.append("source = ?")
            params.append(source)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where_sql = " AND ".join(clauses)
        cursor = self._conn.execute(
            f"SELECT end_reason FROM sessions WHERE {where_sql}",
            params,
        )

        _USER_REASONS = {"user_exit", "user_close", "timeout", "cancelled"}
        _AGENT_REASONS = {"agent_close", "cron_complete", "completed", "compression"}
        _RESET_REASONS = {"reset", "session_reset", "rewind"}

        abandoned = 0
        closed_by_user = 0
        closed_by_agent = 0
        reset = 0
        other = 0
        total = 0

        for row in cursor.fetchall():
            row = dict(row)
            total += 1
            reason = row.get("end_reason")
            if not reason:
                abandoned += 1
            elif reason in _USER_REASONS:
                closed_by_user += 1
            elif reason in _AGENT_REASONS:
                closed_by_agent += 1
            elif reason in _RESET_REASONS:
                reset += 1
            else:
                other += 1

        abandonment_rate = (abandoned / total) if total > 0 else 0.0

        return {
            "total_sessions": total,
            "abandoned": abandoned,
            "closed_by_user": closed_by_user,
            "closed_by_agent": closed_by_agent,
            "reset": reset,
            "other": other,
            "abandonment_rate": round(abandonment_rate, 3),
        }

    # =================================================================
    # Layer 2: LLM scoring + narrative cards
    # =================================================================

    def _score_and_narrate(
        self, signals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str], bool]:
        """Produce 5-axis scores and 4 LLM narrative cards.

        When the LLM is available, scores and narrative cards come from
        a single bounded LLM call.  When it fails, heuristic scores are
        computed from signals and narrative cards are replaced with
        signal-only fallback text.

        Args:
            signals: The full signals dict from :meth:`_extract_signals`.

        Returns:
            A tuple of (scores, llm_cards, llm_available).
        """
        # Try LLM scoring
        llm_result = self._call_llm_for_scoring(signals)
        if llm_result is not None:
            scores, llm_cards = llm_result
            return scores, llm_cards, True

        # Graceful degradation: heuristic scores + fallback cards
        scores = self._compute_heuristic_scores(signals)
        llm_cards = self._build_fallback_cards(signals, scores)
        return scores, llm_cards, False

    def _call_llm_for_scoring(
        self, signals: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, str]]]:
        """Make a single LLM call for scoring + narrative cards.

        Sends only bounded, credential-scrubbed excerpts: signal counts
        + top 5 go-to prompts (≤80 chars) + top 3 crash-out messages
        (≤200 chars).  No full transcripts leave the machine.

        Returns ``None`` if the LLM call fails (caller handles fallback).
        """
        try:
            from agent.auxiliary_client import call_llm
        except ImportError:
            logger.debug("auxiliary_client not available, using heuristic scores")
            return None

        # Read config: behavior.model (null = current model)
        model_override = None
        try:
            from hermes_cli.config import load_config
            config = load_config()
            behavior_config = config.get("behavior", {}) if isinstance(config, dict) else {}
            if isinstance(behavior_config, dict):
                model_override = behavior_config.get("model")
                if isinstance(model_override, str):
                    model_override = model_override.strip() or None
                else:
                    model_override = None
        except Exception as exc:
            logger.debug("Could not load config for behavior model: %s", exc)
            model_override = None

        # Build the bounded LLM prompt
        prompt_text = self._build_llm_prompt(signals)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a behavioral analyst for AI coding agent sessions. "
                    "Based on the following signals from a developer's Hermes Agent "
                    "sessions, produce 5 behavior scores (1-10) and 4 insight cards. "
                    "Be concise, direct, slightly witty."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        try:
            response = call_llm(
                task="behavior",
                model=model_override,  # type: ignore[arg-type]
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                timeout=60,
            )
        except Exception as exc:
            logger.warning("LLM call for behavioral scoring failed: %s", exc)
            return None

        if response is None:
            return None

        # Extract text from response
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Could not extract LLM response content: %s", exc)
            return None

        return self._parse_llm_response(content, signals)

    def _build_llm_prompt(self, signals: Dict[str, Any]) -> str:
        """Build the bounded LLM prompt from signals.

        Only sends aggregated counts + bounded excerpts (top 5 prompts
        ≤80 chars, top 3 crash-out messages ≤200 chars).  All excerpts
        are credential-scrubbed via :func:`_scrub_credentials`.
        """
        lines: List[str] = []
        lines.append(f"User signals (last {signals.get('days', 30)} days):")

        # Source breakdown
        sessions = signals.get("sessions", [])
        source_counts: Counter = Counter()
        for s in sessions:
            source_counts[s.get("source", "unknown")] += 1
        source_str = ", ".join(f"{s}: {c}" for s, c in source_counts.most_common())
        lines.append(f"- Total sessions: {signals['total_sessions']} ({source_str})")

        # Planning rate
        planning = signals.get("planning", {})
        lines.append(f"- Planning rate: {round(planning.get('rate', 0) * 100)}% (plan-phrase before first tool call)")

        # Steering rate
        steering = signals.get("steering", {})
        lines.append(f"- Steering rate: {round(steering.get('rate', 0) * 100)}% (redirect/correct keywords in user messages)")

        # Prompt length
        pl = signals.get("prompt_length", {})
        buckets = pl.get("buckets", {})
        lt10_pct = round((buckets.get("lt_10", 0) / pl.get("total", 1)) * 100) if pl.get("total") else 0
        lines.append(f"- Avg prompt length: {pl.get('average', 0):.0f} words ({lt10_pct}% under 10 words)")

        # Politeness
        politeness = signals.get("politeness", {})
        lines.append(f"- Politeness: {politeness.get('thank_count', 0)} thank-yous, {politeness.get('please_count', 0)} pleases")

        # Crash outs
        crash = signals.get("crash_outs", {})
        lines.append(f"- Crash outs: {crash.get('count', 0)} (ALL CAPS/exclamation messages)")
        for cm in crash.get("messages", [])[:3]:
            scrubbed = _scrub_credentials(cm.get("content", ""))[:200]
            lines.append(f"  - Biggest crash out: \"{scrubbed}\"")

        # Go-to prompts (top 5, ≤80 chars, scrubbed)
        go_to = signals.get("go_to_prompts", {})
        top_prompts = go_to.get("top", [])[:5]
        if top_prompts:
            prompt_strs = [
                f"\"{_scrub_credentials(p['prompt'])[:80]}\" ({p['count']}x)"
                for p in top_prompts
            ]
            lines.append(f"- Top go-to prompts: {', '.join(prompt_strs)}")

        # Agent parallelism
        ap = signals.get("agent_parallelism", {})
        lines.append(f"- Agent parallelism: max {ap.get('max_per_parent', 0)} subagents, avg {ap.get('avg_per_parent', 0)} per parent session")

        # Verification rate
        ver = signals.get("verification", {})
        lines.append(f"- Verification rate: {round(ver.get('rate', 0) * 100)}% (verified after declaring done)")

        # Error recovery
        er = signals.get("error_recovery", {})
        lines.append(f"- Error recovery: {round(er.get('recovery_rate', 0) * 100)}% self-recovered, {round((1 - er.get('recovery_rate', 0)) * 100)}% needed user intervention")

        # Tool calls per user message (rough)
        total_tool = signals.get("total_tool_messages", 0)
        total_user = signals.get("total_user_messages", 0)
        tpm = (total_tool / total_user) if total_user > 0 else 0
        lines.append(f"- Tool calls per user message: {tpm:.1f}")

        # Tool diversity
        td = signals.get("tool_diversity", {})
        lines.append(f"- Tool diversity: {td.get('distinct_tools', 0)} distinct tools, Shannon entropy {td.get('entropy', 0)}")

        # Decision patterns
        dp = signals.get("decision_patterns", {})
        lines.append(f"- Decision patterns: {dp.get('count', 0)} decisions extracted")

        # Subagent dispatch
        sd = signals.get("subagent_dispatch", {})
        lines.append(f"- Subagent dispatches: {sd.get('total', 0)} total, {sd.get('background', 0)} background, {sd.get('foreground', 0)} foreground")

        # Productivity
        pt = signals.get("productivity_timing", {})
        lines.append(f"- Productivity: {pt.get('peak_pct', 0)}% of sessions at peak hour ({pt.get('classification', 'unknown')})")

        # Model preference
        mp = signals.get("model_preference", {})
        model_strs = [
            f"{m['model']} {m['pct']}%"
            for m in mp.get("models", [])[:3]
        ]
        if model_strs:
            lines.append(f"- Model preference: {', '.join(model_strs)}")

        # ── Hermes-exclusive signals ──

        # Skill usage
        su = signals.get("skill_usage", {})
        if su.get("total_loads", 0) > 0:
            lines.append(
                f"- Skills: {su.get('total_loads', 0)} loads, "
                f"{su.get('total_edits', 0)} edits, "
                f"{su.get('distinct_skills', 0)} distinct, "
                f"{round(su.get('adoption_rate', 0) * 100)}% adoption"
            )
            if su.get("most_used"):
                lines.append(f"  - Most used: {su['most_used']}")

        # Memory management
        mm = signals.get("memory_management", {})
        if mm.get("total", 0) > 0:
            lines.append(
                f"- Memory ops: {mm.get('add_count', 0)} adds, "
                f"{mm.get('replace_count', 0)} replaces, "
                f"{mm.get('remove_count', 0)} removes, "
                f"{round(mm.get('correction_rate', 0) * 100)}% correction rate"
            )

        # Cross-session memory
        csm = signals.get("cross_session_memory", {})
        if csm.get("total_searches", 0) > 0:
            lines.append(
                f"- Session search: {csm.get('total_searches', 0)} searches, "
                f"{csm.get('sessions_that_search', 0)} sessions search, "
                f"avg {csm.get('avg_tool_calls_searching', 0)} vs "
                f"{csm.get('avg_tool_calls_not_searching', 0)} tool calls"
            )

        # Cron autonomy
        cron = signals.get("cron_autonomy", {})
        if cron.get("cron_session_count", 0) > 0:
            lines.append(
                f"- Cron: {cron.get('cron_session_count', 0)} sessions, "
                f"{round(cron.get('success_rate', 0) * 100)}% completion"
            )

        # Delegation patterns
        dp_sig = signals.get("delegation_patterns", {})
        if dp_sig.get("total_dispatches", 0) > 0:
            lines.append(
                f"- Delegation: {dp_sig.get('total_dispatches', 0)} dispatches, "
                f"{dp_sig.get('background', 0)} background, "
                f"max {dp_sig.get('max_concurrent', 0)} concurrent"
            )

        # Task management
        tm = signals.get("task_management", {})
        if tm.get("creation_count", 0) > 0:
            cr = tm.get("completion_rate")
            cr_str = f", {round(cr * 100)}% completion" if cr is not None else ""
            lines.append(
                f"- Todo: {tm.get('creation_count', 0)} lists created"
                f"{cr_str}"
            )

        # Skill ROI
        roi = signals.get("skill_roi", {})
        if roi.get("productivity_multiplier", 0) > 0:
            lines.append(
                f"- Skill ROI: {roi.get('productivity_multiplier', 0)}x productivity multiplier "
                f"({roi.get('avg_tool_calls_with', 0)} vs {roi.get('avg_tool_calls_without', 0)} tool calls)"
            )

        # Session abandonment
        sa = signals.get("session_abandonment", {})
        if sa.get("total_sessions", 0) > 0:
            lines.append(
                f"- Session abandonment: {round(sa.get('abandonment_rate', 0) * 100)}% abandoned, "
                f"{sa.get('abandoned', 0)} without closure"
            )

        lines.append("")
        lines.append("Produce:")
        lines.append("1. Five scores (1-10) with one-line rationale each:")
        lines.append("   - Execution Leverage: how much gets done per prompt")
        lines.append("   - Steering: how often you course-correct")
        lines.append("   - Engineering Quality: do you verify and test")
        lines.append("   - Product Thinking: do you plan features and prioritize")
        lines.append("   - Planning: do you plan before acting")
        lines.append("2. Archetype: [name] + 2-sentence rationale")
        lines.append("3. Agent relationship: [name] + 1-sentence evidence")
        lines.append("4. Growth edge: [recommendation targeting lowest-scoring axis]")
        lines.append("5. Biggest crash out: [context — what likely triggered it]")
        lines.append("")
        lines.append("Format each on separate lines. Be concise, direct, slightly witty.")

        return "\n".join(lines)

    def _parse_llm_response(
        self, content: str, signals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Parse the LLM response into scores and narrative cards.

        The LLM is asked to produce scores and cards on separate lines.
        This method uses regex to extract them.  If parsing partially
        fails, it fills gaps with heuristic values.

        Args:
            content: The raw LLM response text.
            signals: The signals dict (for heuristic fallbacks).

        Returns:
            A tuple of (scores, llm_cards).
        """
        # Extract scores with regex
        def _extract_score(axis_name: str) -> Tuple[Optional[int], str]:
            """Extract a 1-10 score and rationale for an axis."""
            # Try patterns like "Execution Leverage: 8/10 - rationale"
            # or "Execution Leverage: 8 - rationale"
            pattern = re.compile(
                rf"{re.escape(axis_name)}:?\s*(\d{{1,2}})\s*(?:/\s*10)?\s*[-—:]\s*(.+)",
                re.IGNORECASE,
            )
            for line in content.split("\n"):
                m = pattern.search(line)
                if m:
                    try:
                        score = int(m.group(1))
                        if 1 <= score <= 10:
                            return score, m.group(2).strip()
                    except ValueError:
                        continue
            return None, ""

        axes = [
            "Execution Leverage",
            "Steering",
            "Engineering Quality",
            "Product Thinking",
            "Planning",
        ]

        scores: Dict[str, Any] = {}
        for axis in axes:
            score, rationale = _extract_score(axis)
            key = axis.lower().replace(" ", "_")
            if score is None:
                # Fallback to heuristic
                h_scores = self._compute_heuristic_scores(signals)
                score = h_scores.get(key, {}).get("score", 5)
                rationale = h_scores.get(key, {}).get("rationale", "Heuristic estimate")
            scores[key] = {"score": score, "rationale": rationale}

        # Extract narrative cards
        def _extract_card(label: str) -> str:
            """Extract a card by label (e.g. 'Archetype')."""
            pattern = re.compile(
                rf"{re.escape(label)}:?\s*(.+)",
                re.IGNORECASE,
            )
            lines = content.split("\n")
            for i, line in enumerate(lines):
                m = pattern.search(line)
                if m:
                    text = m.group(1).strip()
                    # Grab continuation lines (non-blank, non-new-card)
                    for j in range(i + 1, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if not next_line:
                            break
                        # Stop if it looks like another card/score
                        if re.match(r"^(Execution|Steering|Engineering|Product|Planning|Archetype|Agent|Growth|Biggest)", next_line, re.IGNORECASE):
                            break
                        text += " " + next_line
                    return text
            return ""

        llm_cards: Dict[str, str] = {
            "archetype": _extract_card("Archetype") or "The Builder. Your behavior speaks for itself.",
            "agent_relationship": _extract_card("Agent relationship") or "A working partnership.",
            "growth_edge": _extract_card("Growth edge") or self._heuristic_growth_edge(scores),
            "crash_out": _extract_card("Biggest crash out") or self._heuristic_crash_out(signals),
        }

        return scores, llm_cards

    def _compute_heuristic_scores(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compute 5-axis scores from signals using predefined formulas.

        Used when the LLM is unavailable (graceful degradation).  Each
        formula maps signal values to a 1-10 score with a template
        rationale.

        Args:
            signals: The full signals dict.

        Returns:
            A dict with keys: ``execution_leverage``, ``steering``,
            ``engineering_quality``, ``product_thinking``,
            ``planning`` — each a dict with ``score`` and ``rationale``.
        """
        total_user = signals.get("total_user_messages", 1) or 1
        total_tool = signals.get("total_tool_messages", 0)
        total_sessions = signals.get("total_sessions", 1) or 1

        # Execution Leverage: tool calls per user message + tool diversity + parallelism
        tpm = total_tool / total_user if total_user > 0 else 0
        td = signals.get("tool_diversity", {})
        ap = signals.get("agent_parallelism", {})
        sd = signals.get("subagent_dispatch", {})
        el_raw = min(10, tpm / 2)  # 20 tool calls per msg = 10
        el_raw += min(2, td.get("distinct_tools", 0) / 10)
        el_raw += min(2, ap.get("max_per_parent", 0) / 3)
        el_raw += min(1, sd.get("total", 0) / 20)
        el_score = max(1, min(10, round(el_raw)))

        # Steering: steering rate + crash outs + decisions
        steering = signals.get("steering", {})
        crash = signals.get("crash_outs", {})
        decisions = signals.get("decision_patterns", {})
        st_raw = steering.get("rate", 0) * 10  # 100% → 10
        st_raw += min(2, crash.get("count", 0) / 5)
        st_raw += min(2, decisions.get("count", 0) / 10)
        st_score = max(1, min(10, round(st_raw)))

        # Engineering Quality: verification rate + error recovery + planning + tool diversity
        ver = signals.get("verification", {})
        er = signals.get("error_recovery", {})
        planning = signals.get("planning", {})
        eq_raw = ver.get("rate", 0) * 5  # 100% → 5
        eq_raw += er.get("recovery_rate", 0) * 3  # 100% → 3
        eq_raw += planning.get("rate", 0) * 1  # 100% → 1
        eq_raw += min(1, td.get("distinct_tools", 0) / 20)
        eq_score = max(1, min(10, round(eq_raw)))

        # Product Thinking: planning + topic diversity + decisions + prompt complexity
        topics = signals.get("session_topics", {})
        pt_raw = planning.get("rate", 0) * 4  # 100% → 4
        pt_raw += min(3, topics.get("diversity", 0) / 10)  # 30 distinct → 3
        pt_raw += min(2, decisions.get("count", 0) / 10)
        pl_data = signals.get("prompt_length", {})
        avg_len = pl_data.get("average", 0)
        pt_raw += min(1, avg_len / 50)  # 50+ word avg → +1
        pt_score = max(1, min(10, round(pt_raw)))

        # Planning: planning rate * 10
        pl_score = max(1, min(10, round(planning.get("rate", 0) * 10)))

        return {
            "execution_leverage": {
                "score": el_score,
                "rationale": f"{tpm:.1f} tool calls per prompt, {td.get('distinct_tools', 0)} distinct tools",
            },
            "steering": {
                "score": st_score,
                "rationale": f"{round(steering.get('rate', 0) * 100)}% steering rate, {crash.get('count', 0)} crash outs",
            },
            "engineering_quality": {
                "score": eq_score,
                "rationale": f"{round(ver.get('rate', 0) * 100)}% verification rate, {round(er.get('recovery_rate', 0) * 100)}% self-recovery",
            },
            "product_thinking": {
                "score": pt_score,
                "rationale": f"{round(planning.get('rate', 0) * 100)}% plan-first, {topics.get('diversity', 0)} distinct topics",
            },
            "planning": {
                "score": pl_score,
                "rationale": f"{round(planning.get('rate', 0) * 100)}% of sessions start with a plan",
            },
        }

    def _build_fallback_cards(
        self, signals: Dict[str, Any], scores: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build fallback narrative cards when LLM is unavailable.

        Args:
            signals: The full signals dict.
            scores: Heuristic scores dict.

        Returns:
            A dict with keys: ``archetype``, ``agent_relationship``,
            ``growth_edge``, ``crash_out``.
        """
        return {
            "archetype": self._heuristic_archetype(scores),
            "agent_relationship": self._heuristic_agent_relationship(signals),
            "growth_edge": self._heuristic_growth_edge(scores),
            "crash_out": self._heuristic_crash_out(signals),
        }

    def _heuristic_archetype(self, scores: Dict[str, Any]) -> str:
        """Generate a heuristic archetype label from scores."""
        el = scores.get("execution_leverage", {}).get("score", 5)
        st = scores.get("steering", {}).get("score", 5)
        pl = scores.get("planning", {}).get("score", 5)
        if el >= 7 and pl >= 6:
            return "The Orchestrator. You delegate heavily and plan before acting."
        if el >= 7 and st >= 6:
            return "The Director. You delegate but steer hard when things go off track."
        if st >= 7:
            return "The Micromanager. You course-correct often, keeping tight control."
        if pl >= 7:
            return "The Architect. You plan thoroughly before touching code."
        if el <= 3:
            return "The Hands-On Builder. You prefer to do it yourself, prompt by prompt."
        return "The Pragmatist. A balanced mix of delegation, steering, and planning."

    def _heuristic_agent_relationship(self, signals: Dict[str, Any]) -> str:
        """Generate a heuristic agent relationship description."""
        steering = signals.get("steering", {})
        politeness = signals.get("politeness", {})
        crash = signals.get("crash_outs", {})
        rate = steering.get("rate", 0)
        if rate > 0.4:
            return f"Like a junior dev you're training. You course-correct in {round(rate * 100)}% of prompts."
        if politeness.get("total", 0) > 5:
            return "Like a trusted colleague. You're polite and let it run."
        if crash.get("count", 0) > 2:
            return "Like a stubborn mule you're wrangling. There's some friction."
        return "Like a quiet partner. You give direction and let it work."

    def _heuristic_growth_edge(self, scores: Dict[str, Any]) -> str:
        """Generate a growth edge recommendation targeting the lowest axis."""
        axis_labels = {
            "execution_leverage": "Execution Leverage",
            "steering": "Steering",
            "engineering_quality": "Engineering Quality",
            "product_thinking": "Product Thinking",
            "planning": "Planning",
        }
        lowest_key = min(scores, key=lambda k: scores[k].get("score", 5))
        lowest_score = scores[lowest_key].get("score", 5)
        label = axis_labels.get(lowest_key, lowest_key)
        tips = {
            "execution_leverage": "Try delegating more with subagents and using more tools per prompt.",
            "steering": "Let the agent run longer before course-correcting.",
            "engineering_quality": "Add a verification step (run tests, check the file) before saying 'done.'",
            "product_thinking": "Spend more time planning features and prioritizing before diving in.",
            "planning": "Start sessions with a plan before your first tool call.",
        }
        tip = tips.get(lowest_key, "Focus on improving this area.")
        return f"Your {label} score is {lowest_score}/10. {tip}"

    def _heuristic_crash_out(self, signals: Dict[str, Any]) -> str:
        """Generate a heuristic crash-out card from signal data."""
        crash = signals.get("crash_outs", {})
        if crash.get("count", 0) == 0:
            return "No crash outs detected this period. You kept your cool."
        msgs = crash.get("messages", [])
        if msgs:
            top = msgs[0]
            return f"Your biggest crash out: \"{top.get('content', '')}\" (caps lock: {top.get('caps_pct', 0)}%)."
        return f"{crash.get('count', 0)} crash outs this period."

    # =================================================================
    # Deterministic insight cards (from signals, no LLM)
    # =================================================================

    def _build_deterministic_cards(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Build all 11 deterministic insight cards from signals.

        These cards require no LLM — they're computed directly from
        the extracted signal data.

        Returns:
            A dict with 11 card entries (each a dict with title + body).
        """
        cards: Dict[str, Any] = {}

        # Card 5: Prompt style
        cards["prompt_style"] = self._card_prompt_style(signals)

        # Card 6: Go-to prompts
        cards["go_to_prompts"] = self._card_go_to_prompts(signals)

        # Card 7: Politeness
        cards["politeness"] = self._card_politeness(signals)

        # Card 8: Crash out stats
        cards["crash_out_stats"] = self._card_crash_out_stats(signals)

        # Card 9: Model preference
        cards["model_preference"] = self._card_model_preference(signals)

        # Card 10: Productivity timing
        cards["productivity"] = self._card_productivity(signals)

        # Card 11: Shipping timing
        cards["shipping_timing"] = self._card_shipping_timing(signals)

        # Card 12: Agent parallelism
        cards["agent_parallelism"] = self._card_agent_parallelism(signals)

        # Card 13: Longest agent run
        cards["longest_run"] = self._card_longest_run(signals)

        # Card 14: Cryptic prompt
        cards["cryptic_prompt"] = self._card_cryptic_prompt(signals)

        # Card 15: Planning habits
        cards["planning_habits"] = self._card_planning_habits(signals)

        # ── Hermes-exclusive deep insight cards (16-23) ──

        # Card 16: Skill mastery
        cards["skill_mastery"] = self._card_skill_mastery(signals)

        # Card 17: Memory hygiene
        cards["memory_hygiene"] = self._card_memory_hygiene(signals)

        # Card 18: Autonomy level
        cards["autonomy_level"] = self._card_autonomy_level(signals)

        # Card 19: Cross-session memory
        cards["cross_session_memory"] = self._card_cross_session_memory(signals)

        # Card 20: Tool orchestration
        cards["tool_orchestration"] = self._card_tool_orchestration(signals)

        # Card 21: Model effectiveness
        cards["model_effectiveness"] = self._card_model_effectiveness(signals)

        # Card 22: Skill ROI
        cards["skill_roi"] = self._card_skill_roi(signals)

        # Card 23: Session abandonment
        cards["session_abandonment"] = self._card_session_abandonment(signals)

        return cards

    def _card_prompt_style(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Prompt style' card from prompt length distribution."""
        pl = signals.get("prompt_length", {})
        buckets = pl.get("buckets", {})
        total = pl.get("total", 1) or 1
        lt10 = buckets.get("lt_10", 0)
        lt10_pct = round((lt10 / total) * 100) if total > 0 else 0
        longest = pl.get("longest", 0)
        avg = pl.get("average", 0)

        if lt10_pct >= 60:
            title = "Straight to the point"
            body = f"{lt10_pct}% of your prompts are under 10 words. Longest: {longest} words."
        elif avg > 50:
            title = "Detailed and thorough"
            body = f"Average prompt: {avg:.0f} words. Longest: {longest} words. You give context."
        else:
            title = "Balanced communicator"
            body = f"{lt10_pct}% under 10 words, avg {avg:.0f} words. Longest: {longest} words."

        return {"title": title, "body": body}

    def _card_go_to_prompts(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Go-to prompts' card."""
        go_to = signals.get("go_to_prompts", {})
        top = go_to.get("top", [])
        if not top:
            return {"title": "No go-to prompts", "body": "Your prompts are all unique."}
        first = top[0]
        title = f'"{first["prompt"]}"'
        body = f"Used {first['count']} times across {first['sessions']} sessions."
        if len(top) > 1:
            others = ", ".join(f'"{p["prompt"]}" ({p["count"]}x)' for p in top[1:4])
            body += f" Also: {others}."
        return {"title": title, "body": body}

    def _card_politeness(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Politeness' card."""
        politeness = signals.get("politeness", {})
        thanks = politeness.get("thank_count", 0)
        pleases = politeness.get("please_count", 0)
        total = thanks + pleases

        if total == 0:
            return {"title": "All business", "body": "No thank-yous or pleases detected. The robots won't spare you."}
        if total <= 5:
            return {"title": "You're civil", "body": f"{thanks} thank-yous, {pleases} pleases. The robots won't spare you."}
        if total <= 20:
            return {"title": "Polite and professional", "body": f"{thanks} thank-yous, {pleases} pleases. Respectful."}
        return {"title": "Exceedingly polite", "body": f"{thanks} thank-yous, {pleases} pleases. Your agents love you."}

    def _card_crash_out_stats(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Crash out stats' card."""
        crash = signals.get("crash_outs", {})
        count = crash.get("count", 0)
        if count == 0:
            return {"title": "No crash outs", "body": "You kept your cool this period."}
        msgs = crash.get("messages", [])
        if msgs:
            top = msgs[0]
            content = top.get("content", "")
            caps = top.get("caps_pct", 0)
            return {
                "title": f"{count} crash out{'s' if count != 1 else ''} this period",
                "body": f'Worst: "{content}" (caps lock: {caps}%).',
            }
        return {"title": f"{count} crash outs", "body": "Frustration detected but no standout message."}

    def _card_model_preference(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Model preference' card."""
        mp = signals.get("model_preference", {})
        models = mp.get("models", [])
        if not models:
            return {"title": "No model data", "body": "No model preference detected."}
        top = models[0]
        title = f"{top['model']} loyalist" if top["pct"] >= 60 else f"{top['model']} user"
        body_parts = [f"You reach for {top['model']} in {top['pct']}% of sessions"]
        if len(models) > 1:
            others = ", ".join(f"{m['model']} in {m['pct']}%" for m in models[1:4])
            body_parts.append(others)
        body = ".\n".join(body_parts) + "." if len(body_parts) > 1 else body_parts[0] + "."
        return {"title": title, "body": body}

    def _card_productivity(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Productivity timing' card."""
        pt = signals.get("productivity_timing", {})
        classification = pt.get("classification", "unknown")
        peak_hour = pt.get("peak_hour")
        peak_pct = pt.get("peak_pct", 0)

        labels = {
            "night_owl": "Night owl",
            "early_bird": "Early bird",
            "afternoon_grinder": "Afternoon grinder",
            "evening": "Evening worker",
        }
        title = labels.get(classification, "Steady worker")

        if peak_hour is not None:
            ampm = "AM" if peak_hour < 12 else "PM"
            display_hr = peak_hour % 12 or 12
            body = f"{peak_pct}% of your sessions peak around {display_hr} {ampm}."
        else:
            body = "No clear peak detected."

        return {"title": title, "body": body}

    def _card_shipping_timing(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Shipping timing' card."""
        st = signals.get("shipping_timing", {})
        peak_day = st.get("peak_day")
        peak_count = st.get("peak_count", 0)
        if not peak_day:
            return {"title": "No shipping pattern", "body": "No clear peak day detected."}
        plural = peak_day.rstrip("day") + "days" if peak_day.endswith("day") else peak_day + "s"
        title = f"{peak_day}s"
        body = f"Your single biggest push of the period landed on a {peak_day} ({peak_count} sessions)."
        return {"title": title, "body": body}

    def _card_agent_parallelism(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Agent parallelism' card."""
        ap = signals.get("agent_parallelism", {})
        max_pp = ap.get("max_per_parent", 0)
        avg_pp = ap.get("avg_per_parent", 0)
        max_conc = ap.get("max_concurrent", 0)

        if max_pp == 0:
            return {"title": "Solo operator", "body": "No subagent dispatches detected. You work alone."}

        if max_pp >= 5:
            title = "Power user"
        elif max_pp >= 3:
            title = "Parallel delegator"
        else:
            title = "Team player"

        body = f"You've run up to {max_pp} subagents in parallel"
        if max_conc > max_pp:
            body += f", {max_conc} concurrent sessions"
        body += f". Average: {avg_pp} subagents per parent session."
        return {"title": title, "body": body}

    def _card_longest_run(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Longest agent run' card."""
        sd = signals.get("session_duration", {})
        longest = sd.get("longest", 0)
        if longest <= 0:
            return {"title": "No duration data", "body": "No ended sessions with duration found."}

        dur_str = _format_duration(longest)
        sid = sd.get("longest_session_id", "")
        date = sd.get("longest_session_date", "")
        topic = sd.get("longest_topic", "")

        title = f"{dur_str}"
        body_parts = [f"Your longest agent run"]
        if topic:
            body_parts.append(f"on \"{topic}\"")
        body_parts.append(f"before you stepped in")
        if sid:
            suffix = f" (Session: {sid}"
            if date:
                suffix += f", {date}"
            suffix += ")"
            body_parts.append(suffix)
        body = " ".join(body_parts) + "."
        return {"title": title, "body": body}

    def _card_cryptic_prompt(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Cryptic prompt' card."""
        cryptic = signals.get("cryptic_prompts", {})
        prompts = cryptic.get("prompts", [])
        if not prompts:
            return {"title": "No cryptic prompts", "body": "Your late-night prompts are coherent. Impressive."}
        top = prompts[0]
        content = top.get("content", "")
        hour = top.get("hour", 0)
        ampm = "AM" if hour < 12 else "PM"
        display_hr = hour % 12 or 12
        typo_note = "with typos" if top.get("has_typos") else "surprisingly clean"
        return {
            "title": f'"{content}"',
            "body": f"Sent at {display_hr} {ampm} with zero context. {typo_note}. The agent figured it out.",
        }

    def _card_planning_habits(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Build the 'Planning habits' card."""
        planning = signals.get("planning", {})
        rate = planning.get("rate", 0)
        pct = round(rate * 100)

        if pct >= 70:
            title = f"{pct}% plan-first"
            body = "You open with a plan in most sessions. Methodical."
        elif pct >= 40:
            title = f"{pct}% plan-first"
            body = "You open with a plan in some sessions, skipping it for quick fixes."
        elif pct >= 10:
            title = f"{pct}% plan-first"
            body = "You rarely plan first. You'd rather dive in and course-correct."
        else:
            title = "Jump-right-in"
            body = "Almost no planning detected. You prefer action over strategy."
        return {"title": title, "body": body}

    # ── Hermes-exclusive deep insight cards (16-23) ───────────────────

    def _card_skill_mastery(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 16: Skill mastery — cross-references skill usage + skill ROI."""
        su = signals.get("skill_usage", {})
        roi = signals.get("skill_roi", {})
        total_loads = su.get("total_loads", 0)
        distinct = su.get("distinct_skills", 0)
        most_used = su.get("most_used")
        adoption = su.get("adoption_rate", 0)
        avg_with = roi.get("avg_tool_calls_with", 0)
        avg_without = roi.get("avg_tool_calls_without", 0)
        multiplier = roi.get("productivity_multiplier", 0)

        if total_loads == 0:
            return {"title": "No skills loaded", "body": "You haven't loaded any skills this period. Skills are your highest-leverage habit."}

        title = f"{distinct} skills loaded"
        parts = []
        if most_used:
            load_count = su.get("load_counts", {}).get(most_used, 0)
            parts.append(f"{most_used} is your most-used ({load_count}x)")
        if multiplier > 0 and avg_without > 0:
            parts.append(f"Sessions with skills average {avg_with:.0f} tool calls vs {avg_without:.0f} without — {multiplier}x more productive")
        adoption_pct = round(adoption * 100)
        parts.append(f"But you only load skills in {adoption_pct}% of sessions")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_memory_hygiene(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 17: Memory hygiene — from memory management signal."""
        mm = signals.get("memory_management", {})
        total = mm.get("total", 0)
        if total == 0:
            return {"title": "No memory operations", "body": "You haven't written to memory this period. Consider storing reusable context."}
        adds = mm.get("add_count", 0)
        replaces = mm.get("replace_count", 0)
        removes = mm.get("remove_count", 0)
        correction = mm.get("correction_rate", 0)
        correction_pct = round(correction * 100)

        title = f"{total} memory operations"
        body_parts = [f"{replaces} replaces, {adds} adds, {removes} removes"]
        if correction_pct >= 30:
            body_parts.append(f"{correction_pct}% correction rate. You write fast and fix later. Consider planning entries before writing to reduce churn")
        else:
            body_parts.append(f"{correction_pct}% correction rate. Clean memory hygiene")
        body = ". ".join(body_parts) + "."
        return {"title": title, "body": body}

    def _card_autonomy_level(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 18: Autonomy level — cron sessions + delegation + subagent volume."""
        cron = signals.get("cron_autonomy", {})
        delegation = signals.get("delegation_patterns", {})
        ap = signals.get("agent_parallelism", {})
        total_sessions = signals.get("total_sessions", 1) or 1

        cron_count = cron.get("cron_session_count", 0)
        cron_success = cron.get("success_rate", 0)
        dispatches = delegation.get("total_dispatches", 0)
        max_conc = delegation.get("max_concurrent", 0)
        subagents = ap.get("total_subagents", 0)

        if cron_count == 0 and dispatches == 0 and subagents == 0:
            return {"title": "Manual operator", "body": "No cron sessions or subagent dispatches. You drive every session yourself."}

        automated = cron_count + subagents
        auto_pct = round((automated / total_sessions) * 100) if total_sessions > 0 else 0

        title = f"{auto_pct}% automated"
        parts = []
        if cron_count > 0:
            parts.append(f"{cron_count} cron sessions ran autonomously ({round(cron_success * 100)}% completion)")
        if dispatches > 0:
            parts.append(f"{dispatches} subagent dispatches, max {max_conc} concurrent")
        if automated > 0:
            parts.append(f"Your agency works while you don't — {auto_pct}% of your session volume is automated")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_cross_session_memory(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 19: Cross-session memory — session_search usage."""
        csm = signals.get("cross_session_memory", {})
        total_searches = csm.get("total_searches", 0)
        if total_searches == 0:
            return {"title": "No history searches", "body": "You don't search past sessions. You may be missing context from previous work."}
        sessions_search = csm.get("sessions_that_search", 0)
        total_sessions = signals.get("total_sessions", 1) or 1
        search_pct = round((sessions_search / total_sessions) * 100) if total_sessions > 0 else 0
        avg_searching = csm.get("avg_tool_calls_searching", 0)
        avg_not = csm.get("avg_tool_calls_not_searching", 0)

        title = f"{total_searches} history searches"
        parts = [f"across {sessions_search} sessions ({search_pct}%)"]
        if avg_searching > 0 and avg_not > 0:
            parts.append(f"Sessions that search average {avg_searching:.0f} tool calls vs {avg_not:.0f} that don't — you do your most serious work when pulling context from past conversations")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_tool_orchestration(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 20: Tool orchestration — tool diversity + task management."""
        td = signals.get("tool_diversity", {})
        tm = signals.get("task_management", {})
        distinct = td.get("distinct_tools", 0)
        todo_count = tm.get("creation_count", 0)
        sessions_with_todos = tm.get("sessions_with_todos", 0)

        if distinct == 0:
            return {"title": "Single-tool user", "body": "You're using one tool. Expand your toolkit for more leverage."}

        title = f"{distinct} distinct tools"
        parts = [f"max {distinct} in one session"]
        if todo_count > 0:
            parts.append(f"{todo_count} todo lists created")
        parts.append("You orchestrate across browser, terminal, file, web, memory, skills, and cron — not just a prompter")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_model_effectiveness(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 21: Model effectiveness — per-model crash-out and steering rates."""
        me = signals.get("model_effectiveness", {})
        models = me.get("models", [])
        if not models or len(models) == 0:
            return {"title": "No model data", "body": "No model effectiveness data available."}

        # Find the best and worst models by crash-out rate
        sorted_by_crash = sorted(models, key=lambda m: m.get("crash_out_rate", 0))
        best = sorted_by_crash[0]
        worst = sorted_by_crash[-1]

        if best.get("crash_out_rate", 0) == worst.get("crash_out_rate", 0) and len(models) == 1:
            # Only one model
            m = models[0]
            title = f"{m['model']}: {round(m.get('crash_out_rate', 0) * 100)}% crash-out"
            body = f"{round(m.get('steering_rate', 0) * 100)}% steering across {m.get('session_count', 0)} sessions."
            return {"title": title, "body": body}

        title = "Model comparison"
        parts = []
        parts.append(
            f"{best['model']}: {round(best.get('crash_out_rate', 0) * 100)}% crash-out rate, "
            f"{round(best.get('steering_rate', 0) * 100)}% steering"
        )
        parts.append(
            f"{worst['model']}: {round(worst.get('crash_out_rate', 0) * 100)}% crash-out, "
            f"{round(worst.get('steering_rate', 0) * 100)}% steering"
        )
        if best.get("crash_out_rate", 0) < worst.get("crash_out_rate", 0):
            parts.append(f"{best['model']} follows your instructions better — consider it for complex tasks")
        if worst.get("crash_out_rate", 0) > 0.3:
            parts.append(f"{worst['model']} is making you lose patience 1 in {max(1, round(1 / worst.get('crash_out_rate', 1)))} sessions")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_skill_roi(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 22: Skill ROI — productivity multiplier + adoption gap."""
        roi = signals.get("skill_roi", {})
        su = signals.get("skill_usage", {})
        multiplier = roi.get("productivity_multiplier", 0)
        avg_with = roi.get("avg_tool_calls_with", 0)
        avg_without = roi.get("avg_tool_calls_without", 0)
        non_skill = roi.get("non_skill_session_count", 0)
        total_sessions = signals.get("total_sessions", 1) or 1
        skip_pct = round((non_skill / total_sessions) * 100) if total_sessions > 0 else 0

        if multiplier <= 0 or avg_without <= 0:
            if su.get("total_loads", 0) == 0:
                return {"title": "No skill usage", "body": "Loading a skill before starting work is your highest-leverage habit."}
            return {"title": "Skills in use", "body": f"{su.get('distinct_skills', 0)} skills loaded across {roi.get('skill_session_count', 0)} sessions."}

        title = f"{multiplier}x more productive with skills"
        parts = [f"Sessions that load skills are {multiplier}x more productive ({avg_with:.0f} vs {avg_without:.0f} tool calls)"]
        if skip_pct > 0:
            parts.append(f"But {skip_pct}% of your sessions skip skills entirely")
        parts.append("Loading a skill before starting work is your highest-leverage habit")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    def _card_session_abandonment(self, signals: Dict[str, Any]) -> Dict[str, str]:
        """Card 23: Session abandonment — from end_reason distribution."""
        sa = signals.get("session_abandonment", {})
        total = sa.get("total_sessions", 0)
        if total == 0:
            return {"title": "No session data", "body": "No sessions to analyze for abandonment patterns."}
        abandoned = sa.get("abandoned", 0)
        closed_user = sa.get("closed_by_user", 0)
        closed_agent = sa.get("closed_by_agent", 0)
        rate = sa.get("abandonment_rate", 0)
        rate_pct = round(rate * 100)

        if abandoned == 0:
            return {"title": "Clean closures", "body": f"All {total} sessions ended properly. You finish what you start."}

        title = f"{rate_pct}% end without closure"
        parts = [f"{abandoned} abandoned vs {closed_user + closed_agent} closed"]
        parts.append("You start more than you finish")
        parts.append("Consider scoping sessions tighter or using /title to track what's still open")
        body = ". ".join(parts) + "."
        return {"title": title, "body": body}

    # =================================================================
    # Layer 3: Score persistence (SQLite, no LLM)
    # =================================================================

    def _persist_scores(
        self,
        days: int,
        source: Optional[str],
        user_id: Optional[str],
        scores: Dict[str, Any],
        llm_cards: Dict[str, str],
        signals: Dict[str, Any],
    ) -> None:
        """Store scores in the ``behavioral_scores`` table for trend tracking.

        Args:
            days: The days window used for this run.
            source: The source filter (or ``None``).
            user_id: The user ID filter (or ``None``).  Ensures two
                users' scores on a multi-user gateway don't overwrite
                each other.
            scores: The 5-axis scores dict.
            llm_cards: The LLM narrative cards dict.
            signals: The raw signals dict (stored as JSON for audit).
        """
        try:
            # Prepare raw signals JSON (strip the large sessions list to keep
            # the blob manageable — only store the signal values, not raw rows)
            signals_for_storage = {
                k: v for k, v in signals.items()
                if k != "sessions"  # skip the raw session list
            }
            raw_signals_json = json.dumps(signals_for_storage, default=str)

            self._conn.execute(
                """INSERT INTO behavioral_scores
                   (run_timestamp, days_window, source_filter, user_id,
                    execution_leverage, steering, engineering_quality,
                    product_thinking, planning,
                    archetype, agent_relationship, growth_edge, raw_signals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    days,
                    source,
                    user_id,
                    scores.get("execution_leverage", {}).get("score", 5),
                    scores.get("steering", {}).get("score", 5),
                    scores.get("engineering_quality", {}).get("score", 5),
                    scores.get("product_thinking", {}).get("score", 5),
                    scores.get("planning", {}).get("score", 5),
                    llm_cards.get("archetype", ""),
                    llm_cards.get("agent_relationship", ""),
                    llm_cards.get("growth_edge", ""),
                    raw_signals_json,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("Could not persist behavioral scores: %s", exc)

    # =================================================================
    # Formatting
    # =================================================================

    def format_terminal(self, report: Dict[str, Any]) -> str:
        """Format the behavioral profile for terminal display (CLI).

        Renders the full report with box-drawing header, bar-chart
        scores, and all 15 insight cards.

        Args:
            report: The report dict from :meth:`generate`.

        Returns:
            A formatted string for terminal output.
        """
        if report.get("empty"):
            days = report.get("days", 30)
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            return f"  Not enough data for behavioral analysis in the last {days} days{src}.\n  Run some agent sessions first!"

        lines: List[str] = []
        days = report["days"]
        src_filter = report.get("source_filter")
        session_count = report.get("session_count", 0)
        scores = report.get("scores", {})
        cards = report.get("cards", {})
        llm_available = report.get("llm_available", False)

        # Header
        lines.append("")
        lines.append("  ╔══════════════════════════════════════════════════════════╗")
        lines.append("  ║                   🧭 Hermes Behavior                     ║")
        period_label = f"Last {days} days · {session_count} sessions"
        if src_filter:
            period_label += f" ({src_filter})"
        padding = 58 - len(period_label) - 2
        left_pad = padding // 2
        right_pad = padding - left_pad
        lines.append(f"  ║{' ' * left_pad} {period_label} {' ' * right_pad}║")
        lines.append("  ╚══════════════════════════════════════════════════════════╝")
        lines.append("")

        # Scores section
        lines.append("  📊 Behavioral Scores")
        lines.append("  " + "─" * 56)

        axis_labels = [
            ("execution_leverage", "Execution Leverage"),
            ("steering", "Steering"),
            ("engineering_quality", "Engineering Quality"),
            ("product_thinking", "Product Thinking"),
            ("planning", "Planning"),
        ]

        for key, label in axis_labels:
            score_data = scores.get(key, {})
            score = score_data.get("score", 0)
            rationale = score_data.get("rationale", "")
            bar = _bar(score)
            lines.append(f"  {label:<20} {bar}  {score}/10  {rationale}")

        if not llm_available:
            lines.append("  (Heuristic scores — LLM was unavailable)")
        lines.append("")

        # Insight cards section
        lines.append("  🎭 Insight Cards")
        lines.append("  " + "─" * 56)

        # Card 1: Archetype (LLM)
        archetype = cards.get("archetype", "")
        if archetype:
            lines.append(f"  Archetype: {self._first_clause(archetype)}")
            lines.append(f"    {self._rest_clauses(archetype)}")
            lines.append("")

        # Card 2: Agent relationship (LLM)
        agent_rel = cards.get("agent_relationship", "")
        if agent_rel:
            lines.append(f"  Agent relationship: {self._first_clause(agent_rel)}")
            lines.append(f"    {self._rest_clauses(agent_rel)}")
            lines.append("")

        # Card 5: Prompt style
        self._render_card_terminal(lines, "Prompt style", cards.get("prompt_style", {}))

        # Card 6: Go-to prompts
        self._render_card_terminal(lines, "Go-to prompt", cards.get("go_to_prompts", {}))

        # Card 7: Politeness
        self._render_card_terminal(lines, "Politeness", cards.get("politeness", {}))

        # Card 8: Crash out stats
        self._render_card_terminal(lines, "Crash outs", cards.get("crash_out_stats", {}))

        # Card 9: Model preference
        self._render_card_terminal(lines, "Model preference", cards.get("model_preference", {}))

        # Card 10: Productivity
        self._render_card_terminal(lines, "Productivity", cards.get("productivity", {}))

        # Card 11: Shipping timing
        self._render_card_terminal(lines, "Shipping timing", cards.get("shipping_timing", {}))

        # Card 12: Agent parallelism
        self._render_card_terminal(lines, "Agent parallelism", cards.get("agent_parallelism", {}))

        # Card 13: Longest agent run
        self._render_card_terminal(lines, "Longest agent run", cards.get("longest_run", {}))

        # Card 14: Cryptic prompt
        self._render_card_terminal(lines, "Cryptic prompt of the period", cards.get("cryptic_prompt", {}))

        # Card 15: Planning habits
        self._render_card_terminal(lines, "Planning", cards.get("planning_habits", {}))

        # ── Hermes-exclusive deep insight cards (16-23) ──
        self._render_card_terminal(lines, "Skill mastery", cards.get("skill_mastery", {}))
        self._render_card_terminal(lines, "Memory hygiene", cards.get("memory_hygiene", {}))
        self._render_card_terminal(lines, "Autonomy level", cards.get("autonomy_level", {}))
        self._render_card_terminal(lines, "Cross-session memory", cards.get("cross_session_memory", {}))
        self._render_card_terminal(lines, "Tool orchestration", cards.get("tool_orchestration", {}))
        self._render_card_terminal(lines, "Model effectiveness", cards.get("model_effectiveness", {}))
        self._render_card_terminal(lines, "Skill ROI", cards.get("skill_roi", {}))
        self._render_card_terminal(lines, "Session abandonment", cards.get("session_abandonment", {}))

        # Card 3: Growth edge (LLM) — at the end, after seeing all scores
        growth = cards.get("growth_edge", "")
        if growth:
            lines.append(f"  Growth edge: {self._first_clause(growth)}")
            lines.append(f"    {self._rest_clauses(growth)}")
            lines.append("")

        # Card 4: Biggest crash out (LLM) — context
        crash_llm = cards.get("crash_out", "")
        if crash_llm:
            lines.append(f"  Crash out context: {self._first_clause(crash_llm)}")
            lines.append(f"    {self._rest_clauses(crash_llm)}")
            lines.append("")

        return "\n".join(lines)

    def format_gateway(self, report: Dict[str, Any]) -> str:
        """Format the behavioral profile for gateway/messaging (Telegram/Discord).

        Compact markdown format: scores inline, all 15 cards, bold labels.

        Args:
            report: The report dict from :meth:`generate`.

        Returns:
            A formatted markdown string for gateway output.
        """
        if report.get("empty"):
            days = report.get("days", 30)
            return f"Not enough data for behavioral analysis in the last {days} days."

        lines: List[str] = []
        days = report["days"]
        src_filter = report.get("source_filter")
        session_count = report.get("session_count", 0)
        scores = report.get("scores", {})
        cards = report.get("cards", {})
        llm_available = report.get("llm_available", False)

        header = f"🧭 **Hermes Behavior** — Last {days} days"
        if src_filter:
            header += f" ({src_filter})"
        header += f" · {session_count} sessions"
        lines.append(header)
        lines.append("")

        # Scores inline
        lines.append("**📊 Behavioral Scores:**")
        axis_labels = [
            ("execution_leverage", "Execution Leverage"),
            ("steering", "Steering"),
            ("engineering_quality", "Engineering Quality"),
            ("product_thinking", "Product Thinking"),
            ("planning", "Planning"),
        ]
        for key, label in axis_labels:
            score_data = scores.get(key, {})
            score = score_data.get("score", 0)
            rationale = score_data.get("rationale", "")
            lines.append(f"  {label}: {score}/10 — {rationale}")
        if not llm_available:
            lines.append("  _(Heuristic scores — LLM unavailable)_")
        lines.append("")

        # Insight cards
        lines.append("**🎭 Insight Cards:**")
        lines.append("")

        # LLM cards
        archetype = cards.get("archetype", "")
        if archetype:
            lines.append(f"**Archetype:** {archetype}")
        agent_rel = cards.get("agent_relationship", "")
        if agent_rel:
            lines.append(f"**Agent relationship:** {agent_rel}")

        # Deterministic cards
        for label, card_key in [
            ("Prompt style", "prompt_style"),
            ("Go-to prompt", "go_to_prompts"),
            ("Politeness", "politeness"),
            ("Crash outs", "crash_out_stats"),
            ("Model preference", "model_preference"),
            ("Productivity", "productivity"),
            ("Shipping timing", "shipping_timing"),
            ("Agent parallelism", "agent_parallelism"),
            ("Longest agent run", "longest_run"),
            ("Cryptic prompt", "cryptic_prompt"),
            ("Planning", "planning_habits"),
            # Hermes-exclusive deep insight cards (16-23)
            ("Skill mastery", "skill_mastery"),
            ("Memory hygiene", "memory_hygiene"),
            ("Autonomy level", "autonomy_level"),
            ("Cross-session memory", "cross_session_memory"),
            ("Tool orchestration", "tool_orchestration"),
            ("Model effectiveness", "model_effectiveness"),
            ("Skill ROI", "skill_roi"),
            ("Session abandonment", "session_abandonment"),
        ]:
            card = cards.get(card_key, {})
            title = card.get("title", "")
            body = card.get("body", "")
            lines.append(f"**{label}:** {title}")
            if body:
                lines.append(f"  {body}")

        # Growth edge + crash out context (LLM)
        lines.append("")
        growth = cards.get("growth_edge", "")
        if growth:
            lines.append(f"**Growth edge:** {growth}")
        crash_llm = cards.get("crash_out", "")
        if crash_llm:
            lines.append(f"**Crash out context:** {crash_llm}")

        return "\n".join(lines)

    # ── Formatting helpers ─────────────────────────────────────────────

    @staticmethod
    def _render_card_terminal(
        lines: List[str], label: str, card: Dict[str, str]
    ) -> None:
        """Render a single deterministic card to terminal lines.

        Format::
            Label: Title
              Body text

        Args:
            lines: The output list to append to.
            label: The card label (e.g. "Prompt style").
            card: The card dict with ``title`` and ``body`` keys.
        """
        title = card.get("title", "")
        body = card.get("body", "")
        lines.append(f"  {label}: {title}")
        if body:
            lines.append(f"    {body}")
        lines.append("")

    @staticmethod
    def _first_clause(text: str) -> str:
        """Extract the first sentence/clause from a narrative card text."""
        if not text:
            return ""
        # Split on period or newline, take first part
        for sep in [". ", ".\n", "\n"]:
            idx = text.find(sep)
            if idx > 0:
                return text[:idx].strip()
        return text.strip()

    @staticmethod
    def _rest_clauses(text: str) -> str:
        """Extract all clauses after the first from a narrative card text."""
        if not text:
            return ""
        # Find the first separator and return the rest
        for sep in [". ", ".\n", "\n"]:
            idx = text.find(sep)
            if idx > 0:
                rest = text[idx + len(sep):].strip()
                if rest:
                    return rest
        return ""