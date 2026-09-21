"""
Feedback loop (P2-10 — quality signal capture from chat surface).

Captures Sahil's explicit and implicit feedback from Discord/Telegram
messages and converts it into structured quality signals for the
eval harness (P2-2), memory hygiene (P2-8), and the review gate
(P2-1).

Signal types
------------
- **approve** — explicit approval ("looks good", "approved", thumbs-up)
- **reject** — explicit rejection ("no", "redo this", "wrong")
- **correct** — correction with replacement ("actually X not Y")
- **clarify** — follow-up question indicating ambiguity ("what about X?")
- **implicit_skip** — no response to a decision for N hours (negative signal)

Integration points
------------------
1. Eval harness — feedback events become soft labels for golden tasks
2. Memory hygiene — corrections trigger contradiction detection
3. Review gate — agent quality score adjusts review frequency
4. Kanban — reject signals trigger task reopen or reassign

Usage
-----
    from hermes_cli.feedback_loop import FeedbackLoop, SignalType

    loop = FeedbackLoop(store_path="~/.hermes/kensei/feedback.db")
    signal = loop.classify_message("that's wrong — the server runs on port 8080")
    if signal.type == SignalType.CORRECT:
        loop.record_correction(signal, task_id="t_893c7fea")
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# Monotonic counter for unique signal IDs
_signal_counter = itertools.count(int(time.time() * 1000))

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class SignalType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    CORRECT = "correct"
    CLARIFY = "clarify"
    IMPLICIT_SKIP = "implicit_skip"
    UNKNOWN = "unknown"


class SignalStrength(Enum):
    EXPLICIT = "explicit"  # Direct statement: "approved", "wrong", "change X to Y"
    IMPLICIT = "implicit"  # Inference: thumbs-up, fast follow-up, silence
    WEAK = "weak"          # Hedged: "maybe", "could you also", "what about"


@dataclass
class FeedbackSignal:
    """A single classified feedback signal from a chat message."""
    signal_id: str
    message_id: str = ""
    platform: str = ""  # discord, telegram, cli
    channel: str = ""
    signal_type: SignalType = SignalType.UNKNOWN
    strength: SignalStrength = SignalStrength.IMPLICIT
    confidence: float = 0.5  # classifier confidence 0.0-1.0
    linked_task_id: str = ""  # kanban task ID if found
    linked_decision: str = ""  # decision or action being judged
    correction_target: str = ""  # the wrong value being corrected
    correction_replacement: str = ""  # the correct value
    raw_text: str = ""
    timestamp: str = ""


@dataclass
class FeedbackReport:
    """Aggregate feedback quality signal for a time range or agent."""
    total_signals: int = 0
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    correction_rate: float = 0.0
    clarification_rate: float = 0.0
    by_task: dict[str, list[str]] = field(default_factory=dict)  # task_id -> signal_ids
    trend: str = "stable"  # improving, stable, declining


def _make_signal_id() -> str:
    """Generate a unique signal ID."""
    return f"fb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{next(_signal_counter)}"


# ---------------------------------------------------------------------------
# Message classifier
# ---------------------------------------------------------------------------


# Explicit patterns — high confidence
APPROVE_PATTERNS = [
    r"\b(?:approved|looks good|lgtm|ship it|perfect|great|excellent)\b",
    r"\b(?:nice|well done|spot on|nailed it|correct)\b",
    r"^\s*(?:yes|yep|yeah|ok|okay|good|fine)\s*$",
    r"[👍✅👏💯]",
    r"\b(?:merge|deploy|release|go ahead|proceed)\b",
]

REJECT_PATTERNS = [
    r"\b(?:no|nope|wrong|incorrect|bad|terrible|awful)\b",
    r"\b(?:redo|rework|start over|try again|fix this)\b",
    r"\b(?:this is wrong|that's not right|completely off)\b",
    r"[👎❌🚫]",
    r"\b(?:don't\s+(?:do|merge|push|deploy|touch|change|run)|stop|revert|roll back|undo)\b",
]

CORRECT_PATTERNS = [
    # "not X, but Y" or "not X it's Y" — flexible word span
    r"\bnot\s+.{1,40}?\s(?:but|it's|it\s+is)\s+",
    # "actually / should be / ought to be / supposed to be / really"
    r"\b(?:actually|should be|ought to be|supposed to be)\b",
    # "change/replace THING to/with OTHER"
    r"\b(?:change|replace)\s+.+\s+(?:to|with)\s+",
    # "you mean / I meant"
    r"\b(?:you mean|I meant|what I meant was)\b",
]

CLARIFY_PATTERNS = [
    r"\b(?:what about|how about|can you|could you)\b",
    r"\b(?:why|explain|elaborate|clarify)\b",
    r"\?\s*$",
    r"\b(?:I don't understand|confus(?:ed|ing)|not clear)\b",
]


def classify_message(
    text: str,
    *,
    linked_task_id: str = "",
    platform: str = "",
    channel: str = "",
) -> FeedbackSignal:
    """Classify a chat message into a feedback signal.

    Runs regex patterns in priority order: correct > reject > approve > clarify.
    Corrections take highest priority because "actually X not Y" contains
    both a rejection and a replacement.

    Args:
        text: Raw chat message text.
        linked_task_id: Known task ID if context available.
        platform: Source platform (discord, telegram, cli).
        channel: Source channel or chat ID.

    Returns:
        FeedbackSignal with type, strength, and confidence.
    """
    signal_id = _make_signal_id()
    text_lower = text.lower().strip()

    # Priority: correct > reject > approve > clarify
    for pattern in CORRECT_PATTERNS:
        if re.search(pattern, text_lower):
            correction_target, correction_replacement = _extract_correction(text)
            return FeedbackSignal(
                signal_id=signal_id,
                signal_type=SignalType.CORRECT,
                strength=SignalStrength.EXPLICIT,
                confidence=0.85,
                linked_task_id=linked_task_id,
                correction_target=correction_target,
                correction_replacement=correction_replacement,
                raw_text=text,
                platform=platform,
                channel=channel,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    for pattern in REJECT_PATTERNS:
        if re.search(pattern, text_lower):
            return FeedbackSignal(
                signal_id=signal_id,
                signal_type=SignalType.REJECT,
                strength=SignalStrength.EXPLICIT,
                confidence=0.80,
                linked_task_id=linked_task_id,
                raw_text=text,
                platform=platform,
                channel=channel,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    for pattern in APPROVE_PATTERNS:
        if re.search(pattern, text_lower):
            return FeedbackSignal(
                signal_id=signal_id,
                signal_type=SignalType.APPROVE,
                strength=SignalStrength.EXPLICIT,
                confidence=0.80,
                linked_task_id=linked_task_id,
                raw_text=text,
                platform=platform,
                channel=channel,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    for pattern in CLARIFY_PATTERNS:
        if re.search(pattern, text_lower):
            return FeedbackSignal(
                signal_id=signal_id,
                signal_type=SignalType.CLARIFY,
                strength=SignalStrength.IMPLICIT,
                confidence=0.65,
                linked_task_id=linked_task_id,
                raw_text=text,
                platform=platform,
                channel=channel,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    return FeedbackSignal(
        signal_id=signal_id,
        signal_type=SignalType.UNKNOWN,
        strength=SignalStrength.WEAK,
        confidence=0.0,
        linked_task_id=linked_task_id,
        raw_text=text,
        platform=platform,
        channel=channel,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _extract_correction(text: str) -> tuple[str, str]:
    """Extract the wrong value and corrected value from a correction message.

    Heuristic: looks for patterns like 'not X but Y', 'X should be Y', etc.
    """
    # Pattern: "not X but Y" or "not X, it's Y"
    m = re.search(r"not\s+['\"]?(\S+(?:\s+\S+){0,4})['\"]?\s+(?:but|it's|it\s+is)\s+['\"]?(\S+(?:\s+\S+){0,4})['\"]?", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern: "X should be Y" or "X ought to be Y"
    m = re.search(r"(\S+(?:\s+\S+){0,4})\s+(?:should|ought|needs)\s+(?:to\s+)?be\s+(\S+(?:\s+\S+){0,4})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern: "change X to Y" or "replace X with Y"
    m = re.search(r"(?:change|replace)\s+['\"]?(\S+(?:\s+\S+){0,4})['\"]?\s+(?:to|with)\s+['\"]?(\S+(?:\s+\S+){0,4})['\"]?", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return "", ""


# ---------------------------------------------------------------------------
# Feedback store (SQLite)
# ---------------------------------------------------------------------------


DEFAULT_FEEDBACK_DB = "~/.hermes/kensei/feedback.db"


class FeedbackLoop:
    """Persistent feedback loop with SQLite storage.

    Usage:
        loop = FeedbackLoop()
        signal = loop.classify_message("that's wrong, port is 8080")
        loop.record(signal)

        # Query quality signals for review gate
        quality = loop.agent_quality_score(agent="remii-deep", days=7)
        report = loop.report(days=7)
    """

    def __init__(self, store_path: str = DEFAULT_FEEDBACK_DB):
        self.store_path = os.path.expanduser(store_path)
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_signals (
                    signal_id TEXT PRIMARY KEY,
                    message_id TEXT DEFAULT '',
                    platform TEXT DEFAULT '',
                    channel TEXT DEFAULT '',
                    signal_type TEXT DEFAULT 'unknown',
                    strength TEXT DEFAULT 'implicit',
                    confidence REAL DEFAULT 0.5,
                    linked_task_id TEXT DEFAULT '',
                    linked_decision TEXT DEFAULT '',
                    correction_target TEXT DEFAULT '',
                    correction_replacement TEXT DEFAULT '',
                    raw_text TEXT DEFAULT '',
                    timestamp TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_task
                ON feedback_signals(linked_task_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback_signals(signal_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                ON feedback_signals(timestamp)
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.store_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Classification (delegates to module-level)
    # ------------------------------------------------------------------

    def classify_message(
        self,
        text: str,
        *,
        linked_task_id: str = "",
        platform: str = "",
        channel: str = "",
    ) -> FeedbackSignal:
        """Classify a message and optionally record it."""
        return classify_message(
            text,
            linked_task_id=linked_task_id,
            platform=platform,
            channel=channel,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, signal: FeedbackSignal) -> None:
        """Persist a feedback signal to the store."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback_signals
                (signal_id, message_id, platform, channel, signal_type,
                 strength, confidence, linked_task_id, linked_decision,
                 correction_target, correction_replacement, raw_text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id, signal.message_id, signal.platform,
                    signal.channel, signal.signal_type.value, signal.strength.value,
                    signal.confidence, signal.linked_task_id, signal.linked_decision,
                    signal.correction_target, signal.correction_replacement,
                    signal.raw_text, signal.timestamp,
                ),
            )

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def agent_quality_score(
        self,
        *,
        agent: str = "",
        days: int = 7,
    ) -> float:
        """Compute a quality score (0.0–1.0) for an agent based on feedback.

        Score = (approvals - rejections) / total_feedback, clamped to [0, 1].
        Negative score = more rejections than approvals.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = """
            SELECT signal_type, COUNT(*) as cnt
            FROM feedback_signals
            WHERE timestamp >= ?
        """
        params = [cutoff]

        if agent:
            query += " AND linked_decision LIKE ?"
            params.append(f"%{agent}%")

        query += " GROUP BY signal_type"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        counts = {r["signal_type"]: r["cnt"] for r in rows}
        approvals = counts.get("approve", 0)
        rejections = counts.get("reject", 0)
        corrections = counts.get("correct", 0)  # treat as rejection for scoring
        total = approvals + rejections + corrections

        if total == 0:
            return 0.5  # neutral — no data

        score = (approvals - (rejections + corrections)) / total
        return max(0.0, min(1.0, score + 0.5))  # normalise to 0–1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, *, days: int = 7) -> FeedbackReport:
        """Generate a feedback report for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT signal_type, linked_task_id, signal_id, timestamp
                FROM feedback_signals
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (cutoff,),
            ).fetchall()

        total = len(rows)
        counts = {"approve": 0, "reject": 0, "correct": 0, "clarify": 0, "unknown": 0}
        by_task: dict[str, list[str]] = {}

        for r in rows:
            counts[r["signal_type"]] = counts.get(r["signal_type"], 0) + 1
            tid = r["linked_task_id"]
            if tid:
                by_task.setdefault(tid, []).append(r["signal_id"])

        # Trend: compare first half to second half
        half = total // 2 if total > 1 else 0
        if half > 0:
            first_half_approvals = sum(
                1 for r in rows[:half] if r["signal_type"] == "approve"
            )
            second_half_approvals = sum(
                1 for r in rows[half:] if r["signal_type"] == "approve"
            )
            diff = second_half_approvals - first_half_approvals
            trend = "improving" if diff > 0 else ("declining" if diff < 0 else "stable")
        else:
            trend = "stable"

        return FeedbackReport(
            total_signals=total,
            approval_rate=counts["approve"] / total if total else 0,
            rejection_rate=counts["reject"] / total if total else 0,
            correction_rate=counts["correct"] / total if total else 0,
            clarification_rate=counts["clarify"] / total if total else 0,
            by_task=by_task,
            trend=trend,
        )

    # ------------------------------------------------------------------
    # Integration: detect implicit skips (no response to a decision)
    # ------------------------------------------------------------------

    def detect_implicit_skips(
        self,
        *,
        task_id: str,
        decision_timestamp: str,
        silence_hours: int = 4,
    ) -> bool:
        """Check whether a task got no response for N hours after a decision.

        If true, record an IMPLICIT_SKIP signal as a weak negative.
        """
        if not task_id:
            return False

        try:
            decision_dt = datetime.fromisoformat(
                decision_timestamp.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return False

        cutoff = decision_dt + timedelta(hours=silence_hours)
        now = datetime.now(timezone.utc)

        if now < cutoff:
            return False  # still within grace period

        # Check for any feedback on this task after the decision
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM feedback_signals
                WHERE linked_task_id = ? AND timestamp >= ?
                """,
                (task_id, decision_dt.isoformat()),
            ).fetchone()

        if row and row["cnt"] == 0:
            # Record implicit skip
            signal = FeedbackSignal(
                signal_id=_make_signal_id(),
                signal_type=SignalType.IMPLICIT_SKIP,
                strength=SignalStrength.WEAK,
                confidence=0.40,
                linked_task_id=task_id,
                linked_decision=decision_timestamp,
                timestamp=now.isoformat(),
            )
            self.record(signal)
            return True

        return False

    # ------------------------------------------------------------------
    # Integration: link feedback to eval harness (P2-2)
    # ------------------------------------------------------------------

    def signals_as_eval_labels(
        self,
        *,
        task_id: str,
        days: int = 30,
    ) -> list[dict]:
        """Return feedback signals formatted as soft labels for the eval harness.

        Each label maps a signal to {"verdict": "pass"|"fail", "reason": str}.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT signal_type, raw_text, timestamp
                FROM feedback_signals
                WHERE linked_task_id = ? AND timestamp >= ?
                ORDER BY timestamp
                """,
                (task_id, cutoff),
            ).fetchall()

        labels = []
        for r in rows:
            if r["signal_type"] in ("approve",):
                labels.append({"verdict": "pass", "reason": r["raw_text"][:200]})
            elif r["signal_type"] in ("reject", "correct", "implicit_skip"):
                labels.append({"verdict": "fail", "reason": r["raw_text"][:200]})
        return labels
