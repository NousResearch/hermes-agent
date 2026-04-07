#!/usr/bin/env python3
"""
Quality scorer for the finetune pipeline.

Assigns quality scores to extracted sessions using conversation-level signals,
turn-level signals, and sentiment modifiers. No manual labeling required.

Usage:
    python score.py [--input PATH] [--output PATH]
"""

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import (
    EXTRACTED_DIR, SCORED_DIR, FEEDBACK_PATH,
    ensure_dirs, load_config, read_jsonl, append_jsonl, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Correction patterns ──

CORRECTION_PATTERNS = [
    re.compile(r"\bno[,.]?\s+(i\s+meant|actually|that'?s\s+(wrong|not|incorrect))", re.I),
    re.compile(r"\bthat'?s\s+(wrong|incorrect|not\s+(right|what))", re.I),
    re.compile(r"\bactually[,.]?\s+(i\s+(want|need|meant)|it\s+should)", re.I),
    re.compile(r"\bplease\s+(don'?t|stop|instead)", re.I),
    re.compile(r"\bi\s+said\b", re.I),
    re.compile(r"\bnot\s+what\s+i\s+(asked|meant|wanted)", re.I),
]

AFFIRMATION_PATTERNS = [
    re.compile(r"\b(exactly|perfect|great|thanks|thank\s+you|awesome|excellent)\b", re.I),
    re.compile(r"\bthat'?s\s+(right|correct|exactly|perfect|what\s+i\s+(want|need))", re.I),
    re.compile(r"\b(yes|yep|yeah|yup)[!.]?\s*$", re.I),
    re.compile(r"\bworks?\s+(great|perfectly|well)\b", re.I),
    re.compile(r"\bnice\s+(one|work|job)\b", re.I),
]

CONCLUSION_PATTERNS = [
    re.compile(r"\b(thanks|thank\s+you|cheers|appreciate\s+it)\b", re.I),
    re.compile(r"\bgot\s+it\b", re.I),
    re.compile(r"\bthat\s+solves\b", re.I),
    re.compile(r"\ball\s+(set|good|done)\b", re.I),
]

# ── Sentiment lexicon ──

POSITIVE_WORDS = {
    "great", "good", "nice", "excellent", "perfect", "awesome", "wonderful",
    "helpful", "thanks", "love", "brilliant", "fantastic", "amazing", "cool",
    "impressive", "solid", "clean", "elegant",
}

NEGATIVE_WORDS = {
    "bad", "wrong", "broken", "terrible", "awful", "useless", "annoying",
    "frustrating", "confused", "confusing", "ugly", "messy", "worse",
    "horrible", "stupid", "ridiculous", "nonsense", "garbage",
}


def _cosine_similarity_quick(a: str, b: str) -> float:
    """Quick bag-of-words cosine similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a) ** 0.5 * len(words_b) ** 0.5)


class QualityScorer:
    """Score session quality using heuristic signals."""

    def __init__(self, config: dict = None):
        cfg = config or load_config().get("scoring", {})
        self.weights = cfg.get("weights", {})
        self.thresholds = cfg.get("thresholds", {})
        self.w_conv = self.weights.get("conversation_signal", 0.3)
        self.w_turn = self.weights.get("turn_signal", 0.4)
        self.w_sent = self.weights.get("sentiment_modifier", 0.1)
        self.w_judge = self.weights.get("judge_score", 0.2)
        self.good_threshold = self.thresholds.get("good", 0.7)
        self.neutral_threshold = self.thresholds.get("neutral", 0.4)

        # Load manual feedback overrides (session-level + turn-level)
        self.feedback = self._load_feedback()
        self.turn_feedback = self._load_turn_feedback()

    def _load_feedback(self) -> Dict[str, float]:
        """Load session-level overrides from feedback.jsonl.

        Returns a flat dict of session_id -> score for session-level labels
        only. Per-turn labels are loaded separately by _load_turn_feedback.
        Skip records (signal=skip) are excluded so they don't override scores.
        """
        overrides = {}
        for record in read_jsonl(FEEDBACK_PATH):
            sid = record.get("session_id")
            if not sid or "score" not in record:
                continue
            if record.get("turn_index") is not None:
                continue  # turn-level — handled elsewhere
            if record.get("signal") == "skip":
                continue
            overrides[sid] = record["score"]
        return overrides

    def _load_turn_feedback(self) -> Dict[str, Dict[int, float]]:
        """Load per-turn overrides from feedback.jsonl.

        Returns nested dict: session_id -> turn_index (1-based) -> score.
        Used by _score_turns to apply retro labels at turn granularity.
        """
        turn_overrides: Dict[str, Dict[int, float]] = {}
        for record in read_jsonl(FEEDBACK_PATH):
            sid = record.get("session_id")
            turn_idx = record.get("turn_index")
            if not sid or turn_idx is None or "score" not in record:
                continue
            turn_overrides.setdefault(sid, {})[int(turn_idx)] = record["score"]
        return turn_overrides

    # ── Conversation-level signals ──

    def _score_conversation(self, turns: List[Dict]) -> float:
        """Score based on conversation-level patterns."""
        if not turns:
            return 0.5

        user_turns = [t for t in turns if t["role"] == "user"]
        assistant_turns = [t for t in turns if t["role"] == "assistant"]

        if not assistant_turns:
            return 0.3

        signals = []

        # Abrupt termination: session ends shortly after assistant with no resolution
        if len(turns) >= 2:
            last_role = turns[-1]["role"]
            second_last_role = turns[-2]["role"] if len(turns) >= 2 else None
            if last_role == "assistant" and len(user_turns) <= 1:
                # Only one user turn and model responded — might be abandoned
                signals.append(0.3)
            elif last_role == "user":
                # Check if the last user message is a resolution
                last_content = turns[-1].get("content", "") or ""
                if any(p.search(last_content) for p in CONCLUSION_PATTERNS):
                    signals.append(0.9)
                elif any(p.search(last_content) for p in CORRECTION_PATTERNS):
                    signals.append(0.2)

        # Retry/rephrase detection
        for i in range(1, len(user_turns)):
            prev = user_turns[i - 1].get("content", "") or ""
            curr = user_turns[i].get("content", "") or ""
            sim = _cosine_similarity_quick(prev, curr)
            if sim > 0.7:
                signals.append(0.2)  # Likely rephrase

        # Session length vs complexity heuristic
        total_tokens = sum(len((t.get("content") or "").split()) for t in turns)
        tool_calls = sum(1 for t in turns if t.get("tool_calls"))
        complexity = total_tokens / 100 + tool_calls * 2
        turn_ratio = len(turns) / max(complexity, 1)
        if turn_ratio > 5:
            signals.append(0.3)  # Too many turns for the complexity = thrashing
        elif turn_ratio < 2:
            signals.append(0.7)  # Efficient

        # Productive conclusion
        if user_turns:
            last_user = user_turns[-1].get("content", "") or ""
            if any(p.search(last_user) for p in CONCLUSION_PATTERNS):
                signals.append(0.9)

        return sum(signals) / len(signals) if signals else 0.5

    # ── Turn-level signals ──

    def _score_turns(self, turns: List[Dict]) -> List[Tuple[int, float]]:
        """Score individual assistant turns based on user responses."""
        turn_scores = []

        for i, turn in enumerate(turns):
            if turn["role"] != "assistant":
                continue

            # Find the next user turn
            next_user = None
            for j in range(i + 1, len(turns)):
                if turns[j]["role"] == "user":
                    next_user = turns[j]
                    break

            if next_user is None:
                # Last assistant turn with no user response — neutral
                turn_scores.append((i, 0.5))
                continue

            user_content = next_user.get("content", "") or ""
            score = 0.5

            # Direct affirmation
            if any(p.search(user_content) for p in AFFIRMATION_PATTERNS):
                score = max(score, 0.9)

            # Contradiction / correction
            if any(p.search(user_content) for p in CORRECTION_PATTERNS):
                score = min(score, 0.1)

            # Follow-up depth (deeper question, not a correction)
            if (len(user_content.split()) > 15
                    and not any(p.search(user_content) for p in CORRECTION_PATTERNS)):
                score = max(score, 0.7)

            # Artifact adoption: user references code or text from assistant
            assistant_content = turn.get("content", "") or ""
            if assistant_content and user_content:
                # Check if user references specific identifiers from assistant output
                # Look for code tokens the assistant used that the user then references
                assistant_words = set(re.findall(r'\b\w{4,}\b', assistant_content))
                user_words = set(re.findall(r'\b\w{4,}\b', user_content))
                overlap = assistant_words & user_words
                # High overlap of specific terms suggests adoption
                if len(overlap) > 5:
                    score = max(score, 0.75)

            turn_scores.append((i, score))

        return turn_scores

    # ── Sentiment ──

    def _sentiment_modifier(self, turns: List[Dict]) -> float:
        """Compute sentiment modifier from user turns following assistant responses."""
        modifiers = []

        for i, turn in enumerate(turns):
            if turn["role"] != "user" or i == 0:
                continue
            # Check if previous turn was assistant
            if turns[i - 1]["role"] != "assistant":
                continue

            content = (turn.get("content", "") or "").lower()
            words = set(content.split())

            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)

            if pos > neg:
                modifiers.append(0.1)
            elif neg > pos:
                modifiers.append(-0.1)

        if not modifiers:
            return 0.0

        return max(-0.2, min(0.2, sum(modifiers) / len(modifiers)))

    # ── Composite ──

    def score_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single session, returning the session with scoring metadata.

        Returns the original session dict with added 'scoring' key.
        """
        session_id = session.get("session_id", "")
        turns = session.get("turns", [])

        # Check for manual override
        if session_id in self.feedback:
            override = self.feedback[session_id]
            bucket = "good" if override >= self.good_threshold else (
                "neutral" if override >= self.neutral_threshold else "bad"
            )
            session["scoring"] = {
                "composite_score": override,
                "bucket": bucket,
                "manual_override": True,
                "conversation_signal": override,
                "turn_signal": override,
                "sentiment_modifier": 0.0,
                "judge_score": 0.0,
            }
            return session

        conv_score = self._score_conversation(turns)
        turn_scores = self._score_turns(turns)

        # Apply per-turn retro labels (1-based assistant turn index)
        # over the automated scores. The mapping converts 1-based user-facing
        # turn numbers to the 0-based message indices used internally.
        per_turn_overrides = self.turn_feedback.get(session_id, {})
        if per_turn_overrides:
            # Build a mapping: 1-based assistant turn -> internal message index
            assistant_msg_indices = [
                i for i, t in enumerate(turns) if t.get("role") == "assistant"
            ]
            # turn_scores is List[Tuple[msg_idx, score]] — rebuild with overrides applied
            updated = []
            for msg_idx, score in turn_scores:
                # Find which 1-based assistant turn this is
                try:
                    assistant_pos = assistant_msg_indices.index(msg_idx) + 1
                except ValueError:
                    assistant_pos = None
                if assistant_pos is not None and assistant_pos in per_turn_overrides:
                    updated.append((msg_idx, per_turn_overrides[assistant_pos]))
                else:
                    updated.append((msg_idx, score))
            turn_scores = updated

        avg_turn = (sum(s for _, s in turn_scores) / len(turn_scores)
                    if turn_scores else 0.5)
        sentiment = self._sentiment_modifier(turns)

        # No judge score in default mode (would need auxiliary model)
        judge_score = 0.5

        composite = (
            self.w_conv * conv_score
            + self.w_turn * avg_turn
            + self.w_sent * (0.5 + sentiment)  # Center sentiment around 0.5
            + self.w_judge * judge_score
        )
        composite = max(0.0, min(1.0, composite))

        bucket = "good" if composite >= self.good_threshold else (
            "neutral" if composite >= self.neutral_threshold else "bad"
        )

        session["scoring"] = {
            "composite_score": composite,
            "bucket": bucket,
            "manual_override": False,
            "conversation_signal": round(conv_score, 4),
            "turn_signal": round(avg_turn, 4),
            "sentiment_modifier": round(sentiment, 4),
            "judge_score": round(judge_score, 4),
            "turn_scores": [(idx, round(s, 4)) for idx, s in turn_scores],
        }
        return session

    def score_all(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all sessions and write results."""
        ensure_dirs()
        scored = [self.score_session(s) for s in sessions]

        # Bucket counts
        buckets = {"good": 0, "neutral": 0, "bad": 0}
        for s in scored:
            bucket = s.get("scoring", {}).get("bucket", "bad")
            buckets[bucket] = buckets.get(bucket, 0) + 1

        # Write scored output
        if scored:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = SCORED_DIR / f"scored_{ts}.jsonl"
            append_jsonl(output_path, scored)
            logger.info("Scored %d sessions → %s", len(scored), output_path)

        logger.info(
            "Buckets: good=%d, neutral=%d, bad=%d",
            buckets["good"], buckets["neutral"], buckets["bad"],
        )
        return scored

    def get_all_scored(self) -> List[Dict[str, Any]]:
        """Load all previously scored sessions."""
        all_scored = []
        for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
            all_scored.extend(read_jsonl(path))
        return all_scored


def main():
    parser = argparse.ArgumentParser(description="Score extracted sessions")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (default: all extracted)")
    args = parser.parse_args()

    scorer = QualityScorer()

    if args.input:
        sessions = read_jsonl(Path(args.input))
    else:
        # Score all extracted sessions
        sessions = []
        for path in sorted(EXTRACTED_DIR.glob("extract_*.jsonl")):
            sessions.extend(read_jsonl(path))

    if not sessions:
        print("No sessions to score. Run extract.py first.")
        return

    scored = scorer.score_all(sessions)
    buckets = {"good": 0, "neutral": 0, "bad": 0}
    for s in scored:
        b = s.get("scoring", {}).get("bucket", "bad")
        buckets[b] = buckets.get(b, 0) + 1

    print(f"Scored {len(scored)} sessions:")
    print(f"  Good:    {buckets['good']}")
    print(f"  Neutral: {buckets['neutral']}")
    print(f"  Bad:     {buckets['bad']}")


if __name__ == "__main__":
    main()
