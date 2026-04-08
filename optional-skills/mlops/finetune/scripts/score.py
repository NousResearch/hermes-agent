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
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import common
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


# ============================================================================
# Positive signal detectors (Phase 1 — see hermes-finetune-positive-signals-spec)
# ============================================================================
#
# These signals are based on observable outcomes (tools succeeded, artifacts
# referenced, sessions resolved) rather than interpreted intent (sentiment,
# affirmation phrases). They were calibrated on a real Hermes session DB
# with 92% tool-call density. The text-marker fallbacks below preserve
# coverage for casual chat data where tool outcomes are less informative.

# Common stoplist for identifier extraction — Python builtins, common words
# we don't want to count as artifacts.
_IDENTIFIER_STOPLIST: Set[str] = {
    "self", "cls", "true", "false", "none", "null", "return", "yield",
    "import", "from", "class", "def", "lambda", "with", "while", "for",
    "if", "elif", "else", "try", "except", "finally", "raise", "assert",
    "pass", "break", "continue", "global", "nonlocal", "in", "is", "not",
    "and", "or", "as", "print", "len", "str", "int", "float", "list",
    "dict", "tuple", "set", "bool", "type", "isinstance", "hasattr",
    "getattr", "setattr", "open", "range", "enumerate", "zip", "map",
    "filter", "sorted", "reversed", "any", "all", "sum", "min", "max",
    "abs", "round", "input", "format", "repr", "vars", "dir", "help",
    "exec", "eval", "id", "hash", "iter", "next", "object", "super",
    "the", "and", "for", "with", "this", "that", "from", "into", "your",
    "have", "will", "would", "should", "could", "might", "must", "make",
    "made", "find", "found", "look", "loop", "step", "use", "used", "uses",
    "set", "get", "put", "add", "new", "old", "all", "any", "let", "lets",
}

# File path extraction — captures common code/config extensions
_FILE_PATH_PATTERN = re.compile(
    r'(?:^|\s|`|"|\')'
    r'((?:[\w./\-]+/)?[\w\-]+\.(?:py|js|ts|tsx|jsx|md|yaml|yml|json|toml|sh|bash|zsh|rs|go|rb|java|c|cpp|h|hpp|html|css|scss|sql|env|cfg|ini|conf|lock))'
    r'(?:\s|`|"|\'|:|$)'
)

# Identifier extraction — snake_case or CamelCase tokens 4+ chars
_IDENTIFIER_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b')

# Bash command extraction — common command starts inside fenced bash blocks
# or terminal tool calls
_BASH_COMMAND_HEAD = re.compile(
    r'\b(cargo|git|npm|yarn|pnpm|pip|python|python3|node|deno|bun|'
    r'curl|wget|ssh|scp|rsync|sed|awk|grep|rg|find|fd|ls|cat|tail|head|'
    r'docker|kubectl|systemctl|sudo|make|cmake|ninja|go|rustc|gcc|clang|'
    r'mkdir|rm|mv|cp|chmod|chown|tar|zip|unzip|gzip|gunzip)\s'
)

# Patterns that indicate a tool result is a soft failure (returned cleanly
# but produced no useful data)
_SOFT_FAILURE_PATTERNS = [
    re.compile(r'"total_count":\s*0\b'),
    re.compile(r'"results":\s*\[\s*\]'),
    re.compile(r'"matches":\s*\[\s*\]'),
    re.compile(r'"files":\s*\[\s*\]'),
    re.compile(r'\bno results found\b', re.I),
    re.compile(r'\bno matches found\b', re.I),
    re.compile(r'\bnothing to (read|search|find)\b', re.I),
]

# Patterns that indicate a tool result is a hard failure
_HARD_FAILURE_PATTERNS = [
    re.compile(r'"error":\s*"[^"]+"'),
    re.compile(r'\bexit_code["\s:]+[1-9]\d*'),
    re.compile(r'\bpermission denied\b', re.I),
    re.compile(r'\bnot found\b', re.I),
    re.compile(r'\btraceback\s*\(most recent call last\)', re.I),
    re.compile(r'\bsegmentation fault\b', re.I),
    re.compile(r'\bcommand not found\b', re.I),
    re.compile(r'\bsyntax error\b', re.I),
]


def _tool_result_status(content: Any) -> str:
    """
    Classify a tool result content as 'success', 'soft_failure', or 'hard_failure'.

    Soft failure = the tool returned cleanly but produced nothing useful
    (e.g. search with zero results). Treated as failure for scoring.

    Hard failure = the tool errored, traceback, exit code != 0, etc.
    """
    if content is None:
        return "hard_failure"

    text = content if isinstance(content, str) else json.dumps(content)
    if not text.strip():
        return "soft_failure"

    if any(p.search(text) for p in _HARD_FAILURE_PATTERNS):
        return "hard_failure"
    if any(p.search(text) for p in _SOFT_FAILURE_PATTERNS):
        return "soft_failure"
    return "success"


def _collect_tool_results(turns: List[Dict], assistant_idx: int) -> List[Tuple[str, Any]]:
    """
    Collect tool result messages that follow an assistant turn at index
    `assistant_idx` and precede the next user/assistant turn.

    Returns list of (tool_name, content) tuples.
    """
    results = []
    for j in range(assistant_idx + 1, len(turns)):
        t = turns[j]
        if t.get("role") == "tool":
            results.append((t.get("tool_name") or "", t.get("content")))
        elif t.get("role") in ("user", "assistant"):
            break
    return results


def _extract_artifacts(turn: Dict) -> Dict[str, Set[str]]:
    """
    Extract trackable artifacts from an assistant turn.

    Returns a dict with keys: 'paths', 'identifiers', 'commands'.
    Each value is a set of strings (deduplicated).
    """
    content = turn.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text", "") if isinstance(c, dict) else c)
            for c in content
        )

    # Also pull text from tool_calls (the args usually contain paths)
    for tc in (turn.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        args = func.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        content = content + "\n" + str(args)

    paths = set(_FILE_PATH_PATTERN.findall(content))

    # Identifiers from inside fenced code blocks (more precise)
    identifiers: Set[str] = set()
    for code_block in re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL):
        for ident in _IDENTIFIER_PATTERN.findall(code_block):
            if ident.lower() not in _IDENTIFIER_STOPLIST and len(ident) >= 4:
                identifiers.add(ident)

    # Bash commands — first token of each command line
    commands: Set[str] = set()
    for match in _BASH_COMMAND_HEAD.finditer(content):
        commands.add(match.group(1))

    return {
        "paths": paths,
        "identifiers": identifiers,
        "commands": commands,
    }


def _artifact_referenced(turn: Dict, artifact: str) -> bool:
    """Whether `artifact` (a string) appears in this turn's content or tool_calls."""
    content = turn.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text", "") if isinstance(c, dict) else c)
            for c in content
        )

    if artifact in content:
        return True

    for tc in (turn.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        args = func.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        if artifact in str(args):
            return True

    return False


# === Signal 1: Tool success chain ===

def positive_tool_success_chain(
    turns: List[Dict],
    assistant_idx: int,
) -> float:
    """
    Score a turn based on its tool success chain.

    Returns 0.0 if the turn has no tool calls, or if the tool failed.
    Returns 0.5/0.7/0.9/1.0 based on chain links satisfied.
    """
    turn = turns[assistant_idx]
    tool_calls = turn.get("tool_calls") or []
    if not tool_calls:
        return 0.0

    # Link 1: tool succeeded
    results = _collect_tool_results(turns, assistant_idx)
    if not results:
        return 0.0  # tool calls but no results captured — can't verify

    statuses = [_tool_result_status(content) for _, content in results]
    if any(s != "success" for s in statuses):
        return 0.0

    score = 0.5

    # Find the next user turn
    next_user = None
    for j in range(assistant_idx + 1, len(turns)):
        if turns[j].get("role") == "user":
            next_user = turns[j]
            break

    if next_user is None:
        # End of session after success — still credit the chain
        return 0.7

    user_content = (next_user.get("content") or "")

    # Link 2: user did not correct/retry
    if any(p.search(user_content) for p in CORRECTION_PATTERNS):
        return 0.5  # tool succeeded but user corrected
    score = 0.7

    # Link 3: user advanced topic (significant new content beyond a yes/no)
    if len(user_content.split()) > 8:
        score = 0.9

    # Link 4: user referenced tool output
    artifacts = _extract_artifacts(turn)
    all_artifacts = artifacts["paths"] | artifacts["identifiers"] | artifacts["commands"]
    for r_name, r_content in results:
        if r_content:
            r_text = r_content if isinstance(r_content, str) else json.dumps(r_content)
            for token in re.findall(r'[\w./\-]{4,}', r_text)[:50]:
                if token in user_content:
                    return 1.0
    if any(a in user_content for a in all_artifacts):
        return 1.0

    return score


# === Signal 2: Artifact longevity ===

def positive_artifact_longevity(
    turns: List[Dict],
    assistant_idx: int,
) -> float:
    """
    Score a turn based on how long the artifacts it produced persist
    in the conversation.

    Returns 0.0 if no artifacts produced or never referenced.
    """
    turn = turns[assistant_idx]
    artifacts = _extract_artifacts(turn)
    all_artifacts = artifacts["paths"] | artifacts["identifiers"] | artifacts["commands"]
    if not all_artifacts:
        return 0.0

    last_reference = assistant_idx
    was_modified = False

    for i in range(assistant_idx + 1, len(turns)):
        for artifact in all_artifacts:
            if _artifact_referenced(turns[i], artifact):
                last_reference = i
                # Modified = referenced in a write tool call (write_file, patch)
                for tc in (turns[i].get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function", tc)
                    name = (func.get("name") or "").lower()
                    if any(w in name for w in ("write", "patch", "edit", "update")):
                        args = func.get("arguments", "")
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        if artifact in str(args):
                            was_modified = True

    span = last_reference - assistant_idx
    if span == 0:
        return 0.0

    if span <= 2:
        score = 0.5
    elif span <= 5:
        score = 0.7
    elif span <= 10:
        score = 0.85
    else:
        score = 1.0

    if was_modified:
        score = min(1.0, score + 0.1)

    return score


# === Signal 3: Self-correction (tool-outcome-based) ===

def positive_self_correction(
    turns: List[Dict],
    assistant_idx: int,
) -> Optional[float]:
    """
    Score a turn if it's a successful correction of a prior failure.

    Returns None if the turn is not part of a correction pattern.
    """
    # Find the previous assistant turn
    prev_asst_idx = None
    for j in range(assistant_idx - 1, -1, -1):
        if turns[j].get("role") == "assistant":
            prev_asst_idx = j
            break
    if prev_asst_idx is None:
        return None

    # Did the previous assistant turn fail at the tool level?
    prev_results = _collect_tool_results(turns, prev_asst_idx)
    prev_failed = False
    if prev_results:
        statuses = [_tool_result_status(content) for _, content in prev_results]
        prev_failed = any(s != "success" for s in statuses)

    # Fallback: text-marker correction in the user turn between
    user_between = None
    for j in range(prev_asst_idx + 1, assistant_idx):
        if turns[j].get("role") == "user":
            user_between = turns[j]
            break

    if user_between is None:
        return None

    user_content = user_between.get("content") or ""
    text_marker_correction = any(p.search(user_content) for p in CORRECTION_PATTERNS)

    if not prev_failed and not text_marker_correction:
        # Try the artifact-overlap heuristic: user references the same artifacts
        # the previous assistant turn touched
        prev_artifacts = _extract_artifacts(turns[prev_asst_idx])
        all_prev = prev_artifacts["paths"] | prev_artifacts["identifiers"] | prev_artifacts["commands"]
        if not any(a in user_content for a in all_prev):
            return None
        # Also require new constraints (length heuristic)
        if len(user_content.split()) < 15:
            return None

    # Did THIS turn's tool calls succeed?
    this_results = _collect_tool_results(turns, assistant_idx)
    if this_results:
        this_statuses = [_tool_result_status(content) for _, content in this_results]
        if any(s != "success" for s in this_statuses):
            return 0.0  # second attempt also failed

    # Did the user accept (no further correction)?
    next_user = None
    for j in range(assistant_idx + 1, len(turns)):
        if turns[j].get("role") == "user":
            next_user = turns[j]
            break

    if next_user is None:
        return 0.6  # corrected, can't confirm acceptance

    next_content = next_user.get("content") or ""
    if any(p.search(next_content) for p in CORRECTION_PATTERNS):
        return 0.0  # user re-corrected

    # Check if the next assistant turn (if any) also failed
    for j in range(assistant_idx + 1, len(turns)):
        if turns[j].get("role") == "assistant":
            next_results = _collect_tool_results(turns, j)
            if next_results:
                next_statuses = [_tool_result_status(content) for _, content in next_results]
                if any(s != "success" for s in next_statuses):
                    return 0.4
            break

    return 0.9


# === Signal 4: Resolution velocity (tool-chain-completion-based) ===

def detect_resolution(turns: List[Dict]) -> Optional[int]:
    """
    Detect if a session is resolved. Returns the index of the resolution
    (the final successful assistant turn) or None.

    Resolution criteria:
      1. At least 3 turns total
      2. Last assistant turn includes tool calls
      3. All those tool calls succeeded
      4. The user did not respond after the final assistant turn
      5. No correction pattern in the last 3 user turns
    """
    if len(turns) < 3:
        return None

    # Find the last assistant turn
    last_asst_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        return None

    # User must not have responded after
    if any(t.get("role") == "user" for t in turns[last_asst_idx + 1:]):
        return None

    # Last assistant turn must have had tool calls that succeeded
    tool_calls = turns[last_asst_idx].get("tool_calls") or []
    if not tool_calls:
        # Allow text-marker resolution as a fallback
        for t in reversed(turns):
            if t.get("role") == "user":
                content = t.get("content") or ""
                if any(p.search(content) for p in CONCLUSION_PATTERNS):
                    return last_asst_idx
                break
        return None

    results = _collect_tool_results(turns, last_asst_idx)
    if not results:
        return None
    statuses = [_tool_result_status(content) for _, content in results]
    if any(s != "success" for s in statuses):
        return None

    # No correction in the last 3 user turns
    user_turns = [t for t in turns if t.get("role") == "user"]
    for u in user_turns[-3:]:
        content = u.get("content") or ""
        if any(p.search(content) for p in CORRECTION_PATTERNS):
            return None

    return last_asst_idx


def positive_resolution_velocity(
    turns: List[Dict],
    resolution_idx: int,
) -> Dict[int, float]:
    """
    Compute velocity scores for each assistant turn in a resolved session.

    Walks backward from the resolution. Turns close to the resolution
    that did not cause detours score highest.
    """
    scores: Dict[int, float] = {}
    remaining_distance = 0

    for i in range(resolution_idx, -1, -1):
        if turns[i].get("role") != "assistant":
            continue

        # Did this turn's tool calls succeed?
        results = _collect_tool_results(turns, i)
        statuses = [_tool_result_status(content) for _, content in results] if results else []
        prev_failed = bool(statuses) and any(s != "success" for s in statuses)

        # Did the next user turn correct it?
        next_user_corrected = False
        for j in range(i + 1, len(turns)):
            if turns[j].get("role") == "user":
                content = turns[j].get("content") or ""
                if any(p.search(content) for p in CORRECTION_PATTERNS):
                    next_user_corrected = True
                break

        if prev_failed or next_user_corrected:
            scores[i] = 0.2
            remaining_distance += 2
        else:
            proximity = 1.0 / (1.0 + remaining_distance * 0.2)
            scores[i] = max(0.5, proximity)
            remaining_distance = max(0, remaining_distance - 1)

    return scores


# === Signal 5: Token efficiency (tiebreaker only) ===

def positive_efficiency_modifier(
    turn: Dict,
    baseline_score: float,
    category_median_tokens: float,
) -> float:
    """
    Small modifier for turns already scoring >= 0.5. Capped at +/- 0.05.
    """
    if baseline_score < 0.5 or category_median_tokens <= 0:
        return 0.0

    content = turn.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    turn_tokens = max(1, len(str(content).split()))

    ratio = turn_tokens / category_median_tokens
    if ratio <= 0.7:
        return 0.05
    if ratio >= 2.0:
        return -0.05
    return 0.0


class QualityScorer:
    """Score session quality using heuristic signals."""

    def __init__(self, config: dict = None):
        cfg = config or load_config().get("scoring", {})
        self.weights = cfg.get("weights", {})
        self.thresholds = cfg.get("thresholds", {})

        # Two scoring modes:
        #   - "positive_signals" (default): the new outcome-based scorer that
        #     uses tool success chains, artifact longevity, self-correction,
        #     resolution velocity, and token efficiency. Best for power-user
        #     workloads with high tool-call density.
        #   - "legacy": the original sentiment + affirmation pattern scorer.
        #     Kept for backwards compatibility and casual chat data.
        self.scoring_mode = cfg.get("mode", "positive_signals")

        if self.scoring_mode == "positive_signals":
            # New weights from the positive-signals spec §"Updated Default Weights"
            self.w_conv = self.weights.get("conversation_signal", 0.15)
            self.w_negative = self.weights.get("negative_turn_signals", 0.25)
            self.w_positive = self.weights.get("positive_turn_signals", 0.35)
            self.w_sent = self.weights.get("sentiment_modifier", 0.05)
            self.w_manual = self.weights.get("manual_override", 0.20)
            # Legacy aliases (kept so the existing turn signal path still
            # contributes when the positive scorer falls through)
            self.w_turn = self.w_negative
            self.w_judge = 0.0
        else:
            # Legacy weights
            self.w_conv = self.weights.get("conversation_signal", 0.3)
            self.w_turn = self.weights.get("turn_signal", 0.4)
            self.w_sent = self.weights.get("sentiment_modifier", 0.1)
            self.w_judge = self.weights.get("judge_score", 0.2)
            self.w_negative = self.w_turn
            self.w_positive = 0.0
            self.w_manual = 0.0

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
        for record in read_jsonl(common.FEEDBACK_PATH):
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
        for record in read_jsonl(common.FEEDBACK_PATH):
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
        # Negative turn scores (existing) — these handle exclusion
        negative_turn_scores = self._score_turns(turns)

        # Apply per-turn retro labels over the automated negative scores.
        # 1-based user-facing turn numbers map to 0-based message indices.
        per_turn_overrides = self.turn_feedback.get(session_id, {})
        if per_turn_overrides:
            assistant_msg_indices = [
                i for i, t in enumerate(turns) if t.get("role") == "assistant"
            ]
            updated = []
            for msg_idx, score in negative_turn_scores:
                try:
                    assistant_pos = assistant_msg_indices.index(msg_idx) + 1
                except ValueError:
                    assistant_pos = None
                if assistant_pos is not None and assistant_pos in per_turn_overrides:
                    updated.append((msg_idx, per_turn_overrides[assistant_pos]))
                else:
                    updated.append((msg_idx, score))
            negative_turn_scores = updated

        avg_negative = (
            sum(s for _, s in negative_turn_scores) / len(negative_turn_scores)
            if negative_turn_scores else 0.5
        )
        sentiment = self._sentiment_modifier(turns)

        # Positive signal scoring — only in the new mode
        positive_turn_scores: List[Tuple[int, float]] = []
        avg_positive = 0.5  # neutral default
        resolution_idx = None
        velocity_scores: Dict[int, float] = {}

        if self.scoring_mode == "positive_signals" and turns:
            resolution_idx = detect_resolution(turns)
            if resolution_idx is not None:
                velocity_scores = positive_resolution_velocity(turns, resolution_idx)

            for i, t in enumerate(turns):
                if t.get("role") != "assistant":
                    continue
                signals = []

                tsc = positive_tool_success_chain(turns, i)
                if tsc > 0:
                    signals.append(tsc)

                al = positive_artifact_longevity(turns, i)
                if al > 0:
                    signals.append(al)

                sc = positive_self_correction(turns, i)
                if sc is not None and sc > 0:
                    signals.append(sc)

                if i in velocity_scores:
                    signals.append(velocity_scores[i])

                if signals:
                    base = max(signals)
                    # Token efficiency tiebreaker (use a simple median across
                    # this session's assistant turns rather than a global one)
                    asst_token_counts = [
                        max(1, len(str(turns[j].get("content") or "").split()))
                        for j in range(len(turns))
                        if turns[j].get("role") == "assistant"
                    ]
                    median_tokens = (
                        sorted(asst_token_counts)[len(asst_token_counts) // 2]
                        if asst_token_counts else 0
                    )
                    base += positive_efficiency_modifier(t, base, median_tokens)
                    base = max(0.0, min(1.0, base))
                    positive_turn_scores.append((i, base))
                else:
                    # No positive signal applies — neutral
                    positive_turn_scores.append((i, 0.5))

            # Apply per-turn retro overrides to positive scores too
            if per_turn_overrides:
                assistant_msg_indices = [
                    i for i, t in enumerate(turns) if t.get("role") == "assistant"
                ]
                updated_pos = []
                for msg_idx, score in positive_turn_scores:
                    try:
                        assistant_pos = assistant_msg_indices.index(msg_idx) + 1
                    except ValueError:
                        assistant_pos = None
                    if assistant_pos is not None and assistant_pos in per_turn_overrides:
                        updated_pos.append((msg_idx, per_turn_overrides[assistant_pos]))
                    else:
                        updated_pos.append((msg_idx, score))
                positive_turn_scores = updated_pos

            avg_positive = (
                sum(s for _, s in positive_turn_scores) / len(positive_turn_scores)
                if positive_turn_scores else 0.5
            )

        # Composite score
        if self.scoring_mode == "positive_signals":
            composite = (
                self.w_conv * conv_score
                + self.w_negative * avg_negative
                + self.w_positive * avg_positive
                + self.w_sent * (0.5 + sentiment)
                # w_manual is reserved — manual overrides take the early-return
                # path above; it stays 0 here so the math sums to 1.0 with the
                # other weights normalized to 0.80 (the remaining 0.20 is
                # reserved for manual_override when it fires).
            )
        else:
            judge_score = 0.5  # placeholder for legacy mode
            composite = (
                self.w_conv * conv_score
                + self.w_turn * avg_negative
                + self.w_sent * (0.5 + sentiment)
                + self.w_judge * judge_score
            )

        composite = max(0.0, min(1.0, composite))

        bucket = "good" if composite >= self.good_threshold else (
            "neutral" if composite >= self.neutral_threshold else "bad"
        )

        # The exposed turn_scores are the positive scores in the new mode
        # (so format.py picks them up for per-turn extraction); fall back to
        # the negative scores in legacy mode.
        exposed_turn_scores = (
            positive_turn_scores if self.scoring_mode == "positive_signals"
            else negative_turn_scores
        )

        session["scoring"] = {
            "composite_score": composite,
            "bucket": bucket,
            "manual_override": False,
            "scoring_mode": self.scoring_mode,
            "conversation_signal": round(conv_score, 4),
            "negative_turn_signal": round(avg_negative, 4),
            "positive_turn_signal": round(avg_positive, 4),
            "sentiment_modifier": round(sentiment, 4),
            "resolved": resolution_idx is not None,
            "turn_scores": [(idx, round(s, 4)) for idx, s in exposed_turn_scores],
        }
        # Legacy compatibility fields so old format.py / tests still read scores
        session["scoring"]["turn_signal"] = session["scoring"]["positive_turn_signal"] \
            if self.scoring_mode == "positive_signals" else session["scoring"]["negative_turn_signal"]
        session["scoring"]["judge_score"] = 0.0
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
