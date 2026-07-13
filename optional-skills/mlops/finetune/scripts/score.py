#!/usr/bin/env python3
"""
Quality scorer for the finetune pipeline.

Assigns quality scores to extracted sessions using conversation-level signals,
turn-level signals, and sentiment modifiers. No manual labeling required.

Usage:
    python score.py [--input PATH] [--output PATH]
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import common
from common import (
    EXTRACTED_DIR, SCORED_DIR, FEEDBACK_PATH,
    ensure_dirs, load_config, read_jsonl, append_jsonl,
    content_to_text, load_records_dedup, logger,
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

# Patterns where the user is demanding the assistant actually use a tool
# instead of just answering from memory. Used by positive_no_tool_response
# to penalize no-tool turns that the user rejected.
TOOL_DEMAND_PATTERNS = [
    re.compile(r"\b(actually|please)\s+(use|run|call|execute|invoke)\s+", re.I),
    re.compile(r"\buse\s+(the|a)\s+tool\b", re.I),
    re.compile(r"\b(run|execute|invoke)\s+(the|it|that)\b", re.I),
    re.compile(r"\bcheck\s+(it|the\s+(file|code|repo|actual))", re.I),
    re.compile(r"\b(read|grep|search|look\s+at)\s+the\s+(file|code|repo|source)", re.I),
    re.compile(r"\bdon'?t\s+(guess|assume|make\s+(it|that)\s+up)\b", re.I),
    re.compile(r"\bstop\s+guessing\b", re.I),
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

# Patterns that indicate a tool result is a hard failure.
# Deliberately shaped around real failure output — bare phrases like
# "not found" or "syntax error" appear in plenty of successful outputs
# (docs, diffs, search results) and must not zero them out.
_HARD_FAILURE_PATTERNS = [
    re.compile(r'"error":\s*"[^"]+"'),
    re.compile(r'\bexit_code["\s:]+[1-9]\d*'),
    re.compile(r'\bpermission denied\b', re.I),
    re.compile(r'\btraceback\s*\(most recent call last\)', re.I),
    re.compile(r'\bsegmentation fault\b', re.I),
    re.compile(r'\bcommand not found\b', re.I),
    re.compile(r'\bno such file or directory\b', re.I),
    re.compile(r'^\s*(?:error|fatal):', re.I | re.M),   # stderr-style lines
    re.compile(r'\bSyntaxError\b'),                      # Python exception name
    re.compile(r'\bsyntax error (?:near|at|before) \S', re.I),  # shell/SQL
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
    content = content_to_text(turn.get("content"))

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


def _token_in_text(token: str, text: str) -> bool:
    """Word-boundary containment check for artifact/output tokens.

    Tokens may contain '.', '/', '-' so plain \\b doesn't work; require the
    match to not be embedded inside a longer word-ish run (e.g. 'name' must
    not match inside 'rename').
    """
    if not token or not text:
        return False
    pattern = r"(?<![\w./\-])" + re.escape(token) + r"(?![\w./\-])"
    return re.search(pattern, text) is not None


def _looks_like_identifier(token: str) -> bool:
    """Whether a token looks like a path/identifier/value rather than a
    common English word: contains a digit, '.', '/', '_', '-', an internal
    capital (CamelCase/mixedCase), or is unusually long (>= 8 chars)."""
    if len(token) >= 8:
        return True
    if any(c.isdigit() or c in "./_-" for c in token):
        return True
    return any(c.isupper() for c in token[1:])


def _artifact_referenced(turn: Dict, artifact: str) -> bool:
    """Whether `artifact` (a string) appears in this turn's content or tool_calls."""
    content = content_to_text(turn.get("content"))

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

    user_content = content_to_text(next_user.get("content"))

    # Link 2: user did not correct/retry
    if any(p.search(user_content) for p in CORRECTION_PATTERNS):
        return 0.5  # tool succeeded but user corrected
    score = 0.7

    # Link 3: user advanced topic (significant new content beyond a yes/no)
    if len(user_content.split()) > 8:
        score = 0.9

    # Link 4: user referenced tool output (word-boundary match — 'name'
    # must not count as referenced just because the user typed 'rename').
    # Tokens must clear the identifier stoplist AND look identifier-like:
    # common English words in tool output ('this', 'with', 'from') would
    # otherwise saturate every conversational session at 1.0.
    artifacts = _extract_artifacts(turn)
    all_artifacts = artifacts["paths"] | artifacts["identifiers"] | artifacts["commands"]
    for r_name, r_content in results:
        if r_content:
            r_text = r_content if isinstance(r_content, str) else json.dumps(r_content)
            for token in re.findall(r'[\w./\-]{4,}', r_text)[:50]:
                if token.lower() in _IDENTIFIER_STOPLIST:
                    continue
                if not _looks_like_identifier(token):
                    continue
                if _token_in_text(token, user_content):
                    return 1.0
    if any(_token_in_text(a, user_content) for a in all_artifacts):
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

    user_content = content_to_text(user_between.get("content"))
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

    next_content = content_to_text(next_user.get("content"))
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
                content = content_to_text(t.get("content"))
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
        content = content_to_text(u.get("content"))
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
                content = content_to_text(turns[j].get("content"))
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

    content = content_to_text(turn.get("content"))
    turn_tokens = max(1, len(content.split()))

    ratio = turn_tokens / category_median_tokens
    if ratio <= 0.7:
        return 0.05
    if ratio >= 2.0:
        return -0.05
    return 0.0


# === Signal 6: Good no-tool response ===
#
# Rebalances the training set against tool-call dominance: rewards assistant
# turns that correctly chose NOT to call a tool. Without this signal the
# scorer treats every no-tool turn as 0.5 (neutral), so the trained adapter
# drifts toward "always call a tool" — which regressed no_tool_accuracy and
# canary_pass_rate by ~18% in v19. See finetune-no-tool-rebalance notes.

def positive_no_tool_response(
    turns: List[Dict],
    assistant_idx: int,
) -> Optional[float]:
    """
    Score an assistant turn that didn't call any tools, based on whether
    the user accepted the answer.

    Returns None if the turn DID call tools (let other signals handle it),
    or if the assistant turn has no real content. Otherwise returns a
    score in [0.0, 0.95]:

      - 0.0  if the user explicitly demanded a tool / corrected the answer
      - 0.4  if the user followed up with new constraints (ambiguous)
      - 0.6  if this is the final turn (can't confirm acceptance)
      - 0.8  if the user moved on without rejecting (implicit accept)
      - 0.9  if the user explicitly affirmed (thanks/perfect/etc.)
    """
    turn = turns[assistant_idx]
    if turn.get("tool_calls"):
        return None

    content = content_to_text(turn.get("content"))
    if len(content.strip()) < 20:
        # Empty/trivial replies aren't a meaningful "chose not to call a tool"
        return None

    # Find the next user turn
    next_user = None
    for j in range(assistant_idx + 1, len(turns)):
        if turns[j].get("role") == "user":
            next_user = turns[j]
            break

    if next_user is None:
        return 0.6  # last turn — can't confirm, but don't penalize

    next_content = content_to_text(next_user.get("content"))

    # Hard reject: user demanded a tool or corrected
    if any(p.search(next_content) for p in TOOL_DEMAND_PATTERNS):
        return 0.0
    if any(p.search(next_content) for p in CORRECTION_PATTERNS):
        return 0.0

    # Explicit affirmation — checked before the length rule so a long,
    # explicit "perfect, thanks, that solved it because ..." scores 0.9
    # instead of being misread as new constraints.
    if any(p.search(next_content) for p in AFFIRMATION_PATTERNS):
        return 0.9
    if any(p.search(next_content) for p in CONCLUSION_PATTERNS):
        return 0.9

    # Soft signal: user added new constraints (long, substantive follow-up)
    # — answer was likely incomplete but not wrong
    word_count = len(next_content.split())
    if word_count >= 25:
        return 0.4

    # Implicit accept: short follow-up that moves on
    return 0.8


def _is_bad_label(record: Dict) -> bool:
    """Whether a feedback record is an explicit bad label."""
    if record.get("signal") == "bad":
        return True
    try:
        return record.get("signal") is None and float(record.get("score", 0.5)) <= 0.0
    except (TypeError, ValueError):
        return False


class QualityScorer:
    """Score session quality using heuristic signals."""

    def __init__(self, config: dict = None):
        cfg = config or load_config().get("scoring", {})
        self.thresholds = cfg.get("thresholds", {})

        # Two scoring modes:
        #   - "positive_signals" (default): the new outcome-based scorer that
        #     uses tool success chains, artifact longevity, self-correction,
        #     resolution velocity, and token efficiency. Best for power-user
        #     workloads with high tool-call density.
        #   - "legacy": the original sentiment + affirmation pattern scorer.
        #     Kept for backwards compatibility and casual chat data.
        self.scoring_mode = cfg.get("mode", "positive_signals")

        # Each mode reads its own weights dict — scoring.weights_positive
        # for positive_signals, scoring.weights for legacy. The two share
        # key names (conversation_signal, sentiment_modifier), so letting
        # legacy values feed the positive-signals math would silently skew
        # the composite.
        if self.scoring_mode == "positive_signals":
            self.weights = cfg.get("weights_positive", {})
            # Weights from the positive-signals spec §"Updated Default Weights"
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
            self.weights = cfg.get("weights", {})
            self.w_conv = self.weights.get("conversation_signal", 0.3)
            self.w_turn = self.weights.get("turn_signal", 0.4)
            self.w_sent = self.weights.get("sentiment_modifier", 0.1)
            self.w_judge = self.weights.get("judge_score", 0.2)
            self.w_negative = self.w_turn
            self.w_positive = 0.0
            self.w_manual = 0.0

        if self.scoring_mode == "positive_signals":
            # Renormalize the active weights to sum to 1.0 (preserving their
            # relative proportions). w_manual is NOT part of the composite —
            # manual overrides take the early-return path in score_session —
            # so without this the defaults sum to 0.80 and hard-cap the
            # composite at ~0.785, starving the "good" bucket.
            total = self.w_conv + self.w_negative + self.w_positive + self.w_sent
            if total > 0 and abs(total - 1.0) > 1e-9:
                self.w_conv /= total
                self.w_negative /= total
                self.w_positive /= total
                self.w_sent /= total
                self.w_turn = self.w_negative

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
        self.session_bad_label: Set[str] = set()
        for record in read_jsonl(common.FEEDBACK_PATH):
            sid = record.get("session_id")
            if not sid or "score" not in record:
                continue
            if record.get("turn_index") is not None:
                continue  # turn-level — handled elsewhere
            if record.get("signal") == "skip":
                continue
            overrides[sid] = record["score"]
            if _is_bad_label(record):
                self.session_bad_label.add(sid)
            else:
                self.session_bad_label.discard(sid)
        return overrides

    def _load_turn_feedback(self) -> Dict[str, Dict[int, float]]:
        """Load per-turn overrides from feedback.jsonl.

        Returns nested dict: session_id -> turn_index (1-based) -> score.
        Used by _score_turns to apply retro labels at turn granularity.
        Also populates self.turn_bad_label (session_id -> set of 1-based
        turn indices explicitly labeled bad) for hard exclusion downstream.
        """
        turn_overrides: Dict[str, Dict[int, float]] = {}
        self.turn_bad_label: Dict[str, Set[int]] = {}
        for record in read_jsonl(common.FEEDBACK_PATH):
            sid = record.get("session_id")
            turn_idx = record.get("turn_index")
            if not sid or turn_idx is None or "score" not in record:
                continue
            turn_idx = int(turn_idx)
            turn_overrides.setdefault(sid, {})[turn_idx] = record["score"]
            if _is_bad_label(record):
                self.turn_bad_label.setdefault(sid, set()).add(turn_idx)
            else:
                self.turn_bad_label.get(sid, set()).discard(turn_idx)
        return turn_overrides

    # ── Conversation-level signals ──

    def _score_conversation(self, turns: List[Dict]) -> float:
        """Score based on conversation-level patterns."""
        if not turns:
            return 0.5

        user_turns = [t for t in turns if t.get("role") == "user"]
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]

        if not assistant_turns:
            return 0.3

        signals = []

        # Abrupt termination: session ends shortly after assistant with no resolution
        if len(turns) >= 2:
            last_role = turns[-1].get("role")
            if last_role == "assistant" and len(user_turns) <= 1:
                # Only one user turn and model responded — might be abandoned
                signals.append(0.3)
            elif last_role == "user":
                # Check if the last user message is a resolution
                last_content = content_to_text(turns[-1].get("content"))
                if any(p.search(last_content) for p in CONCLUSION_PATTERNS):
                    signals.append(0.9)
                elif any(p.search(last_content) for p in CORRECTION_PATTERNS):
                    signals.append(0.2)

        # Retry/rephrase detection
        for i in range(1, len(user_turns)):
            prev = content_to_text(user_turns[i - 1].get("content"))
            curr = content_to_text(user_turns[i].get("content"))
            sim = _cosine_similarity_quick(prev, curr)
            if sim > 0.7:
                signals.append(0.2)  # Likely rephrase

        # Session length vs complexity heuristic
        total_tokens = sum(
            len(content_to_text(t.get("content")).split()) for t in turns
        )
        tool_calls = sum(1 for t in turns if t.get("tool_calls"))
        complexity = total_tokens / 100 + tool_calls * 2
        turn_ratio = len(turns) / max(complexity, 1)
        if turn_ratio > 5:
            signals.append(0.3)  # Too many turns for the complexity = thrashing
        elif turn_ratio < 2:
            signals.append(0.7)  # Efficient

        # Productive conclusion
        if user_turns:
            last_user = content_to_text(user_turns[-1].get("content"))
            if any(p.search(last_user) for p in CONCLUSION_PATTERNS):
                signals.append(0.9)

        return sum(signals) / len(signals) if signals else 0.5

    # ── Turn-level signals ──

    def _score_turns(self, turns: List[Dict]) -> List[Tuple[int, float]]:
        """Score individual assistant turns based on user responses."""
        turn_scores = []

        for i, turn in enumerate(turns):
            if turn.get("role") != "assistant":
                continue

            # Find the next user turn
            next_user = None
            for j in range(i + 1, len(turns)):
                if turns[j].get("role") == "user":
                    next_user = turns[j]
                    break

            if next_user is None:
                # Last assistant turn with no user response — neutral
                turn_scores.append((i, 0.5))
                continue

            user_content = content_to_text(next_user.get("content"))
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
            assistant_content = content_to_text(turn.get("content"))
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
            if turn.get("role") != "user" or i == 0:
                continue
            # Check if previous turn was assistant
            if turns[i - 1].get("role") != "assistant":
                continue

            content = content_to_text(turn.get("content")).lower()
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

    def _apply_turn_overrides(
        self,
        turn_scores: List[Tuple[int, float]],
        assistant_msg_indices: List[int],
        per_turn_overrides: Dict[int, float],
    ) -> List[Tuple[int, float]]:
        """Apply 1-based per-turn retro overrides onto (msg_idx, score) pairs."""
        if not per_turn_overrides:
            return turn_scores
        updated = []
        for msg_idx, score in turn_scores:
            try:
                assistant_pos = assistant_msg_indices.index(msg_idx) + 1
            except ValueError:
                assistant_pos = None
            if assistant_pos is not None and assistant_pos in per_turn_overrides:
                updated.append((msg_idx, per_turn_overrides[assistant_pos]))
            else:
                updated.append((msg_idx, score))
        return updated

    def _bad_turn_msg_indices(
        self,
        session_id: str,
        assistant_msg_indices: List[int],
        session_is_bad: bool = False,
    ) -> List[int]:
        """Message indices of assistant turns with an effective bad label.

        A turn-specific label always wins over a session-level one: a
        session-level bad label marks every assistant turn bad EXCEPT turns
        with a non-bad turn-level override, and a turn-level bad label marks
        its turn bad regardless of any session-level label.
        """
        turn_overrides = self.turn_feedback.get(session_id, {})
        bad_positions = set(self.turn_bad_label.get(session_id, set()))
        if session_is_bad:
            for pos in range(1, len(assistant_msg_indices) + 1):
                if pos not in turn_overrides:
                    bad_positions.add(pos)
        return sorted(
            assistant_msg_indices[pos - 1]
            for pos in bad_positions
            if 1 <= pos <= len(assistant_msg_indices)
        )

    def score_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single session, returning the session with scoring metadata.

        Returns the original session dict with added 'scoring' key.
        """
        session_id = session.get("session_id", "")
        turns = session.get("turns", [])
        assistant_msg_indices = [
            i for i, t in enumerate(turns) if t.get("role") == "assistant"
        ]
        per_turn_overrides = self.turn_feedback.get(session_id, {})

        # Check for manual session-level override. Turn-specific labels still
        # win at turn granularity (documented precedence), regardless of the
        # order the records were written in.
        if session_id in self.feedback:
            override = self.feedback[session_id]
            bucket = "good" if override >= self.good_threshold else (
                "neutral" if override >= self.neutral_threshold else "bad"
            )
            turn_scores = self._apply_turn_overrides(
                [(i, override) for i in assistant_msg_indices],
                assistant_msg_indices, per_turn_overrides,
            )
            session["scoring"] = {
                "composite_score": override,
                "bucket": bucket,
                "manual_override": True,
                "conversation_signal": override,
                "turn_signal": override,
                "sentiment_modifier": 0.0,
                "judge_score": 0.0,
                "turn_scores": [(idx, round(s, 4)) for idx, s in turn_scores],
                "bad_turn_indices": self._bad_turn_msg_indices(
                    session_id, assistant_msg_indices,
                    session_is_bad=session_id in self.session_bad_label,
                ),
            }
            return session

        conv_score = self._score_conversation(turns)
        # Negative turn scores (existing) — these handle exclusion.
        # Per-turn retro labels override the automated scores;
        # 1-based user-facing turn numbers map to 0-based message indices.
        negative_turn_scores = self._apply_turn_overrides(
            self._score_turns(turns), assistant_msg_indices, per_turn_overrides,
        )

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

            # Session-median assistant token count for the efficiency
            # tiebreaker — computed once, not per turn.
            asst_token_counts = [
                max(1, len(content_to_text(turns[j].get("content")).split()))
                for j in assistant_msg_indices
            ]
            median_tokens = (
                sorted(asst_token_counts)[len(asst_token_counts) // 2]
                if asst_token_counts else 0
            )

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

                ntr = positive_no_tool_response(turns, i)
                if ntr is not None and ntr > 0:
                    signals.append(ntr)

                if signals:
                    base = max(signals)
                    # Token efficiency tiebreaker (session median, hoisted
                    # above the loop)
                    base += positive_efficiency_modifier(t, base, median_tokens)
                    base = max(0.0, min(1.0, base))
                    positive_turn_scores.append((i, base))
                else:
                    # No positive signal applies — neutral
                    positive_turn_scores.append((i, 0.5))

            # Apply per-turn retro overrides to positive scores too
            positive_turn_scores = self._apply_turn_overrides(
                positive_turn_scores, assistant_msg_indices, per_turn_overrides,
            )

            avg_positive = (
                sum(s for _, s in positive_turn_scores) / len(positive_turn_scores)
                if positive_turn_scores else 0.5
            )

        # Composite score
        if self.scoring_mode == "positive_signals":
            # Weights are renormalized in __init__ to sum to 1.0 (w_manual is
            # not part of this sum — manual overrides take the early-return
            # path above), so a perfect session can actually reach 1.0.
            composite = (
                self.w_conv * conv_score
                + self.w_negative * avg_negative
                + self.w_positive * avg_positive
                + self.w_sent * (0.5 + sentiment)
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
            # Assistant message indices explicitly labeled bad via retro —
            # format.py excludes these from training regardless of threshold.
            "bad_turn_indices": self._bad_turn_msg_indices(
                session_id, assistant_msg_indices,
            ),
        }
        # Legacy compatibility fields so old format.py / tests still read scores
        session["scoring"]["turn_signal"] = session["scoring"]["positive_turn_signal"] \
            if self.scoring_mode == "positive_signals" else session["scoring"]["negative_turn_signal"]
        session["scoring"]["judge_score"] = 0.0
        return session

    def score_all(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all sessions and write results.

        A session that raises during scoring is logged and skipped so a
        single malformed session can't abort the whole run.
        """
        ensure_dirs()
        scored = []
        for s in sessions:
            try:
                scored.append(self.score_session(s))
            except Exception as e:
                logger.warning(
                    "Failed to score session %s: %s — skipping it.",
                    s.get("session_id", "<unknown>") if isinstance(s, dict) else "<unknown>",
                    e,
                )

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
        """Load all previously scored sessions, deduped by session_id
        (keeping the record from the newest snapshot)."""
        return load_records_dedup(SCORED_DIR, "scored_*.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Score extracted sessions")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (default: all extracted)")
    args = parser.parse_args()

    scorer = QualityScorer()

    if args.input:
        sessions = read_jsonl(Path(args.input))
    else:
        # Score all extracted sessions (deduped: re-extracted sessions keep
        # the newest copy)
        sessions = load_records_dedup(EXTRACTED_DIR, "extract_*.jsonl")

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
