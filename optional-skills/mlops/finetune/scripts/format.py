#!/usr/bin/env python3
"""
Training data formatter for the finetune pipeline.

Emits one training record per qualifying assistant turn (NOT one per session).
Each record is a (context, target) pair where context = system prompt + a
sliding window of preceding turns and target = a single assistant turn the
model should learn to produce. See docs/finetune/hermes-finetune-design-spec.md
§1.2 for the rationale behind turn-based granularity.

Output format is Axolotl-compatible ShareGPT chat_template JSONL. Train/eval
split is deterministic by session_id hash so all turns from the same session
stay together.

Usage:
    python format.py [--input PATH] [--output-dir PATH] [--eval-ratio 0.1]
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    SCORED_DIR, CLUSTERS_DIR,
    ensure_dirs, load_config, read_jsonl, append_jsonl,
    content_to_text, load_records_dedup, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Canonical system prompt for training data — stripped of ephemeral injections
DEFAULT_SYSTEM_PROMPT = (
    "You are Hermes, a helpful AI assistant made by Nous Research. "
    "You are helpful, harmless, and honest."
)

# Patterns to strip from system prompts during canonicalization
EPHEMERAL_PATTERNS = [
    re.compile(r"<skills_context>.*?</skills_context>", re.DOTALL),
    re.compile(r"<memory_context>.*?</memory_context>", re.DOTALL),
    re.compile(r"<honcho_context>.*?</honcho_context>", re.DOTALL),
    re.compile(r"<context_files>.*?</context_files>", re.DOTALL),
    re.compile(r"Current date and time:.*?\n", re.I),
    re.compile(r"Available tools:.*?\n", re.I),
]


# ── Secret redaction ─────────────────────────────────────────────────────
#
# Conservative pass over all message content before it is written into
# training records. Only obviously-secret shapes are replaced — the goal is
# to never leak a live credential into train.jsonl, without mangling
# ordinary code.

_REDACTED = "[REDACTED]"

_SECRET_PATTERNS: List = [
    # PEM private key blocks
    (re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ), _REDACTED),
    # AWS access key IDs
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _REDACTED),
    # OpenAI-style API keys (sk-..., also sk-proj-...)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), _REDACTED),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), _REDACTED),
    # Slack tokens
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), _REDACTED),
    # JWTs (three base64url segments starting with eyJ)
    (re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b"
    ), _REDACTED),
    # Authorization: Bearer <long-token>
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]{20,}=*"),
     r"\1 " + _REDACTED),
    # password= / token= / api_key= style assignments. Quoted values of 6+
    # chars, or unquoted secret-looking runs of 8+ chars. The trailing
    # (?![\w(]) keeps call expressions like `token = get_token()` intact.
    (re.compile(
        r"(?i)\b(password|passwd|api[_-]?key|apikey|secret[_-]?key|"
        r"access[_-]?token|auth[_-]?token|token|secret)\b"
        r"(\s*[:=]\s*)"
        r"(\"[^\"]{6,}\"|'[^']{6,}'|[A-Za-z0-9_\-/+]{8,}(?![\w(]))"
    ), r"\1\2" + _REDACTED),
]


def redact_secrets(text: str) -> str:
    """Replace obvious secrets (keys, tokens, passwords) with [REDACTED]."""
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _session_hash_bucket(session_id: str, eval_ratio: float = 0.1) -> str:
    """Deterministic train/eval split by session ID hash."""
    h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
    return "eval" if (h % 1000) < (eval_ratio * 1000) else "train"


def _canonicalize_system_prompt(content: str) -> str:
    """Strip ephemeral injections from system prompt for training consistency."""
    if not content:
        return DEFAULT_SYSTEM_PROMPT

    result = content
    for pattern in EPHEMERAL_PATTERNS:
        result = pattern.sub("", result)

    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    # If stripping left almost nothing, use default
    if len(result) < 20:
        return DEFAULT_SYSTEM_PROMPT

    return result


def _normalize_reasoning(content: str) -> str:
    """Convert reasoning scratchpad tags to think tags for training."""
    if not content:
        return content
    content = content.replace("<REASONING_SCRATCHPAD>", "<think>")
    content = content.replace("</REASONING_SCRATCHPAD>", "</think>")
    return content


def _turn_to_sharegpt(turn: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Convert a single session turn to a ShareGPT-format conversation entry.

    Multipart list content is flattened to text, and all values pass through
    the conservative secret-redaction filter before being emitted.
    """
    role = turn.get("role")
    content = content_to_text(turn.get("content"))

    entry: Optional[Dict[str, str]] = None
    if role == "system":
        entry = {"from": "system", "value": _canonicalize_system_prompt(content)}

    elif role == "user":
        entry = {"from": "human", "value": content}

    elif role == "assistant":
        content = _normalize_reasoning(content)
        if turn.get("tool_calls"):
            tool_parts = []
            for tc in turn["tool_calls"]:
                func = tc.get("function", tc)
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_parts.append(f"<tool_call>\n{name}\n{args}\n</tool_call>")
            content = (content + "\n" + "\n".join(tool_parts)) if content else "\n".join(tool_parts)
        entry = {"from": "gpt", "value": content}

    elif role == "tool":
        tool_name = turn.get("tool_name", "")
        prefix = f"<tool_response>\n{tool_name}\n" if tool_name else "<tool_response>\n"
        entry = {"from": "tool", "value": f"{prefix}{content}\n</tool_response>"}

    if entry is not None:
        entry["value"] = redact_secrets(entry["value"])
    return entry


def extract_training_turns(
    session: Dict[str, Any],
    context_window_turns: int = 8,
    min_turn_score: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Extract per-turn training examples from a single session.

    Each output record is a (context, target_turn) pair where:
      - context = system prompt + sliding window of preceding turns
      - target = a single assistant turn the model should learn to produce

    Only assistant turns whose score meets `min_turn_score` are emitted.
    Per-turn scores come from the scoring step (composite turn score) and
    are overridden by retro labels via the score.py per-turn override path.

    Args:
        session: A scored session dict (must have 'scoring.turn_scores' or
                 fall back to a flat composite score).
        context_window_turns: Maximum number of preceding turns to include
                              in each example's context (excluding system).
        min_turn_score: Skip turns whose effective score is below this.

    Returns:
        List of training records in ShareGPT format. Each record's
        conversation list ends on a `gpt` turn so axolotl can train on it.
    """
    turns = session.get("turns", [])
    if not turns:
        return []

    session_id = session.get("session_id", "")
    scoring = session.get("scoring", {})

    # Map message-index → turn score. Build from scoring.turn_scores which
    # is List[Tuple[msg_idx, score]] (the format score.py emits).
    raw_turn_scores = scoring.get("turn_scores", [])
    turn_score_by_msg_idx: Dict[int, float] = {}
    for entry in raw_turn_scores:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            try:
                turn_score_by_msg_idx[int(entry[0])] = float(entry[1])
            except (TypeError, ValueError):
                continue

    # Fallback: if there are no per-turn scores, use the composite score
    # for every assistant turn (worse signal but still functional).
    composite_fallback = scoring.get("composite_score", 0.5)

    # Turns explicitly labeled bad (retro) are excluded unconditionally,
    # regardless of the min_turn_score threshold.
    bad_turn_indices = set(scoring.get("bad_turn_indices") or [])

    # Find the system turn (if any) — always pinned at the start of context
    system_entry: Optional[Dict[str, str]] = None
    if turns and turns[0].get("role") == "system":
        system_entry = _turn_to_sharegpt(turns[0])

    examples: List[Dict[str, Any]] = []

    for i, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue

        # Explicit bad label — never train on it, whatever the threshold
        if i in bad_turn_indices:
            continue

        # Effective score for this assistant turn
        effective_score = turn_score_by_msg_idx.get(i, composite_fallback)
        if effective_score < min_turn_score:
            continue

        # Build the context window: up to context_window_turns preceding
        # turns (any roles), ending just before this assistant turn.
        # Skip the system turn if it's in the window — we add it explicitly.
        window_start = max(0, i - context_window_turns)
        if turns[window_start].get("role") == "system":
            window_start += 1
        context_turns = turns[window_start:i]

        target_entry = _turn_to_sharegpt(turn)
        if target_entry is None or target_entry.get("from") != "gpt":
            continue

        # The target must have non-empty content; otherwise there's
        # nothing for axolotl to compute loss against.
        if not (target_entry.get("value") or "").strip():
            continue

        conversations: List[Dict[str, str]] = []
        if system_entry:
            conversations.append(system_entry)
        else:
            # Synthetic system prompt so every example has consistent priming
            conversations.append({"from": "system", "value": DEFAULT_SYSTEM_PROMPT})

        for ctx_turn in context_turns:
            entry = _turn_to_sharegpt(ctx_turn)
            if entry is not None:
                conversations.append(entry)

        # Must have at least one human turn before the target — otherwise
        # there's no prompt for the model to respond to.
        if not any(c["from"] == "human" for c in conversations):
            continue

        conversations.append(target_entry)

        examples.append({
            "conversations": conversations,
            "session_id": session_id,
            "turn_index_in_session": i,
            "score": effective_score,
        })

    return examples


# Backwards-compat alias for any external callers (and existing tests).
# Returns the FIRST training example produced by the new extractor — only
# useful for sessions where the new extractor produces exactly one record.
def format_session_chatml(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Deprecated: use extract_training_turns(). Kept for backwards compatibility."""
    examples = extract_training_turns(session, min_turn_score=0.0)
    return examples[0] if examples else None


class TrainingFormatter:
    """Format scored sessions into training-ready data."""

    def __init__(self, config: dict = None, eval_ratio: float = 0.1):
        self.config = config or load_config().get("training", {})
        self.eval_ratio = eval_ratio
        # Per-turn extraction knobs (with sane defaults if not in config)
        self.context_window_turns = int(self.config.get("context_window_turns", 8))
        self.min_turn_score = float(self.config.get("min_turn_score", 0.7))

    def format_for_cluster(
        self,
        sessions: List[Dict[str, Any]],
        cluster_id: str,
        output_dir: Path = None,
        min_score: float = None,
    ) -> Dict[str, int]:
        """
        Format sessions for a specific cluster, splitting into train/eval.

        With turn-based extraction, each session may produce zero or many
        training records — one per assistant turn whose effective score
        meets `min_turn_score`. Train/eval split is still keyed on
        `session_id` so all turns from a single session land in the same
        split (no context leakage).

        Args:
            sessions: Scored sessions assigned to this cluster.
            cluster_id: Cluster identifier (e.g., "c-a7f3e2" or "_general").
            output_dir: Override output directory.
            min_score: Minimum per-turn score to include. Falls back to
                       `self.min_turn_score` (from config) when None.

        Returns:
            Dict with 'train' and 'eval' counts (counting *training records*,
            not sessions).
        """
        ensure_dirs()
        out = output_dir or CLUSTERS_DIR / cluster_id
        out.mkdir(parents=True, exist_ok=True)

        threshold = self.min_turn_score if min_score is None else min_score

        train_records = []
        eval_records = []
        sessions_with_data = 0

        for session in sessions:
            examples = extract_training_turns(
                session,
                context_window_turns=self.context_window_turns,
                min_turn_score=threshold,
            )
            if not examples:
                continue
            sessions_with_data += 1

            split = _session_hash_bucket(
                session.get("session_id", ""), self.eval_ratio
            )
            if split == "eval":
                eval_records.extend(examples)
            else:
                train_records.extend(examples)

        # Write output
        train_path = out / "train.jsonl"
        eval_path = out / "eval.jsonl"

        # Overwrite (not append) — each format run produces a fresh split.
        # Always write, even when there are zero records: a run that filters
        # everything out must truncate stale train/eval files from previous
        # runs so they can't be silently trained on.
        for path, records in ((train_path, train_records), (eval_path, eval_records)):
            path.write_text(
                "".join(
                    json.dumps(r, ensure_ascii=False) + "\n" for r in records
                ),
                encoding="utf-8",
            )

        counts = {"train": len(train_records), "eval": len(eval_records)}
        logger.info(
            "Formatted cluster %s: %d sessions → train=%d turns, eval=%d turns "
            "(min_turn_score=%.2f, context_window=%d)",
            cluster_id, sessions_with_data,
            counts["train"], counts["eval"],
            threshold, self.context_window_turns,
        )
        return counts

    def format_all_scored(
        self, scored_sessions: List[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Format all scored sessions into the _general cluster.

        Use this before clustering is available, or for sessions
        not assigned to any cluster.
        """
        if scored_sessions is None:
            # Dedupe by session_id, keeping the newest snapshot's record
            scored_sessions = load_records_dedup(SCORED_DIR, "scored_*.jsonl")

        return self.format_for_cluster(scored_sessions, "_general")


def main():
    parser = argparse.ArgumentParser(description="Format scored sessions for training")
    parser.add_argument("--input", type=str, default=None,
                        help="Input scored JSONL (default: all scored)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: clusters/_general/)")
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Fraction held out for eval (default: 0.1)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Minimum per-turn quality score to include "
                             "(default: training.min_turn_score from config)")
    args = parser.parse_args()

    formatter = TrainingFormatter(eval_ratio=args.eval_ratio)

    if args.input:
        sessions = read_jsonl(Path(args.input))
    else:
        # Dedupe by session_id, keeping the newest snapshot's record
        sessions = load_records_dedup(SCORED_DIR, "scored_*.jsonl")

    if not sessions:
        print("No scored sessions found. Run score.py first.")
        return

    out_dir = Path(args.output_dir) if args.output_dir else None
    counts = formatter.format_for_cluster(
        sessions, "_general", output_dir=out_dir, min_score=args.min_score,
    )
    print(f"Formatted: train={counts['train']}, eval={counts['eval']}")


if __name__ == "__main__":
    main()
