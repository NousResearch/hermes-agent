#!/usr/bin/env python3
"""
Training data formatter for the finetune pipeline.

Converts scored sessions to Axolotl-compatible chat_template JSONL format.
Handles ChatML tokenization, system prompt canonicalization, and train/eval splitting.

Usage:
    python format.py [--input PATH] [--output-dir PATH] [--eval-ratio 0.1]
"""

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    SCORED_DIR, CLUSTERS_DIR,
    ensure_dirs, load_config, read_jsonl, append_jsonl, logger,
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


def format_session_chatml(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a scored session to Axolotl chat_template format.

    Returns a dict with 'conversations' key in ShareGPT format,
    compatible with Axolotl's chat_template dataset type.
    """
    turns = session.get("turns", [])
    if not turns:
        return None

    conversations = []

    for turn in turns:
        role = turn["role"]
        content = turn.get("content", "") or ""

        if role == "system":
            content = _canonicalize_system_prompt(content)
            conversations.append({"from": "system", "value": content})

        elif role == "user":
            conversations.append({"from": "human", "value": content})

        elif role == "assistant":
            content = _normalize_reasoning(content)

            # Include tool calls in content for chat_template training
            if turn.get("tool_calls"):
                tool_parts = []
                for tc in turn["tool_calls"]:
                    func = tc.get("function", tc)
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    if isinstance(args, dict):
                        import json
                        args = json.dumps(args)
                    tool_parts.append(f"<tool_call>\n{name}\n{args}\n</tool_call>")

                if content:
                    content = content + "\n" + "\n".join(tool_parts)
                else:
                    content = "\n".join(tool_parts)

            conversations.append({"from": "gpt", "value": content})

        elif role == "tool":
            # Tool results as a special turn
            tool_name = turn.get("tool_name", "")
            prefix = f"<tool_response>\n{tool_name}\n" if tool_name else "<tool_response>\n"
            conversations.append({
                "from": "tool",
                "value": f"{prefix}{content}\n</tool_response>",
            })

    # Ensure we have at least one system prompt
    if not conversations or conversations[0]["from"] != "system":
        conversations.insert(0, {"from": "system", "value": DEFAULT_SYSTEM_PROMPT})

    # Ensure conversation has at least one human + gpt exchange
    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not has_human or not has_gpt:
        return None

    return {
        "conversations": conversations,
        "session_id": session.get("session_id", ""),
        "score": session.get("scoring", {}).get("composite_score", 0.5),
        "bucket": session.get("scoring", {}).get("bucket", "neutral"),
    }


class TrainingFormatter:
    """Format scored sessions into training-ready data."""

    def __init__(self, config: dict = None, eval_ratio: float = 0.1):
        self.config = config or load_config().get("training", {})
        self.eval_ratio = eval_ratio

    def format_for_cluster(
        self,
        sessions: List[Dict[str, Any]],
        cluster_id: str,
        output_dir: Path = None,
        min_score: float = 0.0,
    ) -> Dict[str, int]:
        """
        Format sessions for a specific cluster, splitting into train/eval.

        Args:
            sessions: Scored sessions assigned to this cluster.
            cluster_id: Cluster identifier (e.g., "c-a7f3e2" or "_general").
            output_dir: Override output directory.
            min_score: Minimum composite score to include (default: include all).

        Returns:
            Dict with 'train' and 'eval' counts.
        """
        ensure_dirs()
        out = output_dir or CLUSTERS_DIR / cluster_id
        out.mkdir(parents=True, exist_ok=True)

        train_records = []
        eval_records = []

        for session in sessions:
            score = session.get("scoring", {}).get("composite_score", 0.0)
            if score < min_score:
                continue

            formatted = format_session_chatml(session)
            if formatted is None:
                continue

            split = _session_hash_bucket(
                session.get("session_id", ""), self.eval_ratio
            )
            if split == "eval":
                eval_records.append(formatted)
            else:
                train_records.append(formatted)

        # Write output
        train_path = out / "train.jsonl"
        eval_path = out / "eval.jsonl"

        # Overwrite (not append) — each format run produces a fresh split
        if train_records:
            train_path.write_text(
                "\n".join(
                    __import__("json").dumps(r, ensure_ascii=False)
                    for r in train_records
                ) + "\n",
                encoding="utf-8",
            )
        if eval_records:
            eval_path.write_text(
                "\n".join(
                    __import__("json").dumps(r, ensure_ascii=False)
                    for r in eval_records
                ) + "\n",
                encoding="utf-8",
            )

        counts = {"train": len(train_records), "eval": len(eval_records)}
        logger.info(
            "Formatted cluster %s: train=%d, eval=%d",
            cluster_id, counts["train"], counts["eval"],
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
            scored_sessions = []
            for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
                scored_sessions.extend(read_jsonl(path))

        return self.format_for_cluster(scored_sessions, "_general")


def main():
    parser = argparse.ArgumentParser(description="Format scored sessions for training")
    parser.add_argument("--input", type=str, default=None,
                        help="Input scored JSONL (default: all scored)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: clusters/_general/)")
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Fraction held out for eval (default: 0.1)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum quality score to include")
    args = parser.parse_args()

    formatter = TrainingFormatter(eval_ratio=args.eval_ratio)

    if args.input:
        sessions = read_jsonl(Path(args.input))
    else:
        sessions = []
        for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
            sessions.extend(read_jsonl(path))

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
