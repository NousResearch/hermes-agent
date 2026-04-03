#!/usr/bin/env python3
"""Trajectory pipeline: raw conversation traces → Atropos-ready training data.

Reads trajectory_samples.jsonl (ShareGPT format from agent/trajectory.py),
applies quality filtering and deduplication, then outputs training data in
formats suitable for Atropos SFT and DPO training.

Usage:
    python scripts/trajectory_pipeline.py [options]

    # Basic: filter and convert
    python scripts/trajectory_pipeline.py --input trajectory_samples.jsonl --output-dir ./training_data

    # With quality scoring (slower, uses LLM)
    python scripts/trajectory_pipeline.py --score-quality --sample-rate 0.3

    # Include failed trajectories for DPO pairs
    python scripts/trajectory_pipeline.py --failed-input failed_trajectories.jsonl --output-dir ./training_data
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))


def load_trajectories(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL trajectory file."""
    entries = []
    path = Path(filepath)
    if not path.exists():
        logger.warning("File not found: %s", filepath)
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_num, e)
    return entries


def count_turns(trajectory: Dict[str, Any]) -> int:
    """Count user turns in a trajectory."""
    conversations = trajectory.get("conversations", [])
    return sum(1 for msg in conversations if msg.get("from") == "human")


def compute_message_hash(trajectory: Dict[str, Any]) -> str:
    """Hash user messages for deduplication."""
    conversations = trajectory.get("conversations", [])
    user_messages = [
        msg.get("value", "")
        for msg in conversations
        if msg.get("from") == "human"
    ]
    combined = "\n---\n".join(user_messages)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def filter_trajectories(
    entries: List[Dict[str, Any]],
    min_turns: int = 3,
    completed_only: bool = True,
) -> List[Dict[str, Any]]:
    """Apply basic quality filters."""
    filtered = []
    for entry in entries:
        if completed_only and not entry.get("completed", False):
            continue
        if count_turns(entry) < min_turns:
            continue
        filtered.append(entry)
    return filtered


def deduplicate(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate trajectories based on user message content hash."""
    seen_hashes = set()
    unique = []
    for entry in entries:
        h = compute_message_hash(entry)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(entry)
    return unique


def to_atropos_sft(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a trajectory to Atropos SFT format.

    Atropos SFT expects:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "metadata": {...}
    }
    """
    conversations = trajectory.get("conversations", [])
    messages = []

    for msg in conversations:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        role = role_map.get(msg.get("from", ""), msg.get("from", ""))
        content = msg.get("value", "")
        if role and content:
            messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "metadata": {
            "model": trajectory.get("model", "unknown"),
            "timestamp": trajectory.get("timestamp", ""),
            "source": "hermes-trajectory-pipeline",
        },
    }


def to_atropos_dpo(
    good_trajectory: Dict[str, Any],
    bad_trajectory: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Create a DPO pair from a good and bad trajectory.

    DPO format:
    {
        "prompt": [user messages leading to divergence],
        "chosen": "good assistant response",
        "rejected": "bad assistant response",
        "metadata": {...}
    }
    """
    good_convs = good_trajectory.get("conversations", [])
    bad_convs = bad_trajectory.get("conversations", [])

    if not good_convs or not bad_convs:
        return None

    # Find first user message (prompt)
    prompt_messages = []
    for msg in good_convs:
        if msg.get("from") == "human":
            prompt_messages.append({
                "role": "user",
                "content": msg.get("value", ""),
            })
            break

    if not prompt_messages:
        return None

    # Get first assistant response from each
    good_response = ""
    for msg in good_convs:
        if msg.get("from") == "gpt":
            good_response = msg.get("value", "")
            break

    bad_response = ""
    for msg in bad_convs:
        if msg.get("from") == "gpt":
            bad_response = msg.get("value", "")
            break

    if not good_response or not bad_response:
        return None
    if good_response == bad_response:
        return None

    return {
        "prompt": prompt_messages,
        "chosen": good_response,
        "rejected": bad_response,
        "metadata": {
            "good_model": good_trajectory.get("model", "unknown"),
            "bad_model": bad_trajectory.get("model", "unknown"),
            "source": "hermes-trajectory-pipeline",
        },
    }


def write_jsonl(entries: List[Dict[str, Any]], filepath: Path) -> int:
    """Write entries as JSONL. Returns count written."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Trajectory pipeline")
    parser.add_argument("--input", type=str, default="trajectory_samples.jsonl",
                        help="Input trajectory file (default: trajectory_samples.jsonl)")
    parser.add_argument("--failed-input", type=str, default=None,
                        help="Failed trajectories file for DPO pairs")
    parser.add_argument("--output-dir", type=str, default="./training_data",
                        help="Output directory (default: ./training_data)")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="Minimum user turns per trajectory (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load
    logger.info("Loading trajectories from %s...", args.input)
    good_entries = load_trajectories(args.input)
    logger.info("  Loaded: %d trajectories", len(good_entries))

    if not good_entries:
        logger.error("No trajectories found. Generate some via normal Hermes usage first.")
        sys.exit(1)

    # Filter
    filtered = filter_trajectories(good_entries, min_turns=args.min_turns)
    logger.info("  After filtering (min_turns=%d, completed=True): %d", args.min_turns, len(filtered))

    # Deduplicate
    unique = deduplicate(filtered)
    logger.info("  After dedup: %d (removed %d duplicates)", len(unique), len(filtered) - len(unique))

    if args.dry_run:
        logger.info("\nDry run — no files written.")
        logger.info("Would produce:")
        logger.info("  SFT: %d examples → %s/sft_data.jsonl", len(unique), output_dir)
        if args.failed_input:
            failed = load_trajectories(args.failed_input)
            logger.info("  DPO: up to %d pairs → %s/dpo_data.jsonl", min(len(unique), len(failed)), output_dir)
        return

    # Convert to SFT format
    sft_entries = [to_atropos_sft(t) for t in unique]
    sft_count = write_jsonl(sft_entries, output_dir / "sft_data.jsonl")
    logger.info("  Wrote %d SFT examples to %s/sft_data.jsonl", sft_count, output_dir)

    # DPO pairs (if failed trajectories provided)
    if args.failed_input:
        failed_entries = load_trajectories(args.failed_input)
        logger.info("  Loaded %d failed trajectories for DPO", len(failed_entries))

        dpo_entries = []
        for good, bad in zip(unique, failed_entries):
            pair = to_atropos_dpo(good, bad)
            if pair:
                dpo_entries.append(pair)

        if dpo_entries:
            dpo_count = write_jsonl(dpo_entries, output_dir / "dpo_data.jsonl")
            logger.info("  Wrote %d DPO pairs to %s/dpo_data.jsonl", dpo_count, output_dir)
        else:
            logger.info("  No valid DPO pairs generated")

    # Summary
    logger.info("\nPipeline complete:")
    logger.info("  Input: %d trajectories", len(good_entries))
    logger.info("  Output: %d SFT examples", len(sft_entries))
    logger.info("  Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
