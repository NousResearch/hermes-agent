#!/usr/bin/env python3
"""
Sample and Compress HuggingFace Datasets

Downloads trajectories from multiple HuggingFace datasets, randomly samples them,
and runs trajectory compression to fit within a target token budget.

Usage:
    python scripts/sample_and_compress.py

    # Custom sample size
    python scripts/sample_and_compress.py --total_samples=5000

    # Custom output name
    python scripts/sample_and_compress.py --output_name=compressed_16k
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


# Default datasets to sample from
DEFAULT_DATASETS = [
    "NousResearch/swe-terminus-agent-glm-kimi-minimax",
    "NousResearch/hermes-agent-megascience-sft1",
    "NousResearch/Hermes-Agent-Thinking-GLM-4.7-SFT2",
    "NousResearch/Hermes-Agent-Thinking-GLM-4.7-SFT1",
    "NousResearch/terminal-tasks-glm-hermes-agent",
]


def load_dataset_from_hf(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "NousResearch/dataset-name")

    Returns:
        List of trajectory entries
    """
    from datasets import load_dataset

    print(f"   Loading {dataset_name}...")

    try:
        # Try loading with default config
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"   ⚠️  Error loading {dataset_name}: {e}")
        return []

    # Convert to list of dicts
    entries = []
    for item in ds:
        # Handle different possible formats
        if "conversations" in item:
            entries.append({"conversations": item["conversations"]})
        elif "messages" in item:
            # Convert messages format to conversations format if needed
            entries.append({"conversations": item["messages"]})
        else:
            # Assume the whole item is the entry
            entries.append(dict(item))

    print(f"   ✅ Loaded {len(entries):,} entries from {dataset_name}")
    return entries


# Global tokenizer for multiprocessing (set in worker init)
_TOKENIZER = None


def _init_tokenizer_worker(tokenizer_name: str):
    """Initialize tokenizer in worker process."""
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def _count_tokens_for_entry(entry: Dict) -> Tuple[Dict, int]:
    """
    Count tokens for a single entry (used in parallel processing).

    Args:
        entry: Trajectory entry with 'conversations' field

    Returns:
        Tuple of (entry, token_count)
    """
    global _TOKENIZER

    conversations = entry.get("conversations", [])
    if not conversations:
        return entry, 0

    total = 0
    for turn in conversations:
        value = turn.get("value", "")
        if value:
            try:
                total += len(_TOKENIZER.encode(value))
            except Exception:
                # Fallback to character estimate
                total += len(value) // 4

    return entry, total


def sample_from_datasets(
    datasets: List[str],
    total_samples: int,
    min_tokens: int = 16000,
    tokenizer_name: str = "moonshotai/Kimi-K2-Thinking",
    seed: int = 42,
    num_proc: int = 8,
) -> List[Dict[str, Any]]:
    """
    Load all datasets, filter by token count, then randomly sample from combined pool.

    Args:
        datasets: List of HuggingFace dataset names
        total_samples: Total number of samples to collect
        min_tokens: Minimum token count to include (only sample trajectories >= this)
        tokenizer_name: HuggingFace tokenizer for counting tokens
        seed: Random seed for reproducibility
        num_proc: Number of parallel processes for tokenization

    Returns:
        List of sampled trajectory entries
    """
    from functools import partial
    from multiprocessing import Pool

    random.seed(seed)

    print(f"\n📥 Loading {len(datasets)} datasets...")
    print(f"   Minimum tokens: {min_tokens:,} (filtering smaller trajectories)")
    print(f"   Parallel workers: {num_proc}")
    print()

    # Load ALL entries from all datasets into one pool
    all_entries = []

    for dataset_name in datasets:
        entries = load_dataset_from_hf(dataset_name)

        if not entries:
            print(f"   ⚠️  Skipping {dataset_name} (no entries loaded)")
            continue

        # Add source metadata to each entry
        for entry in entries:
            entry["_source_dataset"] = dataset_name

        all_entries.extend(entries)

    print(f"\n📊 Total entries loaded: {len(all_entries):,}")

    # Filter by token count using parallel processing
    print(f"\n🔍 Filtering trajectories with >= {min_tokens:,} tokens (using {num_proc} workers)...")

    filtered_entries = []
    token_counts = []

    # Use multiprocessing for token counting
    with Pool(processes=num_proc, initializer=_init_tokenizer_worker, initargs=(tokenizer_name,)) as pool:
        # Process in chunks and show progress
        chunk_size = 1000
        processed = 0

        for result in pool.imap_unordered(_count_tokens_for_entry, all_entries, chunksize=100):
            entry, token_count = result
            processed += 1

            if processed % chunk_size == 0:
                print(f"   Processed {processed:,}/{len(all_entries):,}...", end="\r")

            if token_count >= min_tokens:
                entry["_original_tokens"] = token_count
                filtered_entries.append(entry)
                token_counts.append(token_count)

    print(f"\n   ✅ Found {len(filtered_entries):,} trajectories >= {min_tokens:,} tokens")

    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"   📈 Token stats: min={min(token_counts):,}, max={max(token_counts):,}, avg={avg_tokens:,.0f}")

    # Random sample from the filtered pool
    if len(filtered_entries) <= total_samples:
        print(f"\n⚠️  Only {len(filtered_entries):,} trajectories available, using all of them")
        sampled = filtered_entries
    else:
        sampled = random.sample(filtered_entries, total_samples)
        print(f"\n✅ Randomly sampled {len(sampled):,} trajectories from pool of {len(filtered_entries):,}")

    # Show source distribution
    source_counts = {}
    for entry in sampled:
        source = entry.get("_source_dataset", "unknown").split("/")[-1]
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n📌 Sample distribution by source:")
    for source, count in sorted(source_counts.items()):
        print(f"      {source}: {count:,}")

    # Shuffle
    random.shuffle(sampled)

    return sampled


def save_samples_for_compression(samples: List[Dict[str, Any]], output_dir: Path, batch_size: int = 100):
    """
    Save samples to JSONL files for trajectory compression.

    Args:
        samples: List of trajectory entries
        output_dir: Directory to save JSONL files
        batch_size: Number of entries per file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into batches
    num_batches = (len(samples) + batch_size - 1) // batch_size

    print(f"\n💾 Saving {len(samples)} samples to {output_dir}")
    print(f"   Batch size: {batch_size}, Total batches: {num_batches}")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(samples))
        batch = samples[start_idx:end_idx]

        output_file = output_dir / f"batch_{i}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in batch:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"   ✅ Saved {num_batches} batch files")


def run_compression(input_dir: Path, output_dir: Path, config_path: str):
    """
    Run trajectory compression on the sampled data.

    Args:
        input_dir: Directory containing JSONL files to compress
        output_dir: Directory for compressed output
        config_path: Path to compression config YAML
    """
    # Import the compressor
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor

    print(f"\n🗜️  Running trajectory compression...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Config: {config_path}")

    # Load config
    config = CompressionConfig.from_yaml(config_path)

    # Initialize compressor
    compressor = TrajectoryCompressor(config)

    # Run compression
    compressor.process_directory(input_dir, output_dir)


def merge_output_to_single_jsonl(input_dir: Path, output_file: Path):
    """
    Merge all JSONL files in a directory into a single JSONL file.

    Args:
        input_dir: Directory containing JSONL files
        output_file: Output JSONL file path
    """
    print(f"\n📦 Merging output files into {output_file.name}...")

    all_entries = []
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        if jsonl_file.name == output_file.name:
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    # Write merged file
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"   ✅ Merged {len(all_entries):,} entries into {output_file.name}")
    return output_file


def main(
    total_samples: int = 2500,
    output_name: str = "compressed_agentic",
    datasets: str = None,
    config: str = "configs/trajectory_compression.yaml",
    seed: int = 42,
    batch_size: int = 100,
    min_tokens: int = 16000,
    num_proc: int = 8,
    skip_download: bool = False,
):
    """
    Sample trajectories from HuggingFace datasets and run compression.

    Args:
        total_samples: Total number of samples to collect (default: 2500)
        output_name: Name for output directory/file (default: "compressed_agentic")
        datasets: Comma-separated list of dataset names (uses defaults if not provided)
        config: Path to compression config YAML
        seed: Random seed for reproducibility
        batch_size: Number of entries per JSONL file during processing
        min_tokens: Minimum token count to filter trajectories (default: 16000)
        num_proc: Number of parallel workers for tokenization (default: 8)
        skip_download: Skip download and use existing sampled data
    """
    print("=" * 70)
    print("📊 TRAJECTORY SAMPLING AND COMPRESSION")
    print("=" * 70)

    # Parse datasets
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
    else:
        dataset_list = DEFAULT_DATASETS

    print(f"\n📋 Configuration:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Min tokens filter: {min_tokens:,}")
    print(f"   Parallel workers: {num_proc}")
    print(f"   Datasets: {len(dataset_list)}")
    for ds in dataset_list:
        print(f"      - {ds}")
    print(f"   Output name: {output_name}")
    print(f"   Config: {config}")
    print(f"   Seed: {seed}")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    sampled_dir = base_dir / "data" / f"{output_name}_raw"
    compressed_dir = base_dir / "data" / f"{output_name}_batches"
    final_output = base_dir / "data" / f"{output_name}.jsonl"

    if not skip_download:
        # Step 1: Download, filter by token count, and sample from combined pool
        samples = sample_from_datasets(dataset_list, total_samples, min_tokens=min_tokens, seed=seed, num_proc=num_proc)

        if not samples:
            print("❌ No samples collected. Exiting.")
            return

        # Step 2: Save to JSONL files
        save_samples_for_compression(samples, sampled_dir, batch_size)
    else:
        print(f"\n⏭️  Skipping download, using existing data in {sampled_dir}")

    # Step 3: Run compression
    config_path = base_dir / config
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    run_compression(sampled_dir, compressed_dir, str(config_path))

    # Step 4: Merge into single JSONL file
    merge_output_to_single_jsonl(compressed_dir, final_output)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Raw samples:        {sampled_dir}")
    print(f"📁 Compressed batches: {compressed_dir}")
    print(f"📁 Final output:       {final_output}")
    print(f"\nTo upload to HuggingFace:")
    print(f"   huggingface-cli upload NousResearch/{output_name} {final_output}")


if __name__ == "__main__":
    fire.Fire(main)
