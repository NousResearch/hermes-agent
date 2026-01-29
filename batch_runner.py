#!/usr/bin/env python3
"""
Batch Agent Runner

This module provides parallel batch processing capabilities for running the agent
across multiple prompts from a dataset. It includes:
- Dataset loading and batching
- Parallel batch processing with multiprocessing
- Checkpointing for fault tolerance and resumption
- Trajectory saving in the proper format (from/value pairs)
- Tool usage statistics aggregation across all batches

Usage:
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
    
    # Resume an interrupted run
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
    
    # Use a specific toolset distribution
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --distribution=image_gen
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, Manager, Lock
import traceback

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
import fire

from run_agent import AIAgent
from toolset_distributions import (
    get_distribution, 
    list_distributions, 
    sample_toolsets_from_distribution,
    validate_distribution
)


# Global configuration for worker processes
_WORKER_CONFIG = {}

# All possible tools - used to ensure consistent schema across all trajectory entries
# This is required because Arrow/Parquet (used by HuggingFace datasets) needs identical schemas
ALL_POSSIBLE_TOOLS = {
    'terminal', 'web_search', 'web_extract',
    'vision_analyze', 'image_generate', 'mixture_of_agents',
    # Browser automation tools
    'browser_navigate', 'browser_snapshot', 'browser_click',
    'browser_type', 'browser_scroll', 'browser_back',
    'browser_press', 'browser_close', 'browser_get_images',
    'browser_vision'
}

# Default stats for tools that weren't used
DEFAULT_TOOL_STATS = {'count': 0, 'success': 0, 'failure': 0}


def _normalize_tool_stats(tool_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """
    Normalize tool_stats to include all possible tools with consistent schema.
    
    This ensures HuggingFace datasets can load the JSONL without schema mismatch errors.
    Tools that weren't used get zero counts.
    
    Args:
        tool_stats (Dict): Raw tool statistics from extraction
        
    Returns:
        Dict: Normalized tool statistics with all tools present
    """
    normalized = {}
    
    # Add all possible tools with defaults
    for tool in ALL_POSSIBLE_TOOLS:
        if tool in tool_stats:
            normalized[tool] = tool_stats[tool].copy()
        else:
            normalized[tool] = DEFAULT_TOOL_STATS.copy()
    
    # Also include any unexpected tools (in case new tools are added)
    for tool, stats in tool_stats.items():
        if tool not in normalized:
            normalized[tool] = stats.copy()
    
    return normalized


def _normalize_tool_error_counts(tool_error_counts: Dict[str, int]) -> Dict[str, int]:
    """
    Normalize tool_error_counts to include all possible tools.
    
    Args:
        tool_error_counts (Dict): Raw error counts mapping
        
    Returns:
        Dict: Normalized error counts with all tools present
    """
    normalized = {}
    
    # Add all possible tools with zero defaults
    for tool in ALL_POSSIBLE_TOOLS:
        normalized[tool] = tool_error_counts.get(tool, 0)
    
    # Also include any unexpected tools
    for tool, count in tool_error_counts.items():
        if tool not in normalized:
            normalized[tool] = count
    
    return normalized


def _extract_tool_stats(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Extract tool usage statistics from message history.
    
    Args:
        messages (List[Dict]): Message history
        
    Returns:
        Dict: Tool statistics with counts and success/failure rates
    """
    tool_stats = {}
    
    # Track tool calls and their results
    tool_calls_map = {}  # Map tool_call_id to tool name
    
    for msg in messages:
        # Track tool calls from assistant messages
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                
                # Initialize stats for this tool if not exists
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                tool_stats[tool_name]["count"] += 1
                tool_calls_map[tool_call_id] = tool_name
        
        # Track tool responses
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            
            # Determine if tool call was successful
            is_success = True
            try:
                # Try to parse as JSON and check for actual error values
                content_json = json.loads(content) if isinstance(content, str) else content
                
                if isinstance(content_json, dict):
                    # Check if error field exists AND has a non-null value
                    if "error" in content_json and content_json["error"] is not None:
                        is_success = False
                    
                    # Special handling for terminal tool responses
                    # Terminal wraps its response in a "content" field
                    if "content" in content_json and isinstance(content_json["content"], dict):
                        inner_content = content_json["content"]
                        # Check for actual error (non-null error field)
                        # Note: non-zero exit codes are not failures - the model can self-correct
                        if inner_content.get("error") is not None:
                            is_success = False
                    
                    # Check for "success": false pattern used by some tools
                    if content_json.get("success") is False:
                        is_success = False
                        
            except:
                # If not JSON, check if content is empty or explicitly states an error
                # Note: We avoid simple substring matching to prevent false positives
                if not content:
                    is_success = False
                # Only mark as failure if it explicitly starts with "Error:" or "ERROR:"
                elif content.strip().lower().startswith("error:"):
                    is_success = False
            
            # Update success/failure count
            if tool_call_id in tool_calls_map:
                tool_name = tool_calls_map[tool_call_id]
                if is_success:
                    tool_stats[tool_name]["success"] += 1
                else:
                    tool_stats[tool_name]["failure"] += 1
    
    return tool_stats


def _process_single_prompt(
    prompt_index: int,
    prompt_data: Dict[str, Any],
    batch_num: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single prompt with the agent.
    
    Args:
        prompt_index (int): Index of prompt in dataset
        prompt_data (Dict): Prompt data containing 'prompt' field
        batch_num (int): Batch number
        config (Dict): Configuration dict with agent parameters
        
    Returns:
        Dict: Result containing trajectory, stats, and metadata
    """
    prompt = prompt_data["prompt"]
    
    try:
        # Sample toolsets from distribution for this prompt
        selected_toolsets = sample_toolsets_from_distribution(config["distribution"])
        
        if config.get("verbose"):
            print(f"   Prompt {prompt_index}: Using toolsets {selected_toolsets}")
        
        # Initialize agent with sampled toolsets and log prefix for identification
        log_prefix = f"[B{batch_num}:P{prompt_index}]"
        agent = AIAgent(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config["model"],
            max_iterations=config["max_iterations"],
            enabled_toolsets=selected_toolsets,
            save_trajectories=False,  # We handle saving ourselves
            verbose_logging=config.get("verbose", False),
            ephemeral_system_prompt=config.get("ephemeral_system_prompt"),
            log_prefix_chars=config.get("log_prefix_chars", 100),
            log_prefix=log_prefix,
            providers_allowed=config.get("providers_allowed"),
            providers_ignored=config.get("providers_ignored"),
            providers_order=config.get("providers_order"),
            provider_sort=config.get("provider_sort"),
        )

        # Run the agent with task_id to ensure each task gets its own isolated VM
        result = agent.run_conversation(prompt, task_id=f"task_{prompt_index}")
        
        # Extract tool usage statistics
        tool_stats = _extract_tool_stats(result["messages"])
        
        # Convert to trajectory format (using existing method)
        trajectory = agent._convert_to_trajectory_format(
            result["messages"],
            prompt,
            result["completed"]
        )
        
        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": trajectory,
            "tool_stats": tool_stats,
            "completed": result["completed"],
            "partial": result.get("partial", False),
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat(),
                "model": config["model"]
            }
        }
    
    except Exception as e:
        print(f"❌ Error processing prompt {prompt_index}: {e}")
        if config.get("verbose"):
            traceback.print_exc()
        
        return {
            "success": False,
            "prompt_index": prompt_index,
            "error": str(e),
            "trajectory": None,
            "tool_stats": {},
            "toolsets_used": [],
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat()
            }
        }


def _process_batch_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a single batch of prompts.
    
    Args:
        args (Tuple): (batch_num, batch_data, output_dir, completed_prompts, config)
        
    Returns:
        Dict: Batch results with statistics
    """
    batch_num, batch_data, output_dir, completed_prompts_set, config = args
    
    output_dir = Path(output_dir)
    print(f"\n🔄 Batch {batch_num}: Starting ({len(batch_data)} prompts)")
    
    # Output file for this batch
    batch_output_file = output_dir / f"batch_{batch_num}.jsonl"
    
    # Filter out already completed prompts
    prompts_to_process = [
        (idx, data) for idx, data in batch_data
        if idx not in completed_prompts_set
    ]
    
    if not prompts_to_process:
        print(f"✅ Batch {batch_num}: Already completed (skipping)")
        return {
            "batch_num": batch_num,
            "processed": 0,
            "skipped": len(batch_data),
            "tool_stats": {},
            "completed_prompts": []
        }
    
    print(f"   Processing {len(prompts_to_process)} prompts (skipping {len(batch_data) - len(prompts_to_process)} already completed)")
    
    # Initialize aggregated stats for this batch
    batch_tool_stats = {}
    completed_in_batch = []
    
    # Process each prompt sequentially in this batch
    for prompt_index, prompt_data in prompts_to_process:
        # Process the prompt
        result = _process_single_prompt(
            prompt_index,
            prompt_data,
            batch_num,
            config
        )
        
        # Save trajectory if successful
        if result["success"] and result["trajectory"]:
            # Get and normalize tool stats for consistent schema across all entries
            raw_tool_stats = result.get("tool_stats", {})
            tool_stats = _normalize_tool_stats(raw_tool_stats)
            
            # Create normalized tool_error_counts mapping tool names to their failure counts
            raw_error_counts = {
                tool_name: stats.get("failure", 0) 
                for tool_name, stats in raw_tool_stats.items()
            }
            tool_error_counts = _normalize_tool_error_counts(raw_error_counts)
            
            trajectory_entry = {
                "prompt_index": prompt_index,
                "conversations": result["trajectory"],
                "metadata": result["metadata"],
                "completed": result["completed"],
                "partial": result.get("partial", False),  # True if stopped due to invalid tool calls
                "api_calls": result["api_calls"],
                "toolsets_used": result["toolsets_used"],
                "tool_stats": tool_stats,  # Full stats: {tool: {count, success, failure}} - normalized
                "tool_error_counts": tool_error_counts  # Simple: {tool: failure_count} - normalized
            }
            
            # Append to batch output file
            with open(batch_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_entry, ensure_ascii=False) + "\n")
        
        # Aggregate tool statistics
        for tool_name, stats in result.get("tool_stats", {}).items():
            if tool_name not in batch_tool_stats:
                batch_tool_stats[tool_name] = {
                    "count": 0,
                    "success": 0,
                    "failure": 0
                }
            
            batch_tool_stats[tool_name]["count"] += stats["count"]
            batch_tool_stats[tool_name]["success"] += stats["success"]
            batch_tool_stats[tool_name]["failure"] += stats["failure"]
        
        # Only mark as completed if successfully saved (failed prompts can be retried on resume)
        if result["success"] and result["trajectory"]:
            completed_in_batch.append(prompt_index)
            status = "⚠️  partial" if result.get("partial") else "✅"
            print(f"   {status} Prompt {prompt_index} completed")
        else:
            print(f"   ❌ Prompt {prompt_index} failed (will retry on resume)")
    
    print(f"✅ Batch {batch_num}: Completed ({len(prompts_to_process)} prompts processed)")
    
    return {
        "batch_num": batch_num,
        "processed": len(prompts_to_process),
        "skipped": len(batch_data) - len(prompts_to_process),
        "tool_stats": batch_tool_stats,
        "completed_prompts": completed_in_batch
    }


class BatchRunner:
    """
    Manages batch processing of agent prompts with checkpointing and statistics.
    """
    
    def __init__(
        self,
        dataset_file: str,
        batch_size: int,
        run_name: str,
        distribution: str = "default",
        max_iterations: int = 10,
        base_url: str = None,
        api_key: str = None,
        model: str = "claude-opus-4-20250514",
        num_workers: int = 4,
        verbose: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
    ):
        """
        Initialize the batch runner.

        Args:
            dataset_file (str): Path to the dataset JSONL file with 'prompt' field
            batch_size (int): Number of prompts per batch
            run_name (str): Name for this run (used for checkpointing and output)
            distribution (str): Toolset distribution to use (default: "default")
            max_iterations (int): Max iterations per agent run
            base_url (str): Base URL for model API
            api_key (str): API key for model
            model (str): Model name to use
            num_workers (int): Number of parallel workers
            verbose (bool): Enable verbose logging
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
        """
        self.dataset_file = Path(dataset_file)
        self.batch_size = batch_size
        self.run_name = run_name
        self.distribution = distribution
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.num_workers = num_workers
        self.verbose = verbose
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.log_prefix_chars = log_prefix_chars
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        
        # Validate distribution
        if not validate_distribution(distribution):
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(list_distributions().keys())}")
        
        # Setup output directory
        self.output_dir = Path("data") / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        # Statistics file
        self.stats_file = self.output_dir / "statistics.json"
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Create batches
        self.batches = self._create_batches()
        
        print(f"📊 Batch Runner Initialized")
        print(f"   Dataset: {self.dataset_file} ({len(self.dataset)} prompts)")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total batches: {len(self.batches)}")
        print(f"   Run name: {self.run_name}")
        print(f"   Distribution: {self.distribution}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Workers: {self.num_workers}")
        if self.ephemeral_system_prompt:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"   🔒 Ephemeral system prompt: '{prompt_preview}'")
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load dataset from JSONL file.
        
        Returns:
            List[Dict]: List of dataset entries
        """
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        dataset = []
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    if 'prompt' not in entry:
                        print(f"⚠️  Warning: Line {line_num} missing 'prompt' field, skipping")
                        continue
                    dataset.append(entry)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not dataset:
            raise ValueError(f"No valid entries found in dataset file: {self.dataset_file}")
        
        return dataset
    
    def _create_batches(self) -> List[List[Tuple[int, Dict[str, Any]]]]:
        """
        Split dataset into batches with indices.
        
        Returns:
            List of batches, where each batch is a list of (index, entry) tuples
        """
        batches = []
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [(idx, entry) for idx, entry in enumerate(self.dataset[i:i + self.batch_size], start=i)]
            batches.append(batch)
        
        return batches
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint data if it exists.
        
        Returns:
            Dict: Checkpoint data with completed prompt indices
        """
        if not self.checkpoint_file.exists():
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any], lock: Optional[Lock] = None):
        """
        Save checkpoint data.
        
        Args:
            checkpoint_data (Dict): Checkpoint data to save
            lock (Lock): Optional lock for thread-safe access
        """
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        if lock:
            with lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _scan_completed_prompts_by_content(self) -> set:
        """
        Scan all batch files and extract completed prompts by their actual content.
        
        This provides a more robust resume mechanism that matches on prompt text
        rather than indices, allowing recovery even if indices don't match.
        
        Returns:
            set: Set of prompt texts that have been successfully processed
        """
        completed_prompts = set()
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        if not batch_files:
            return completed_prompts
        
        print(f"📂 Scanning {len(batch_files)} batch files for completed prompts...")
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Skip failed entries - we want to retry these
                            if entry.get("failed", False):
                                continue
                            
                            # Extract the human/user prompt from conversations
                            conversations = entry.get("conversations", [])
                            for msg in conversations:
                                if msg.get("from") == "human":
                                    prompt_text = msg.get("value", "").strip()
                                    if prompt_text:
                                        completed_prompts.add(prompt_text)
                                    break  # Only need the first human message
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"  ⚠️  Warning: Error reading {batch_file.name}: {e}")
        
        return completed_prompts
    
    def _filter_dataset_by_completed(self, completed_prompts: set) -> Tuple[List[Dict], List[int]]:
        """
        Filter the dataset to exclude prompts that have already been completed.
        
        Args:
            completed_prompts: Set of prompt texts that have been completed
            
        Returns:
            Tuple of (filtered_dataset, skipped_indices)
        """
        filtered_dataset = []
        skipped_indices = []
        
        for idx, entry in enumerate(self.dataset):
            # Extract prompt from the dataset entry
            prompt_text = entry.get("prompt", "").strip()
            
            # Also check conversations format
            if not prompt_text:
                conversations = entry.get("conversations", [])
                for msg in conversations:
                    role = msg.get("role") or msg.get("from")
                    if role in ("user", "human"):
                        prompt_text = (msg.get("content") or msg.get("value", "")).strip()
                        break
            
            if prompt_text in completed_prompts:
                skipped_indices.append(idx)
            else:
                # Keep original index for tracking
                filtered_dataset.append((idx, entry))
        
        return filtered_dataset, skipped_indices
    
    def run(self, resume: bool = False):
        """
        Run the batch processing pipeline.
        
        Args:
            resume (bool): Whether to resume from checkpoint
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Batch Processing")
        print("=" * 70)
        
        # Smart resume: scan batch files by content to find completed prompts
        completed_prompt_texts = set()
        if resume:
            completed_prompt_texts = self._scan_completed_prompts_by_content()
            if completed_prompt_texts:
                print(f"   Found {len(completed_prompt_texts)} already-completed prompts by content matching")
        
        # Filter dataset to only include unprocessed prompts
        if resume and completed_prompt_texts:
            filtered_entries, skipped_indices = self._filter_dataset_by_completed(completed_prompt_texts)
            
            if not filtered_entries:
                print("\n✅ All prompts have already been processed!")
                return
            
            # Recreate batches from filtered entries (keeping original indices for tracking)
            batches_to_process = []
            for i in range(0, len(filtered_entries), self.batch_size):
                batch = filtered_entries[i:i + self.batch_size]
                batches_to_process.append(batch)
            
            self.batches = batches_to_process
            
            # Print prominent resume summary
            print("\n" + "=" * 70)
            print("📊 RESUME SUMMARY")
            print("=" * 70)
            print(f"   Original dataset size:     {len(self.dataset):,} prompts")
            print(f"   Already completed:         {len(skipped_indices):,} prompts")
            print(f"   ─────────────────────────────────────────")
            print(f"   🎯 RESUMING WITH:          {len(filtered_entries):,} prompts")
            print(f"   New batches created:       {len(batches_to_process)}")
            print("=" * 70 + "\n")
        
        # Initialize checkpoint data (needed for saving at the end)
        checkpoint_data = {
            "run_name": self.run_name,
            "completed_prompts": [],
            "batch_stats": {},
            "last_updated": None
        }
        
        # Prepare configuration for workers
        config = {
            "distribution": self.distribution,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "log_prefix_chars": self.log_prefix_chars,
            "providers_allowed": self.providers_allowed,
            "providers_ignored": self.providers_ignored,
            "providers_order": self.providers_order,
            "provider_sort": self.provider_sort,
        }
        
        # For backward compatibility, still track by index (but this is secondary to content matching)
        completed_prompts_set = set()
        
        # Aggregate statistics across all batches
        total_tool_stats = {}
        
        start_time = time.time()
        
        print(f"\n🔧 Initializing {self.num_workers} worker processes...")
        
        # Process batches in parallel
        with Pool(processes=self.num_workers) as pool:
            # Create tasks for each batch
            tasks = [
                (
                    batch_num,
                    batch_data,
                    str(self.output_dir),  # Convert Path to string for pickling
                    completed_prompts_set,
                    config
                )
                for batch_num, batch_data in enumerate(self.batches)
            ]
            
            print(f"✅ Created {len(tasks)} batch tasks")
            print(f"🚀 Starting parallel batch processing...\n")
            
            # Use rich Progress for better visual tracking with persistent bottom bar
            # redirect_stdout/stderr lets rich manage all output so progress bar stays clean
            results = []
            console = Console(force_terminal=True)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]📦 Batches"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as progress:
                task = progress.add_task("Processing", total=len(tasks))
                
                # Temporarily suppress DEBUG logging to avoid bar interference
                root_logger = logging.getLogger()
                original_level = root_logger.level
                root_logger.setLevel(logging.WARNING)
                
                try:
                    for result in pool.imap_unordered(_process_batch_worker, tasks):
                        results.append(result)
                        progress.update(task, advance=1)
                finally:
                    root_logger.setLevel(original_level)
        
        # Aggregate all batch statistics and update checkpoint
        all_completed_prompts = list(completed_prompts_set)
        for batch_result in results:
            # Add newly completed prompts
            all_completed_prompts.extend(batch_result.get("completed_prompts", []))
            
            # Aggregate tool stats
            for tool_name, stats in batch_result.get("tool_stats", {}).items():
                if tool_name not in total_tool_stats:
                    total_tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                total_tool_stats[tool_name]["count"] += stats["count"]
                total_tool_stats[tool_name]["success"] += stats["success"]
                total_tool_stats[tool_name]["failure"] += stats["failure"]
        
        # Save final checkpoint
        checkpoint_data["completed_prompts"] = all_completed_prompts
        self._save_checkpoint(checkpoint_data)
        
        # Calculate success rates
        for tool_name in total_tool_stats:
            stats = total_tool_stats[tool_name]
            total_calls = stats["success"] + stats["failure"]
            if total_calls > 0:
                stats["success_rate"] = round(stats["success"] / total_calls * 100, 2)
                stats["failure_rate"] = round(stats["failure"] / total_calls * 100, 2)
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
        
        # Combine ALL batch files in directory into a single trajectories.jsonl file
        # This includes both old batches (from previous runs) and new batches (from resume)
        # Also filter out corrupted entries (where model generated invalid tool names)
        combined_file = self.output_dir / "trajectories.jsonl"
        print(f"\n📦 Combining ALL batch files into {combined_file.name}...")
        
        VALID_TOOLS = {'web_search', 'web_extract', 'terminal', 'vision_analyze', 
                       'image_generate', 'mixture_of_agents',
                       # Browser automation tools
                       'browser_navigate', 'browser_snapshot', 'browser_click',
                       'browser_type', 'browser_scroll', 'browser_back',
                       'browser_press', 'browser_close', 'browser_get_images',
                       'browser_vision'}
        
        total_entries = 0
        filtered_entries = 0
        batch_files_found = 0
        
        # Find ALL batch files in the output directory (handles resume merging old + new)
        all_batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for batch_file in all_batch_files:
                batch_files_found += 1
                batch_num = batch_file.stem.split("_")[1]  # Extract batch number for logging
                
                with open(batch_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        total_entries += 1
                        try:
                            data = json.loads(line)
                            tool_stats = data.get('tool_stats', {})
                            
                            # Check for invalid tool names (model hallucinations)
                            invalid_tools = [k for k in tool_stats.keys() if k not in VALID_TOOLS]
                            
                            if invalid_tools:
                                filtered_entries += 1
                                invalid_preview = invalid_tools[0][:50] + "..." if len(invalid_tools[0]) > 50 else invalid_tools[0]
                                print(f"   ⚠️  Filtering corrupted entry (batch {batch_num}): invalid tool '{invalid_preview}'")
                                continue
                            
                            outfile.write(line)
                        except json.JSONDecodeError:
                            filtered_entries += 1
                            print(f"   ⚠️  Filtering invalid JSON entry (batch {batch_num})")
        
        if filtered_entries > 0:
            print(f"⚠️  Filtered {filtered_entries} corrupted entries out of {total_entries} total")
        print(f"✅ Combined {batch_files_found} batch files into trajectories.jsonl ({total_entries - filtered_entries} entries)")
        
        # Save final statistics
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "total_batches": len(self.batches),
            "batch_size": self.batch_size,
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "tool_statistics": total_tool_stats
        }
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Prompts processed this run: {sum(r.get('processed', 0) for r in results)}")
        print(f"✅ Total trajectories in merged file: {total_entries - filtered_entries}")
        print(f"✅ Total batch files merged: {batch_files_found}")
        print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")
        print(f"\n📈 Tool Usage Statistics:")
        print("-" * 70)
        
        if total_tool_stats:
            # Sort by count descending
            sorted_tools = sorted(
                total_tool_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            print(f"{'Tool Name':<25} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<12}")
            print("-" * 70)
            for tool_name, stats in sorted_tools:
                print(
                    f"{tool_name:<25} "
                    f"{stats['count']:<10} "
                    f"{stats['success']:<10} "
                    f"{stats['failure']:<10} "
                    f"{stats['success_rate']:.1f}%"
                )
        else:
            print("No tool calls were made during this run.")
        
        print(f"\n💾 Results saved to: {self.output_dir}")
        print(f"   - Trajectories: trajectories.jsonl (combined)")
        print(f"   - Individual batches: batch_*.jsonl (for debugging)")
        print(f"   - Statistics: {self.stats_file.name}")
        print(f"   - Checkpoint: {self.checkpoint_file.name}")


def main(
    dataset_file: str = None,
    batch_size: int = None,
    run_name: str = None,
    distribution: str = "default",
    model: str = "anthropic/claude-sonnet-4-20250514",
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    max_turns: int = 10,
    num_workers: int = 4,
    resume: bool = False,
    verbose: bool = False,
    list_distributions: bool = False,
    ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100,
    providers_allowed: str = None,
    providers_ignored: str = None,
    providers_order: str = None,
    provider_sort: str = None,
):
    """
    Run batch processing of agent prompts from a dataset.

    Args:
        dataset_file (str): Path to JSONL file with 'prompt' field in each entry
        batch_size (int): Number of prompts per batch
        run_name (str): Name for this run (used for output and checkpointing)
        distribution (str): Toolset distribution to use (default: "default")
        model (str): Model name to use (default: "claude-opus-4-20250514")
        api_key (str): API key for model authentication
        base_url (str): Base URL for model API
        max_turns (int): Maximum number of tool calling iterations per prompt (default: 10)
        num_workers (int): Number of parallel worker processes (default: 4)
        resume (bool): Resume from checkpoint if run was interrupted (default: False)
        verbose (bool): Enable verbose logging (default: False)
        list_distributions (bool): List available toolset distributions and exit
        ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
        providers_allowed (str): Comma-separated list of OpenRouter providers to allow (e.g. "anthropic,openai")
        providers_ignored (str): Comma-separated list of OpenRouter providers to ignore (e.g. "together,deepinfra")
        providers_order (str): Comma-separated list of OpenRouter providers to try in order (e.g. "anthropic,openai,google")
        provider_sort (str): Sort providers by "price", "throughput", or "latency" (OpenRouter only)
        
    Examples:
        # Basic usage
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
        
        # Resume interrupted run
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
        
        # Use specific distribution
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=image_test --distribution=image_gen
        
        # With ephemeral system prompt (not saved to dataset)
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --ephemeral_system_prompt="You are a helpful assistant focused on image generation."
        
        # List available distributions
        python batch_runner.py --list_distributions
    """
    # Handle list distributions
    if list_distributions:
        from toolset_distributions import list_distributions as get_all_dists, print_distribution_info
        
        print("📊 Available Toolset Distributions")
        print("=" * 70)
        
        all_dists = get_all_dists()
        for dist_name in sorted(all_dists.keys()):
            print_distribution_info(dist_name)
        
        print("\n💡 Usage:")
        print("  python batch_runner.py --dataset_file=data.jsonl --batch_size=10 \\")
        print("                         --run_name=my_run --distribution=<name>")
        return
    
    # Validate required arguments
    if not dataset_file:
        print("❌ Error: --dataset_file is required")
        return
    
    if not batch_size or batch_size < 1:
        print("❌ Error: --batch_size must be a positive integer")
        return
    
    if not run_name:
        print("❌ Error: --run_name is required")
        return
    
    # Parse provider preferences (comma-separated strings to lists)
    providers_allowed_list = [p.strip() for p in providers_allowed.split(",")] if providers_allowed else None
    providers_ignored_list = [p.strip() for p in providers_ignored.split(",")] if providers_ignored else None
    providers_order_list = [p.strip() for p in providers_order.split(",")] if providers_order else None
    
    # Initialize and run batch runner
    try:
        runner = BatchRunner(
            dataset_file=dataset_file,
            batch_size=batch_size,
            run_name=run_name,
            distribution=distribution,
            max_iterations=max_turns,
            base_url=base_url,
            api_key=api_key,
            model=model,
            num_workers=num_workers,
            verbose=verbose,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            providers_allowed=providers_allowed_list,
            providers_ignored=providers_ignored_list,
            providers_order=providers_order_list,
            provider_sort=provider_sort,
        )

        runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    fire.Fire(main)

