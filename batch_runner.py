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
- Cluster failure detection and graceful shutdown (morph, firecrawl, API errors)
- Configurable failure thresholds with automatic data consolidation

Usage:
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run

    # Resume an interrupted run
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume

    # Use a specific toolset distribution
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --distribution=image_gen

    # Configure tool failure thresholds
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                           --max_tool_failures=20 --max_tool_failure_rate=0.3
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
import re

import fire

from run_agent import AIAgent
from toolset_distributions import (
    get_distribution,
    list_distributions,
    sample_toolsets_from_distribution,
    validate_distribution
)
from safe_print import safe_print


# Global configuration for worker processes
_WORKER_CONFIG = {}


def _extract_tool_errors_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool errors from message history with tool names.

    Args:
        messages (List[Dict]): Message history

    Returns:
        List[Dict]: List of tool errors with tool name, error message, and context
    """
    tool_errors = []
    tool_calls_map = {}  # Map tool_call_id to tool name

    for msg in messages:
        # Track tool calls from assistant messages
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                tool_calls_map[tool_call_id] = tool_name

        # Check tool responses for errors
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")

            # Determine if tool call had an error
            has_error = False
            error_msg = None

            try:
                content_json = json.loads(content) if isinstance(content, str) else content

                if isinstance(content_json, dict):
                    # Check if error field exists AND has a non-null value
                    if "error" in content_json and content_json["error"] is not None:
                        has_error = True
                        error_msg = str(content_json["error"])

                    # Special handling for terminal tool responses
                    if "content" in content_json and isinstance(content_json["content"], dict):
                        inner_content = content_json["content"]
                        if inner_content.get("error") is not None or inner_content.get("exit_code", 0) != 0:
                            has_error = True
                            error_msg = inner_content.get("error") or f"Exit code: {inner_content.get('exit_code')}"

                    # Check for "success": false pattern
                    if content_json.get("success") is False:
                        has_error = True
                        if not error_msg:
                            error_msg = str(content_json.get("message", content_json.get("error", "Unknown error")))

            except:
                # If not JSON, check if content explicitly states an error
                if content.strip().lower().startswith("error:"):
                    has_error = True
                    error_msg = content.strip()

            # Record error if found
            if has_error and tool_call_id in tool_calls_map:
                tool_name = tool_calls_map[tool_call_id]
                tool_errors.append({
                    "tool_name": tool_name,
                    "error_message": error_msg or "Unknown error",
                    "full_content": content[:500]  # Keep first 500 chars of full response
                })

    return tool_errors


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
                        # Check for actual error (non-null error field or non-zero exit code)
                        has_error = (inner_content.get("error") is not None or 
                                   inner_content.get("exit_code", 0) != 0)
                        if has_error:
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
        
        # Initialize agent with sampled toolsets
        agent = AIAgent(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config["model"],
            max_iterations=config["max_iterations"],
            enabled_toolsets=selected_toolsets,
            save_trajectories=False,  # We handle saving ourselves
            verbose_logging=config.get("verbose", False),
            ephemeral_system_prompt=config.get("ephemeral_system_prompt"),
            log_prefix_chars=config.get("log_prefix_chars", 100)
        )

        # Run the agent with task_id to ensure each task gets its own isolated VM
        result = agent.run_conversation(prompt, task_id=f"task_{prompt_index}")

        # Extract tool usage statistics
        tool_stats = _extract_tool_stats(result["messages"])

        # Extract tool errors from conversation
        tool_errors = _extract_tool_errors_from_messages(result["messages"])

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
            "tool_errors": tool_errors,
            "completed": result["completed"],
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat(),
                "model": config["model"]
            }
        }
    
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        safe_print(f"[bold red]❌ Error processing prompt {prompt_index}:[/bold red] {error_msg}")
        if config.get("verbose"):
            safe_print(tb)

        return {
            "success": False,
            "prompt_index": prompt_index,
            "error": error_msg,
            "traceback": tb,
            "tool_errors": [],
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
    all_tool_errors = []  # Track all tool errors in this batch
    exception_errors = []  # Track top-level exceptions

    # Process each prompt sequentially in this batch
    for prompt_index, prompt_data in prompts_to_process:
        # Process the prompt
        result = _process_single_prompt(
            prompt_index,
            prompt_data,
            batch_num,
            config
        )

        # Track tool errors from the conversation
        if result.get("tool_errors"):
            for tool_error in result["tool_errors"]:
                all_tool_errors.append({
                    "prompt_index": prompt_index,
                    "tool_name": tool_error["tool_name"],
                    "error_message": tool_error["error_message"],
                    "full_content": tool_error.get("full_content", "")
                })

        # Track top-level exceptions (not tool errors)
        if not result["success"]:
            exception_errors.append({
                "prompt_index": prompt_index,
                "error": result.get("error", "Unknown error"),
                "traceback": result.get("traceback", "")
            })
            safe_print(f"[bold red]❌ Exception in prompt {prompt_index}:[/bold red] {result.get('error', '')[:100]}")

        # Save trajectory if successful
        if result["success"] and result["trajectory"]:
            trajectory_entry = {
                "prompt_index": prompt_index,
                "conversations": result["trajectory"],
                "metadata": result["metadata"],
                "completed": result["completed"],
                "api_calls": result["api_calls"],
                "toolsets_used": result["toolsets_used"]
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
        
        completed_in_batch.append(prompt_index)
        print(f"   ✅ Prompt {prompt_index} completed")
    
    print(f"✅ Batch {batch_num}: Completed ({len(prompts_to_process)} prompts processed)")
    
    return {
        "batch_num": batch_num,
        "processed": len(prompts_to_process),
        "skipped": len(batch_data) - len(prompts_to_process),
        "tool_stats": batch_tool_stats,
        "completed_prompts": completed_in_batch,
        "tool_errors": all_tool_errors,
        "exception_errors": exception_errors
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
        max_tool_failures: int = 10,
        max_tool_failure_rate: float = 0.5,
        keep_recent_errors: int = 5,
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
            max_tool_failures (int): Maximum number of tool failures before stopping (default: 10)
            max_tool_failure_rate (float): Maximum tool failure rate (0.0-1.0) before stopping (default: 0.5)
            keep_recent_errors (int): Number of recent errors to keep per tool (default: 5)
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
        self.max_tool_failures = max_tool_failures
        self.max_tool_failure_rate = max_tool_failure_rate
        self.keep_recent_errors = keep_recent_errors
        
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
        
        safe_print("[bold cyan]📊 Batch Runner Initialized[/bold cyan]")
        safe_print(f"   Dataset: {self.dataset_file} ({len(self.dataset)} prompts)")
        safe_print(f"   Batch size: {self.batch_size}")
        safe_print(f"   Total batches: {len(self.batches)}")
        safe_print(f"   Run name: {self.run_name}")
        safe_print(f"   Distribution: {self.distribution}")
        safe_print(f"   Output directory: {self.output_dir}")
        safe_print(f"   Workers: {self.num_workers}")
        safe_print(f"   [yellow]Tool failure limits:[/yellow]")
        safe_print(f"      Max failures: {self.max_tool_failures}")
        safe_print(f"      Max failure rate: {self.max_tool_failure_rate:.1%}")
        safe_print(f"      Keep recent errors: {self.keep_recent_errors}")
        if self.ephemeral_system_prompt:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            safe_print(f"   🔒 Ephemeral system prompt: '{prompt_preview}'")
    
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

    def _consolidate_data(self, num_batches: int, tool_stats: Dict[str, Dict[str, int]],
                          start_time: float, tool_errors_by_tool: Dict[str, List[Dict]],
                          exception_errors: List[Dict], early_exit: bool = False, exit_reason: str = None):
        """
        Consolidate batch data into trajectories.jsonl and save statistics.

        Args:
            num_batches (int): Number of batches processed
            tool_stats (Dict): Aggregated tool statistics
            start_time (float): Start time of the run
            tool_errors_by_tool (Dict): Tool errors grouped by tool name with k most recent
            exception_errors (List): Top-level exceptions
            early_exit (bool): Whether this is an early exit
            exit_reason (str): Reason for early exit
        """
        # Combine all batch files into a single trajectories.jsonl file
        combined_file = self.output_dir / "trajectories.jsonl"
        safe_print(f"\n[cyan]📦 Combining batch files into {combined_file.name}...[/cyan]")

        entries_written = 0
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for batch_num in range(num_batches):
                batch_file = self.output_dir / f"batch_{batch_num}.jsonl"
                if batch_file.exists():
                    with open(batch_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                            entries_written += 1

        safe_print(f"[green]✅ Combined {num_batches} batch files into trajectories.jsonl ({entries_written} entries)[/green]")

        # Calculate success rates for tool stats
        for tool_name in tool_stats:
            stats = tool_stats[tool_name]
            total_calls = stats["success"] + stats["failure"]
            if total_calls > 0:
                stats["success_rate"] = round(stats["success"] / total_calls * 100, 2)
                stats["failure_rate"] = round(stats["failure"] / total_calls * 100, 2)
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0

        # Save final statistics
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "total_batches": len(self.batches),
            "batches_processed": num_batches,
            "batch_size": self.batch_size,
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "early_exit": early_exit,
            "exit_reason": exit_reason,
            "tool_errors": tool_errors_by_tool,
            "exception_errors": exception_errors[:self.keep_recent_errors],  # Keep k most recent
            "tool_statistics": tool_stats
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
    
    
    def run(self, resume: bool = False):
        """
        Run the batch processing pipeline.
        
        Args:
            resume (bool): Whether to resume from checkpoint
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Batch Processing")
        print("=" * 70)
        
        # Load checkpoint
        checkpoint_data = self._load_checkpoint() if resume else {
            "run_name": self.run_name,
            "completed_prompts": [],
            "batch_stats": {},
            "last_updated": None
        }
        
        if resume and checkpoint_data.get("completed_prompts"):
            print(f"📂 Resuming from checkpoint ({len(checkpoint_data['completed_prompts'])} prompts already completed)")
        
        # Prepare configuration for workers
        config = {
            "distribution": self.distribution,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "log_prefix_chars": self.log_prefix_chars
        }
        
        # Get completed prompts set
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        
        # Aggregate statistics across all batches
        total_tool_stats = {}
        tool_errors_by_tool = {}  # {tool_name: [list of k most recent errors]}
        all_exception_errors = []
        all_completed_prompts = list(completed_prompts_set)
        total_processed = len(completed_prompts_set)
        total_tool_errors = 0
        early_exit = False
        exit_reason = None

        start_time = time.time()

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

            # Process batches and check tool failure threshold after each batch
            for batch_num, task in enumerate(tasks):
                # Process single batch
                result = pool.apply(_process_batch_worker, (task,))

                # Update statistics
                all_completed_prompts.extend(result.get("completed_prompts", []))
                total_processed += result.get("processed", 0)

                # Aggregate tool stats
                for tool_name, stats in result.get("tool_stats", {}).items():
                    if tool_name not in total_tool_stats:
                        total_tool_stats[tool_name] = {
                            "count": 0,
                            "success": 0,
                            "failure": 0
                        }

                    total_tool_stats[tool_name]["count"] += stats["count"]
                    total_tool_stats[tool_name]["success"] += stats["success"]
                    total_tool_stats[tool_name]["failure"] += stats["failure"]

                # Aggregate tool errors (keep k most recent per tool)
                for tool_error in result.get("tool_errors", []):
                    tool_name = tool_error["tool_name"]
                    if tool_name not in tool_errors_by_tool:
                        tool_errors_by_tool[tool_name] = []

                    # Add error and keep only k most recent
                    tool_errors_by_tool[tool_name].append(tool_error)
                    if len(tool_errors_by_tool[tool_name]) > self.keep_recent_errors:
                        tool_errors_by_tool[tool_name] = tool_errors_by_tool[tool_name][-self.keep_recent_errors:]

                    total_tool_errors += 1

                # Track exception errors
                all_exception_errors.extend(result.get("exception_errors", []))

                # Check tool failure thresholds
                if total_processed > 0:
                    tool_failure_rate = total_tool_errors / total_processed

                    # Check absolute count threshold
                    if total_tool_errors >= self.max_tool_failures:
                        early_exit = True
                        exit_reason = f"Exceeded maximum tool failures ({total_tool_errors}/{self.max_tool_failures})"
                        safe_print(f"\n[bold red]🛑 STOPPING: {exit_reason}[/bold red]")
                        break

                    # Check rate threshold
                    if tool_failure_rate >= self.max_tool_failure_rate:
                        early_exit = True
                        exit_reason = f"Exceeded tool failure rate ({tool_failure_rate:.2%} >= {self.max_tool_failure_rate:.2%})"
                        safe_print(f"\n[bold red]🛑 STOPPING: {exit_reason}[/bold red]")
                        break

                # Update checkpoint after each batch
                checkpoint_data["completed_prompts"] = all_completed_prompts
                self._save_checkpoint(checkpoint_data)

        # Save final checkpoint
        checkpoint_data["completed_prompts"] = all_completed_prompts
        self._save_checkpoint(checkpoint_data)

        # Consolidate data and save statistics
        num_batches_processed = batch_num + 1 if early_exit else len(self.batches)
        self._consolidate_data(
            num_batches_processed,
            total_tool_stats,
            start_time,
            tool_errors_by_tool,
            all_exception_errors,
            early_exit,
            exit_reason
        )
        
        # Print summary
        safe_print("\n" + "=" * 70)
        if early_exit:
            safe_print("[bold yellow]⚠️  BATCH PROCESSING STOPPED EARLY[/bold yellow]")
            safe_print(f"[yellow]Reason: {exit_reason}[/yellow]")
        else:
            safe_print("[bold green]📊 BATCH PROCESSING COMPLETE[/bold green]")
        safe_print("=" * 70)

        safe_print(f"✅ Total prompts processed: {total_processed}")
        safe_print(f"✅ Batches completed: {num_batches_processed}/{len(self.batches)}")
        safe_print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")

        # Tool error summary
        if tool_errors_by_tool:
            total_errors = sum(len(errors) for errors in tool_errors_by_tool.values())
            safe_print(f"\n[bold red]🚨 Tool Errors: {total_tool_errors} total ({len(tool_errors_by_tool)} tools)[/bold red]")
            safe_print("[red]-[/red]" * 70)

            # Sort tools by error count
            sorted_tools = sorted(
                tool_errors_by_tool.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            for tool_name, errors in sorted_tools:
                # Count unique error messages
                unique_errors = {}
                for error in errors:
                    error_msg = error["error_message"][:100]  # Truncate for grouping
                    if error_msg not in unique_errors:
                        unique_errors[error_msg] = []
                    unique_errors[error_msg].append(error)

                safe_print(f"\n  [red]{tool_name}:[/red] {len(errors)} errors ({len(unique_errors)} unique)")

                # Show up to 3 most recent unique error types
                for idx, (error_msg, instances) in enumerate(list(unique_errors.items())[:3]):
                    error_preview = error_msg if len(error_msg) <= 100 else error_msg[:97] + "..."
                    safe_print(f"    [{idx+1}] [dim]{error_preview}[/dim] (x{len(instances)})")
                    # Show one example with prompt index
                    example = instances[-1]  # Most recent
                    safe_print(f"        [dim]Prompt {example['prompt_index']}[/dim]")

                if len(unique_errors) > 3:
                    safe_print(f"    [dim]... and {len(unique_errors) - 3} more error types[/dim]")

            tool_failure_rate = total_tool_errors / total_processed if total_processed > 0 else 0
            safe_print(f"\n  [red]Tool failure rate: {tool_failure_rate:.2%}[/red]")

        # Exception errors
        if all_exception_errors:
            safe_print(f"\n[bold red]💥 Top-level Exceptions: {len(all_exception_errors)}[/bold red]")
            safe_print("[red]-[/red]" * 70)
            for error in all_exception_errors[:self.keep_recent_errors]:
                error_preview = error["error"][:100]
                if len(error["error"]) > 100:
                    error_preview += "..."
                safe_print(f"  Prompt {error['prompt_index']}: [dim]{error_preview}[/dim]")

        safe_print(f"\n[cyan]📈 Tool Usage Statistics:[/cyan]")
        safe_print("-" * 70)

        if total_tool_stats:
            # Sort by count descending
            sorted_tools = sorted(
                total_tool_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )

            safe_print(f"{'Tool Name':<25} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<12}")
            safe_print("-" * 70)
            for tool_name, stats in sorted_tools:
                safe_print(
                    f"{tool_name:<25} "
                    f"{stats['count']:<10} "
                    f"{stats['success']:<10} "
                    f"{stats['failure']:<10} "
                    f"{stats.get('success_rate', 0):.1f}%"
                )
        else:
            safe_print("No tool calls were made during this run.")

        safe_print(f"\n[cyan]💾 Results saved to:[/cyan] {self.output_dir}")
        safe_print(f"   - Trajectories: trajectories.jsonl (combined)")
        safe_print(f"   - Individual batches: batch_*.jsonl (for debugging)")
        safe_print(f"   - Statistics: {self.stats_file.name}")
        safe_print(f"   - Checkpoint: {self.checkpoint_file.name}")

        if early_exit:
            safe_print(f"\n[bold yellow]ℹ️  Run was stopped early due to tool failures.[/bold yellow]")
            safe_print(f"[yellow]   Check {self.stats_file.name} for detailed error information including tracebacks.[/yellow]")
            safe_print(f"[yellow]   You can resume this run later with --resume flag.[/yellow]")


def main(
    dataset_file: str = None,
    batch_size: int = None,
    run_name: str = None,
    distribution: str = "default",
    model: str = "claude-opus-4-20250514",
    api_key: str = None,
    base_url: str = "https://api.anthropic.com/v1/",
    max_turns: int = 10,
    num_workers: int = 4,
    resume: bool = False,
    verbose: bool = False,
    list_distributions: bool = False,
    ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100,
    max_tool_failures: int = 10,
    max_tool_failure_rate: float = 0.5,
    keep_recent_errors: int = 5,
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
        max_tool_failures (int): Maximum number of tool failures before stopping (default: 10)
        max_tool_failure_rate (float): Maximum tool failure rate (0.0-1.0) before stopping (default: 0.5)
        keep_recent_errors (int): Number of recent errors to keep per tool for reporting (default: 5)

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

        # With custom tool failure thresholds
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --max_tool_failures=20 --max_tool_failure_rate=0.3 --keep_recent_errors=10

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
            max_tool_failures=max_tool_failures,
            max_tool_failure_rate=max_tool_failure_rate,
            keep_recent_errors=keep_recent_errors
        )

        runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    fire.Fire(main)

