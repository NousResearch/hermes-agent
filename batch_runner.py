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
                # Try to parse as JSON and check for error field
                content_json = json.loads(content) if isinstance(content, str) else content
                if isinstance(content_json, dict) and "error" in content_json:
                    is_success = False
            except:
                # If not JSON, check if content contains error indicators
                if not content or "error" in content.lower():
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
            verbose_logging=config.get("verbose", False)
        )
        
        # Run the agent
        result = agent.run_conversation(prompt)
        
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
        verbose: bool = False
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
                    json.dump(checkpoint_data, f, indent=2)
        else:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    
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
            "verbose": self.verbose
        }
        
        # Get completed prompts set
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        
        # Aggregate statistics across all batches
        total_tool_stats = {}
        
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
            
            # Use map to process batches in parallel
            results = pool.map(_process_batch_worker, tasks)
        
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
        
        # Combine all batch files into a single trajectories.jsonl file
        combined_file = self.output_dir / "trajectories.jsonl"
        print(f"\n📦 Combining batch files into {combined_file.name}...")
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for batch_num in range(len(self.batches)):
                batch_file = self.output_dir / f"batch_{batch_num}.jsonl"
                if batch_file.exists():
                    with open(batch_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
        
        print(f"✅ Combined {len(self.batches)} batch files into trajectories.jsonl")
        
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
            json.dump(final_stats, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Total prompts processed: {len(self.dataset)}")
        print(f"✅ Total batches: {len(self.batches)}")
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
    model: str = "claude-opus-4-20250514",
    api_key: str = None,
    base_url: str = "https://api.anthropic.com/v1/",
    max_turns: int = 10,
    num_workers: int = 4,
    resume: bool = False,
    verbose: bool = False,
    list_distributions: bool = False
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
        
    Examples:
        # Basic usage
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
        
        # Resume interrupted run
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
        
        # Use specific distribution
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=image_test --distribution=image_gen
        
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
            verbose=verbose
        )
        
        runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    fire.Fire(main)

