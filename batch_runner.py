#!/usr/bin/env python3
"""
Batch Agent Runner

This module provides parallel batch processing capabilities for running the agent
across multiple prompts from a dataset. It includes:
- Dataset loading
- Concurrent processing with asyncio (Producer-Consumer pattern)
- Checkpointing for fault tolerance and resumption
- Trajectory saving in the proper format (from/value pairs)
- Tool usage statistics aggregation across all prompts
- Cluster failure detection and graceful shutdown (morph, firecrawl, API errors)
- Configurable failure thresholds with automatic data consolidation

Usage:
    python batch_runner.py --dataset_file=data.jsonl --run_name=my_run

    # Resume an interrupted run
    python batch_runner.py --dataset_file=data.jsonl --run_name=my_run --resume

    # Use a specific toolset distribution
    python batch_runner.py --dataset_file=data.jsonl --run_name=my_run --distribution=image_gen

    # Configure tool failure thresholds
    python batch_runner.py --dataset_file=data.jsonl --run_name=my_run \\
                           --max_tool_failures=20 --max_tool_failure_rate=0.3 --min_tool_calls_for_rate=10
"""

import json
import logging
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
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


# Canonical names for the terminal tool (old & new implementations)
_TERMINAL_TOOL_NAMES = {"terminal", "terminal_tool", "simple_terminal_tool"}


def _is_terminal_tool_name(tool_name: Optional[str]) -> bool:
    """Return True if the given tool name corresponds to a terminal tool."""
    return bool(tool_name) and tool_name.lower() in _TERMINAL_TOOL_NAMES


def _terminal_tool_failed(content_json: Dict[str, Any]) -> bool:
    """
    Determine whether the terminal tool itself failed (not the user command).

    Terminal failures are indicated by explicit status flags or negative exit codes.
    Regular command failures (non-zero positive exit codes, stderr, timeouts) are not counted.
    """
    if not isinstance(content_json, dict):
        return False

    status = str(content_json.get("status", "")).lower()
    if status in {"error", "disabled"}:
        return True

    exit_code = content_json.get("exit_code")
    if isinstance(exit_code, int) and exit_code < 0:
        return True

    return False


def _categorize_error_type(error_message: str) -> str:
    """
    Categorize an error message into a failure type.

    Args:
        error_message (str): The error message to categorize

    Returns:
        str: Category of the error
    """
    error_lower = error_message.lower()

    # Common error patterns
    if "timeout" in error_lower or "timed out" in error_lower:
        return "Timeout"
    elif "connection" in error_lower or "connect" in error_lower:
        return "Connection Error"
    elif "rate limit" in error_lower or "ratelimit" in error_lower or "429" in error_lower:
        return "Rate Limit"
    elif "authentication" in error_lower or "auth" in error_lower or "unauthorized" in error_lower or "401" in error_lower:
        return "Authentication"
    elif "not found" in error_lower or "404" in error_lower:
        return "Not Found"
    elif "permission" in error_lower or "forbidden" in error_lower or "403" in error_lower:
        return "Permission Denied"
    elif "invalid" in error_lower or "malformed" in error_lower or "bad request" in error_lower or "400" in error_lower:
        return "Invalid Input"
    elif "out of memory" in error_lower or "oom" in error_lower:
        return "Out of Memory"
    elif "network" in error_lower:
        return "Network Error"
    elif "server error" in error_lower or "500" in error_lower or "502" in error_lower or "503" in error_lower:
        return "Server Error"
    elif "vm" in error_lower and ("fail" in error_lower or "error" in error_lower):
        return "VM Error"
    else:
        return "Other"


def _extract_tool_errors_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool errors from message history with tool names.

    Args:
        messages (List[Dict]): Message history

    Returns:
        List[Dict]: List of tool errors with tool name, error message, error type, and context
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
                    # Get tool name for special handling
                    tool_name = tool_calls_map.get(tool_call_id, "unknown")

                    # Special handling for terminal tool outputs
                    if _is_terminal_tool_name(tool_name):
                        if _terminal_tool_failed(content_json):
                            has_error = True
                            # Prefer explicit error text, fall back to status or generic message
                            error_msg = str(
                                content_json.get("error")
                                or content_json.get("status")
                                or "Terminal tool failure"
                            )
                    else:
                        # For other tools, check if error field exists AND has a non-null value
                        if "error" in content_json and content_json["error"] is not None:
                            has_error = True
                            error_msg = str(content_json["error"])

                        # Check nested content structure (some tools wrap responses)
                        if "content" in content_json and isinstance(content_json["content"], dict):
                            inner_content = content_json["content"]
                            if inner_content.get("error") is not None:
                                has_error = True
                                error_msg = inner_content.get("error")

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
                error_message = error_msg or "Unknown error"
                tool_errors.append({
                    "tool_name": tool_name,
                    "error_message": error_message,
                    "error_type": _categorize_error_type(error_message),
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
                    # Get tool name for special handling
                    tool_name = tool_calls_map.get(tool_call_id, "unknown")

                    # Special handling for terminal tool: only count as failure when the tool itself fails
                    if _is_terminal_tool_name(tool_name):
                        if _terminal_tool_failed(content_json):
                            is_success = False
                    else:
                        # For other tools, check if error field exists AND has a non-null value
                        if "error" in content_json and content_json["error"] is not None:
                            is_success = False

                        # Check nested content structure (some tools wrap responses)
                        if "content" in content_json and isinstance(content_json["content"], dict):
                            inner_content = content_json["content"]
                            # Check for actual error (non-null error field)
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


async def _process_single_prompt(
    prompt_index: int,
    prompt_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single prompt with the agent.
    
    Args:
        prompt_index (int): Index of prompt in dataset
        prompt_data (Dict): Prompt data containing 'prompt' field
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
            log_prefix_chars=config.get("log_prefix_chars", 100),
            prokletor_client=config.get("prokletor_client"),
            prokletor_formatter=config.get("prokletor_formatter")
        )

        # Run the agent with task_id to ensure each task gets its own isolated VM
        result = await agent.run_conversation(prompt, task_id=f"task_{prompt_index}")

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

        # Get profiling stats from the result
        profiling_stats = result.get("profiling_stats", {"tools": {}, "api_calls": {}})

        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": trajectory,
            "tool_stats": tool_stats,
            "tool_errors": tool_errors,
            "profiling_stats": profiling_stats,
            "completed": result["completed"],
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {
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
            "profiling_stats": {"tools": {}, "api_calls": {}},
            "trajectory": None,
            "tool_stats": {},
            "toolsets_used": [],
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }


async def worker(
    work_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    config: Dict[str, Any]
):
    """
    Consumer worker that processes prompts from the work queue.
    """
    while True:
        try:
            task = await work_queue.get()
            if task is None:
                # Sentinel to stop worker
                work_queue.task_done()
                break
            
            prompt_index, prompt_data = task
            
            result = await _process_single_prompt(prompt_index, prompt_data, config)
            
            await result_queue.put(result)
            work_queue.task_done()
            
        except Exception as e:
            print(f"Error in worker: {e}")
            if 'task' in locals() and task is not None:
                work_queue.task_done()


class BatchRunner:
    """
    Manages batch processing of agent prompts with checkpointing and statistics.
    """
    
    def __init__(
        self,
        dataset_file: str,
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
        max_tool_failures: float = float("inf"),
        max_tool_failure_rate: float = 0.5,
        keep_recent_errors: int = 5,
        min_tool_calls_for_rate: int = 10,
        prokletor_client: str = None,
        prokletor_formatter: str = None,
    ):
        """
        Initialize the batch runner.

        Args:
            dataset_file (str): Path to the dataset JSONL file with 'prompt' field
            run_name (str): Name for this run (used for checkpointing and output)
            distribution (str): Toolset distribution to use (default: "default")
            max_iterations (int): Max iterations per agent run
            base_url (str): Base URL for model API
            api_key (str): API key for model
            model (str): Model name to use
            num_workers (int): Number of parallel workers (default: 4)
            verbose (bool): Enable verbose logging
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
            max_tool_failures (float): Maximum number of tool failures before stopping (default: inf for unlimited)
            max_tool_failure_rate (float): Maximum tool failure rate (0.0-1.0) before stopping (default: 0.5)
            keep_recent_errors (int): Number of recent errors to keep per tool (default: 5)
            min_tool_calls_for_rate (int): Minimum number of tool calls before checking failure rate (default: 10)
            prokletor_client (str): Name of the prokletor client to use
            prokletor_formatter (str): Name of the prokletor formatter to use
        """
        self.dataset_file = Path(dataset_file)
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
        self.min_tool_calls_for_rate = min_tool_calls_for_rate
        self.prokletor_client = prokletor_client
        self.prokletor_formatter = prokletor_formatter
        
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

        # Errors file
        self.errors_file = self.output_dir / "errors.json"
        
        # Trajectories file
        self.trajectories_file = self.output_dir / "trajectories.jsonl"
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        safe_print("[bold cyan]📊 Batch Runner Initialized[/bold cyan]")
        safe_print(f"   Dataset: {self.dataset_file} ({len(self.dataset)} prompts)")
        safe_print(f"   Run name: {self.run_name}")
        safe_print(f"   Distribution: {self.distribution}")
        safe_print(f"   Output directory: {self.output_dir}")
        safe_print(f"   Workers: {self.num_workers}")
        safe_print(f"   [yellow]Tool failure limits:[/yellow]")
        safe_print(f"      Max failures: {self.max_tool_failures}")
        safe_print(f"      Max failure rate: {self.max_tool_failure_rate:.1%}")
        safe_print(f"      Min tool calls for rate check: {self.min_tool_calls_for_rate}")
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
                "last_updated": None
            }
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        Save checkpoint data.

        Args:
            checkpoint_data (Dict): Checkpoint data to save
        """
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    def _save_final_stats(
        self, 
        num_processed: int,
        tool_stats: Dict[str, Dict[str, int]],
        start_time: float,
        tool_errors_by_tool: Dict[str, List[Dict]],
        exception_errors: List[Dict],
        early_exit: bool = False,
        exit_reason: str = None,
        profiling_stats_list: List[Dict] = None
    ):
        """
        Save final statistics and errors.
        """
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

        # Build failure type breakdown for each tool
        failure_type_breakdown = {}
        for tool_name, errors in tool_errors_by_tool.items():
            failure_types = {}
            for error in errors:
                error_type = error.get("error_type", "Other")
                if error_type not in failure_types:
                    failure_types[error_type] = 0
                failure_types[error_type] += 1

            # Calculate percentages
            total_failures = len(errors)
            failure_type_breakdown[tool_name] = {
                "total_failures": total_failures,
                "types": {
                    error_type: {
                        "count": count,
                        "percentage": round((count / total_failures) * 100, 2)
                    }
                    for error_type, count in failure_types.items()
                }
            }

        # Save error information to separate file
        error_data = {
            "run_name": self.run_name,
            "completed_at": datetime.now().isoformat(),
            "total_tool_errors": sum(len(errors) for errors in tool_errors_by_tool.values()),
            "total_exception_errors": len(exception_errors),
            "tool_errors": tool_errors_by_tool,
            "failure_type_breakdown": failure_type_breakdown,
            "exception_errors": exception_errors[:self.keep_recent_errors]  # Keep k most recent
        }

        with open(self.errors_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)

        # Aggregate profiling statistics if available
        aggregated_profiling_stats = None
        if profiling_stats_list:
            try:
                from profiling import aggregate_profiling_stats, print_aggregated_statistics
                aggregated_profiling_stats = aggregate_profiling_stats(profiling_stats_list)
                
                # Display aggregated profiling statistics
                print_aggregated_statistics(aggregated_profiling_stats, detailed=True)
            except ImportError:
                pass

        # Save final statistics
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "processed": num_processed,
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "early_exit": early_exit,
            "exit_reason": exit_reason,
            "tool_statistics": tool_stats,
            "profiling_statistics": aggregated_profiling_stats
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

    async def _run_async(self, resume: bool = False):
        """
        Async implementation of the batch runner pipeline.
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Batch Processing")
        print("=" * 70)
        
        # Load checkpoint
        checkpoint_data = self._load_checkpoint() if resume else {
            "run_name": self.run_name,
            "completed_prompts": [],
            "last_updated": None
        }
        
        if resume and checkpoint_data.get("completed_prompts"):
            print(f"📂 Resuming from checkpoint ({len(checkpoint_data['completed_prompts'])} prompts already completed)")
        
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        
        # Prepare queues
        work_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        
        # Enqueue prompts to process
        prompts_to_process = []
        for idx, entry in enumerate(self.dataset):
            if idx not in completed_prompts_set:
                prompts_to_process.append((idx, entry))
                work_queue.put_nowait((idx, entry))
        
        total_to_process = len(prompts_to_process)
        if total_to_process == 0:
            print("✅ All prompts already completed.")
            return

        # Worker configuration
        worker_config = {
            "distribution": self.distribution,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "log_prefix_chars": self.log_prefix_chars,
            "prokletor_client": self.prokletor_client,
            "prokletor_formatter": self.prokletor_formatter
        }
        
        # Start workers
        workers = []
        for _ in range(min(self.num_workers, total_to_process)):
            w = asyncio.create_task(worker(work_queue, result_queue, worker_config))
            workers.append(w)
            
        print(f"   Processing {total_to_process} prompts with {len(workers)} workers...")
        
        # Aggregate statistics
        total_tool_stats = {}
        all_profiling_stats = []
        tool_errors_by_tool = {}
        all_exception_errors = []
        total_tool_errors = 0
        early_exit = False
        exit_reason = None
        processed_count = 0
        
        start_time = time.time()
        
        # Process results as they arrive
        try:
            while processed_count < total_to_process:
                result = await result_queue.get()
                processed_count += 1
                
                prompt_index = result["prompt_index"]
                
                # Track exceptions
                if not result["success"]:
                    safe_print(f"[bold red]❌ Exception in prompt {prompt_index}:[/bold red] {result.get('error', '')[:100]}")
                    all_exception_errors.append({
                        "prompt_index": prompt_index,
                        "error": result.get("error", "Unknown error"),
                        "traceback": result.get("traceback", "")
                    })
                else:
                    print(f"   ✅ Prompt {prompt_index} completed")
                    
                    # Save trajectory immediately
                    if result.get("trajectory"):
                        trajectory_entry = {
                            "prompt_index": prompt_index,
                            "conversations": result["trajectory"],
                            "metadata": result["metadata"],
                            "completed": result["completed"],
                            "api_calls": result["api_calls"],
                            "toolsets_used": result["toolsets_used"]
                        }
                        with open(self.trajectories_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(trajectory_entry, ensure_ascii=False) + "\n")
                
                # Aggregate tool stats
                for tool_name, stats in result.get("tool_stats", {}).items():
                    if tool_name not in total_tool_stats:
                        total_tool_stats[tool_name] = {"count": 0, "success": 0, "failure": 0}
                    
                    total_tool_stats[tool_name]["count"] += stats["count"]
                    total_tool_stats[tool_name]["success"] += stats["success"]
                    total_tool_stats[tool_name]["failure"] += stats["failure"]
                
                # Collect profiling stats
                if result.get("profiling_stats"):
                    all_profiling_stats.append(result["profiling_stats"])
                
                # Aggregate tool errors
                for tool_error in result.get("tool_errors", []):
                    tool_name = tool_error["tool_name"]
                    if tool_name not in tool_errors_by_tool:
                        tool_errors_by_tool[tool_name] = []
                    
                    tool_errors_by_tool[tool_name].append(tool_error)
                    # Keep only k most recent
                    if len(tool_errors_by_tool[tool_name]) > self.keep_recent_errors:
                        tool_errors_by_tool[tool_name] = tool_errors_by_tool[tool_name][-self.keep_recent_errors:]
                    
                    total_tool_errors += 1
                
                # Update checkpoint
                completed_prompts_set.add(prompt_index)
                checkpoint_data["completed_prompts"] = list(completed_prompts_set)
                self._save_checkpoint(checkpoint_data)
                
                # Check failure thresholds
                total_tool_calls = sum(stats["count"] for stats in total_tool_stats.values())
                
                if total_tool_errors >= self.max_tool_failures:
                    early_exit = True
                    exit_reason = f"Exceeded maximum tool failures ({total_tool_errors}/{self.max_tool_failures})"
                    break
                
                if total_tool_calls >= self.min_tool_calls_for_rate:
                    tool_failure_rate = total_tool_errors / total_tool_calls
                    if tool_failure_rate >= self.max_tool_failure_rate:
                        early_exit = True
                        exit_reason = f"Exceeded tool failure rate ({tool_failure_rate:.2%})"
                        break
        
        except asyncio.CancelledError:
            early_exit = True
            exit_reason = "Run cancelled"
        finally:
            # Stop all workers
            for _ in range(len(workers)):
                work_queue.put_nowait(None)
            await asyncio.gather(*workers, return_exceptions=True)
            
        if early_exit:
            safe_print(f"\n[bold red]🛑 STOPPING: {exit_reason}[/bold red]")
        
        # Save final statistics
        self._save_final_stats(
            processed_count,
            total_tool_stats,
            start_time,
            tool_errors_by_tool,
            all_exception_errors,
            early_exit,
            exit_reason,
            all_profiling_stats
        )
        
        # Summary output
        safe_print("\n" + "=" * 70)
        safe_print(f"✅ Total prompts processed: {processed_count}/{total_to_process}")
        safe_print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")
        
        if tool_errors_by_tool:
            safe_print(f"\n[bold red]🚨 Tool Errors: {total_tool_errors} total[/bold red]")
            # Simplified error printing here, full detail is in json
            for tool_name, errors in tool_errors_by_tool.items():
                safe_print(f"  {tool_name}: {len(errors)} errors")
        
        safe_print(f"\n[cyan]💾 Results saved to:[/cyan] {self.output_dir}")

    def run(self, resume: bool = False):
        """
        Run the batch processing pipeline (sync wrapper).
        """
        asyncio.run(self._run_async(resume))


def main(
    dataset_file: str = None,
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
    max_tool_failures: float = float("inf"),
    max_tool_failure_rate: float = 0.5,
    keep_recent_errors: int = 5,
    min_tool_calls_for_rate: int = 10,
    prokletor_client: str = None,
    prokletor_formatter: str = None,
):
    """
    Run batch processing of agent prompts from a dataset.

    Args:
        dataset_file (str): Path to JSONL file with 'prompt' field in each entry
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
        max_tool_failures (float): Maximum number of tool failures before stopping (default: inf for unlimited)
        max_tool_failure_rate (float): Maximum tool failure rate (0.0-1.0) before stopping (default: 0.5)
        keep_recent_errors (int): Number of recent errors to keep per tool for reporting (default: 5)
        min_tool_calls_for_rate (int): Minimum number of tool calls before checking failure rate (default: 10)
        prokletor_client (str): Name of the prokletor client to use
        prokletor_formatter (str): Name of the prokletor formatter to use

    Examples:
        # Basic usage
        python batch_runner.py --dataset_file=data.jsonl --run_name=my_run
        
        # Resume interrupted run
        python batch_runner.py --dataset_file=data.jsonl --run_name=my_run --resume
        
        # Use specific distribution
        python batch_runner.py --dataset_file=data.jsonl --run_name=image_test --distribution=image_gen
    """
    # Handle list distributions
    if list_distributions:
        from toolset_distributions import list_distributions as get_all_dists, print_distribution_info
        
        print("📊 Available Toolset Distributions")
        print("=" * 70)
        
        all_dists = get_all_dists()
        for dist_name in sorted(all_dists.keys()):
            print_distribution_info(dist_name)
        return
    
    # Validate required arguments
    if not dataset_file:
        print("❌ Error: --dataset_file is required")
        return
    
    if not run_name:
        print("❌ Error: --run_name is required")
        return
    
    # Initialize and run batch runner
    try:
        runner = BatchRunner(
            dataset_file=dataset_file,
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
            keep_recent_errors=keep_recent_errors,
            min_tool_calls_for_rate=min_tool_calls_for_rate,
            prokletor_client=prokletor_client,
            prokletor_formatter=prokletor_formatter
        )

        runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    fire.Fire(main)
