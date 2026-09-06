#!/usr/bin/env python3
"""Batch Agent Runner — run the agent over a JSONL prompt dataset in parallel.

Batches are processed by a multiprocessing pool with per-batch ``batch_N.jsonl`` output,
checkpointing for ``--resume``, trajectories in from/value format, and tool-usage
statistics aggregated across all batches. See ``main`` (fire CLI) for usage.
"""

# hermes_bootstrap must be the very first import — UTF-8 stdio on Windows, no-op on POSIX.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Partial ``hermes update`` (git reset landed, ``uv pip install -e .`` did not):
    # only Windows UTF-8 stdio setup is skipped.
    pass

import json
import logging
import os
import time
import traceback
from datetime import datetime
from multiprocessing import Lock, Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from model_tools import TOOL_TO_TOOLSET_MAP
from run_agent import AIAgent
from toolset_distributions import (
    list_distributions,
    sample_toolsets_from_distribution,
    validate_distribution,
)

logger = logging.getLogger(__name__)

# Auto-derived from model_tools so it stays in sync as tools are added. Gives every
# trajectory a consistent tool_stats schema (Arrow/Parquet for HF datasets) and filters
# corrupted entries (hallucinated tool names) when combining trajectories.
ALL_POSSIBLE_TOOLS = set(TOOL_TO_TOOLSET_MAP.keys())

DEFAULT_TOOL_STATS = {'count': 0, 'success': 0, 'failure': 0}
_REASONING_KEYS = ("total_assistant_turns", "turns_with_reasoning", "turns_without_reasoning")

# BatchRunner.__init__ parameters stored as same-named attributes.
_RUNNER_FIELDS = (
    "batch_size", "run_name", "distribution", "max_iterations", "base_url", "api_key", "model",
    "num_workers", "verbose", "ephemeral_system_prompt", "log_prefix_chars", "providers_allowed",
    "providers_ignored", "providers_order", "provider_sort", "openrouter_min_coding_score",
    "max_tokens", "reasoning_config", "prefill_messages", "max_samples",
)
# BatchRunner attributes forwarded verbatim to every AIAgent in the worker config.
_AGENT_PASSTHROUGH = (
    "base_url", "api_key", "ephemeral_system_prompt", "providers_allowed", "providers_ignored",
    "providers_order", "provider_sort", "openrouter_min_coding_score", "max_tokens",
    "reasoning_config", "prefill_messages",
)


def _normalize_tool_stats(tool_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """All possible tools with zero defaults (consistent HF schema), plus any unexpected tools."""
    normalized = {
        tool: tool_stats[tool].copy() if tool in tool_stats else DEFAULT_TOOL_STATS.copy()
        for tool in ALL_POSSIBLE_TOOLS
    }
    for tool, stats in tool_stats.items():
        if tool not in normalized:
            normalized[tool] = stats.copy()
    return normalized


def _normalize_tool_error_counts(tool_error_counts: Dict[str, int]) -> Dict[str, int]:
    """All possible tools with zero defaults, plus any unexpected tools."""
    normalized = {tool: tool_error_counts.get(tool, 0) for tool in ALL_POSSIBLE_TOOLS}
    for tool, count in tool_error_counts.items():
        if tool not in normalized:
            normalized[tool] = count
    return normalized


def _merge_tool_stats(total: Dict[str, Dict[str, int]], tool_stats: Dict[str, Dict[str, int]]) -> None:
    """Add per-tool count/success/failure from *tool_stats* into *total* in place."""
    for tool_name, stats in tool_stats.items():
        agg = total.setdefault(tool_name, DEFAULT_TOOL_STATS.copy())
        agg["count"] += stats["count"]
        agg["success"] += stats["success"]
        agg["failure"] += stats["failure"]


def _merge_reasoning_stats(total: Dict[str, int], reasoning_stats: Dict[str, Any]) -> None:
    """Add the turn counters from *reasoning_stats* into *total* in place."""
    for key in total:
        total[key] += reasoning_stats.get(key, 0)


def _tool_call_succeeded(content) -> bool:
    """Judge a tool response: JSON with a non-null ``error`` / ``success: false`` fails;
    non-JSON fails only when empty or starting with ``Error:`` (no substring matching, to
    avoid false positives). Non-zero exit codes are NOT failures — the model self-corrects."""
    try:
        content_json = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, ValueError, TypeError):
        return bool(content) and not content.strip().lower().startswith("error:")
    if not isinstance(content_json, dict):
        return True
    if content_json.get("error") is not None:
        return False
    # Terminal wraps its response in a "content" field.
    inner_content = content_json.get("content")
    if isinstance(inner_content, dict) and inner_content.get("error") is not None:
        return False
    return content_json.get("success") is not False


def _extract_tool_stats(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per-tool call counts and success/failure tallies from a message history."""
    tool_stats = {}
    tool_calls_map = {}  # tool_call_id -> tool name

    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not tool_call or not isinstance(tool_call, dict): continue
                tool_name = tool_call["function"]["name"]
                tool_stats.setdefault(tool_name, DEFAULT_TOOL_STATS.copy())["count"] += 1
                tool_calls_map[tool_call["id"]] = tool_name
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            is_success = _tool_call_succeeded(msg.get("content", ""))
            if tool_call_id in tool_calls_map:
                tool_stats[tool_calls_map[tool_call_id]]["success" if is_success else "failure"] += 1

    return tool_stats


def _turn_has_reasoning(msg: Dict[str, Any]) -> bool:
    """``<REASONING_SCRATCHPAD>`` in content, or a non-empty native ``reasoning`` field."""
    if "<REASONING_SCRATCHPAD>" in (msg.get("content", "") or ""):
        return True
    return bool(msg.get("reasoning", "").strip()) if msg.get("reasoning") else False


def _extract_reasoning_stats(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count assistant turns with reasoning vs without."""
    assistant_turns = [msg for msg in messages if msg.get("role") == "assistant"]
    total = len(assistant_turns)
    with_reasoning = sum(1 for msg in assistant_turns if _turn_has_reasoning(msg))
    return {
        "total_assistant_turns": total,
        "turns_with_reasoning": with_reasoning,
        "turns_without_reasoning": total - with_reasoning,
        "has_any_reasoning": with_reasoning > 0,
    }


def _failure_result(prompt_index: int, batch_num: int, error: str) -> Dict[str, Any]:
    """Result dict for a prompt that produced no trajectory."""
    return {
        "success": False,
        "prompt_index": prompt_index,
        "error": error,
        "trajectory": None,
        "tool_stats": {},
        "toolsets_used": [],
        "metadata": {"batch_num": batch_num, "timestamp": datetime.now().isoformat()},
    }


def _prepare_container_image(
    prompt_index: int, prompt_data: Dict[str, Any], batch_num: int, task_id: str, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Register the dataset row's per-prompt container image (``image``/``docker_image``)
    for this task's sandbox (Docker, Modal, Singularity, Daytona).

    For Docker the image is verified (local cache, then pull) before spending tokens on the
    agent loop; Modal pulls server-side so no local check. Returns a failure result when the
    pull fails, else ``None``.
    """
    container_image = prompt_data.get("image") or prompt_data.get("docker_image")
    if not container_image:
        return None
    env_type = os.getenv("TERMINAL_ENV", "local")
    if env_type == "docker":
        import subprocess as _sp
        try:
            probe = _sp.run(
                ["docker", "image", "inspect", container_image],
                capture_output=True, timeout=10,
            )
            if probe.returncode != 0:
                if config.get("verbose"):
                    print(f"   Prompt {prompt_index}: Pulling docker image {container_image}...", flush=True)
                pull = _sp.run(
                    ["docker", "pull", container_image],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600,
                )
                if pull.returncode != 0:
                    return _failure_result(
                        prompt_index, batch_num,
                        f"Docker image not available: {container_image}\n{pull.stderr[:500]}",
                    )
        except FileNotFoundError:
            pass  # Docker CLI not installed — skip check (e.g., Modal backend)
        except Exception as img_err:
            if config.get("verbose"):
                print(f"   Prompt {prompt_index}: Docker image check failed: {img_err}", flush=True)
    from tools.terminal_tool import register_task_env_overrides
    overrides = {
        "docker_image": container_image,
        "modal_image": container_image,
        "singularity_image": f"docker://{container_image}",
        "daytona_image": container_image,
    }
    if prompt_data.get("cwd"):
        overrides["cwd"] = prompt_data["cwd"]
    register_task_env_overrides(task_id, overrides)
    if config.get("verbose"):
        print(f"   Prompt {prompt_index}: Using container image {container_image}")
    return None


def _process_single_prompt(
    prompt_index: int,
    prompt_data: Dict[str, Any],
    batch_num: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the agent on one prompt; returns trajectory, stats and metadata (or a failure result)."""
    prompt = prompt_data["prompt"]
    task_id = f"task_{prompt_index}"
    failure = _prepare_container_image(prompt_index, prompt_data, batch_num, task_id, config)
    if failure is not None:
        return failure

    try:
        selected_toolsets = sample_toolsets_from_distribution(config["distribution"])

        if config.get("verbose"):
            print(f"   Prompt {prompt_index}: Using toolsets {selected_toolsets}")
        agent = AIAgent(
            model=config["model"],
            max_iterations=config["max_iterations"],
            enabled_toolsets=selected_toolsets,
            save_trajectories=False,  # We handle saving ourselves
            verbose_logging=config.get("verbose", False),
            log_prefix_chars=config.get("log_prefix_chars", 100),
            log_prefix=f"[B{batch_num}:P{prompt_index}]",
            skip_context_files=True,  # Don't pollute trajectories with SOUL.md/AGENTS.md
            skip_memory=True,  # Don't use persistent memory in batch runs
            **{key: config.get(key) for key in _AGENT_PASSTHROUGH},
        )

        # task_id ensures each task gets its own isolated VM
        result = agent.run_conversation(prompt, task_id=task_id)

        # Stats before conversion — keep the original evaluation order.
        tool_stats = _extract_tool_stats(result["messages"])
        reasoning_stats = _extract_reasoning_stats(result["messages"])
        trajectory = agent._convert_to_trajectory_format(result["messages"], prompt, result["completed"])

        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": trajectory,
            "tool_stats": tool_stats,
            "reasoning_stats": reasoning_stats,
            "completed": result["completed"],
            # Sibling of the non-empty-response return below (#64686): the classifier's failure_reason must
            # survive the empty-response normalization path too, or downstream consumers (TUI billing
            # surface, transient-failure persistence) lose the structured reason exactly when the run
            # produced no text.
            "partial": result.get("partial", False),
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {"batch_num": batch_num, "timestamp": datetime.now().isoformat(), "model": config["model"]},
        }
    except Exception as e:
        print(f"❌ Error processing prompt {prompt_index}: {e}")
        if config.get("verbose"):
            traceback.print_exc()
        return _failure_result(prompt_index, batch_num, str(e))


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON row and fsync so a crash never loses an acknowledged prompt."""
    from hermes_cli.config import artifact_file_mode
    from utils import open_private_append

    with open_private_append(path, mode=artifact_file_mode()) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _process_batch_worker(args: Tuple) -> Dict[str, Any]:
    """Pool worker: process one batch of ``(index, prompt_data)`` sequentially.

    ``args`` is ``(batch_num, batch_data, output_dir, completed_prompts, config)``.
    """
    batch_num, batch_data, output_dir, completed_prompts_set, config = args
    output_dir = Path(output_dir)
    print(f"\n🔄 Batch {batch_num}: Starting ({len(batch_data)} prompts)")
    batch_output_file = output_dir / f"batch_{batch_num}.jsonl"
    prompts_to_process = [(idx, data) for idx, data in batch_data if idx not in completed_prompts_set]

    if not prompts_to_process:
        print(f"✅ Batch {batch_num}: Already completed (skipping)")
        return {"batch_num": batch_num, "processed": 0, "skipped": len(batch_data), "tool_stats": {}, "completed_prompts": []}
    print(f"   Processing {len(prompts_to_process)} prompts (skipping {len(batch_data) - len(prompts_to_process)} already completed)")
    batch_tool_stats = {}
    batch_reasoning_stats = dict.fromkeys(_REASONING_KEYS, 0)
    completed_in_batch = []
    discarded_no_reasoning = 0

    for prompt_index, prompt_data in prompts_to_process:
        result = _process_single_prompt(prompt_index, prompt_data, batch_num, config)

        if result["success"] and result["trajectory"]:
            reasoning = result.get("reasoning_stats", {})
            if not reasoning.get("has_any_reasoning", True):
                print(f"   🚫 Prompt {prompt_index} discarded (no reasoning in any turn)")
                discarded_no_reasoning += 1
                completed_in_batch.append(prompt_index)
                # Tombstone row (#93527): resume filters by scanning batch_*.jsonl
                # rows for prompt content, so a discarded sample without a row would
                # be re-run at full cost on every restart. The merge step excludes
                # tombstones from trajectories.jsonl.
                _append_jsonl(batch_output_file, {
                    "prompt_index": prompt_index,
                    "discarded": "no_reasoning",
                    "prompt": _entry_prompt_text(prompt_data),
                })
                continue

            # Normalize for a consistent schema across all entries.
            raw_tool_stats = result.get("tool_stats", {})
            raw_error_counts = {
                tool_name: stats.get("failure", 0)
                for tool_name, stats in raw_tool_stats.items()
            }
            _append_jsonl(batch_output_file, {
                "prompt_index": prompt_index,
                "conversations": result["trajectory"],
                "metadata": result["metadata"],
                "completed": result["completed"],
                "partial": result.get("partial", False),  # True if stopped due to invalid tool calls
                "api_calls": result["api_calls"],
                "toolsets_used": result["toolsets_used"],
                "tool_stats": _normalize_tool_stats(raw_tool_stats),  # {tool: {count, success, failure}}
                "tool_error_counts": _normalize_tool_error_counts(raw_error_counts)  # {tool: failure_count}
            })
        _merge_tool_stats(batch_tool_stats, result.get("tool_stats", {}))
        _merge_reasoning_stats(batch_reasoning_stats, result.get("reasoning_stats", {}))

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
        "reasoning_stats": batch_reasoning_stats,
        "discarded_no_reasoning": discarded_no_reasoning,
        "completed_prompts": completed_in_batch
    }


def _entry_prompt_text(entry: Dict) -> str:
    """Human prompt text from a dataset/trajectory entry: flat ``prompt``, ShareGPT
    ``conversations`` (from/value), chat ``conversations``/``messages`` (role/content),
    or a no-reasoning discard tombstone."""
    if not isinstance(entry, dict):
        return ""
    text = str(entry.get("prompt") or "").strip()
    if text:
        return text
    for key in ("conversations", "messages"):
        for msg in entry.get(key, []) or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from")
            if role in {"user", "human"}:
                text = str(msg.get("content") or msg.get("value") or "").strip()
                if text:
                    return text
    return ""


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _chunk(entries: List[Tuple[int, Dict[str, Any]]], size: int) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Split ``(index, entry)`` tuples into batches of *size*, preserving original indices."""
    return [entries[i:i + size] for i in range(0, len(entries), size)]


class BatchRunner:
    """Manages batch processing of agent prompts with checkpointing and statistics."""

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
        openrouter_min_coding_score: Optional[float] = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        max_samples: int = None,
    ):
        """Load the dataset (truncated to *max_samples*), validate *distribution*, create batches.

        ``ephemeral_system_prompt`` is used during execution but NOT saved to trajectories.
        ``prefill_messages`` are prepended as few-shot context; Anthropic Sonnet/Opus 4.6+
        reject a trailing assistant-role prefill (400) — use user-role priming for those.
        """
        params = dict(locals())
        self.dataset_file = Path(dataset_file)
        for name in _RUNNER_FIELDS:
            setattr(self, name, params[name])

        if not validate_distribution(distribution):
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(list_distributions().keys())}")
        self.output_dir = Path("data") / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.stats_file = self.output_dir / "statistics.json"
        self.dataset = self._load_dataset()
        if self.max_samples and self.max_samples < len(self.dataset):
            full_count = len(self.dataset)
            self.dataset = self.dataset[:self.max_samples]
            print(f"✂️  Truncated dataset from {full_count} to {self.max_samples} samples (--max_samples)")
        self.batches = self._create_batches()
        print("📊 Batch Runner Initialized")
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
        """Load JSONL entries that have a ``prompt`` field; skip blank/invalid lines."""
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
        """Split the dataset into batches of ``(index, entry)`` tuples."""
        return _chunk(list(enumerate(self.dataset)), self.batch_size)

    def _empty_checkpoint(self) -> Dict[str, Any]:
        return {"run_name": self.run_name, "completed_prompts": [], "batch_stats": {}, "last_updated": None}

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Checkpoint data (completed prompt indices), or an empty one if missing/unreadable."""
        if not self.checkpoint_file.exists():
            return self._empty_checkpoint()

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            return self._empty_checkpoint()

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any], lock: Optional[Lock] = None):
        """Atomically write *checkpoint_data* (stamped ``last_updated``), under *lock* if given."""
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        from utils import atomic_json_write
        if lock:
            with lock:
                atomic_json_write(self.checkpoint_file, checkpoint_data)
        else:
            atomic_json_write(self.checkpoint_file, checkpoint_data)

    def _scan_completed_prompts_by_content(self) -> set:
        """Prompt texts already processed, scanned from every ``batch_*.jsonl``.

        Matching on content rather than index lets resume recover even when indices
        don't line up. Failed entries are skipped (retried); discard tombstones count
        as completed (#93527) — re-running would just re-discard.
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
                            if entry.get("failed", False):
                                continue
                            prompt_text = _entry_prompt_text(entry)
                            if prompt_text:
                                completed_prompts.add(prompt_text)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"  ⚠️  Warning: Error reading {batch_file.name}: {e}")

        return completed_prompts

    def _filter_dataset_by_completed(self, completed_prompts: set) -> Tuple[List[Dict], List[int]]:
        """Return ``([(index, entry)] not yet completed, [skipped indices])``."""
        filtered_dataset = []
        skipped_indices = []

        for idx, entry in enumerate(self.dataset):
            prompt_text = entry.get("prompt", "").strip()

            # Also check conversations format
            if not prompt_text:
                conversations = entry.get("conversations", [])
                for msg in conversations:
                    role = msg.get("role") or msg.get("from")
                    if role in {"user", "human"}:
                        prompt_text = (msg.get("content") or msg.get("value", "")).strip()
                        break

            if prompt_text in completed_prompts:
                skipped_indices.append(idx)
            else:
                filtered_dataset.append((idx, entry))

        return filtered_dataset, skipped_indices

    def _apply_resume(self) -> bool:
        """Rebuild ``self.batches`` from unprocessed prompts. False when nothing is left to run."""
        completed_prompt_texts = self._scan_completed_prompts_by_content()
        if not completed_prompt_texts:
            return True
        print(f"   Found {len(completed_prompt_texts)} already-completed prompts by content matching")
        filtered_entries, skipped_indices = self._filter_dataset_by_completed(completed_prompt_texts)

        if not filtered_entries:
            print("\n✅ All prompts have already been processed!")
            return False
        self.batches = _chunk(filtered_entries, self.batch_size)
        _banner("📊 RESUME SUMMARY")
        print(f"   Original dataset size:     {len(self.dataset):,} prompts")
        print(f"   Already completed:         {len(skipped_indices):,} prompts")
        print("   ─────────────────────────────────────────")
        print(f"   🎯 RESUMING WITH:          {len(filtered_entries):,} prompts")
        print(f"   New batches created:       {len(self.batches)}")
        print("=" * 70 + "\n")
        return True

    def _worker_config(self) -> Dict[str, Any]:
        """Picklable agent configuration for worker processes.

        ``self.api_key`` may be a zero-arg callable (Azure Foundry Entra ID bearer provider
        from ``agent.azure_identity_adapter``), which is not safely picklable across the
        Pool boundary. Drop it and let each worker rebuild its own provider via
        ``resolve_runtime_provider()`` from ``model.auth_mode`` in config.yaml
        (azure-identity caches in-process, so each worker gets its own short-lived cache).
        """
        if callable(self.api_key) and not isinstance(self.api_key, str):
            worker_api_key = None
            print(
                "ℹ️  Detected Entra ID bearer provider — workers will rebuild "
                "credentials from config.yaml in each process.",
                flush=True,
            )
        else:
            worker_api_key = self.api_key
        config = {key: getattr(self, key) for key in _AGENT_PASSTHROUGH}
        config["api_key"] = worker_api_key
        for key in ("distribution", "model", "max_iterations", "verbose", "log_prefix_chars"):
            config[key] = getattr(self, key)
        return config

    def _run_pool(self, config, checkpoint_data, completed_prompts_set, checkpoint_lock) -> List[Dict[str, Any]]:
        """Process all batches in a worker pool, checkpointing after each result."""
        print(f"\n🔧 Initializing {self.num_workers} worker processes...")

        with Pool(processes=self.num_workers) as pool:
            # output_dir as str for pickling
            tasks = [
                (batch_num, batch_data, str(self.output_dir), completed_prompts_set, config)
                for batch_num, batch_data in enumerate(self.batches)
            ]
            print(f"✅ Created {len(tasks)} batch tasks")
            print("🚀 Starting parallel batch processing...\n")

            # rich Progress gives a persistent bottom bar; stdout/stderr are NOT
            # redirected so worker prints stay visible.
            results = []
            console = Console(force_terminal=True)
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]📦 Batches"), BarColumn(bar_width=40),
                MofNCompleteColumn(), TextColumn("•"), TimeRemainingColumn(),
                console=console, refresh_per_second=2, transient=False,
                redirect_stdout=False, redirect_stderr=False,
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

                        # Incremental checkpoint update (so resume works after crash)
                        try:
                            batch_num = result.get('batch_num')
                            completed = result.get('completed_prompts', []) or []
                            completed_prompts_set.update(completed)

                            if isinstance(batch_num, int):
                                checkpoint_data.setdefault('batch_stats', {})[str(batch_num)] = {
                                    key: result.get(key, 0) for key in ('processed', 'skipped', 'discarded_no_reasoning')
                                }
                            checkpoint_data['completed_prompts'] = sorted(completed_prompts_set)
                            self._save_checkpoint(checkpoint_data, lock=checkpoint_lock)
                        except Exception as ckpt_err:
                            # Don't fail the run if checkpoint write fails
                            print(f"⚠️  Warning: Failed to save incremental checkpoint: {ckpt_err}")
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted — terminating batch workers...")
                    pool.terminate()
                    pool.join()
                    raise
                except Exception as e:
                    logger.error("Batch worker failed: %s", e, exc_info=True)
                    pool.terminate()
                    pool.join()
                    raise
                finally:
                    root_logger.setLevel(original_level)
        return results

    def _combine_batch_files(self) -> Tuple[int, int]:
        """Merge ALL ``batch_*.jsonl`` (old runs + resume) into ``trajectories.jsonl``.

        Drops corrupted entries (hallucinated tool names, invalid JSON) and discard
        tombstones (#93527, resume bookkeeping only). Returns ``(kept, files_found)``.
        """
        combined_file = self.output_dir / "trajectories.jsonl"
        print(f"\n📦 Combining ALL batch files into {combined_file.name}...")
        total_entries = 0
        filtered_entries = 0
        tombstone_entries = 0
        batch_files_found = 0
        all_batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))

        from hermes_cli.config import artifact_file_mode
        from utils import open_private_append

        with open_private_append(combined_file, mode=artifact_file_mode()) as outfile:
            outfile.truncate(0)
            for batch_file in all_batch_files:
                batch_files_found += 1
                batch_num = batch_file.stem.split("_")[1]  # Extract batch number for logging

                with open(batch_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        total_entries += 1
                        try:
                            data = json.loads(line)

                            if data.get("discarded"):
                                tombstone_entries += 1
                                continue
                            tool_stats = data.get('tool_stats', {})
                            invalid_tools = [k for k in tool_stats if k not in ALL_POSSIBLE_TOOLS]

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
        kept = total_entries - filtered_entries - tombstone_entries
        print(f"✅ Combined {batch_files_found} batch files into trajectories.jsonl ({kept} entries)")
        return kept, batch_files_found

    def _print_summary(self, results, total_tool_stats, total_reasoning_stats, kept, batch_files_found, start_time) -> None:
        _banner("📊 BATCH PROCESSING COMPLETE")
        print(f"✅ Prompts processed this run: {sum(r.get('processed', 0) for r in results)}")
        print(f"✅ Total trajectories in merged file: {kept}")
        print(f"✅ Total batch files merged: {batch_files_found}")
        print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")
        print("\n📈 Tool Usage Statistics:")
        print("-" * 70)

        if total_tool_stats:
            sorted_tools = sorted(total_tool_stats.items(), key=lambda x: x[1]["count"], reverse=True)
            print(f"{'Tool Name':<25} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<12}")
            print("-" * 70)
            for tool_name, stats in sorted_tools:
                print(f"{tool_name:<25} {stats['count']:<10} {stats['success']:<10} {stats['failure']:<10} {stats['success_rate']:.1f}%")
        else:
            print("No tool calls were made during this run.")
        total_discarded = sum(r.get("discarded_no_reasoning", 0) for r in results)

        print("\n🧠 Reasoning Coverage:")
        print("-" * 70)
        total_turns = total_reasoning_stats["total_assistant_turns"]
        with_reasoning = total_reasoning_stats["turns_with_reasoning"]
        without_reasoning = total_reasoning_stats["turns_without_reasoning"]
        if total_turns > 0:
            pct_with = round(with_reasoning / total_turns * 100, 1)
            pct_without = round(without_reasoning / total_turns * 100, 1)
            print(f"   Total assistant turns:    {total_turns:,}")
            print(f"   With reasoning:           {with_reasoning:,} ({pct_with}%)")
            print(f"   Without reasoning:        {without_reasoning:,} ({pct_without}%)")
        else:
            print("   No assistant turns recorded.")
        if total_discarded > 0:
            print(f"   🚫 Samples discarded (zero reasoning): {total_discarded:,}")
        print(f"\n💾 Results saved to: {self.output_dir}")
        print("   - Trajectories: trajectories.jsonl (combined)")
        print("   - Individual batches: batch_*.jsonl (for debugging)")
        print(f"   - Statistics: {self.stats_file.name}")
        print(f"   - Checkpoint: {self.checkpoint_file.name}")

    def run(self, resume: bool = False):
        """Run the batch pipeline; with *resume*, skip prompts already present in batch files."""
        _banner("🚀 Starting Batch Processing")

        if resume and not self._apply_resume():
            return

        # Load existing checkpoint (so resume doesn't clobber prior progress)
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data.get("run_name") != self.run_name:
            checkpoint_data = self._empty_checkpoint()
        config = self._worker_config()

        # Index tracking is secondary to content matching (backward compatibility).
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        start_time = time.time()

        # Checkpoint writes happen in the parent process; keep a lock for safety.
        checkpoint_lock = Lock()
        results = self._run_pool(config, checkpoint_data, completed_prompts_set, checkpoint_lock)
        total_tool_stats = {}
        total_reasoning_stats = dict.fromkeys(_REASONING_KEYS, 0)
        for batch_result in results:
            _merge_tool_stats(total_tool_stats, batch_result.get("tool_stats", {}))
            _merge_reasoning_stats(total_reasoning_stats, batch_result.get("reasoning_stats", {}))

        # Final checkpoint is best-effort; incremental writes already happened.
        try:
            checkpoint_data["completed_prompts"] = sorted(completed_prompts_set)
            self._save_checkpoint(checkpoint_data, lock=checkpoint_lock)
        except Exception as ckpt_err:
            print(f"⚠️  Warning: Failed to save final checkpoint: {ckpt_err}")

        for stats in total_tool_stats.values():
            total_calls = stats["success"] + stats["failure"]
            stats["success_rate"] = round(stats["success"] / total_calls * 100, 2) if total_calls > 0 else 0.0
            stats["failure_rate"] = round(stats["failure"] / total_calls * 100, 2) if total_calls > 0 else 0.0
        kept, batch_files_found = self._combine_batch_files()
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "total_batches": len(self.batches),
            "batch_size": self.batch_size,
            # Snapshot the CLI-level credential/runtime fields BEFORE mutating them so a failed in-place
            # agent swap can roll the whole CLI back to the old working model. Otherwise the broken
            # credentials staged below leak into the next turn's resolution even though the agent itself
            # rolled back (#50163).
            # Snapshot CLI-level fields before mutation so a failed in-place swap rolls the whole CLI back
            # to the old working model (#50163).
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "tool_statistics": total_tool_stats,
            "reasoning_statistics": total_reasoning_stats,
            "discarded_no_reasoning": sum(r.get("discarded_no_reasoning", 0) for r in results),
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        self._print_summary(results, total_tool_stats, total_reasoning_stats, kept, batch_files_found, start_time)


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    """Comma-separated CLI string to a list of stripped items; ``None`` when empty."""
    return [p.strip() for p in value.split(",")] if value else None


def main(
    dataset_file: str = None,
    batch_size: int = None,
    run_name: str = None,
    distribution: str = "default",
    model: str = "anthropic/claude-sonnet-4.6",
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
    max_tokens: int = None,
    reasoning_effort: str = None,
    reasoning_disabled: bool = False,
    prefill_messages_file: str = None,
    max_samples: int = None,
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
        max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
        reasoning_effort (str): Reasoning effort: "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra" (default: "medium")
        reasoning_disabled (bool): Completely disable reasoning/thinking tokens (default: False)
        prefill_messages_file (str): Path to JSON file containing prefill messages (list of {role, content} dicts)
        max_samples (int): Only process the first N samples from the dataset (optional, processes all if not set)
        
    Examples:
        # Basic usage
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
        
        # Resume interrupted run
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
        
        # Use specific distribution
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=image_test --distribution=image_gen
        
        # With disabled reasoning and max tokens
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --reasoning_disabled --max_tokens=128000
        
        # With prefill messages from file
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --prefill_messages_file=configs/prefill_opus.json
        
        # List available distributions
        python batch_runner.py --list_distributions
    """
    if list_distributions:
        from toolset_distributions import print_distribution_info
        print("📊 Available Toolset Distributions")
        print("=" * 70)
        all_dists = list_distributions()
        for dist_name in sorted(all_dists.keys()):
            print_distribution_info(dist_name)

        print("\n💡 Usage:")
        print("  python batch_runner.py --dataset_file=data.jsonl --batch_size=10 \\")
        print("                         --run_name=my_run --distribution=<name>")
        return

    for invalid, message in (
        (not dataset_file, "--dataset_file is required"),
        (not batch_size or batch_size < 1, "--batch_size must be a positive integer"),
        (not run_name, "--run_name is required"),
    ):
        if invalid:
            print(f"❌ Error: {message}")
            raise SystemExit(1)

    # --reasoning_disabled takes priority, then --reasoning_effort, then default (medium)
    reasoning_config = None
    if reasoning_disabled:
        reasoning_config = {"effort": "none"}
        print("🧠 Reasoning: DISABLED (effort=none)")
    elif reasoning_effort:
        valid_efforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
        if reasoning_effort not in valid_efforts:
            print(f"❌ Error: --reasoning_effort must be one of: {', '.join(valid_efforts)}")
            raise SystemExit(1)
        reasoning_config = {"enabled": True, "effort": reasoning_effort}
        print(f"🧠 Reasoning effort: {reasoning_effort}")
    prefill_messages = None
    if prefill_messages_file:
        try:
            with open(prefill_messages_file, 'r', encoding='utf-8') as f:
                prefill_messages = json.load(f)
            if not isinstance(prefill_messages, list):
                print("❌ Error: prefill_messages_file must contain a JSON array of messages")
                raise SystemExit(1)
            print(f"💬 Loaded {len(prefill_messages)} prefill messages from {prefill_messages_file}")
        except Exception as e:
            print(f"❌ Error loading prefill messages: {e}")
            raise SystemExit(1)

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
            providers_allowed=_split_csv(providers_allowed),
            providers_ignored=_split_csv(providers_ignored),
            providers_order=_split_csv(providers_order),
            provider_sort=provider_sort,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            max_samples=max_samples,
        )
        runner.run(resume=resume)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
