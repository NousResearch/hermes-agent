"""
Endless Terminals Environment for Hermes-Agent + Atropos RL.

Loads pre-generated terminal tasks from HuggingFace dataset and scores
agent performance using test execution in the agent's sandbox.

Uses hermes-agent backends (modal, docker, local) with per-task Docker images
extracted from container.def files. Tests run in the same sandbox the agent
used, following the Terminal Bench 2 pattern.

Dataset: https://huggingface.co/datasets/obiwan96/endless-terminals-train

Run:
  python environments/endless_terminals/endless_terminals_env.py process \
    --config environments/endless_terminals/default.yaml
"""

import asyncio
import logging
import os
import random
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

# Ensure hermes-agent root is on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.base import ScoredDataGroup, ScoredDataItem
from atroposlib.type_definitions import Item

# Monkey-patch atroposlib's ManagedServer to forward chat_template_kwargs from
# extra_body into apply_chat_template. Should be upstreamed to atroposlib.
def _patch_managed_server():
    import inspect
    from atroposlib.envs.server_handling import managed_server as _ms

    _ManagedServer = _ms.ManagedServer

    # Patch 1: add extra_template_kwargs + chat_template_kwargs forwarding
    # Only apply if not already in the installed version
    _needs_patch1 = "extra_template_kwargs" not in inspect.signature(_ManagedServer._convert_messages_to_prompt).parameters

    if _needs_patch1:
        def _convert_messages_to_prompt(self, messages, tools=None, extra_template_kwargs=None):
            if tools and self._get_translator():
                messages = self._get_translator().convert_messages_for_template(messages)
            if self.tokenizer is None:
                return "\n".join([f"{m['role']}: {m.get('content', '')}" for m in messages])
            if hasattr(self.tokenizer, "apply_chat_template"):
                add_generation_prompt = len(messages) == 0 or messages[-1].get("role") != "assistant"
                if not self._preserve_think_blocks:
                    messages = self._protect_think_blocks(messages)
                template_kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
                if tools:
                    template_kwargs["tools"] = tools
                if extra_template_kwargs:
                    template_kwargs.update(extra_template_kwargs)
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
                prompt = prompt.replace(self._THINK_OPEN, "<think>")
                prompt = prompt.replace(self._THINK_CLOSE, "</think>")
                return prompt
            return "\n".join([f"{m['role']}: {m.get('content', '')}" for m in messages])

        _orig_chat_completion = _ManagedServer.chat_completion

        async def chat_completion(self, **kwargs):
            extra_body = kwargs.get("extra_body", {}) or {}
            chat_template_kwargs = extra_body.get("chat_template_kwargs", None)
            self._pending_chat_template_kwargs = chat_template_kwargs
            try:
                return await _orig_chat_completion(self, **kwargs)
            finally:
                self._pending_chat_template_kwargs = None

        def _convert_messages_to_prompt_with_stash(self, messages, tools=None, extra_template_kwargs=None):
            if extra_template_kwargs is None:
                extra_template_kwargs = getattr(self, "_pending_chat_template_kwargs", None)
            return _convert_messages_to_prompt(self, messages, tools=tools, extra_template_kwargs=extra_template_kwargs)

        _ManagedServer._convert_messages_to_prompt = _convert_messages_to_prompt_with_stash
        _ManagedServer.chat_completion = chat_completion
        print("[patch] ManagedServer: support chat_template_kwargs in extra_body")

    # Patch 2: fix _find_extending_node and _compute_input_ids to normalize out
    # Qwen3's empty think prefill before prefix matching. The model echoes the
    # prefill (<think>\n\n</think>\n\n or <think>\n</think>\n\n) in its completion,
    # but the chat template strips it when re-rendering prior turns, causing the
    # prefix match to fail every turn.
    import re as _re
    _EMPTY_THINK_RE = _re.compile(r"<think>\n*</think>\n\n")

    def _normalize_think(text):
        return _EMPTY_THINK_RE.sub("", text)

    if not getattr(_ManagedServer._find_extending_node, "_think_prefill_patched", False):

        def _find_extending_node(self, input_text):
            if not self.current_nodes:
                return None
            norm_input = _normalize_think(input_text)
            for node in self.current_nodes:
                if norm_input.startswith(_normalize_think(node.full_text)):
                    return node
            return None

        _find_extending_node._think_prefill_patched = True
        _ManagedServer._find_extending_node = _find_extending_node

        _orig_compute_input_ids = _ManagedServer._compute_input_ids

        def _compute_input_ids(self, input_text, extending_node):
            if extending_node is not None:
                norm_input = _normalize_think(input_text)
                norm_existing = _normalize_think(extending_node.full_text)
                new_suffix = norm_input[len(norm_existing):]
                if new_suffix:
                    new_tokens = self.tokenizer.encode(new_suffix, add_special_tokens=False)
                    return extending_node.tokens + new_tokens
                return extending_node.tokens.copy()
            return self.tokenizer.encode(input_text, add_special_tokens=True)

        _ManagedServer._compute_input_ids = _compute_input_ids
        print("[patch] ManagedServer: Qwen3 think prefill normalization for extend")

_patch_managed_server()

# Silence vLLM's hermes_tool_parser ERROR logs by patching extract_tool_calls
# to suppress its logger. vLLM configures its own logging on import so
# setLevel/filters applied before import get reset.
def _silence_hermes_tool_parser():
    try:
        from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
        import logging as _logging
        _logging.getLogger("vllm.tool_parsers.hermes_tool_parser").setLevel(_logging.CRITICAL)
    except Exception:
        pass

_silence_hermes_tool_parser()

from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.agent_loop import AgentResult
from environments.tool_context import ToolContext
from tools.terminal_tool import (
    register_task_env_overrides,
    clear_task_env_overrides,
    cleanup_vm,
)

logger = logging.getLogger(__name__)

# Add endless-terminals to path for imports
ENDLESS_TERMINALS_PATH = os.getenv(
    "ENDLESS_TERMINALS_PATH",
    str(Path.home() / "Desktop" / "Projects" / "endless-terminals")
)
sys.path.insert(0, ENDLESS_TERMINALS_PATH)


class EndlessTerminalsEnvConfig(HermesAgentEnvConfig):
    """Configuration for Endless Terminals environment."""

    # Dataset settings
    use_dataset: bool = Field(
        default=True,
        description="Load tasks from HuggingFace dataset (recommended). If False, generate procedurally."
    )
    dataset_name: str = Field(
        default="obiwan96/endless-terminals-train",
        description="HuggingFace dataset name"
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use"
    )
    dataset_cache_dir: str = Field(
        default="~/.cache/huggingface/datasets",
        description="HuggingFace datasets cache directory"
    )
    tasks_base_dir: str = Field(
        default="",
        description="Base directory containing task_* folders. If empty, uses paths from dataset."
    )

    # Test execution
    test_timeout_s: int = Field(default=60, description="Test execution timeout (seconds)")

    # Docker image fallback
    default_docker_image: str = Field(
        default="ubuntu:22.04",
        description="Default Docker image if container.def parsing fails"
    )

    # Agent defaults
    max_agent_turns: int = Field(default=32, description="Max turns for agent (increased for long traces)")

    # Evaluation settings
    num_eval_tasks: int = Field(
        default=10,
        description="Number of tasks to run during periodic evaluation"
    )
    eval_split_ratio: float = Field(
        default=0.1,
        description="Fraction of dataset to hold out for evaluation (0.0-1.0)"
    )
    overfit_task_index: Optional[int] = Field(
        default=None,
        description="If set, always sample this dataset index (for overfitting tests)"
    )

    max_concurrent_containers: int = Field(
        default=16,
        description="Max number of Docker containers running simultaneously"
    )


class EndlessTerminalsEnv(HermesAgentBaseEnv):
    """
    Endless Terminals environment using pre-generated HuggingFace dataset.

    Loads terminal tasks from dataset, runs agent with terminal tools,
    and scores by executing tests in the agent's sandbox using ToolContext.
    """

    name = "endless_terminals_env"
    env_config_cls = EndlessTerminalsEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[EndlessTerminalsEnvConfig, List["APIServerConfig"]]:
        """
        Default configuration for Endless Terminals environment.

        This is used when no config file is provided, but note that when using
        --config, the YAML is loaded differently and this may not be called.
        """
        from atroposlib.envs.server_handling.server_manager import APIServerConfig

        env_config = EndlessTerminalsEnvConfig(
            enabled_toolsets=["terminal", "file"],
            max_agent_turns=32,
            terminal_backend="local",
            use_dataset=True,
            tasks_base_dir="",
            group_size=1,
            total_steps=1,
            use_wandb=False,
        )

        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4.5",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset = None
        self._train_dataset = None
        self._eval_dataset = None
        self._dataset_indices = []
        self._current_index = 0

        # Metrics tracking for wandb - single buffer with dicts
        self._metrics_buffer = []

        # Semaphore to cap concurrent Docker containers
        self._container_sem = asyncio.Semaphore(self.config.max_concurrent_containers)

        # Debug: check server config
        if hasattr(self, 'server') and hasattr(self.server, 'servers'):
            for i, srv in enumerate(self.server.servers):
                logger.debug(f"Server {i}: model_name={getattr(srv.config, 'model_name', 'NONE')}")

    async def setup(self):
        """Load dataset from HuggingFace or local directory."""
        if not self.config.use_dataset:
            logger.info("Using procedural task generation (not implemented yet)")
            return

        # If tasks_base_dir is set, load from local directory instead of HuggingFace
        if self.config.tasks_base_dir:
            tasks_base = Path(os.path.expanduser(self.config.tasks_base_dir))

            # Resolve to absolute path if relative
            if not tasks_base.is_absolute():
                tasks_base = Path.cwd() / tasks_base

            tasks_base = tasks_base.resolve()

            if not tasks_base.exists():
                raise RuntimeError(f"tasks_base_dir not found: {tasks_base}")

            logger.info(f"Loading tasks from local directory: {tasks_base}")

            # Find all task_* directories
            task_dirs = sorted(tasks_base.glob("task_*"))
            logger.info(f"Found {len(task_dirs)} task directories")

            if not task_dirs:
                # Debug: show what's actually in the directory
                all_items = list(tasks_base.iterdir())
                logger.warning(f"Directory contains {len(all_items)} items:")
                for item in all_items[:10]:
                    logger.warning(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                raise RuntimeError(f"No task_* directories found in {tasks_base}")

            # Create fake dataset items (just the directory paths)
            self._dataset = [
                {
                    "description": f"Task from {task_dir.name}",
                    "extra_info": {"task_dir": str(task_dir)},
                }
                for task_dir in task_dirs
            ]

            logger.info(f"Loaded {len(self._dataset)} tasks from local directory")

            self._split_dataset()
            return

        # Otherwise, load from HuggingFace
        logger.info(f"Loading dataset from HuggingFace: {self.config.dataset_name}")

        try:
            from datasets import load_dataset

            self._dataset = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    cache_dir=os.path.expanduser(self.config.dataset_cache_dir)
                )
            )

            logger.info(f"Loaded {len(self._dataset)} tasks from HuggingFace")

            self._split_dataset()

        except Exception as e:
            logger.error(f"ERROR loading dataset: {e}")
            raise

    def _split_dataset(self):
        """Split dataset into train and eval sets based on eval_split_ratio."""
        if self._dataset is None or len(self._dataset) == 0:
            raise RuntimeError("Cannot split empty dataset")

        total_size = len(self._dataset)
        eval_size = int(total_size * self.config.eval_split_ratio)
        train_size = total_size - eval_size

        all_indices = list(range(total_size))
        random.shuffle(all_indices)

        train_indices = all_indices[:train_size]
        eval_indices = all_indices[train_size:]

        if isinstance(self._dataset, list):
            self._train_dataset = [self._dataset[i] for i in train_indices]
            self._eval_dataset = [self._dataset[i] for i in eval_indices]
        else:
            self._train_dataset = self._dataset.select(train_indices)
            self._eval_dataset = self._dataset.select(eval_indices)

        self._dataset_indices = list(range(len(self._train_dataset)))
        random.shuffle(self._dataset_indices)
        self._current_index = 0

        logger.info(
            f"Split dataset: {len(self._train_dataset)} train, "
            f"{len(self._eval_dataset)} eval "
            f"(ratio={self.config.eval_split_ratio:.1%})"
        )

    async def get_next_item(self) -> Item:
        """Sample next task from training dataset."""
        if self._train_dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        # Get next task (with wraparound)
        if self.config.overfit_task_index is not None:
            # Index into the full dataset (before train/eval split) for stability
            idx = self.config.overfit_task_index % len(self._dataset)
            task = self._dataset[idx]
        else:
            idx = self._dataset_indices[self._current_index]
            self._current_index += 1
            if self._current_index >= len(self._dataset_indices):
                random.shuffle(self._dataset_indices)
                self._current_index = 0
                logger.info("Reshuffled dataset (completed one epoch)")
            task = self._train_dataset[idx]

        # Extract task directory path
        task_dir = task.get("extra_info", {}).get("task_dir")
        if not task_dir:
            task_dir = task.get("reward_spec", {}).get("ground_truth")

        # Resolve task directory path
        if task_dir:
            task_dir_path = Path(task_dir)
            # If tasks_base_dir is configured and path doesn't exist, reconstruct it
            if self.config.tasks_base_dir and not task_dir_path.exists():
                original_path = Path(task_dir)
                task_name = original_path.name
                task_dir_path = Path(os.path.expanduser(self.config.tasks_base_dir)) / task_name
        else:
            logger.error("No task directory path found in dataset item")
            return await self.get_next_item()

        # Verify directory exists
        if not task_dir_path.exists():
            logger.warning(f"Task dir not found: {task_dir_path}")
            logger.warning("Hint: Set tasks_base_dir to directory containing task_* folders")
            return await self.get_next_item()  # Try next task

        # Look for test file in tests/ subdirectory first, then at root
        final_test = task_dir_path / "tests" / "test_final_state.py"
        if not final_test.exists():
            final_test = task_dir_path / "test_final_state.py"

        # Verify test file exists
        if not final_test.exists():
            logger.warning(f"Missing test file in {task_dir_path} (checked tests/ and root)")
            return await self.get_next_item()

        # Parse container.def to extract Docker image
        # Check environment/ subdirectory first, then root
        container_def = task_dir_path / "environment" / "container.def"
        if not container_def.exists():
            container_def = task_dir_path / "container.def"
        docker_image = self._parse_docker_image_from_def(container_def)

        # Try to load description from instruction.md or task.json
        # Always prefer instruction.md on disk over dataset placeholder descriptions
        description = ""
        instruction_md = task_dir_path / "instruction.md"
        if instruction_md.exists():
            try:
                description = instruction_md.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to load instruction.md for {task_dir_path.name}: {e}")

        # Fallback to task.json in environment/
        if not description:
            task_json = task_dir_path / "environment" / "task.json"
            if task_json.exists():
                try:
                    import json
                    task_data = json.loads(task_json.read_text())
                    description = task_data.get("description", "") or task_data.get("instruction", "")
                except Exception as e:
                    logger.warning(f"Failed to load task.json for {task_dir_path.name}: {e}")

        if not description:
            description = f"Complete the task in {task_dir_path.name}"

        return {
            "task_id": f"{task_dir_path.name}",
            "task_name": task_dir_path.name,
            "description": description,
            "task_dir": str(task_dir_path),
            "final_test": str(final_test),
            "docker_image": docker_image,
            "dataset_index": idx,
        }

    def format_prompt(self, item: Item) -> str:
        """Return the task description for the agent."""
        return str(item.get("description", ""))

    def _parse_docker_image_from_def(self, container_def_path: Path) -> str:
        """
        Extract the Docker base image for a task.

        Tries sources in order:
        1. Dockerfile in the same directory (or parent environment/ dir)
           → parse "FROM <image>" line
        2. container.def with "Bootstrap: docker" → parse "From: <image>"
        3. Falls back to default_docker_image

        Singularity defs with "Bootstrap: localimage" / "From: ./foo.sif"
        are skipped since those aren't valid Docker image names.
        """
        task_dir = container_def_path.parent

        # --- Try Dockerfile first (most reliable for Docker backend) ---
        for dockerfile_path in [
            task_dir / "Dockerfile",
            task_dir.parent / "Dockerfile",   # task_dir might be environment/
        ]:
            if dockerfile_path.exists():
                try:
                    content = dockerfile_path.read_text()
                    match = re.search(
                        r'^FROM\s+(\S+)', content, re.MULTILINE | re.IGNORECASE
                    )
                    if match:
                        image = match.group(1).strip()
                        logger.info(f"Extracted Docker image from Dockerfile: {image}")
                        return image
                except Exception as e:
                    logger.warning(f"Failed to parse {dockerfile_path}: {e}")

        # --- Fallback: container.def with Bootstrap: docker ---
        if container_def_path.exists():
            try:
                content = container_def_path.read_text()
                # Only use From: line if it's a docker-based def (not localimage/sif)
                bootstrap_match = re.search(
                    r'^Bootstrap:\s*(\S+)', content, re.MULTILINE | re.IGNORECASE
                )
                bootstrap = bootstrap_match.group(1).lower() if bootstrap_match else ""

                if bootstrap == "docker":
                    from_match = re.search(
                        r'^From:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE
                    )
                    if from_match:
                        image = from_match.group(1).strip()
                        logger.info(f"Extracted Docker image from container.def: {image}")
                        return image
                else:
                    logger.debug(
                        f"Skipping container.def (Bootstrap: {bootstrap}), not a Docker source"
                    )
            except Exception as e:
                logger.warning(f"Failed to parse {container_def_path}: {e}")

        logger.warning(f"Could not extract Docker image from {task_dir}, using default")
        return self.config.default_docker_image

    # Per-task lock to prevent parallel builds of the same image
    _build_locks: Dict[str, threading.Lock] = {}
    _build_locks_lock = threading.Lock()

    def _build_task_docker_image(self, item: Item) -> str:
        """
        Build a Docker image from the task's Dockerfile.

        Each task has a custom Dockerfile that installs dependencies (pytest, etc.)
        and sets up the initial filesystem state. We build it once and cache by
        task name so repeated runs reuse the image.

        Thread-safe: uses per-task locks so parallel collect_trajectory() calls
        for the same task (group_size > 1) don't race on docker build.

        Returns the built image tag, or falls back to default_docker_image.
        """
        import subprocess

        task_name = item.get("task_name", "unknown")
        task_dir = item.get("task_dir", "")

        # Look for Dockerfile
        dockerfile_paths = [
            Path(task_dir) / "environment" / "Dockerfile",
            Path(task_dir) / "Dockerfile",
        ]
        dockerfile = None
        for p in dockerfile_paths:
            if p.exists():
                dockerfile = p
                break

        if not dockerfile:
            logger.debug(f"Task {task_name}: no Dockerfile found, using default image")
            return self.config.default_docker_image

        # Build with a deterministic tag so we cache across runs
        image_tag = f"hermes-et:{task_name}"

        # Acquire per-task lock so only one thread builds, others wait
        with self._build_locks_lock:
            if task_name not in self._build_locks:
                self._build_locks[task_name] = threading.Lock()
            build_lock = self._build_locks[task_name]

        with build_lock:
            # Check if image already exists (another thread may have built it)
            try:
                result = subprocess.run(
                    ["docker", "image", "inspect", image_tag],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.debug(f"Task {task_name}: reusing cached image {image_tag}")
                    return image_tag
            except Exception:
                pass

            # Build the image with BuildKit (required for --mount in Dockerfiles)
            logger.info(f"Task {task_name}: building Docker image from {dockerfile}...")
            build_env = {**os.environ, "DOCKER_BUILDKIT": "1"}
            try:
                result = subprocess.run(
                    ["docker", "build", "-t", image_tag, "-f", str(dockerfile), str(dockerfile.parent)],
                    capture_output=True, text=True, timeout=300, env=build_env,
                )
                if result.returncode == 0:
                    logger.info(f"Task {task_name}: built image {image_tag}")
                    return image_tag
                else:
                    logger.error(
                        f"Task {task_name}: docker build failed (exit {result.returncode}): "
                        f"{result.stderr[-500:]}"
                    )
                    return self.config.default_docker_image
            except subprocess.TimeoutExpired:
                logger.error(f"Task {task_name}: docker build timed out")
                return self.config.default_docker_image
            except Exception as e:
                logger.error(f"Task {task_name}: docker build error: {e}")
                return self.config.default_docker_image

    async def collect_trajectories(self, item: Item):
        """
        Override to aggregate inference_logprobs into the ScoredDataGroup.

        The base atropos collect_trajectories aggregates tokens/masks/scores but
        silently drops inference_logprobs. Tinker-atropos needs inference_logprobs
        for importance-sampling loss. We call collect_trajectory group_size times
        ourselves and build the ScoredDataGroup directly.
        """
        # Resolve toolsets once for the whole group
        self._current_group_tools = self._resolve_tools_for_group()

        tasks = [self.collect_trajectory(item) for _ in range(self.config.group_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        group = ScoredDataGroup()
        group["tokens"] = []
        group["masks"] = []
        group["scores"] = []
        group["inference_logprobs"] = []
        group["messages"] = []

        backlog = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("collect_trajectory failed: %s", result)
                continue
            scored_item, item_backlog = result
            if scored_item is None:
                continue
            group["tokens"].append(scored_item["tokens"])
            group["masks"].append(scored_item["masks"])
            group["scores"].append(scored_item["scores"])
            group["inference_logprobs"].append(scored_item.get("inference_logprobs", [1.0] * len(scored_item["tokens"])))
            if scored_item.get("messages"):
                group["messages"].append(scored_item["messages"])
            backlog.extend(item_backlog)

        if not group["tokens"]:
            return None, backlog

        return group, backlog

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Override to register per-task Docker image before running the agent.

        Builds the task's Dockerfile into a Docker image (cached by task name),
        then registers it as the container for this rollout's task_id.
        """
        import uuid
        from environments.agent_loop import HermesAgentLoop

        task_id = str(uuid.uuid4())
        task_name = item.get("task_name", "unknown")

        # Build task-specific Docker image (cached across runs)
        docker_image = await asyncio.get_event_loop().run_in_executor(
            None, self._build_task_docker_image, item
        )

        logger.debug(f"collect_trajectory START for {task_name}")

        # Register image override for this task_id (set both docker_image and
        # modal_image so the correct one is used regardless of backend)
        logger.debug(f"Registering Docker image: {docker_image}")
        register_task_env_overrides(task_id, {
            "docker_image": docker_image,
            "modal_image": docker_image,
        })
        logger.info(
            f"Task {task_name}: registered Docker image {docker_image} for task_id {task_id[:8]}"
        )
        logger.debug("Docker image registered")

        try:
            # Get group-level tools (resolved once in collect_trajectories)
            logger.debug("Resolving tools...")
            if self._current_group_tools is None:
                tools, valid_names = self._resolve_tools_for_group()
            else:
                tools, valid_names = self._current_group_tools
            logger.debug(f"Tools resolved: {len(tools)} tools")

            # Build initial messages
            logger.debug("Building initial messages...")
            messages: List[Dict[str, Any]] = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": self.format_prompt(item)})
            logger.debug("Messages built, starting agent loop...")

            # Run the agent loop
            result: AgentResult
            managed_state: Optional[Dict[str, Any]] = None
            async with self._container_sem:
                if self._use_managed_server():
                    try:
                        # No tool_call_parser — let the chat template render tool calls
                        # natively so full_text is consistent across turns (enabling
                        # multi-turn sequence extension). agent_loop has a fallback
                        # to parse raw <tool_call> content.
                        managed_ctx = self.server.managed_server(
                            tokenizer=self.tokenizer,
                            preserve_think_blocks=True,
                        )

                        async with managed_ctx as managed:
                            agent = HermesAgentLoop(
                                server=managed,
                                tool_schemas=tools,
                                valid_tool_names=valid_names,
                                max_turns=self.config.max_agent_turns,
                                task_id=task_id,
                                temperature=self.config.agent_temperature,
                                max_tokens=self.config.max_token_length,
                                extra_body=self.config.extra_body,
                            )
                            result = await agent.run(messages)

                            managed_state = managed.get_state()
                    except NotImplementedError:
                        logger.warning("ManagedServer not available. Falling back to direct server mode.")
                        agent = HermesAgentLoop(
                            server=self.server,
                            tool_schemas=tools,
                            valid_tool_names=valid_names,
                            max_turns=self.config.max_agent_turns,
                            task_id=task_id,
                            temperature=self.config.agent_temperature,
                            max_tokens=self.config.max_token_length,
                            extra_body=self.config.extra_body,
                        )
                        result = await agent.run(messages)
                else:
                    # Phase 1: OpenAI server
                    agent = HermesAgentLoop(
                        server=self.server,
                        tool_schemas=tools,
                        valid_tool_names=valid_names,
                        max_turns=self.config.max_agent_turns,
                        task_id=task_id,
                        temperature=self.config.agent_temperature,
                        max_tokens=self.config.max_token_length,
                        extra_body=self.config.extra_body,
                    )
                    result = await agent.run(messages)

            # Skip reward computation if agent produced no output
            only_system_and_user = all(
                msg.get("role") in ("system", "user") for msg in result.messages
            )
            if result.turns_used == 0 or only_system_and_user:
                logger.warning(
                    "Agent loop produced no output (turns=%d). Skipping trajectory.",
                    result.turns_used,
                )
                # Return None to skip this trajectory (likely an API failure)
                return None, []
            else:
                # Compute reward using ToolContext
                ctx = ToolContext(task_id)
                try:
                    reward = await self.compute_reward(item, result, ctx)
                except Exception as e:
                    logger.error("compute_reward failed: %s", e)
                    reward = 0.0
                finally:
                    ctx.cleanup()


            # Track metrics for wandb logging
            task_metrics = {
                "test_passed": 1.0 if reward > 0.5 else 0.0,
                "reward": reward,
                "turns_used": result.turns_used,
                "finished_naturally": result.finished_naturally,
                "docker_image": docker_image,
                "num_tool_errors": len(result.tool_errors),
            }

            # Include detailed tool errors if any occurred
            if result.tool_errors:
                task_metrics["tool_errors"] = [
                    {
                        "turn": err.turn,
                        "tool": err.tool_name,
                        "error": err.error[:200],
                    }
                    for err in result.tool_errors
                ]

            self._metrics_buffer.append(task_metrics)

            # ============================================================================
            # Build ScoredDataItem from result
            # ============================================================================
            # Phase 2: Extract tokens/masks/logprobs from first SequenceNode
            # Phase 1: Create placeholder tokens for OpenAI-style servers
            # ============================================================================
            # Build return value.
            # The base collect_trajectories() calls this group_size times and
            # aggregates using .append(), so we must return a ScoredDataItem
            # (scalar scores, single token list, etc.) -- NOT a ScoredDataGroup
            # with list-valued fields, which would cause double-wrapping and
            # "unhashable type: list" errors in handle_send_to_api.
            nodes = (managed_state or {}).get("nodes", []) if managed_state else []

            if nodes:
                # Phase 2: Use last node — ManagedServer extends the sequence each turn,
                # carrying forward masked_tokens so all assistant turns are unmasked.
                node = nodes[-1]
                tokens = node.tokens
                masks = node.masked_tokens
                if hasattr(node, "logprobs") and node.logprobs:
                    inference_logprobs = node.logprobs
                    real_logprobs = [lp for lp in inference_logprobs if lp != 1.0]
                    logger.info(f"Phase 2: {len(tokens)} tokens, {len(real_logprobs)} generated across {len(nodes)} nodes, logprob_mean={sum(real_logprobs)/max(len(real_logprobs),1):.3f}")
                else:
                    inference_logprobs = [1.0] * len(tokens)
                    logger.warning(f"Phase 2: node has no logprobs! Falling back to placeholders.")
            else:
                # Phase 1: placeholder tokens
                logger.warning(f"Phase 1 fallback: managed_state empty, using placeholder logprobs. IS loss will be zero!")
                full_text = "\n".join(
                    msg.get("content", "") for msg in result.messages if msg.get("content")
                )
                if self.tokenizer:
                    tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
                else:
                    tokens = list(range(min(len(full_text) // 4, 128)))
                masks = [-100] + tokens[1:]
                inference_logprobs = [1.0] * len(tokens)

            if not tokens:
                return None, []
            
            print(reward)

            scored_item: ScoredDataItem = {
                "tokens": tokens,
                "masks": masks,
                "scores": reward,
                "messages": result.messages,
                "inference_logprobs": inference_logprobs,
            }

            return scored_item, []

        finally:
            # Clean up task overrides and sandbox
            clear_task_env_overrides(task_id)
            try:
                cleanup_vm(task_id)
            except Exception as e:
                logger.debug(f"VM cleanup for {task_id[:8]}: {e}")

    async def compute_reward(
        self,
        item: Item,
        result: AgentResult,
        ctx: ToolContext
    ) -> float:
        """
        Run final tests in the agent's sandbox and return binary reward.

        Uses ToolContext to execute pytest in the SAME sandbox the agent used,
        following the Terminal Bench 2 verification pattern. No separate
        Apptainer execution needed.

        Returns 1.0 if tests pass, 0.0 otherwise.
        """
        task_name = item.get("task_name", "unknown")
        final_test_path = Path(item.get("final_test", ""))

        if not final_test_path.exists():
            logger.error(f"Task {task_name}: test file not found at {final_test_path}")
            return 0.0

        logger.info(f"Task {task_name}: running tests in sandbox...")

        try:
            # Run tests in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            reward = await loop.run_in_executor(
                None,
                self._run_tests_in_sandbox,
                final_test_path,
                ctx,
                task_name,
            )

            status = "PASS" if reward == 1.0 else "FAIL"
            logger.info(f"Task {task_name}: {status} (reward={reward})")
            return reward

        except Exception as e:
            logger.error(f"Task {task_name}: test execution failed: {e}", exc_info=True)
            return 0.0

    def _run_tests_in_sandbox(
        self,
        test_file_path: Path,
        ctx: ToolContext,
        task_name: str,
    ) -> float:
        """
        Upload test file to sandbox and execute pytest with granular scoring.

        Runs in thread pool (via run_in_executor) to avoid blocking the event loop
        with synchronous ToolContext calls.

        Scoring (0.0 to 1.0):
          - Base: fraction of tests passed (e.g., 3/5 = 0.6)
          - All tests pass: 1.0

        This granular approach ensures score variance within GRPO groups,
        avoiding the "Scores are the same in a group, skipping..." problem
        that occurs with binary 0/1 rewards.

        Args:
            test_file_path: Local path to test_final_state.py
            ctx: ToolContext scoped to the agent's sandbox
            task_name: For logging

        Returns:
            Float between 0.0 and 1.0 based on test pass ratio
        """
        try:
            # Upload test file to a writable temp directory
            # (Using /workspace fails on macOS local backend because / is read-only)
            test_content = test_file_path.read_text()
            test_dir = f"/tmp/_hermes_test_{os.getpid()}"
            test_path = f"{test_dir}/test_final_state.py"
            ctx.write_file(test_path, test_content)
            logger.debug(f"Task {task_name}: uploaded test file to {test_path}")

            # Run pytest -v for per-test pass/fail details
            result = ctx.terminal(
                f"cd {test_dir} && python3 -m pytest -v test_final_state.py 2>&1",
                timeout=self.config.test_timeout_s,
            )

            exit_code = result.get("exit_code", -1)
            output = result.get("output", "")

            if exit_code == 0:
                logger.debug(f"Task {task_name}: all tests passed")
                return 1.0

            # Parse pytest -v output for granular scoring
            # Lines like: "test_final_state.py::test_name PASSED"
            # or:         "test_final_state.py::test_name FAILED"
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            errored = len(re.findall(r" ERROR", output))
            total = passed + failed + errored

            if total > 0:
                # Score is fraction of tests passed
                reward = passed / total
                logger.info(
                    f"Task {task_name}: {passed}/{total} tests passed "
                    f"(reward={reward:.3f})"
                )
                return reward
            else:
                # Couldn't parse output — fall back to binary
                # Also try the summary line: "X passed, Y failed"
                summary_match = re.search(
                    r"(\d+) passed(?:.*?(\d+) failed)?", output
                )
                if summary_match:
                    p = int(summary_match.group(1))
                    f = int(summary_match.group(2) or 0)
                    total = p + f
                    if total > 0:
                        reward = p / total
                        logger.info(
                            f"Task {task_name}: {p}/{total} tests passed "
                            f"from summary (reward={reward:.3f})"
                        )
                        return reward

                output_preview = output[-500:] if output else "(no output)"
                logger.info(
                    f"Task {task_name}: tests failed, couldn't parse counts "
                    f"(exit_code={exit_code})\n{output_preview}"
                )
                return 0.0

        except Exception as e:
            logger.error(f"Task {task_name}: error running tests: {e}")
            return 0.0

    async def evaluate(self):
        """
        Periodic evaluation on holdout eval set.

        Runs the agent on num_eval_tasks from the held-out eval set
        (never seen during training). Returns metrics for wandb logging.
        """
        if self._eval_dataset is None:
            logger.warning("Cannot evaluate: eval dataset not loaded")
            return {}

        if len(self._eval_dataset) == 0:
            logger.warning("Eval dataset is empty")
            return {}

        # Use min of num_eval_tasks and actual eval set size
        num_tasks = min(self.config.num_eval_tasks, len(self._eval_dataset))
        logger.info(f"Starting evaluation on {num_tasks} held-out tasks...")

        eval_metrics = {
            "rewards": [],
            "passes": [],
            "turns": [],
            "natural_finishes": [],
        }

        # Sample from eval set (holdout)
        import random
        eval_indices = random.sample(range(len(self._eval_dataset)), num_tasks)

        for idx in eval_indices:
            task = self._eval_dataset[idx]

            # Build item using same logic as get_next_item
            task_dir = task.get("extra_info", {}).get("task_dir")
            if not task_dir:
                task_dir = task.get("reward_spec", {}).get("ground_truth")

            if not task_dir:
                continue

            task_dir_path = Path(task_dir)
            if self.config.tasks_base_dir and not task_dir_path.exists():
                original_path = Path(task_dir)
                task_name = original_path.name
                task_dir_path = Path(os.path.expanduser(self.config.tasks_base_dir)) / task_name

            if not task_dir_path.exists():
                continue

            # Find test file
            final_test = task_dir_path / "tests" / "test_final_state.py"
            if not final_test.exists():
                final_test = task_dir_path / "test_final_state.py"
            if not final_test.exists():
                continue

            # Parse Docker image
            container_def = task_dir_path / "environment" / "container.def"
            if not container_def.exists():
                container_def = task_dir_path / "container.def"
            docker_image = self._parse_docker_image_from_def(container_def)

            # Load description
            description = task.get("description", "")
            instruction_md = task_dir_path / "instruction.md"
            if not description and instruction_md.exists():
                try:
                    description = instruction_md.read_text().strip()
                except Exception:
                    pass

            item = {
                "description": description,
                "final_test": str(final_test),
                "docker_image": docker_image,
            }

            # Run agent on this task
            try:
                import uuid
                task_id = str(uuid.uuid4())

                # Register task environment
                from model_tools import register_task_env_overrides
                register_task_env_overrides(task_id, {"modal_image": docker_image})

                # Build messages
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": description or "Complete the task."},
                ]

                # Get tools
                from model_tools import get_tool_definitions
                tools = get_tool_definitions(self.config.enabled_toolsets)
                valid_names = {t["function"]["name"] for t in tools}

                # Run agent
                from environments.agent_loop import HermesAgentLoop
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                    extra_body=self.config.extra_body,
                )
                result = await agent.run(messages)

                # Compute reward
                from environments.tool_context import ToolContext
                ctx = ToolContext(task_id)
                try:
                    reward = await self.compute_reward(item, result, ctx)
                except Exception as e:
                    logger.warning(f"Eval reward computation failed: {e}")
                    reward = 0.0
                finally:
                    ctx.cleanup()

                # Track metrics
                eval_metrics["rewards"].append(reward)
                eval_metrics["passes"].append(1.0 if reward > 0.5 else 0.0)
                eval_metrics["turns"].append(result.turns_used)
                eval_metrics["natural_finishes"].append(1.0 if result.finished_naturally else 0.0)

            except Exception as e:
                logger.error(f"Eval task failed: {e}")
                continue
            finally:
                # Cleanup
                from model_tools import clear_task_env_overrides, cleanup_vm
                clear_task_env_overrides(task_id)
                cleanup_vm(task_id)

        # Aggregate metrics
        if not eval_metrics["rewards"]:
            logger.warning("No eval tasks completed successfully")
            return {}

        aggregated = {
            "eval/pass_rate": sum(eval_metrics["passes"]) / len(eval_metrics["passes"]),
            "eval/avg_reward": sum(eval_metrics["rewards"]) / len(eval_metrics["rewards"]),
            "eval/avg_turns": sum(eval_metrics["turns"]) / len(eval_metrics["turns"]),
            "eval/natural_finish_rate": sum(eval_metrics["natural_finishes"]) / len(eval_metrics["natural_finishes"]),
            "eval/num_tasks": len(eval_metrics["rewards"]),
        }

        logger.info(f"Evaluation complete: pass_rate={aggregated['eval/pass_rate']:.2%}, avg_turns={aggregated['eval/avg_turns']:.1f}")
        return aggregated

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log Endless Terminals specific metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Aggregate metrics from buffer
        if self._metrics_buffer:
            # Test pass rate
            test_passes = [m["test_passed"] for m in self._metrics_buffer]
            accuracy = sum(test_passes) / len(test_passes)
            wandb_metrics["train/accuracy"] = accuracy
            wandb_metrics["endless_terminals/test_pass_rate"] = accuracy
            wandb_metrics["endless_terminals/num_tests_passed"] = sum(test_passes)
            wandb_metrics["endless_terminals/num_tests_total"] = len(test_passes)

            # Turns used statistics
            turns = [m["turns_used"] for m in self._metrics_buffer]
            wandb_metrics["endless_terminals/avg_turns_used"] = sum(turns) / len(turns)
            wandb_metrics["endless_terminals/max_turns_used"] = max(turns)
            wandb_metrics["endless_terminals/min_turns_used"] = min(turns)

            # Natural finish rate (did model stop on its own vs hitting max turns)
            natural_finishes = [1.0 if m["finished_naturally"] else 0.0 for m in self._metrics_buffer]
            wandb_metrics["endless_terminals/natural_finish_rate"] = sum(natural_finishes) / len(natural_finishes)

            # Tool error statistics
            total_tool_errors = sum(m["num_tool_errors"] for m in self._metrics_buffer)
            wandb_metrics["endless_terminals/total_tool_errors"] = total_tool_errors
            wandb_metrics["endless_terminals/avg_tool_errors_per_task"] = total_tool_errors / len(self._metrics_buffer)

            # Docker image distribution (count unique images used)
            docker_images = [m["docker_image"] for m in self._metrics_buffer]
            unique_images = set(docker_images)
            wandb_metrics["endless_terminals/num_unique_docker_images"] = len(unique_images)

            # Log most common errors if any
            all_errors = []
            for m in self._metrics_buffer:
                if "tool_errors" in m:
                    all_errors.extend(m["tool_errors"])

            if all_errors:
                # Count error types
                error_tools = {}
                for err in all_errors:
                    tool = err["tool"]
                    error_tools[tool] = error_tools.get(tool, 0) + 1

                # Log top 3 error-prone tools
                for i, (tool, count) in enumerate(sorted(error_tools.items(), key=lambda x: x[1], reverse=True)[:3]):
                    wandb_metrics[f"endless_terminals/errors_by_tool/{tool}"] = count

            # Clear buffer after logging
            self._metrics_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    EndlessTerminalsEnv.cli()
