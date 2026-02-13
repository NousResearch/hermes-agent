"""
Endless Terminals Environment for Hermes-Agent + Atropos RL.

Runs terminal tasks from the Endless Terminals dataset.
Supports three modes:
  1. Local directory: tasks from a local folder of task_* dirs (default)
  2. HuggingFace dataset: tasks from a HF dataset
  3. Procedural: generate tasks on-the-fly via LLM (requires vLLM)

Each task provides a Dockerfile that defines the initial environment.
The agent solves the task using terminal commands inside a Docker container.
Scoring is done by running pytest on `test_final_state.py` in the container.

Run (standalone process mode):
  python -m atropos.envs.endless_terminals_env process \
    --env.use_wandb false \
    --env.total_steps 100 \
    --env.group_size 4

Run (Tinker serve mode):
  # Terminal 1: run-api
  # Terminal 2: python launch_training.py --config configs/endless_terminals.yaml
  # Terminal 3:
  TINKER_CONFIG=configs/endless_terminals.yaml \
  ENDLESS_TERMINALS_DIR=/path/to/endless-terminals \
    python -m atropos.envs.endless_terminals_env serve
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import APIServerConfig, Item

from ..agent import AgentConfig
from ..backends.docker_direct_backend import (
    DockerDirectBackend,
    build_docker_image,
    docker_image_exists,
)
from ..tools import ToolCall
from .agent_env import AgentEnv, AgentEnvConfig

load_dotenv()


# ---------------------------------------------------------------------------
# Tinker integration
# ---------------------------------------------------------------------------
# When TINKER_CONFIG is set, we load model/training params from the Tinker YAML.
# Custom env fields (ENDLESS_TERMINALS_DIR, etc.) are always read from env vars.
TINKER_CONFIG = os.getenv("TINKER_CONFIG", "")


def _load_tinker_config():
    """Load TinkerAtroposConfig if available, else return None."""
    if not TINKER_CONFIG:
        return None
    config_path = Path(TINKER_CONFIG)
    if not config_path.exists():
        print(f"[EndlessTerminalsEnv] TINKER_CONFIG={TINKER_CONFIG} not found, ignoring", flush=True)
        return None
    try:
        from tinker_atropos.config import TinkerAtroposConfig
        config = TinkerAtroposConfig.from_yaml(config_path)
        print(f"[EndlessTerminalsEnv] Loaded Tinker config from {config_path}", flush=True)
        return config
    except ImportError:
        print("[EndlessTerminalsEnv] tinker_atropos not installed, ignoring TINKER_CONFIG", flush=True)
        return None
    except Exception as e:
        print(f"[EndlessTerminalsEnv] Error loading Tinker config: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class EndlessTerminalsEnvConfig(AgentEnvConfig):
    """Configuration for Endless Terminals environment."""

    # ---- Local directory mode (primary) ----
    use_local_dir: bool = Field(
        default=True,
        description="Load tasks from a local directory of task_* folders.",
    )
    local_tasks_dir: str = Field(
        default="",
        description="Path to directory containing task_* folders. Required if use_local_dir=True.",
    )
    prebuild_images: bool = Field(
        default=False,
        description="Pre-build ALL Docker images during setup (slow but avoids build-during-training).",
    )
    max_concurrent_builds: int = Field(
        default=4,
        description="Max parallel Docker image builds during pre-build.",
    )

    # ---- HuggingFace dataset mode ----
    use_dataset: bool = Field(
        default=False,
        description="Load tasks from HuggingFace dataset.",
    )
    dataset_name: str = Field(
        default="obiwan96/endless-terminals-train",
        description="HuggingFace dataset name (if use_dataset=True)",
    )
    dataset_split: str = Field(default="train")
    dataset_cache_dir: str = Field(default="~/.cache/huggingface/datasets")
    tasks_base_dir: str = Field(
        default="",
        description="Base directory containing task_* folders (for dataset mode path resolution).",
    )

    # ---- Procedural generation mode ----
    task_gen_model: str = Field(default="Qwen/Qwen3-32B")
    task_gen_temperature: float = Field(default=1.0)
    task_gen_max_tokens: int = Field(default=2048)

    # ---- Container / scoring ----
    container_build_timeout_s: float = Field(default=600.0, description="Docker build timeout")
    test_timeout_s: int = Field(default=120, description="Test execution timeout (seconds)")
    keep_failed_tasks: bool = Field(default=False)

    # ---- Agent defaults ----
    agent_max_steps: int = Field(default=32)
    agent_temperature: float = Field(default=0.7)

    # ---- Docker image prefix ----
    docker_image_prefix: str = Field(
        default="endless-terminals",
        description="Docker image name prefix for built task images.",
    )

    # ---- Server defaults ----
    server_base_url: str = Field(default="http://127.0.0.1:8080")
    server_model: str = Field(default="hermes-4-36b")
    tokenizer_name: str = Field(default="NousResearch/Hermes-4.3-36B")


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class EndlessTerminalsEnv(AgentEnv[EndlessTerminalsEnvConfig]):
    """
    Endless Terminals environment.

    Each task:
      1. Has a Dockerfile defining the initial container state
      2. Has an instruction.md describing what the agent should do
      3. Has tests/test_final_state.py to verify completion

    Flow per trajectory:
      1. get_next_item() → picks a task
      2. setup_trajectory_workspace() → builds Docker image, registers with backend
      3. Agent solves task via terminal commands (docker exec in the container)
      4. verify_and_score_trajectory() → runs pytest in container, returns binary reward
    """

    name = "endless_terminals_env"
    env_config_cls = EndlessTerminalsEnvConfig

    def __init__(
        self,
        config: EndlessTerminalsEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self._iteration = 0

        # Local dir mode
        self._local_tasks: List[Dict[str, Any]] = []
        self._local_task_indices: List[int] = []
        self._local_current_index = 0

        # Eval split (held-out tasks)
        self._eval_tasks: List[Dict[str, Any]] = []

        # Training metrics
        self._train_scores_buffer: List[float] = []
        self._eval_metrics: List[tuple] = []

        # HF dataset mode
        self._dataset = None
        self._dataset_indices: List[int] = []
        self._dataset_current_index = 0

        # Docker image cache: task_name -> image_tag
        self._image_cache: Dict[str, str] = {}
        self._build_lock = asyncio.Lock()

    # ---- Config init (CLI) ----

    @classmethod
    def config_init(cls) -> Tuple[EndlessTerminalsEnvConfig, List[APIServerConfig]]:
        """
        Initialize config.

        Two modes:
          1. Tinker mode: TINKER_CONFIG env var points to a Tinker YAML.
             Model, training params, and server config come from the YAML.
          2. Standalone mode: Everything from env vars (ATROPOS_SERVER_*, etc.)

        In both modes, Endless Terminals-specific fields (ENDLESS_TERMINALS_DIR,
        PREBUILD_IMAGES, etc.) are always read from env vars.
        """
        tinker_cfg = _load_tinker_config()

        # ── Endless Terminals-specific fields (always from env vars) ──
        local_tasks_dir = os.getenv("ENDLESS_TERMINALS_DIR", "")
        use_local_dir = bool(local_tasks_dir)

        if tinker_cfg is not None:
            # ── Tinker mode ─────────────────────────────────────────
            print("[EndlessTerminalsEnv] Using Tinker config", flush=True)

            env_config = EndlessTerminalsEnvConfig(
                # Standard Atropos fields from Tinker YAML
                tokenizer_name=tinker_cfg.base_model,
                group_size=tinker_cfg.group_size,
                use_wandb=tinker_cfg.use_wandb,
                rollout_server_url=tinker_cfg.atropos_api_url,
                total_steps=tinker_cfg.num_steps,
                batch_size=tinker_cfg.batch_size,
                steps_per_eval=tinker_cfg.steps_per_eval,
                max_token_length=tinker_cfg.max_token_env_length,
                max_num_workers=tinker_cfg.max_num_workers,
                max_batches_offpolicy=tinker_cfg.max_batches_offpolicy,
                ensure_scores_are_not_same=tinker_cfg.ensure_scores_are_not_same,
                wandb_name=f"{tinker_cfg.wandb_run_name}-env",
                include_messages=True,

                # Tooling: terminal only
                enabled_toolsets=["terminal"],
                disabled_toolsets=[],

                # Agent config
                agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "32")),
                agent_temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),

                # Docker-direct backend (no Nomad needed)
                tool_pool_mode="docker_direct",
                sandbox_image="ubuntu:22.04",
                purge_job_on_start=False,
                purge_job_on_shutdown=False,

                # Endless Terminals fields
                use_local_dir=use_local_dir,
                local_tasks_dir=local_tasks_dir,
                prebuild_images=os.getenv("PREBUILD_IMAGES", "false").lower() == "true",
                use_dataset=os.getenv("USE_DATASET", "false").lower() == "true",
                dataset_name=os.getenv("ENDLESS_DATASET", "obiwan96/endless-terminals-train"),
                container_build_timeout_s=float(os.getenv("CONTAINER_BUILD_TIMEOUT", "600")),
                test_timeout_s=int(os.getenv("TEST_TIMEOUT", "120")),
            )

            server_configs = [
                APIServerConfig(
                    model_name=tinker_cfg.base_model,
                    base_url=tinker_cfg.inference_api_url + "/v1",
                    api_key="x",
                    server_type="sglang",
                    num_requests_for_eval=tinker_cfg.num_requests_for_eval,
                    timeout=600,  # Longer timeout for multi-step agent trajectories
                ),
            ]
            return env_config, server_configs

        else:
            # ── Standalone mode (env vars) ──────────────────────────
            base_url = (
                os.getenv("ATROPOS_SERVER_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or "http://127.0.0.1:8080"
            )
            model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "hermes-4-36b"
            api_key = (
                os.getenv("ATROPOS_SERVER_API_KEY")
                or os.getenv("NOUS_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or "local"
            )

            env_config = EndlessTerminalsEnvConfig(
                tokenizer_name=os.getenv("ATROPOS_TOKENIZER_NAME") or "NousResearch/Hermes-4.3-36B",
                group_size=int(os.getenv("ATROPOS_GROUP_SIZE", "4")),
                use_wandb=os.getenv("USE_WANDB", "false").lower() == "true",
                include_messages=True,
                total_steps=int(os.getenv("ATROPOS_TOTAL_STEPS", "1000")),
                batch_size=int(os.getenv("ATROPOS_BATCH_SIZE", "32")),
                server_base_url=base_url,
                server_model=model,

                # Tooling
                enabled_toolsets=["terminal"],
                disabled_toolsets=[],

                # Agent
                agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "32")),
                agent_temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),

                # Docker-direct backend
                tool_pool_mode="docker_direct",
                sandbox_image="ubuntu:22.04",
                purge_job_on_start=False,
                purge_job_on_shutdown=False,

                # Endless Terminals fields
                use_local_dir=use_local_dir,
                local_tasks_dir=local_tasks_dir,
                prebuild_images=os.getenv("PREBUILD_IMAGES", "false").lower() == "true",
                use_dataset=os.getenv("USE_DATASET", "false").lower() == "true",
                dataset_name=os.getenv("ENDLESS_DATASET", "obiwan96/endless-terminals-train"),
                task_gen_model=os.getenv("TASK_GEN_MODEL", "Qwen/Qwen3-32B"),
                container_build_timeout_s=float(os.getenv("CONTAINER_BUILD_TIMEOUT", "600")),
                test_timeout_s=int(os.getenv("TEST_TIMEOUT", "120")),
            )

            server_configs = [
                APIServerConfig(
                    model_name=model,
                    base_url=f"{base_url.rstrip('/')}/v1",
                    api_key=api_key,
                    num_max_requests_at_once=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
                    num_requests_for_eval=int(os.getenv("MAX_EVAL_REQUESTS", "4")),
                    timeout=300,
                )
            ]
            return env_config, server_configs

    # ---- Setup ----

    async def setup_agent_env(self) -> None:
        """Env-specific setup: scan tasks and optionally pre-build images."""
        if self.config.use_local_dir:
            await self._setup_local_dir()
        elif self.config.use_dataset:
            await self._setup_hf_dataset()
        else:
            print("[EndlessTerminalsEnv] Using procedural task generation", flush=True)

    async def _setup_local_dir(self) -> None:
        """Scan local directory for task_* folders."""
        tasks_dir = Path(self.config.local_tasks_dir).expanduser().resolve()
        if not tasks_dir.is_dir():
            raise RuntimeError(f"local_tasks_dir does not exist: {tasks_dir}")

        print(f"[EndlessTerminalsEnv] Scanning {tasks_dir} for tasks...", flush=True)

        tasks = []
        for entry in sorted(tasks_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("task_"):
                continue

            # Validate required files
            dockerfile = entry / "environment" / "Dockerfile"
            instruction = entry / "instruction.md"
            test_final = entry / "tests" / "test_final_state.py"

            if not dockerfile.exists():
                continue
            if not instruction.exists():
                continue
            if not test_final.exists():
                continue

            # Read task metadata
            task_json_path = entry / "environment" / "task.json"
            description = instruction.read_text(encoding="utf-8").strip()

            truth = ""
            if task_json_path.exists():
                try:
                    task_json = json.loads(task_json_path.read_text(encoding="utf-8"))
                    # task.json may have a richer description; prefer instruction.md
                    truth = task_json.get("truth", "")
                except Exception:
                    pass

            tasks.append({
                "task_name": entry.name,
                "task_dir": str(entry),
                "dockerfile": str(dockerfile),
                "description": description,
                "truth": truth,
                "test_final": str(test_final),
            })

        if not tasks:
            raise RuntimeError(f"No valid task_* directories found in {tasks_dir}")

        # Split into train and eval (hold out ~5% for eval, min 10, max 50)
        random.shuffle(tasks)
        eval_count = max(10, min(50, len(tasks) // 20))
        eval_count = min(eval_count, len(tasks) // 2)  # Never more than half

        self._eval_tasks = tasks[:eval_count]
        self._local_tasks = tasks[eval_count:]
        self._local_task_indices = list(range(len(self._local_tasks)))
        random.shuffle(self._local_task_indices)
        self._local_current_index = 0

        print(
            f"[EndlessTerminalsEnv] Found {len(tasks)} valid tasks "
            f"({len(self._local_tasks)} train, {len(self._eval_tasks)} eval)",
            flush=True,
        )

        # Optionally pre-build all Docker images
        if self.config.prebuild_images:
            await self._prebuild_images()

    async def _prebuild_images(self) -> None:
        """Pre-build Docker images for all tasks."""
        print(f"[EndlessTerminalsEnv] Pre-building Docker images...", flush=True)
        sem = asyncio.Semaphore(self.config.max_concurrent_builds)
        built = 0
        skipped = 0
        failed = 0

        async def _build_one(task: Dict[str, Any]) -> None:
            nonlocal built, skipped, failed
            image_tag = self._image_tag_for_task(task["task_name"])

            if docker_image_exists(image_tag):
                self._image_cache[task["task_name"]] = image_tag
                skipped += 1
                return

            async with sem:
                ok = await build_docker_image(
                    task["dockerfile"], image_tag,
                    timeout_s=self.config.container_build_timeout_s,
                )
                if ok:
                    self._image_cache[task["task_name"]] = image_tag
                    built += 1
                else:
                    failed += 1

        await asyncio.gather(*[_build_one(t) for t in self._local_tasks])
        print(
            f"[EndlessTerminalsEnv] Pre-build: {built} built, {skipped} cached, {failed} failed",
            flush=True,
        )

    async def _setup_hf_dataset(self) -> None:
        """Load HuggingFace dataset."""
        print(f"[EndlessTerminalsEnv] Loading dataset: {self.config.dataset_name}", flush=True)
        try:
            from datasets import load_dataset

            loop = asyncio.get_event_loop()
            self._dataset = await loop.run_in_executor(
                None,
                lambda: load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    cache_dir=os.path.expanduser(self.config.dataset_cache_dir),
                ),
            )
            self._dataset_indices = list(range(len(self._dataset)))
            random.shuffle(self._dataset_indices)
            self._dataset_current_index = 0
            print(f"[EndlessTerminalsEnv] Loaded {len(self._dataset)} tasks from dataset", flush=True)
        except Exception as e:
            print(f"[EndlessTerminalsEnv] ERROR loading dataset: {e}", flush=True)
            raise

    # ---- Image helpers ----

    def _image_tag_for_task(self, task_name: str) -> str:
        return f"{self.config.docker_image_prefix}:{task_name}"

    async def _ensure_image(self, task: Dict[str, Any]) -> str:
        """Ensure the Docker image for a task is built. Returns image tag."""
        task_name = task["task_name"]
        image_tag = self._image_tag_for_task(task_name)

        # Fast path: already cached
        if task_name in self._image_cache:
            return self._image_cache[task_name]

        async with self._build_lock:
            # Double-check after acquiring lock
            if task_name in self._image_cache:
                return self._image_cache[task_name]

            # Check if image exists in Docker
            if docker_image_exists(image_tag):
                self._image_cache[task_name] = image_tag
                return image_tag

            # Build it
            print(f"[EndlessTerminalsEnv] Building image {image_tag}...", flush=True)
            ok = await build_docker_image(
                task["dockerfile"], image_tag,
                timeout_s=self.config.container_build_timeout_s,
            )
            if not ok:
                raise RuntimeError(f"Failed to build Docker image for {task_name}")

            self._image_cache[task_name] = image_tag
            return image_tag

    # ---- Item generation ----

    async def get_next_item(self) -> Item:
        self._iteration += 1

        if self.config.use_local_dir and self._local_tasks:
            return self._get_next_local_item()
        elif self.config.use_dataset and self._dataset is not None:
            return self._get_next_dataset_item()
        else:
            return self._get_fallback_item()

    def _get_next_local_item(self) -> Item:
        """Pick the next task from local directories."""
        idx = self._local_task_indices[self._local_current_index]
        task = self._local_tasks[idx]

        self._local_current_index += 1
        if self._local_current_index >= len(self._local_task_indices):
            random.shuffle(self._local_task_indices)
            self._local_current_index = 0
            print("[EndlessTerminalsEnv] Reshuffled local tasks (epoch complete)", flush=True)

        return {
            "task_id": f"local_{self._iteration:06d}_{task['task_name']}",
            "task_name": task["task_name"],
            "description": task["description"],
            "truth": task.get("truth", ""),
            "task_dir": task["task_dir"],
            "dockerfile": task["dockerfile"],
            "test_final": task["test_final"],
            "from_local_dir": True,
        }

    def _get_next_dataset_item(self) -> Item:
        """Pick the next task from HuggingFace dataset."""
        idx = self._dataset_indices[self._dataset_current_index]
        task = self._dataset[idx]

        self._dataset_current_index += 1
        if self._dataset_current_index >= len(self._dataset_indices):
            random.shuffle(self._dataset_indices)
            self._dataset_current_index = 0
            print("[EndlessTerminalsEnv] Reshuffled dataset (epoch complete)", flush=True)

        # Resolve task directory
        task_dir = task.get("extra_info", {}).get("task_dir") or task.get("reward_spec", {}).get("ground_truth", "")
        if self.config.tasks_base_dir:
            task_name = Path(task_dir).name
            task_dir = str(Path(self.config.tasks_base_dir) / task_name)

        task_dir_path = Path(task_dir)
        return {
            "task_id": f"dataset_{self._iteration:06d}_{task_dir_path.name}",
            "task_name": task_dir_path.name,
            "description": task.get("description", ""),
            "task_dir": task_dir,
            "dockerfile": str(task_dir_path / "environment" / "Dockerfile"),
            "test_final": str(task_dir_path / "tests" / "test_final_state.py"),
            "from_dataset": True,
        }

    def _get_fallback_item(self) -> Item:
        return {
            "task_id": f"fallback_{self._iteration:06d}",
            "task_name": "fallback",
            "description": (
                "Create a file named 'hello.txt' in /home/user/ containing "
                "the text 'Hello, World!' on a single line."
            ),
            "task_dir": "",
            "dockerfile": "",
            "test_final": "",
        }

    # ---- AgentEnv hooks ----

    def build_task(self, item: Item) -> str:
        """Return the task prompt for the agent."""
        return str(item.get("description", ""))

    def build_agent_config(self, item: Item) -> AgentConfig:
        return AgentConfig(
            max_steps=self.config.agent_max_steps,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.agent_max_tokens,
            tool_delay_s=self.config.agent_tool_delay_s,
        )

    async def setup_trajectory_workspace(
        self,
        item: Item,
        *,
        trajectory_id: str,
        exec_tool,
    ) -> Dict[str, Any]:
        """
        Build the Docker image for this task and register it with the backend.

        The DockerDirectBackend will start a container from this image when the
        agent makes its first tool call (lazy acquisition via ToolExecutor).
        """
        task_name = item.get("task_name", "unknown")
        dockerfile = item.get("dockerfile", "")

        if not dockerfile or not Path(dockerfile).exists():
            print(f"[EndlessTerminalsEnv] WARNING: No Dockerfile for {task_name}", flush=True)
            return {"image": "ubuntu:22.04"}

        # Build/get Docker image
        image_tag = await self._ensure_image({
            "task_name": task_name,
            "dockerfile": dockerfile,
        })

        # Register image with the DockerDirect backend
        if isinstance(self._backend, DockerDirectBackend):
            self._backend.register_image(trajectory_id, image_tag)

        return {"image": image_tag, "task_name": task_name}

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        """Not used — scoring happens in verify_and_score_trajectory."""
        return 0.0

    async def verify_and_score_trajectory(
        self,
        item: Item,
        final_response: str,
        *,
        trajectory_id: str,
        exec_tool,
        agent_result=None,
        workspace_meta=None,
    ) -> tuple[float, Dict[str, Any]]:
        """
        Run test_final_state.py inside the container and return binary reward.
        """
        task_id = item.get("task_id", "unknown")
        test_final = item.get("test_final", "")

        if not test_final or not Path(test_final).exists():
            print(f"[EndlessTerminalsEnv] No test file for {task_id}", flush=True)
            return 0.0, {"error": "No test file"}

        print(f"[EndlessTerminalsEnv] Scoring {task_id}...", flush=True)

        try:
            # Read the test file and base64-encode it for safe transfer
            test_content = Path(test_final).read_text(encoding="utf-8")
            encoded = base64.b64encode(test_content.encode("utf-8")).decode("ascii")

            # Write test file into the container and run pytest
            # We write to /tmp to avoid interfering with the agent's workspace
            # Use printf + heredoc to avoid quoting issues with single quotes in base64
            verify_cmd = (
                f"printf '%s' '{encoded}' | base64 -d > /tmp/_test_final_state.py && "
                f"cd /home/user && "
                f"python3 -m pytest /tmp/_test_final_state.py -v --tb=short 2>&1; "
                f"echo \"EXIT_CODE=$?\""
            )

            result = await exec_tool(ToolCall(
                name="terminal",
                arguments={"command": verify_cmd},
            ))

            output = result.output if hasattr(result, "output") else str(result)

            # Check if pytest passed
            # Look for EXIT_CODE=0 at the end (most reliable)
            success = "EXIT_CODE=0" in output

            score = 1.0 if success else 0.0

            metadata = {
                "task_id": task_id,
                "success": success,
                "test_output": output[-2000:] if len(output) > 2000 else output,
                "total_tool_calls": agent_result.total_tool_calls if agent_result else 0,
            }

            self._train_scores_buffer.append(score)
            print(f"[EndlessTerminalsEnv] {task_id} → score={score}", flush=True)
            return score, metadata

        except Exception as e:
            print(f"[EndlessTerminalsEnv] Error scoring {task_id}: {e}", flush=True)
            return 0.0, {"error": str(e)}

    # ---- WandB logging ----

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log training metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Training pass rate since last log
        if self._train_scores_buffer:
            wandb_metrics["train/percent_correct"] = (
                sum(self._train_scores_buffer) / len(self._train_scores_buffer)
            )
            wandb_metrics["train/num_trajectories"] = len(self._train_scores_buffer)
            self._train_scores_buffer = []

        # Eval metrics (populated by evaluate())
        for key, value in self._eval_metrics:
            wandb_metrics[key] = value
        self._eval_metrics = []

        await super().wandb_log(wandb_metrics)

    # ---- Evaluation ----

    async def evaluate(self, *args, **kwargs):
        """
        Run the agent on held-out eval tasks and report pass rate.

        Each eval task: build Docker container → run agent (temp=0) → pytest → score.
        This is expensive (full agent trajectories), so we only eval a subset.
        """
        import time as _time

        if not self._eval_tasks:
            return {}

        start_time = _time.time()
        eval_sample_size = min(len(self._eval_tasks), 20)
        eval_subset = random.sample(self._eval_tasks, eval_sample_size)

        print(
            f"[EndlessTerminalsEnv] Running evaluation on {eval_sample_size} tasks...",
            flush=True,
        )

        scores = []
        samples = []

        for task_info in eval_subset:
            task_name = task_info["task_name"]
            description = task_info["description"]

            try:
                # Build Docker image
                image_tag = await self._ensure_image(task_info)

                # Run agent with temp=0 for deterministic eval
                eval_tid = f"eval_{uuid.uuid4().hex[:8]}"

                # Register image with backend
                if isinstance(self._backend, DockerDirectBackend):
                    self._backend.register_image(eval_tid, image_tag)

                async def _exec(call, _tid=eval_tid):
                    return await self._tool_executor.execute(_tid, call)

                from ..agent import AtroposAgent as _AtroposAgent

                agent = _AtroposAgent(
                    server=self.server,
                    tokenizer=self.tokenizer,
                    tools=self.tools,
                    config=AgentConfig(
                        max_steps=self.config.agent_max_steps,
                        temperature=0.0,  # Deterministic for eval
                        max_tokens=self.config.agent_max_tokens,
                    ),
                    execute_tool=_exec,
                )

                result = await agent.run(description)

                # Score: run pytest in the container
                score = 0.0
                test_final = task_info.get("test_final", "")
                if result.success and test_final and Path(test_final).exists():
                    test_content = Path(test_final).read_text(encoding="utf-8")
                    encoded = base64.b64encode(test_content.encode("utf-8")).decode("ascii")
                    verify_cmd = (
                        f"printf '%s' '{encoded}' | base64 -d > /tmp/_test_final_state.py && "
                        f"cd /home/user && "
                        f"python3 -m pytest /tmp/_test_final_state.py -v --tb=short 2>&1; "
                        f'echo "EXIT_CODE=$?"'
                    )
                    test_result = await _exec(ToolCall(
                        name="terminal",
                        arguments={"command": verify_cmd},
                    ))
                    test_output = test_result.output if hasattr(test_result, "output") else ""
                    if "EXIT_CODE=0" in test_output:
                        score = 1.0

                scores.append(score)
                samples.append({
                    "task": task_name,
                    "score": score,
                    "tool_calls": result.total_tool_calls,
                    "success": result.success,
                })

                # Cleanup
                await self._tool_executor.release_trajectory(eval_tid, reset_workspace=True)

                print(f"  [eval] {task_name} → {score}", flush=True)

            except Exception as e:
                print(f"  [eval] {task_name} → ERROR: {e}", flush=True)
                scores.append(0.0)
                samples.append({"task": task_name, "score": 0.0, "error": str(e)})

        end_time = _time.time()

        percent_correct = sum(scores) / len(scores) if scores else 0.0

        print(
            f"[EndlessTerminalsEnv] Eval: {percent_correct:.1%} pass rate "
            f"({sum(scores):.0f}/{len(scores)}) in {end_time - start_time:.0f}s",
            flush=True,
        )

        # Store for wandb_log to pick up
        self._eval_metrics.append(("eval/percent_correct", percent_correct))
        self._eval_metrics.append(("eval/num_tasks", len(scores)))
        self._eval_metrics.append(("eval/duration_s", end_time - start_time))

        # Log via atroposlib
        eval_metrics = {
            "eval/percent_correct": percent_correct,
            "eval/num_tasks": len(scores),
        }
        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.agent_max_tokens,
            },
        )


if __name__ == "__main__":
    EndlessTerminalsEnv.cli()




