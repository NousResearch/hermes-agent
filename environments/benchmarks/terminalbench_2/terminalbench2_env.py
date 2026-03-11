"""
TerminalBench2Env -- Terminal-Bench 2.0 Evaluation Environment

Evaluates agentic LLMs on challenging terminal tasks from Terminal-Bench 2.0.
Each task provides a unique Docker environment (pre-built on Docker Hub), a natural
language instruction, and a test suite for verification. The agent uses terminal +
file tools to complete the task, then the test suite runs inside the same sandbox.

This is an eval-only environment (not a training environment). It is designed to
be run via the `evaluate` subcommand:

    python environments/terminalbench2_env.py evaluate \\
        --env.dataset_name NousResearch/terminal-bench-2

The evaluate flow:
    1. setup()     -- Loads the TB2 dataset from HuggingFace
    2. evaluate()  -- Iterates over all tasks, running each through:
        a. rollout_and_score_eval()  -- Per-task agent loop + test verification
            - Resolves Docker image (pre-built Hub image or Dockerfile fallback)
            - Registers per-task Modal sandbox via register_task_env_overrides()
            - Runs the HermesAgentLoop (terminal + file tools)
            - Uploads test suite and runs test.sh in the same sandbox
            - Returns binary pass/fail result
        b. Aggregates per-task, per-category, and overall pass rates
        c. Logs results via evaluate_log() and wandb

Key features:
  - Per-task Modal sandboxes using pre-built Docker Hub images
  - Binary reward: 1.0 if all tests pass, 0.0 otherwise
  - Concurrency-controlled parallel evaluation via asyncio.Semaphore
  - Per-task, per-category, and aggregate pass rate tracking
"""

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import Field

from atroposlib.envs.base import EvalHandlingEnum
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext
from tools.terminal_tool import (
    register_task_env_overrides,
    clear_task_env_overrides,
    cleanup_vm,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class TerminalBench2EvalConfig(HermesAgentEnvConfig):
    """
    Configuration for the Terminal-Bench 2.0 evaluation environment.

    Extends HermesAgentEnvConfig with TB2-specific settings for dataset loading,
    test execution, task filtering, and eval concurrency.
    """

    # --- Dataset ---
    dataset_name: str = Field(
        default="NousResearch/terminal-bench-2",
        description="HuggingFace dataset containing TB2 tasks.",
    )

    # --- Test execution ---
    test_timeout: int = Field(
        default=180,
        description="Timeout in seconds for running the test suite after agent completes.",
    )

    # --- Image strategy ---
    force_build: bool = Field(
        default=False,
        description="If True, always build from Dockerfile (ignore docker_image). "
        "Useful for testing custom Dockerfiles.",
    )

    # --- Task filtering (comma-separated from CLI) ---
    task_filter: Optional[str] = Field(
        default=None,
        description="Comma-separated task names to run (e.g., 'fix-git,git-multibranch'). "
        "If not set, all tasks are run.",
    )
    skip_tasks: Optional[str] = Field(
        default=None,
        description="Comma-separated task names to skip on top of the default skip list.",
    )

    # --- Per-task wall-clock timeout ---
    task_timeout: int = Field(
        default=1800,
        description="Maximum wall-clock seconds per task (agent loop + verification). "
        "Tasks exceeding this are scored as FAIL. Default 30 minutes.",
    )

    # --- Concurrency control ---
    max_concurrent_tasks: int = Field(
        default=8,
        description="Maximum number of tasks to run concurrently. "
        "Limits concurrent Modal sandbox creations to avoid async/threading deadlocks. "
        "Modal has internal limits and creating too many sandboxes simultaneously "
        "causes blocking calls to deadlock inside the thread pool.",
    )

    # --- Eval concurrency ---
    eval_concurrency: int = Field(
        default=0,
        description="Maximum number of tasks to evaluate in parallel. "
        "0 means unlimited (all tasks run concurrently). "
        "Set to 8 for local backends to avoid overwhelming the machine.",
    )


# Tasks that cannot run properly on Modal and are excluded from scoring.
MODAL_INCOMPATIBLE_TASKS = {
    "qemu-startup",        # Needs KVM/hardware virtualization
    "qemu-alpine-ssh",     # Needs KVM/hardware virtualization
    "crack-7z-hash",       # Password brute-force -- too slow for cloud sandbox timeouts
}


# =============================================================================
# Tar extraction helper
# =============================================================================

def _extract_base64_tar(b64_data: str, target_dir: Path):
    """Extract a base64-encoded tar.gz archive into target_dir."""
    if not b64_data:
        return
    raw = base64.b64decode(b64_data)
    buf = io.BytesIO(raw)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=str(target_dir))


# =============================================================================
# Main Environment
# =============================================================================

class TerminalBench2EvalEnv(HermesAgentBaseEnv):
    """
    Terminal-Bench 2.0 evaluation environment (eval-only, no training).

    Inherits from HermesAgentBaseEnv for:
      - Terminal backend setup (os.environ["TERMINAL_ENV"])
      - Tool resolution via _resolve_tools_for_group()
      - Monkey patches for async-safe tool operation
      - Wandb trajectory formatting

    The evaluate flow (triggered by `environment.py evaluate`):
      1. setup()    -- Load dataset from HuggingFace
      2. evaluate() -- Run all tasks through rollout_and_score_eval()

    Each task in rollout_and_score_eval():
      1. Resolve Docker image (pre-built Hub image or Dockerfile fallback)
      2. Register per-task Modal sandbox override
      3. Run HermesAgentLoop with terminal + file tools
      4. Upload test suite and execute test.sh in the same sandbox
      5. Check /logs/verifier/reward.txt for pass/fail
      6. Clean up sandbox, overrides, and temp files
    """

    name = "terminal-bench-2"
    env_config_cls = TerminalBench2EvalConfig

    @classmethod
    def config_init(cls) -> Tuple[TerminalBench2EvalConfig, List[APIServerConfig]]:
        """
        Default configuration for Terminal-Bench 2.0 evaluation.

        Uses eval-only settings:
          - eval_handling=STOP_TRAIN so the eval flow runs cleanly
          - steps_per_eval=1, total_steps=1 so eval triggers immediately
          - group_size=1 (one rollout per group, each task is expensive)

        Uses Modal terminal backend (cloud-isolated sandbox per task) and
        OpenRouter with Claude for inference.
        """
        env_config = TerminalBench2EvalConfig(
            # Terminal + file tools only (the agent interacts via shell commands)
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,

            # Agent settings -- TB2 tasks are complex, need many turns
            max_agent_turns=60,
            max_token_length=***
            agent_temperature=0.6,
            system_prompt=None,

            # Modal backend for per-task cloud-isolated sandboxes
            terminal_backend="modal",
            terminal_timeout=300,   # 5 min per command (builds, pip install, etc.)

            # Test execution timeout (TB2 test scripts can install deps like pytest)
            test_timeout=180,

            # 89 tasks run in parallel, each needs a thread for tool calls
            tool_pool_size=128,

            # --- Eval-only Atropos settings ---
            # These settings make the env work as an eval-only environment:
            #   - STOP_TRAIN: pauses training during eval (standard for eval envs)
            #   - steps_per_eval=1, total_steps=1: eval triggers immediately
            #   - group_size=1: one rollout per group (each task is expensive)
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            group_size=1,
            steps_per_eval=1,
            total_steps=1,

            tokenizer_name="NousRe...1-8B",
            use_wandb=True,
            wandb_name="terminal-bench-2",
            ensure_scores_are_not_same=False,  # Binary rewards may all be 0 or 1
        )

        # OpenRouter with Claude -- API key loaded from .env
        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4",
                server_type="openai",
                api_key=os.get...EY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs

    # =========================================================================
    # Setup -- load dataset
    # =========================================================================

    async def setup(self):
        """Load the Terminal-Bench 2.0 dataset from HuggingFace."""
        from datasets import load_dataset

        # Auto-set terminal_lifetime to task_timeout + 120s so sandboxes
        # never get killed during an active task, but still get cleaned up
        # promptly after the task times out.
        lifetime = self.config.task_timeout + 120
        self.config.terminal_lifetime = lifetime
        os.environ["TERMINAL_LIFETIME_SECONDS"] = str(lifetime)
        print(f"  Terminal lifetime auto-set to {lifetime}s (task_timeout + 120s)")

        print(f"Loading TB2 dataset from: {self.config.dataset_name}")
        ds = load_dataset(self.config.dataset_name, split="train")

        # Apply task filters (comma-separated strings from CLI)
        tasks = list(ds)
        if self.config.task_filter:
            allowed = {name.strip() for name in self.config.task_filter.split(",")}
            tasks = [t for t in tasks if t["task_name"] in allowed]
            print(f"  Filtered to {len(tasks)} tasks: {sorted(allowed)}")

        # Skip tasks incompatible with the current backend (e.g., QEMU on Modal)
        # plus any user-specified skip_tasks
        skip = set(MODAL_INCOMPATIBLE_TASKS) if self.config.terminal_backend == "modal" else set()
        if self.config.skip_tasks:
            skip |= {name.strip() for name in self.config.skip_tasks.split(",")}
        if skip:
            before = len(tasks)
            tasks = [t for t in tasks if t["task_name"] not in skip]
            skipped = before - len(tasks)
            if skipped > 0:
                print(f"  Skipped {skipped} incompatible tasks: {sorted(skip & {t['task_name'] for t in ds})}")

        self.all_eval_items = tasks
        self.iter = 0

        # Build category index for per-category metrics
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        for i, task in enumerate(self.all_eval_items):
            self.category_index[task.get("category", "unknown")].append(i)

        # Reward tracking for wandb logging
        self.eval_metrics: List[Tuple[str, float]] = []

        # Streaming JSONL writer -- saves each task's full conversation
        # immediately on completion so data is preserved even on Ctrl+C.
        # Timestamped filename so each run produces a unique file.
        import datetime
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._streaming_path = os.path.join(log_dir, f"samples_{run_ts}.jsonl")
        self._streaming_file = open(self._streaming_path, "w")
        self._streaming_lock = __import__("threading").Lock()
        print(f"  Streaming results to: {self._streaming_path}")

        print(f"TB2 ready: {len(self.all_eval_items)} tasks across {len(self.category_index)} categories")
        for cat, indices in sorted(self.category_index.items()):
            print(f"  {cat}: {len(indices)} tasks")

    def _save_result(self, result: Dict[str, Any]):
        """Write a single task result to the streaming JSONL file immediately."""
        if not hasattr(self, "_streaming_file") or self._streaming_file.closed:
            return
        with self._streaming_lock:
            self._streaming_file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            self._streaming_file.flush()

    # =========================================================================
    # Training pipeline stubs -- NOT used in eval-only mode
    # =========================================================================
    # These satisfy the abstract method requirements from HermesAgentBaseEnv.
    # The evaluate subcommand calls setup() -> evaluate() directly, bypassing
    # the training pipeline entirely.

    async def get_next_item(self):
        """Return next item (stub -- not used in eval-only mode)."""
        item = self.all_eval_items[self.iter % len(self.all_eval_items)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Return the task's instruction as the user prompt."""
        return item["instruction"]

    async def compute_reward(self, item, result, ctx) -> float:
        """Compute reward (stub -- actual verification is in rollout_and_score_eval)."""
        return 0.0

    async def collect_trajectories(self, item):
        """Collect trajectories (stub -- not used in eval-only mode)."""
        return None, []

    async def score(self, rollout_group_data):
        """Score rollouts (stub -- not used in eval-only mode)."""
        return None

    # =========================================================================
    # Docker image resolution
    # =========================================================================

    def _resolve_task_image(
        self, item: Dict[str, Any], task_name: str
    ) -> Tuple[str, Optional[Path]]:
        """
        Resolve the Docker image for a task, with fallback to Dockerfile.

        Strategy (mirrors Harbor's approach):
        1. If force_build=True, always build from Dockerfile in environment_tar
        2. If docker_image is available, use the pre-built Docker Hub image (fast)
        3. Otherwise, extract Dockerfile from environment_tar and build (slow)

        Returns:
            (modal_image, temp_dir) -- modal_image is a Docker Hub name or a
            Dockerfile path. temp_dir is set if we extracted files that need
            cleanup later.
        """
        docker_image = item.get("docker_image", "")
        environment_tar = item.get("environment_tar", "")

        # Fast path: use pre-built Docker Hub image
        if docker_image and not self.config.force_build:
            logger.info("Task %s: using pre-built image %s", task_name, docker_image)
            return docker_image, None

        # Slow path: extract Dockerfile from environment_tar and build
        if environment_tar:
            task_dir = Path(tempfile.mkdtemp(prefix=f"tb2-{task_name}-"))
            _extract_base64_tar(environment_tar, task_dir)
            dockerfile_path = task_dir / "Dockerfile"
            if dockerfile_path.exists():
                logger.info(
                    "Task %s: building from Dockerfile (force_build=%s, docker_image=%s)",
                    task_name, self.config.force_build, bool(docker_image),
                )
                return str(dockerfile_path), task_dir

        # Neither available -- fall back to Hub image if force_build was True
        if docker_image:
            logger.warning(
                "Task %s: force_build=True but no environment_tar, "
                "falling back to docker_image %s", task_name, docker_image,
            )
            return docker_image, None

        return "", None

    # =========================================================================
    # Per-task evaluation -- agent loop + test verification
    # =========================================================================

    async def rollout_and_score_eval(self, eval_item: Dict[str, Any]) -> Dict:
        """
        Evaluate a single TB2 task: run the agent loop, then verify with tests.

        This is the core evaluation method. For each task it:
        1. Resolves the Docker image and registers the Modal sandbox override
        2. Runs HermesAgentLoop with terminal + file tools
        3. Uploads the test suite into the sandbox
        4. Executes test.sh and checks the result
        5. Cleans up the sandbox and temp files

        Args:
            eval_item: A single TB2 task dict from the dataset

        Returns:
            Dict with 'passed' (bool), 'reward' (float), 'task_name' (str),
            'category' (str), and optional debug info
        """
        task_name = eval_item.get("task_name", "unknown")
        category = eval_item.get("category", "unknown")
        task_id = str(uuid.uuid4())
        task_dir = None  # Set if we extract a Dockerfile (needs cleanup)

        from tqdm import tqdm
        tqdm.write(f"  [START] {task_name} (task_id={task_id[:8]})")
        task_start = time.time()

        try:
            # --- 1. Resolve Docker image ---
            modal_image, task_dir = self._resolve_task_image(eval_item, task_name)
            if not modal_image:
                logger.error("Task %s: no docker_image or environment_tar, skipping", task_name)
                return {
                    "passed": False, "reward": 0.0,
                    "task_name": task_name, "category": category,
                    "error": "no_image",
                }

            # --- 2. Register per-task image override ---
            # Set both modal_image and docker_image so the task image is used
            # regardless of which backend is configured.
            register_task_env_overrides(task_id, {
                "modal_image": modal_image,
                "docker_image": modal_image,
                "cwd": "/app",
            })
            logger.info(
                "Task %s: registered image override for task_id %s",
                task_name, task_id[:8],
            )

            # --- 3. Resolve tools and build messages ---
            tools, valid_names = self._resolve_tools_for_group()

            messages: List[Dict[str, Any]] = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": self.format_prompt(eval_item)})

            # --- 4. Run agent loop ---
            # Use ManagedServer (Phase 2) for vLLM/SGLang backends to get
            # token-level tracking via /generate. Falls back to direct
            # ServerManager (Phase 1) for OpenAI endpoints.
            if self._use_managed_server():
                async with self.server.managed_server(
                    tokenizer=self.tokenizer,
                    preserve_think_blocks=bool(self.config.thinking_mode),
                ) as managed:
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
            else:
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

            # --- 5. Verify -- run test suite in the agent's sandbox ---
            # Skip verification if the agent produced no meaningful output
            only_system_and_user = all(
                msg.get("role") in ("system", "user") for msg in result.messages
            )
            if result.turns_used == 0 or only_system_and_user:
                logger.warning(
                    "Task %s: agent produced no output (turns=%d). Reward=0.",
                    task_name, result.turns_used,
                )
                reward = 0.0
            else:
                # Run tests in a thread so the blocking ctx.terminal() calls
