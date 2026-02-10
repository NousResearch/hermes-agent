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
        description="Comma-separated task names to run (e.g., 'fix-git,broken-pipe'). "
        "If not set, all tasks are run.",
    )
    skip_tasks: Optional[str] = Field(
        default=None,
        description="Comma-separated task names to skip (e.g., 'heavy-task,slow-task').",
    )



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
            max_token_length=16000,
            agent_temperature=0.6,
            system_prompt=(
                "You are a skilled software engineer and system administrator with "
                "access to a terminal and file tools. You are working inside a Linux "
                "container environment. Complete the user's task by using the available "
                "tools. Be methodical: explore the environment first, plan your approach, "
                "then execute step by step. Verify your work before finishing."
            ),

            # Modal backend for per-task cloud-isolated sandboxes
            terminal_backend="modal",

            # Test execution timeout (TB2 test scripts can install deps like pytest)
            test_timeout=180,

            # --- Eval-only Atropos settings ---
            # These settings make the env work as an eval-only environment:
            #   - STOP_TRAIN: pauses training during eval (standard for eval envs)
            #   - steps_per_eval=1, total_steps=1: eval triggers immediately
            #   - group_size=1: one rollout per group (each task is expensive)
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            group_size=1,
            steps_per_eval=1,
            total_steps=1,

            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
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
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
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

        print(f"Loading TB2 dataset from: {self.config.dataset_name}")
        ds = load_dataset(self.config.dataset_name, split="train")

        # Apply task filters (comma-separated strings from CLI)
        tasks = list(ds)
        if self.config.task_filter:
            allowed = {name.strip() for name in self.config.task_filter.split(",")}
            tasks = [t for t in tasks if t["task_name"] in allowed]
            print(f"  Filtered to {len(tasks)} tasks: {sorted(allowed)}")
        if self.config.skip_tasks:
            skip = {name.strip() for name in self.config.skip_tasks.split(",")}
            tasks = [t for t in tasks if t["task_name"] not in skip]
            print(f"  After skip_tasks: {len(tasks)} tasks (skipped: {sorted(skip)})")

        self.all_eval_items = tasks
        self.iter = 0

        # Build category index for per-category metrics
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        for i, task in enumerate(self.all_eval_items):
            self.category_index[task.get("category", "unknown")].append(i)

        # Reward tracking for wandb logging
        self.eval_metrics: List[Tuple[str, float]] = []

        print(f"TB2 ready: {len(self.all_eval_items)} tasks across {len(self.category_index)} categories")
        for cat, indices in sorted(self.category_index.items()):
            print(f"  {cat}: {len(indices)} tasks")

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

            # --- 2. Register per-task Modal image override ---
            register_task_env_overrides(task_id, {"modal_image": modal_image})
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
            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=task_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
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
                ctx = ToolContext(task_id)
                try:
                    reward = self._run_tests(eval_item, ctx, task_name)
                except Exception as e:
                    logger.error("Task %s: test verification failed: %s", task_name, e)
                    reward = 0.0
                finally:
                    ctx.cleanup()

            passed = reward == 1.0
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {task_name} (turns={result.turns_used})")
            logger.info(
                "Task %s: reward=%.1f, turns=%d, finished=%s",
                task_name, reward, result.turns_used, result.finished_naturally,
            )

            return {
                "passed": passed,
                "reward": reward,
                "task_name": task_name,
                "category": category,
                "turns_used": result.turns_used,
                "finished_naturally": result.finished_naturally,
            }

        except Exception as e:
            logger.error("Task %s: rollout failed: %s", task_name, e, exc_info=True)
            print(f"  [ERROR] {task_name}: {e}")
            return {
                "passed": False, "reward": 0.0,
                "task_name": task_name, "category": category,
                "error": str(e),
            }

        finally:
            # --- Cleanup: clear overrides, sandbox, and temp files ---
            clear_task_env_overrides(task_id)
            try:
                cleanup_vm(task_id)
            except Exception as e:
                logger.debug("VM cleanup for %s: %s", task_id[:8], e)
            if task_dir and task_dir.exists():
                shutil.rmtree(task_dir, ignore_errors=True)

    def _run_tests(
        self, item: Dict[str, Any], ctx: ToolContext, task_name: str
    ) -> float:
        """
        Upload and execute the test suite in the agent's sandbox, then
        download the verifier output locally to read the reward.

        Follows Harbor's verification pattern:
        1. Upload tests/ directory into the sandbox
        2. Execute test.sh inside the sandbox
        3. Download /logs/verifier/ directory to a local temp dir
        4. Read reward.txt locally with native Python I/O

        Downloading locally avoids issues with the file_read tool on
        the Modal VM and matches how Harbor handles verification.

        TB2 test scripts (test.sh) typically:
        1. Install pytest via uv/pip
        2. Run pytest against the test files in /tests/
        3. Write results to /logs/verifier/reward.txt

        Args:
            item: The TB2 task dict (contains tests_tar, test_sh)
            ctx: ToolContext scoped to this task's sandbox
            task_name: For logging

        Returns:
            1.0 if tests pass, 0.0 otherwise
        """
        tests_tar = item.get("tests_tar", "")
        test_sh = item.get("test_sh", "")

        if not test_sh:
            logger.warning("Task %s: no test_sh content, reward=0", task_name)
            return 0.0

        # Create required directories in the sandbox
        ctx.terminal("mkdir -p /tests /logs/verifier")

        # Upload test files into the sandbox (binary-safe via base64)
        if tests_tar:
            tests_temp = Path(tempfile.mkdtemp(prefix=f"tb2-tests-{task_name}-"))
            try:
                _extract_base64_tar(tests_tar, tests_temp)
                ctx.upload_dir(str(tests_temp), "/tests")
            except Exception as e:
                logger.warning("Task %s: failed to upload test files: %s", task_name, e)
            finally:
                shutil.rmtree(tests_temp, ignore_errors=True)

        # Write the test runner script (test.sh)
        ctx.write_file("/tests/test.sh", test_sh)
        ctx.terminal("chmod +x /tests/test.sh")

        # Execute the test suite
        logger.info(
            "Task %s: running test suite (timeout=%ds)",
            task_name, self.config.test_timeout,
        )
        test_result = ctx.terminal(
            "bash /tests/test.sh",
            timeout=self.config.test_timeout,
        )

        exit_code = test_result.get("exit_code", -1)
        output = test_result.get("output", "")

        # Download the verifier output directory locally, then read reward.txt
        # with native Python I/O. This avoids issues with file_read on the
        # Modal VM and matches Harbor's verification pattern.
        reward = 0.0
        local_verifier_dir = Path(tempfile.mkdtemp(prefix=f"tb2-verifier-{task_name}-"))
        try:
            ctx.download_dir("/logs/verifier", str(local_verifier_dir))

            reward_file = local_verifier_dir / "reward.txt"
            if reward_file.exists() and reward_file.stat().st_size > 0:
                content = reward_file.read_text().strip()
                if content == "1":
                    reward = 1.0
                elif content == "0":
                    reward = 0.0
                else:
                    # Unexpected content -- try parsing as float
                    try:
                        reward = float(content)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Task %s: reward.txt content unexpected (%r), "
                            "falling back to exit_code=%d",
                            task_name, content, exit_code,
                        )
                        reward = 1.0 if exit_code == 0 else 0.0
            else:
                # reward.txt not written -- fall back to exit code
                logger.warning(
                    "Task %s: reward.txt not found after download, "
                    "falling back to exit_code=%d",
                    task_name, exit_code,
                )
                reward = 1.0 if exit_code == 0 else 0.0
        except Exception as e:
            logger.warning(
                "Task %s: failed to download verifier dir: %s, "
                "falling back to exit_code=%d",
                task_name, e, exit_code,
            )
            reward = 1.0 if exit_code == 0 else 0.0
        finally:
            shutil.rmtree(local_verifier_dir, ignore_errors=True)

        # Log test output for debugging failures
        if reward == 0.0:
            output_preview = output[-500:] if output else "(no output)"
            logger.info(
                "Task %s: FAIL (exit_code=%d)\n%s",
                task_name, exit_code, output_preview,
            )

        return reward

    # =========================================================================
    # Evaluate -- main entry point for the eval subcommand
    # =========================================================================

    async def evaluate(self, *args, **kwargs) -> None:
        """
        Run Terminal-Bench 2.0 evaluation over all tasks.

        This is the main entry point when invoked via:
            python environments/terminalbench2_env.py evaluate

        Runs all tasks through rollout_and_score_eval() via asyncio.gather()
        (same pattern as GPQA and other Atropos eval envs). Aggregates
        per-task, per-category, and overall pass rates, then logs to wandb
        and evaluate_log().
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting Terminal-Bench 2.0 Evaluation")
        print(f"{'='*60}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Total tasks: {len(self.all_eval_items)}")
        print(f"  Max agent turns: {self.config.max_agent_turns}")
        print(f"  Terminal backend: {self.config.terminal_backend}")
        print(f"{'='*60}\n")

        # Fire all tasks -- Atropos / Modal handle scheduling
        from tqdm.asyncio import tqdm_asyncio
        eval_tasks = [
            self.rollout_and_score_eval(item) for item in self.all_eval_items
        ]
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating TB2")

        end_time = time.time()

        # Filter out None results (shouldn't happen, but be safe)
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return

        # ---- Compute metrics ----
        total = len(valid_results)
        passed = sum(1 for r in valid_results if r.get("passed"))
        overall_pass_rate = passed / total if total > 0 else 0.0

        # Per-category breakdown
        cat_results: Dict[str, List[Dict]] = defaultdict(list)
        for r in valid_results:
            cat_results[r.get("category", "unknown")].append(r)

        # Build metrics dict
        eval_metrics = {
            "eval/pass_rate": overall_pass_rate,
            "eval/total_tasks": total,
            "eval/passed_tasks": passed,
            "eval/evaluation_time_seconds": end_time - start_time,
        }

        # Per-category metrics
        for category, cat_items in sorted(cat_results.items()):
            cat_passed = sum(1 for r in cat_items if r.get("passed"))
            cat_total = len(cat_items)
            cat_pass_rate = cat_passed / cat_total if cat_total > 0 else 0.0
            cat_key = category.replace(" ", "_").replace("-", "_").lower()
            eval_metrics[f"eval/pass_rate_{cat_key}"] = cat_pass_rate

        # Store metrics for wandb_log
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # ---- Print summary ----
        print(f"\n{'='*60}")
        print("Terminal-Bench 2.0 Evaluation Results")
        print(f"{'='*60}")
        print(f"Overall Pass Rate: {overall_pass_rate:.4f} ({passed}/{total})")
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")

        print("\nCategory Breakdown:")
        for category, cat_items in sorted(cat_results.items()):
            cat_passed = sum(1 for r in cat_items if r.get("passed"))
            cat_total = len(cat_items)
            cat_rate = cat_passed / cat_total if cat_total > 0 else 0.0
            print(f"  {category}: {cat_rate:.1%} ({cat_passed}/{cat_total})")

        # Print individual task results
        print("\nTask Results:")
        for r in sorted(valid_results, key=lambda x: x.get("task_name", "")):
            status = "PASS" if r.get("passed") else "FAIL"
            turns = r.get("turns_used", "?")
            error = r.get("error", "")
            extra = f" (error: {error})" if error else ""
            print(f"  [{status}] {r['task_name']} (turns={turns}){extra}")

        print(f"{'='*60}\n")

        # Build sample records for evaluate_log
        samples = [
            {
                "task_name": r.get("task_name"),
                "category": r.get("category"),
                "passed": r.get("passed"),
                "reward": r.get("reward"),
                "turns_used": r.get("turns_used"),
                "error": r.get("error"),
            }
            for r in valid_results
        ]

        # Log evaluation results
        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.agent_temperature,
                    "max_tokens": self.config.max_token_length,
                    "max_agent_turns": self.config.max_agent_turns,
                    "terminal_backend": self.config.terminal_backend,
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

    # =========================================================================
    # Wandb logging
    # =========================================================================

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log TB2-specific metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add stored eval metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    TerminalBench2EvalEnv.cli()
