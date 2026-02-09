"""
Endless Terminals Environment for Hermes-Agent + Atropos RL.

Procedurally generates terminal tasks on-demand during RL training.
Uses the Endless Terminals task generation pipeline to create diverse Linux tasks,
builds Apptainer containers, and scores trajectories based on test execution.

Run (process mode):
  uv run python -m atropos.envs.endless_terminals_env process \
    --env.use_wandb false \
    --env.total_steps 100 \
    --env.group_size 4
"""

from __future__ import annotations

import os
import sys
import asyncio
import tempfile
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import APIServerConfig, Item

from ..agent import AgentConfig
from .agent_env import AgentEnv, AgentEnvConfig

load_dotenv()

# Add endless-terminals to path for imports
ENDLESS_TERMINALS_PATH = os.getenv(
    "ENDLESS_TERMINALS_PATH",
    str(Path.home() / "Desktop" / "Projects" / "endless-terminals")
)
sys.path.insert(0, ENDLESS_TERMINALS_PATH)


class EndlessTerminalsEnvConfig(AgentEnvConfig):
    """Configuration for Endless Terminals environment."""

    # Task generation
    task_gen_model: str = Field(
        default="Qwen/Qwen3-32B",
        description="Model for task generation (via vLLM)"
    )
    task_gen_temperature: float = Field(default=1.0, description="Temperature for task generation")
    task_gen_max_tokens: int = Field(default=2048, description="Max tokens for task generation")

    # Container settings
    base_container_image: str = Field(
        default="ubuntu:22.04",
        description="Base Apptainer image for task containers"
    )
    container_timeout_s: int = Field(default=180, description="Container build timeout (seconds)")

    # Test execution
    test_timeout_s: int = Field(default=60, description="Test execution timeout (seconds)")

    # Workspace
    workspace_dir: str = Field(
        default="/tmp/endless_terminals_workspace",
        description="Directory for task workspaces (cleaned up after scoring)"
    )
    keep_failed_tasks: bool = Field(
        default=False,
        description="Keep task directories for failed trajectories (for debugging)"
    )

    # Agent defaults
    agent_max_steps: int = Field(default=32, description="Max steps for agent (increased for long traces)")
    agent_temperature: float = Field(default=0.7, description="Agent sampling temperature")

    # Server defaults
    server_base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL for OpenAI-compatible chat server"
    )
    server_model: str = Field(default="hermes-4-36b", description="Model name")
    tokenizer_name: str = Field(
        default="NousResearch/Hermes-4.3-36B",
        description="Tokenizer for RL"
    )


class EndlessTerminalsEnv(AgentEnv[EndlessTerminalsEnvConfig]):
    """
    Endless Terminals environment with procedural task generation.

    This environment generates terminal tasks on-demand during RL training:
    1. get_next_item(): Generate task via LLM, build container/tests
    2. Agent solves task using terminal tool
    3. verify_and_score_trajectory(): Run tests, return binary reward
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
        self._workspace_dir = Path(config.workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def config_init(cls) -> Tuple[EndlessTerminalsEnvConfig, List[APIServerConfig]]:
        """Initialize config from environment variables."""
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

            # Tooling: terminal only for Endless Terminals
            enabled_toolsets=["terminal"],
            disabled_toolsets=[],

            # Agent config
            agent_max_steps=32,  # Increased for long traces
            agent_temperature=0.7,

            # Task generation
            task_gen_model=os.getenv("TASK_GEN_MODEL", "Qwen/Qwen3-32B"),

            # Sandbox config
            tool_pool_mode=os.getenv("TOOL_POOL_MODE", "nomad"),
            sandbox_image=os.getenv("ATROPOS_SANDBOX_IMAGE", "atropos-sandbox:local"),
            purge_job_on_start=True,
            purge_job_on_shutdown=True,
        )

        server_configs = [
            APIServerConfig(
                model_name=model,
                base_url=f"{base_url.rstrip('/')}/v1",
                api_key=api_key,
                num_max_requests_at_once=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
                num_requests_for_eval=int(os.getenv("MAX_EVAL_REQUESTS", "4")),
                timeout=300,  # Longer timeout for multi-step tasks
            )
        ]
        return env_config, server_configs

    async def setup_agent_env(self) -> None:
        """Environment-specific setup (no-op for now)."""
        print("[EndlessTerminalsEnv] setup_agent_env(): ready", flush=True)

    async def get_next_item(self) -> Item:
        """
        Generate a new task on-demand.

        This is the core of procedural generation:
        1. Call Endless Terminals task generation LLM
        2. Build Apptainer container with tests
        3. Return task item with workspace info
        """
        self._iteration += 1
        task_id = f"task_{self._iteration:06d}_{uuid.uuid4().hex[:8]}"

        print(f"[EndlessTerminalsEnv] Generating task {task_id}...", flush=True)

        try:
            # Generate task description via LLM
            task_data = await self._generate_task()

            # Create workspace directory
            task_dir = self._workspace_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            # Generate test files
            initial_test_path = task_dir / "test_initial_state.py"
            final_test_path = task_dir / "test_final_state.py"

            await self._generate_test_files(
                task_data,
                initial_test_path,
                final_test_path
            )

            # Generate and build container
            container_def_path = task_dir / "container.def"
            container_sif_path = task_dir / "container.sif"

            await self._build_container(
                task_data,
                initial_test_path,
                container_def_path,
                container_sif_path
            )

            print(f"[EndlessTerminalsEnv] Task {task_id} ready!", flush=True)

            return {
                "task_id": task_id,
                "description": task_data["description"],
                "truth": task_data.get("truth", ""),
                "task_dir": str(task_dir),
                "container_sif": str(container_sif_path),
                "initial_test": str(initial_test_path),
                "final_test": str(final_test_path),
            }

        except Exception as e:
            print(f"[EndlessTerminalsEnv] Failed to generate task {task_id}: {e}", flush=True)
            # Return a simple fallback task
            return self._get_fallback_task(task_id)

    def build_task(self, item: Item) -> str:
        """Return the task description for the agent."""
        return str(item.get("description", ""))

    def build_agent_config(self, item: Item) -> AgentConfig:
        """Build agent config for this task."""
        return AgentConfig(
            max_steps=self.config.agent_max_steps,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.agent_max_tokens,
            tool_delay_s=self.config.agent_tool_delay_s,
        )

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        """
        Score is computed in verify_and_score_trajectory.
        This method is required by base class but not used.
        """
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
        Run final tests in container and return binary reward.

        Returns:
            (score, metadata) where score is 1.0 if tests pass, 0.0 otherwise
        """
        task_id = item.get("task_id", "unknown")
        container_sif = item.get("container_sif")
        final_test = item.get("final_test")
        task_dir = item.get("task_dir")

        if not container_sif or not final_test:
            return 0.0, {"error": "Missing container or test files"}

        print(f"[EndlessTerminalsEnv] Scoring task {task_id}...", flush=True)

        try:
            # Run final tests in container
            success = await self._run_final_tests(
                Path(container_sif),
                Path(final_test)
            )

            score = 1.0 if success else 0.0

            # Cleanup workspace (unless keeping failed tasks)
            if score == 0.0 and self.config.keep_failed_tasks:
                print(f"[EndlessTerminalsEnv] Keeping failed task at {task_dir}", flush=True)
            else:
                await self._cleanup_task(Path(task_dir))

            metadata = {
                "task_id": task_id,
                "success": success,
                "total_tool_calls": agent_result.total_tool_calls if agent_result else 0,
            }

            print(f"[EndlessTerminalsEnv] Task {task_id} score: {score}", flush=True)
            return score, metadata

        except Exception as e:
            print(f"[EndlessTerminalsEnv] Error scoring task {task_id}: {e}", flush=True)
            return 0.0, {"error": str(e)}

    # -------------------------------------------------------------------------
    # Helper methods for task generation and testing
    # -------------------------------------------------------------------------

    async def _generate_task(self) -> Dict[str, str]:
        """Generate a task using Endless Terminals LLM pipeline."""
        from generator.task_template_gen import generate_templates_batch

        # Generate single task using batch API
        loop = asyncio.get_event_loop()
        tasks = await loop.run_in_executor(
            None,
            lambda: generate_templates_batch(
                batch_size=1,
                model=self.config.task_gen_model,
                temperature=self.config.task_gen_temperature,
                max_tokens=self.config.task_gen_max_tokens,
                max_concurrency=1,
            )
        )

        if not tasks:
            raise RuntimeError("Task generation failed")

        return tasks[0]

    async def _generate_test_files(
        self,
        task_data: Dict[str, str],
        initial_test_path: Path,
        final_test_path: Path
    ) -> None:
        """Generate test files for the task."""
        from generator.initial_state_test_gen import generate_test_template as gen_initial
        from generator.completion_test_gen import generate_test_template as gen_final

        description = task_data["description"]
        truth = task_data.get("truth", "")

        loop = asyncio.get_event_loop()

        # Generate initial state test
        initial_test = await loop.run_in_executor(
            None,
            lambda: gen_initial(
                description,
                truth,
                temperature=0.6,
                model=self.config.task_gen_model,
            )
        )
        initial_test_path.write_text(initial_test, encoding="utf-8")

        # Generate final state test
        final_test = await loop.run_in_executor(
            None,
            lambda: gen_final(
                description,
                truth,
                temperature=0.6,
                model=self.config.task_gen_model,
            )
        )
        final_test_path.write_text(final_test, encoding="utf-8")

    async def _build_container(
        self,
        task_data: Dict[str, str],
        initial_test_path: Path,
        container_def_path: Path,
        container_sif_path: Path
    ) -> None:
        """Build Apptainer container for the task."""
        from generator.apptainer_def_gen import iterate_def_template

        description = task_data["description"]
        truth = task_data.get("truth", "")
        initial_test = initial_test_path.read_text(encoding="utf-8")

        loop = asyncio.get_event_loop()

        # Generate container definition (with retries)
        def_text = await loop.run_in_executor(
            None,
            lambda: iterate_def_template(
                description,
                truth,
                initial_test,
                max_rounds=3,  # Reduced retries for speed
                num_completions=4,
                model=self.config.task_gen_model,
                temperature=0.6,
                max_tokens=1024,
            )
        )

        if not def_text:
            raise RuntimeError("Container definition generation failed")

        container_def_path.write_text(def_text, encoding="utf-8")

        # Build SIF file
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["apptainer", "build", str(container_sif_path), str(container_def_path)],
                capture_output=True,
                text=True,
                timeout=self.config.container_timeout_s,
            )
        )

        if result.returncode != 0:
            raise RuntimeError(f"Container build failed: {result.stderr}")

    async def _run_final_tests(
        self,
        container_sif: Path,
        final_test_path: Path
    ) -> bool:
        """Run final tests in the container."""
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "apptainer", "exec",
                        "--fakeroot",
                        "--userns",
                        "--writable-tmpfs",
                        "--cleanenv",
                        str(container_sif),
                        "pytest", "-q",
                        str(final_test_path.name),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.config.test_timeout_s,
                    cwd=str(final_test_path.parent),
                )
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[EndlessTerminalsEnv] Test timeout for {final_test_path}", flush=True)
            return False
        except Exception as e:
            print(f"[EndlessTerminalsEnv] Test execution error: {e}", flush=True)
            return False

    async def _cleanup_task(self, task_dir: Path) -> None:
        """Clean up task workspace."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: shutil.rmtree(task_dir, ignore_errors=True)
        )

    def _get_fallback_task(self, task_id: str) -> Item:
        """Return a simple fallback task if generation fails."""
        return {
            "task_id": task_id,
            "description": (
                "Create a file named 'hello.txt' in /home/user/ containing "
                "the text 'Hello, World!' on a single line."
            ),
            "truth": "File: /home/user/hello.txt, Content: Hello, World!",
            "task_dir": "",
            "container_sif": "",
            "initial_test": "",
            "final_test": "",
        }


if __name__ == "__main__":
    EndlessTerminalsEnv.cli()
