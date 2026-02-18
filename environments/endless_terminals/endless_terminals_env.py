"""
Endless Terminals Environment for Hermes-Agent + Atropos RL.

Loads pre-generated terminal tasks from HuggingFace dataset and scores
agent performance using test execution in Apptainer containers.

Dataset: https://huggingface.co/datasets/obiwan96/endless-terminals-train

Run:
  python environments/endless_terminals/endless_terminals_env.py process \
    --config environments/endless_terminals/default.yaml
"""

import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

# Ensure hermes-agent root is on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.base import ScoredDataItem
from atroposlib.type_definitions import Item

from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.agent_loop import AgentResult
from environments.tool_context import ToolContext

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

    # Agent defaults
    max_agent_turns: int = Field(default=32, description="Max turns for agent (increased for long traces)")


class EndlessTerminalsEnv(HermesAgentBaseEnv[EndlessTerminalsEnvConfig]):
    """
    Endless Terminals environment using pre-generated HuggingFace dataset.

    Loads terminal tasks from dataset, runs agent with terminal tools,
    and scores by executing tests in Apptainer containers.
    """

    name = "endless_terminals_env"
    env_config_cls = EndlessTerminalsEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset = None
        self._dataset_indices = []
        self._current_index = 0

    async def setup(self):
        """Load HuggingFace dataset."""
        if not self.config.use_dataset:
            print("[EndlessTerminalsEnv] Using procedural task generation (not implemented yet)", flush=True)
            return

        print(f"[EndlessTerminalsEnv] Loading dataset: {self.config.dataset_name}", flush=True)

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

            # Create shuffled indices
            self._dataset_indices = list(range(len(self._dataset)))
            random.shuffle(self._dataset_indices)
            self._current_index = 0

            print(f"[EndlessTerminalsEnv] Loaded {len(self._dataset)} tasks from dataset", flush=True)

        except Exception as e:
            print(f"[EndlessTerminalsEnv] ERROR loading dataset: {e}", flush=True)
            raise

    async def get_next_item(self) -> Item:
        """Sample next task from dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        # Get next task (with wraparound)
        idx = self._dataset_indices[self._current_index]
        task = self._dataset[idx]

        # Advance to next task
        self._current_index += 1
        if self._current_index >= len(self._dataset_indices):
            # Reshuffle for next epoch
            random.shuffle(self._dataset_indices)
            self._current_index = 0
            print("[EndlessTerminalsEnv] Reshuffled dataset (completed one epoch)", flush=True)

        # Extract task directory path
        task_dir = task.get("extra_info", {}).get("task_dir")
        if not task_dir:
            task_dir = task.get("reward_spec", {}).get("ground_truth")

        # If tasks_base_dir is configured, reconstruct path
        if self.config.tasks_base_dir:
            original_path = Path(task_dir)
            task_name = original_path.name
            task_dir_path = Path(self.config.tasks_base_dir) / task_name
        else:
            task_dir_path = Path(task_dir)

        # Verify directory exists
        if not task_dir_path.exists():
            print(f"[EndlessTerminalsEnv] WARNING: Task dir not found: {task_dir_path}", flush=True)
            print(f"[EndlessTerminalsEnv] Hint: Set tasks_base_dir to directory containing task_* folders", flush=True)
            return await self.get_next_item()  # Try next task

        container_sif = task_dir_path / "container.sif"
        final_test = task_dir_path / "test_final_state.py"

        # Verify files exist
        if not container_sif.exists() or not final_test.exists():
            print(f"[EndlessTerminalsEnv] WARNING: Missing files in {task_dir_path}", flush=True)
            return await self.get_next_item()

        return {
            "task_id": f"{task_dir_path.name}",
            "description": task.get("description", ""),
            "task_dir": str(task_dir_path),
            "container_sif": str(container_sif),
            "final_test": str(final_test),
            "dataset_index": idx,
        }

    def format_prompt(self, item: Item) -> str:
        """Return the task description for the agent."""
        return str(item.get("description", ""))

    async def compute_reward(
        self,
        item: Item,
        result: AgentResult,
        ctx: ToolContext
    ) -> float:
        """
        Run final tests in container and return binary reward.

        Returns 1.0 if tests pass, 0.0 otherwise.
        """
        task_id = item.get("task_id", "unknown")
        container_sif = Path(item.get("container_sif", ""))
        final_test = Path(item.get("final_test", ""))

        if not container_sif.exists() or not final_test.exists():
            print(f"[EndlessTerminalsEnv] ERROR: Missing test files for {task_id}", flush=True)
            return 0.0

        print(f"[EndlessTerminalsEnv] Running tests for {task_id}...", flush=True)

        try:
            # Run final tests in container
            success = await self._run_tests_in_container(container_sif, final_test)
            score = 1.0 if success else 0.0

            print(f"[EndlessTerminalsEnv] Task {task_id} score: {score}", flush=True)
            return score

        except Exception as e:
            print(f"[EndlessTerminalsEnv] ERROR scoring {task_id}: {e}", flush=True)
            return 0.0

    async def _run_tests_in_container(
        self,
        container_sif: Path,
        final_test_path: Path
    ) -> bool:
        """Run pytest in Apptainer container."""
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

    async def evaluate(self):
        """Periodic evaluation (optional)."""
        return {}


if __name__ == "__main__":
    EndlessTerminalsEnv.cli()
