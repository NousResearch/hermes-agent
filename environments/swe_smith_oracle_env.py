"""
SWE-smith-oracle environment (ported to HermesAgentBaseEnv).

Trains models to fix real GitHub repositories:
- Clones a public GitHub repo at a specific commit
- Runs an agent loop with terminal tool to apply a fix
- Verifies by running pytest with nodeids from the dataset
- Reward: 1.0 if all tests pass, 0.0 otherwise

Dataset: NousResearch/SWE-smith-oracle (train split; does NOT use SWE-bench eval set).

Usage:
    # Process mode (OpenAI server, no training):
    python environments/swe_smith_oracle_env.py process \\
        --env.data_path_to_save_groups data/swe_oracle_output.jsonl

    # With Modal sandbox backend:
    python environments/swe_smith_oracle_env.py process \\
        --env.tool_pool_mode modal \\
        --env.modal_image python:3.11
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import Field

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item

from environments.agent_loop import AgentResult
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

class SweSmithOracleEnvConfig(HermesAgentEnvConfig):
    """Config for SWE-smith-oracle environment."""

    dataset_name: str = Field(default="NousResearch/SWE-smith-oracle")
    dataset_split: str = Field(default="train")
    max_items: int = Field(default=0, description="0 = no limit")
    shuffle: bool = Field(default=True)
    seed: int = Field(default=0)

    python_only: bool = Field(default=True, description="Filter to Python-evaluable rows")
    score_include_fail_to_pass: bool = Field(
        default=True,
        description="Score tests on PASS_TO_PASS ∪ FAIL_TO_PASS. "
        "Disable to only run PASS_TO_PASS (faster but weaker signal).",
    )

    prompt_mode: str = Field(
        default="problem_statement",
        description="'problem_statement' (fast) or 'problem_statement+text' (includes dataset 'text').",
    )

    repo_base_url: str = Field(default="https://github.com", description="Base URL for repo cloning")
    install_timeout_s: float = Field(default=600.0)
    test_timeout_s: float = Field(default=600.0)


# =============================================================================
# Environment
# =============================================================================

class SweSmithOracleEnv(HermesAgentBaseEnv):
    """
    SWE-smith-oracle environment for training models to fix real GitHub repos.

    Uses proper OpenAI-spec tool calling via HermesAgentBaseEnv.
    The model gets terminal access to inspect, edit, and test the repository.
    """

    name = "swe-smith-oracle"
    env_config_cls = SweSmithOracleEnvConfig

    def __init__(
        self,
        config: SweSmithOracleEnvConfig,
        server_configs,
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self._dataset = None
        self._indices: List[int] = []
        self._cursor = 0

    @classmethod
    def config_init(cls) -> Tuple[SweSmithOracleEnvConfig, List[APIServerConfig]]:
        """Default config — reads from ATROPOS_SERVER_* env vars."""
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "Hermes-4.3-36B"
        api_key = (
            os.getenv("ATROPOS_SERVER_API_KEY")
            or os.getenv("NOUS_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "local"
        )

        env_config = SweSmithOracleEnvConfig(
            tokenizer_name=os.getenv("ATROPOS_TOKENIZER_NAME") or "NousResearch/Hermes-4.3-36B",
            group_size=1,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=8192,
            wandb_name="swe_smith_oracle",
            # Terminal tool for the agent
            enabled_toolsets=["terminal"],
            terminal_backend=os.getenv("TERMINAL_ENV", "local"),
            # Longer agent turns for SWE tasks
            max_agent_turns=50,
            agent_temperature=0.7,
            system_prompt=(
                "You are a senior software engineer. You have access to a terminal "
                "to inspect and fix repositories. Use non-interactive commands only. "
                "Each terminal command runs in a fresh shell."
            ),
            tool_call_parser="hermes",
            # Sandbox settings (used when tool_pool_mode != "default")
            sandbox_image=os.getenv("ATROPOS_SANDBOX_IMAGE") or "atropos-sandbox:local",
            purge_job_on_start=True,
            purge_job_on_shutdown=True,
        )

        server_configs = [
            APIServerConfig(
                model_name=model,
                base_url=base_url,
                api_key=api_key,
                server_type="openai",
                health_check=False,
                timeout=int(os.getenv("ATROPOS_SERVER_TIMEOUT_S") or "300"),
            ),
        ]

        return env_config, server_configs

    # =========================================================================
    # Dataset loading
    # =========================================================================

    async def setup(self):
        """Load SWE-smith-oracle dataset."""
        from datasets import load_dataset

        t0 = time.perf_counter()
        print(
            f"[SweSmithOracleEnv] loading dataset {self.config.dataset_name}:{self.config.dataset_split} "
            f"(python_only={self.config.python_only}, max_items={self.config.max_items or 'all'})",
            flush=True,
        )
        ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        self._dataset = ds

        indices: List[int] = []
        for idx in range(len(ds)):
            row = ds[idx]
            if self.config.python_only and not self._is_python_row(row):
                continue
            indices.append(idx)

        if self.config.shuffle:
            rnd = random.Random(self.config.seed)
            rnd.shuffle(indices)

        if self.config.max_items and self.config.max_items > 0:
            indices = indices[: self.config.max_items]

        self._indices = indices
        self._cursor = 0

        print(
            f"[SweSmithOracleEnv] loaded {len(self._indices)} items "
            f"in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def _is_python_row(self, row: Dict[str, Any]) -> bool:
        nodeids = row.get("PASS_TO_PASS")
        if not isinstance(nodeids, list) or not nodeids:
            return False
        return all(isinstance(nid, str) and ".py::" in nid for nid in nodeids)

    async def get_next_item(self) -> Item:
        if not self._dataset or not self._indices:
            raise RuntimeError("Dataset not initialized")
        if self._cursor >= len(self._indices):
            self._cursor = 0
        idx = self._indices[self._cursor]
        self._cursor += 1
        return dict(self._dataset[idx])

    # =========================================================================
    # Prompt formatting
    # =========================================================================

    def _repo_name(self, item: Item) -> str:
        repo = item.get("repo") or ""
        if isinstance(repo, str) and "/" in repo:
            return repo.split("/")[-1]
        return "repo"

    def format_prompt(self, item: Item) -> str:
        """Build the SWE task prompt."""
        repo = item.get("repo") or ""
        base_commit = item.get("base_commit") or ""
        problem = str(item.get("problem_statement") or "")
        context = str(item.get("text") or "")
        repo_dir = self._repo_name(item)

        nodeids = self._tests_for_item(item)
        tests_list = "\n".join(f"- {t}" for t in nodeids)

        context_block = ""
        prompt_mode = (self.config.prompt_mode or "problem_statement").strip().lower()
        if prompt_mode == "problem_statement+text" and context:
            context_block = f"\nAdditional context:\n{context}\n"

        return (
            f"Fix the repository so the specified tests pass.\n\n"
            f"Repository: {repo} (checked out at base_commit={base_commit})\n"
            f"Workspace path: ./{repo_dir}\n\n"
            "Constraints:\n"
            "- Use the terminal tool to inspect, edit, and verify the repository.\n"
            f"- Start by inspecting the repo (e.g. `ls`, `cd ./{repo_dir}`, `git status`).\n"
            "- Use a workspace-local virtualenv (.venv) to avoid cross-run contamination.\n"
            "- Use non-interactive commands only.\n"
            "- Prefer `. .venv/bin/activate` or `.venv/bin/python ...` (POSIX compatible).\n\n"
            f"Problem statement:\n{problem}\n\n"
            f"{context_block}"
            f"Run these tests to verify:\n{tests_list}\n\n"
            "When done, briefly describe what you changed and confirm tests pass."
        )

    # =========================================================================
    # Test helpers
    # =========================================================================

    def _tests_for_item(self, item: Item) -> List[str]:
        tests: List[str] = []
        if self.config.score_include_fail_to_pass:
            for key in ("PASS_TO_PASS", "FAIL_TO_PASS"):
                nodeids = item.get(key)
                if isinstance(nodeids, list):
                    tests.extend([n for n in nodeids if isinstance(n, str)])
        else:
            nodeids = item.get("PASS_TO_PASS")
            if isinstance(nodeids, list):
                tests.extend([n for n in nodeids if isinstance(n, str)])
        return sorted(dict.fromkeys(tests))

    def _chunk_nodeids(self, nodeids: List[str], max_per_chunk: int = 50) -> List[List[str]]:
        return [nodeids[i : i + max_per_chunk] for i in range(0, len(nodeids), max_per_chunk)]

    # =========================================================================
    # Reward: run pytest in the terminal
    # =========================================================================

    async def compute_reward(
        self, item: Item, result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Verify by running pytest with the dataset's nodeids.

        Uses ToolContext.terminal() to run commands in the same
        terminal session as the agent (same task_id = same sandbox).
        """
        repo_dir = self._repo_name(item)

        # Don't reward trajectories that never used tools
        tool_call_count = sum(
            len(msg.get("tool_calls", []))
            for msg in result.messages
            if msg.get("role") == "assistant"
        )
        if tool_call_count == 0:
            print(f"[SweSmithOracleEnv] No tool calls made; score=0.0", flush=True)
            return 0.0

        nodeids = self._tests_for_item(item)
        if not nodeids:
            return 0.0

        # Install deps + run tests
        print(f"[SweSmithOracleEnv] Verifying: installing deps + running tests", flush=True)
        setup_result = ctx.terminal(
            f"cd {repo_dir} && "
            "python -m venv .venv && "
            ". .venv/bin/activate && "
            "python -m pip install -U pip setuptools wheel && "
            "python -m pip install -e . && "
            "python -m pip install pytest",
            timeout=int(self.config.install_timeout_s),
        )
        if setup_result.get("exit_code", 1) != 0:
            print(f"[SweSmithOracleEnv] Install failed; score=0.0", flush=True)
            return 0.0

        # Run test chunks
        chunks = self._chunk_nodeids(nodeids, max_per_chunk=50)
        for chunk_idx, chunk in enumerate(chunks):
            joined = " ".join(chunk)
            test_result = ctx.terminal(
                f"cd {repo_dir} && . .venv/bin/activate && python -m pytest -q {joined}",
                timeout=int(self.config.test_timeout_s),
            )
            if test_result.get("exit_code", 1) != 0:
                print(f"[SweSmithOracleEnv] Tests failed (chunk {chunk_idx}); score=0.0", flush=True)
                return 0.0

        print(f"[SweSmithOracleEnv] All tests passed; score=1.0", flush=True)
        return 1.0

    # =========================================================================
    # Evaluation (minimal for now)
    # =========================================================================

    async def evaluate(self, *args, **kwargs):
        """Placeholder evaluation — SWE tasks are too expensive for frequent eval."""
        start_time = time.time()
        await self.evaluate_log(
            metrics={"eval/placeholder": 0.0},
            samples=[],
            start_time=start_time,
            end_time=time.time(),
        )


if __name__ == "__main__":
    SweSmithOracleEnv.cli()
