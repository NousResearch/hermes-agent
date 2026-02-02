"""
SWE-smith-oracle benchmark environment (Phase 4.7).

This environment is intentionally minimal:
- prepares a sandbox workspace by cloning a public GitHub repo at `base_commit`
- runs an AtroposAgent tool loop to apply a fix
- verifies by running pytest nodeids from the dataset (reward = pass/fail)

Dataset: NousResearch/SWE-smith-oracle (train; does NOT use SWE-bench eval set).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import APIServerConfig, Item

from ..agent import AgentConfig
from ..tools import ToolCall
from .agent_env import AgentEnv, AgentEnvConfig


class SweSmithOracleEnvConfig(AgentEnvConfig):
    dataset_name: str = Field(default="NousResearch/SWE-smith-oracle")
    dataset_split: str = Field(default="train")
    max_items: int = Field(default=0, description="0 = no limit")
    shuffle: bool = Field(default=True)
    seed: int = Field(default=0)

    python_only: bool = Field(default=True, description="Filter to Python-evaluable rows")
    score_include_fail_to_pass: bool = Field(
        default=False,
        description="If true, score tests on PASS_TO_PASS ∪ FAIL_TO_PASS; else PASS_TO_PASS only.",
    )

    repo_base_url: str = Field(default="https://github.com", description="Base URL for repo cloning")
    install_timeout_s: float = Field(default=600.0)
    test_timeout_s: float = Field(default=600.0)


class SweSmithOracleEnv(AgentEnv[SweSmithOracleEnvConfig]):
    """
    SWE-smith-oracle AgentEnv.

    This is designed for benchmarking multiplexed slot execution vs naive container-per-trajectory.
    """

    name = "swe_smith_oracle_env"
    env_config_cls = SweSmithOracleEnvConfig

    def __init__(
        self,
        config: SweSmithOracleEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self._dataset = None
        self._indices: List[int] = []
        self._cursor = 0

    @classmethod
    def config_init(cls) -> Tuple[SweSmithOracleEnvConfig, List[APIServerConfig]]:
        # Defaults for running the env via CLI in offline `process` mode.
        # Override via env vars or `--env.*` flags as needed.
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:11434"
        )
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "glm-4.7-flash"
        api_key = os.getenv("ATROPOS_SERVER_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

        env_config = SweSmithOracleEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",  # tokenization only
            group_size=1,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=8192,
            inference_weight=1.0,
            wandb_name="swe_smith_oracle",
        )

        server_configs = [
            APIServerConfig(
                model_name=model,
                base_url=f"{base_url.rstrip('/')}/v1",
                api_key=api_key,
                num_max_requests_at_once=1,
                num_requests_for_eval=1,
                timeout=300,
            ),
        ]

        return env_config, server_configs

    async def setup_agent_env(self) -> None:
        from datasets import load_dataset

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
            f"SweSmithOracleEnv loaded {len(self._indices)} items from {self.config.dataset_name}:{self.config.dataset_split}"
        )

    def _is_python_row(self, row: Dict[str, Any]) -> bool:
        nodeids = row.get("PASS_TO_PASS")
        if not isinstance(nodeids, list) or not nodeids:
            return False
        for nid in nodeids:
            if not isinstance(nid, str) or ".py::" not in nid:
                return False
        return True

    async def get_next_item(self) -> Item:
        if not self._dataset or not self._indices:
            raise RuntimeError("Dataset not initialized (did setup() run?)")
        if self._cursor >= len(self._indices):
            self._cursor = 0
        idx = self._indices[self._cursor]
        self._cursor += 1
        return dict(self._dataset[idx])

    def _repo_name(self, item: Item) -> str:
        repo = item.get("repo") or ""
        if isinstance(repo, str) and "/" in repo:
            return repo.split("/")[-1]
        return "repo"

    def build_task(self, item: Item) -> str:
        repo = item.get("repo") or ""
        base_commit = item.get("base_commit") or ""
        problem = item.get("problem_statement") or ""
        context = item.get("text") or ""

        nodeids = self._tests_for_item(item)
        tests_preview = "\n".join(f"- {t}" for t in nodeids[:50])
        if len(nodeids) > 50:
            tests_preview += f"\n- ... ({len(nodeids) - 50} more)"

        repo_dir = self._repo_name(item)
        return (
            "You are a senior software engineer. Fix the repository so the specified tests pass.\n\n"
            f"Repository: {repo} (checked out at base_commit={base_commit})\n"
            f"Workspace path: ./{repo_dir}\n\n"
            "Constraints:\n"
            "- Use a workspace-local virtualenv (e.g. inside the repo at ./.venv) to avoid cross-run contamination.\n"
            "- Use non-interactive commands only.\n\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            "Additional context:\n"
            f"{context}\n\n"
            "Run these tests to verify:\n"
            f"{tests_preview}\n\n"
            "When done, briefly describe what you changed and confirm tests pass."
        )

    def build_agent_config(self, item: Item) -> AgentConfig:  # noqa: ARG002
        # SWE tasks are longer than the simple test env.
        return AgentConfig(
            max_steps=self.config.agent_max_steps,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.agent_max_tokens,
            tool_delay_s=self.config.agent_tool_delay_s,
        )

    async def setup_trajectory_workspace(self, item: Item, *, trajectory_id: str, exec_tool) -> Dict[str, Any]:
        _ = trajectory_id
        repo = item.get("repo")
        base_commit = item.get("base_commit")
        if not isinstance(repo, str) or not isinstance(base_commit, str):
            raise RuntimeError("Invalid dataset row: missing repo/base_commit")

        repo_dir = self._repo_name(item)
        clone_url = f"{self.config.repo_base_url.rstrip('/')}/{repo}.git"

        # Clone and checkout the base commit.
        clone_cmd = f"rm -rf {repo_dir} && git clone {clone_url} {repo_dir}"
        res = await exec_tool(ToolCall(name="terminal", arguments={"command": clone_cmd, "timeout": self.config.install_timeout_s}))
        if not res.success:
            raise RuntimeError(f"git clone failed: {res.error}\n{res.output}")

        checkout_cmd = f"cd {repo_dir} && git checkout {base_commit}"
        res = await exec_tool(ToolCall(name="terminal", arguments={"command": checkout_cmd, "timeout": self.config.install_timeout_s}))
        if not res.success:
            raise RuntimeError(f"git checkout failed: {res.error}\n{res.output}")

        # Best-effort baseline python env.
        setup_cmd = (
            f"cd {repo_dir} && "
            "python -m venv .venv && "
            ". .venv/bin/activate && "
            "python -m pip install -U pip setuptools wheel && "
            "python -m pip install -e . && "
            "python -m pip install pytest"
        )
        await exec_tool(ToolCall(name="terminal", arguments={"command": setup_cmd, "timeout": self.config.install_timeout_s}))

        return {"repo_dir": repo_dir, "base_commit": base_commit}

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
        # Stable order for reproducibility.
        return sorted(dict.fromkeys(tests))

    def _chunk_nodeids(self, nodeids: List[str], max_per_chunk: int = 50) -> List[List[str]]:
        chunks: List[List[str]] = []
        for i in range(0, len(nodeids), max_per_chunk):
            chunks.append(nodeids[i : i + max_per_chunk])
        return chunks

    async def verify_and_score_trajectory(
        self,
        item: Item,
        final_response: str,  # noqa: ARG002
        *,
        trajectory_id: str,
        exec_tool,
    ) -> tuple[float, Dict[str, Any]]:
        _ = trajectory_id
        repo_dir = self._repo_name(item)
        nodeids = self._tests_for_item(item)
        if not nodeids:
            return 0.0, {"error": "No tests provided"}

        chunks = self._chunk_nodeids(nodeids, max_per_chunk=50)
        for chunk_idx, chunk in enumerate(chunks):
            joined = " ".join(chunk)
            cmd = f"cd {repo_dir} && . .venv/bin/activate && python -m pytest -q {joined}"
            res = await exec_tool(
                ToolCall(
                    name="terminal",
                    arguments={"command": cmd, "timeout": self.config.test_timeout_s},
                )
            )
            if not res.success:
                return 0.0, {"failed_chunk": chunk_idx, "error": res.error, "output": res.output}

        return 1.0, {"passed": True}

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        # Not used; scoring happens in verify_and_score_trajectory.
        _ = (item, final_response)
        return 0.0


if __name__ == "__main__":
    SweSmithOracleEnv.cli()
