"""
SWE-smith-oracle environment.

This environment is intentionally minimal:
- prepares a sandbox workspace by cloning a public GitHub repo at `base_commit`
- runs an AtroposAgent tool loop to apply a fix
- verifies by running pytest nodeids from the dataset (reward = pass/fail)
- Python only (no multi-language support currently, need to properly bauild & add to dropbox)
- TODO: Get the other nonpython sandboxes up and running, then add a config knob to switch between them per row
- oh and add to dockerhub

Dataset: NousResearch/SWE-smith-oracle (train; does NOT use SWE-bench eval set).
"""

from __future__ import annotations

import os
import random
import time
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
        default=True,
        description=(
            "If true (default), score tests on PASS_TO_PASS ∪ FAIL_TO_PASS. "
            "Disable to only run PASS_TO_PASS (faster but weaker signal)."
        ),
    )

    prompt_mode: str = Field(
        default="problem_statement",
        description="Task prompt content: 'problem_statement' (fast) or 'problem_statement+text' (slower, includes dataset 'text').",
    )

    repo_base_url: str = Field(default="https://github.com", description="Base URL for repo cloning")
    install_timeout_s: float = Field(default=600.0)
    test_timeout_s: float = Field(default=600.0)

    tokenizer_name: str = Field(default="NousResearch/Hermes-4.3-36B", description="Tokenizer name for RL tokenization")


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
        base_url_raw = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
        base_url = base_url_raw.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "hermes-4-36b"
        api_key = os.getenv("ATROPOS_SERVER_API_KEY") or os.getenv("NOUS_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

        env_config = SweSmithOracleEnvConfig(
            tokenizer_name=os.getenv("ATROPOS_TOKENIZER_NAME") or "NousResearch/Hermes-4.3-36B",
            group_size=1,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=8192,
            inference_weight=1.0,
            wandb_name="swe_smith_oracle",
            enabled_toolsets=["terminal"],
            disabled_toolsets=[],
            sandbox_image=os.getenv("ATROPOS_SANDBOX_IMAGE") or "atropos-sandbox:local",
            purge_job_on_start=True,
            purge_job_on_shutdown=True,
        )

        server_configs = [
            APIServerConfig(
                model_name=model,
                base_url=base_url,
                api_key=api_key,
                num_max_requests_at_once=1,
                num_requests_for_eval=1,
                timeout=int(os.getenv("ATROPOS_SERVER_TIMEOUT_S") or "300"),
            ),
        ]

        return env_config, server_configs

    async def setup_agent_env(self) -> None:
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
            f"[SweSmithOracleEnv] loaded {len(self._indices)} items from {self.config.dataset_name}:{self.config.dataset_split} "
            f"in {time.perf_counter() - t0:.2f}s",
            flush=True,
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
        print(f"[SweSmithOracleEnv] get_next_item() cursor={self._cursor}/{len(self._indices)}", flush=True)
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
        problem = str(item.get("problem_statement") or "")
        context = str(item.get("text") or "")

        nodeids = self._tests_for_item(item)
        tests_list = "\n".join(f"- {t}" for t in nodeids)

        repo_dir = self._repo_name(item)

        tests_block = (
            "Run these tests to verify:\n"
            f"{tests_list}\n\n"
            "When done, briefly describe what you changed and confirm tests pass."
        )

        prompt_mode = (self.config.prompt_mode or "problem_statement").strip().lower()
        if prompt_mode not in {"problem_statement", "problem_statement+text"}:
            raise ValueError(
                f"Invalid prompt_mode={self.config.prompt_mode!r}. "
                "Expected 'problem_statement' or 'problem_statement+text'."
            )

        context_block = ""
        if prompt_mode == "problem_statement+text" and context:
            # Note: We intentionally do NOT truncate/cap here. This mode is for debugging / richer prompts and can be slow.
            context_block = f"\nAdditional context:\n{context}\n"

        return (
            "You are a senior software engineer. Fix the repository so the specified tests pass.\n\n"
            f"Repository: {repo} (checked out at base_commit={base_commit})\n"
            f"Workspace path: ./{repo_dir}\n\n"
            "Constraints:\n"
            "- You MUST use the terminal tool to inspect, edit, and verify the repository. Do not respond with a patch file.\n"
            f"- Start by inspecting the repo (e.g. `ls`, `cd ./{repo_dir}`, `git status`).\n"
            "- Use a workspace-local virtualenv (e.g. inside the repo at ./.venv) to avoid cross-run contamination.\n"
            "- Use non-interactive commands only.\n\n"
            "- Terminal commands run under POSIX /bin/sh and each tool call runs in a fresh shell (no persisted env vars).\n"
            "  Avoid bash-only `source`; prefer `. .venv/bin/activate` or `.venv/bin/python ...`.\n\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            f"{context_block}\n"
            f"{tests_block}"
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
        t0 = time.perf_counter()
        repo = item.get("repo")
        base_commit = item.get("base_commit")
        instance_id = item.get("instance_id") or item.get("id") or item.get("problem_id")
        if not isinstance(repo, str) or not isinstance(base_commit, str):
            raise RuntimeError("Invalid dataset row: missing repo/base_commit")

        repo_dir = self._repo_name(item)
        clone_url = f"{self.config.repo_base_url.rstrip('/')}/{repo}.git"
        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} setup_trajectory_workspace(): "
            f"repo={repo} base_commit={base_commit} instance_id={instance_id} dir=./{repo_dir}",
            flush=True,
        )

        # Repo setup strategy:
        # - Maintain a shared, per-container bare repo cache under /data/repo_cache
        # - For each trajectory, create an isolated git worktree under the slot workspace
        # This avoids cloning/fetching full repos per trajectory and is crucial for multiplexing.

        def _repo_cache_slug(repo_name: str) -> str:
            return repo_name.replace("/", "__")

        repo_slug = _repo_cache_slug(repo)
        cache_root = "/data/repo_cache"
        bare_repo = f"{cache_root}/{repo_slug}.git"
        lock_file = f"{cache_root}/.locks/{repo_slug}.lock"

        # Use flock to serialize operations that mutate the shared bare repo (fetch/worktree).
        # util-linux (flock) is included in the sandbox image.
        worktree_cmd = (
            "set -e; "
            f"rm -rf {repo_dir}; "
            f"mkdir -p {cache_root}/.locks; "
            f": > {lock_file}; "
            f"flock -x {lock_file} sh -lc '"
            f"set -e; "
            "export GIT_TERMINAL_PROMPT=0; "
            "export GIT_LFS_SKIP_SMUDGE=1; "
            f"if [ ! -d \"{bare_repo}\" ]; then "
            f"  git init --bare \"{bare_repo}\"; "
            f"  git -C \"{bare_repo}\" remote add origin \"{clone_url}\"; "
            "fi; "
            f"git -C \"{bare_repo}\" remote set-url origin \"{clone_url}\"; "
            f"git -C \"{bare_repo}\" worktree prune || true; "
            f"if ! git -C \"{bare_repo}\" cat-file -e \"{base_commit}^{{commit}}\" 2>/dev/null; then "
            f"  git -C \"{bare_repo}\" fetch --depth 1 origin \"{base_commit}\" || true; "
            "fi; "
            f"if ! git -C \"{bare_repo}\" cat-file -e \"{base_commit}^{{commit}}\" 2>/dev/null; then "
            f"  git -C \"{bare_repo}\" fetch --prune origin; "
            "fi; "
            f"git --git-dir=\"{bare_repo}\" worktree add --detach \"{repo_dir}\" \"{base_commit}\"; "
            "'"
        )

        print(f"[SweSmithOracleEnv] tid={trajectory_id} preparing worktree from repo cache", flush=True)
        res = await exec_tool(
            ToolCall(
                name="terminal",
                arguments={"command": worktree_cmd, "timeout": self.config.install_timeout_s},
            )
        )
        if not res.success:
            raise RuntimeError(
                "git worktree setup failed "
                f"(repo={repo}, base_commit={base_commit}, instance_id={instance_id}): {res.error}\n{res.output}"
            )

        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} setup_trajectory_workspace(): worktree ready in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
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
        agent_result=None,
        workspace_meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Dict[str, Any]]:
        _ = trajectory_id
        repo_dir = self._repo_name(item)

        # Training correctness: do not reward trajectories that never actually used tools.
        if agent_result is not None and getattr(agent_result, "total_tool_calls", 0) <= 0:
            print(
                f"[SweSmithOracleEnv] tid={trajectory_id} verify (dataset_tests): no tool calls; score=0.0",
                flush=True,
            )
            return 0.0, {
                "verification_mode": "dataset_tests",
                "error": "No tool calls were made by the agent",
            }

        nodeids = self._tests_for_item(item)
        if not nodeids:
            return 0.0, {"error": "No tests provided"}

        print(f"[SweSmithOracleEnv] tid={trajectory_id} verify (dataset_tests): ensuring venv + deps", flush=True)
        setup_cmd = (
            f"cd {repo_dir} && "
            "python -m venv .venv && "
            ". .venv/bin/activate && "
            "python -m pip install -U pip setuptools wheel && "
            "python -m pip install -e . && "
            "python -m pip install pytest"
        )
        setup_res = await exec_tool(
            ToolCall(name="terminal", arguments={"command": setup_cmd, "timeout": self.config.install_timeout_s})
        )
        verification_messages = [{"role": "user", "content": setup_res.to_xml()}]
        if not setup_res.success:
            return 0.0, {
                "verification_mode": "dataset_tests",
                "phase": "install",
                "error": setup_res.error,
                "output": setup_res.output,
                "verification_messages": verification_messages,
            }

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
            verification_messages.append({"role": "user", "content": res.to_xml()})
            if not res.success:
                return 0.0, {
                    "verification_mode": "dataset_tests",
                    "phase": "pytest",
                    "failed_chunk": chunk_idx,
                    "error": res.error,
                    "output": res.output,
                    "verification_messages": verification_messages,
                }

        return 1.0, {"verification_mode": "dataset_tests", "passed": True, "verification_messages": verification_messages}

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        # Not used; scoring happens in verify_and_score_trajectory.
        _ = (item, final_response)
        return 0.0


if __name__ == "__main__":
    SweSmithOracleEnv.cli()
