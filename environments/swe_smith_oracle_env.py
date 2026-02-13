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
            enabled_toolsets=["terminal", "file"],
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
                server_type="vllm",
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
    # Sandbox hooks: setup_trajectory_workspace + verify_and_score_trajectory
    # =========================================================================

    async def setup_trajectory_workspace(
        self, item: Item, *, trajectory_id: str, exec_tool
    ) -> Dict[str, Any]:
        """
        Prepare a sandbox workspace: bare repo cache + git worktree.

        Uses flock-serialized bare repo cache under /data/repo_cache so
        multiple trajectories sharing a sandbox don't clone the same repo
        in parallel. Each trajectory gets an isolated worktree at the
        specified base_commit.

        Args:
            item: Dataset row with repo, base_commit, etc.
            trajectory_id: Unique trajectory ID
            exec_tool: async callable(tool_name, args, timeout) -> ExecutionResult

        Returns:
            Dict with repo_dir, base_commit metadata
        """
        import time as _time

        t0 = _time.perf_counter()
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

        # Bare repo cache + worktree strategy (same as atropos/envs/swe_smith_oracle_env.py)
        repo_slug = repo.replace("/", "__")
        cache_root = "/data/repo_cache"
        bare_repo = f"{cache_root}/{repo_slug}.git"
        lock_file = f"{cache_root}/.locks/{repo_slug}.lock"

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
            "bash",
            {"command": worktree_cmd},
            timeout=self.config.install_timeout_s,
        )
        if not res.success:
            raise RuntimeError(
                f"git worktree setup failed "
                f"(repo={repo}, base_commit={base_commit}, instance_id={instance_id}): "
                f"{res.error}\n{res.output}"
            )

        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} setup_trajectory_workspace(): "
            f"worktree ready in {_time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        return {"repo_dir": repo_dir, "base_commit": base_commit}

    async def verify_and_score_trajectory(
        self,
        item: Item,
        result: AgentResult,
        *,
        trajectory_id: str,
        exec_tool,
        workspace_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        In-sandbox verification: install deps + run pytest with dataset nodeids.

        Args:
            item: Dataset row
            result: Agent's rollout result
            trajectory_id: Unique trajectory ID
            exec_tool: async callable(tool_name, args, timeout) -> ExecutionResult
            workspace_meta: From setup_trajectory_workspace (has repo_dir)

        Returns:
            (reward, metadata) tuple
        """
        repo_dir = (workspace_meta or {}).get("repo_dir") or self._repo_name(item)

        # Don't reward trajectories that never used tools
        tool_call_count = sum(
            len(msg.get("tool_calls", []))
            for msg in result.messages
            if msg.get("role") == "assistant"
        )
        if tool_call_count == 0:
            print(
                f"[SweSmithOracleEnv] tid={trajectory_id} verify: no tool calls; score=0.0",
                flush=True,
            )
            return 0.0, {"error": "No tool calls were made by the agent"}

        nodeids = self._tests_for_item(item)
        if not nodeids:
            return 0.0, {"error": "No tests provided"}

        # Install dependencies
        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} verify: installing deps + running tests",
            flush=True,
        )
        setup_cmd = (
            f"cd {repo_dir} && "
            "python -m venv .venv && "
            ". .venv/bin/activate && "
            "python -m pip install -U pip setuptools wheel && "
            "python -m pip install -e . && "
            "python -m pip install pytest"
        )
        setup_res = await exec_tool(
            "bash", {"command": setup_cmd}, timeout=self.config.install_timeout_s
        )
        if not setup_res.success:
            print(
                f"[SweSmithOracleEnv] tid={trajectory_id} install failed; score=0.0",
                flush=True,
            )
            return 0.0, {
                "phase": "install",
                "error": setup_res.error,
                "output": setup_res.output,
            }

        # Run test chunks
        chunks = self._chunk_nodeids(nodeids, max_per_chunk=50)
        for chunk_idx, chunk in enumerate(chunks):
            joined = " ".join(chunk)
            cmd = f"cd {repo_dir} && . .venv/bin/activate && python -m pytest -q {joined}"
            res = await exec_tool(
                "bash", {"command": cmd}, timeout=self.config.test_timeout_s
            )
            if not res.success:
                print(
                    f"[SweSmithOracleEnv] tid={trajectory_id} tests failed (chunk {chunk_idx}); score=0.0",
                    flush=True,
                )
                return 0.0, {
                    "phase": "pytest",
                    "failed_chunk": chunk_idx,
                    "error": res.error,
                    "output": res.output,
                }

        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} all tests passed; score=1.0",
            flush=True,
        )
        return 1.0, {"passed": True}

    # =========================================================================
    # Reward: run pytest in the terminal (local / non-sandbox path)
    # =========================================================================

    async def compute_reward(
        self, item: Item, result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Verify by running pytest with the dataset's nodeids.

        Reward structure (shaped to give training signal even when model can't solve tasks):
          - 0.0:  No tool calls at all
          - 0.05: Per valid tool call (up to 0.3 max for tool-call shaping)
          - 0.4:  Successfully installed deps
          - 1.0:  All tests pass

        The partial rewards for tool calls help the model learn to USE tools
        before it can learn to use them CORRECTLY. This is critical for cold-start
        training where the base model barely makes any tool calls.
        """
        repo_dir = self._repo_name(item)

        # Count tool calls. Prefer the agent-loop metrics if present:
        # - attempted: model called a known tool name
        # - schema_valid: args were a dict (no coercion/double-decoding)
        fallback_count = sum(
            len(msg.get("tool_calls", []))
            for msg in result.messages
            if msg.get("role") == "assistant"
        )

        attempted = getattr(result, "tool_calls_attempted", fallback_count)
        schema_valid = getattr(result, "tool_calls_schema_valid", fallback_count)

        if attempted == 0:
            print(f"[SweSmithOracleEnv] No tool calls made; score=0.0", flush=True)
            return 0.0

        # Shaping: reward attempting tool use a little, but reward schema-valid calls more.
        # Full credit per call is still 0.05 when schema_valid.
        attempt_reward = min(attempted * 0.02, 0.10)
        schema_reward = min(schema_valid * 0.03, 0.20)
        tool_call_reward = min(attempt_reward + schema_reward, 0.30)

        nodeids = self._tests_for_item(item)
        if not nodeids:
            # No tests defined — just reward tool usage
            print(f"[SweSmithOracleEnv] No tests defined; score={tool_call_reward:.2f} (tool calls)", flush=True)
            return tool_call_reward

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
            print(f"[SweSmithOracleEnv] Install failed; score={tool_call_reward:.2f} (tool calls only)", flush=True)
            return tool_call_reward

        # Partial reward for successful install
        install_reward = 0.4

        # Run test chunks
        chunks = self._chunk_nodeids(nodeids, max_per_chunk=50)
        for chunk_idx, chunk in enumerate(chunks):
            joined = " ".join(chunk)
            test_result = ctx.terminal(
                f"cd {repo_dir} && . .venv/bin/activate && python -m pytest -q {joined}",
                timeout=int(self.config.test_timeout_s),
            )
            if test_result.get("exit_code", 1) != 0:
                print(f"[SweSmithOracleEnv] Tests failed (chunk {chunk_idx}); score={install_reward:.2f} (install ok)", flush=True)
                return install_reward

        print(f"[SweSmithOracleEnv] All tests passed; score=1.0", flush=True)
        return 1.0

    # =========================================================================
    # Token truncation — keep start of trajectory, truncate from end
    # =========================================================================

    def _build_scored_item(self, item, result, reward):
        """
        Override to truncate tokens/masks from the END to fit within max_token_len.

        Intuition (from NeurIPS finding): the start of the trajectory is most important
        for shifting the model distribution. Truncating from the end only costs ~2-3%
        vs handling the full sequence, but avoids the "Token length is too long" discard
        that throws away entire groups including valid training signal.
        """
        scored_item, remaining = super()._build_scored_item(item, result, reward)
        if scored_item is None:
            return scored_item, remaining

        # Use config.max_token_length as the truncation limit.
        # self.max_token_len comes from the trainer via /info, but may be -1
        # if the trainer hasn't registered yet (race condition).
        max_len = self.max_token_len
        if max_len <= 0:
            # Fallback to config value
            max_len = getattr(self.config, 'max_token_length', 0)
        if max_len <= 0:
            return scored_item, remaining

        # Leave some margin (64 tokens) to avoid edge cases with padding alignment
        truncate_to = max_len - 64

        tokens = scored_item.get("tokens")
        masks = scored_item.get("masks")

        if tokens is not None and len(tokens) >= max_len:
            orig_len = len(tokens)
            scored_item["tokens"] = tokens[:truncate_to]
            if masks is not None and len(masks) >= max_len:
                scored_item["masks"] = masks[:truncate_to]
            logger.info(
                "Truncated trajectory from %d to %d tokens (max_token_len=%d)",
                orig_len, truncate_to, max_len,
            )

        return scored_item, remaining

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
