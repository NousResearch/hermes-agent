"""
SWE-smith-oracle environment.

This environment is intentionally minimal:
- prepares a sandbox workspace by cloning a public GitHub repo at `base_commit`
- runs an AtroposAgent tool loop to apply a fix
- verifies by running pytest nodeids from the dataset (reward = pass/fail)
- Python only (no multi-language support currently, need to properly bauild & add to dropbox)

Dataset: NousResearch/SWE-smith-oracle (train; does NOT use SWE-bench eval set).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    verification_mode: Literal["pytest", "install"] = Field(
        default="install",
        description="How to score trajectories: 'pytest' runs dataset tests, 'install' scores based on repo install success.",
    )

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
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
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

        # The dataset "text" field can be extremely large (e.g. includes large code blobs
        # and long test lists). In local dev and bring-up runs this can make the first LLM
        # call appear "hung" while the model chews through a massive prompt. Keep a cap.
        def _cap(s: str, n: int) -> tuple[str, bool]:
            if len(s) <= n:
                return s, False
            return s[:n], True

        problem, problem_trunc = _cap(problem, 8_000)
        context, context_trunc = _cap(context, 12_000)

        nodeids = self._tests_for_item(item)
        tests_preview = "\n".join(f"- {t}" for t in nodeids[:50])
        if len(nodeids) > 50:
            tests_preview += f"\n- ... ({len(nodeids) - 50} more)"

        repo_dir = self._repo_name(item)
        verify_note = ""
        if self.config.verification_mode == "install":
            verify_note = (
                "\nVerification for this run is INSTALL-ONLY:\n"
                "- Your goal is to make `python -m pip install -e .` succeed in a repo-local venv (./.venv).\n"
                "- You may skip running pytest to save time.\n"
            )

        trunc_note = ""
        if problem_trunc or context_trunc:
            trunc_note = (
                "\nNOTE: Some context was truncated to keep prompts manageable in local dev.\n"
                f"- problem_statement_truncated={problem_trunc}\n"
                f"- text_truncated={context_trunc}\n"
            )

        tests_block = (
            "Run these tests to verify:\n"
            f"{tests_preview}\n\n"
            "When done, briefly describe what you changed and confirm tests pass."
        )
        if self.config.verification_mode == "install":
            # Keep install-only prompts short and avoid huge test lists.
            tests_block = (
                "When done, briefly describe what you changed and confirm that "
                "`python -m pip install -e .` succeeds."
            )

        return (
            "You are a senior software engineer. Fix the repository so the specified tests pass.\n\n"
            f"Repository: {repo} (checked out at base_commit={base_commit})\n"
            f"Workspace path: ./{repo_dir}\n\n"
            "Constraints:\n"
            "- Use a workspace-local virtualenv (e.g. inside the repo at ./.venv) to avoid cross-run contamination.\n"
            "- Use non-interactive commands only.\n\n"
            f"{verify_note}\n"
            f"{trunc_note}\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            "Additional context:\n"
            f"{context}\n\n"
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

        # Prefer a lightweight "fetch by sha" to avoid pulling full history.
        # If it fails (some servers disallow fetching unadvertised objects, or we hit
        # shallow-object edge cases), fall back to a full clone.
        clone_attempts: list[tuple[str, str]] = []
        clone_attempts.append(
            (
                "shallow_fetch_sha",
                (
                    f"rm -rf {repo_dir} && "
                    f"git init {repo_dir} && "
                    f"cd {repo_dir} && "
                    "export GIT_TERMINAL_PROMPT=0 && "
                    "export GIT_LFS_SKIP_SMUDGE=1 && "
                    "git config advice.detachedHead false && "
                    f"git remote add origin {clone_url} && "
                    f"git fetch --depth 1 origin {base_commit} && "
                    "git checkout -q FETCH_HEAD"
                ),
            )
        )
        clone_attempts.append(
            (
                "full_clone_checkout",
                (
                    f"rm -rf {repo_dir} && "
                    f"GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git clone {clone_url} {repo_dir} && "
                    f"cd {repo_dir} && "
                    "git config advice.detachedHead false && "
                    f"git checkout -q {base_commit}"
                ),
            )
        )

        clone_res = None
        for label, cmd in clone_attempts:
            t_attempt = time.perf_counter()
            print(f"[SweSmithOracleEnv] tid={trajectory_id} clone attempt: {label}", flush=True)
            res = await exec_tool(
                ToolCall(
                    name="terminal",
                    arguments={"command": cmd, "timeout": self.config.install_timeout_s},
                )
            )
            clone_res = res
            if res.success:
                print(
                    f"[SweSmithOracleEnv] tid={trajectory_id} clone ok ({label}) in {time.perf_counter() - t_attempt:.2f}s",
                    flush=True,
                )
                break
            print(
                f"[SweSmithOracleEnv] tid={trajectory_id} clone failed ({label}) in {time.perf_counter() - t_attempt:.2f}s: "
                f"{res.error}",
                flush=True,
            )

        if clone_res is None or not clone_res.success:
            err = clone_res.error if clone_res is not None else "unknown"
            out = clone_res.output if clone_res is not None else ""
            raise RuntimeError(
                "git clone/checkout failed "
                f"(repo={repo}, base_commit={base_commit}, instance_id={instance_id}): {err}\n{out}"
            )

        print(
            f"[SweSmithOracleEnv] tid={trajectory_id} setup_trajectory_workspace(): clone complete in {time.perf_counter() - t0:.2f}s",
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
        agent_result=None,  # noqa: ARG002
        workspace_meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Dict[str, Any]]:
        _ = trajectory_id
        repo_dir = self._repo_name(item)

        if self.config.verification_mode == "install":
            print(f"[SweSmithOracleEnv] tid={trajectory_id} verify (install): running pip install -e .", flush=True)
            t0 = time.perf_counter()
            install_cmd = (
                f"cd {repo_dir} && "
                "python -m venv .venv && "
                ". .venv/bin/activate && "
                "python -m pip install -U pip setuptools wheel && "
                "python -m pip install -e ."
            )
            res = await exec_tool(
                ToolCall(name="terminal", arguments={"command": install_cmd, "timeout": self.config.install_timeout_s})
            )
            ok = bool(res.success)
            print(
                f"[SweSmithOracleEnv] tid={trajectory_id} verify (install): {'ok' if ok else 'fail'} "
                f"in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
            return (1.0 if ok else 0.0), {
                "verification_mode": "install",
                "install_success": ok,
                "error": res.error,
            }

        nodeids = self._tests_for_item(item)
        if not nodeids:
            return 0.0, {"error": "No tests provided"}

        print(f"[SweSmithOracleEnv] tid={trajectory_id} verify (pytest): ensuring venv + deps", flush=True)
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
        if not setup_res.success:
            return 0.0, {
                "verification_mode": "pytest",
                "phase": "install",
                "error": setup_res.error,
                "output": setup_res.output,
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
            if not res.success:
                return 0.0, {"failed_chunk": chunk_idx, "error": res.error, "output": res.output}

        return 1.0, {"verification_mode": "pytest", "passed": True}

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        # Not used; scoring happens in verify_and_score_trajectory.
        _ = (item, final_response)
        return 0.0


if __name__ == "__main__":
    SweSmithOracleEnv.cli()
