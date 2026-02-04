"""
ToolServer routing smoke environment.

Validates that:
  - sandbox tools run through Nomad SlotPool (terminal -> bash in sandbox)
  - external tools run through ToolServer (skills_list)

This env uses ToolServer in-process by default (`tool_server_url="inprocess"`),
so it is self-contained for local testing.

Run:
  uv run python -m atropos.envs.toolserver_smoke_env process --env.use_wandb false --env.total_steps 1 --env.group_size 1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import APIServerConfig, Item

from ..agent import AgentConfig, AgentResult
from .agent_env import AgentEnv, AgentEnvConfig

load_dotenv()


class ToolServerSmokeEnvConfig(AgentEnvConfig):
    server_base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL for an OpenAI-compatible chat server (without /v1).",
    )
    server_model: str = Field(default="hermes-4-36b", description="Model name")
    tokenizer_name: str = Field(default="NousResearch/Hermes-4.3-36B", description="Tokenizer name for RL tokenization")


class ToolServerSmokeEnv(AgentEnv[ToolServerSmokeEnvConfig]):
    name = "toolserver_smoke_env"
    env_config_cls = ToolServerSmokeEnvConfig

    def __init__(
        self,
        config: ToolServerSmokeEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self._iter = 0

    @classmethod
    def config_init(cls) -> Tuple[ToolServerSmokeEnvConfig, List[APIServerConfig]]:
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "hermes-4-36b"
        api_key = os.getenv("ATROPOS_SERVER_API_KEY") or os.getenv("NOUS_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

        env_config = ToolServerSmokeEnvConfig(
            tokenizer_name=os.getenv("ATROPOS_TOKENIZER_NAME") or "NousResearch/Hermes-4.3-36B",
            group_size=1,
            use_wandb=False,
            include_messages=True,
            ensure_scores_are_not_same=False,
            total_steps=1,
            batch_size=1,
            server_base_url=base_url,
            server_model=model,
            enabled_toolsets=["terminal", "skills"],
            disabled_toolsets=[],
            # Self-contained ToolServer for local smoke.
            tool_server_url="inprocess",
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
                timeout=120,
            )
        ]
        return env_config, server_configs

    async def setup_agent_env(self) -> None:
        return None

    async def get_next_item(self) -> Item:
        self._iter += 1
        return {
            "prompt": (
                "You MUST call exactly one tool per assistant message.\n"
                "\n"
                "Step 1) Call the skills_list tool (no arguments), then stop.\n"
                "Step 2) After you receive the tool response, call the terminal tool to run:\n"
                "python -c \"print('ok')\"\n"
                "Step 3) After you receive the terminal tool response, answer with just: ok\n"
                "\n"
                "Tool call format requirements:\n"
                "- Every tool call MUST be a complete XML block with a closing tag.\n"
                "- Do NOT emit a second <tool_call> in the same assistant message.\n"
                "\n"
                "Example:\n"
                "<tool_call>{\"name\": \"skills_list\", \"arguments\": {}}</tool_call>\n"
                "Do not include anything else in your final answer."
            )
        }

    def build_task(self, item: Item) -> str:
        return str(item.get("prompt") or "")

    def build_agent_config(self, item: Item) -> AgentConfig:  # noqa: ARG002
        return AgentConfig(
            max_steps=min(10, int(self.config.agent_max_steps)),
            temperature=0.2,
            max_tokens=None,
        )

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        _ = (item, final_response)
        return 0.0

    async def verify_and_score_trajectory(
        self,
        item: Item,
        final_response: str,
        *,
        trajectory_id: str,  # noqa: ARG002
        exec_tool,  # noqa: ARG002
        agent_result: AgentResult | None = None,
        workspace_meta: Dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[float, Dict[str, Any]]:
        if agent_result is None:
            return 0.0, {"error": "Missing agent_result"}

        called = {c.name for s in agent_result.steps for c in s.tool_calls}
        need = {"skills_list", "terminal"}
        if not need.issubset(called):
            return 0.0, {"error": f"Missing tool calls: {sorted(need - called)}", "called": sorted(called)}

        terminal_ok = False
        for step in agent_result.steps:
            for call, res in zip(step.tool_calls, step.tool_results):
                if call.name != "terminal":
                    continue
                if res.success and (res.output or "").strip().splitlines()[-1].strip() == "ok":
                    terminal_ok = True

        score = 1.0 if terminal_ok and (final_response or "").strip() == "ok" else 0.0
        return score, {"called": sorted(called), "final": (final_response or "").strip()}


if __name__ == "__main__":
    ToolServerSmokeEnv.cli()
