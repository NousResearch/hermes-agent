"""
Nomad sandbox terminal smoke environment (training-oriented).

Validates, end-to-end:
  BaseEnv.process -> AgentEnv -> ToolExecutor (batched) -> Nomad SlotPool -> sandbox_server

It forces the model to use a sandbox tool by asking it to run a command that
generates a high-entropy token inside the sandbox, then repeat it exactly.

Run (process mode):
  uv run python -m atropos.envs.sandbox_terminal_smoke_env process --env.use_wandb false --env.total_steps 2 --env.group_size 1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import APIServerConfig, Item

from ..agent import AgentConfig, AgentResult
from ..tools import ToolCall
from .agent_env import AgentEnv, AgentEnvConfig

load_dotenv()

STRICT_TOOLCALL_SYSTEM_PROMPT = None


def _forced_tool_item() -> Item:
    # Use double quotes in the shell command and show JSON escaping explicitly.
    # This avoids invalid JSON escapes like `\\'` (not valid JSON) that some models produce.
    cmd = 'python -c "import secrets; print(secrets.token_hex(16))"'
    return {
        "command": cmd,
        "prompt": (
            "You MUST use the terminal tool.\n"
            "Run this exact command:\n"
            f"{cmd}\n"
            "When you call the tool, use valid JSON inside <tool_call>. Example:\n"
            '<tool_call>{"name": "terminal", "arguments": {"command": '
            '"python -c \\\\"import secrets; print(secrets.token_hex(16))\\\\""}}'
            "</tool_call>\n"
            "Then respond with EXACTLY what it printed (the hex token) and nothing else.\n"
            "Do not guess. Do not explain."
        ),
    }


class SandboxTerminalSmokeEnvConfig(AgentEnvConfig):
    server_base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL for an OpenAI-compatible chat server (without /v1).",
    )
    server_model: str = Field(default="hermes-4-36b", description="Model name")
    tokenizer_name: str = Field(default="NousResearch/Hermes-4.3-36B", description="Tokenizer name for RL tokenization")


class SandboxTerminalSmokeEnv(AgentEnv[SandboxTerminalSmokeEnvConfig]):
    name = "sandbox_terminal_smoke_env"
    env_config_cls = SandboxTerminalSmokeEnvConfig

    def __init__(
        self,
        config: SandboxTerminalSmokeEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self._iter = 0

    @classmethod
    def config_init(cls) -> Tuple[SandboxTerminalSmokeEnvConfig, List[APIServerConfig]]:
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "hermes-4-36b"
        api_key = os.getenv("ATROPOS_SERVER_API_KEY") or os.getenv("NOUS_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

        env_config = SandboxTerminalSmokeEnvConfig(
            tokenizer_name=os.getenv("ATROPOS_TOKENIZER_NAME") or "NousResearch/Hermes-4.3-36B",
            group_size=1,
            use_wandb=False,
            include_messages=True,
            ensure_scores_are_not_same=False,
            total_steps=2,
            batch_size=1,
            server_base_url=base_url,
            server_model=model,
            # Tooling: sandbox-only terminal.
            enabled_toolsets=["terminal"],
            disabled_toolsets=[],
            # Default to Nomad sandboxing; users can override via --env.* args.
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
        return _forced_tool_item()

    def build_task(self, item: Item) -> str:
        return str(item.get("prompt") or "")

    def build_agent_config(self, item: Item) -> AgentConfig:  # noqa: ARG002
        # Avoid imposing max_tokens by default; tool-tag responses can be long for some models.
        return AgentConfig(
            max_steps=min(8, int(self.config.agent_max_steps)),
            temperature=0.2,
            max_tokens=None,
            system_prompt=STRICT_TOOLCALL_SYSTEM_PROMPT,
        )

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        # Scoring happens in verify_and_score_trajectory so we can inspect tool results.
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

        observed: str = ""
        tool_ok = False
        for step in agent_result.steps:
            for res in step.tool_results:
                if not res.success:
                    return 0.0, {"error": res.error, "output": res.output}
                out = (res.output or "").strip()
                if out:
                    observed = out.splitlines()[-1].strip()
                    tool_ok = True

        final = (final_response or "").strip()
        score = 1.0 if tool_ok and agent_result.total_tool_calls > 0 and observed and final == observed else 0.0
        return score, {"observed": observed, "tool_calls": agent_result.total_tool_calls, "command": item.get("command")}


if __name__ == "__main__":
    SandboxTerminalSmokeEnv.cli()
