"""
Hermes-Agent (Atropos-compatible) smoke environment.

This is a minimal `BaseEnv` environment that uses Hermes-Agent's Atropos-backed
runner (`AtroposAIAgent`) and can be exercised via `BaseEnv`'s `process` mode.

This deliberately does NOT use slot multiplexing / sandboxes yet (stage 1).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, Item

load_dotenv()


def _build_forced_tool_item() -> Item:
    """
    Construct a task that *cannot* be completed reliably without executing a tool.

    We generate a high-entropy token *inside the tool execution* and ask the agent to
    repeat it exactly. Scoring verifies that:
      - a terminal tool call occurred (role="tool" message present), and
      - the final answer matches the tool stdout exactly.
    """
    return {
        "command": "python -c \"import secrets; print(secrets.token_hex(16))\"",
        "prompt": (
            "Use the terminal tool to run:\n"
            "python -c \"import secrets; print(secrets.token_hex(16))\"\n"
            "Then answer with EXACTLY what it printed and nothing else."
        ),
    }


TEST_ITEMS: List[Item] = [
    _build_forced_tool_item(),
    _build_forced_tool_item(),
]


class HermesCompatTestEnvConfig(BaseEnvConfig):
    """Config for HermesCompatTestEnv."""

    server_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for an OpenAI-compatible chat server (without /v1).",
    )
    server_model: str = Field(default="glm-4.7-flash", description="Model name")


class HermesCompatTestEnv(BaseEnv):
    """
    Minimal BaseEnv that runs Hermes-Agent's Atropos-compatible agent loop.

    Run (process mode):
      uv run atropos-agent-hermes-compat-test process --env.use_wandb false --env.total_steps 2 --env.group_size 1
    """

    name = "hermes_compat_test_env"
    env_config_cls = HermesCompatTestEnvConfig

    def __init__(
        self,
        config: HermesCompatTestEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config=config, server_configs=server_configs, slurm=slurm, testing=testing)
        self._iter = 0

        from atropos_compatible_agent import AtroposAIAgent  # noqa: WPS433

        # Only expose terminal for this smoke env.
        self._agent = AtroposAIAgent(
            server=self.server,
            tokenizer=self.tokenizer,
            model=getattr(config, "server_model", "local"),
            max_iterations=8,
            enabled_toolsets=["terminal"],
            tool_delay=0.0,
            # Let the server decide token limits; we care about tool calling correctness here.
            max_tokens=None,
            temperature=None,
        )

    @classmethod
    def config_init(cls) -> Tuple[HermesCompatTestEnvConfig, List[APIServerConfig]]:
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:11434"
        )
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "glm-4.7-flash"
        # Never pass through real API keys in this smoke env (they will be printed by BaseEnv config logging).
        # Local OpenAI-compatible servers typically ignore the API key anyway.
        api_key = "local"

        env_config = HermesCompatTestEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=1,
            use_wandb=False,
            include_messages=True,
            ensure_scores_are_not_same=False,
            total_steps=2,
            batch_size=1,
            server_base_url=base_url,
            server_model=model,
        )

        server_configs = [
            APIServerConfig(
                server_type="openai",
                model_name=model,
                base_url=f"{base_url}/v1",
                api_key=api_key,
                num_max_requests_at_once=1,
                num_requests_for_eval=1,
                timeout=120,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        return None

    async def get_next_item(self) -> Item:
        # Regenerate token per task to avoid leakage across steps.
        item = _build_forced_tool_item()
        self._iter += 1
        return item

    async def collect_trajectory(self, item: Item):
        prompt = item.get("prompt", "")

        result = await self._agent.run_conversation_async(
            prompt,
            task_id=str(uuid.uuid4()),
        )

        final = (result.get("final_response") or "").strip()

        # Verify the agent actually executed the tool by extracting stdout from the tool message.
        observed: str = ""
        saw_tool = False
        for msg in result.get("messages", []):
            if msg.get("role") == "tool":
                saw_tool = True
                # Tool messages contain JSON strings from terminal tool.
                try:
                    payload = json.loads(msg.get("content") or "{}")
                    out = (payload.get("output") or "").strip()
                    if out:
                        observed = out.splitlines()[-1].strip()
                except Exception:
                    continue
        # Pass if:
        # - a tool call occurred, and
        # - the final answer matches the observed stdout exactly.
        score = 1.0 if saw_tool and observed and final == observed else 0.0

        # Tokenization fallback: build tokens/masks from final prompt + completion.
        # Note: this is sufficient for smoke testing; production training should
        # use a backend that supports ManagedServer state tracking.
        system_prompt = result.get("system_prompt")
        messages: List[Dict[str, str]] = result.get("messages", [])
        prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages

        if system_prompt:
            prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in prompt_messages])
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

        output_tokens = self.tokenizer.encode(final, add_special_tokens=False)

        scored = {
            "tokens": prompt_tokens + output_tokens,
            "masks": ([-100] * len(prompt_tokens)) + output_tokens,
            "scores": score,
            "messages": prompt_messages + [{"role": "assistant", "content": final}],
        }

        return scored, []

    async def evaluate(self, *args, **kwargs):  # noqa: ARG002
        # Minimal eval hook for BaseEnv abstract method.
        return {}


if __name__ == "__main__":
    HermesCompatTestEnv.cli()
