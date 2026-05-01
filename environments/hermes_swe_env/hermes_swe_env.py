"""
HermesSweEnv -- SWE-Bench Style Environment with Modal Sandboxes

A concrete environment for software engineering tasks where the model writes code
and the reward function runs tests to verify correctness. Uses Modal terminal
backend for cloud-isolated sandboxes per rollout.

The reward function uses ToolContext.terminal() to run test commands in the same
Modal sandbox the model used during its agentic loop. All filesystem state from
the model's tool calls is preserved for verification.

Usage:
    # Phase 1: OpenAI server type
    vllm serve YourModel --tool-parser hermes
    run-api
    python environments/hermes_swe_env.py serve \\
        --openai.base_url http://localhost:8000/v1 \\
        --openai.model_name YourModel \\
        --openai.server_type openai \\
        --env.dataset_name bigcode/humanevalpack \\
        --env.terminal_backend modal

    # Phase 2: VLLM server type (full RL training)
    python environments/hermes_swe_env.py serve \\
        --openai.base_url http://localhost:8000/v1 \\
        --openai.model_name YourModel \\
        --openai.server_type vllm \\
        --env.tool_call_parser hermes \\
        --env.terminal_backend modal
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datasets import load_dataset

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item

from environments.agent_loop import AgentResult
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


class HermesSweEnvConfig(HermesAgentEnvConfig):
    """Config with defaults for SWE-bench style tasks."""

    pass  # Inherits all fields, overrides defaults in config_init


class HermesSweEnv(HermesAgentBaseEnv):
    """
    SWE-bench style environment using Modal terminal backend.

    The model gets a coding task, uses terminal + file + web tools to solve it,
    and the reward function runs tests in the same Modal sandbox to verify.

    Subclass this for specific SWE datasets (HumanEval, SWE-bench, etc.)
    and customize format_prompt() and compute_reward() as needed.
    """

    name = "hermes-swe"
    env_config_cls = HermesSweEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[HermesSweEnvConfig, List[APIServerConfig]]:
        """
        Default configuration for the SWE environment.

        Uses Modal terminal backend for cloud isolation and terminal + file + web toolsets.
        """
        env_config = HermesSweEnvConfig(
            # Toolsets: terminal for running code, file for reading/writing, web for docs
            enabled_toolsets=["terminal", "file", "web"],
            disabled_toolsets=None,
            distribution=None,
            # Agent settings -- SWE tasks need more turns
            max_agent_turns=30,
            max_token_length=4096,
            agent_temperature=1.0,
            system_prompt=(
                "You are a skilled software engineer. You have access to a terminal, "
                "file tools, and web search. Use these tools to complete the coding task. "
                "Write clean, working code and verify it runs correctly before finishing."
            ),
            # Modal backend for cloud-isolated sandboxes
            terminal_backend="modal",
            # Dataset -- override via CLI for your specific SWE dataset
            dataset_name="bigcode/humanevalpack",
            dataset_split="test",
            prompt_field="prompt",
            # Atropos settings
            group_size=4,
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            tool_call_parser="hermes",
            steps_per_eval=50,
            total_steps=500,
            use_wandb=True,
            wandb_name="hermes-swe",
        )

        server_configs = [
            APIServerConfig(
                base_url="http://localhost:8000/v1",
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                server_type="openai",  # Phase 1; switch to "vllm" for Phase 2
                api_key="",
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Load the SWE dataset."""
        if self.config.dataset_name:
            self.dataset = load_dataset(
                self.config.dataset_name, split=self.config.dataset_split
            )
        else:
            # Placeholder if no dataset specified
            self.dataset = []
        self.iter = 0
        self.reward_buffer: List[float] = []

    async def get_next_item(self) -> Dict[str, Any]:
        """Cycle through the SWE dataset."""
        if not self.dataset:
            raise ValueError("No dataset loaded. Set dataset_name in config.")
        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """
        Format the SWE task prompt.

        Override this in subclasses for different dataset formats.
        Default assumes the dataset has a 'prompt' field and optionally a 'test' field.
        """
        prompt = item.get(self.config.prompt_field, "")

        # If the dataset has test information, include it in the prompt
        test_info = item.get("test", item.get("test_code", item.get("tests", "")))
        if test_info:
            prompt += f"\n\nTests to pass:\n{test_info}"

        return prompt

    async def compute_reward(
        self, item: Dict[str, Any], result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Score by running tests in the model's Modal sandbox.

        Default implementation:
        - If the dataset item has a 'test' or 'test_code' field, run it
        - Check exit code: 0 = pass, non-zero = fail
        - Partial credit for file creation

        Override this in subclasses for more sophisticated reward logic.
        """
        # Find the test command from the dataset item
        test_code = item.get("test", item.get("test_code", item.get("tests", "")))

        if test_code:
            # Run the test in the model's sandbox
            test_result = ctx.terminal(
                f'cd /workspace && python3 -c "{test_code}"', timeout=60
            )

            if test_result["exit_code"] == 0:
                self.reward_buffer.append(1.0)
                return 1.0

        # Partial credit: check if the model created any Python files
        file_check = ctx.terminal("find /workspace -name '*.py' -newer /tmp/.start_marker 2>/dev/null | head -5")
        if file_check["exit_code"] == 0 and file_check.get("output", "").strip():
            self.reward_buffer.append(0.1)
            return 0.1

        self.reward_buffer.append(0.0)
        return 0.0

    async def evaluate(self, *args, **kwargs):
        """
        Run evaluation on a held-out set.

        Override for dataset-specific evaluation logic.
        """
        start_time = time.time()
        end_time = time.time()

        eval_metrics = {"eval/placeholder": 0.0}
        await self.evaluate_log(
            metrics=eval_metrics,
            start_time=start_time,
            end_time=end_time,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log SWE-specific metrics."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
            wandb_metrics["train/pass_rate"] = sum(
                1 for r in self.reward_buffer if r == 1.0
            ) / len(self.reward_buffer)
            self.reward_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    HermesSweEnv.cli()
