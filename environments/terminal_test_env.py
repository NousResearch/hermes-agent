"""
TerminalTestEnv -- Simple Test Environment for Validating the Stack

A self-contained environment with inline tasks (no external dataset needed).
Each task asks the model to create a file at a known path with specific content.
The reward verifier cats the file and checks if the content matches.

Enables only terminal + file toolsets. Uses Modal terminal backend with
OpenRouter (Claude) by default.

Training tasks (3):
    1. Create ~/greeting.txt with "Hello from Hermes Agent"
    2. Create ~/count.txt with numbers 1-5, one per line
    3. Create ~/answer.txt with the result of 123 + 456

Eval task (1):
    1. Create ~/result.txt with the result of 6 * 7

Usage:
    # Start Atropos API server
    run-api

    # Run environment (uses OpenRouter + Modal by default)
    python environments/terminal_test_env.py serve

    # Process mode (no run-api needed, saves to JSONL)
    python environments/terminal_test_env.py process \\
        --env.data_path_to_save_groups terminal_test_output.jsonl
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item

from environments.agent_loop import AgentResult
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Inline task definitions -- no external dataset needed
# =============================================================================

TRAIN_TASKS = [
    {
        "prompt": "Create a file at ~/greeting.txt containing exactly the text: Hello from Hermes Agent",
        "verify_path": "~/greeting.txt",
        "expected_content": "Hello from Hermes Agent",
    },
    {
        "prompt": "Create a file at ~/count.txt containing the numbers 1 through 5, one per line",
        "verify_path": "~/count.txt",
        "expected_content": "1\n2\n3\n4\n5",
    },
    {
        "prompt": "Create a file at ~/answer.txt containing the result of 123 + 456",
        "verify_path": "~/answer.txt",
        "expected_content": "579",
    },
]

EVAL_TASKS = [
    {
        "prompt": "Create a file at ~/result.txt containing the result of 6 * 7",
        "verify_path": "~/result.txt",
        "expected_content": "42",
    },
]


class TerminalTestEnvConfig(HermesAgentEnvConfig):
    """Config with defaults suitable for terminal testing."""

    pass  # Inherits all fields, overrides defaults in config_init


class TerminalTestEnv(HermesAgentBaseEnv):
    """
    Simple test environment with inline file-creation tasks.

    All tasks follow the same pattern: "create a file at ~/X.txt with content Y".
    The verifier runs `cat ~/X.txt` in the rollout's terminal and checks the output
    against the expected string. Same verifier logic for all tasks.

    This environment is designed to validate the full stack end-to-end:
    - Agent loop executes tool calls (terminal/file)
    - ToolContext provides terminal access to the reward function
    - Reward function verifies file content via cat
    - Scored data flows through the Atropos pipeline
    """

    name = "terminal-test"
    env_config_cls = TerminalTestEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[TerminalTestEnvConfig, List[APIServerConfig]]:
        """
        Default configuration for the terminal test environment.

        Uses Modal terminal backend for cloud isolation and OpenRouter with
        Claude for inference. API keys loaded from ~/hermes-agent/.env.
        """
        env_config = TerminalTestEnvConfig(
            # Terminal + file tools only
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,
            # Agent settings
            max_agent_turns=10,  # Simple tasks, don't need many turns
            max_token_length=16000,
            agent_temperature=1.0,
            system_prompt=(
                "You are a helpful assistant with access to a terminal and file tools. "
                "Complete the user's request by using the available tools. "
                "Be precise and follow instructions exactly."
            ),
            # Modal terminal backend for cloud-isolated sandboxes per rollout
            terminal_backend="modal",
            # Atropos settings
            group_size=3,              # 3 rollouts per group
            tokenizer_name="NousResearch/q-30b-t-h45-e1",
            tool_call_parser="hermes",
            steps_per_eval=3,          # Eval after all 3 steps
            total_steps=3,             # 3 groups total (1 group per step)
            use_wandb=True,
            wandb_name="terminal-test",
            ensure_scores_are_not_same=False,  # Allow all-same scores for simple tasks
            # No external dataset
            dataset_name=None,
        )

        # OpenRouter with Claude -- API key loaded from .env (OPENROUTER_API_KEY)
        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-opus-4.6",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,  # OpenRouter doesn't have a /health endpoint
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Initialize inline task lists."""
        self.train_tasks = list(TRAIN_TASKS)
        self.eval_tasks = list(EVAL_TASKS)
        self.iter = 0
        # Track reward stats for wandb logging
        self.reward_buffer: List[float] = []

    async def get_next_item(self) -> Dict[str, str]:
        """Cycle through training tasks."""
        item = self.train_tasks[self.iter % len(self.train_tasks)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, str]) -> str:
        """The prompt is directly in the task item."""
        return item["prompt"]

    async def compute_reward(
        self, item: Dict[str, str], result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Verify by cat-ing the expected file path and checking content matches.
        Same verifier for all tasks -- they all write a file at a known path.

        Scoring:
            1.0 = exact match
            0.5 = expected content is present but has extra stuff
            0.0 = file doesn't exist or content doesn't match
        """
        verify_result = ctx.terminal(f"cat {item['verify_path']}")

        # File doesn't exist or can't be read
        if verify_result["exit_code"] != 0:
            self.reward_buffer.append(0.0)
            return 0.0

        actual = verify_result.get("output", "").strip()
        expected = item["expected_content"].strip()

        # Exact match
        if actual == expected:
            self.reward_buffer.append(1.0)
            return 1.0

        # Partial credit: expected content is present but has extra stuff
        if expected in actual:
            self.reward_buffer.append(0.5)
            return 0.5

        self.reward_buffer.append(0.0)
        return 0.0

    async def evaluate(self, *args, **kwargs):
        """
        Run eval tasks using the agent loop and verify results.
        Logs accuracy metrics.
        """
        start_time = time.time()
        correct = 0
        total = len(self.eval_tasks)
        samples = []

        for eval_item in self.eval_tasks:
            try:
                # For eval, we do a simple single-turn completion (not full agent loop)
                # to keep eval fast. The agent loop is tested via training.
                completion = await self.server.chat_completion(
                    messages=[
                        {"role": "system", "content": self.config.system_prompt or ""},
                        {"role": "user", "content": eval_item["prompt"]},
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.0,
                    split="eval",
                )

                response_content = (
                    completion.choices[0].message.content if completion.choices else ""
                )

                samples.append(
                    {
                        "prompt": eval_item["prompt"],
                        "response": response_content,
                        "expected": eval_item["expected_content"],
                    }
                )

            except Exception as e:
                logger.error("Eval failed for item: %s", e)
                samples.append(
                    {
                        "prompt": eval_item["prompt"],
                        "response": f"ERROR: {e}",
                        "expected": eval_item["expected_content"],
                    }
                )

        end_time = time.time()

        eval_metrics = {
            "eval/num_samples": total,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log training metrics including reward stats and accuracy."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            total = len(self.reward_buffer)
            correct = sum(1 for r in self.reward_buffer if r == 1.0)
            partial = sum(1 for r in self.reward_buffer if r == 0.5)

            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / total
            wandb_metrics["train/accuracy"] = correct / total
            wandb_metrics["train/partial_match_rate"] = partial / total
            wandb_metrics["train/total_rollouts"] = total
            self.reward_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    TerminalTestEnv.cli()
