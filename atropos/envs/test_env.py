"""
Simple test environment for validating the atropos-agent setup.

This environment uses a local OpenAI-compatible server for LLM testing to verify:
- BaseEnv extension works correctly
- API communication via OpenAI-compatible endpoint
- Basic trajectory collection

This is a minimal environment for testing, not production use.
"""

import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    Item,
)

from ..agent import AgentConfig
from .agent_env import AgentEnv, AgentEnvConfig

# Load environment variables from .env file
load_dotenv()


# Simple test prompts for validation
TEST_PROMPTS = [
    {
        "prompt": "What is 2 + 2? Answer with just the number.",
        "expected": "4",
    },
    {
        "prompt": "What is the capital of France? Answer with just the city name.",
        "expected": "Paris",
    },
    {
        "prompt": "What color is the sky on a clear day? Answer with just the color.",
        "expected": "Blue",
    },
    {
        "prompt": "How many days are in a week? Answer with just the number.",
        "expected": "7",
    },
    {
        "prompt": "What is 10 * 5? Answer with just the number.",
        "expected": "50",
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions concisely and directly. "
    "When asked for a simple answer, provide just that answer without explanation."
)


class SimpleTestEnvConfig(AgentEnvConfig):
    """Configuration for the simple test environment."""

    server_base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL for an OpenAI-compatible server (without /v1)",
    )
    server_model: str = Field(
        default="hermes-4-36b",
        description="Model name",
    )


class SimpleTestEnv(AgentEnv[SimpleTestEnvConfig]):
    """
    A simple test environment to validate the atropos-agent setup.
    
    Uses a local OpenAI-compatible LLM endpoint with basic question-answering tasks.
    Scoring is based on whether the response contains the expected answer.
    """

    name = "simple_test_env"
    env_config_cls = SimpleTestEnvConfig

    def __init__(
        self,
        config: SimpleTestEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.iter = 0
        self.test_prompts = TEST_PROMPTS
        self.percent_correct_buffer: List[float] = []

    @classmethod
    def config_init(cls) -> Tuple[SimpleTestEnvConfig, List[APIServerConfig]]:
        """
        Initialize configuration with local server settings from environment variables.
        """
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://127.0.0.1:8080"
        )
        model = os.getenv("ATROPOS_SERVER_MODEL") or os.getenv("LLM_MODEL") or "hermes-4-36b"
        api_key = os.getenv("ATROPOS_SERVER_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

        env_config = SimpleTestEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",  # For tokenization only
            group_size=4,
            use_wandb=False,  # Disable wandb for simple testing
            rollout_server_url="http://localhost:8000",
            total_steps=10,
            batch_size=16,
            steps_per_eval=5,
            max_token_length=2048,
            inference_weight=1.0,
            wandb_name="simple_test",
            server_base_url=base_url,
            server_model=model,
        )

        # OpenAI-compatible servers typically expose chat completions at /v1.
        server_configs = [
            APIServerConfig(
                model_name=model,
                base_url=f"{base_url}/v1",
                api_key=api_key,
                num_max_requests_at_once=4,
                num_requests_for_eval=8,
                timeout=120,  # Local models may be slower
            ),
        ]

        return env_config, server_configs

    async def setup_agent_env(self):
        """Setup the environment - load test data."""
        print(f"SimpleTestEnv setup complete. {len(self.test_prompts)} test prompts loaded.")
        print(f"Using server at: {self.config.server_base_url}")
        print(f"Model: {self.config.server_model}")

    async def get_next_item(self) -> Item:
        """Get the next test prompt."""
        item = self.test_prompts[self.iter % len(self.test_prompts)]
        self.iter += 1
        return item

    def build_task(self, item: Item) -> str:
        return item["prompt"]

    def build_agent_config(self, item: Item) -> AgentConfig:  # noqa: ARG002
        return AgentConfig(
            max_steps=5,
            temperature=0.7,
            max_tokens=256,
            system_prompt=SYSTEM_PROMPT,
        )

    async def score_trajectory(self, item: Item, final_response: str) -> float:
        expected = item["expected"].lower()
        response_lower = (final_response or "").lower()
        score = 1.0 if expected in response_lower else 0.0
        self.percent_correct_buffer.append(score)
        return score

    async def evaluate(self, *args, **kwargs):
        """
        Simple evaluation - run through all test prompts once.
        """
        correct = 0
        total = len(self.test_prompts)

        for item in self.test_prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]

            response = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=256,
                temperature=0.0,  # Greedy for eval
                split="eval",
            )

            response_text = response.choices[0].message.content or ""
            expected = item["expected"].lower()

            if expected in response_text.lower():
                correct += 1

        accuracy = correct / total
        print(f"Evaluation: {correct}/{total} = {accuracy:.2%} accuracy")
        return {"eval_accuracy": accuracy}

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics (simplified for testing)."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            avg_correct = sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            wandb_metrics["train/percent_correct"] = avg_correct
            print(f"Train accuracy: {avg_correct:.2%}")
            self.percent_correct_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    # Allow running as CLI
    SimpleTestEnv.cli()
