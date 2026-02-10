"""
GSM8kAgentEnv -- Math Reasoning with Tool Use (Python REPL)

An agentic RL environment where models solve GSM8k math problems using
a Python interpreter tool. Uses proper OpenAI-spec tool calling via
HermesAgentBaseEnv (not ICL).

The model:
1. Receives a math problem
2. Can call the `terminal` tool to run Python code (`python3 -c "..."`)
3. Provides a final answer in \\boxed{} format
4. Gets reward: 1.0 if correct, 0.0 if wrong

Usage:
    # Phase 1 (OpenRouter, no training):
    python environments/gsm8k_agent_env.py process \\
        --env.data_path_to_save_groups gsm8k_agent_output.jsonl

    # Phase 2 (VLLM + Tinker training):
    run-api
    python launch_training.py --config configs/gsm8k_agent.yaml
    python environments/gsm8k_agent_env.py serve
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path
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
# Math verification helpers
# =============================================================================

def _verify_math_answer(model_response: str, gold_answer: str) -> bool:
    """
    Verify if the model's response contains the correct answer.
    Uses math_verify for robust LaTeX comparison, falls back to string matching.
    """
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        gold_parsed = parse(
            f"\\boxed{{{gold_answer}}}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        # Strip <think> blocks if present
        answer_text = model_response
        if "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[-1]

        answer_parsed = parse(
            answer_text,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        return bool(verify(answer_parsed, gold_parsed))

    except ImportError:
        # Fallback: simple string matching for \\boxed{answer}
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, model_response)
        if matches:
            model_answer = matches[-1].strip().replace(",", "")
            gold_clean = gold_answer.strip().replace(",", "")
            return model_answer == gold_clean
        return False


# =============================================================================
# Environment Config
# =============================================================================

class GSM8kAgentEnvConfig(HermesAgentEnvConfig):
    """Config with defaults for GSM8k agent environment."""
    pass


# =============================================================================
# Environment
# =============================================================================

class GSM8kAgentEnv(HermesAgentBaseEnv):
    """
    GSM8k math environment with Python REPL tool calling.

    Models solve grade-school math problems by reasoning step by step
    and using Python (via the terminal tool) for calculations.

    Exercises the full agentic RL training loop:
    - Model receives math problem
    - Makes tool calls to compute (python3 -c "...")
    - Provides final answer in \\boxed{}
    - Reward: binary (1.0 correct, 0.0 wrong)
    """

    name = "gsm8k-agent"
    env_config_cls = GSM8kAgentEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[GSM8kAgentEnvConfig, List[APIServerConfig]]:
        """
        Default config using terminal tool.

        Reads from environment variables (set in .env):
            ATROPOS_SERVER_BASE_URL  - Inference server URL
            ATROPOS_SERVER_MODEL     - Model name on the server
            ATROPOS_TOKENIZER_NAME   - HuggingFace tokenizer name
            ATROPOS_SERVER_API_KEY   - API key for the server
        """
        # Resolve inference server settings from env
        base_url = (
            os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        model = (
            os.getenv("ATROPOS_SERVER_MODEL")
            or os.getenv("LLM_MODEL")
            or "Hermes-4.3-36B"
        )

        api_key = (
            os.getenv("ATROPOS_SERVER_API_KEY")
            or os.getenv("NOUS_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

        tokenizer = (
            os.getenv("ATROPOS_TOKENIZER_NAME")
            or os.getenv("ATROPOS_TOKENIZER")
            or "NousResearch/Hermes-4.3-36B"
        )

        env_config = GSM8kAgentEnvConfig(
            # Terminal + file toolsets (same as terminal_test_env.py)
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,
            # Agent settings
            max_agent_turns=5,          # Math problems don't need many turns
            max_token_length=2048,      # Room for reasoning + code
            agent_temperature=1.0,
            system_prompt=(
                "You are a helpful math assistant. You have access to a terminal "
                "where you can run Python code to help solve problems.\n\n"
                "When you need to calculate something, use the terminal tool with "
                "a command like: python3 -c \"print(2 + 2)\"\n\n"
                "When you have the final answer, write it inside \\boxed{} like: \\boxed{42}\n\n"
                "Work step by step. Use Python to verify your reasoning."
            ),
            # Terminal backend (local for testing, modal for production)
            terminal_backend=os.getenv("TERMINAL_ENV", "local"),
            # Parser -- hermes format for Hermes models
            tool_call_parser="hermes",
            # Atropos settings
            group_size=4,
            tokenizer_name=tokenizer,
            steps_per_eval=5,
            total_steps=10,
            use_wandb=bool(os.getenv("WANDB_API_KEY")),
            wandb_name="gsm8k-agent",
            ensure_scores_are_not_same=False,
            # No external dataset (we load GSM8k ourselves)
            dataset_name=None,
        )

        server_configs = [
            APIServerConfig(
                base_url=base_url,
                model_name=model,
                server_type="openai",
                api_key=api_key,
                health_check=False,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Load GSM8k dataset."""
        from datasets import load_dataset

        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = [
            {
                "question": item["question"],
                "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
            }
            for item in test_data
        ]
        self.iter = 0
        self.reward_buffer: List[float] = []
        self.tool_use_buffer: List[int] = []
        print(f"[GSM8kAgentEnv] Loaded {len(self.train)} train, {len(self.test)} test examples")

    async def get_next_item(self) -> Dict[str, str]:
        """Cycle through training problems."""
        item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return {
            "question": item["question"],
            "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
        }

    def format_prompt(self, item: Dict[str, str]) -> str:
        """Format the math problem as a user message."""
        return item["question"]

    async def compute_reward(
        self, item: Dict[str, str], result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Score: verify the model's \\boxed{} answer against the gold answer.

        The agent has full access to terminal via ctx, but for GSM8k we just
        check the final answer from the conversation.
        """
        # Get the last assistant message content
        final_text = ""
        for msg in reversed(result.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"]
                break

        correct = _verify_math_answer(final_text, item["gold_answer"])
        reward = 1.0 if correct else 0.0

        self.reward_buffer.append(reward)
        # Count tool calls in this trajectory
        tool_call_count = sum(
            len(msg.get("tool_calls", []))
            for msg in result.messages
            if msg.get("role") == "assistant"
        )
        self.tool_use_buffer.append(tool_call_count)

        return reward

    async def evaluate(self, *args, **kwargs):
        """Evaluate on a subset of the test set (greedy, no tools for speed)."""
        start_time = time.time()
        correct = 0
        total = 0
        samples = []

        eval_subset = self.test[:30]  # Small subset for quick eval

        for item in eval_subset:
            try:
                completion = await self.server.chat_completion(
                    messages=[
                        {"role": "system", "content": self.config.system_prompt or ""},
                        {"role": "user", "content": item["question"]},
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.0,
                    split="eval",
                )

                response = completion.choices[0].message.content or ""
                is_correct = _verify_math_answer(response, item["gold_answer"])

                if is_correct:
                    correct += 1
                total += 1

                samples.append({
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "response": response[:500],
                    "correct": is_correct,
                })

            except Exception as e:
                logger.error("Eval failed: %s", e)
                total += 1

        percent_correct = correct / total if total > 0 else 0
        end_time = time.time()

        await self.evaluate_log(
            metrics={"eval/percent_correct": percent_correct, "eval/total": total},
            samples=samples,
            start_time=start_time,
            end_time=end_time,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log training metrics."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            wandb_metrics["train/percent_correct"] = sum(self.reward_buffer) / len(self.reward_buffer)
            wandb_metrics["train/total_rollouts"] = len(self.reward_buffer)
            self.reward_buffer = []

        if self.tool_use_buffer:
            wandb_metrics["train/avg_tool_calls"] = sum(self.tool_use_buffer) / len(self.tool_use_buffer)
            wandb_metrics["train/tool_use_rate"] = sum(1 for t in self.tool_use_buffer if t > 0) / len(self.tool_use_buffer)
            self.tool_use_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    GSM8kAgentEnv.cli()
