"""
OuroborosEnv -- RL Environment with Council-Based Reward

Uses the adversarial council to evaluate agent research output, producing
multi-signal rewards and DPO preference pairs for training.

The agent is given research questions and must use web search, file tools,
and terminal to produce comprehensive, evidence-backed analysis. The council
evaluates the output quality and generates rewards.

Usage:
    # Process mode (SFT data gen, no run-api needed)
    python environments/ouroboros_env.py process \\
        --env.data_path_to_save_groups ouroboros_output.jsonl

    # Serve mode (with Atropos API server)
    python environments/ouroboros_env.py serve \\
        --openai.base_url http://localhost:8000/v1
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item
from pydantic import Field

from environments.agent_loop import AgentResult
from environments.council_evaluator import CouncilEvaluator
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Research questions dataset (inline, no external dependency)
# =============================================================================

RESEARCH_QUESTIONS = [
    # Technology & Computing
    {
        "question": "What are the security implications of post-quantum cryptography migration for existing blockchain networks?",
        "category": "technology",
    },
    {
        "question": "How do different consensus mechanisms (PoW, PoS, DPoS, DAG) compare in terms of energy efficiency, decentralization, and security?",
        "category": "technology",
    },
    {
        "question": "What is the current state of homomorphic encryption and when will it be practical for real-world applications?",
        "category": "technology",
    },
    {
        "question": "How do zero-knowledge proofs enable privacy while maintaining auditability in financial systems?",
        "category": "technology",
    },
    {
        "question": "What are the trade-offs between monolithic and modular blockchain architectures?",
        "category": "technology",
    },
    # Economics & Finance
    {
        "question": "What would be the economic impact of a central bank digital currency (CBDC) on commercial banking?",
        "category": "economics",
    },
    {
        "question": "How do automated market makers (AMMs) compare to traditional order book exchanges in terms of capital efficiency?",
        "category": "economics",
    },
    {
        "question": "What historical evidence exists for or against the effectiveness of universal basic income?",
        "category": "economics",
    },
    {
        "question": "How do mechanism design principles apply to token economics and governance?",
        "category": "economics",
    },
    {
        "question": "What are the systemic risks of algorithmic stablecoins based on historical examples?",
        "category": "economics",
    },
    # Science & Research
    {
        "question": "What is the current evidence for and against the lab leak hypothesis for COVID-19?",
        "category": "science",
    },
    {
        "question": "How close are we to achieving artificial general intelligence, and what are the key remaining challenges?",
        "category": "science",
    },
    {
        "question": "What are the most promising approaches to nuclear fusion and their realistic timelines?",
        "category": "science",
    },
    {
        "question": "How effective are different geoengineering approaches for climate change mitigation?",
        "category": "science",
    },
    {
        "question": "What is the replication crisis in social psychology and how should it change our confidence in published findings?",
        "category": "science",
    },
    # Governance & Society
    {
        "question": "How do different models of internet governance (multi-stakeholder, state-led, corporate) affect innovation and freedom?",
        "category": "governance",
    },
    {
        "question": "What are the arguments for and against intellectual property rights in the age of AI-generated content?",
        "category": "governance",
    },
    {
        "question": "How effective are economic sanctions as a foreign policy tool, based on historical evidence?",
        "category": "governance",
    },
    {
        "question": "What are the implications of autonomous weapons systems for international law and ethics?",
        "category": "governance",
    },
    {
        "question": "How should AI systems be regulated to balance innovation with safety?",
        "category": "governance",
    },
    # Strategy & Decision-Making
    {
        "question": "When is it rational to use prediction markets vs. expert committees for forecasting?",
        "category": "strategy",
    },
    {
        "question": "What does the evidence say about the effectiveness of diversification vs. concentration in investment portfolios?",
        "category": "strategy",
    },
    {
        "question": "How should organizations balance exploration vs. exploitation in R&D allocation?",
        "category": "strategy",
    },
    {
        "question": "What are the failure modes of democratic governance in decentralized autonomous organizations?",
        "category": "strategy",
    },
    {
        "question": "How do different approaches to risk management (quantitative vs. scenario-based) perform under tail risk events?",
        "category": "strategy",
    },
]

EVAL_QUESTIONS = [
    {
        "question": "Should proof-of-stake replace proof-of-work for all blockchain networks? Analyze the trade-offs comprehensively.",
        "category": "technology",
    },
    {
        "question": "What is the optimal approach to AI alignment and what are the strongest arguments against current safety research?",
        "category": "science",
    },
    {
        "question": "Analyze the case for and against cryptocurrency as legal tender, using El Salvador as a case study.",
        "category": "economics",
    },
]


# =============================================================================
# Environment Config
# =============================================================================


class OuroborosEnvConfig(HermesAgentEnvConfig):
    """Configuration for the Ouroboros council-reward environment."""

    council_model: str = Field(
        default="nousresearch/hermes-3-llama-3.1-70b",
        description="Model for council evaluation. Can differ from the agent model.",
    )
    min_confidence_threshold: float = Field(
        default=0.3,
        description="Minimum council confidence for positive reward (0.0-1.0).",
    )
    dpo_output_dir: str = Field(
        default="data/ouroboros_dpo",
        description="Directory to save DPO preference pairs.",
    )


# =============================================================================
# Environment
# =============================================================================


class OuroborosEnv(HermesAgentBaseEnv):
    """RL environment that uses the adversarial council for reward.

    The agent researches questions using web search, file tools, and terminal.
    The council evaluates output quality across accuracy, depth, evidence,
    and falsifiability. DPO pairs are extracted from council disagreements.
    """

    name = "ouroboros"
    env_config_cls = OuroborosEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[OuroborosEnvConfig, List[APIServerConfig]]:
        """Default configuration for the ouroboros environment."""
        env_config = OuroborosEnvConfig(
            # Enable web + file + terminal for research
            enabled_toolsets=["web", "file", "terminal", "council"],
            disabled_toolsets=None,
            distribution=None,
            # Agent settings
            max_agent_turns=20,
            max_token_length=16000,
            agent_temperature=1.0,
            system_prompt=(
                "You are a research agent. Your task is to thoroughly research "
                "the given question using web search and other available tools. "
                "Provide a comprehensive, evidence-backed analysis with specific "
                "data points, sources, and counter-arguments. "
                "Be thorough but concise. Cite your sources."
            ),
            # Terminal backend
            terminal_backend="local",
            terminal_timeout=120,
            # Council evaluation
            council_model="nousresearch/hermes-3-llama-3.1-70b",
            min_confidence_threshold=0.3,
            # Atropos settings
            group_size=2,
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            steps_per_eval=10,
            total_steps=50,
            use_wandb=True,
            wandb_name="ouroboros",
            ensure_scores_are_not_same=False,
            dataset_name=None,
        )

        # Default to OpenRouter
        server_configs = [
            APIServerConfig(
                base_url=os.getenv(
                    "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                model_name=os.getenv(
                    "OPENAI_MODEL", "nousresearch/hermes-3-llama-3.1-70b"
                ),
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                health_check=False,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Initialize research questions and council evaluator."""
        self.questions = list(RESEARCH_QUESTIONS)
        self.eval_questions = list(EVAL_QUESTIONS)
        self.iter = 0
        self.evaluator = CouncilEvaluator(model=self.config.council_model)
        self.dpo_buffer: List[Dict] = []
        self.reward_buffer: List[float] = []

        # Create DPO output directory
        dpo_dir = Path(self.config.dpo_output_dir)
        dpo_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Ouroboros env initialized: %d train questions, %d eval questions",
            len(self.questions),
            len(self.eval_questions),
        )

    async def get_next_item(self) -> Dict[str, str]:
        """Cycle through research questions."""
        item = self.questions[self.iter % len(self.questions)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, str]) -> str:
        """Format the research question as a prompt."""
        return (
            f"Research this question thoroughly using web search and analysis:\n\n"
            f"{item['question']}\n\n"
            f"Provide a comprehensive, evidence-backed analysis. Include specific "
            f"data points, historical examples, and counter-arguments. Cite sources."
        )

    async def compute_reward(
        self, item: Dict[str, str], result: AgentResult, ctx: ToolContext
    ) -> float:
        """Score the rollout using the council evaluator.

        Multi-signal reward:
          - 60% council confidence score
          - 20% tool usage (did the agent actually use tools?)
          - 20% source citation (did the response include evidence?)
        """
        # Extract agent's final response from messages
        agent_output = self._extract_final_response(result.messages)

        if not agent_output or len(agent_output) < 50:
            self.reward_buffer.append(0.0)
            return 0.0

        try:
            # Council evaluates the research
            verdict = await self.evaluator.evaluate(
                content=agent_output,
                question=item["question"],
                criteria=["accuracy", "depth", "evidence", "falsifiability"],
            )

            # Store DPO pairs
            dpo_pairs = self.evaluator.extract_dpo_pairs(verdict)
            self.dpo_buffer.extend(dpo_pairs)

            # Multi-signal reward
            council_score = verdict.confidence_score / 100.0
            tool_usage = min(1.0, result.turns_used / 5)  # reward using tools
            has_sources = 1.0 if verdict.sources else 0.0

            reward = 0.6 * council_score + 0.2 * tool_usage + 0.2 * has_sources
            reward = max(0.0, min(1.0, reward))

        except Exception as e:
            logger.error("Council evaluation failed: %s", e)
            # Fallback: simple length-based reward
            reward = min(1.0, len(agent_output) / 2000) * 0.3

        self.reward_buffer.append(reward)
        return reward

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on held-out questions and log metrics."""
        start_time = time.time()
        samples = []

        for eval_item in self.eval_questions[:3]:
            try:
                completion = await self.server.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": self.config.system_prompt or "",
                        },
                        {"role": "user", "content": self.format_prompt(eval_item)},
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.0,
                    split="eval",
                )

                response = (
                    completion.choices[0].message.content
                    if completion.choices
                    else ""
                )
                samples.append({
                    "prompt": eval_item["question"],
                    "response": response[:500],
                    "category": eval_item.get("category", ""),
                })
            except Exception as e:
                logger.error("Eval failed: %s", e)
                samples.append({
                    "prompt": eval_item["question"],
                    "response": f"ERROR: {e}",
                    "category": eval_item.get("category", ""),
                })

        end_time = time.time()

        eval_metrics = {
            "eval/num_samples": len(samples),
            "eval/dpo_pairs_total": len(self.dpo_buffer),
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
        )

        # Save accumulated DPO pairs
        self._save_dpo_pairs()

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log training metrics including council scores."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            total = len(self.reward_buffer)
            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / total
            wandb_metrics["train/total_rollouts"] = total
            wandb_metrics["train/dpo_pairs_buffered"] = len(self.dpo_buffer)
            self.reward_buffer = []

        await super().wandb_log(wandb_metrics)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _extract_final_response(messages: List[Dict[str, Any]]) -> str:
        """Extract the agent's final text response from conversation messages."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                if len(content) > 50:  # skip short tool-call-only turns
                    return content
        # Fallback: concatenate all assistant messages
        parts = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                parts.append(msg["content"])
        return "\n".join(parts)

    def _save_dpo_pairs(self):
        """Save accumulated DPO pairs to disk."""
        if not self.dpo_buffer:
            return

        output_path = Path(self.config.dpo_output_dir) / "dpo_pairs.jsonl"
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                for pair in self.dpo_buffer:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            logger.info(
                "Saved %d DPO pairs to %s", len(self.dpo_buffer), output_path
            )
            self.dpo_buffer = []
        except Exception as e:
            logger.error("Failed to save DPO pairs: %s", e)


if __name__ == "__main__":
    OuroborosEnv.cli()
