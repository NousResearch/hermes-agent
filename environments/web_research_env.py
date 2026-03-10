"""
WebResearchEnv — RL Environment for Multi-Step Web Research
============================================================

Trains models to do accurate, efficient, multi-source web research.

Reward signals:
  - Answer correctness  (LLM judge, 0.0–1.0)
  - Source diversity    (used ≥2 distinct domains)
  - Efficiency          (penalizes excessive tool calls)
  - Tool usage          (bonus for actually using web tools)

Dataset: FRAMES benchmark (Google, 2024) — multi-hop factual questions
  HuggingFace: google/frames-benchmark
  Fallback:    built-in sample questions (no HF token needed)

Usage:
    # Phase 1 (OpenAI-compatible server)
    python environments/web_research_env.py serve \\
        --openai.base_url http://localhost:8000/v1 \\
        --openai.model_name YourModel \\
        --openai.server_type openai

    # Process mode (offline data generation)
    python environments/web_research_env.py process \\
        --env.data_path_to_save_groups data/web_research.jsonl

    # Standalone eval
    python environments/web_research_env.py evaluate \\
        --openai.base_url http://localhost:8000/v1 \\
        --openai.model_name YourModel

Built by: github.com/jackx707
Inspired by: GroceryMind — production Hermes agent doing live web research
             across German grocery stores (firecrawl + hermes-agent)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import Field

# Ensure hermes-agent root is on path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ---------------------------------------------------------------------------
# Optional HuggingFace datasets import
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item

from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.agent_loop import AgentResult
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback sample dataset (used when HuggingFace is unavailable)
# Multi-hop questions requiring real web search to answer.
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    {
        "question": "What is the current population of the capital city of the country that won the 2022 FIFA World Cup?",
        "answer": "Buenos Aires has approximately 3 million people in the city proper, or around 15 million in the greater metro area.",
        "difficulty": "medium",
        "hops": 2,
    },
    {
        "question": "Who is the CEO of the company that makes the most widely used open-source container orchestration platform?",
        "answer": "The Linux Foundation oversees Kubernetes. CNCF (Cloud Native Computing Foundation) is the specific body — it does not have a traditional CEO but has an executive director.",
        "difficulty": "medium",
        "hops": 2,
    },
    {
        "question": "What programming language was used to write the original version of the web framework used by Instagram?",
        "answer": "Django, which Instagram was built on, is written in Python.",
        "difficulty": "easy",
        "hops": 2,
    },
    {
        "question": "In what year was the university founded where the inventor of the World Wide Web currently holds a professorship?",
        "answer": "Tim Berners-Lee holds a professorship at MIT (founded 1861) and the University of Southampton (founded 1952).",
        "difficulty": "hard",
        "hops": 3,
    },
    {
        "question": "What is the latest stable version of the programming language that ranks #1 on the TIOBE index as of this year?",
        "answer": "Python is currently #1 on TIOBE. The latest stable version should be verified via the official python.org site.",
        "difficulty": "medium",
        "hops": 2,
    },
    {
        "question": "How many employees does the parent company of Instagram have?",
        "answer": "Meta Platforms (parent of Instagram) employs approximately 70,000+ people as of recent reports.",
        "difficulty": "medium",
        "hops": 2,
    },
    {
        "question": "What is the current interest rate set by the central bank of the country where the Eiffel Tower is located?",
        "answer": "The European Central Bank sets rates for France/eurozone. The current rate should be verified — it has changed frequently in 2023-2025.",
        "difficulty": "hard",
        "hops": 2,
    },
    {
        "question": "Which company acquired the startup founded by the creator of Oculus VR?",
        "answer": "Palmer Luckey founded Oculus VR, which was acquired by Facebook (now Meta). He later founded Anduril Industries.",
        "difficulty": "medium",
        "hops": 2,
    },
    {
        "question": "What is the market cap of the company that owns the most popular search engine in Russia?",
        "answer": "Yandex (now split into separate entities after 2024 restructuring). Current market cap should be verified via financial sources.",
        "difficulty": "hard",
        "hops": 2,
    },
    {
        "question": "What was the GDP growth rate of the country that hosted the most recent Summer Olympics?",
        "answer": "Paris, France hosted the 2024 Summer Olympics. France's recent GDP growth should be verified via World Bank or IMF data.",
        "difficulty": "hard",
        "hops": 2,
    },
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class WebResearchEnvConfig(HermesAgentEnvConfig):
    """Configuration for the web research RL environment."""

    # Reward weights
    correctness_weight: float = Field(
        default=0.6,
        description="Weight for answer correctness in reward (LLM judge score).",
    )
    tool_usage_weight: float = Field(
        default=0.2,
        description="Weight for tool usage signal (did the model actually use web tools?).",
    )
    efficiency_weight: float = Field(
        default=0.2,
        description="Weight for efficiency signal (penalizes excessive tool calls).",
    )
    diversity_bonus: float = Field(
        default=0.1,
        description="Bonus reward for citing ≥2 distinct domains.",
    )

    # Efficiency thresholds
    efficient_max_calls: int = Field(
        default=5,
        description="Maximum tool calls before efficiency penalty begins.",
    )
    heavy_penalty_calls: int = Field(
        default=10,
        description="Tool call count where efficiency penalty steepens.",
    )

    # Eval
    eval_size: int = Field(
        default=20,
        description="Number of held-out items for evaluation.",
    )
    eval_split_ratio: float = Field(
        default=0.1,
        description="Fraction of dataset to hold out for evaluation (0.0–1.0).",
    )

    # Dataset
    dataset_name: str = Field(
        default="google/frames-benchmark",
        description="HuggingFace dataset name for research questions.",
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WebResearchEnv(HermesAgentBaseEnv):
    """
    RL environment for training multi-step web research skills.

    The model is given a factual question requiring 2-3 hops of web research
    and must use web_search / web_extract tools to find and synthesize the answer.

    Reward is multi-signal:
      60% — answer correctness (LLM judge)
      20% — tool usage (did the model actually search the web?)
      20% — efficiency (penalizes >5 tool calls)

    Bonus +0.1 for source diversity (≥2 distinct domains cited).
    """

    name = "web-research"
    env_config_cls = WebResearchEnvConfig

    # Default toolsets for this environment — web + file for saving notes
    default_toolsets = ["web", "file"]

    @classmethod
    def config_init(cls) -> Tuple[WebResearchEnvConfig, List[APIServerConfig]]:
        """Default configuration for the web research environment."""
        env_config = WebResearchEnvConfig(
            enabled_toolsets=["web", "file"],
            max_agent_turns=15,
            agent_temperature=1.0,
            system_prompt=(
                "You are a highly capable research agent. When asked a factual question, "
                "always use web_search to find current, accurate information before answering. "
                "Cite at least 2 sources. Be concise and accurate."
            ),
            group_size=4,
            total_steps=1000,
            steps_per_eval=100,
            use_wandb=True,
            wandb_name="web-research",
        )

        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4.5",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._items: list[dict] = []
        self._eval_items: list[dict] = []
        self._index: int = 0

        # Metrics tracking for wandb
        self._reward_buffer: list[float] = []
        self._correctness_buffer: list[float] = []
        self._tool_usage_buffer: list[float] = []
        self._efficiency_buffer: list[float] = []
        self._diversity_buffer: list[float] = []

    # ------------------------------------------------------------------
    # 1. Setup — load dataset
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Load the FRAMES benchmark or fall back to built-in samples."""
        if HF_AVAILABLE:
            try:
                logger.info("Loading FRAMES benchmark from HuggingFace...")
                ds = load_dataset(self.config.dataset_name, split="test")
                self._items = [
                    {
                        "question": row["Prompt"],
                        "answer": row["Answer"],
                        "difficulty": row.get("reasoning_types", "unknown"),
                        "hops": 2,
                    }
                    for row in ds
                ]
                # Hold out for eval
                eval_size = max(
                    self.config.eval_size,
                    int(len(self._items) * self.config.eval_split_ratio),
                )
                random.shuffle(self._items)
                self._eval_items = self._items[:eval_size]
                self._items = self._items[eval_size:]
                logger.info(
                    f"Loaded {len(self._items)} train / {len(self._eval_items)} eval items "
                    f"from FRAMES benchmark."
                )
                return
            except Exception as e:
                logger.warning(f"Could not load FRAMES from HuggingFace: {e}. Using built-in samples.")

        # Fallback
        random.shuffle(SAMPLE_QUESTIONS)
        split = max(1, len(SAMPLE_QUESTIONS) * 8 // 10)
        self._items = SAMPLE_QUESTIONS[:split]
        self._eval_items = SAMPLE_QUESTIONS[split:]
        logger.info(
            f"Using built-in sample dataset: {len(self._items)} train / "
            f"{len(self._eval_items)} eval items."
        )

    # ------------------------------------------------------------------
    # 2. get_next_item — return the next question
    # ------------------------------------------------------------------

    async def get_next_item(self) -> dict:
        """Return the next item, cycling through the dataset."""
        if not self._items:
            raise RuntimeError("Dataset is empty. Did you call setup()?")
        item = self._items[self._index % len(self._items)]
        self._index += 1
        return item

    # ------------------------------------------------------------------
    # 3. format_prompt — build the user-facing prompt
    # ------------------------------------------------------------------

    def format_prompt(self, item: dict) -> str:
        """Format the research question as a task prompt."""
        return (
            f"Research the following question thoroughly using web search. "
            f"You MUST search the web to find current, accurate information — "
            f"do not rely solely on your training data.\n\n"
            f"Question: {item['question']}\n\n"
            f"Requirements:\n"
            f"- Use web_search and/or web_extract tools to find information\n"
            f"- Search at least 2 different sources\n"
            f"- Provide a concise, accurate answer (2-4 sentences)\n"
            f"- Cite the sources you used"
        )

    # ------------------------------------------------------------------
    # 4. compute_reward — multi-signal scoring
    # ------------------------------------------------------------------

    async def compute_reward(
        self,
        item: dict,
        result: AgentResult,
        ctx: ToolContext,
    ) -> float:
        """
        Multi-signal reward function:

          correctness_weight * correctness  — LLM judge comparing answer to ground truth
          tool_usage_weight  * tool_used    — binary: did the model use web tools?
          efficiency_weight  * efficiency   — penalizes wasteful tool usage
          + diversity_bonus                 — source diversity (≥2 distinct domains)
        """
        # Extract final response from messages (last assistant message with content)
        final_response = ""
        tools_used: list[str] = []
        for msg in reversed(result.messages):
            if msg.get("role") == "assistant" and msg.get("content") and not final_response:
                final_response = msg["content"]
            # Collect tool names from tool call messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    if name:
                        tools_used.append(name)
        tool_call_count: int = result.turns_used or len(tools_used)

        cfg = self.config

        # ---- Signal 1: Answer correctness (LLM judge) ----------------
        correctness = await self._llm_judge(
            question=item["question"],
            expected=item["answer"],
            model_answer=final_response,
        )

        # ---- Signal 2: Web tool usage --------------------------------
        web_tools = {"web_search", "web_extract", "search", "firecrawl"}
        tool_used = 1.0 if any(t in web_tools for t in tools_used) else 0.0

        # ---- Signal 3: Efficiency ------------------------------------
        if tool_call_count <= cfg.efficient_max_calls:
            efficiency = 1.0
        elif tool_call_count <= cfg.heavy_penalty_calls:
            efficiency = 1.0 - (tool_call_count - cfg.efficient_max_calls) * 0.08
        else:
            efficiency = max(0.0, 1.0 - (tool_call_count - cfg.efficient_max_calls) * 0.12)

        # ---- Bonus: Source diversity ---------------------------------
        domains = self._extract_domains(final_response)
        diversity = cfg.diversity_bonus if len(domains) >= 2 else 0.0

        # ---- Combine ------------------------------------------------
        reward = (
            cfg.correctness_weight * correctness
            + cfg.tool_usage_weight * tool_used
            + cfg.efficiency_weight * efficiency
            + diversity
        )
        reward = min(1.0, max(0.0, reward))  # clamp to [0, 1]

        # Track for wandb
        self._reward_buffer.append(reward)
        self._correctness_buffer.append(correctness)
        self._tool_usage_buffer.append(tool_used)
        self._efficiency_buffer.append(efficiency)
        self._diversity_buffer.append(diversity)

        logger.debug(
            f"Reward breakdown — correctness={correctness:.2f}, "
            f"tool_used={tool_used:.1f}, efficiency={efficiency:.2f}, "
            f"diversity={diversity:.1f} → total={reward:.3f}"
        )

        return reward

    # ------------------------------------------------------------------
    # 5. evaluate — run on held-out eval split
    # ------------------------------------------------------------------

    async def evaluate(self, *args, **kwargs) -> None:
        """Run evaluation on the held-out split using the agent loop."""
        import time

        items = self._eval_items
        if not items:
            logger.warning("No eval items available.")
            return

        eval_size = min(self.config.eval_size, len(items))
        eval_items = items[:eval_size]

        logger.info(f"Running eval on {len(eval_items)} questions...")
        start_time = time.time()
        samples = []

        for item in eval_items:
            try:
                # Use the base env's agent loop for eval (same as training)
                prompt = self.format_prompt(item)
                completion = await self.server.chat_completion(
                    messages=[
                        {"role": "system", "content": self.config.system_prompt or ""},
                        {"role": "user", "content": prompt},
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.0,
                    split="eval",
                )

                response_content = (
                    completion.choices[0].message.content if completion.choices else ""
                )

                # Score the response
                correctness = await self._llm_judge(
                    question=item["question"],
                    expected=item["answer"],
                    model_answer=response_content,
                )

                samples.append({
                    "prompt": item["question"],
                    "response": response_content,
                    "expected": item["answer"],
                    "correctness": correctness,
                })

            except Exception as e:
                logger.error(f"Eval error on item: {e}")
                samples.append({
                    "prompt": item["question"],
                    "response": f"ERROR: {e}",
                    "expected": item["answer"],
                    "correctness": 0.0,
                })

        end_time = time.time()

        # Compute metrics
        correctness_scores = [s["correctness"] for s in samples]
        eval_metrics = {
            "eval/mean_correctness": (
                sum(correctness_scores) / len(correctness_scores)
                if correctness_scores else 0.0
            ),
            "eval/n_items": len(samples),
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
        )

    # ------------------------------------------------------------------
    # 6. wandb_log — custom metrics
    # ------------------------------------------------------------------

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None) -> None:
        """Log reward breakdown metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self._reward_buffer:
            n = len(self._reward_buffer)
            wandb_metrics["train/mean_reward"] = sum(self._reward_buffer) / n
            wandb_metrics["train/mean_correctness"] = sum(self._correctness_buffer) / n
            wandb_metrics["train/mean_tool_usage"] = sum(self._tool_usage_buffer) / n
            wandb_metrics["train/mean_efficiency"] = sum(self._efficiency_buffer) / n
            wandb_metrics["train/mean_diversity"] = sum(self._diversity_buffer) / n
            wandb_metrics["train/total_rollouts"] = n

            # Accuracy buckets
            wandb_metrics["train/correct_rate"] = (
                sum(1 for c in self._correctness_buffer if c >= 0.7) / n
            )
            wandb_metrics["train/tool_usage_rate"] = (
                sum(1 for t in self._tool_usage_buffer if t > 0) / n
            )

            # Clear buffers
            self._reward_buffer.clear()
            self._correctness_buffer.clear()
            self._tool_usage_buffer.clear()
            self._efficiency_buffer.clear()
            self._diversity_buffer.clear()

        await super().wandb_log(wandb_metrics)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _llm_judge(
        self,
        question: str,
        expected: str,
        model_answer: str,
    ) -> float:
        """
        Use the server's LLM to judge answer correctness.
        Falls back to keyword heuristic if LLM call fails.
        """
        if not model_answer or not model_answer.strip():
            return 0.0

        judge_prompt = (
            "You are an impartial judge evaluating the quality of an AI research answer.\n\n"
            f"Question: {question}\n\n"
            f"Reference answer: {expected}\n\n"
            f"Model answer: {model_answer}\n\n"
            "Score the model answer on a scale from 0.0 to 1.0 where:\n"
            "  1.0 = fully correct and complete\n"
            "  0.7 = mostly correct with minor gaps\n"
            "  0.4 = partially correct\n"
            "  0.1 = mentions relevant topic but wrong or very incomplete\n"
            "  0.0 = completely wrong or no answer\n\n"
            "Consider: factual accuracy, completeness, and relevance.\n"
            'Respond with ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}'
        )

        try:
            response = await self.server.chat_completion(
                messages=[{"role": "user", "content": judge_prompt}],
                n=1,
                max_tokens=150,
                temperature=0.0,
                split="eval",
            )
            text = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_judge_json(text)
            if parsed is not None:
                return float(parsed)
        except Exception as e:
            logger.debug(f"LLM judge failed: {e}. Using heuristic.")

        return self._heuristic_score(expected, model_answer)

    @staticmethod
    def _parse_judge_json(text: str) -> Optional[float]:
        """Extract the score float from LLM judge JSON response."""
        try:
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(clean)
            score = float(data.get("score", -1))
            if 0.0 <= score <= 1.0:
                return score
        except Exception:
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if match:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
        return None

    @staticmethod
    def _heuristic_score(expected: str, model_answer: str) -> float:
        """Lightweight keyword overlap score as fallback."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
            "at", "to", "for", "with", "and", "or", "but", "it", "its",
            "this", "that", "as", "by", "from", "be", "has", "have", "had",
        }

        def tokenize(text: str) -> set:
            tokens = re.findall(r'\b\w+\b', text.lower())
            return {t for t in tokens if t not in stopwords and len(t) > 2}

        expected_tokens = tokenize(expected)
        answer_tokens = tokenize(model_answer)

        if not expected_tokens:
            return 0.5

        overlap = len(expected_tokens & answer_tokens)
        union = len(expected_tokens | answer_tokens)

        jaccard = overlap / union if union > 0 else 0.0
        recall = overlap / len(expected_tokens)
        return min(1.0, 0.4 * jaccard + 0.6 * recall)

    @staticmethod
    def _extract_domains(text: str) -> set:
        """Extract unique domains from URLs cited in the response."""
        urls = re.findall(r'https?://[^\s\)>\]"\']+', text)
        domains = set()
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower().lstrip("www.")
                if domain:
                    domains.add(domain)
            except Exception:
                pass
        return domains


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    WebResearchEnv.cli()
