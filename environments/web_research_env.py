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
    python environments/web_research_env.py serve \
        --openai.base_url http://localhost:8000/v1 \
        --openai.model_name YourModel \
        --openai.server_type openai

    # With eval split
    python environments/web_research_env.py serve \
        --openai.base_url http://localhost:8000/v1 \
        --openai.model_name YourModel \
        --env.eval_every 50 \
        --env.eval_size 20

    # Standalone eval (no training server needed)
    python environments/web_research_env.py eval \
        --openai.base_url http://localhost:8000/v1 \
        --openai.model_name YourModel

Built by: github.com/jackx707
Inspired by: GroceryMind — production Hermes agent doing live web research
             across German grocery stores (firecrawl + hermes-agent)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Optional HuggingFace datasets import
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from environments.hermes_base_env import HermesAgentBaseEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback sample dataset (used when HuggingFace is unavailable)
# These are multi-hop questions that require real web search to answer.
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
      20% — efficiency (penalizes >6 tool calls)

    Bonus +0.1 for source diversity (≥2 distinct domains cited).
    """

    name = "web-research"

    # Default toolsets for this environment — web + file for saving notes
    default_toolsets = ["web", "file"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._items: list[dict] = []
        self._eval_items: list[dict] = []
        self._index: int = 0
        self._total_scored: int = 0
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    # 1. Setup — load dataset
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Load the FRAMES benchmark or fall back to built-in samples."""
        if HF_AVAILABLE:
            try:
                logger.info("Loading FRAMES benchmark from HuggingFace...")
                ds = load_dataset("google/frames-benchmark", split="test")
                self._items = [
                    {
                        "question": row["Prompt"],
                        "answer": row["Answer"],
                        "difficulty": row.get("reasoning_types", "unknown"),
                        "hops": 2,
                    }
                    for row in ds
                ]
                # Hold out 10% for eval
                eval_size = max(20, len(self._items) // 10)
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
        """
        Format the research question as a task prompt.
        Instructs the model to use web search and cite sources.
        """
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
        result: dict,
        ctx: Any,  # ToolContext
    ) -> float:
        """
        Multi-signal reward function:

          0.6 * correctness   — LLM judge comparing answer to ground truth
          0.2 * tool_used     — binary: did the model use web tools?
          0.2 * efficiency    — penalizes wasteful tool usage
          +0.1 bonus          — source diversity (≥2 distinct domains)
        """
        final_response: str = result.get("final_response", "")
        tools_used: list[str] = result.get("tools_used", [])
        tool_call_count: int = result.get("tool_call_count", len(tools_used))

        # ---- Signal 1: Answer correctness (LLM judge) ----------------
        correctness = await self._llm_judge(
            question=item["question"],
            expected=item["answer"],
            model_answer=final_response,
            ctx=ctx,
        )

        # ---- Signal 2: Web tool usage --------------------------------
        web_tools = {"web_search", "web_extract", "search", "firecrawl"}
        tool_used = 1.0 if any(t in web_tools for t in tools_used) else 0.0

        # ---- Signal 3: Efficiency ------------------------------------
        # Ideal: 2-5 tool calls. Penalise beyond 6, hard cap at 15.
        if tool_call_count <= 5:
            efficiency = 1.0
        elif tool_call_count <= 10:
            efficiency = 1.0 - (tool_call_count - 5) * 0.08
        else:
            efficiency = max(0.0, 1.0 - (tool_call_count - 5) * 0.12)

        # ---- Bonus: Source diversity ---------------------------------
        domains = self._extract_domains(final_response)
        diversity_bonus = 0.1 if len(domains) >= 2 else 0.0

        # ---- Combine ------------------------------------------------
        reward = (
            0.6 * correctness
            + 0.2 * tool_used
            + 0.2 * efficiency
            + diversity_bonus
        )
        reward = min(1.0, max(0.0, reward))  # clamp to [0, 1]

        # Track running stats
        self._total_scored += 1
        self._total_reward += reward

        logger.debug(
            f"Reward breakdown — correctness={correctness:.2f}, "
            f"tool_used={tool_used:.1f}, efficiency={efficiency:.2f}, "
            f"diversity_bonus={diversity_bonus:.1f} → total={reward:.3f}"
        )

        return reward

    # ------------------------------------------------------------------
    # 5. evaluate — run on held-out eval split
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        *args: Any,
        eval_size: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Run evaluation on the held-out split.
        Returns a dict of metrics for logging.
        """
        items = self._eval_items
        if eval_size:
            items = items[:eval_size]

        if not items:
            logger.warning("No eval items available.")
            return {}

        logger.info(f"Running eval on {len(items)} questions...")

        rewards = []
        correctness_scores = []

        for item in items:
            try:
                # Run the agent on each eval question
                result = await self._run_agent_on_item(item)
                reward = await self.compute_reward(item, result, ctx=None)
                rewards.append(reward)

                # Also track raw correctness separately
                if result.get("final_response"):
                    correctness_scores.append(
                        await self._llm_judge(
                            question=item["question"],
                            expected=item["answer"],
                            model_answer=result["final_response"],
                            ctx=None,
                        )
                    )
            except Exception as e:
                logger.error(f"Eval error on item: {e}")
                rewards.append(0.0)

        metrics = {
            "eval/mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "eval/mean_correctness": (
                sum(correctness_scores) / len(correctness_scores)
                if correctness_scores else 0.0
            ),
            "eval/n_items": len(rewards),
            "train/mean_reward_so_far": (
                self._total_reward / self._total_scored
                if self._total_scored > 0 else 0.0
            ),
        }

        logger.info(
            f"Eval complete — mean_reward={metrics['eval/mean_reward']:.3f}, "
            f"mean_correctness={metrics['eval/mean_correctness']:.3f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _llm_judge(
        self,
        question: str,
        expected: str,
        model_answer: str,
        ctx: Any,
    ) -> float:
        """
        Use an LLM to judge whether `model_answer` correctly addresses
        `question` compared to `expected`. Returns a float in [0, 1].

        Uses the agent's own inference client if ctx is available,
        otherwise falls back to a lightweight heuristic.
        """
        if not model_answer or not model_answer.strip():
            return 0.0

        # Build judge prompt
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
            "Respond with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}"
        )

        # Try using ctx for inference (Phase 2 / live training)
        if ctx is not None and hasattr(ctx, "chat_completion"):
            try:
                response = await ctx.chat_completion(
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                text = response.get("content", "")
                parsed = self._parse_judge_json(text)
                if parsed is not None:
                    return float(parsed)
            except Exception as e:
                logger.debug(f"LLM judge via ctx failed: {e}. Using heuristic.")

        # Fallback: keyword overlap heuristic
        return self._heuristic_score(expected, model_answer)

    @staticmethod
    def _parse_judge_json(text: str) -> Optional[float]:
        """Extract the score float from LLM judge JSON response."""
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(clean)
            score = float(data.get("score", -1))
            if 0.0 <= score <= 1.0:
                return score
        except Exception:
            # Try regex fallback
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if match:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
        return None

    @staticmethod
    def _heuristic_score(expected: str, model_answer: str) -> float:
        """
        Lightweight keyword overlap score as fallback when no LLM is available.
        Extracts meaningful tokens and computes Jaccard similarity.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
            "at", "to", "for", "with", "and", "or", "but", "it", "its",
            "this", "that", "as", "by", "from", "be", "has", "have", "had",
        }

        def tokenize(text: str) -> set:
            tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
            return {t for t in tokens if t not in stopwords and len(t) > 2}

        expected_tokens = tokenize(expected)
        answer_tokens = tokenize(model_answer)

        if not expected_tokens:
            return 0.5  # Can't judge

        overlap = len(expected_tokens & answer_tokens)
        union = len(expected_tokens | answer_tokens)

        jaccard = overlap / union if union > 0 else 0.0
        # Recall-weighted: reward covering expected content
        recall = overlap / len(expected_tokens)
        return min(1.0, 0.4 * jaccard + 0.6 * recall)

    @staticmethod
    def _extract_domains(text: str) -> set:
        """
        Extract unique domains from URLs cited in the response.
        Used to measure source diversity.
        """
        urls = re.findall(r'https?://[^\s\)>\]"\']+', text)
        domains = set()
        for url in urls:
            try:
                parsed = urlparse(url)
                # Normalize: strip www.
                domain = parsed.netloc.lower().lstrip("www.")
                if domain:
                    domains.add(domain)
            except Exception:
                pass
        return domains

    async def _run_agent_on_item(self, item: dict) -> dict:
        """
        Stub for running agent during eval. In Phase 1/2, this is handled
        by the Atropos framework's rollout mechanism. Provided here for
        standalone eval compatibility.
        """
        # In real usage, the framework calls get_next_item + format_prompt
        # and runs the agent. This stub returns an empty result for safety.
        return {
            "final_response": "",
            "tools_used": [],
            "tool_call_count": 0,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    WebResearchEnv.cli()
