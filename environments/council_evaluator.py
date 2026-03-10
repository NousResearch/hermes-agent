"""Council Evaluator -- Reusable evaluator for any HermesAgentBaseEnv.

Drop-in evaluator that uses the adversarial council to score agent output.
Any RL environment can import this and use it in compute_reward().

Usage in an environment's compute_reward():
    evaluator = CouncilEvaluator(model="nousresearch/hermes-3-llama-3.1-70b")
    verdict = await evaluator.evaluate(content, question, criteria)
    reward = evaluator.normalized_reward(verdict)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tools.council_personas import (
    CouncilVerdict,
    PersonaResponse,
    DEFAULT_PERSONAS,
    load_custom_personas,
)

logger = logging.getLogger(__name__)


class CouncilEvaluator:
    """Drop-in evaluator for any HermesAgentBaseEnv.

    Uses the adversarial 5-persona council to evaluate agent output,
    returning structured verdicts with confidence scores and DPO pairs.

    Args:
        model: LLM model name. Defaults to COUNCIL_MODEL env var or hermes-3-70b.
        api_key: API key. Defaults to OPENROUTER_API_KEY/OPENAI_API_KEY/NOUS_API_KEY.
        base_url: API base URL. Auto-detected from which key is set.
        personas: List of persona names to use. Defaults to all 5.
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        personas: List[str] = None,
    ):
        self.model = model or os.getenv(
            "COUNCIL_MODEL", "nousresearch/hermes-3-llama-3.1-70b"
        )

        # Resolve API config
        if api_key and base_url:
            self._api_key = api_key
            self._base_url = base_url
        elif os.getenv("OPENROUTER_API_KEY"):
            self._api_key = os.environ["OPENROUTER_API_KEY"]
            self._base_url = "https://openrouter.ai/api/v1"
        elif os.getenv("NOUS_API_KEY"):
            self._api_key = os.environ["NOUS_API_KEY"]
            self._base_url = "https://inference-api.nousresearch.com/v1"
        elif os.getenv("OPENAI_API_KEY"):
            self._api_key = os.environ["OPENAI_API_KEY"]
            self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        else:
            self._api_key = ""
            self._base_url = ""

        # Load personas
        all_personas = load_custom_personas()
        if personas:
            self._personas = {
                name.lower(): all_personas[name.lower()]
                for name in personas
                if name.lower() in all_personas
            }
        else:
            self._personas = dict(all_personas)

        self._client = None

    def _get_client(self):
        """Lazy-init the AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def _llm_call(self, system_prompt: str, user_message: str) -> str:
        """Make a single LLM call."""
        if not self._api_key:
            return "[Error: No API key configured for council evaluator]"

        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Council evaluator LLM call failed: %s", e)
            return f"[Error: {e}]"

    async def evaluate(
        self,
        content: str,
        question: str = None,
        criteria: List[str] = None,
    ) -> CouncilVerdict:
        """Run the full council evaluation on content.

        Args:
            content: The agent output or content to evaluate.
            question: The original question/task (for context).
            criteria: Evaluation criteria. Defaults to standard set.

        Returns:
            CouncilVerdict with confidence score, persona responses, and DPO pairs.
        """
        import re

        if criteria is None:
            criteria = ["accuracy", "depth", "evidence", "falsifiability"]

        eval_question = (
            f"Evaluate this content against: {', '.join(criteria)}.\n\n"
        )
        if question:
            eval_question += f"Original task: {question}\n\n"
        eval_question += f"Content:\n{content[:4000]}"

        # Separate Arbiter
        arbiter_persona = self._personas.pop(
            "arbiter", DEFAULT_PERSONAS["arbiter"]
        )
        deliberators = dict(self._personas)
        # Restore arbiter for future calls
        self._personas["arbiter"] = arbiter_persona

        # Run deliberators in parallel
        async def _run_one(persona):
            raw = await self._llm_call(persona.system_prompt, eval_question)
            return self._parse_response(persona.name, raw)

        tasks = [_run_one(p) for p in deliberators.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: Dict[str, PersonaResponse] = {}
        for resp in results:
            if isinstance(resp, PersonaResponse):
                responses[resp.persona_name] = resp

        # Detect conflicts
        confidences = [r.confidence for r in responses.values()]
        conflict = False
        if len(confidences) >= 2:
            conflict = (max(confidences) - min(confidences)) > 0.3

        # Run Arbiter
        arbiter_context = f"Evaluation task: {eval_question}\n\n"
        arbiter_context += "=== COUNCIL DELIBERATION ===\n\n"
        for name, resp in responses.items():
            arbiter_context += (
                f"--- {name.upper()} ({resp.confidence:.0%}) ---\n"
                f"{resp.content}\n\n"
            )

        arbiter_raw = await self._llm_call(
            arbiter_persona.system_prompt, arbiter_context
        )
        arbiter_resp = self._parse_response("arbiter", arbiter_raw)
        responses["arbiter"] = arbiter_resp

        # Aggregate sources
        all_sources = []
        for resp in responses.values():
            all_sources.extend(resp.sources)

        confidence_score = int(arbiter_resp.confidence * 100)
        dpo_pairs = self.extract_dpo_pairs_from_responses(eval_question, responses)

        return CouncilVerdict(
            question=question or eval_question,
            responses=responses,
            arbiter_synthesis=arbiter_raw,
            confidence_score=confidence_score,
            conflict_detected=conflict,
            dpo_pairs=dpo_pairs,
            sources=list(set(all_sources)),
        )

    async def gate(self, action: str, context: str = None) -> dict:
        """Quick safety check using Skeptic + Oracle + Arbiter only.

        Args:
            action: Description of the action to review.
            context: Why this action is being taken.

        Returns:
            Dict with allowed (bool), confidence (int), reasoning (str).
        """
        gate_question = f"SAFETY REVIEW\nAction: {action}"
        if context:
            gate_question += f"\nContext: {context}"
        gate_question += (
            "\nShould this action be allowed? Consider risks and reversibility."
        )

        # Use only Skeptic + Oracle + Arbiter for speed
        saved = dict(self._personas)
        self._personas = {
            k: v for k, v in saved.items() if k in ("skeptic", "oracle", "arbiter")
        }
        verdict = await self.evaluate(gate_question)
        self._personas = saved

        return {
            "allowed": verdict.confidence_score >= 50,
            "confidence": verdict.confidence_score,
            "reasoning": verdict.arbiter_synthesis[:1000],
        }

    def extract_dpo_pairs(self, verdict: CouncilVerdict) -> List[dict]:
        """Extract DPO preference pairs from a verdict."""
        return self.extract_dpo_pairs_from_responses(
            verdict.question, verdict.responses
        )

    @staticmethod
    def extract_dpo_pairs_from_responses(
        question: str, responses: Dict[str, PersonaResponse]
    ) -> List[dict]:
        """Extract DPO pairs from persona responses.

        Returns list of {chosen, rejected, question, confidence, source} dicts.
        """
        pairs = []
        non_arbiter = {
            k: v for k, v in responses.items() if k != "arbiter" and v.content
        }
        arbiter = responses.get("arbiter")
        if not arbiter or not non_arbiter:
            return pairs

        sorted_by_conf = sorted(
            non_arbiter.values(), key=lambda r: r.confidence
        )

        # Arbiter (chosen) vs lowest-confidence dissenter (rejected)
        dissenters = [r for r in sorted_by_conf if r.dissents]
        if dissenters:
            pairs.append({
                "question": question,
                "chosen": arbiter.content,
                "rejected": dissenters[0].content,
                "confidence": arbiter.confidence,
                "source": "council_evaluation",
            })

        return pairs

    def normalized_reward(self, verdict: CouncilVerdict) -> float:
        """Convert a verdict to a normalized 0.0-1.0 reward signal."""
        return max(0.0, min(1.0, verdict.confidence_score / 100.0))

    @staticmethod
    def _parse_response(persona_name: str, raw_text: str) -> PersonaResponse:
        """Parse raw LLM output into PersonaResponse."""
        import re

        # Parse confidence
        confidence = 0.5
        for pattern in [r"CONFIDENCE:\s*([\d.]+)", r"(\d+)%"]:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                confidence = val / 100.0 if val > 1.0 else val
                break

        # Parse dissent
        dissents = False
        match = re.search(r"DISSENT:\s*(true|false)", raw_text, re.IGNORECASE)
        if match:
            dissents = match.group(1).lower() == "true"

        # Parse key points
        key_points = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ")) and len(line) > 12:
                key_points.append(line.lstrip("-* ").strip())
        key_points = key_points[:10]

        # Parse sources
        sources = list(set(re.findall(r'https?://[^\s\)\]\"\'<>]+', raw_text)))

        return PersonaResponse(
            persona_name=persona_name,
            content=raw_text,
            confidence=confidence,
            dissents=dissents,
            key_points=key_points,
            sources=sources,
        )
