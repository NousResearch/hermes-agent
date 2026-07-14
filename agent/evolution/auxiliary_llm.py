"""Auxiliary LLM client for Evolution Engine — uses Hermes' existing model.

No separate API key needed. Uses the same model the user already configured
during Hermes install. Falls back through the standard auxiliary resolution
chain (main model → OpenRouter → Nous Portal → native Anthropic → etc.).

Config (optional — defaults to main model):
  auxiliary:
    evolution:
      provider: auto     # or "openai", "anthropic", "deepseek", etc.
      model: auto         # or a specific model name
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EvolutionLLMClient:
    """Thin wrapper around Hermes' existing auxiliary_client for evolution tasks."""

    def __init__(self):
        self._available = None  # Lazy check

    @property
    def is_available(self) -> bool:
        """Check if any LLM backend is available through Hermes' resolver."""
        if self._available is None:
            try:
                from agent.auxiliary_client import _resolve_auto
                result = _resolve_auto(task=None)
                self._available = result is not None
            except Exception:
                self._available = False
        return self._available

    def analyze_sync(self, prompt: str) -> str:
        """Run failure analysis using Hermes' configured model."""
        return self._call(prompt, max_tokens=2048, temperature=0.3)

    def propose_sync(self, prompt: str) -> str:
        """Generate improvement proposals using Hermes' configured model."""
        return self._call(prompt, max_tokens=4096, temperature=0.3)

    def judge_sync(self, rubric: str, trajectory_json: str) -> Dict[str, Any]:
        """LLM-as-judge using Hermes' configured model."""
        prompt = (
            f"Score this agent performance against the rubric.\n\n"
            f"RUBRIC:\n{rubric}\n\n"
            f"TRAJECTORY:\n{trajectory_json[:8000]}\n\n"
            f'Respond with ONLY JSON: {{"passed": true/false, "score": 0.0-1.0, "reasoning": "..."}}'
        )
        response = self._call(prompt, max_tokens=512, temperature=0.0)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"passed": False, "score": 0.0, "reasoning": "Failed to parse"}

    # Legacy async methods — delegate to sync
    async def analyze(self, prompt: str) -> str:
        return self.analyze_sync(prompt)

    async def propose(self, prompt: str) -> str:
        return self.propose_sync(prompt)

    async def judge(self, rubric: str, trajectory_json: str) -> Dict[str, Any]:
        return self.judge_sync(rubric, trajectory_json)

    def _call(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Call Hermes' auxiliary LLM with auto-resolved provider."""
        try:
            from agent.auxiliary_client import call_llm
            response = call_llm(
                task="evolution",
                messages=[
                    {"role": "system", "content": "You are an AI engineering assistant. Respond with ONLY valid JSON. No markdown, no explanation outside the JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = None
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
            elif isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
            return content or "{}"
        except Exception as e:
            logger.debug("Evolution LLM call failed: %s", e)
            # Fallback: return empty JSON so callers degrade gracefully
            return "{}"


def get_evolution_llm() -> Optional[EvolutionLLMClient]:
    """Return the evolution LLM client. None if no backend available."""
    client = EvolutionLLMClient()
    return client if client.is_available else None
