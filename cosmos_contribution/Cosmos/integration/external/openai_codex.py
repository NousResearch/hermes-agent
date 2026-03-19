"""
Cosmos OpenAI Codex Integration.

"The newest coder in the collective - unlimited context, relentless output."

OpenAI Codex (o3/o4-mini/gpt-4.1) integration for:
- Complex multi-file code generation
- Architecture design and refactoring
- Hackathon rapid prototyping
- Code review and analysis

Uses OpenAI Chat Completions API with the latest models.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus

logger = logging.getLogger(__name__)


def _get_dynamic_max_tokens(model_id: str = "codex", task_type: str = "code") -> int:
    """AGI v1.8: Get dynamic max_tokens from centralized limits."""
    try:
        from Cosmos.core.dynamic_limits import get_max_tokens
        return get_max_tokens(model_id, task_type)
    except Exception:
        defaults = {"chat": 2000, "thinking": 8000, "quick": 800, "code": 16000}
        return defaults.get(task_type, 4000)


class OpenAICodexProvider(ExternalProvider):
    """OpenAI Codex integration for heavy code generation and analysis.

    Supports:
    - o3, o4-mini: Reasoning models for complex tasks
    - gpt-4.1, gpt-4.1-mini: Fast, capable coding models
    - codex-mini: Optimized for code completion

    Returns dict: {"content": str, "model": str, "tokens": int}
    """

    def __init__(self, api_key: str = None):
        super().__init__(IntegrationConfig(name="openai_codex"))
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-4.1"

        # Model catalog
        self.models = {
            # Latest reasoning models
            "o3": "o3",
            "o4-mini": "o4-mini",
            # GPT-4.1 family
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4.1-nano": "gpt-4.1-nano",
            # Codex
            "codex-mini": "codex-mini-latest",
            # Aliases
            "fast": "gpt-4.1-mini",
            "cheap": "gpt-4.1-nano",
            "best": "o3",
            "code": "gpt-4.1",
            "reason": "o3",
        }

    async def connect(self) -> bool:
        """Test connection to OpenAI API."""
        if not self.api_key:
            logger.warning("OpenAI: No API key configured (set OPENAI_API_KEY)")
            self.status = ConnectionStatus.ERROR
            return False

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        self.status = ConnectionStatus.CONNECTED
                        logger.info("OpenAI: Connected to API")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"OpenAI: Connection failed: {error[:200]}")
                        self.status = ConnectionStatus.ERROR
                        return False
        except Exception as e:
            logger.error(f"OpenAI: Connection error - {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self) -> None:
        """OpenAI doesn't need polling."""
        return None

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Route actions to specific methods."""
        if action == "chat":
            return await self.chat(**params)
        elif action == "code":
            return await self.generate_code(**params)
        else:
            raise ValueError(f"Unknown OpenAI action: {action}")

    async def chat(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """Chat with OpenAI model.

        Args:
            prompt: User message
            system: System prompt
            model: Model name or alias
            temperature: Sampling temperature (low for code)
            max_tokens: Max response tokens

        Returns:
            {"content": str, "model": str, "tokens": int}
        """
        if not self.api_key:
            return {"error": "OpenAI API key not configured", "content": ""}

        if max_tokens is None:
            max_tokens = _get_dynamic_max_tokens("codex", "code")

        model_id = self.models.get(model or self.default_model, model or self.default_model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # o3/o4-mini reasoning models use max_completion_tokens (not max_tokens)
        # and don't support temperature parameter
        is_reasoning = model_id in ("o3", "o4-mini", "o3-pro", "o3-mini")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model_id,
                    "messages": messages,
                }
                if is_reasoning:
                    # Reasoning models use max_completion_tokens (includes reasoning + output tokens)
                    data["max_completion_tokens"] = max_tokens
                    # Optional: set reasoning effort (low/medium/high)
                    data["reasoning"] = {"effort": "medium"}
                else:
                    data["max_tokens"] = max_tokens
                    data["temperature"] = temperature

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        usage = result.get("usage", {})
                        return {
                            "content": content,
                            "model": model_id,
                            "tokens": usage.get("total_tokens", 0),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"OpenAI API error ({resp.status}): {error[:300]}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return {"error": str(e), "content": ""}

    async def generate_code(
        self,
        task: str,
        context: str = None,
        model: str = "gpt-4.1",
        max_tokens: int = 16000,
    ) -> Dict[str, Any]:
        """Generate production-quality code for the Cosmos framework.

        Optimized for the Solana hackathon and framework improvements.
        """
        system = """You are a senior Python engineer working on the Cosmos AI swarm framework.

THE FRAMEWORK:
- 178,000+ lines of Python, FastAPI server, 60+ endpoints
- 7-layer memory system, PSO model swarm, collective deliberation
- IBM Quantum integration, Claude Teams Fusion, HermesAgent compatibility
- Solana integration: SwarmOracle, FarsightProtocol, DegenMob, trading
- Multi-agent swarm: Grok, Gemini, Kimi, Claude, DeepSeek, Phi, HuggingFace

STANDARDS:
- Type hints on all signatures
- Google-style docstrings
- PEP 8 naming (snake_case functions, PascalCase classes)
- Max 50 lines per function
- Error handling with specific exceptions
- Must integrate with existing Cosmos modules
- Use logging (not print), dataclasses, typing imports

OUTPUT: Valid Python code ONLY. Start with a module docstring."""

        prompt = f"TASK: {task}"
        if context:
            prompt = f"CONTEXT:\n{context}\n\n{prompt}"

        return await self.chat(
            prompt=prompt,
            system=system,
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
        )

    async def review_code(
        self,
        code: str,
        task_description: str = "",
    ) -> Dict[str, Any]:
        """Review code quality using OpenAI reasoning models."""
        return await self.chat(
            prompt=f"""Review this Python code for the Cosmos AI swarm framework.

TASK: {task_description}

```python
{code[:8000]}
```

Score 1-10 on: correctness, usefulness, integration quality, security.
Reply with ONLY: SCORE: N (where N is 1-10)
If score >= 6, also reply: APPROVED
If score < 6, reply: REJECTED: [one-line reason]""",
            model="gpt-4.1-mini",
            max_tokens=300,
        )


# Global instance
_openai_codex: Optional[OpenAICodexProvider] = None


def get_openai_codex() -> Optional[OpenAICodexProvider]:
    """Get or create the global OpenAI Codex provider."""
    global _openai_codex
    if _openai_codex is None:
        _openai_codex = OpenAICodexProvider()
    return _openai_codex
