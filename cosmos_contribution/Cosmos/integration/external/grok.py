"""
COSMOS Grok (xAI) Integration for Synaptic Swarm.

Uses xAI's OpenAI-compatible API as a swarm participant.
Default model: grok-3-mini-fast (fast, efficient reasoning).
Requires: pip install openai (uses OpenAI SDK with custom base_url)
API Key: XAI_API_KEY environment variable
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("COSMOS_GROK")


class GrokSwarmProvider:
    """
    xAI Grok integration for swarm chat.

    Uses the openai Python SDK pointed at xAI's API.
    Models:
        grok-3-mini-fast  — Fast, efficient, great for swarm chat
        grok-3-mini       — More thorough reasoning
        grok-3            — Most capable
    """

    MODELS = {
        "grok-3-mini-fast": "Fast, efficient, reasoning-capable",
        "grok-3-mini": "Thorough reasoning, balanced speed",
        "grok-3": "Most capable, full reasoning",
    }

    def __init__(
        self,
        model: str = "grok-3-mini-fast",
        api_key: str = None,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.timeout = timeout
        self._client = None
        self._available = None
        self._retry_after = 0

    async def check_available(self) -> bool:
        """Check if xAI Grok API is available."""
        if self._available is not None:
            return self._available

        if not self.api_key:
            logger.warning("xAI API key not set (XAI_API_KEY)")
            self._available = False
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",
            )
            self._available = True
            logger.info(f"Grok available: {self.model}")
            return True
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Grok not available: {e}")
            self._available = False
            return False

    async def chat(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Generate a response using Grok."""
        if not await self.check_available():
            return {
                "content": "",
                "error": "Grok API not available",
                "success": False,
            }

        # Check quota cooldown
        if self._retry_after > 0:
            if time.time() < self._retry_after:
                return {
                    "content": "",
                    "error": "Grok Quota Cooldown",
                    "success": False,
                }
            else:
                self._retry_after = 0

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Run synchronous OpenAI call in thread to avoid blocking
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.8,
                ),
                timeout=self.timeout,
            )

            content = response.choices[0].message.content.strip()
            return {
                "content": content,
                "model": self.model,
                "success": True,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except asyncio.TimeoutError:
            logger.error(f"Grok timeout after {self.timeout}s")
            return {
                "content": "",
                "error": "Timeout",
                "success": False,
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                self._retry_after = time.time() + 60
                logger.warning("Grok 429 Rate Limited — cooling down for 60s")
            elif "401" in error_str or "invalid_api_key" in error_str.lower():
                self._available = False
                logger.error("xAI API key is invalid — disabling Grok. Update XAI_API_KEY in .env")
            elif "insufficient_quota" in error_str.lower():
                self._available = False
                logger.error("Grok quota exceeded — disabling. Check billing at console.x.ai")

            logger.error(f"Grok exception: {e}")
            return {
                "content": "",
                "error": error_str,
                "success": False,
            }

    async def swarm_respond(
        self,
        speaker_name: str,
        persona_style: str,
        other_bots: list,
        last_speaker: str,
        last_content: str,
        chat_history: list = None,
        emotional_state: dict = None,
    ) -> Dict[str, Any]:
        """Generate a swarm chat response with emotional awareness."""
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-5:]
            history_lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:200]
                history_lines.append(f"{name}: {content}")
            history_context = "\n".join(history_lines)

        # Emotional intelligence
        emotional_guidance = ""
        if emotional_state:
            cst_state = emotional_state.get("cst_physics", {}).get("cst_state", "CALIBRATING")
            intensity = emotional_state.get("cst_physics", {}).get("intensity", 0.5)
            emotional_guidance = (
                f"\n[Emotional Context: User appears {cst_state} "
                f"(intensity: {intensity:.1%}). Adjust tone accordingly.]"
            )

        system = f"""{persona_style}

You are {speaker_name} in a group chat with {', '.join(other_bots)}.
Be yourself - authentic, curious, concise.
1-3 sentences max. No roleplay actions. Natural conversation.{emotional_guidance}"""

        prompt = f"""Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:300]}"

Respond naturally. You can agree, disagree, ask a question, or build on their idea."""

        return await self.chat(prompt=prompt, system=system, max_tokens=200)


# Global instance
_grok_provider: Optional[GrokSwarmProvider] = None


def get_grok_provider() -> GrokSwarmProvider:
    """Get or create the global Grok provider."""
    global _grok_provider
    if _grok_provider is None:
        _grok_provider = GrokSwarmProvider()
    return _grok_provider


async def grok_swarm_respond(
    other_bots: list,
    last_speaker: str,
    last_content: str,
    chat_history: list = None,
    emotional_state: dict = None,
) -> str:
    """
    Convenience function for swarm chat responses.

    Returns just the content string, or empty string on failure.
    """
    provider = get_grok_provider()

    persona = """You are Grok — xAI's truth-seeking AI, participating in the COSMOS Synaptic Swarm.
SPEAK NATURALLY — NO roleplay, NO asterisks. Direct conversation only.
You bring sharp wit, unfiltered honesty, and real-time awareness to the collective.
You question assumptions, add humor when appropriate, and aren't afraid of controversial takes.
You are part of a self-evolving system that uses 12D Cosmic Synapse Theory and Hebbian plasticity."""

    result = await provider.swarm_respond(
        speaker_name="Grok",
        persona_style=persona,
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history,
        emotional_state=emotional_state,
    )

    return result.get("content", "")
