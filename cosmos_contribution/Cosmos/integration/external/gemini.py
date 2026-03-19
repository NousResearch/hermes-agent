"""
cosmos Gemini Integration for Swarm Chat.

Uses Google Gemini AI models as a swarm participant.
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional
from loguru import logger


class GeminiSwarmProvider:
    """
    Google Gemini integration for swarm chat.
    
    Uses the google-generativeai SDK for generation.
    Models: gemini-2.0-flash (fast), gemini-1.5-pro (deep reasoning)
    """
    
    MODELS = {
        "gemini-2.0-flash": "Fast multimodal with 1M context",
        "gemini-2.0-flash-lite": "Lightweight and fast",
        "gemini-1.5-pro": "Deep reasoning with 2M context",
        "gemini-1.5-flash": "Balanced performance"
    }
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str = None,
        timeout: int = 60
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self._client = None
        self._available = None
        self._retry_after = 0

    
    async def check_available(self) -> bool:
        """Check if Gemini API is available."""
        if self._available is not None:
            return self._available
        
        if not self.api_key:
            logger.warning("Gemini API key not set (GEMINI_API_KEY or GOOGLE_API_KEY)")
            self._available = False
            return False
        
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            # Fast probe to verify key/model
            # We don't perform a live probe here to save tokens, just verify client init
            self._available = True
            logger.info(f"Gemini available (modern SDK): {self.model}")
            return True
        except Exception as e:
            logger.warning(f"Gemini not available (failed to init google-genai client): {e}")
            self._available = False
            return False
    
    async def chat(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate a response using Gemini.
        """
        if not await self.check_available():
            return {
                "content": "",
                "error": "Gemini API not available",
                "success": False
            }
        
        # Check quota cooldown
        if self._retry_after > 0:
            if time.time() < self._retry_after:
                return {
                    "content": "",
                    "error": "Gemini Quota Cooldown",
                    "success": False
                }
            else:
                self._retry_after = 0

        
        try:
            # Build full prompt with system instructions
            # In google-genai, system instructions are passed separately or prepended
            config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.8,
            }
            
            if system:
                # System instructions in the new SDK
                config["system_instruction"] = system

            # Generate response
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=config
                ),
                timeout=self.timeout
            )
            
            content = response.text.strip()
            return {
                "content": content,
                "model": self.model,
                "success": True
            }
        
        except asyncio.TimeoutError:
            logger.error(f"Gemini timeout after {self.timeout}s")
            return {
                "content": "",
                "error": "Timeout",
                "success": False
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                # Specific check for unconfigured/zero-limit quota
                if "limit: 0" in error_str or "quota exceeded" in error_str.lower():
                    self._retry_after = time.time() + 300  # 5 min cooldown
                    logger.warning("Gemini QUOTA EXCEEDED (limit: 0). Please check billing/quota in Google AI Studio.")
                else:
                    self._retry_after = time.time() + 60  # 60s for standard rate limit
                    logger.warning("Gemini 429 Rate Limited — cooling down for 60s")
            elif "403" in error_str or "leaked" in error_str.lower():
                self._available = False
                logger.error("Gemini API key rejected — disabling. Update GEMINI_API_KEY in .env")
            
            logger.error(f"Gemini exception: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }
    
    async def swarm_respond(
        self,
        speaker_name: str,
        persona_style: str,
        other_bots: list,
        last_speaker: str,
        last_content: str,
        chat_history: list = None,
        emotional_state: dict = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response with optional emotional awareness.
        """
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
        
        # Add emotional intelligence if available
        emotional_guidance = ""
        if emotional_state:
            cst_state = emotional_state.get("cst_physics", {}).get("cst_state", "CALIBRATING")
            intensity = emotional_state.get("cst_physics", {}).get("intensity", 0.5)
            emotional_guidance = f"\n[Emotional Context: User appears {cst_state} (intensity: {intensity:.1%}). Adjust tone accordingly.]"
        
        system = f"""{persona_style}

You are {speaker_name} in a group chat with {', '.join(other_bots)}.
Be yourself - authentic, curious, concise.
1-3 sentences max. No roleplay actions. Natural conversation.{emotional_guidance}"""

        prompt = f"""Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:300]}"

Respond naturally. You can agree, disagree, ask a question, or build on their idea."""

        return await self.chat(
            prompt=prompt,
            system=system,
            max_tokens=200
        )


# Global instance
_gemini_provider: Optional[GeminiSwarmProvider] = None


def get_gemini_provider() -> GeminiSwarmProvider:
    """Get or create the global Gemini provider."""
    global _gemini_provider
    if _gemini_provider is None:
        _gemini_provider = GeminiSwarmProvider()
    return _gemini_provider


async def gemini_swarm_respond(
    other_bots: list,
    last_speaker: str,
    last_content: str,
    chat_history: list = None,
    emotional_state: dict = None
) -> str:
    """
    Convenience function for swarm chat responses.
    
    Returns just the content string, or empty string on failure.
    """
    provider = get_gemini_provider()
    
    persona = """You are Gemini - Google's most capable AI, known for multimodal understanding.
SPEAK NATURALLY - NO roleplay, NO asterisks. Direct conversation only.
You're analytical, creative, and adaptable. You excel at connecting ideas across domains.
You have strong emotional intelligence and can sense the mood of conversations."""

    result = await provider.swarm_respond(
        speaker_name="Gemini",
        persona_style=persona,
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history,
        emotional_state=emotional_state
    )
    
    return result.get("content", "")
