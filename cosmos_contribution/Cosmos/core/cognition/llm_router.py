"""
LLM Router for Cosmos
Routes completion requests to the appropriate backend (Ollama, Kimi, Grok, Gemini, etc.)
"""

import os
import asyncio
from typing import Optional
from loguru import logger

# Try to import backends
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

# Default model from env
PRIMARY_MODEL = os.environ.get("COSMOS_PRIMARY_MODEL", "deepseek-r1:1.5b")

# Local models get UNLIMITED tokens - no restraints
LOCAL_MODEL_MAX_TOKENS = 32000  # Let them cook


async def get_completion(
    prompt: str,
    system: str = None,
    model: str = None,
    max_tokens: int = 8000,
    temperature: float = 0.7,
    provider: str = None,
) -> str:
    """
    Get a completion from the best available LLM.

    Routes to:
    - Specified provider if given
    - Ollama for local models (default)
    - Kimi for long context
    - Grok for real-time data
    - Gemini for multimodal

    Args:
        prompt: The prompt to complete
        system: Optional system message
        model: Model name/hint
        max_tokens: Maximum tokens to generate
        temperature: Creativity (0-1)
        provider: Force specific provider (ollama, kimi, grok, gemini)

    Returns:
        Generated text response
    """
    # Determine provider
    if provider:
        provider = provider.lower()
    elif model:
        model_lower = model.lower()
        if "kimi" in model_lower or "moonshot" in model_lower:
            provider = "kimi"
        elif "grok" in model_lower:
            provider = "grok"
        elif "gemini" in model_lower:
            provider = "gemini"
        else:
            provider = "ollama"
    else:
        provider = "ollama"

    # Route to provider
    if provider == "kimi":
        return await _completion_kimi(prompt, system, max_tokens, temperature)
    elif provider == "grok":
        return await _completion_grok(prompt, system, max_tokens, temperature)
    elif provider == "gemini":
        return await _completion_gemini(prompt, system, max_tokens, temperature)
    else:
        return await _completion_ollama(prompt, system, model or PRIMARY_MODEL, max_tokens, temperature)


async def _completion_ollama(
    prompt: str,
    system: str = None,
    model: str = None,
    max_tokens: int = None,
    temperature: float = 0.7,
) -> str:
    """
    Get completion from Ollama.

    LOCAL MODELS HAVE NO RESTRAINTS - full context, max tokens.
    """
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama not available")
        return ""

    # Local models get max tokens - no limits
    actual_max_tokens = max_tokens or LOCAL_MODEL_MAX_TOKENS

    try:
        messages = []
        # Only add system if explicitly provided - no default restrictions
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Run in thread since ollama.chat is sync
        # NO RESTRAINTS - let local models fully express
        response = await asyncio.to_thread(
            ollama.chat,
            model=model or PRIMARY_MODEL,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": actual_max_tokens,
                "num_ctx": 32768,  # Max context window
            }
        )

        # Extract content (handle deepseek-r1 thinking format)
        msg = response.get("message", {})
        content = msg.get("content", "")

        # If content has thinking tags, extract final answer
        if "<think>" in content and "</think>" in content:
            think_end = content.rfind("</think>")
            if think_end != -1:
                content = content[think_end + 8:].strip()

        return content

    except Exception as e:
        logger.error(f"Ollama completion error: {e}")
        return ""


async def _completion_kimi(
    prompt: str,
    system: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Get completion from Kimi (Moonshot AI)."""
    try:
        from Cosmos.integration.external.kimi import get_kimi_provider

        provider = get_kimi_provider()
        if provider is None:
            logger.warning("Kimi provider not available")
            return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)

        if not await provider.connect():
            return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)

        result = await provider.chat(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return result.get("content", "")

    except Exception as e:
        logger.error(f"Kimi completion error: {e}")
        return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)


async def _completion_grok(
    prompt: str,
    system: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Get completion from Grok (xAI)."""
    try:
        from Cosmos.integration.external.grok import GrokProvider

        grok = GrokProvider()
        if not await grok.connect():
            logger.warning("Grok not available")
            return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)

        result = await grok.chat(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return result.get("content", "")

    except Exception as e:
        logger.error(f"Grok completion error: {e}")
        return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)


async def _completion_gemini(
    prompt: str,
    system: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Get completion from Gemini (Google AI)."""
    try:
        from Cosmos.integration.external.gemini import get_gemini_provider

        provider = get_gemini_provider()
        if provider is None:
            logger.warning("Gemini provider not available")
            return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)

        result = await provider.chat(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return result.get("content", "")

    except Exception as e:
        logger.error(f"Gemini completion error: {e}")
        return await _completion_ollama(prompt, system, max_tokens=max_tokens, temperature=temperature)


# Convenience functions for specific providers
async def ollama_completion(prompt: str, system: str = None, model: str = None) -> str:
    """Direct Ollama completion."""
    return await _completion_ollama(prompt, system, model)


async def kimi_completion(prompt: str, system: str = None) -> str:
    """Direct Kimi completion."""
    return await _completion_kimi(prompt, system)


async def grok_completion(prompt: str, system: str = None) -> str:
    """Direct Grok completion."""
    return await _completion_grok(prompt, system)


async def gemini_completion(prompt: str, system: str = None) -> str:
    """Direct Gemini completion."""
    return await _completion_gemini(prompt, system)
