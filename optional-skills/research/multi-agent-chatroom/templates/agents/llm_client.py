# agents/llm_client.py
"""Unified LLM API client for DeepSeek, OpenAI, and Anthropic."""

import os
import httpx
from typing import Optional


class LLMClient:
    """Supports DeepSeek, OpenAI, and Anthropic providers."""

    def __init__(self, provider: str, model: str, api_key: str = None,
                 temperature: float = 0.3, max_tokens: int = 16384):
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._resolve_key(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _resolve_key(self, provider: str) -> str:
        """Resolve API key from environment variables."""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")

        # Check environment first
        key = os.getenv(env_var, "")
        if key:
            return key

        # Check common env file locations
        for env_path in [
            os.path.expanduser("~/.env"),
            os.path.expanduser("~/.env"),
        ]:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(env_var + "="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")

        return ""

    async def chat(self, system_prompt: str, user_message: str,
                   temperature: float = None, max_tokens: int = None) -> str:
        """Send a chat completion and return response text."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            return await self._chat_anthropic(system_prompt, user_message, temp, max_tok)
        else:
            # OpenAI-compatible: openai, deepseek
            return await self._chat_openai_compatible(system_prompt, user_message, temp, max_tok)

    async def _chat_anthropic(self, system: str, user: str, temp: float, max_tok: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tok,
            "temperature": temp,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    async def _chat_openai_compatible(self, system: str, user: str,
                                       temp: float, max_tok: int) -> str:
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
        }
        base_url = base_urls.get(self.provider, "https://api.openai.com/v1")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tok,
            "temperature": temp,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers, json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # Convenience method for multi-turn
    async def chat_multi(self, system_prompt: str, messages: list[dict],
                         temperature: float = None, max_tokens: int = None) -> str:
        """Send multi-turn messages and return response."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": max_tok,
                "temperature": temp,
            }
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers, json=body,
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]
        else:
            base_urls = {"openai": "https://api.openai.com/v1",
                         "deepseek": "https://api.deepseek.com/v1"}
            base_url = base_urls.get(self.provider, "https://api.openai.com/v1")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": max_tok,
                "temperature": temp,
            }
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers, json=body,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
