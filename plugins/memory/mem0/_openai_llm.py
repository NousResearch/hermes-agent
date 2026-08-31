"""OpenAI-only LLM adapter for Mem0 OSS mode."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.llms.openai import OpenAILLM


class DirectOpenAILLM(OpenAILLM):
    """Use OpenAI credentials and requests regardless of router environment."""

    def __init__(
        self,
        config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None,
    ):
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            config = OpenAIConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                reasoning_effort=getattr(config, "reasoning_effort", None),
                http_client_proxies=config.http_client_proxies,
                is_reasoning_model=getattr(config, "is_reasoning_model", None),
            )

        if not config.model:
            config.model = "gpt-5-mini"

        # Older, partial, and manually edited configs may predate the setup
        # marker. Keep the exact default model safe at runtime without
        # overriding an explicit user choice or changing the persisted config.
        if config.model == "gpt-5-mini" and config.is_reasoning_model is None:
            config.is_reasoning_model = True

        # Bypass OpenAILLM.__init__: it intentionally selects OpenRouter when
        # OPENROUTER_API_KEY is present. LLMBase still owns validation and
        # supported-parameter filtering for parity with Mem0's implementation.
        LLMBase.__init__(self, config)

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for the Hermes Mem0 OSS provider"
            )

        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})

        # OpenRouter-only fields are deliberately not added here. ``store`` is
        # opt-in so OpenAI-compatible endpoints do not receive unknown fields.
        if self.config.store is not None:
            params["store"] = self.config.store

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        parsed_response = self._parse_response(response, tools)
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception:
                logging.error("Error running Mem0 OpenAI response callback")
        return parsed_response
