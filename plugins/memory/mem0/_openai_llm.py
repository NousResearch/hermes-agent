"""OpenAI-only LLM adapter for Mem0 OSS mode."""

from __future__ import annotations

import logging
import os
import time as _time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.llms.openai import OpenAILLM

# BaseLlmConfig fields copied into OpenAIConfig; the last two may be absent on older mem0.
_COPIED_FIELDS = ("model", "temperature", "api_key", "max_tokens", "top_p", "top_k", "enable_vision", "vision_details", "http_client_proxies")
_OPTIONAL_FIELDS = ("reasoning_effort", "is_reasoning_model")

# OpenRouter-style routers report 402 (in-flight or credit budget exhausted) plus
# transient 429/5xx; retry with backoff instead of failing the memory append.
# Some 402s remedy with "wait for in-flight requests to settle" — honor
# Retry-After (capped) exactly as the provider asks.
_RETRYABLE_STATUS_CODES = {402, 408, 409, 429, 500, 502, 503, 504}
_MAX_RETRY_ATTEMPTS = 3
_MAX_RETRY_WAIT_SECS = 120


def _retry_wait_seconds(exc: Exception) -> float:
    """Extract Retry-After (seconds or HTTP-date) from an OpenAI status error."""
    try:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", {}) if resp is not None else {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if not raw:
            return 5.0
        try:
            return max(1.0, float(raw))
        except ValueError:
            parsed = datetime.strptime(str(raw), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
            return max(1.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 5.0


class DirectOpenAILLM(OpenAILLM):
    """Use OpenAI credentials and requests regardless of router environment."""

    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            fields = {k: getattr(config, k) for k in _COPIED_FIELDS}
            fields.update({k: getattr(config, k, None) for k in _OPTIONAL_FIELDS})
            config = OpenAIConfig(**fields)
        if not config.model:
            config.model = "gpt-5-mini"
        # Configs predating the setup marker: keep the default model reasoning-safe
        # without overriding an explicit user choice.
        if config.model == "gpt-5-mini" and config.is_reasoning_model is None:
            config.is_reasoning_model = True
        # Bypass OpenAILLM.__init__ (it picks OpenRouter when OPENROUTER_API_KEY is
        # set); LLMBase still owns validation and supported-parameter filtering.
        LLMBase.__init__(self, config)
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required for the Hermes Mem0 OSS provider")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")

    def generate_response(self, messages: List[Dict[str, str]], response_format=None, tools: Optional[List[Dict]] = None, tool_choice: str = "auto", **kwargs):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})
        # No OpenRouter-only fields; ``store`` is opt-in so OpenAI-compatible endpoints never receive unknown fields.
        if self.config.store is not None:
            params["store"] = self.config.store
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"], params["tool_choice"] = tools, tool_choice
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.client.chat.completions.create(**params)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                is_retryable = status in _RETRYABLE_STATUS_CODES or status is None  # network-level failure
                if not is_retryable or attempt > _MAX_RETRY_ATTEMPTS:
                    raise
                wait = min(_retry_wait_seconds(exc), _MAX_RETRY_WAIT_SECS)
                logging.warning(
                    "Mem0 OpenAI call failed (attempt %d/%d, status=%s); retrying in %.0fs",
                    attempt, _MAX_RETRY_ATTEMPTS, status, wait,
                )
                _time.sleep(wait)
        parsed_response = self._parse_response(response, tools)
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception:
                logging.error("Error running Mem0 OpenAI response callback")
        return parsed_response
