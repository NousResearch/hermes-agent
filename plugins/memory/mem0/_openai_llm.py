"""OpenAI-only LLM adapter for Mem0 OSS mode."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union, cast

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.llms.openai import OpenAILLM

# BaseLlmConfig fields copied into OpenAIConfig; the last two may be absent on older mem0.
_COPIED_FIELDS = ("model", "temperature", "api_key", "max_tokens", "top_p", "top_k", "enable_vision", "vision_details", "http_client_proxies")
_OPTIONAL_FIELDS = ("reasoning_effort", "is_reasoning_model")


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
        # OPENAI_API_KEY / OPENAI_BASE_URL are profile credentials: read them through the secret
        # scope, never raw os.environ, or a multiplexed secondary's memory extraction runs on the
        # default profile's OpenAI account (and its proxy).
        from agent.secret_scope import get_secret
        api_key = self.config.api_key or get_secret("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key is required for the Hermes Mem0 OSS provider")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=self.config.openai_base_url or get_secret("OPENAI_BASE_URL", "") or "https://api.openai.com/v1")

    @staticmethod
    def _fallback_settings() -> dict:
        """Read the optional extraction fallback from the active profile's Mem0 config."""
        from hermes_constants import get_hermes_home
        from utils import read_json_or_empty

        config = read_json_or_empty(get_hermes_home() / "mem0.json")
        fallback = (((config.get("oss") or {}).get("llm") or {}).get("fallback"))
        if not isinstance(fallback, dict) or fallback.get("enabled") is not True:
            return {}
        return fallback

    @staticmethod
    def _valid_extraction_json(content: object, response_format: object) -> bool:
        """Mem0's JSON extraction response must be an object with a memory list."""
        if not isinstance(response_format, dict) or response_format.get("type") != "json_object":
            return True
        if not isinstance(content, str) or not content.strip():
            return False
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return False
        return isinstance(parsed, dict) and isinstance(parsed.get("memory"), list)

    @staticmethod
    def _transient_codex_error(exc: Exception) -> bool:
        """Limit automatic failover to retryable HTTP and transport failures."""
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 429} or status >= 500
        return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError", "TimeoutException", "ReadTimeout",
        }

    @staticmethod
    def _is_extraction_request(messages: List[Dict[str, str]], response_format: object,
                               tools: Optional[List[Dict]]) -> bool:
        """Only a tool-free Mem0 JSON extraction may use the Codex fallback."""
        if tools or not isinstance(response_format, dict) or response_format.get("type") != "json_object":
            return False
        return not any(
            isinstance(message, dict) and (message.get("role") == "tool" or message.get("tool_calls"))
            for message in messages
        )

    def _codex_fallback(self, messages: List[Dict[str, str]], response_format: object, settings: dict) -> str:
        """Run Mem0's JSON extraction using this profile's existing Codex OAuth session."""
        model = str(settings.get("model") or "").strip()
        if not model:
            raise RuntimeError("Mem0 extraction fallback is enabled but has no model")
        if not isinstance(response_format, dict) or response_format.get("type") != "json_object":
            raise RuntimeError("Codex fallback is restricted to Mem0 JSON extraction requests")

        from hermes_cli.auth import resolve_codex_runtime_credentials
        from agent.codex_headers import codex_cloudflare_headers
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        from agent.codex_runtime import _consume_codex_event_stream
        from agent.auxiliary_client import _parse_codex_final_response
        from openai import OpenAI

        # A fallback must not refresh or write auth/config state as a side effect of memory sync.
        credentials = resolve_codex_runtime_credentials(read_only=True, refresh_if_expiring=False)
        token, base_url = credentials.get("api_key"), credentials.get("base_url")
        if not token or not base_url:
            raise RuntimeError("OpenAI Codex OAuth credentials are unavailable for Mem0 fallback")

        instructions = "\n\n".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") in {"system", "developer"}
        ).strip()
        instructions = (
            instructions
            + "\n\nReturn only valid JSON with a top-level `memory` array. "
            + "Each extracted fact must be an object with a non-empty `text` field; "
            + 'if there are no durable facts, return `{"memory": []}`. Do not use Markdown fences.'
        ).strip()
        conversation = [
            message for message in messages
            if message.get("role") not in {"system", "developer"}
        ]
        input_items = _chat_messages_to_responses_input(
            conversation,
            current_issuer_kind="codex_backend",
            current_issuer_model=model,
            native_compaction_eligible=False,
        )
        if not input_items:
            raise RuntimeError("Mem0 extraction fallback received no user input")

        try:
            timeout = float(settings.get("timeout_seconds") or 90)
        except (TypeError, ValueError):
            timeout = 90.0
        client = OpenAI(
            api_key=token,
            base_url=base_url,
            default_headers=codex_cloudflare_headers(token, base_url=base_url),
            timeout=timeout,
        )
        try:
            last_error = None
            for attempt in range(2):
                stream = None
                try:
                    stream = client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=cast(Any, input_items),
                        text={"format": {"type": "json_object"}},
                        store=False,
                        stream=True,
                    )
                    # Some Codex-compatible hosts accept stream=True but return a completed
                    # Responses object instead of an iterable event stream.
                    final = stream if hasattr(stream, "output") else _consume_codex_event_stream(stream, model=model)
                    if getattr(final, "status", None) != "completed":
                        raise RuntimeError(
                            f"Codex fallback ended with status {getattr(final, 'status', None)!r}"
                        )
                    text_parts, tool_calls, _usage = _parse_codex_final_response(final)
                    if tool_calls:
                        raise RuntimeError("Codex fallback unexpectedly returned tool calls")
                    content = "".join(text_parts).strip()
                    if self._valid_extraction_json(content, response_format):
                        return content
                    raise ValueError("Codex fallback returned invalid Mem0 extraction JSON")
                except Exception as exc:
                    last_error = exc
                    retry_invalid_json = isinstance(exc, ValueError) and "invalid Mem0 extraction JSON" in str(exc)
                    if attempt or not (retry_invalid_json or self._transient_codex_error(exc)):
                        raise
                    logging.warning("Mem0 Codex fallback retrying once (error=%s)", type(exc).__name__)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
            raise last_error or RuntimeError("Mem0 Codex fallback failed")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

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
        is_json_extraction = self._is_extraction_request(messages, response_format, tools)
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            settings = self._fallback_settings() if is_json_extraction else {}
            if not settings or not self._transient_codex_error(exc):
                raise
            return self._codex_fallback(messages, response_format, settings)
        parsed_response = self._parse_response(response, tools)
        content = parsed_response.get("content") if isinstance(parsed_response, dict) else parsed_response
        if is_json_extraction and not self._valid_extraction_json(content, response_format):
            settings = self._fallback_settings()
            if settings:
                return self._codex_fallback(messages, response_format, settings)
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception:
                logging.error("Error running Mem0 OpenAI response callback")
        return parsed_response
