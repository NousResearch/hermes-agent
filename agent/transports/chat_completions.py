"""OpenAI Chat Completions transport.

Handles the default api_mode ('chat_completions') used by ~16 OpenAI-compatible
providers (OpenRouter, Nous, NVIDIA, Qwen, Ollama, DeepSeek, xAI, Kimi, etc.).

Messages and tools are already in OpenAI format — convert_messages and
convert_tools are near-identity.  The complexity lives in build_kwargs
which has provider-specific conditionals for max_tokens defaults,
reasoning configuration, temperature handling, and extra_body assembly.
"""

import copy
from typing import Any, Dict, List, Optional

from agent.lmstudio_reasoning import resolve_lmstudio_effort
from agent.moonshot_schema import is_moonshot_model, sanitize_moonshot_tools
from agent.prompt_builder import DEVELOPER_ROLE_MODELS
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


def _build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Translate Hermes/OpenRouter-style reasoning config to Gemini thinkingConfig."""
    if reasoning_config is None or not isinstance(reasoning_config, dict):
        return None

    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("google/"):
        normalized_model = normalized_model.split("/", 1)[1]

    # ``thinking_config`` is a Gemini-only request parameter. The same
    # ``gemini`` provider also serves Gemma (and historically PaLM/Bard);
    # those reject the field with HTTP 400 "Unknown name 'thinking_config':
    # Cannot find field" — including the polite ``{"includeThoughts": False}``
    # form. Omit the field entirely on non-Gemini models. (#17426)
    if not normalized_model.startswith("gemini"):
        return None

    if reasoning_config.get("enabled") is False:
        # Gemini can hide thought parts even when internal thinking still
        # happens; omit thinkingLevel to avoid model-specific validation quirks.
        return {"includeThoughts": False}

    effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if effort == "none":
        return {"includeThoughts": False}

    thinking_config: Dict[str, Any] = {"includeThoughts": True}

    # Gemini 2.5 accepts thinkingBudget; don't guess a budget from Hermes'
    # coarse effort levels. ``includeThoughts`` alone is enough to surface
    # thought parts without risking request validation errors.
    if normalized_model.startswith("gemini-2.5-"):
        return thinking_config

    if effort not in {"minimal", "low", "medium", "high", "xhigh"}:
        effort = "medium"

    # Gemini 3 Flash documents low/medium/high thinking levels; Gemini 3 Pro
    # is stricter (low/high). Clamp Hermes' wider effort set to what each
    # family accepts so we never forward an undocumented level verbatim.
    if normalized_model.startswith(("gemini-3", "gemini-3.1")):
        if "flash" in normalized_model:
            if effort in {"minimal", "low"}:
                thinking_config["thinkingLevel"] = "low"
            elif effort in {"high", "xhigh"}:
                thinking_config["thinkingLevel"] = "high"
            else:
                thinking_config["thinkingLevel"] = "medium"
        elif "pro" in normalized_model:
            thinking_config["thinkingLevel"] = (
                "high" if effort in {"high", "xhigh"} else "low"
            )

    return thinking_config


def _snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Convert Gemini thinking config keys to the OpenAI-compat field names."""
    if not isinstance(config, dict) or not config:
        return None

    translated: Dict[str, Any] = {}
    if isinstance(config.get("includeThoughts"), bool):
        translated["include_thoughts"] = config["includeThoughts"]
    if isinstance(config.get("thinkingLevel"), str) and config["thinkingLevel"].strip():
        translated["thinking_level"] = config["thinkingLevel"].strip().lower()
    if isinstance(config.get("thinkingBudget"), (int, float)):
        translated["thinking_budget"] = int(config["thinkingBudget"])
    return translated or None


def _is_gemini_openai_compat_base_url(base_url: Any) -> bool:
    normalized = str(base_url or "").strip().rstrip("/").lower()
    if not normalized:
        return False
    if "generativelanguage.googleapis.com" not in normalized:
        return False
    return normalized.endswith("/openai")


class ChatCompletionsTransport(ProviderTransport):
    """Transport for api_mode='chat_completions'.

    The default path for OpenAI-compatible providers.
    """

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Messages are already in OpenAI format — sanitize Codex leaks only.

        Strips Codex Responses API fields (``codex_reasoning_items`` /
        ``codex_message_items`` on the message, ``call_id``/``response_item_id``
        on tool_calls) that strict chat-completions providers reject with 400/422.
        """
        needs_sanitize = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg or "codex_message_items" in msg:
                needs_sanitize = True
                break
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and ("call_id" in tc or "response_item_id" in tc):
                        needs_sanitize = True
                        break
                if needs_sanitize:
                    break

        if not needs_sanitize:
            return messages

        sanitized = copy.deepcopy(messages)
        for msg in sanitized:
            if not isinstance(msg, dict):
                continue
            msg.pop("codex_reasoning_items", None)
            msg.pop("codex_message_items", None)
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc.pop("call_id", None)
                        tc.pop("response_item_id", None)
        return sanitized

    def convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build chat.completions.create() kwargs.

        This is the most complex transport method — it handles ~16 providers
        via params rather than subclasses.

        params:
            timeout: float — API call timeout
            max_tokens: int | None — user-configured max tokens
            ephemeral_max_output_tokens: int | None — one-shot override (error recovery)
            max_tokens_param_fn: callable — returns {max_tokens: N} or {max_completion_tokens: N}
            reasoning_config: dict | None
            request_overrides: dict | None
            session_id: str | None
            qwen_session_metadata: dict | None — {sessionId, promptId} precomputed
            model_lower: str — lowercase model name for pattern matching
            # Provider detection flags (all optional, default False)
            is_openrouter: bool
            is_nous: bool
            is_qwen_portal: bool
            is_github_models: bool
            is_nvidia_nim: bool
            is_kimi: bool
            is_lmstudio: bool
            is_custom_provider: bool
            ollama_num_ctx: int | None
            # Provider routing
            provider_preferences: dict | None
            # Qwen-specific
            qwen_prepare_fn: callable | None — runs AFTER codex sanitization
            qwen_prepare_inplace_fn: callable | None — in-place variant for deepcopied lists
            # Temperature
            fixed_temperature: Any — from _fixed_temperature_for_model()
            omit_temperature: bool
            # Reasoning
            supports_reasoning: bool
            github_reasoning_extra: dict | None
            lmstudio_reasoning_options: list[str] | None  # raw allowed_options from /api/v1/models
            # Claude on OpenRouter/Nous max output
            anthropic_max_output: int | None
            # Extra
            extra_body_additions: dict | None — pre-built extra_body entries
        """
        # Codex sanitization: drop reasoning_items / call_id / response_item_id
        sanitized = self.convert_messages(messages)

        # Qwen portal prep AFTER codex sanitization.  If sanitize already
        # deepcopied, reuse that copy via the in-place variant to avoid a
        # second deepcopy.
        is_qwen = params.get("is_qwen_portal", False)
        if is_qwen:
            qwen_prep = params.get("qwen_prepare_fn")
            qwen_prep_inplace = params.get("qwen_prepare_inplace_fn")
            if sanitized is messages:
                if qwen_prep is not None:
                    sanitized = qwen_prep(sanitized)
            else:
                # Already deepcopied — transform in place
                if qwen_prep_inplace is not None:
                    qwen_prep_inplace(sanitized)
                elif qwen_prep is not None:
                    sanitized = qwen_prep(sanitized)

        # Developer role swap for GPT-5/Codex models
        model_lower = params.get("model_lower", (model or "").lower())
        if (
            sanitized
            and isinstance(sanitized[0], dict)
            and sanitized[0].get("role") == "system"
            and any(p in model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            sanitized = list(sanitized)
            sanitized[0] = {**sanitized[0], "role": "developer"}

        api_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": sanitized,
        }

        timeout = params.get("timeout")
        if timeout is not None:
            api_kwargs["timeout"] = timeout

        # Temperature
        fixed_temp = params.get("fixed_temperature")
        omit_temp = params.get("omit_temperature", False)
        if omit_temp:
            api_kwargs.pop("temperature", None)
        elif fixed_temp is not None:
            api_kwargs["temperature"] = fixed_temp

        # Qwen metadata (caller precomputes {sessionId, promptId})
        qwen_meta = params.get("qwen_session_metadata")
        if qwen_meta and is_qwen:
            api_kwargs["metadata"] = qwen_meta

        # Tools
        if tools:
            # Moonshot/Kimi uses a stricter flavored JSON Schema.  Rewriting
            # tool parameters here keeps aggregator routes (Nous, OpenRouter,
            # etc.) compatible, in addition to direct moonshot.ai endpoints.
            if is_moonshot_model(model):
                tools = sanitize_moonshot_tools(tools)
            api_kwargs["tools"] = tools

        # max_tokens resolution — priority: ephemeral > user > provider default
        max_tokens_fn = params.get("max_tokens_param_fn")
        ephemeral = params.get("ephemeral_max_output_tokens")
        max_tokens = params.get("max_tokens")
        anthropic_max_out = params.get("anthropic_max_output")
        is_nvidia_nim = params.get("is_nvidia_nim", False)
        is_kimi = params.get("is_kimi", False)
        is_tokenhub = params.get("is_tokenhub", False)
        reasoning_config = params.get("reasoning_config")

        if ephemeral is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(ephemeral))
        elif max_tokens is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(max_tokens))
        elif is_nvidia_nim and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(16384))
        elif is_qwen and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(65536))
        elif is_kimi and max_tokens_fn:
            # Kimi/Moonshot: 32000 matches Kimi CLI's default
            api_kwargs.update(max_tokens_fn(32000))
        elif anthropic_max_out is not None:
            api_kwargs["max_tokens"] = anthropic_max_out

        # Kimi: top-level reasoning_effort (unless thinking disabled)
        if is_kimi:
            _kimi_thinking_off = bool(
                reasoning_config
                and isinstance(reasoning_config, dict)
                and reasoning_config.get("enabled") is False
            )
            if not _kimi_thinking_off:
                _kimi_effort = "medium"
                if reasoning_config and isinstance(reasoning_config, dict):
                    _e = (reasoning_config.get("effort") or "").strip().lower()
                    if _e in ("low", "medium", "high"):
                        _kimi_effort = _e
                api_kwargs["reasoning_effort"] = _kimi_effort

        # Tencent TokenHub: top-level reasoning_effort (unless thinking disabled)
        if is_tokenhub:
            _tokenhub_thinking_off = bool(
                reasoning_config
                and isinstance(reasoning_config, dict)
                and reasoning_config.get("enabled") is False
            )
            if not _tokenhub_thinking_off:
                _tokenhub_effort = "high"
                if reasoning_config and isinstance(reasoning_config, dict):
                    _e = (reasoning_config.get("effort") or "").strip().lower()
                    if _e in ("low", "medium", "high"):
                        _tokenhub_effort = _e
                api_kwargs["reasoning_effort"] = _tokenhub_effort

        # LM Studio: top-level reasoning_effort. Only emit when the model
        # declares reasoning support via /api/v1/models capabilities (gated
        # upstream by params["supports_reasoning"]). resolve_lmstudio_effort
        # is shared with run_agent's summary path so both stay in sync.
        if params.get("is_lmstudio", False) and params.get("supports_reasoning", False):
            _lm_effort = resolve_lmstudio_effort(
                reasoning_config,
                params.get("lmstudio_reasoning_options"),
            )
            if _lm_effort is not None:
                api_kwargs["reasoning_effort"] = _lm_effort

        # extra_body assembly
        extra_body: Dict[str, Any] = {}

        is_openrouter = params.get("is_openrouter", False)
        is_nous = params.get("is_nous", False)
        is_github_models = params.get("is_github_models", False)
        provider_name = str(params.get("provider_name") or "").strip().lower()
        base_url = params.get("base_url")

        provider_prefs = params.get("provider_preferences")
        if provider_prefs and is_openrouter:
            extra_body["provider"] = provider_prefs

        # Kimi extra_body.thinking
        if is_kimi:
            _kimi_thinking_enabled = True
            if reasoning_config and isinstance(reasoning_config, dict):
                if reasoning_config.get("enabled") is False:
                    _kimi_thinking_enabled = False
            extra_body["thinking"] = {
                "type": "enabled" if _kimi_thinking_enabled else "disabled",
            }

        # Reasoning. LM Studio is handled above via top-level reasoning_effort,
        # so skip emitting extra_body.reasoning for it.
        if params.get("supports_reasoning", False) and not params.get("is_lmstudio", False):
            if is_github_models:
                gh_reasoning = params.get("github_reasoning_extra")
                if gh_reasoning is not None:
                    extra_body["reasoning"] = gh_reasoning
            else:
                if reasoning_config is not None:
                    rc = dict(reasoning_config)
                    if is_nous and rc.get("enabled") is False:
                        pass  # omit for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        if is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        # Ollama num_ctx
        ollama_ctx = params.get("ollama_num_ctx")
        if ollama_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_ctx
            extra_body["options"] = options

        # Ollama/custom think=false
        if params.get("is_custom_provider", False):
            if reasoning_config and isinstance(reasoning_config, dict):
                _effort = (reasoning_config.get("effort") or "").strip().lower()
                _enabled = reasoning_config.get("enabled", True)
                if _effort == "none" or _enabled is False:
                    extra_body["think"] = False

        if is_qwen:
            extra_body["vl_high_resolution_images"] = True

        if provider_name == "gemini":
            raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if _is_gemini_openai_compat_base_url(base_url):
                thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
                if thinking_config:
                    openai_compat_extra = extra_body.get("extra_body", {})
                    google_extra = openai_compat_extra.get("google", {})
                    google_extra["thinking_config"] = thinking_config
                    openai_compat_extra["google"] = google_extra
                    extra_body["extra_body"] = openai_compat_extra
            elif raw_thinking_config:
                extra_body["thinking_config"] = raw_thinking_config
        elif provider_name == "google-gemini-cli":
            thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if thinking_config:
                extra_body["thinking_config"] = thinking_config

        # Merge any pre-built extra_body additions
        additions = params.get("extra_body_additions")
        if additions:
            extra_body.update(additions)

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # Request overrides last (service_tier etc.)
        overrides = params.get("request_overrides")
        if overrides:
            api_kwargs.update(overrides)

        return api_kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize OpenAI ChatCompletion to NormalizedResponse.

        For chat_completions, this is near-identity — the response is already
        in OpenAI format.  extra_content on tool_calls (Gemini thought_signature)
        is preserved via ToolCall.provider_data.  reasoning_details (OpenRouter
        unified format) and reasoning_content (DeepSeek/Moonshot) are also
        preserved for downstream replay.
        """
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                # Preserve provider-specific extras on the tool call.
                # Gemini 3 thinking models attach extra_content with
                # thought_signature — without replay on the next turn the API
                # rejects the request with 400.
                tc_provider_data: Dict[str, Any] = {}
                extra = getattr(tc, "extra_content", None)
                if extra is None and hasattr(tc, "model_extra"):
                    extra = (tc.model_extra or {}).get("extra_content")
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        try:
                            extra = extra.model_dump()
                        except Exception:
                            pass
                    tc_provider_data["extra_content"] = extra
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    provider_data=tc_provider_data or None,
                ))

        usage = None
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        # Preserve reasoning fields separately.  DeepSeek/Moonshot use
        # ``reasoning_content``; others use ``reasoning``.  Downstream code
        # (_extract_reasoning, thinking-prefill retry) reads both distinctly,
        # so keep them apart in provider_data rather than merging.
        reasoning = getattr(msg, "reasoning", None)
        reasoning_content = getattr(msg, "reasoning_content", None)

        provider_data: Dict[str, Any] = {}
        if reasoning_content:
            provider_data["reasoning_content"] = reasoning_content
        rd = getattr(msg, "reasoning_details", None)
        if rd:
            provider_data["reasoning_details"] = rd

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check that response has valid choices."""
        if response is None:
            return False
        if not hasattr(response, "choices") or response.choices is None:
            return False
        if not response.choices:
            return False
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract OpenRouter/OpenAI cache stats from prompt_tokens_details."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            return None
        cached = getattr(details, "cached_tokens", 0) or 0
        written = getattr(details, "cache_write_tokens", 0) or 0
        if cached or written:
            return {"cached_tokens": cached, "creation_tokens": written}
        return None


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("chat_completions", ChatCompletionsTransport)
