"""OpenAI Chat Completions transport (default api_mode for OpenAI-compatible providers).

Messages/tools are already OpenAI-shaped, so convert_* are near-identity; the
provider-specific work lives in build_kwargs (max_tokens, reasoning, extra_body).
"""

import json
from typing import Any
from urllib.parse import urlparse

from agent.lmstudio_reasoning import resolve_lmstudio_effort
from agent.reasoning_effort import (
    KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES, OPENAI_COMPAT_WIRE_EFFORTS, TOKENHUB_EFFORTS, clamp_effort,
    kimi_supported_efforts, requested_effort,
)
from agent.moonshot_schema import is_moonshot_model, sanitize_moonshot_tools
from agent.prompt_builder import DEVELOPER_ROLE_MODELS
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

# xAI reserves ``tool_search`` for its server-side tool (HTTP 400 on client
# declarations); aliased on the wire, mapped back in normalize_response.
# xAI's chat-completions API reserves the function name ``tool_search`` for its own server-side tool and
# rejects any request declaring a client function with that name (HTTP 400 "The function name tool_search is
# reserved for the tool_search tool", #95003). The Tool Search bridge (tools/tool_search.py) assembles its
# client-side discovery tool under the same literal name for every provider, so Grok providers are unusable
# whenever the bridge is active. Mirror the web_search treatment in transports/codex.py
# (_rename_client_web_search_for_xai): alias the wire declaration and map the alias back in
# normalize_response. The alias value matches _CODEX_TOOL_SEARCH_ALIAS from the Codex-side fix for the same
# reserved-name class (#83122) so the two transports stay consistent.
_XAI_TOOL_SEARCH_ALIAS = "hermes_tool_search"

# Persistence-only / cross-transport message keys that strict OpenAI-compatible
# providers reject with HTTP 400 ("Extra inputs are not permitted").
_STRIP_MSG_KEYS = (
    "codex_reasoning_items", "codex_message_items", "tool_name", "effect_disposition", "timestamp",
    "platform_message_id", "api_content", "anthropic_content_blocks", "bedrock_content_blocks",
)
_STRIP_TC_KEYS = ("call_id", "response_item_id")
_HIGH_EFFORTS = {"high", "xhigh", "max", "ultra"}


def _rename_tool_search_bridge_for_xai(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Alias the client ``tool_search`` declaration for xAI; returns ``(tools, {alias: "tool_search"})``.

    If a real tool already holds ``hermes_tool_search``, the bridge takes a ``_2``/``_3`` suffix.
    """
    from agent.transports.codex import _alias_reserved_tools

    return _alias_reserved_tools(
        tools, ("tool_search",), name_of=lambda t: (t.get("function") or {}).get("name"),
        rename=lambda t, alias: {**t, "function": {**t["function"], "name": alias}},
    )


def _static_prompt_instructions(messages: list[dict[str, Any]]) -> str:
    """Stable leading system/developer prefix used for cache routing (later messages are conversation state)."""
    first = messages[0] if messages and isinstance(messages[0], dict) else {}
    if first.get("role") not in {"system", "developer"}:
        return ""
    content = first.get("content")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(content or "")


def _add_prompt_cache_key(
    api_kwargs: dict[str, Any], *, messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None, supports_prompt_cache_key: bool,
    session_id: str | None = None, cache_scope_id: str | None = None,
) -> None:
    """Add a content-addressed ``prompt_cache_key`` only for a capable endpoint.

    ``cache_scope_id`` (compression-lineage root) beats ``session_id`` so the key
    survives compression rotation. A caller-supplied key is authoritative but is
    bounded to OpenAI's 64-char cap in place. Shares the Responses transport's hash
    so equivalent prefixes hit one bucket across modes.

    ``cache_scope_id``, when provided, is the rotation-stable logical scope (compression-lineage root —
    agent/prompt_cache_scope.py) and takes precedence over the physical ``session_id`` so the key survives
    context-compression session rotation (#79017).
    """
    # Stable prompt-cache routing for the Codex/Responses aux path, mirroring the main transport
    # (agent/transports/codex.py::build_kwargs, which sets prompt_cache_key =
    # _content_cache_key(instructions, tools)). Without this, MoA acting-aggregator and other auxiliary
    # Responses calls stay cache-cold while the main Responses transport is warm (issue #53735). The key is
    # content-addressed from the static prefix (instructions + tool schemas) so it stays warm across
    # turns/fires. Guard the top-level field the same way the main transport does: xAI Responses takes the
    # key in extra_body (not top-level) and GitHub/Copilot Responses opts out of cache-key routing entirely
    # — for those hosts, skip it here.
    from agent.transports.codex import (
        _bound_prompt_cache_key_field, _cache_scope_from_session_id, _content_cache_key
    )

    containers = [c for c in (api_kwargs, api_kwargs.get("extra_body")) if isinstance(c, dict) and "prompt_cache_key" in c]
    if containers:
        for c in containers:
            _bound_prompt_cache_key_field(c)
        return
    if not supports_prompt_cache_key:
        return
    cache_key = _content_cache_key(
        _static_prompt_instructions(messages), tools, _cache_scope_from_session_id(cache_scope_id or session_id),
    )
    if cache_key:
        api_kwargs["prompt_cache_key"] = cache_key


def _reasoning_config_for_model(model: str, reasoning_config: dict | None) -> dict | None:
    """Clamp Hermes' extended effort set (``ultra``) to the OpenAI-compat wire vocabulary.

    Hermes' internal effort set extends the wire vocabulary with ``ultra`` (the /reasoning command documents
    none..xhigh|max|ultra). OpenAI- compatible wires — OpenRouter chief among them — accept exactly
    max|xhigh|high|medium|low|minimal|none and reject the extension with HTTP 400 (#89503). Clamp against
    the declared wire vocabulary via the shared policy in ``agent.reasoning_effort``; provider profiles with
    narrower sets clamp again downstream.
    """
    if not isinstance(reasoning_config, dict):
        return reasoning_config
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS) if effort else effort
    return {**reasoning_config, "effort": clamped} if clamped != effort else reasoning_config


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge nested provider request bodies without erasing sibling keys."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged

def _build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Translate Hermes/OpenRouter-style reasoning config to Gemini thinkingConfig."""
    if not isinstance(reasoning_config, dict):
        return None
    normalized_model = (model or "").strip().lower().removeprefix("google/")
    # Gemini-only; Gemma/PaLM on the same provider 400 on the field even as ``{"includeThoughts": False}``.
    # ``thinking_config`` is a Gemini-only request parameter. The same ``gemini`` provider also serves Gemma
    # (and historically PaLM/Bard); those reject the field with HTTP 400 "Unknown name 'thinking_config':
    # Cannot find field" — including the polite ``{"includeThoughts": False}`` form. Omit the field entirely
    # on non-Gemini models. (#17426)
    if not normalized_model.startswith("gemini"):
        return None
    effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if reasoning_config.get("enabled") is False or effort == "none":
        return {"includeThoughts": False}
    thinking_config: dict[str, Any] = {"includeThoughts": True}
    # Gemini 2.5 takes thinkingBudget; don't guess one from coarse effort levels.
    if normalized_model.startswith("gemini-2.5-"):
        return thinking_config
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
        effort = "medium"
    # Gemini 3 Flash documents low/medium/high; Gemini 3 Pro only low/high.
    if normalized_model.startswith(("gemini-3", "gemini-3.1")):
        if "flash" in normalized_model:
            thinking_config["thinkingLevel"] = (
                "low" if effort in {"minimal", "low"} else "high" if effort in _HIGH_EFFORTS else "medium"
            )
        elif "pro" in normalized_model:
            thinking_config["thinkingLevel"] = "high" if effort in _HIGH_EFFORTS else "low"
    return thinking_config


def _snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Convert Gemini thinking config keys to the OpenAI-compat field names."""
    if not isinstance(config, dict) or not config:
        return None
    translated: dict[str, Any] = {}
    include, level, budget = config.get("includeThoughts"), config.get("thinkingLevel"), config.get("thinkingBudget")
    if isinstance(include, bool):
        translated["include_thoughts"] = include
    if isinstance(level, str) and level.strip():
        translated["thinking_level"] = level.strip().lower()
    if isinstance(budget, (int, float)):
        translated["thinking_budget"] = int(budget)
    return translated or None


def _raise_gemini_thinking_max_tokens(model: str, reasoning_config: dict | None, requested: Any) -> Any:
    """Raise Gemini output caps that thinking tokens (billed against max_tokens) would otherwise exhaust."""
    thinking_config = _build_gemini_thinking_config(model, reasoning_config)
    if not thinking_config:
        return requested
    from agent.gemini_native_adapter import _effective_gemini_max_output_tokens

    return _effective_gemini_max_output_tokens(requested, thinking_config)


def _is_gemini_openai_compat_base_url(base_url: Any) -> bool:
    normalized = str(base_url or "").strip().rstrip("/").lower()
    return bool(normalized) and "generativelanguage.googleapis.com" in normalized and normalized.endswith("/openai")


def _is_openai_api_base_url(base_url: Any) -> bool:
    """True only for the exact api.openai.com host (implies ``prompt_cache_key`` support).

    Not a substring match: Azure / strict compat endpoints stay opt-in via ``supports_prompt_cache_key``.
    """
    try:
        return (urlparse(str(base_url or "").strip()).hostname or "").lower() == "api.openai.com"
    except Exception:
        return False


def _model_consumes_thought_signature(model: Any) -> bool:
    """True for Gemini-family targets, which require tool-call ``extra_content`` (thought_signature) replay.

    Every other strict provider rejects it, so it is stripped for non-Gemini targets.
    """
    m = str(model or "").lower()
    return "gemini" in m or "gemma" in m


def _attr_or_model_extra(obj: Any, name: str) -> Any:
    """``obj.<name>``, else the same key from pydantic ``model_extra`` (some SDKs park fields there)."""
    value = getattr(obj, name, None)
    if value is None and hasattr(obj, "model_extra"):
        value = (obj.model_extra if isinstance(obj.model_extra, dict) else {}).get(name)
    return value


def _dump_extra_content(extra: Any) -> Any:
    """Plain-dict form of a pydantic ``extra_content``; older pydantic lacks ``warnings=``, so retry without it."""
    if hasattr(extra, "model_dump"):
        for dump_kwargs in ({"warnings": False}, {}):
            try:
                return extra.model_dump(**dump_kwargs)
            except TypeError:
                continue
            except Exception:
                break
    return extra


def _pareto_score(raw: Any) -> float | None:
    """Coding-score floor for the Pareto router as a float in [0, 1], else None."""
    try:
        score = float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return score if score is not None and 0.0 <= score <= 1.0 else None


def _swap_developer_role(sanitized: list, model_lower: str) -> list:
    """GPT-5/Codex models take a ``developer`` role instead of ``system``."""
    if (
        sanitized and isinstance(sanitized[0], dict) and sanitized[0].get("role") == "system"
        and any(p in model_lower for p in DEVELOPER_ROLE_MODELS)
    ):
        return [{**sanitized[0], "role": "developer"}, *sanitized[1:]]
    return sanitized


def _apply_max_tokens(api_kwargs: dict, model: str, reasoning_config: Any, params: dict, profile_max: Any = None) -> None:
    """Resolve max_tokens — priority: ephemeral > user > profile default > anthropic_max_output."""
    max_tokens_fn = params.get("max_tokens_param_fn")
    for candidate in (params.get("ephemeral_max_output_tokens"), params.get("max_tokens")):
        if candidate is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(_raise_gemini_thinking_max_tokens(model, reasoning_config, candidate)))
            return
    if profile_max and max_tokens_fn:
        api_kwargs.update(max_tokens_fn(_raise_gemini_thinking_max_tokens(model, reasoning_config, profile_max)))
    elif params.get("anthropic_max_output") is not None:
        api_kwargs["max_tokens"] = params["anthropic_max_output"]


def _base_kwargs(model: str, sanitized: list, tools: Any, params: dict, profile: Any = None) -> dict[str, Any]:
    """Shared ``{model, messages[, temperature][, timeout][, tools]}`` scaffold for both build paths.

    ``temperature`` is profile-path only: ``fixed_temperature`` beats the caller's; ``OMIT_TEMPERATURE`` sends none.
    """
    api_kwargs: dict[str, Any] = {"model": model, "messages": sanitized}
    if profile is not None:
        from providers.base import OMIT_TEMPERATURE

        if profile.fixed_temperature is OMIT_TEMPERATURE:
            pass
        elif profile.fixed_temperature is not None:
            api_kwargs["temperature"] = profile.fixed_temperature
        elif params.get("temperature") is not None:
            api_kwargs["temperature"] = params["temperature"]
    if params.get("timeout") is not None:
        api_kwargs["timeout"] = params["timeout"]
    if tools:
        # Moonshot/Kimi uses a stricter JSON Schema flavor; rewriting here also covers aggregator routes.
        api_kwargs["tools"] = sanitize_moonshot_tools(tools) if is_moonshot_model(model) else tools
    return api_kwargs


def _finish_kwargs(api_kwargs: dict[str, Any], sanitized: list, params: dict, *, supports_prompt_cache_key: bool) -> dict[str, Any]:
    """Tail shared by both build paths: content-addressed prompt_cache_key, then return."""
    _add_prompt_cache_key(
        api_kwargs, messages=sanitized, tools=api_kwargs.get("tools"), supports_prompt_cache_key=supports_prompt_cache_key,
        session_id=params.get("session_id"), cache_scope_id=params.get("cache_scope_id"),
    )
    return api_kwargs


def _sanitize_message(msg: Any, strip_extra_content: bool) -> dict | None:
    """Sanitized copy of ``msg``, or None when nothing needs stripping.

    Drops persistence sidecars, ``_``-prefixed scaffolding markers, tool-call ``call_id`` /
    ``response_item_id`` (and ``extra_content`` unless Gemini), and an assistant
    ``tool_calls: []`` / ``null`` (strict providers reject both).
    """
    if not isinstance(msg, dict):
        return None
    strip_keys = [k for k in msg if k in _STRIP_MSG_KEYS or (isinstance(k, str) and k.startswith("_"))]
    out_msg = {k: v for k, v in msg.items() if k not in strip_keys}
    tool_calls = msg.get("tool_calls")
    copied_tool_calls = None
    if msg.get("role") == "assistant" and "tool_calls" in msg and (tool_calls is None or (isinstance(tool_calls, list) and not tool_calls)):
        out_msg.pop("tool_calls", None)
        strip_keys.append("tool_calls")
    elif isinstance(tool_calls, list):
        for tc_idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            keys = [k for k in _STRIP_TC_KEYS if k in tc]
            if strip_extra_content and "extra_content" in tc:
                keys.append("extra_content")
            if keys:
                if copied_tool_calls is None:
                    copied_tool_calls = list(tool_calls)
                copied_tool_calls[tc_idx] = {k: v for k, v in tc.items() if k not in keys}
        if copied_tool_calls is not None:
            out_msg["tool_calls"] = copied_tool_calls
    return out_msg if strip_keys or copied_tool_calls is not None else None


class ChatCompletionsTransport(ProviderTransport):
    """Transport for api_mode='chat_completions'."""

    # ``{alias: original}`` of the most recent request. ``None`` = no request recorded
    # (normalize-only call sites) -> static alias; ``{}`` = no aliases emitted.
    _last_wire_aliases: dict[str, str] | None = None

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Strip internal fields that strict chat-completions providers reject (HTTP 400/422).

        Returns the input list unchanged when nothing needs sanitizing.
        """
        strip_extra_content = not _model_consumes_thought_signature(kwargs.get("model"))
        sanitized_pairs = [(m, _sanitize_message(m, strip_extra_content)) for m in messages]
        if all(s is None for _, s in sanitized_pairs):
            return messages
        return [m if s is None else s for m, s in sanitized_pairs]

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    def build_kwargs(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **params,
    ) -> dict[str, Any]:
        """Build chat.completions.create() kwargs.

        With ``provider_profile`` every quirk comes from the profile; the legacy flag
        path below (is_kimi, is_openrouter, ...) is only reached for unregistered providers.
        """
        sanitized = self.convert_messages(messages, model=model)
        _profile = params.get("provider_profile")
        if _profile:
            return self._build_kwargs_from_profile(_profile, model, sanitized, tools, params)

        sanitized = _swap_developer_role(sanitized, params.get("model_lower", (model or "").lower()))
        api_kwargs = _base_kwargs(model, sanitized, tools, params)

        is_kimi = params.get("is_kimi", False)
        is_lmstudio = params.get("is_lmstudio", False)
        supports_reasoning = params.get("supports_reasoning", False)
        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))
        _apply_max_tokens(api_kwargs, model, reasoning_config, params)

        # Kimi / TokenHub / LM Studio: top-level reasoning_effort (unless thinking disabled).
        thinking_off = isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False
        _e = requested_effort(reasoning_config)
        if is_kimi and not thinking_off:
            # K3 = low/high/max (server default high), K2-era = low/medium/high (default medium).
            _supported = kimi_supported_efforts(model)
            is_k3 = _supported is KIMI_K3_EFFORTS
            api_kwargs["reasoning_effort"] = (
                ("high" if is_k3 else "medium") if _e is None
                else clamp_effort(_e, _supported, KIMI_K3_OVERRIDES if is_k3 else None)
            )
        if params.get("is_tokenhub", False) and not thinking_off:
            api_kwargs["reasoning_effort"] = "high" if _e is None else clamp_effort(_e, TOKENHUB_EFFORTS)
        if is_lmstudio and supports_reasoning:
            _lm_effort = resolve_lmstudio_effort(reasoning_config, params.get("lmstudio_reasoning_options"))
            if _lm_effort is not None:
                api_kwargs["reasoning_effort"] = _lm_effort

        extra_body: dict[str, Any] = {}
        is_openrouter = params.get("is_openrouter", False)
        base_url = params.get("base_url")
        if is_openrouter and params.get("provider_preferences"):
            extra_body["provider"] = params["provider_preferences"]
        # Pareto Code router plugin (same shape as the OpenRouter profile path).
        if is_openrouter and model == "openrouter/pareto-code":
            _pareto_score_f = _pareto_score(params.get("openrouter_min_coding_score"))
            if _pareto_score_f is not None:
                extra_body["plugins"] = [{"id": "pareto-router", "min_coding_score": _pareto_score_f}]
        if is_kimi:
            extra_body["thinking"] = {"type": "disabled" if thinking_off else "enabled"}

        # LM Studio is handled above via top-level reasoning_effort.
        if supports_reasoning and not is_lmstudio:
            if params.get("is_github_models", False):
                if params.get("github_reasoning_extra") is not None:
                    extra_body["reasoning"] = params["github_reasoning_extra"]
            else:
                _effort = (reasoning_config.get("effort", "medium") or "medium") if reasoning_config and isinstance(reasoning_config, dict) else "medium"
                # Honor explicit "thinking off" like the profile path — never re-enable it.
                off = thinking_off or _effort == "none"
                extra_body["reasoning"] = {"enabled": not off, "effort": "none" if off else _effort}

        if str(params.get("provider_name") or "").strip().lower() == "gemini":
            raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if _is_gemini_openai_compat_base_url(base_url):
                thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
                if thinking_config:
                    openai_compat_extra = extra_body.get("extra_body", {})
                    openai_compat_extra["google"] = {**openai_compat_extra.get("google", {}), "thinking_config": thinking_config}
                    extra_body["extra_body"] = openai_compat_extra
            elif raw_thinking_config:
                extra_body["thinking_config"] = raw_thinking_config

        if params.get("extra_body_additions"):
            extra_body.update(params["extra_body_additions"])
        if extra_body:
            api_kwargs["extra_body"] = extra_body
        if params.get("request_overrides"):
            api_kwargs.update(params["request_overrides"])
        return _finish_kwargs(
            api_kwargs, sanitized, params,
            supports_prompt_cache_key=bool(params.get("supports_prompt_cache_key")) or _is_openai_api_base_url(base_url),
        )

    def _build_kwargs_from_profile(self, profile, model, sanitized, tools, params):
        """Build API kwargs from a ProviderProfile — every quirk comes from the profile object."""
        sanitized = _swap_developer_role(profile.prepare_messages(sanitized), (model or "").lower())
        api_kwargs = _base_kwargs(model, sanitized, tools, params, profile=profile)

        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))
        # Profiles fronting several backends override get_max_tokens() per model.
        _apply_max_tokens(api_kwargs, model, reasoning_config, params, profile_max=profile.get_max_tokens(model))

        extra_body_from_profile, top_level_from_profile = profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config, supports_reasoning=params.get("supports_reasoning", False),
            qwen_session_metadata=params.get("qwen_session_metadata"), model=model,
            base_url=params.get("base_url"), ollama_num_ctx=params.get("ollama_num_ctx"),
            session_id=params.get("session_id"), request_overrides=params.get("request_overrides"),
        )
        api_kwargs.update(top_level_from_profile)

        extra_body: dict[str, Any] = {}
        profile_body = profile.build_extra_body(
            session_id=params.get("session_id"), provider_preferences=params.get("provider_preferences"), model=model,
            base_url=params.get("base_url"), reasoning_config=reasoning_config,
            openrouter_min_coding_score=params.get("openrouter_min_coding_score"),
        )
        for part in (profile_body, extra_body_from_profile, params.get("extra_body_additions")):
            if part:
                extra_body = _deep_merge_dict(extra_body, part)
        for k, v in (params.get("request_overrides") or {}).items():
            if k == "extra_body" and isinstance(v, dict):
                extra_body = _deep_merge_dict(extra_body, v)
            else:
                api_kwargs[k] = v

        if extra_body:
            # Native Gemini speaks Google's REST schema: OpenAI-style extra_body
            # keys (tags, reasoning, provider, ...) are unknown fields -> HTTP 400.
            # The native client only reads thinking_config, so drop everything else.
            try:
                from agent.gemini_native_adapter import is_native_gemini_base_url
                _native_gemini = is_native_gemini_base_url(params.get("base_url"))
            except Exception:
                _native_gemini = False
            if _native_gemini:
                extra_body = {k: v for k, v in extra_body.items() if k in ("thinking_config", "thinkingConfig")}
            if extra_body:
                api_kwargs["extra_body"] = extra_body
        return _finish_kwargs(
            api_kwargs, sanitized, params, supports_prompt_cache_key=bool(getattr(profile, "supports_prompt_cache_key", False)),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize an OpenAI ChatCompletion.

        Gemini ``extra_content`` rides on ToolCall.provider_data; ``reasoning_content`` and
        ``reasoning_details`` stay distinct in provider_data because downstream reads them so.
        """
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        _fr = getattr(choice, "finish_reason", None)
        finish_reason = (str(_fr) if isinstance(_fr, int) else _fr) or "stop"  # Poolside returns int finish_reason

        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [tc for tc in (self._normalize_tool_call(tc) for tc in msg.tool_calls) if tc is not None]

        usage = Usage.from_openai(response.usage) if hasattr(response, "usage") and response.usage else None

        # Fields some SDKs park in pydantic ``model_extra`` rather than as attributes.
        reasoning_content = _attr_or_model_extra(msg, "reasoning_content")
        provider_data: dict[str, Any] = {}
        if reasoning_content is not None:
            provider_data["reasoning_content"] = reasoning_content
        if getattr(msg, "reasoning_details", None):
            provider_data["reasoning_details"] = msg.reasoning_details

        # OpenAI structured refusal (``message.refusal`` set, ``content`` empty); without
        # promotion the loop retries a deterministic refusal as an empty response.
        content = getattr(msg, "content", None)
        refusal = _attr_or_model_extra(msg, "refusal")
        if isinstance(refusal, str) and refusal.strip():
            provider_data["refusal"] = refusal
            # Terminal ``content_filter`` only when the refusal is the sole payload.
            if not (isinstance(content, str) and content.strip()) and not tool_calls:
                content = refusal
                if finish_reason in (None, "stop"):
                    finish_reason = "content_filter"

        return NormalizedResponse(
            content=content, tool_calls=tool_calls, finish_reason=finish_reason,
            reasoning=getattr(msg, "reasoning", None), usage=usage, provider_data=provider_data or None,
        )

    def _normalize_tool_call(self, tc: Any) -> ToolCall | None:
        """One SDK tool call -> ToolCall; None when it lacks a function/name (matches Relay's codec)."""
        tc_function = getattr(tc, "function", None)
        name = getattr(tc_function, "name", None)
        if tc_function is None or name is None:
            return None
        # Reverse only aliases THIS request emitted; a real ``hermes_tool_search`` tool stays itself.
        alias_map = self._last_wire_aliases
        if alias_map is None:
            name = "tool_search" if name == _XAI_TOOL_SEARCH_ALIAS else name
        else:
            name = alias_map.get(name, name)
        arguments = getattr(tc_function, "arguments", None)
        extra = _attr_or_model_extra(tc, "extra_content")
        return ToolCall(
            id=getattr(tc, "id", None), name=name, arguments="{}" if arguments is None else arguments,
            provider_data=None if extra is None else {"extra_content": _dump_extra_content(extra)},
        )

    def validate_response(self, response: Any) -> bool:
        """Check that response has valid choices."""
        return bool(response is not None and getattr(response, "choices", None))

    def extract_cache_stats(self, response: Any) -> dict[str, int] | None:
        """Cache stats from prompt_tokens_details (OpenRouter/OpenAI) or DeepSeek's top-level prompt_cache_hit_tokens."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        written = getattr(details, "cache_write_tokens", 0) or 0 if details else 0
        cached = cached or getattr(usage, "prompt_cache_hit_tokens", 0) or 0  # DeepSeek native
        return {"cached_tokens": cached, "creation_tokens": written} if cached or written else None


from agent.transports import register_transport  # noqa: E402

register_transport("chat_completions", ChatCompletionsTransport)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
