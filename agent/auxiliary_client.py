"""Shared auxiliary client router for side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks (auto mode):
  1. OpenRouter  (OPENROUTER_API_KEY)
  2. Nous Portal (~/.hermes/auth.json active provider)
  3. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
  4. Codex OAuth (Responses API via chatgpt.com with gpt-5.3-codex,
     wrapped to look like a chat.completions client)
  5. Native Anthropic
  6. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
  7. None

Resolution order for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is one of the supported vision backends below
  2. OpenRouter
  3. Nous Portal
  4. Codex OAuth (gpt-5.3-codex supports vision via Responses API)
  5. Native Anthropic
  6. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  7. None

Per-task provider overrides (e.g. AUXILIARY_VISION_PROVIDER,
CONTEXT_COMPRESSION_PROVIDER) can force a specific provider for each task.
Default "auto" follows the chains above.

Per-task model overrides (e.g. AUXILIARY_VISION_MODEL,
AUXILIARY_WEB_EXTRACT_MODEL) let callers use a different model slug
than the provider's default.

Per-task direct endpoint overrides (e.g. AUXILIARY_VISION_BASE_URL,
AUXILIARY_VISION_API_KEY) let callers route a specific auxiliary task to a
custom OpenAI-compatible endpoint without touching the main model settings.

Payment / credit exhaustion fallback:
  When a resolved provider returns HTTP 402 or a credit-related error,
  call_llm() automatically retries with the next available provider in the
  auto-detection chain.  This handles the common case where a user depletes
  their OpenRouter balance but has Codex OAuth or another provider available.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path  # noqa: F401 — used by test mocks
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from agent.credential_pool import load_pool
from hermes_cli.config import get_hermes_home
from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "minimax": "MiniMax-M2.7-highspeed",
    "minimax-cn": "MiniMax-M2.7-highspeed",
    "anthropic": "claude-haiku-4-5-20251001",
    "ai-gateway": "google/gemini-3-flash",
    "opencode-zen": "gemini-3-flash",
    "opencode-go": "glm-5",
    "kilocode": "google/gemini-3-flash-preview",
}

# OpenRouter app attribution headers
_OR_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-OpenRouter-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}

# Nous Portal extra_body for product attribution.
# Callers should pass this as extra_body in chat.completions.create()
# when the auxiliary client is backed by Nous Portal.
NOUS_EXTRA_BODY = {"tags": ["product=hermes-agent"]}

# Set at resolve time — True if the auxiliary client points to Nous Portal
auxiliary_is_nous: bool = False

# Default auxiliary models per provider
_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
_NOUS_MODEL = "google/gemini-3-flash-preview"
_NOUS_FREE_TIER_VISION_MODEL = "xiaomi/mimo-v2-omni"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"

# Codex fallback: uses the Responses API (the only endpoint the Codex
# OAuth token can access) with a fast model for auxiliary tasks.
# ChatGPT-backed Codex accounts currently reject gpt-5.3-codex for these
# auxiliary flows, while gpt-5.2-codex remains broadly available and supports
# vision via Responses.
_CODEX_AUX_MODEL = "gpt-5.2-codex"
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    # Use the PooledCredential.runtime_api_key property which handles
    # provider-specific fallback (e.g. agent_key for nous).
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    # runtime_base_url handles provider-specific logic (e.g. nous prefers inference_base_url).
    # Fall back through inference_base_url and base_url for non-PooledCredential entries.
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


# ── Codex Responses → chat.completions adapter ─────────────────────────────
# All auxiliary consumers call client.chat.completions.create(**kwargs) and
# read response.choices[0].message.content. This adapter translates those
# calls to the Codex Responses API so callers don't need any changes.


def _convert_content_for_responses(content: Any) -> Any:
    """Convert chat.completions content to Responses API format.

    chat.completions uses:
      {"type": "text", "text": "..."}
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Responses API uses:
      {"type": "input_text", "text": "..."}
      {"type": "input_image", "image_url": "data:image/png;base64,..."}

    If content is a plain string, it's returned as-is (the Responses API
    accepts strings directly for text-only messages).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    converted: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            # chat.completions nests the URL: {"image_url": {"url": "..."}}
            image_data = part.get("image_url", {})
            url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data)
            entry: Dict[str, Any] = {"type": "input_image", "image_url": url}
            # Preserve detail if specified
            detail = image_data.get("detail") if isinstance(image_data, dict) else None
            if detail:
                entry["detail"] = detail
            converted.append(entry)
        elif ptype in ("input_text", "input_image"):
            # Already in Responses format — pass through
            converted.append(part)
        else:
            # Unknown content type — try to preserve as text
            text = part.get("text", "")
            if text:
                converted.append({"type": "input_text", "text": text})

    return converted or ""


class _CodexCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through the Codex Responses streaming API."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)

        # Separate system/instructions from conversation messages.
        # Convert chat.completions multimodal content blocks to Responses
        # API format (input_text / input_image instead of text / image_url).
        instructions = "You are a helpful assistant."
        input_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                input_msgs.append({
                    "role": role,
                    "content": _convert_content_for_responses(content),
                })

        resp_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_msgs or [{"role": "user", "content": ""}],
            "store": False,
        }

        # Note: the Codex endpoint (chatgpt.com/backend-api/codex) does NOT
        # support max_output_tokens or temperature — omit to avoid 400 errors.

        # Tools support for flush_memories and similar callers
        tools = kwargs.get("tools")
        if tools:
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                converted.append({
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            if converted:
                resp_kwargs["tools"] = converted

        # Stream and collect the response
        text_parts: List[str] = []
        tool_calls_raw: List[Any] = []
        usage = None

        try:
            # Collect output items and text deltas during streaming —
            # the Codex backend can return empty response.output from
            # get_final_response() even when items were streamed.
            collected_output_items: List[Any] = []
            collected_text_deltas: List[str] = []
            has_function_calls = False
            with self._client.responses.stream(**resp_kwargs) as stream:
                for _event in stream:
                    _etype = getattr(_event, "type", "")
                    if _etype == "response.output_item.done":
                        _done = getattr(_event, "item", None)
                        if _done is not None:
                            collected_output_items.append(_done)
                    elif "output_text.delta" in _etype:
                        _delta = getattr(_event, "delta", "")
                        if _delta:
                            collected_text_deltas.append(_delta)
                    elif "function_call" in _etype:
                        has_function_calls = True
                final = stream.get_final_response()

            # Backfill empty output from collected stream events
            _output = getattr(final, "output", None)
            if isinstance(_output, list) and not _output:
                if collected_output_items:
                    final.output = list(collected_output_items)
                    logger.debug(
                        "Codex auxiliary: backfilled %d output items from stream events",
                        len(collected_output_items),
                    )
                elif collected_text_deltas and not has_function_calls:
                    # Only synthesize text when no tool calls were streamed —
                    # a function_call response with incidental text should not
                    # be collapsed into a plain-text message.
                    assembled = "".join(collected_text_deltas)
                    final.output = [SimpleNamespace(
                        type="message", role="assistant", status="completed",
                        content=[SimpleNamespace(type="output_text", text=assembled)],
                    )]
                    logger.debug(
                        "Codex auxiliary: synthesized from %d deltas (%d chars)",
                        len(collected_text_deltas), len(assembled),
                    )

            # Extract text and tool calls from the Responses output.
            # Items may be SDK objects (attrs) or dicts (raw/fallback paths),
            # so use a helper that handles both shapes.
            def _item_get(obj: Any, key: str, default: Any = None) -> Any:
                val = getattr(obj, key, None)
                if val is None and isinstance(obj, dict):
                    val = obj.get(key, default)
                return val if val is not None else default

            for item in getattr(final, "output", []):
                item_type = _item_get(item, "type")
                if item_type == "message":
                    for part in (_item_get(item, "content") or []):
                        ptype = _item_get(part, "type")
                        if ptype in ("output_text", "text"):
                            text_parts.append(_item_get(part, "text", ""))
                elif item_type == "function_call":
                    tool_calls_raw.append(SimpleNamespace(
                        id=_item_get(item, "call_id", ""),
                        type="function",
                        function=SimpleNamespace(
                            name=_item_get(item, "name", ""),
                            arguments=_item_get(item, "arguments", "{}"),
                        ),
                    ))

            resp_usage = getattr(final, "usage", None)
            if resp_usage:
                usage = SimpleNamespace(
                    prompt_tokens=getattr(resp_usage, "input_tokens", 0),
                    completion_tokens=getattr(resp_usage, "output_tokens", 0),
                    total_tokens=getattr(resp_usage, "total_tokens", 0),
                )
        except Exception as exc:
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise

        content = "".join(text_parts).strip() or None

        # Build a response that looks like chat.completions
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls_raw or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="stop" if not tool_calls_raw else "tool_calls",
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _CodexChatShim:
    """Wraps the adapter to provide client.chat.completions.create()."""

    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter


class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper that routes through Codex Responses API.

    Consumers can call client.chat.completions.create(**kwargs) as normal.
    Also exposes .api_key and .base_url for introspection by async wrappers.
    """

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        adapter = _CodexCompletionsAdapter(real_client, model)
        self.chat = _CodexChatShim(adapter)
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class _AsyncCodexCompletionsAdapter:
    """Async version of the Codex Responses adapter.

    Wraps the sync adapter via asyncio.to_thread() so async consumers
    (web_tools, session_search) can await it as normal.
    """

    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCodexChatShim:
    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter


class AsyncCodexAuxiliaryClient:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create()."""

    def __init__(self, sync_wrapper: "CodexAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncCodexCompletionsAdapter(sync_adapter)
        self.chat = _AsyncCodexChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url


class _AnthropicCompletionsAdapter:
    """OpenAI-client-compatible adapter for Anthropic Messages API."""

    def __init__(self, real_client: Any, model: str, is_oauth: bool = False):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth

    def create(self, **kwargs) -> Any:
        from agent.anthropic_adapter import build_anthropic_kwargs, normalize_anthropic_response

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2000
        temperature = kwargs.get("temperature")

        normalized_tool_choice = None
        if isinstance(tool_choice, str):
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type", "")).lower()
            if choice_type == "function":
                normalized_tool_choice = tool_choice.get("function", {}).get("name")
            elif choice_type in {"auto", "required", "none"}:
                normalized_tool_choice = choice_type

        anthropic_kwargs = build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_config=None,
            tool_choice=normalized_tool_choice,
            is_oauth=self._is_oauth,
        )
        if temperature is not None:
            anthropic_kwargs["temperature"] = temperature

        response = self._client.messages.create(**anthropic_kwargs)
        assistant_message, finish_reason = normalize_anthropic_response(response)

        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        choice = SimpleNamespace(
            index=0,
            message=assistant_message,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _AnthropicChatShim:
    def __init__(self, adapter: _AnthropicCompletionsAdapter):
        self.completions = adapter


class AnthropicAuxiliaryClient:
    """OpenAI-client-compatible wrapper over a native Anthropic client."""

    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool = False):
        self._real_client = real_client
        adapter = _AnthropicCompletionsAdapter(real_client, model, is_oauth=is_oauth)
        self.chat = _AnthropicChatShim(adapter)
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, "close", None)
        if callable(close_fn):
            close_fn()


class _AsyncAnthropicCompletionsAdapter:
    def __init__(self, sync_adapter: _AnthropicCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncAnthropicChatShim:
    def __init__(self, adapter: _AsyncAnthropicCompletionsAdapter):
        self.completions = adapter


class AsyncAnthropicAuxiliaryClient:
    def __init__(self, sync_wrapper: "AnthropicAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncAnthropicCompletionsAdapter(sync_adapter)
        self.chat = _AsyncAnthropicChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url


def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.hermes/auth.json for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens,
    otherwise None.
    """
    pool_present, entry = _select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }

    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        # Must have at least an access_token or agent_key
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    """Extract the best API key from a Nous provider state dict."""
    return provider.get("agent_key") or provider.get("access_token", "")


def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


def _read_codex_access_token() -> Optional[str]:
    """Read a valid, non-expired Codex OAuth access token from Hermes auth store."""
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        return token or None

    try:
        from hermes_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        # Check JWT expiry — expired tokens block the auto chain and
        # prevent fallback to working providers (e.g. Anthropic).
        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass  # Non-JWT token or decode error — use as-is

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order.

    Returns (client, model) for the first provider with usable runtime
    credentials, or (None, None) if none are configured.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        if provider_id == "anthropic":
            return _try_anthropic()

        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue

            base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            model = _API_KEY_PROVIDER_AUX_MODELS.get(provider_id, "default")
            logger.debug("Auxiliary text client: %s (%s) via pool", pconfig.name, model)
            extra = {}
            if "api.kimi.com" in base_url.lower():
                extra["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
            elif "api.githubcopilot.com" in base_url.lower():
                from hermes_cli.models import copilot_default_headers

                extra["default_headers"] = copilot_default_headers()
            return OpenAI(api_key=api_key, base_url=base_url, **extra), model

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            continue

        base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        model = _API_KEY_PROVIDER_AUX_MODELS.get(provider_id, "default")
        logger.debug("Auxiliary text client: %s (%s)", pconfig.name, model)
        extra = {}
        if "api.kimi.com" in base_url.lower():
            extra["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
        elif "api.githubcopilot.com" in base_url.lower():
            from hermes_cli.models import copilot_default_headers

            extra["default_headers"] = copilot_default_headers()
        return OpenAI(api_key=api_key, base_url=base_url, **extra), model

    return None, None


# ── Provider resolution helpers ─────────────────────────────────────────────

def _get_auxiliary_provider(task: str = "") -> str:
    """Read the provider override for a specific auxiliary task.

    Checks AUXILIARY_{TASK}_PROVIDER first (e.g. AUXILIARY_VISION_PROVIDER),
    then CONTEXT_{TASK}_PROVIDER (for the compression section's summary_provider),
    then falls back to "auto".  Returns one of: "auto", "openrouter", "nous", "main".
    """
    if task:
        for prefix in ("AUXILIARY_", "CONTEXT_"):
            val = os.getenv(f"{prefix}{task.upper()}_PROVIDER", "").strip().lower()
            if val and val != "auto":
                return val
    return "auto"


def _get_auxiliary_env_override(task: str, suffix: str) -> Optional[str]:
    """Read an auxiliary env override from AUXILIARY_* or CONTEXT_* prefixes."""
    if not task:
        return None
    for prefix in ("AUXILIARY_", "CONTEXT_"):
        val = os.getenv(f"{prefix}{task.upper()}_{suffix}", "").strip()
        if val:
            return val
    return None


def _try_openrouter() -> Tuple[Optional[OpenAI], Optional[str]]:
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        or_key = _pool_runtime_api_key(entry)
        if not or_key:
            return None, None
        base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
        logger.debug("Auxiliary client: OpenRouter via pool")
        return OpenAI(api_key=or_key, base_url=base_url,
                       default_headers=_OR_HEADERS), _OPENROUTER_MODEL

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        return None, None
    logger.debug("Auxiliary client: OpenRouter")
    return OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL,
                   default_headers=_OR_HEADERS), _OPENROUTER_MODEL


def _try_nous() -> Tuple[Optional[OpenAI], Optional[str]]:
    nous = _read_nous_auth()
    if not nous:
        return None, None
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug("Auxiliary client: Nous Portal")
    if nous.get("source") == "pool":
        model = "gemini-3-flash"
    else:
        model = _NOUS_MODEL
    # Free-tier users can't use paid auxiliary models — use the free
    # multimodal model instead so vision/browser-vision still works.
    try:
        from hermes_cli.models import check_nous_free_tier
        if check_nous_free_tier():
            model = _NOUS_FREE_TIER_VISION_MODEL
            logger.debug("Free-tier Nous account — using %s for auxiliary/vision", model)
    except Exception:
        pass
    return (
        OpenAI(
            api_key=_nous_api_key(nous),
            base_url=str(nous.get("inference_base_url") or _nous_base_url()).rstrip("/"),
        ),
        model,
    )


def _read_main_model() -> str:
    """Read the user's configured main model from config.yaml.

    config.yaml model.default is the single source of truth for the active
    model. Environment variables are no longer consulted.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default", "")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ""


def _read_main_provider() -> str:
    """Read the user's configured main provider from config.yaml.

    Returns the lowercase provider id (e.g. "alibaba", "openrouter") or ""
    if not configured.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get("provider", "")
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ""


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint the same way the main CLI does.

    This covers both env-driven OPENAI_BASE_URL setups and config-saved custom
    endpoints where the base URL lives in config.yaml instead of the live
    environment.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        return None, None

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None

    custom_base = custom_base.strip().rstrip("/")
    if "openrouter.ai" in custom_base.lower():
        # requested='custom' falls back to OpenRouter when no custom endpoint is
        # configured. Treat that as "no custom endpoint" for auxiliary routing.
        return None, None

    # Local servers (Ollama, llama.cpp, vLLM, LM Studio) don't require auth.
    # Use a placeholder key — the OpenAI SDK requires a non-empty string but
    # local servers ignore the Authorization header.  Same fix as cli.py
    # _ensure_runtime_credentials() (PR #2556).
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    return custom_base, custom_key.strip()


def _current_custom_base_url() -> str:
    custom_base, _ = _resolve_custom_runtime()
    return custom_base or ""


def _try_custom_endpoint() -> Tuple[Optional[OpenAI], Optional[str]]:
    custom_base, custom_key = _resolve_custom_runtime()
    if not custom_base or not custom_key:
        return None, None
    model = _read_main_model() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s)", model)
    return OpenAI(api_key=custom_key, base_url=custom_base), model


def _try_codex() -> Tuple[Optional[Any], Optional[str]]:
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if not codex_token:
            return None, None
        base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", _CODEX_AUX_MODEL)
    real_client = OpenAI(api_key=codex_token, base_url=base_url)
    return CodexAuxiliaryClient(real_client, _CODEX_AUX_MODEL), _CODEX_AUX_MODEL


def _try_anthropic() -> Tuple[Optional[Any], Optional[str]]:
    try:
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    except ImportError:
        return None, None

    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present:
        if entry is None:
            return None, None
        token = _pool_runtime_api_key(entry)
    else:
        entry = None
        token = resolve_anthropic_token()
    if not token:
        return None, None

    # Allow base URL override from config.yaml model.base_url, but only
    # when the configured provider is anthropic — otherwise a non-Anthropic
    # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url:
                    base_url = cfg_base_url
    except Exception:
        pass

    from agent.anthropic_adapter import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _API_KEY_PROVIDER_AUX_MODELS.get("anthropic", "claude-haiku-4-5-20251001")
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        # The anthropic_adapter module imports fine but the SDK itself is
        # missing — build_anthropic_client raises ImportError at call time
        # when _anthropic_sdk is None.  Treat as unavailable.
        return None, None
    return AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model


def _resolve_forced_provider(forced: str) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Resolve a specific forced provider.  Returns (None, None) if creds missing."""
    if forced == "openrouter":
        client, model = _try_openrouter()
        if client is None:
            logger.warning("auxiliary.provider=openrouter but OPENROUTER_API_KEY not set")
        return client, model

    if forced == "nous":
        client, model = _try_nous()
        if client is None:
            logger.warning("auxiliary.provider=nous but Nous Portal not configured (run: hermes auth)")
        return client, model

    if forced == "codex":
        client, model = _try_codex()
        if client is None:
            logger.warning("auxiliary.provider=codex but no Codex OAuth token found (run: hermes model)")
        return client, model

    if forced == "main":
        # "main" = skip OpenRouter/Nous, use the main chat model's credentials.
        for try_fn in (_try_custom_endpoint, _try_codex, _resolve_api_key_provider):
            client, model = try_fn()
            if client is not None:
                return client, model
        logger.warning("auxiliary.provider=main but no main endpoint credentials found")
        return None, None

    # Unknown provider name — fall through to auto
    logger.warning("Unknown auxiliary.provider=%r, falling back to auto", forced)
    return None, None


_AUTO_PROVIDER_LABELS = {
    "_try_openrouter": "openrouter",
    "_try_nous": "nous",
    "_try_custom_endpoint": "local/custom",
    "_try_codex": "openai-codex",
    "_resolve_api_key_provider": "api-key",
}

_AGGREGATOR_PROVIDERS = frozenset({"openrouter", "nous"})


def _get_provider_chain() -> List[tuple]:
    """Return the ordered provider detection chain.

    Built at call time (not module level) so that test patches
    on the ``_try_*`` functions are picked up correctly.
    """
    return [
        ("openrouter", _try_openrouter),
        ("nous", _try_nous),
        ("local/custom", _try_custom_endpoint),
        ("openai-codex", _try_codex),
        ("api-key", _resolve_api_key_provider),
    ]


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion rather than rate limiting.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # OpenRouter and other providers include "credits" or "afford" in 402 bodies,
    # but sometimes wrap them in 429 or other codes.
    if status in (402, 429, None):
        if any(kw in err_lower for kw in ("credits", "insufficient funds",
                                           "can only afford", "billing",
                                           "payment required")):
            return True
    return False


def _try_payment_fallback(
    failed_provider: str,
    task: str = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit error.

    Iterates the standard auto-detection chain, skipping the provider that
    returned a payment error.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    # Normalise the failed provider label for matching.
    skip = failed_provider.lower().strip()
    # Also skip Step-1 main-provider path if it maps to the same backend.
    # (e.g. main_provider="openrouter" → skip "openrouter" in chain)
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    # Map common resolved_provider values back to chain labels.
    _alias_to_label = {"openrouter": "openrouter", "nous": "nous",
                       "openai-codex": "openai-codex", "codex": "openai-codex",
                       "custom": "local/custom", "local/custom": "local/custom"}
    skip_chain_labels = {_alias_to_label.get(s, s) for s in skip_labels}

    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: payment error on %s — falling back to %s (%s)",
                task or "call", failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: payment error on %s and no fallback available (tried: %s)",
        task or "call", failed_provider, ", ".join(tried),
    )
    return None, None, ""


def _resolve_auto() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain.

    Priority:
      1. If the user's main provider is NOT an aggregator (OpenRouter / Nous),
         use their main provider + main model directly.  This ensures users on
         Alibaba, DeepSeek, ZAI, etc. get auxiliary tasks handled by the same
         provider they already have credentials for — no OpenRouter key needed.
      2. OpenRouter → Nous → custom → Codex → API-key providers (original chain).
    """
    global auxiliary_is_nous
    auxiliary_is_nous = False  # Reset — _try_nous() will set True if it wins

    # ── Step 1: non-aggregator main provider → use main model directly ──
    main_provider = _read_main_provider()
    main_model = _read_main_model()
    if (main_provider and main_model
            and main_provider not in _AGGREGATOR_PROVIDERS
            and main_provider not in ("auto", "custom", "")):
        client, resolved = resolve_provider_client(main_provider, main_model)
        if client is not None:
            logger.info("Auxiliary auto-detect: using main provider %s (%s)",
                        main_provider, resolved or main_model)
            return client, resolved or main_model

    # ── Step 2: aggregator / fallback chain ──────────────────────────────
    tried = []
    for label, try_fn in _get_provider_chain():
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info("Auxiliary auto-detect: using %s (%s) — skipped: %s",
                            label, model or "default", ", ".join(tried))
            else:
                logger.info("Auxiliary auto-detect: using %s (%s)", label, model or "default")
            return client, model
        tried.append(label)
    logger.warning("Auxiliary auto-detect: no provider available (tried: %s). "
                   "Compression, summarization, and memory flush will not work. "
                   "Set OPENROUTER_API_KEY or configure a local model in config.yaml.",
                   ", ".join(tried))
    return None, None


# ── Centralized Provider Router ─────────────────────────────────────────────
#
# resolve_provider_client() is the single entry point for creating a properly
# configured client given a (provider, model) pair.  It handles auth lookup,
# base URL resolution, provider-specific headers, and API format differences
# (Chat Completions vs Responses API for Codex).
#
# All auxiliary consumer code should go through this or the public helpers
# below — never look up auth env vars ad-hoc.


def _to_async_client(sync_client, model: str):
    """Convert a sync client to its async counterpart, preserving Codex routing."""
    from openai import AsyncOpenAI

    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return AsyncAnthropicAuxiliaryClient(sync_client), model

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    base_lower = str(sync_client.base_url).lower()
    if "openrouter" in base_lower:
        async_kwargs["default_headers"] = dict(_OR_HEADERS)
    elif "api.githubcopilot.com" in base_lower:
        from hermes_cli.models import copilot_default_headers

        async_kwargs["default_headers"] = copilot_default_headers()
    elif "api.kimi.com" in base_lower:
        async_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
    return AsyncOpenAI(**async_kwargs), model


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: given a provider name and optional model, return a
    configured client with the correct auth, base URL, and API format.

    The returned client always exposes ``.chat.completions.create()`` — for
    Codex/Responses API providers, an adapter handles the translation
    transparently.

    Args:
        provider: Provider identifier.  One of:
            "openrouter", "nous", "openai-codex" (or "codex"),
            "zai", "kimi-coding", "minimax", "minimax-cn",
            "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
            "auto" (full auto-detection chain).
        model: Model slug override.  If None, uses the provider's default
               auxiliary model.
        async_mode: If True, return an async-compatible client.
        raw_codex: If True, return a raw OpenAI client for Codex providers
            instead of wrapping in CodexAuxiliaryClient.  Use this when
            the caller needs direct access to responses.stream() (e.g.,
            the main agent loop).
        explicit_base_url: Optional direct OpenAI-compatible endpoint.
        explicit_api_key: Optional API key paired with explicit_base_url.

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    # Normalise aliases
    provider = (provider or "auto").strip().lower()
    if provider == "codex":
        provider = "openai-codex"
    if provider == "main":
        # Resolve to the user's actual main provider so named custom providers
        # and non-aggregator providers (DeepSeek, Alibaba, etc.) work correctly.
        main_prov = _read_main_provider()
        if main_prov and main_prov not in ("auto", "main", ""):
            provider = main_prov
        else:
            provider = "custom"

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved = _resolve_auto()
        if client is None:
            return None, None
        # When auto-detection lands on a non-OpenRouter provider (e.g. a
        # local server), an OpenRouter-formatted model override like
        # "google/gemini-3-flash-preview" won't work.  Drop it and use
        # the provider's own default model instead.
        if model and "/" in model and resolved and "/" not in resolved:
            logger.debug(
                "Dropping OpenRouter-format model %r for non-OpenRouter "
                "auxiliary provider (using %r instead)", model, resolved)
            model = None
        final_model = model or resolved
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── OpenRouter ───────────────────────────────────────────────────
    if provider == "openrouter":
        client, default = _try_openrouter()
        if client is None:
            logger.warning("resolve_provider_client: openrouter requested "
                           "but OPENROUTER_API_KEY not set")
            return None, None
        final_model = model or default
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── Nous Portal (OAuth) ──────────────────────────────────────────
    if provider == "nous":
        client, default = _try_nous()
        if client is None:
            logger.warning("resolve_provider_client: nous requested "
                           "but Nous Portal not configured (run: hermes auth)")
            return None, None
        final_model = model or default
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── OpenAI Codex (OAuth → Responses API) ─────────────────────────
    if provider == "openai-codex":
        if raw_codex:
            # Return the raw OpenAI client for callers that need direct
            # access to responses.stream() (e.g., the main agent loop).
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning("resolve_provider_client: openai-codex requested "
                               "but no Codex OAuth token found (run: hermes model)")
                return None, None
            final_model = model or _CODEX_AUX_MODEL
            raw_client = OpenAI(api_key=codex_token, base_url=_CODEX_AUX_BASE_URL)
            return (raw_client, final_model)
        # Standard path: wrap in CodexAuxiliaryClient adapter
        client, default = _try_codex()
        if client is None:
            logger.warning("resolve_provider_client: openai-codex requested "
                           "but no Codex OAuth token found (run: hermes model)")
            return None, None
        final_model = model or default
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY) ───────────
    if provider == "custom":
        if explicit_base_url:
            custom_base = explicit_base_url.strip()
            custom_key = (
                (explicit_api_key or "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or "no-key-required"  # local servers don't need auth
            )
            if not custom_base:
                logger.warning(
                    "resolve_provider_client: explicit custom endpoint requested "
                    "but base_url is empty"
                )
                return None, None
            final_model = model or _read_main_model() or "gpt-4o-mini"
            client = OpenAI(api_key=custom_key, base_url=custom_base)
            return (_to_async_client(client, final_model) if async_mode
                    else (client, final_model))
        # Try custom first, then codex, then API-key providers
        for try_fn in (_try_custom_endpoint, _try_codex,
                       _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = model or default
                return (_to_async_client(client, final_model) if async_mode
                        else (client, final_model))
        logger.warning("resolve_provider_client: custom/main requested "
                       "but no endpoint credentials found")
        return None, None

    # ── Named custom providers (config.yaml custom_providers list) ───
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider
        custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = custom_entry.get("base_url", "").strip()
            custom_key = custom_entry.get("api_key", "").strip() or "no-key-required"
            if custom_base:
                final_model = model or _read_main_model() or "gpt-4o-mini"
                client = OpenAI(api_key=custom_key, base_url=custom_base)
                logger.debug(
                    "resolve_provider_client: named custom provider %r (%s)",
                    provider, final_model)
                return (_to_async_client(client, final_model) if async_mode
                        else (client, final_model))
            logger.warning(
                "resolve_provider_client: named custom provider %r has no base_url",
                provider)
            return None, None
    except ImportError:
        pass

    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("hermes_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        logger.warning("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        if provider == "anthropic":
            client, default_model = _try_anthropic()
            if client is None:
                logger.warning("resolve_provider_client: anthropic requested but no Anthropic credentials found")
                return None, None
            final_model = model or default_model
            return (_to_async_client(client, final_model) if async_mode else (client, final_model))

        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            if provider == "copilot":
                tried_sources.append("gh auth token")
            logger.debug("resolve_provider_client: provider %s has no API "
                         "key configured (tried: %s)",
                         provider, ", ".join(tried_sources))
            return None, None

        base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url

        default_model = _API_KEY_PROVIDER_AUX_MODELS.get(provider, "")
        final_model = model or default_model

        # Provider-specific headers
        headers = {}
        if "api.kimi.com" in base_url.lower():
            headers["User-Agent"] = "KimiCLI/1.0"
        elif "api.githubcopilot.com" in base_url.lower():
            from hermes_cli.models import copilot_default_headers

            headers.update(copilot_default_headers())

        client = OpenAI(api_key=api_key, base_url=base_url,
                        **({"default_headers": headers} if headers else {}))
        logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    elif pconfig.auth_type in ("oauth_device_code", "oauth_external"):
        # OAuth providers — route through their specific try functions
        if provider == "nous":
            return resolve_provider_client("nous", model, async_mode)
        if provider == "openai-codex":
            return resolve_provider_client("openai-codex", model, async_mode)
        # Other OAuth providers not directly supported
        logger.warning("resolve_provider_client: OAuth provider %s not "
                       "directly supported, try 'auto'", provider)
        return None, None

    logger.warning("resolve_provider_client: unhandled auth_type %s for %s",
                   pconfig.auth_type, provider)
    return None, None


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(task: str = "") -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    Args:
        task: Optional task name ("compression", "web_extract") to check
              for a task-specific provider override.

    Callers may override the returned model with a per-task env var
    (e.g. CONTEXT_COMPRESSION_MODEL, AUXILIARY_WEB_EXTRACT_MODEL).
    """
    provider, model, base_url, api_key = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )


def get_async_text_auxiliary_client(task: str = ""):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )


_VISION_AUTO_PROVIDER_ORDER = (
    "openrouter",
    "nous",
    "openai-codex",
    "anthropic",
    "custom",
)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    provider = (provider or "auto").strip().lower()
    if provider == "codex":
        return "openai-codex"
    if provider == "main":
        # Resolve to actual main provider — named custom providers and
        # non-aggregator providers need to pass through as their real name.
        main_prov = _read_main_provider()
        if main_prov and main_prov not in ("auto", "main", ""):
            return main_prov
        return "custom"
    return provider


def _resolve_strict_vision_backend(provider: str) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "openrouter":
        return _try_openrouter()
    if provider == "nous":
        return _try_nous()
    if provider == "openai-codex":
        return _try_codex()
    if provider == "anthropic":
        return _try_anthropic()
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def _preferred_main_vision_provider() -> Optional[str]:
    """Return the selected main provider when it is also a supported vision backend."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            provider = _normalize_vision_provider(model_cfg.get("provider", ""))
            if provider in _VISION_AUTO_PROVIDER_ORDER:
                return provider
    except Exception:
        pass
    return None


def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    This is the single source of truth for setup, tool gating, and runtime
    auto-routing of vision tasks. The selected main provider is preferred when
    it is also a known-good vision backend; otherwise Hermes falls back through
    the standard conservative order.
    """
    ordered = list(_VISION_AUTO_PROVIDER_ORDER)
    preferred = _preferred_main_vision_provider()
    if preferred in ordered:
        ordered.remove(preferred)
        ordered.insert(0, preferred)
    return [provider for provider in ordered if _strict_vision_backend_available(provider)]


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    requested, resolved_model, resolved_base_url, resolved_api_key = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        client, final_model = resolve_provider_client(
            "custom",
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
        )
        if client is None:
            return "custom", None, None
        return "custom", client, final_model

    if requested == "auto":
        ordered = list(_VISION_AUTO_PROVIDER_ORDER)
        preferred = _preferred_main_vision_provider()
        if preferred in ordered:
            ordered.remove(preferred)
            ordered.insert(0, preferred)

        for candidate in ordered:
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)
        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(requested)
        return _finalize(requested, sync_client, default_model)

    client, final_model = _get_cached_client(requested, resolved_model, async_mode)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_vision_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for vision/multimodal auxiliary tasks."""
    _, client, final_model = resolve_vision_provider_client(async_mode=False)
    return client, final_model


def get_async_vision_auxiliary_client():
    """Return (async_client, model_slug) for async vision consumers."""
    _, client, final_model = resolve_vision_provider_client(async_mode=True)
    return client, final_model


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs for auxiliary API calls.
    
    Includes Nous Portal product tags when the auxiliary client is backed
    by Nous Portal. Returns empty dict otherwise.
    """
    return dict(NOUS_EXTRA_BODY) if auxiliary_is_nous else {}


def auxiliary_max_tokens_param(value: int) -> dict:
    """Return the correct max tokens kwarg for the auxiliary client's provider.
    
    OpenRouter and local models use 'max_tokens'. Direct OpenAI with newer
    models (gpt-4o, o-series, gpt-5+) requires 'max_completion_tokens'.
    The Codex adapter translates max_tokens internally, so we use max_tokens
    for it as well.
    """
    custom_base = _current_custom_base_url()
    or_key = os.getenv("OPENROUTER_API_KEY")
    # Only use max_completion_tokens for direct OpenAI custom endpoints
    if (not or_key
            and _read_nous_auth() is None
            and "api.openai.com" in custom_base.lower()):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


# ── Centralized LLM Call API ────────────────────────────────────────────────
#
# call_llm() and async_call_llm() own the full request lifecycle:
#   1. Resolve provider + model from task config (or explicit args)
#   2. Get or create a cached client for that provider
#   3. Format request args for the provider + model (max_tokens handling, etc.)
#   4. Make the API call
#   5. Return the response
#
# Every auxiliary LLM consumer should use these instead of manually
# constructing clients and calling .chat.completions.create().

# Client cache: (provider, async_mode, base_url, api_key) -> (client, default_model)
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.
    """
    import inspect

    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            client = entry[0]
            if client is None:
                continue
            # Mark any async httpx transport as closed first (prevents __del__
            # from scheduling aclose() on a dead event loop).
            _force_close_async_httpx(client)
            # Sync clients: close the httpx connection pool cleanly.
            # Async clients: skip — we already neutered __del__ above.
            try:
                close_fn = getattr(client, "close", None)
                if close_fn and not inspect.iscoroutinefunction(close_fn):
                    close_fn()
            except Exception:
                pass
        _client_cache.clear()


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                _force_close_async_httpx(client)
                stale_keys.append(key)
        for key in stale_keys:
            del _client_cache[key]


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues (especially in gateway
    mode where _run_async() may spawn fresh loops in worker threads), the
    cache key for async clients includes the current event loop's identity
    so each loop gets its own client instance.
    """
    # Include loop identity for async clients to prevent cross-loop reuse.
    # httpx.AsyncClient (inside AsyncOpenAI) is bound to the loop where it
    # was created — reusing it on a different loop causes deadlocks (#2681).
    loop_id = 0
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
            loop_id = id(current_loop)
        except RuntimeError:
            pass
    cache_key = (provider, async_mode, base_url or "", api_key or "", loop_id)
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # A cached async client whose loop has been closed will raise
                # "Event loop is closed" when httpx tries to clean up its
                # transport.  Discard the stale client and create a fresh one.
                if cached_loop is not None and cached_loop.is_closed():
                    _force_close_async_httpx(cached_client)
                    del _client_cache[cache_key]
                else:
                    return cached_client, model or cached_default
            else:
                return cached_client, model or cached_default
    # Build outside the lock
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    if client is not None:
        # For async clients, remember which loop they were created on so we
        # can detect stale entries later.
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                client, default_model, _ = _client_cache[cache_key]
    return client, model or default_model


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model/base_url/api_key args (always win)
      2. Env var overrides (AUXILIARY_{TASK}_*, CONTEXT_{TASK}_*)
      3. Config file (auxiliary.{task}.* or compression.*)
      4. "auto" (full auto-detection chain)

    Returns (provider, model, base_url, api_key) where model may be None
    (use provider default). When base_url is set, provider is forced to
    "custom" and the task uses that direct endpoint.
    """
    config = {}
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None

    if task:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except ImportError:
            config = {}

        aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
        task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
        if not isinstance(task_config, dict):
            task_config = {}
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None

        # Backwards compat: compression section has its own keys.
        # The auxiliary.compression defaults to provider="auto", so treat
        # both None and "auto" as "not explicitly configured".
        if task == "compression" and (not cfg_provider or cfg_provider == "auto"):
            comp = config.get("compression", {}) if isinstance(config, dict) else {}
            if isinstance(comp, dict):
                cfg_provider = comp.get("summary_provider", "").strip() or None
                cfg_model = cfg_model or comp.get("summary_model", "").strip() or None
                _sbu = comp.get("summary_base_url") or ""
                cfg_base_url = cfg_base_url or _sbu.strip() or None

    env_model = _get_auxiliary_env_override(task, "MODEL") if task else None
    resolved_model = model or env_model or cfg_model

    if base_url:
        return "custom", resolved_model, base_url, api_key
    if provider:
        return provider, resolved_model, base_url, api_key

    if task:
        env_base_url = _get_auxiliary_env_override(task, "BASE_URL")
        env_api_key = _get_auxiliary_env_override(task, "API_KEY")
        if env_base_url:
            return "custom", resolved_model, env_base_url, env_api_key or cfg_api_key

        env_provider = _get_auxiliary_provider(task)
        if env_provider != "auto":
            return env_provider, resolved_model, None, None

        if cfg_base_url:
            return "custom", resolved_model, cfg_base_url, cfg_api_key
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, None, None
        return "auto", resolved_model, None, None

    return "auto", resolved_model, None, None


_DEFAULT_AUX_TIMEOUT = 30.0


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """Read timeout from auxiliary.{task}.timeout in config, falling back to *default*."""
    if not task:
        return default
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except ImportError:
        return default
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    raw = task_config.get("timeout")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        # Codex adapter handles max_tokens internally; OpenRouter/Nous use max_tokens.
        # Direct OpenAI api.openai.com with newer models needs max_completion_tokens.
        if provider == "custom":
            custom_base = base_url or _current_custom_base_url()
            if "api.openai.com" in custom_base.lower():
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    if tools:
        kwargs["tools"] = tools

    # Provider-specific extra_body
    merged_extra = dict(extra_body or {})
    if provider == "nous" or auxiliary_is_nous:
        merged_extra.setdefault("tags", []).extend(["product=hermes-agent"])
    if merged_extra:
        kwargs["extra_body"] = merged_extra

    return kwargs


def call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
) -> Any:
    """Centralized synchronous LLM call.

    Resolves provider + model (from task config, explicit args, or auto-detect),
    handles auth, request formatting, and model-specific arg adjustments.

    Args:
        task: Auxiliary task name ("compression", "vision", "web_extract",
              "session_search", "skills_hub", "mcp", "flush_memories").
              Reads provider:model from config/env. Ignored if provider is set.
        provider: Explicit provider override.
        model: Explicit model override.
        messages: Chat messages list.
        temperature: Sampling temperature (None = provider default).
        max_tokens: Max output tokens (handles max_tokens vs max_completion_tokens).
        tools: Tool definitions (for function calling).
        timeout: Request timeout in seconds (None = read from auxiliary.{task}.timeout config).
        extra_body: Additional request body fields.

    Returns:
        Response object with .choices[0].message.content

    Raises:
        RuntimeError: If no provider is configured.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            async_mode=False,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=False,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )
        if client is None:
            # When the user explicitly chose a non-OpenRouter provider but no
            # credentials were found, fail fast instead of silently routing
            # through OpenRouter (which causes confusing 404s).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                raise RuntimeError(
                    f"Provider '{_explicit}' is set in config.yaml but no API key "
                    f"was found. Set the {_explicit.upper()}_API_KEY environment "
                    f"variable, or switch to a different provider with `hermes model`."
                )
            # For auto/custom with no credentials, try the full auto chain
            # rather than hardcoding OpenRouter (which may be depleted).
            # Pass model=None so each provider uses its own default —
            # resolved_model may be an OpenRouter-format slug that doesn't
            # work on other providers.
            if not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client("auto")
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup")

    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)

    # Log what we're about to do — makes auxiliary operations visible
    _base_info = str(getattr(client, "base_url", resolved_base_url) or "")
    if task:
        logger.info("Auxiliary %s: using %s (%s)%s",
                     task, resolved_provider or "auto", final_model or "default",
                     f" at {_base_info}" if _base_info and "openrouter" not in _base_info else "")

    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=extra_body,
        base_url=resolved_base_url)

    # Handle max_tokens vs max_completion_tokens retry, then payment fallback.
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as first_err:
        err_str = str(first_err)
        if "max_tokens" in err_str or "unsupported_parameter" in err_str:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as retry_err:
                # If the max_tokens retry also hits a payment error,
                # fall through to the payment fallback below.
                if not _is_payment_error(retry_err):
                    raise
                first_err = retry_err

        # ── Payment / credit exhaustion fallback ──────────────────────
        # When the resolved provider returns 402 or a credit-related error,
        # try alternative providers instead of giving up.  This handles the
        # common case where a user runs out of OpenRouter credits but has
        # Codex OAuth or another provider available.
        if _is_payment_error(first_err):
            fb_client, fb_model, fb_label = _try_payment_fallback(
                resolved_provider, task)
            if fb_client is not None:
                fb_kwargs = _build_call_kwargs(
                    fb_label, fb_model, messages,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, timeout=effective_timeout,
                    extra_body=extra_body)
                return fb_client.chat.completions.create(**fb_kwargs)
        raise


def extract_content_or_reasoning(response) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.

    Mirrors the main agent loop's behavior when a reasoning model (DeepSeek-R1,
    Qwen-QwQ, etc.) returns ``content=None`` with reasoning in structured fields.

    Resolution order:
      1. ``message.content`` — strip inline think/reasoning blocks, check for
         remaining non-whitespace text.
      2. ``message.reasoning`` / ``message.reasoning_content`` — direct
         structured reasoning fields (DeepSeek, Moonshot, Novita, etc.).
      3. ``message.reasoning_details`` — OpenRouter unified array format.

    Returns the best available text, or ``""`` if nothing found.
    """
    import re

    msg = response.choices[0].message
    content = (msg.content or "").strip()

    if content:
        # Strip inline think/reasoning blocks (mirrors _strip_think_blocks)
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>"
            r".*?"
            r"</(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned

    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = getattr(msg, field, None)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())

    details = getattr(msg, "reasoning_details", None)
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = (
                    detail.get("summary")
                    or detail.get("content")
                    or detail.get("text")
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))

    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return ""


async def async_call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
) -> Any:
    """Centralized asynchronous LLM call.

    Same as call_llm() but async. See call_llm() for full documentation.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            async_mode=True,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=True,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=True,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )
        if client is None:
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                raise RuntimeError(
                    f"Provider '{_explicit}' is set in config.yaml but no API key "
                    f"was found. Set the {_explicit.upper()}_API_KEY environment "
                    f"variable, or switch to a different provider with `hermes model`."
                )
            if not resolved_base_url:
                logger.warning("Provider %s unavailable, falling back to openrouter",
                               resolved_provider)
                client, final_model = _get_cached_client(
                    "openrouter", resolved_model or _OPENROUTER_MODEL,
                    async_mode=True)
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup")

    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)

    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=extra_body,
        base_url=resolved_base_url)

    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as first_err:
        err_str = str(first_err)
        if "max_tokens" in err_str or "unsupported_parameter" in err_str:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            return await client.chat.completions.create(**kwargs)
        raise
