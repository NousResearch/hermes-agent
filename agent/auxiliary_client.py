"""Shared auxiliary OpenAI client for cheap/fast side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks (auto mode):
  1. OpenRouter  (OPENROUTER_API_KEY)
  2. Nous Portal (~/.hermes/auth.json active provider)
  3. Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY)
  4. Codex OAuth (Responses API via chatgpt.com with gpt-5.3-codex,
     wrapped to look like a chat.completions client)
  5. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
     — checked via PROVIDER_REGISTRY entries with auth_type='api_key'
  6. None

Resolution order for vision/multimodal tasks (auto mode):
  1. OpenRouter
  2. Nous Portal
  3. None  (steps 3-5 are skipped — they may not support multimodal)

Per-task provider overrides (e.g. AUXILIARY_VISION_PROVIDER,
CONTEXT_COMPRESSION_PROVIDER) can force a specific provider for each task:
"openrouter", "nous", "openai", or "main" (= steps 3-5).
Default "auto" follows the chains above.

Per-task model overrides (e.g. AUXILIARY_VISION_MODEL,
AUXILIARY_WEB_EXTRACT_MODEL) let callers use a different model slug
than the provider's default.
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = {
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "minimax": "MiniMax-M2.5-highspeed",
    "minimax-cn": "MiniMax-M2.5-highspeed",
}

# OpenRouter app attribution headers
_OR_HEADERS = {
    "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
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
_NOUS_MODEL = "gemini-3-flash"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_AUTH_JSON_PATH = Path.home() / ".hermes" / "auth.json"

# OpenAI direct: uses OPENAI_API_KEY with the official API endpoint.
# gpt-4o-mini is cheap/fast and supports vision — good default for auxiliary tasks.
_OPENAI_AUX_MODEL = "gpt-4o-mini"
_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Codex fallback: uses the Responses API (the only endpoint the Codex
# OAuth token can access) with a fast model for auxiliary tasks.
_CODEX_AUX_MODEL = "gpt-5.3-codex"
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


# ── Codex Responses → chat.completions adapter ─────────────────────────────
# All auxiliary consumers call client.chat.completions.create(**kwargs) and
# read response.choices[0].message.content. This adapter translates those
# calls to the Codex Responses API so callers don't need any changes.

class _CodexCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through the Codex Responses streaming API."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature")

        # Separate system/instructions from conversation messages
        instructions = "You are a helpful assistant."
        input_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content
            else:
                input_msgs.append({"role": role, "content": content})

        resp_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_msgs or [{"role": "user", "content": ""}],
            "stream": True,
            "store": False,
        }

        max_tokens = kwargs.get("max_output_tokens") or kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            resp_kwargs["max_output_tokens"] = int(max_tokens)
        if temperature is not None:
            resp_kwargs["temperature"] = temperature

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
            with self._client.responses.stream(**resp_kwargs) as stream:
                for _event in stream:
                    pass
                final = stream.get_final_response()

            # Extract text and tool calls from the Responses output
            for item in getattr(final, "output", []):
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for part in getattr(item, "content", []):
                        ptype = getattr(part, "type", None)
                        if ptype in ("output_text", "text"):
                            text_parts.append(getattr(part, "text", ""))
                elif item_type == "function_call":
                    tool_calls_raw.append(SimpleNamespace(
                        id=getattr(item, "call_id", ""),
                        type="function",
                        function=SimpleNamespace(
                            name=getattr(item, "name", ""),
                            arguments=getattr(item, "arguments", "{}"),
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


def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.hermes/auth.json for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens,
    otherwise None.
    """
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
    """Read a valid Codex OAuth access token from Hermes auth store (~/.hermes/auth.json)."""
    try:
        from hermes_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if isinstance(access_token, str) and access_token.strip():
            return access_token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order.

    Returns (client, model) for the first provider whose env var is set,
    or (None, None) if none are configured.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        # Check if any of the provider's env vars are set
        api_key = ""
        for env_var in pconfig.api_key_env_vars:
            val = os.getenv(env_var, "").strip()
            if val:
                api_key = val
                break
        if not api_key:
            continue
        # Resolve base URL (with optional env-var override)
        # Kimi Code keys (sk-kimi-) need api.kimi.com/coding/v1
        env_url = ""
        if pconfig.base_url_env_var:
            env_url = os.getenv(pconfig.base_url_env_var, "").strip()
        if env_url:
            base_url = env_url.rstrip("/")
        elif provider_id == "kimi-coding" and api_key.startswith("sk-kimi-"):
            base_url = "https://api.kimi.com/coding/v1"
        else:
            base_url = pconfig.inference_base_url
        model = _API_KEY_PROVIDER_AUX_MODELS.get(provider_id, "default")
        logger.debug("Auxiliary text client: %s (%s)", pconfig.name, model)
        extra = {}
        if "api.kimi.com" in base_url.lower():
            extra["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
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


def _try_openrouter() -> Tuple[Optional[OpenAI], Optional[str]]:
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
    return (
        OpenAI(api_key=_nous_api_key(nous), base_url=_nous_base_url()),
        _NOUS_MODEL,
    )


def _try_openai() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try OpenAI direct API (api.openai.com) using OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, None
    logger.debug("Auxiliary client: OpenAI direct (%s)", _OPENAI_AUX_MODEL)
    return OpenAI(api_key=api_key, base_url=_OPENAI_BASE_URL), _OPENAI_AUX_MODEL


def _try_custom_endpoint() -> Tuple[Optional[OpenAI], Optional[str]]:
    custom_base = os.getenv("OPENAI_BASE_URL")
    custom_key = os.getenv("OPENAI_API_KEY")
    if not custom_base or not custom_key:
        return None, None
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s)", model)
    return OpenAI(api_key=custom_key, base_url=custom_base), model


def _try_codex() -> Tuple[Optional[Any], Optional[str]]:
    codex_token = _read_codex_access_token()
    if not codex_token:
        return None, None
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", _CODEX_AUX_MODEL)
    real_client = OpenAI(api_key=codex_token, base_url=_CODEX_AUX_BASE_URL)
    return CodexAuxiliaryClient(real_client, _CODEX_AUX_MODEL), _CODEX_AUX_MODEL


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
            logger.warning("auxiliary.provider=nous but Nous Portal not configured (run: hermes login)")
        return client, model

    if forced == "openai":
        client, model = _try_openai()
        if client is None:
            logger.warning("auxiliary.provider=openai but OPENAI_API_KEY not set")
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


def _resolve_auto() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain: OpenRouter → Nous → custom → Codex → API-key → None."""
    for try_fn in (_try_openrouter, _try_nous, _try_custom_endpoint,
                   _try_codex, _resolve_api_key_provider):
        client, model = try_fn()
        if client is not None:
            return client, model
    logger.debug("Auxiliary client: none available")
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
    forced = _get_auxiliary_provider(task)
    if forced != "auto":
        return _resolve_forced_provider(forced)
    return _resolve_auto()


def get_async_text_auxiliary_client(task: str = ""):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    from openai import AsyncOpenAI

    sync_client, model = get_text_auxiliary_client(task)
    if sync_client is None:
        return None, None

    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    if "openrouter" in str(sync_client.base_url).lower():
        async_kwargs["default_headers"] = dict(_OR_HEADERS)
    elif "api.kimi.com" in str(sync_client.base_url).lower():
        async_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
    return AsyncOpenAI(**async_kwargs), model


def get_vision_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for vision/multimodal auxiliary tasks.

    Checks AUXILIARY_VISION_PROVIDER for a forced provider, otherwise
    auto-detects.  Callers may override the returned model with
    AUXILIARY_VISION_MODEL.

    In auto mode, only OpenRouter and Nous Portal are tried because they
    are known to support multimodal (Gemini).  Custom endpoints, Codex,
    and API-key providers are skipped — they may not handle vision input
    and would produce confusing errors.  To use one of those providers
    for vision, set AUXILIARY_VISION_PROVIDER explicitly.
    """
    forced = _get_auxiliary_provider("vision")
    if forced != "auto":
        return _resolve_forced_provider(forced)
    # Auto: only multimodal-capable providers
    for try_fn in (_try_openrouter, _try_nous):
        client, model = try_fn()
        if client is not None:
            return client, model
    logger.debug("Auxiliary vision client: none available (auto only tries OpenRouter/Nous)")
    return None, None


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
    # Check if any auxiliary task is explicitly forced to "openai"
    for task in ("vision", "web_extract", "compression"):
        if _get_auxiliary_provider(task) == "openai":
            return {"max_completion_tokens": value}
    custom_base = os.getenv("OPENAI_BASE_URL", "")
    or_key = os.getenv("OPENROUTER_API_KEY")
    # Only use max_completion_tokens for direct OpenAI custom endpoints
    if (not or_key
            and _read_nous_auth() is None
            and "api.openai.com" in custom_base.lower()):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}
