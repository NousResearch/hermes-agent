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
  3. Codex OAuth (gpt-5.3-codex supports vision via Responses API)
  4. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  5. None  (API-key providers like z.ai/Kimi/MiniMax are skipped —
     they may not support multimodal)

Per-task provider overrides (e.g. AUXILIARY_VISION_PROVIDER,
CONTEXT_COMPRESSION_PROVIDER) can force a specific provider for each task:
"openrouter", "nous", "codex", or "main" (= steps 3-5).
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
    "anthropic": "claude-haiku-4-5-20251001",
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

# Codex fallback: uses the Responses API (the only endpoint the Codex
# OAuth token can access) with a fast model for auxiliary tasks.
_CODEX_AUX_MODEL = "gpt-5.3-codex"
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


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
        temperature = kwargs.get("temperature")

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


def _try_custom_endpoint() -> Tuple[Optional[OpenAI], Optional[str]]:
    custom_base = os.getenv("OPENAI_BASE_URL")
    custom_key = os.getenv("OPENAI_API_KEY")
    if not custom_base or not custom_key:
        return None, None
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
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


def _resolve_auto() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain: OpenRouter → Nous → custom → Codex → API-key → None."""
    for try_fn in (_try_openrouter, _try_nous, _try_custom_endpoint,
                   _try_codex, _resolve_api_key_provider):
        client, model = try_fn()
        if client is not None:
            return client, model
    logger.debug("Auxiliary client: none available")
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

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    base_lower = str(sync_client.base_url).lower()
    if "openrouter" in base_lower:
        async_kwargs["default_headers"] = dict(_OR_HEADERS)
    elif "api.kimi.com" in base_lower:
        async_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.0"}
    return AsyncOpenAI(**async_kwargs), model


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
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

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    # Normalise aliases
    provider = (provider or "auto").strip().lower()
    if provider == "codex":
        provider = "openai-codex"
    if provider == "main":
        provider = "custom"

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved = _resolve_auto()
        if client is None:
            return None, None
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
                           "but Nous Portal not configured (run: hermes login)")
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

    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, _resolve_kimi_base_url
    except ImportError:
        logger.debug("hermes_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        logger.warning("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        # Find the first configured API key
        api_key = ""
        for env_var in pconfig.api_key_env_vars:
            api_key = os.getenv(env_var, "").strip()
            if api_key:
                break
        if not api_key:
            logger.warning("resolve_provider_client: provider %s has no API "
                           "key configured (tried: %s)",
                           provider, ", ".join(pconfig.api_key_env_vars))
            return None, None

        # Resolve base URL (env override → provider-specific logic → default)
        base_url_override = os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""
        if provider == "kimi-coding":
            base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, base_url_override)
        elif base_url_override:
            base_url = base_url_override
        else:
            base_url = pconfig.inference_base_url

        default_model = _API_KEY_PROVIDER_AUX_MODELS.get(provider, "")
        final_model = model or default_model

        # Provider-specific headers
        headers = {}
        if "api.kimi.com" in base_url.lower():
            headers["User-Agent"] = "KimiCLI/1.0"

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
    forced = _get_auxiliary_provider(task)
    if forced != "auto":
        return resolve_provider_client(forced)
    return resolve_provider_client("auto")


def get_async_text_auxiliary_client(task: str = ""):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    forced = _get_auxiliary_provider(task)
    if forced != "auto":
        return resolve_provider_client(forced, async_mode=True)
    return resolve_provider_client("auto", async_mode=True)


def get_vision_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for vision/multimodal auxiliary tasks.

    Checks AUXILIARY_VISION_PROVIDER for a forced provider, otherwise
    auto-detects.  Callers may override the returned model with
    AUXILIARY_VISION_MODEL.

    In auto mode, only providers known to support multimodal are tried:
    OpenRouter, Nous Portal, and Codex OAuth (gpt-5.3-codex supports
    vision via the Responses API).  Custom endpoints and API-key
    providers are skipped — they may not handle vision input.  To use
    them, set AUXILIARY_VISION_PROVIDER explicitly.
    """
    forced = _get_auxiliary_provider("vision")
    if forced != "auto":
        return resolve_provider_client(forced)
    # Auto: try providers known to support multimodal first, then fall
    # back to the user's custom endpoint.  Many local models (Qwen-VL,
    # LLaVA, Pixtral, etc.) support vision — skipping them entirely
    # caused silent failures for local-only users.
    for try_fn in (_try_openrouter, _try_nous, _try_codex,
                   _try_custom_endpoint):
        client, model = try_fn()
        if client is not None:
            return client, model
    logger.debug("Auxiliary vision client: none available")
    return None, None


def get_async_vision_auxiliary_client():
    """Return (async_client, model_slug) for async vision consumers.

    Properly handles Codex routing — unlike manually constructing
    AsyncOpenAI from a sync client, this preserves the Responses API
    adapter for Codex providers.

    Returns (None, None) when no provider is available.
    """
    sync_client, model = get_vision_auxiliary_client()
    if sync_client is None:
        return None, None
    return _to_async_client(sync_client, model)


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
    custom_base = os.getenv("OPENAI_BASE_URL", "")
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

# Client cache: (provider, async_mode) -> (client, default_model)
_client_cache: Dict[tuple, tuple] = {}


def _get_cached_client(
    provider: str, model: str = None, async_mode: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider."""
    cache_key = (provider, async_mode)
    if cache_key in _client_cache:
        cached_client, cached_default = _client_cache[cache_key]
        return cached_client, model or cached_default
    client, default_model = resolve_provider_client(provider, model, async_mode)
    if client is not None:
        _client_cache[cache_key] = (client, default_model)
    return client, model or default_model


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
) -> Tuple[str, Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model args (always win)
      2. Env var overrides (AUXILIARY_{TASK}_PROVIDER, etc.)
      3. Config file (auxiliary.{task}.provider/model or compression.*)
      4. "auto" (full auto-detection chain)

    Returns (provider, model) where model may be None (use provider default).
    """
    if provider:
        return provider, model

    if task:
        # Check env var overrides first
        env_provider = _get_auxiliary_provider(task)
        if env_provider != "auto":
            # Check for env var model override too
            env_model = None
            for prefix in ("AUXILIARY_", "CONTEXT_"):
                val = os.getenv(f"{prefix}{task.upper()}_MODEL", "").strip()
                if val:
                    env_model = val
                    break
            return env_provider, model or env_model

        # Read from config file
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except ImportError:
            return "auto", model

        # Check auxiliary.{task} section
        aux = config.get("auxiliary", {})
        task_config = aux.get(task, {})
        cfg_provider = task_config.get("provider", "").strip() or None
        cfg_model = task_config.get("model", "").strip() or None

        # Backwards compat: compression section has its own keys
        if task == "compression" and not cfg_provider:
            comp = config.get("compression", {})
            cfg_provider = comp.get("summary_provider", "").strip() or None
            cfg_model = cfg_model or comp.get("summary_model", "").strip() or None

        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, model or cfg_model
        return "auto", model or cfg_model

    return "auto", model


def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
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
            custom_base = os.getenv("OPENAI_BASE_URL", "")
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
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = 30.0,
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
        timeout: Request timeout in seconds.
        extra_body: Additional request body fields.

    Returns:
        Response object with .choices[0].message.content

    Raises:
        RuntimeError: If no provider is configured.
    """
    resolved_provider, resolved_model = _resolve_task_provider_model(
        task, provider, model)

    client, final_model = _get_cached_client(resolved_provider, resolved_model)
    if client is None:
        # Fallback: try openrouter
        if resolved_provider != "openrouter":
            logger.warning("Provider %s unavailable, falling back to openrouter",
                           resolved_provider)
            client, final_model = _get_cached_client(
                "openrouter", resolved_model or _OPENROUTER_MODEL)
    if client is None:
        raise RuntimeError(
            f"No LLM provider configured for task={task} provider={resolved_provider}. "
            f"Run: hermes setup")

    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=timeout, extra_body=extra_body)

    # Handle max_tokens vs max_completion_tokens retry
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as first_err:
        err_str = str(first_err)
        if "max_tokens" in err_str or "unsupported_parameter" in err_str:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            return client.chat.completions.create(**kwargs)
        raise


async def async_call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = 30.0,
    extra_body: dict = None,
) -> Any:
    """Centralized asynchronous LLM call.

    Same as call_llm() but async. See call_llm() for full documentation.
    """
    resolved_provider, resolved_model = _resolve_task_provider_model(
        task, provider, model)

    client, final_model = _get_cached_client(
        resolved_provider, resolved_model, async_mode=True)
    if client is None:
        if resolved_provider != "openrouter":
            logger.warning("Provider %s unavailable, falling back to openrouter",
                           resolved_provider)
            client, final_model = _get_cached_client(
                "openrouter", resolved_model or _OPENROUTER_MODEL,
                async_mode=True)
    if client is None:
        raise RuntimeError(
            f"No LLM provider configured for task={task} provider={resolved_provider}. "
            f"Run: hermes setup")

    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=timeout, extra_body=extra_body)

    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as first_err:
        err_str = str(first_err)
        if "max_tokens" in err_str or "unsupported_parameter" in err_str:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            return await client.chat.completions.create(**kwargs)
        raise
