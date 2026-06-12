"""
API key validation for Hermes Agent.

Validates provider API keys by making lightweight probe requests
(model listing) to each provider's inference endpoint. Uses httpx
for HTTP requests — NOT the OpenAI SDK.

Strategy per provider:
  - OpenAI-compatible (most providers): GET /v1/models with Bearer auth
  - Anthropic: GET /v1/models with x-api-key header
  - Google/Gemini: GET /v1/models with ?key= query param
  - Ollama/local: skipped (no validation needed)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Mapping from environment variable name to provider identifier.
# This covers all API-key-based providers in PROVIDER_REGISTRY.
# Keys checked in priority order (first match wins).
ENV_VAR_TO_PROVIDER: Dict[str, str] = {
    # OpenAI-compatible
    "OPENAI_API_KEY": "openai-api",
    "OPENAI_BASE_URL": "openai-api",
    # Anthropic
    "ANTHROPIC_API_KEY": "anthropic",
    "ANTHROPIC_TOKEN": "anthropic",
    "CLAUDE_CODE_OAUTH_TOKEN": "anthropic",
    "ANTHROPIC_BASE_URL": "anthropic",
    # Google / Gemini
    "GOOGLE_API_KEY": "gemini",
    "GEMINI_API_KEY": "gemini",
    "GEMINI_BASE_URL": "gemini",
    # xAI
    "XAI_API_KEY": "xai",
    "XAI_BASE_URL": "xai",
    # DeepSeek
    "DEEPSEEK_API_KEY": "deepseek",
    "DEEPSEEK_BASE_URL": "deepseek",
    # NVIDIA NIM
    "NVIDIA_API_KEY": "nvidia",
    "NVIDIA_BASE_URL": "nvidia",
    # Hugging Face
    "HF_TOKEN": "huggingface",
    "HF_BASE_URL": "huggingface",
    # Alibaba / Qwen Cloud
    "DASHSCOPE_API_KEY": "alibaba",
    "DASHSCOPE_BASE_URL": "alibaba",
    # Kimi / Moonshot
    "KIMI_API_KEY": "kimi-coding",
    "KIMI_CODING_API_KEY": "kimi-coding",
    "KIMI_BASE_URL": "kimi-coding",
    # GLM / Z.AI
    "GLM_API_KEY": "zai",
    "ZAI_API_KEY": "zai",
    "Z_AI_API_KEY": "zai",
    "GLM_BASE_URL": "zai",
    # StepFun
    "STEPFUN_API_KEY": "stepfun",
    "STEPFUN_BASE_URL": "stepfun",
    # MiniMax
    "MINIMAX_API_KEY": "minimax",
    "MINIMAX_BASE_URL": "minimax",
    # MiniMax (China)
    "MINIMAX_CN_API_KEY": "minimax-cn",
    "MINIMAX_CN_BASE_URL": "minimax-cn",
    # LM Studio
    "LM_API_KEY": "lmstudio",
    "LM_BASE_URL": "lmstudio",
    # GitHub Copilot
    "COPILOT_GITHUB_TOKEN": "copilot",
    "GH_TOKEN": "copilot",
    "GITHUB_TOKEN": "copilot",
    "COPILOT_API_BASE_URL": "copilot",
    # Ollama Cloud
    "OLLAMA_API_KEY": "ollama-cloud",
    "OLLAMA_BASE_URL": "ollama-cloud",
    # Arcee AI
    "ARCEEAI_API_KEY": "arcee",
    "ARCEE_BASE_URL": "arcee",
    # GMI Cloud
    "GMI_API_KEY": "gmi",
    "GMI_BASE_URL": "gmi",
    # OpenCode Zen
    "OPENCODE_ZEN_API_KEY": "opencode-zen",
    "OPENCODE_ZEN_BASE_URL": "opencode-zen",
    # OpenCode Go
    "OPENCODE_GO_API_KEY": "opencode-go",
    "OPENCODE_GO_BASE_URL": "opencode-go",
    # Kilo Code
    "KILOCODE_API_KEY": "kilocode",
    "KILOCODE_BASE_URL": "kilocode",
    # Xiaomi MiMo
    "XIAOMI_API_KEY": "xiaomi",
    "XIAOMI_BASE_URL": "xiaomi",
    # Tencent TokenHub
    "TOKENHUB_API_KEY": "tencent-tokenhub",
    "TOKENHUB_BASE_URL": "tencent-tokenhub",
    # Azure Foundry
    "AZURE_FOUNDRY_API_KEY": "azure-foundry",
    "AZURE_FOUNDRY_BASE_URL": "azure-foundry",
    # OpenCode Zen env vars
    "OPENCODE_ZEN_KEY": "opencode-zen",
}

# Providers that use Anthropic-style API (x-api-key header, /v1/messages)
_ANTHROPIC_STYLE_PROVIDERS = {
    "anthropic",
    "minimax",
    "minimax-cn",
}

# Providers that use Google/Gemini-style API (?key= query param)
_GOOGLE_STYLE_PROVIDERS = {
    "gemini",
}

# Providers that should be skipped (local/no-auth)
_SKIP_PROVIDERS = {
    "ollama",  # local Ollama — no API key needed
    "lmstudio",  # local LM Studio — no API key needed
}


def provider_for_env_var(env_var: str) -> Optional[str]:
    """Map an environment variable name to a provider identifier.

    Args:
        env_var: The environment variable name (e.g., "OPENAI_API_KEY").

    Returns:
        The provider identifier string (e.g., "openai-api"), or None if
        the env var is not recognized as belonging to a known provider.
    """
    return ENV_VAR_TO_PROVIDER.get(env_var)


def _get_default_base_url(provider: str) -> str:
    """Return the default inference base URL for a provider, or empty string."""
    # Provider base URLs — mirroring the defaults from PROVIDER_REGISTRY
    _BASE_URLS: Dict[str, str] = {
        "openai-api": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "xai": "https://api.x.ai/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "huggingface": "https://router.huggingface.co/v1",
        "alibaba": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "alibaba-coding-plan": "https://coding-intl.dashscope.aliyuncs.com/v1",
        "kimi-coding": "https://api.moonshot.ai/v1",
        "kimi-coding-cn": "https://api.moonshot.cn/v1",
        "zai": "https://api.z.ai/api/paas/v4",
        "stepfun": "https://api.stepfun.com/v1",
        "minimax": "https://api.minimax.io/anthropic",
        "minimax-cn": "https://api.minimaxi.com/anthropic",
        "lmstudio": "http://127.0.0.1:1234/v1",
        "copilot": "https://models.github.ai/v1",
        "ollama-cloud": "https://api.ollama.cloud/v1",
        "arcee": "https://api.arcee.ai/api/v1",
        "gmi": "https://api.gmi-serving.com/v1",
        "opencode-zen": "https://opencode.ai/zen/v1",
        "opencode-go": "https://opencode.ai/zen/go/v1",
        "kilocode": "https://api.kilo.ai/api/gateway",
        "xiaomi": "https://api.xiaomimimo.com/v1",
        "tencent-tokenhub": "https://tokenhub.tencentmaas.com/v1",
        "azure-foundry": "",
    }
    return _BASE_URLS.get(provider, "")


def validate_api_key(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout: int = 10,
) -> Tuple[bool, str]:
    """Validate a provider API key by probing the models endpoint.

    Makes a lightweight GET request to the provider's model listing endpoint
    to verify the API key is valid and has access.

    Args:
        provider: The provider identifier (e.g., "openai-api", "anthropic",
                  "gemini").
        api_key: The API key to validate.
        base_url: Optional base URL override. If not provided, uses the
                  provider's default inference base URL.
        timeout: HTTP request timeout in seconds (default: 10).

    Returns:
        A tuple of (success, message):
        - (True, "") on success — key is valid.
        - (True, "billing_warning") on HTTP 402 — key works but has no
          billing configured.
        - (False, error_message) on any failure.

    Strategy:
        - OpenAI-compatible providers → GET /v1/models with ``Authorization:
          Bearer <api_key>``.
        - Anthropic / Anthropic-style providers → GET /v1/models with
          ``x-api-key: <api_key>`` header.
        - Google/Gemini → GET /v1/models with ``?key=<api_key>`` query param.
        - Ollama / local providers → skipped (returns True, "").
    """
    if not api_key:
        return False, "API key is empty"

    if provider in _SKIP_PROVIDERS:
        logger.info("Skipping validation for local provider: %s", provider)
        return True, ""

    resolved_base_url = base_url or _get_default_base_url(provider)
    if not resolved_base_url:
        return False, f"Unknown provider: {provider}"

    # Strip trailing slashes for consistent URL construction
    resolved_base_url = resolved_base_url.rstrip("/")

    try:
        if provider in _ANTHROPIC_STYLE_PROVIDERS:
            return _validate_anthropic_style(resolved_base_url, api_key, timeout)
        elif provider in _GOOGLE_STYLE_PROVIDERS:
            return _validate_google_style(resolved_base_url, api_key, timeout)
        else:
            # Default: OpenAI-compatible (Bearer token auth)
            return _validate_openai_compat(resolved_base_url, api_key, timeout)
    except Exception as exc:
        logger.warning("API key validation failed for %s: %s", provider, exc)
        return False, str(exc)


def _validate_openai_compat(
    base_url: str, api_key: str, timeout: int
) -> Tuple[bool, str]:
    """Validate against an OpenAI-compatible /v1/models endpoint.

    Uses ``Authorization: Bearer <api_key>`` header.
    """
    models_url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    logger.debug("Validating OpenAI-compat key: GET %s", models_url)
    return _make_probe_request("GET", models_url, headers, None, timeout)


def _validate_anthropic_style(
    base_url: str, api_key: str, timeout: int
) -> Tuple[bool, str]:
    """Validate against an Anthropic-style /v1/models endpoint.

    Uses ``x-api-key: <api_key>`` header (Anthropic auth).
    """
    models_url = f"{base_url}/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Accept": "application/json",
    }

    logger.debug("Validating Anthropic-style key: GET %s", models_url)
    return _make_probe_request("GET", models_url, headers, None, timeout)


def _validate_google_style(
    base_url: str, api_key: str, timeout: int
) -> Tuple[bool, str]:
    """Validate against a Google/Gemini-style /v1/models endpoint.

    Uses ``?key=<api_key>`` query parameter.
    """
    models_url = f"{base_url}/models?key={api_key}"
    headers = {
        "Accept": "application/json",
    }

    logger.debug("Validating Google-style key: GET %s (key hidden)", models_url.replace(api_key, "***"))
    return _make_probe_request("GET", models_url, headers, None, timeout)


def _make_probe_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Any,
    timeout: int,
) -> Tuple[bool, str]:
    """Make an HTTP probe request and interpret the result.

    Returns:
        (True, "") on 2xx success.
        (True, "billing_warning") on 402 (key works, no billing).
        (False, error_msg) on any other failure.
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=headers)
            elif method.upper() == "HEAD":
                response = client.head(url, headers=headers)
            else:
                response = client.request(method, url, headers=headers, json=json_body)
    except httpx.TimeoutException:
        return False, f"Request timed out after {timeout}s"
    except httpx.ConnectError as exc:
        return False, f"Connection error: {exc}"
    except httpx.HTTPError as exc:
        return False, f"HTTP error: {exc}"

    status = response.status_code

    if status == 402:
        # 402 Payment Required — key is valid but no billing configured
        logger.info("API key validated but billing needed (HTTP 402)")
        return True, "billing_warning"

    if 200 <= status < 300:
        logger.debug("API key validated successfully (HTTP %d)", status)
        return True, ""

    # Error responses — extract meaningful message from body if possible
    error_msg = _extract_error_message(response, status)
    return False, error_msg


def _extract_error_message(response: httpx.Response, status: int) -> str:
    """Try to extract a human-readable error message from an API response."""
    try:
        body = response.json()
        # Try common error field paths
        error = body.get("error", {})
        if isinstance(error, dict):
            msg = error.get("message", "") or error.get("code", "")
            if msg:
                return f"HTTP {status}: {msg}"
        elif isinstance(error, str):
            return f"HTTP {status}: {error}"
        # Also check top-level message field
        if "message" in body:
            return f"HTTP {status}: {body['message']}"
    except (ValueError, KeyError, TypeError):
        pass

    # Fallback: just use status text
    return f"HTTP {status}: {response.reason_phrase or 'Request failed'}"
