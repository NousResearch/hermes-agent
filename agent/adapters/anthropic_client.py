"""Client-building functions for the Anthropic adapter."""

import logging
import sys
from typing import Any, Optional

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]

from agent.adapters import anthropic_auth as _auth

logger = logging.getLogger(__name__)

# Allow tests to patch agent.anthropic_adapter._anthropic_sdk.
# The façade re-exports _anthropic_sdk; this getter resolves from the
# façade module FIRST so that test patches propagate into the client.
def _get_sdk():
    # Check façade first (tests patch agent.anthropic_adapter._anthropic_sdk)
    mod = sys.modules.get("agent.anthropic_adapter")
    if mod is not None:
        facade_sdk = getattr(mod, "_anthropic_sdk", None)
        if facade_sdk is not None:
            return facade_sdk
    # Fallback: local import
    return _anthropic_sdk


def build_anthropic_client(api_key: str, base_url: str = None):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    Returns an anthropic.Anthropic instance.
    """
    if _get_sdk() is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    normalized_base_url = _auth._normalize_base_url_text(base_url)
    kwargs = {
        "timeout": Timeout(timeout=900.0, connect=10.0),
    }
    if normalized_base_url:
        kwargs["base_url"] = normalized_base_url
    common_betas = _auth._common_betas_for_base_url(normalized_base_url)

    if _auth._requires_bearer_auth(normalized_base_url):
        # Some Anthropic-compatible providers (e.g. MiniMax) expect the API key in
        # Authorization: Bearer *** for regular API keys. Route those endpoints
        # through auth_token so the SDK sends Bearer auth instead of x-api-key.
        # Check this before OAuth token shape detection because MiniMax secrets do
        # not use Anthropic's sk-ant-api prefix and would otherwise be misread as
        # Anthropic OAuth/setup tokens.
        kwargs["auth_token"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _auth._is_third_party_anthropic_endpoint(base_url):
        # Third-party proxies (Azure AI Foundry, AWS Bedrock, etc.) use their
        # own API keys with x-api-key auth. Skip OAuth detection — their keys
        # don't follow Anthropic's sk-ant-* prefix convention and would be
        # misclassified as OAuth tokens.
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _auth._is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + Claude Code identity.
        # Anthropic routes OAuth requests based on user-agent and headers;
        # without Claude Code's fingerprint, requests get intermittent 500s.
        all_betas = common_betas + _auth._OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {
            "anthropic-beta": ",".join(all_betas),
            "user-agent": f"claude-cli/{_auth._get_claude_code_version()} (external, cli)",
            "x-app": "cli",
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _get_sdk().Anthropic(**kwargs)


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format, case-insensitive)
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: claude-opus-4.6 → claude-opus-4-6), unless
      preserve_dots is True (e.g. for Alibaba/DashScope: qwen3.5-plus).
    """
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots:
        # OpenRouter uses dots for version separators (claude-opus-4.6),
        # Anthropic uses hyphens (claude-opus-4-6). Convert dots to hyphens.
        model = model.replace(".", "-")
    return model
