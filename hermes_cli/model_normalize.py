"""Per-provider model name normalization.

Different LLM providers expect model identifiers in different formats:

- **Aggregators** (OpenRouter, Nous, AI Gateway, Kilo Code) need
  ``vendor/model`` slugs like ``anthropic/claude-sonnet-4.6``.
- **Anthropic** native API expects bare names with dots replaced by
  hyphens: ``claude-sonnet-4-6``.
- **Copilot** expects bare names *with* dots preserved:
  ``claude-sonnet-4.6``.
- **OpenCode Zen** follows the same dot-to-hyphen convention as
  Anthropic: ``claude-sonnet-4-6``.
- **OpenCode Go** preserves dots in model names: ``minimax-m2.7``.
- **DeepSeek** only accepts two model identifiers:
  ``deepseek-chat`` and ``deepseek-reasoner``.
- **Custom** and remaining providers pass the name through as-is.

This module centralises that translation so callers can simply write::

    api_model = normalize_model_for_provider(user_input, provider)

Inspired by Clawdbot's ``normalizeAnthropicModelId`` pattern.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Vendor prefix mapping
# ---------------------------------------------------------------------------
# Maps the first hyphen-delimited token of a bare model name to the vendor
# slug used by aggregator APIs (OpenRouter, Nous, etc.).
#
# Example: "claude-sonnet-4.6" -> first token "claude" -> vendor "anthropic"
#          -> aggregator slug: "anthropic/claude-sonnet-4.6"

_VENDOR_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "gemma": "google",
    "deepseek": "deepseek",
    "glm": "z-ai",
    "kimi": "moonshotai",
    "minimax": "minimax",
    "grok": "x-ai",
    "qwen": "qwen",
    "mimo": "xiaomi",
    "nemotron": "nvidia",
    "llama": "meta-llama",
    "step": "stepfun",
    "trinity": "arcee-ai",
}

# Providers whose APIs consume vendor/model slugs.
_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset({
    "openrouter",
    "nous",
    "ai-gateway",
    "kilocode",
})

# Providers that want bare names with dots replaced by hyphens.
_DOT_TO_HYPHEN_PROVIDERS: frozenset[str] = frozenset({
    "anthropic",
    "opencode-zen",
})

# Providers that want bare names with dots preserved.
_STRIP_VENDOR_ONLY_PROVIDERS: frozenset[str] = frozenset({
    "copilot",
    "copilot-acp",
})

# Providers whose own naming is authoritative -- pass through unchanged.
_PASSTHROUGH_PROVIDERS: frozenset[str] = frozenset({
    "gemini",
    "zai",
    "kimi-coding",
    "minimax",
    "minimax-cn",
    "alibaba",
    "qwen-oauth",
    "huggingface",
    "openai-codex",
    "custom",
})

# ---------------------------------------------------------------------------
# DeepSeek special handling
# ---------------------------------------------------------------------------
# DeepSeek's API only recognises exactly two model identifiers.  We map
# common aliases and patterns to the canonical names.

_DEEPSEEK_REASONER_KEYWORDS: frozenset[str] = frozenset({
    "reasoner",
    "r1",
    "think",
    "reasoning",
    "cot",
})

_DEEPSEEK_CANONICAL_MODELS: frozenset[str] = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
})


def _normalize_for_deepseek(model_name: str) -> str:
    """Map any model input to one of DeepSeek's two accepted identifiers.

    Rules:
    - Already ``deepseek-chat`` or ``deepseek-reasoner`` -> pass through.
    - Contains any reasoner keyword (r1, think, reasoning, cot, reasoner)
      -> ``deepseek-reasoner``.
    - Everything else -> ``deepseek-chat``.

    Args:
        model_name: The bare model name (vendor prefix already stripped).

    Returns:
        One of ``"deepseek-chat"`` or ``"deepseek-reasoner"``.
    """
    bare = _strip_vendor_prefix(model_name).lower()

    if bare in _DEEPSEEK_CANONICAL_MODELS:
        return bare

    # Check for reasoner-like keywords anywhere in the name
    for keyword in _DEEPSEEK_REASONER_KEYWORDS:
        if keyword in bare:
            return "deepseek-reasoner"

    return "deepseek-chat"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _strip_vendor_prefix(model_name: str) -> str:
    """Remove a ``vendor/`` prefix if present.

    Examples::

        >>> _strip_vendor_prefix("anthropic/claude-sonnet-4.6")
        'claude-sonnet-4.6'
        >>> _strip_vendor_prefix("claude-sonnet-4.6")
        'claude-sonnet-4.6'
        >>> _strip_vendor_prefix("meta-llama/llama-4-scout")
        'llama-4-scout'
    """
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _dots_to_hyphens(model_name: str) -> str:
    """Replace dots with hyphens in a model name.

    Anthropic's native API uses hyphens where marketing names use dots:
    ``claude-sonnet-4.6`` -> ``claude-sonnet-4-6``.
    """
    return model_name.replace(".", "-")


def detect_vendor(model_name: str) -> Optional[str]:
    """Detect the vendor slug from a bare model name.

    Uses the first hyphen-delimited token of the model name to look up
    the corresponding vendor in ``_VENDOR_PREFIXES``.  Also handles
    case-insensitive matching and special patterns.

    Args:
        model_name: A model name, optionally already including a
            ``vendor/`` prefix.  If a prefix is present it is used
            directly.

    Returns:
        The vendor slug (e.g. ``"anthropic"``, ``"openai"``) or ``None``
        if no vendor can be confidently detected.

    Examples::

        >>> detect_vendor("claude-sonnet-4.6")
        'anthropic'
        >>> detect_vendor("gpt-5.4-mini")
        'openai'
        >>> detect_vendor("anthropic/claude-sonnet-4.6")
        'anthropic'
        >>> detect_vendor("my-custom-model")
    """
    name = model_name.strip()
    if not name:
        return None

    # If there's already a vendor/ prefix, extract it
    if "/" in name:
        return name.split("/", 1)[0].lower() or None

    name_lower = name.lower()

    # Try first hyphen-delimited token (exact match)
    first_token = name_lower.split("-")[0]
    if first_token in _VENDOR_PREFIXES:
        return _VENDOR_PREFIXES[first_token]

    # Handle patterns where the first token includes version digits,
    # e.g. "qwen3.5-plus" -> first token "qwen3.5", but prefix is "qwen"
    for prefix, vendor in _VENDOR_PREFIXES.items():
        if name_lower.startswith(prefix):
            return vendor

    return None


def _prepend_vendor(model_name: str) -> str:
    """Prepend the detected ``vendor/`` prefix if missing.

    Used for aggregator providers that require ``vendor/model`` format.
    If the name already contains a ``/``, it is returned as-is.
    If no vendor can be detected, the name is returned unchanged
    (aggregators may still accept it or return an error).

    Examples::

        >>> _prepend_vendor("claude-sonnet-4.6")
        'anthropic/claude-sonnet-4.6'
        >>> _prepend_vendor("anthropic/claude-sonnet-4.6")
        'anthropic/claude-sonnet-4.6'
        >>> _prepend_vendor("my-custom-thing")
        'my-custom-thing'
    """
    if "/" in model_name:
        return model_name

    vendor = detect_vendor(model_name)
    if vendor:
        return f"{vendor}/{model_name}"
    return model_name


# ---------------------------------------------------------------------------
# Main normalisation entry point
# ---------------------------------------------------------------------------

def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    """Translate a model name into the format the target provider's API expects.

    This is the primary entry point for model name normalisation.  It
    accepts any user-facing model identifier and transforms it for the
    specific provider that will receive the API call.

    Args:
        model_input: The model name as provided by the user or config.
            Can be bare (``"claude-sonnet-4.6"``), vendor-prefixed
            (``"anthropic/claude-sonnet-4.6"``), or already in native
            format (``"claude-sonnet-4-6"``).
        target_provider: The canonical Hermes provider id, e.g.
            ``"openrouter"``, ``"anthropic"``, ``"copilot"``,
            ``"deepseek"``, ``"custom"``.  Should already be normalised
            via ``hermes_cli.models.normalize_provider()``.

    Returns:
        The model identifier string that the target provider's API
        expects.

    Raises:
        No exceptions -- always returns a best-effort string.

    Examples::

        >>> normalize_model_for_provider("claude-sonnet-4.6", "openrouter")
        'anthropic/claude-sonnet-4.6'

        >>> normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic")
        'claude-sonnet-4-6'

        >>> normalize_model_for_provider("anthropic/claude-sonnet-4.6", "copilot")
        'claude-sonnet-4.6'

        >>> normalize_model_for_provider("openai/gpt-5.4", "copilot")
        'gpt-5.4'

        >>> normalize_model_for_provider("claude-sonnet-4.6", "opencode-zen")
        'claude-sonnet-4-6'

        >>> normalize_model_for_provider("deepseek-v3", "deepseek")
        'deepseek-chat'

        >>> normalize_model_for_provider("deepseek-r1", "deepseek")
        'deepseek-reasoner'

        >>> normalize_model_for_provider("my-model", "custom")
        'my-model'

        >>> normalize_model_for_provider("claude-sonnet-4.6", "zai")
        'claude-sonnet-4.6'
    """
    name = (model_input or "").strip()
    if not name:
        return name

    provider = (target_provider or "").strip().lower()

    # --- Aggregators: need vendor/model format ---
    if provider in _AGGREGATOR_PROVIDERS:
        return _prepend_vendor(name)

    # --- Anthropic / OpenCode: strip vendor, dots -> hyphens ---
    if provider in _DOT_TO_HYPHEN_PROVIDERS:
        bare = _strip_vendor_prefix(name)
        return _dots_to_hyphens(bare)

    # --- Copilot: strip vendor, keep dots ---
    if provider in _STRIP_VENDOR_ONLY_PROVIDERS:
        return _strip_vendor_prefix(name)

    # --- DeepSeek: map to one of two canonical names ---
    if provider == "deepseek":
        return _normalize_for_deepseek(name)

    # --- Custom & all others: pass through as-is ---
    return name


# ---------------------------------------------------------------------------
# Batch / convenience helpers
# ---------------------------------------------------------------------------

