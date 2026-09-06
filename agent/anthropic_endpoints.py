"""Endpoint-family detection for Anthropic-compatible base URLs.

A dozen services speak the Anthropic Messages API but differ in auth style, accepted beta
headers, and request quirks (MiniMax, Kimi/Moonshot, DeepSeek, OpenCode, Azure AI Foundry, Nous
Portal, Bedrock). Every such difference is decided from the configured base URL, so the
predicates live together here as pure functions (no I/O, SDK or credentials) that both
``agent/anthropic_adapter.py`` and ``agent/anthropic_message_convert.py`` can import without a
cycle.
"""

import re
from urllib.parse import urlparse

from utils import base_url_host_matches, base_url_hostname

_MINIMAX_ANTHROPIC_PREFIXES = ("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic")


def _normalize_base_url_text(base_url) -> str:
    """Coerce a base URL (str or ``httpx.URL``) to a stripped string; "" when falsy."""
    return str(base_url).strip() if base_url else ""


def _normalized_lower(base_url) -> str:
    """``_normalize_base_url_text`` + rstrip("/") + lower(), the shape most predicates match on."""
    return _normalize_base_url_text(base_url).rstrip("/").lower()


def _is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """Any non-anthropic.com endpoint (own x-api-key keys; skip OAuth detection). No base_url =
    direct Anthropic API."""
    normalized = _normalized_lower(base_url)
    return bool(normalized) and "anthropic.com" not in normalized


def _is_kimi_coding_endpoint(base_url: str | None) -> bool:
    """Kimi's /coding endpoint, which requires a claude-code User-Agent."""
    return _normalized_lower(base_url).startswith("https://api.kimi.com/coding")


def _is_opencode_endpoint(base_url: str | None) -> bool:
    """OpenCode's Zen/Go relay (opencode.ai)."""
    return base_url_host_matches(base_url or "", "opencode.ai")


# Kimi / Moonshot family model-name prefixes: official slugs (``kimi-k2.5``, ``kimi_thinking``,
# ``moonshot-v1-8k``) and release lines (``k1.5-…``, ``k2-thinking``, ``k25-…``, ``k3.x``/``k3-…``).
# Matched case-insensitively after stripping any ``vendor/`` prefix.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_", "moonshot-", "moonshot_", "k1.", "k1-", "k2.", "k2-", "k25", "k2.5", "k3.", "k3-",
)
# Bare release slugs with no separator suffix (Kimi Coding Plan serves K3 as exactly ``k3``);
# exact-match so unrelated names sharing the prefix don't match.
_KIMI_FAMILY_EXACT_SLUGS = frozenset({"k3"})


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower().rsplit("/", 1)[-1]  # ``moonshotai/kimi-k2.5`` -> ``kimi-k2.5``
    return bool(m) and (m in _KIMI_FAMILY_EXACT_SLUGS or m.startswith(_KIMI_FAMILY_MODEL_PREFIXES))


def _is_kimi_family_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """Any Kimi / Moonshot Anthropic-Messages endpoint: the /coding endpoint, any api.kimi.com /
    moonshot.ai / moonshot.cn host, or any endpoint (e.g. a private gateway) whose *model* is in
    the Kimi family — the upstream enforces Kimi's thinking semantics regardless of hostname.
    Decides whether unsigned reasoning_content-derived thinking blocks are preserved on replay."""
    return (
        _is_kimi_coding_endpoint(base_url)
        or any(base_url_host_matches(base_url or "", d) for d in ("api.kimi.com", "moonshot.ai", "moonshot.cn"))
        or _model_name_is_kimi_family(model)
    )


def _is_deepseek_anthropic_endpoint(base_url: str | None) -> bool:
    """DeepSeek's ``/anthropic`` route. In thinking mode DeepSeek requires prior-turn ``thinking``
    blocks to round-trip while the generic third-party path strips them; its blocks are unsigned,
    so it gets the same strip-signed / keep-unsigned policy as Kimi. Pinned to the ``/anthropic``
    path so the OpenAI-compatible base URL is not misclassified.

    Per DeepSeek's published compatibility matrix the blocks are unsigned (no Anthropic-proprietary
    signature, no ``redacted_thinking`` support), so this endpoint is handled with the same strip-signed /
    keep-unsigned policy used for Kimi's ``/coding`` endpoint. See hermes-agent#16748.
    """
    return base_url_host_matches(base_url or "", "api.deepseek.com") and "/anthropic" in _normalized_lower(base_url)


def _is_nous_portal_endpoint(base_url: str | None) -> bool:
    """Nous Portal's Anthropic Messages route (Bearer JWT, verbatim catalog ids, native
    thinking-signature replay). Trusted hosts only: prod ``inference-api.nousresearch.com`` or the
    operator-set ``NOUS_INFERENCE_BASE_URL`` host (exact hostname equality, so neither lookalike
    domains nor sibling hosts of the override match)."""
    if base_url_host_matches(base_url or "", "inference-api.nousresearch.com"):
        return True
    try:
        from hermes_cli.auth import _nous_inference_env_override
        override = _nous_inference_env_override()
    except Exception:
        return False
    override_host = base_url_hostname(override) if override else ""
    return bool(override_host) and base_url_hostname(base_url or "") == override_host


# Claude generations whose API keeps EVERY prior turn's thinking blocks in context (and in the prompt
# cache) when they are passed back unchanged: Opus >= 4.5, Sonnet >= 4.6, and the Fable/Mythos
# families. Earlier models (Sonnet <= 4.5, all Haiku) are "last turn only": the API strips older
# blocks itself. Anything we cannot classify is treated as keep-all, because dropping a block the
# API would have kept rewrites the cached prefix from that point (measured: ~80% of uncached input
# in a 1,393-agent run), while passing back a block the API drops costs nothing.
_LAST_TURN_ONLY_THINKING_RE = re.compile(
    r"(haiku)|(sonnet[-_ ]?4[-_.]?[0-5]\b)|(opus[-_ ]?4[-_.]?[0-4]\b)|(claude[-_ ]?3)", re.IGNORECASE
)


def _model_keeps_all_thinking(model: str | None) -> bool:
    """True when the API preserves prior-turn thinking blocks for this Claude model."""
    return not _LAST_TURN_ONLY_THINKING_RE.search(model or "")


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Providers needing ``Authorization: Bearer`` instead of ``x-api-key``: MiniMax, Azure AI
    Foundry, Palantir Foundry's LLM proxy, CommandCode, Nous Portal. Palantir/CommandCode use
    hostname matching (not substring) so ``evil.com/palantirfoundry`` paths don't trigger it."""
    normalized = _normalized_lower(base_url)
    return (
        _is_nous_portal_endpoint(base_url)
        or normalized.startswith(_MINIMAX_ANTHROPIC_PREFIXES)
        or "azure.com" in normalized
        or base_url_host_matches(normalized, "palantirfoundry.com")
        or base_url_host_matches(normalized, "api.commandcode.ai")
    )


def _base_url_needs_context_1m_beta(base_url: str | None) -> bool:
    """Endpoints that still gate 1M context behind a beta (Azure)."""
    return "azure.com" in _normalize_base_url_text(base_url).lower()


def _is_minimax_anthropic_endpoint(base_url: str | None) -> bool:
    """MiniMax's Anthropic-compatible endpoints, which reject the fine-grained-tool-streaming and
    context-1m betas (stripped even though MiniMax also uses Bearer auth)."""
    return _normalized_lower(base_url).startswith(_MINIMAX_ANTHROPIC_PREFIXES)


def _is_azure_anthropic_endpoint(base_url: str | None) -> bool:
    """Azure-hosted Anthropic Messages endpoints serving ``/anthropic``: modern Foundry
    (``*.services.ai.azure.*``) and legacy Azure OpenAI (``*.openai.azure.*``) hosts; opts them
    into ``api-version`` query plumbing. Deliberately no finite TLD allow-list, so
    sovereign/private clouds work."""
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host_padded = f".{(parsed.hostname or '').lower().rstrip('.')}."
    is_azure_host = ".services.ai.azure." in host_padded or ".openai.azure." in host_padded
    return is_azure_host and "/anthropic" in (parsed.path or "").lower()
