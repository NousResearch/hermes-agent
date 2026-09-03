"""Endpoint-family detection for Anthropic-compatible base URLs.

Hermes talks to a dozen services that speak the Anthropic Messages API but
differ in auth style, accepted beta headers, and request quirks: MiniMax,
Kimi/Moonshot, DeepSeek, OpenCode, Azure AI Foundry, the Nous portal, Bedrock.
Every one of those differences is decided by inspecting the configured base
URL, so the predicates live together here instead of being scattered through
client construction and message conversion.

Pure functions over a base-URL string - no I/O, no SDK, no credentials - which
is what lets both ``agent/anthropic_adapter.py`` and
``agent/anthropic_message_convert.py`` depend on this module without a cycle.

``agent.anthropic_adapter`` re-exports every name below.
"""

from urllib.parse import urlparse

from utils import base_url_host_matches, base_url_hostname


def _normalize_base_url_text(base_url) -> str:
    """Normalize SDK/base transport URL values to a plain string for inspection.

    Some client objects expose ``base_url`` as an ``httpx.URL`` instead of a raw
    string.  Provider/auth detection should accept either shape.
    """
    if not base_url:
        return ""
    return str(base_url).strip()


def _is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for non-Anthropic endpoints using the Anthropic Messages API.

    Third-party proxies (Microsoft Foundry, AWS Bedrock, self-hosted) authenticate
    with their own API keys via x-api-key, not Anthropic OAuth tokens. OAuth
    detection should be skipped for these endpoints.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False  # No base_url = direct Anthropic API
    normalized = normalized.rstrip("/").lower()
    if "anthropic.com" in normalized:
        return False  # Direct Anthropic API — OAuth applies
    return True  # Any other endpoint is a third-party proxy


def _is_kimi_coding_endpoint(base_url: str | None) -> bool:
    """Return True for Kimi's /coding endpoint that requires claude-code UA."""
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return normalized.rstrip("/").lower().startswith("https://api.kimi.com/coding")


def _is_opencode_endpoint(base_url: str | None) -> bool:
    """Return True for OpenCode's Zen/Go relay (opencode.ai)."""
    return base_url_host_matches(base_url or "", "opencode.ai")


# Model-name prefixes that identify the Kimi / Moonshot family.  Covers
# - official slugs: ``kimi-k2.5``, ``kimi_thinking``, ``moonshot-v1-8k``
# - common release lines: ``k1.5-...``, ``k2-thinking``, ``k25-...``, ``k2.5-...``,
#   and the bare Coding Plan slug ``k3`` (plus ``k3.x``/``k3-...`` variants)
# Matched case-insensitively against the post-``normalize_model_name`` form,
# so a caller's ``provider/vendor/model`` slug is handled the same as a
# bare name.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
    "k3.", "k3-",
)

# Bare release slugs with no separator suffix (Kimi Coding Plan serves K3
# as the exact slug ``k3``). Kept exact-match so unrelated model names that
# merely start with the same characters don't get misclassified.
_KIMI_FAMILY_EXACT_SLUGS = frozenset({"k3"})


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    # Strip vendor prefix (e.g. ``moonshotai/kimi-k2.5`` → ``kimi-k2.5``)
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    if m in _KIMI_FAMILY_EXACT_SLUGS:
        return True
    return m.startswith(_KIMI_FAMILY_MODEL_PREFIXES)


def _is_kimi_family_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """Return True for any Kimi / Moonshot Anthropic-Messages-speaking endpoint.

    Broader than ``_is_kimi_coding_endpoint`` — matches:

    - Kimi's official ``/coding`` URL (legacy check, preserved)
    - Any ``api.kimi.com`` / ``moonshot.ai`` / ``moonshot.cn`` host
    - Custom or proxied endpoints whose *model* name is in the Kimi / Moonshot
      family (``kimi-*``, ``moonshot-*``, ``k1.*``, ``k2.*``, …).  Users with
      ``api_mode: anthropic_messages`` on a private gateway fronting Kimi
      fall into this branch — the upstream still enforces Kimi's thinking
      semantics (reasoning_content required on every replayed tool-call
      message) regardless of the gateway's hostname.

    Used to decide whether to drop Anthropic's ``thinking`` kwarg and to
    preserve unsigned reasoning_content-derived thinking blocks on replay.
    See hermes-agent#13848, #17057.
    """
    if _is_kimi_coding_endpoint(base_url):
        return True
    for _domain in ("api.kimi.com", "moonshot.ai", "moonshot.cn"):
        if base_url_host_matches(base_url or "", _domain):
            return True
    if _model_name_is_kimi_family(model):
        return True
    return False


def _is_deepseek_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for DeepSeek's Anthropic-compatible endpoint.

    DeepSeek's ``/anthropic`` route speaks the Anthropic Messages protocol
    but, when thinking mode is enabled, requires the ``thinking`` blocks
    from prior assistant turns to round-trip on subsequent requests — the
    generic third-party path strips them and triggers HTTP 400::

        The content[].thinking in the thinking mode must be passed back
        to the API.

    Per DeepSeek's published compatibility matrix the blocks are unsigned
    (no Anthropic-proprietary signature, no ``redacted_thinking`` support),
    so this endpoint is handled with the same strip-signed / keep-unsigned
    policy used for Kimi's ``/coding`` endpoint.  The match is pinned to
    the ``/anthropic`` path so the OpenAI-compatible ``api.deepseek.com``
    base URL (which never reaches this adapter) is not misclassified.
    See hermes-agent#16748.
    """
    if not base_url_host_matches(base_url or "", "api.deepseek.com"):
        return False
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return "/anthropic" in normalized.rstrip("/").lower()


def _is_nous_portal_endpoint(base_url: str | None) -> bool:
    """Return True for Nous Portal's Anthropic Messages route.

    Portal serves its ``anthropic/*`` catalog natively at
    ``https://inference-api.nousresearch.com/v1/messages``.  Portal-specific
    behaviours key off this: Bearer JWT auth, verbatim catalog model ids,
    and native thinking-signature replay.

    Trusted hosts only:

    1. Prod hostname ``inference-api.nousresearch.com``
    2. The operator-set ``NOUS_INFERENCE_BASE_URL`` hostname (staging/preview)

    Lookalikes such as ``inference-api.nousresearch.com.attacker.test`` are
    rejected (hostname match, not substring).
    """
    if base_url_host_matches(base_url or "", "inference-api.nousresearch.com"):
        return True
    try:
        from hermes_cli.auth import _nous_inference_env_override

        override = _nous_inference_env_override()
    except Exception:
        return False
    if not override:
        return False
    # Exact host equality (not subdomain) so the env override can't broaden
    # into sibling hosts the operator did not set.
    override_host = base_url_hostname(override)
    return bool(override_host) and base_url_hostname(base_url or "") == override_host


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Return True for Anthropic-compatible providers that require Bearer auth.

    Some third-party /anthropic endpoints implement Anthropic's Messages API but
    require Authorization: Bearer instead of Anthropic's native x-api-key header.
    MiniMax's global and China Anthropic-compatible endpoints, Azure AI
    Foundry's Anthropic-style endpoint, Palantir Foundry's LLM proxy, and Nous
    Portal's Messages route follow this pattern.
    """
    if _is_nous_portal_endpoint(base_url):
        return True
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return (
        normalized.startswith(("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic"))
        or "azure.com" in normalized
        # Palantir Foundry LLM proxy (<org>.palantirfoundry.com/api/v2/llm/proxy/anthropic)
        # rejects x-api-key with 401 and requires Authorization: Bearer.
        # Hostname match (not substring) so e.g. evil.com/palantirfoundry
        # paths don't trigger Bearer auth.
        or base_url_host_matches(normalized, "palantirfoundry.com")
        # xKiro's /v1/messages endpoint accepts the same Bearer key as its
        # OpenAI-compatible routes.
        or base_url_host_matches(normalized, "api.xkiro.com")
        # CommandCode's /provider/v1/messages endpoint uses Bearer auth,
        # not Anthropic's native x-api-key header. Hostname match for the
        # same reason as above.
        or base_url_host_matches(normalized, "api.commandcode.ai")
    )


def _base_url_needs_context_1m_beta(base_url: str | None) -> bool:
    """Return True for endpoints that still gate 1M context behind a beta."""
    normalized = _normalize_base_url_text(base_url).lower()
    if not normalized:
        return False
    return "azure.com" in normalized


def _is_minimax_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for MiniMax's Anthropic-compatible endpoints.

    MiniMax rejects the fine-grained-tool-streaming and context-1m betas;
    those need to be stripped even though MiniMax also uses Bearer auth.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return normalized.startswith(
        ("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic")
    )


def _is_azure_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for Azure-hosted Anthropic Messages endpoints.

    Covers both the modern Foundry host family (``*.services.ai.azure.*``)
    and the legacy Azure OpenAI host family (``*.openai.azure.*``) when
    serving Anthropic's ``/anthropic`` route. Used to opt-in those hosts
    to the ``api-version`` query-param plumbing required by Azure.

    Intentionally avoids a finite allow-list of TLD suffixes so it works
    across sovereign / private Azure clouds.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = (parsed.path or "").lower()
    host_padded = f".{host}."
    is_foundry_host = ".services.ai.azure." in host_padded
    is_legacy_azoai_host = ".openai.azure." in host_padded
    return (is_foundry_host or is_legacy_azoai_host) and "/anthropic" in path
