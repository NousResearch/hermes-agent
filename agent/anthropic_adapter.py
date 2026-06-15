"""Anthropic Messages API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Anthropic's Messages API. Follows the same pattern as the codex_responses
adapter — all provider-specific logic is isolated here.

Auth supports:
  - Regular API keys (sk-ant-api*) → x-api-key header
  - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
  - Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json) → Bearer auth
"""

import copy
import json
import logging
import os
import platform
import secrets
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from typing import Any, Dict, List, Optional, Tuple
from utils import base_url_host_matches, normalize_proxy_env_vars

# NOTE: `import anthropic` is deliberately NOT at module top — the SDK pulls
# ~220 ms of imports (anthropic.types, anthropic.lib.tools._beta_runner, etc.)
# and the 3 usage sites (build_anthropic_client, build_anthropic_bedrock_client,
# read_claude_code_credentials_from_keychain) are all on cold user-triggered
# paths. Access via the `_get_anthropic_sdk()` accessor below, which caches
# the module after the first call and returns None on ImportError.
_anthropic_sdk: Any = ...  # sentinel — None means "tried and missing"


def _get_anthropic_sdk():
    """Return the ``anthropic`` SDK module, importing lazily. None if not installed."""
    global _anthropic_sdk
    if _anthropic_sdk is ...:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("provider.anthropic", prompt=False)
        except ImportError:
            pass
        except Exception:
            # FeatureUnavailable — fall through to ImportError handling below
            pass
        try:
            import anthropic as _sdk
            _anthropic_sdk = _sdk
        except ImportError:
            _anthropic_sdk = None
    return _anthropic_sdk

logger = logging.getLogger(__name__)

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}
# Hermes effort → Anthropic adaptive-thinking effort (output_config.effort).
# Anthropic exposes 5 levels on 4.7+: low, medium, high, xhigh, max.
# Opus/Sonnet 4.6 only expose 4 levels: low, medium, high, max — no xhigh.
# We preserve xhigh as xhigh on 4.7+ (the recommended default for coding/
# agentic work) and downgrade it to max on pre-4.7 adaptive models (which
# is the strongest level they accept).  "minimal" is a legacy alias that
# maps to low on every model.  See:
# https://platform.claude.com/docs/en/about-claude/models/migration-guide
ADAPTIVE_EFFORT_MAP = {
    "max":     "max",
    "xhigh":   "xhigh",
    "high":    "high",
    "medium":  "medium",
    "low":     "low",
    "minimal": "low",
}

# ── Anthropic thinking-mode classification ────────────────────────────
# Claude 4.6 replaced budget-based extended thinking with *adaptive* thinking,
# and 4.7 additionally forbids the manual ``thinking`` block entirely and drops
# temperature/top_p/top_k.  Newer Claude releases (4.8, and named models like
# claude-fable-5) follow the same modern contract — but they share no common
# version substring, so an allowlist of version numbers ("4.6", "4.7", …) goes
# stale the moment a model ships without a recognized number and silently
# routes it down the legacy manual-thinking path.
#
# Instead we DEFAULT unknown Claude models to the modern contract and keep an
# explicit *legacy* list of the older Claude families that still require manual
# thinking.  This mirrors _get_anthropic_max_output's "default to newest" design
# (future models are unlikely to regress to the older contract), so each new
# Claude release works without a code change.
#
# Non-Claude Anthropic-Messages models (minimax, qwen3, GLM, …) are NOT Claude,
# so they fall through to the legacy path automatically — exactly what those
# manual-thinking endpoints need.

# Older Claude families that DON'T support adaptive thinking (manual thinking
# with budget_tokens only). Substring-matched against the model name.
_LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS = (
    "claude-3",          # 3, 3.5, 3.7
    "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0",
    "claude-opus-4-2025", "claude-sonnet-4-2025",  # date-stamped 4.0 IDs
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-haiku-4-5", "claude-haiku-4.5",
)

# Older Claude families that DON'T accept the "xhigh" effort level (4.6 only
# supports low/medium/high/max). xhigh arrived with Opus 4.7. Adaptive models
# not in this list (4.7, 4.8, fable, future) accept xhigh.
_NO_XHIGH_CLAUDE_SUBSTRINGS = (
    "claude-opus-4-6", "claude-opus-4.6",
    "claude-sonnet-4-6", "claude-sonnet-4.6",
)


def _is_claude_model(model: str | None) -> bool:
    return "claude" in (model or "").lower()


_FAST_MODE_SUPPORTED_SUBSTRINGS = ("opus-4-6", "opus-4.6")

# ── Max output token limits per Anthropic model ───────────────────────
# Source: Anthropic docs + Cline model catalog.  Anthropic's API requires
# max_tokens as a mandatory field.  Previously we hardcoded 16384, which
# starves thinking-enabled models (thinking tokens count toward the limit).
_ANTHROPIC_OUTPUT_LIMITS = {
    # Mythos-class named models (claude-fable-5, …) — 1M context, reasoning
    "claude-fable":      128_000,
    # Claude 4.8
    "claude-opus-4-8":   128_000,
    # Claude 4.7
    "claude-opus-4-7":   128_000,
    # Claude 4.6
    "claude-opus-4-6":   128_000,
    "claude-sonnet-4-6":  64_000,
    # Claude 4.5
    "claude-opus-4-5":    64_000,
    "claude-sonnet-4-5":  64_000,
    "claude-haiku-4-5":   64_000,
    # Claude 4
    "claude-opus-4":      32_000,
    "claude-sonnet-4":    64_000,
    # Claude 3.7
    "claude-3-7-sonnet": 128_000,
    # Claude 3.5
    "claude-3-5-sonnet":   8_192,
    "claude-3-5-haiku":    8_192,
    # Claude 3
    "claude-3-opus":       4_096,
    "claude-3-sonnet":     4_096,
    "claude-3-haiku":      4_096,
    # Third-party Anthropic-compatible providers
    "minimax":            131_072,
    # Qwen models via DashScope Anthropic-compatible endpoint
    # DashScope enforces max_tokens ∈ [1, 65536]
    "qwen3":               65_536,
}

# For any model not in the table, assume the highest current limit.
# Future Anthropic models are unlikely to have *less* output capacity.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Look up the max output token limit for an Anthropic model.

    Uses substring matching against _ANTHROPIC_OUTPUT_LIMITS so date-stamped
    model IDs (claude-sonnet-4-5-20250929) and variant suffixes (:1m, :fast)
    resolve correctly.  Longest-prefix match wins to avoid e.g. "claude-3-5"
    matching before "claude-3-5-sonnet".

    Normalizes dots to hyphens so that model names like
    ``anthropic/claude-opus-4.6`` match the ``claude-opus-4-6`` table key.
    """
    m = model.lower().replace(".", "-")
    best_key = ""
    best_val = _ANTHROPIC_DEFAULT_OUTPUT_LIMIT
    for key, val in _ANTHROPIC_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def _resolve_positive_anthropic_max_tokens(value) -> Optional[int]:
    """Return ``value`` floored to a positive int, or ``None`` if it is not a
    finite positive number. Ported from openclaw/openclaw#66664.

    Anthropic's Messages API rejects ``max_tokens`` values that are 0,
    negative, non-integer, or non-finite with HTTP 400. Python's ``or``
    idiom (``max_tokens or fallback``) correctly catches ``0`` but lets
    negative ints and fractional floats (``-1``, ``0.5``) through to the
    API, producing a user-visible failure instead of a local error.
    """
    # Booleans are a subclass of int — exclude explicitly so ``True`` doesn't
    # silently become 1 and ``False`` doesn't become 0.
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        import math
        if not math.isfinite(value):
            return None
    except Exception:
        return None
    floored = int(value)  # truncates toward zero for floats
    return floored if floored > 0 else None


def _resolve_anthropic_messages_max_tokens(
    requested,
    model: str,
    context_length: Optional[int] = None,
) -> int:
    """Resolve the ``max_tokens`` budget for an Anthropic Messages call.

    Prefers ``requested`` when it is a positive finite number; otherwise
    falls back to the model's output ceiling. Raises ``ValueError`` if no
    positive budget can be resolved (should not happen with current model
    table defaults, but guards against a future regression where
    ``_get_anthropic_max_output`` could return ``0``).

    Separately, callers apply a context-window clamp — this resolver does
    not, to keep the positive-value contract independent of endpoint
    specifics.

    Ported from openclaw/openclaw#66664 (resolveAnthropicMessagesMaxTokens).
    """
    resolved = _resolve_positive_anthropic_max_tokens(requested)
    if resolved is not None:
        return resolved
    fallback = _get_anthropic_max_output(model)
    if fallback > 0:
        return fallback
    raise ValueError(
        f"Anthropic Messages adapter requires a positive max_tokens value for "
        f"model {model!r}; got {requested!r} and no model default resolved."
    )


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True for Claude models that use adaptive thinking (4.6+).

    Defaults *unknown* Claude models to adaptive (the modern contract) and
    only returns False for the explicit legacy list of older Claude families
    that require manual budget-based thinking. Non-Claude Anthropic-Messages
    models (minimax, qwen3, …) return False so they keep the manual path.
    """
    if not _is_claude_model(model):
        return False
    m = model.lower()
    return not any(v in m for v in _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS)


def _supports_xhigh_effort(model: str) -> bool:
    """Return True for models that accept the 'xhigh' adaptive effort level.

    Opus 4.7 introduced xhigh as a distinct level between high and max.
    Pre-4.7 adaptive models (Opus/Sonnet 4.6) only accept low/medium/high/max
    and reject xhigh with an HTTP 400. Callers should downgrade xhigh→max
    when this returns False.

    Defaults unknown adaptive Claude models to accepting xhigh (4.7+ contract);
    only the 4.6 family and legacy manual-thinking models are excluded.
    """
    if not _supports_adaptive_thinking(model):
        return False
    m = model.lower()
    return not any(v in m for v in _NO_XHIGH_CLAUDE_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """Return True for models that 400 on any non-default temperature/top_p/top_k.

    Opus 4.7 introduced this restriction; later Claude releases follow it.
    Defaults unknown Claude models to forbidding sampling params (the modern
    contract). The 4.6 family still accepts them, and the legacy manual-thinking
    families (4.5 and older) accept them too, so both are excluded. Non-Claude
    models are unaffected. Callers should omit these fields entirely rather than
    passing zero/default values (the API rejects anything non-null).
    """
    if not _is_claude_model(model):
        return False
    m = model.lower()
    # 4.6 family is adaptive but still accepts sampling params.
    if any(v in m for v in _NO_XHIGH_CLAUDE_SUBSTRINGS):
        return False
    return not any(v in m for v in _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS)


def _supports_fast_mode(model: str) -> bool:
    """Return True for models that support Anthropic Fast Mode (speed=fast).

    Per Anthropic docs, fast mode is currently supported on Opus 4.6 only.
    Sending ``speed: "fast"`` to any other Claude model (including Opus 4.7)
    returns HTTP 400. This guard prevents silently 400'ing when stale config
    or older callers leave fast mode enabled across a model upgrade.
    """
    return any(v in model for v in _FAST_MODE_SUPPORTED_SUBSTRINGS)


# Beta headers for enhanced features that are safe on ordinary/native Anthropic
# requests. As of Opus 4.7 (2026-04-16), these are GA on Claude 4.6+ — the
# beta headers are still accepted (harmless no-op) but not required. Kept
# here so older Claude (4.5, 4.1) + compatible endpoints that still gate on
# the headers continue to get the enhanced features.
#
# Do NOT include ``context-1m-2025-08-07`` here. Anthropic returns HTTP 400
# ("long context beta is not yet available for this subscription") for
# accounts without the long-context beta, which breaks normal short auxiliary
# calls like title generation/session summarization.
#
# ``context-1m-2025-08-07`` is still required to unlock the 1M context window
# on Claude Opus 4.6/4.7 and Sonnet 4.6 when served via AWS Bedrock or Azure
# AI Foundry. Add it only for those endpoint-specific paths below.
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]
# MiniMax's Anthropic-compatible endpoints fail tool-use requests when
# the fine-grained tool streaming beta is present.  Omit it so tool calls
# fall back to the provider's default response path.
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
# 1M context beta. Native Anthropic does not get this by default because some
# subscriptions reject it, but Bedrock/Azure still need it for 1M context.
_CONTEXT_1M_BETA = "context-1m-2025-08-07"

# Fast mode beta — enables the ``speed: "fast"`` request parameter for
# significantly higher output token throughput on Opus 4.6 (~2.5x).
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
_FAST_MODE_BETA = "fast-mode-2026-02-01"

# Additional beta headers required for OAuth/subscription auth.
# Matches what Claude Code (and pi-ai / OpenCode) send.
_OAUTH_ONLY_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
]

# Claude Code identity — required for OAuth requests to be routed correctly.
# Without these, Anthropic's infrastructure intermittently 500s OAuth traffic.
# The version must stay reasonably current — Anthropic rejects OAuth requests
# when the spoofed user-agent version is too far behind the actual release.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"
_claude_code_version_cache: Optional[str] = None


def _detect_claude_code_version() -> str:
    """Detect the installed Claude Code version, fall back to a static constant.

    Anthropic's OAuth infrastructure validates the user-agent version and may
    reject requests with a version that's too old.  Detecting dynamically means
    users who keep Claude Code updated never hit stale-version 400s.
    """
    import subprocess as _sp

    for cmd in ("claude", "claude-code"):
        try:
            result = _sp.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output is like "2.1.74 (Claude Code)" or just "2.1.74"
                version = result.stdout.strip().split()[0]
                if version and version[0].isdigit():
                    return version
        except Exception:
            pass
    return _CLAUDE_CODE_VERSION_FALLBACK


_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
_MCP_TOOL_PREFIX = "mcp__"


def _get_claude_code_version() -> str:
    """Lazily detect the installed Claude Code version when OAuth headers need it."""
    global _claude_code_version_cache
    if _claude_code_version_cache is None:
        _claude_code_version_cache = _detect_claude_code_version()
    return _claude_code_version_cache


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an Anthropic OAuth/setup token.

    Positively identifies Anthropic OAuth tokens by their key format:
    - ``sk-ant-`` prefix (but NOT ``sk-ant-api``) → setup tokens, managed keys
    - ``eyJ`` prefix → JWTs from the Anthropic OAuth flow
    - ``cc-`` prefix → Claude Code OAuth access tokens (from CLAUDE_CODE_OAUTH_TOKEN)

    Non-Anthropic keys (MiniMax, Alibaba, etc.) don't match any pattern
    and correctly return False.
    """
    if not key:
        return False
    # Regular Anthropic Console API keys — x-api-key auth, never OAuth
    if key.startswith("sk-ant-api"):
        return False
    # Anthropic-issued tokens (setup-tokens sk-ant-oat-*, managed keys)
    if key.startswith("sk-ant-"):
        return True
    # JWTs from Anthropic OAuth flow
    if key.startswith("eyJ"):
        return True
    # Claude Code OAuth access tokens (opaque, from CLAUDE_CODE_OAUTH_TOKEN)
    if key.startswith("cc-"):
        return True
    return False


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


# Model-name prefixes that identify the Kimi / Moonshot family.  Covers
# - official slugs: ``kimi-k2.5``, ``kimi_thinking``, ``moonshot-v1-8k``
# - common release lines: ``k1.5-...``, ``k2-thinking``, ``k25-...``, ``k2.5-...``
# Matched case-insensitively against the post-``normalize_model_name`` form,
# so a caller's ``provider/vendor/model`` slug is handled the same as a
# bare name.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
)


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    # Strip vendor prefix (e.g. ``moonshotai/kimi-k2.5`` → ``kimi-k2.5``)
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
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


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Return True for Anthropic-compatible providers that require Bearer auth.

    Some third-party /anthropic endpoints implement Anthropic's Messages API but
    require Authorization: Bearer instead of Anthropic's native x-api-key header.
    MiniMax's global and China Anthropic-compatible endpoints, and Azure AI
    Foundry's Anthropic-style endpoint follow this pattern.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return (
        normalized.startswith(("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic"))
        or "azure.com" in normalized
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


def _common_betas_for_base_url(
    base_url: str | None,
    *,
    drop_context_1m_beta: bool = False,
) -> list[str]:
    """Return the beta headers that are safe for the configured endpoint.

    MiniMax's Anthropic-compatible endpoints (Bearer-auth) reject requests
    that include Anthropic's ``fine-grained-tool-streaming`` beta — every
    tool-use message triggers a connection error. They also reject the
    1M-context beta. Azure AI Foundry's Anthropic endpoint also uses
    Bearer auth but keeps both betas (it needs the 1M beta for 1M context).

    The ``context-1m-2025-08-07`` beta is not sent to native Anthropic by
    default because some subscriptions reject it. Add it only for endpoint
    families that still require it for 1M context, currently Microsoft Foundry.
    Bedrock uses its own client helper below and opts in explicitly.

    ``drop_context_1m_beta=True`` strips the 1M-context beta from any path that
    would otherwise include it after a subscription/endpoint rejects the beta.
    """
    betas = list(_COMMON_BETAS)
    if _base_url_needs_context_1m_beta(base_url) and not drop_context_1m_beta:
        betas.append(_CONTEXT_1M_BETA)
    if _is_minimax_anthropic_endpoint(base_url):
        _stripped = {_TOOL_STREAMING_BETA, _CONTEXT_1M_BETA}
        return [b for b in betas if b not in _stripped]
    if drop_context_1m_beta:
        return [b for b in betas if b != _CONTEXT_1M_BETA]
    return betas


def _build_anthropic_client_with_bearer_hook(
    token_provider,
    base_url: str = None,
    timeout: float = None,
    *,
    drop_context_1m_beta: bool = False,
):
    """Anthropic-on-Foundry Entra ID variant of :func:`build_anthropic_client`.

    Anthropic SDK 0.86.0 stores ``api_key`` / ``auth_token`` as static
    strings; there is no callable-token contract. To get per-request
    bearer refresh (Microsoft's documented Foundry pattern), we hand
    the SDK a custom ``httpx.Client`` whose request event hook mints a
    fresh JWT from the Entra credential chain and rewrites
    ``Authorization: Bearer <jwt>`` on every outbound request. The SDK
    ignores its own auth logic when ``http_client`` is provided (the
    hook strips any pre-set Authorization).

    The placeholder ``auth_token`` is required because the SDK raises
    ``AnthropicError`` at construction if neither ``api_key`` nor
    ``auth_token`` is set — but the hook overrides it per-request so
    the placeholder value never reaches Azure.
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for Azure Foundry Anthropic-style "
            "endpoints with Entra ID auth. Install with: pip install 'anthropic>=0.39.0'"
        )

    normalize_proxy_env_vars()

    from httpx import Timeout
    from agent.azure_identity_adapter import build_bearer_http_client

    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    timeout_obj = Timeout(timeout=float(_read_timeout), connect=10.0)

    # Strip any trailing /v1 — the Anthropic SDK appends /v1/messages.
    normalized_base_url = _normalize_base_url_text(base_url)
    if normalized_base_url:
        import re as _re
        normalized_base_url = _re.sub(r"/v1/?$", "", normalized_base_url.rstrip("/"))

    http_client = build_bearer_http_client(token_provider, timeout=timeout_obj)

    kwargs = {
        "timeout": timeout_obj,
        "http_client": http_client,
        # The SDK requires *something* for api_key/auth_token. Our
        # event hook overrides Authorization per request so this value
        # is never sent. The sentinel string makes accidental leaks
        # diagnosable in logs.
        "auth_token": "entra-id-bearer-via-http-hook",
    }

    if normalized_base_url:
        if _is_azure_anthropic_endpoint(normalized_base_url) and "api-version" not in normalized_base_url:
            kwargs["base_url"] = normalized_base_url
            kwargs["default_query"] = {"api-version": "2025-04-15"}
        else:
            kwargs["base_url"] = normalized_base_url

    common_betas = _common_betas_for_base_url(
        normalized_base_url,
        drop_context_1m_beta=drop_context_1m_beta,
    )
    if common_betas:
        kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _anthropic_sdk.Anthropic(**kwargs)


def build_anthropic_client(
    api_key,
    base_url: str = None,
    timeout: float = None,
    *,
    drop_context_1m_beta: bool = False,
):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    ``api_key`` accepts either:

    * a static ``str`` — the historical contract for all key-based and
      OAuth flows.
    * a ``Callable[[], str]`` — an Entra ID bearer token provider from
      :mod:`agent.azure_identity_adapter`. The Anthropic SDK itself
      requires a static string, so when given a callable we construct
      a custom ``httpx.Client`` with a request event hook that mints a
      fresh JWT per outbound request and rewrites the ``Authorization``
      header. The SDK never sees the callable directly.

    If *timeout* is provided it overrides the default 900s read timeout.  The
    connect timeout stays at 10s.  Callers pass this from the per-provider /
    per-model ``request_timeout_seconds`` config so Anthropic-native and
    Anthropic-compatible providers respect the same knob as OpenAI-wire
    providers.

    ``drop_context_1m_beta=True`` strips ``context-1m-2025-08-07`` from the
    client-level ``anthropic-beta`` header. Used by the reactive OAuth retry
    path in ``run_agent.py`` when a subscription rejects the beta; leave at
    its default on fresh clients so 1M-capable subscriptions keep the
    capability.

    Returns an anthropic.Anthropic instance.
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )

    # Callable api_key → Entra ID bearer provider path. Delegated to a
    # helper so the existing static-key code below stays unchanged.
    if callable(api_key) and not isinstance(api_key, str):
        return _build_anthropic_client_with_bearer_hook(
            api_key, base_url, timeout,
            drop_context_1m_beta=drop_context_1m_beta,
        )

    normalize_proxy_env_vars()

    from httpx import Timeout

    normalized_base_url = _normalize_base_url_text(base_url)
    if normalized_base_url:
        import re as _re
        normalized_base_url = _re.sub(r"/v1/?$", "", normalized_base_url.rstrip("/"))
    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    kwargs = {
        "timeout": Timeout(timeout=float(_read_timeout), connect=10.0),
    }
    if normalized_base_url:
        # Azure Anthropic endpoints require an ``api-version`` query parameter.
        # Pass it via default_query so the SDK appends it to every request URL
        # without corrupting the base_url (appending it directly produces
        # malformed paths like /anthropic?api-version=.../v1/messages).
        if _is_azure_anthropic_endpoint(normalized_base_url) and "api-version" not in normalized_base_url:
            kwargs["base_url"] = normalized_base_url.rstrip("/")
            kwargs["default_query"] = {"api-version": "2025-04-15"}
        else:
            kwargs["base_url"] = normalized_base_url
    common_betas = _common_betas_for_base_url(
        normalized_base_url,
        drop_context_1m_beta=drop_context_1m_beta,
    )

    if _is_kimi_coding_endpoint(base_url):
        # Kimi's /coding endpoint requires User-Agent: claude-code/0.1.0
        # to be recognized as a valid Coding Agent. Without it, returns 403.
        # Check this BEFORE _requires_bearer_auth since both match api.kimi.com/coding.
        kwargs["api_key"] = api_key
        kwargs["default_headers"] = {
            "User-Agent": "claude-code/0.1.0",
            **( {"anthropic-beta": ",".join(common_betas)} if common_betas else {} )
        }
    elif _requires_bearer_auth(normalized_base_url):
        # Some Anthropic-compatible providers (e.g. MiniMax) expect the API key in
        # Authorization: Bearer *** for regular API keys. Route those endpoints
        # through auth_token so the SDK sends Bearer auth instead of x-api-key.
        # Check this before OAuth token shape detection because MiniMax secrets do
        # not use Anthropic's sk-ant-api prefix and would otherwise be misread as
        # Anthropic OAuth/setup tokens.
        kwargs["auth_token"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_third_party_anthropic_endpoint(base_url):
        # Third-party proxies (Microsoft Foundry, AWS Bedrock, etc.) use their
        # own API keys with x-api-key auth. Skip OAuth detection — their keys
        # don't follow Anthropic's sk-ant-* prefix convention and would be
        # misclassified as OAuth tokens.
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + Claude Code identity.
        # Anthropic routes OAuth requests based on user-agent and headers;
        # without Claude Code's fingerprint, requests get intermittent 500s.
        all_betas = common_betas + _OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {
            "anthropic-beta": ",".join(all_betas),
            "user-agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            "x-app": "cli",
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _anthropic_sdk.Anthropic(**kwargs)


def build_anthropic_bedrock_client(region: str):
    """Create an AnthropicBedrock client for Bedrock Claude models.

    Uses the Anthropic SDK's native Bedrock adapter, which provides full
    Claude feature parity: prompt caching, thinking budgets, adaptive
    thinking, fast mode — features not available via the Converse API.

    Attaches the common Anthropic beta headers as client-level defaults so
    that Bedrock-hosted Claude models get the same enhanced features as
    native Anthropic. The ``context-1m-2025-08-07`` beta in particular
    unlocks the 1M context window for Opus 4.6/4.7 on Bedrock — without
    it, Bedrock caps these models at 200K even though the Anthropic API
    serves them with 1M natively.

    Auth uses the boto3 default credential chain (IAM roles, SSO, env vars).
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Bedrock provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    if not hasattr(_anthropic_sdk, "AnthropicBedrock"):
        raise ImportError(
            "anthropic.AnthropicBedrock not available. "
            "Upgrade with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    return _anthropic_sdk.AnthropicBedrock(
        aws_region=region,
        timeout=Timeout(timeout=900.0, connect=10.0),
        default_headers={"anthropic-beta": ",".join([*_COMMON_BETAS, _CONTEXT_1M_BETA])},
    )


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read Claude Code OAuth credentials from the macOS Keychain.

    Claude Code >=2.1.114 stores credentials in the macOS Keychain under the
    service name "Claude Code-credentials" rather than (or in addition to)
    the JSON file at ~/.claude/.credentials.json.

    The password field contains a JSON string with the same claudeAiOauth
    structure as the JSON file.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Read the "Claude Code-credentials" generic password entry
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials",
             "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None

    if result.returncode != 0:
        logger.debug("Keychain: no entry found for 'Claude Code-credentials'")
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None

    oauth_data = data.get("claudeAiOauth")
    if oauth_data and isinstance(oauth_data, dict):
        access_token = oauth_data.get("accessToken", "")
        if access_token:
            return {
                "accessToken": access_token,
                "refreshToken": oauth_data.get("refreshToken", ""),
                "expiresAt": oauth_data.get("expiresAt", 0),
                "source": "macos_keychain",
            }

    return None


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials.

    Checks two sources in order:
      1. macOS Keychain (Darwin only) — "Claude Code-credentials" entry
      2. ~/.claude/.credentials.json file

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    # Try macOS Keychain first (covers Claude Code >=2.1.114)
    kc_creds = _read_claude_code_credentials_from_keychain()
    if kc_creds:
        return kc_creds

    # Fall back to JSON file
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth_data = data.get("claudeAiOauth")
            if oauth_data and isinstance(oauth_data, dict):
                access_token = oauth_data.get("accessToken", "")
                if access_token:
                    return {
                        "accessToken": access_token,
                        "refreshToken": oauth_data.get("refreshToken", ""),
                        "expiresAt": oauth_data.get("expiresAt", 0),
                        "source": "claude_code_credentials_file",
                    }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)

    return None


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def refresh_anthropic_oauth_pure(refresh_token: str, *, use_json: bool = False) -> Dict[str, Any]:
    """Refresh an Anthropic OAuth token without mutating local credential files."""
    import time
    import urllib.parse
    import urllib.request

    if not refresh_token:
        raise ValueError("refresh_token is required")

    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    if use_json:
        data = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/x-www-form-urlencoded"

    token_endpoints = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ]
    last_error = None
    for endpoint in token_endpoints:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": content_type,
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            last_error = exc
            logger.debug("Anthropic token refresh failed at %s: %s", endpoint, exc)
            continue

        access_token = result.get("access_token", "")
        if not access_token:
            raise ValueError("Anthropic refresh response was missing access_token")
        next_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in", 3600)
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "expires_at_ms": int(time.time() * 1000) + (expires_in * 1000),
        }

    if last_error is not None:
        raise last_error
    raise ValueError("Anthropic token refresh failed")


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Attempt to refresh an expired Claude Code OAuth token."""
    refresh_token = creds.get("refreshToken", "")
    if not refresh_token:
        logger.debug("No refresh token available — cannot refresh")
        return None

    try:
        refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
        _write_claude_code_credentials(
            refreshed["access_token"],
            refreshed["refresh_token"],
            refreshed["expires_at_ms"],
        )
        logger.debug("Successfully refreshed Claude Code OAuth token")
        return refreshed["access_token"]
    except Exception as e:
        logger.debug("Failed to refresh Claude Code token: %s", e)
        return None


def _write_claude_code_credentials(
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
    *,
    scopes: Optional[list] = None,
) -> None:
    """Write refreshed credentials back to ~/.claude/.credentials.json.

    The optional *scopes* list (e.g. ``["user:inference", "user:profile", ...]``)
    is persisted so that Claude Code's own auth check recognises the credential
    as valid.  Claude Code >=2.1.81 gates on the presence of ``"user:inference"``
    in the stored scopes before it will use the token.
    """
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        # Read existing file to preserve other fields
        existing = {}
        if cred_path.exists():
            existing = json.loads(cred_path.read_text(encoding="utf-8"))

        oauth_data: Dict[str, Any] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        if scopes is not None:
            oauth_data["scopes"] = scopes
        elif "claudeAiOauth" in existing and "scopes" in existing["claudeAiOauth"]:
            # Preserve previously-stored scopes when the refresh response
            # does not include a scope field.
            oauth_data["scopes"] = existing["claudeAiOauth"]["scopes"]

        existing["claudeAiOauth"] = oauth_data

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process random suffix avoids collisions between concurrent
        # writers and stale leftovers from a prior crashed write.
        _tmp_cred = cred_path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            # Create the temp file atomically at 0o600. The previous
            # write_text + post-replace chmod opened a TOCTOU window where
            # both the temp file and the destination briefly inherited the
            # process umask (commonly 0o644 = world-readable), exposing
            # Claude Code OAuth tokens to other local users between create
            # and chmod. Mirrors agent/google_oauth.py (#19673) and
            # tools/mcp_oauth.py (#21148). Parent dir (~/.claude/) is
            # owned by Claude Code itself, so we leave its mode alone.
            fd = os.open(
                str(_tmp_cred),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(_tmp_cred, cred_path)
        except OSError:
            try:
                _tmp_cred.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, IOError) as e:
        logger.debug("Failed to write refreshed credentials: %s", e)


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    if creds:
        logger.debug("Claude Code credentials expired — attempting refresh")
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            return refreshed
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.

    Hermes historically persisted setup tokens into ANTHROPIC_TOKEN. That makes
    later refresh impossible because the static env token wins before we ever
    inspect Claude Code's refreshable credential file. If we have a refreshable
    Claude Code credential record, prefer it over the static env OAuth token.
    """
    if not env_token or not _is_oauth_token(env_token) or not isinstance(creds, dict):
        return None
    if not creds.get("refreshToken"):
        return None

    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug(
            "Preferring Claude Code credential file over static env OAuth token so refresh can proceed"
        )
        return resolved
    return None


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      4. ANTHROPIC_API_KEY env var (regular API key, or legacy fallback)

    Returns the token string or None.
    """
    creds = read_claude_code_credentials()

    # 1. Hermes-managed OAuth/setup token env var
    token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, creds)
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, creds)
        if preferred:
            return preferred
        return cc_token

    # 3. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(creds)
    if resolved_claude_token:
        return resolved_claude_token

    # 4. Regular API key, or a legacy OAuth token saved in ANTHROPIC_API_KEY.
    # This remains as a compatibility fallback for pre-migration Hermes configs.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    return None


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively and return the resulting token.

    Checks multiple sources after the subprocess completes:
      1. Claude Code credential files (may be written by the subprocess)
      2. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_TOKEN env vars

    Returns the token string, or None if no credentials were obtained.
    Raises FileNotFoundError if the 'claude' CLI is not installed.
    """
    import shutil
    import subprocess

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )

    # Run interactively — stdin/stdout/stderr inherited so the user can
    # complete the OAuth login prompt. Must keep inherited stdin; the TUI-EOF
    # concern does not apply to an interactive login the user explicitly
    # invokes.  noqa: subprocess-stdin
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None

    # Check if credentials were saved to Claude Code's config files
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]

    # Check env vars that may have been set
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
_HERMES_OAUTH_FILE = get_hermes_home() / ".anthropic_oauth.json"


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    import base64
    import hashlib
    import secrets

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import secrets
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    from urllib.parse import urlencode

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print()
    print("Authorize Hermes with your Claude Pro/Max subscription.")
    print()
    print("╭─ Claude Pro/Max Authorization ────────────────────╮")
    print("│                                                   │")
    print("│  Open this link in your browser:                  │")
    print("╰───────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()

    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open_gui
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior

    if _can_open_gui():
        try:
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
        except Exception:
            pass

    print()
    print("After authorizing, you'll see a code. Paste it below.")
    print()
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not auth_code:
        print("No code entered.")
        return None

    splits = auth_code.split("#")
    code = splits[0]
    received_state = splits[1] if len(splits) > 1 else ""

    # Validate state to prevent CSRF (RFC 6749 §10.12)
    if received_state != oauth_state:
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=exchange_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")
        return None

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_ms": expires_at_ms,
    }


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    if _HERMES_OAUTH_FILE.exists():
        try:
            data = json.loads(_HERMES_OAUTH_FILE.read_text(encoding="utf-8"))
            if data.get("accessToken"):
                return data
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read Hermes OAuth credentials: %s", e)
    return None


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def _is_bedrock_model_id(model: str) -> bool:
    """Detect AWS Bedrock model IDs that use dots as namespace separators.

    Bedrock model IDs come in two forms:
    - Bare:    ``anthropic.claude-opus-4-7``
    - Regional (inference profiles): ``us.anthropic.claude-sonnet-4-5-v1:0``

    In both cases the dots separate namespace components, not version
    numbers, and must be preserved verbatim for the Bedrock API.
    """
    lower = model.lower()
    # Regional inference-profile prefixes
    if any(lower.startswith(p) for p in ("global.", "us.", "eu.", "ap.", "jp.")):
        return True
    # Bare Bedrock model IDs: provider.model-family
    if lower.startswith("anthropic."):
        return True
    return False


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format, case-insensitive)
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: claude-opus-4.6 → claude-opus-4-6), unless
      preserve_dots is True (e.g. for Alibaba/DashScope: qwen3.5-plus).
    - Preserves Bedrock model IDs (``anthropic.claude-opus-4-7``) and
      regional inference profiles (``us.anthropic.claude-*``) whose dots
      are namespace separators, not version separators.
    """
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots:
        # Bedrock model IDs use dots as namespace separators
        # (e.g. "anthropic.claude-opus-4-7", "us.anthropic.claude-*").
        # These must not be converted to hyphens.  See issue #12295.
        if _is_bedrock_model_id(model):
            return model
        # Only convert dots to hyphens for Anthropic/Claude models.
        # Non-Anthropic models (gpt-5.4, gemini-2.5, etc.) use dots
        # as part of their canonical names.  See issue #17171.
        _lower = model.lower()
        if _lower.startswith("claude-") or _lower.startswith("anthropic/"):
            model = model.replace(".", "-")
    return model


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool call ID for the Anthropic API.

    Anthropic requires IDs matching [a-zA-Z0-9_-]. Replace invalid
    characters with underscores and ensure non-empty.
    """
    import re
    if not tool_id:
        return "tool_0"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return sanitized or "tool_0"


def _normalize_tool_input_schema(schema: Any) -> Dict[str, Any]:
    """Normalize tool schemas before sending them to Anthropic.

    Anthropic's tool schema validator rejects nullable unions such as
    ``anyOf: [{"type": "string"}, {"type": "null"}]`` that Pydantic/MCP
    commonly emits for optional fields. Tool optionality is represented by
    the parent ``required`` array, so we delegate to the shared
    ``strip_nullable_unions`` helper to collapse nullable unions to the
    non-null branch while preserving metadata like description/default.

    ``keep_nullable_hint=False`` because the Anthropic validator does not
    recognize the OpenAPI-style ``nullable: true`` extension and strict
    schema-to-grammar converters may reject unknown keywords.

    Top-level ``oneOf``/``allOf``/``anyOf`` are also stripped here: the
    Anthropic API rejects union keywords at the schema root with a generic
    HTTP 400. Several upstream and plugin tools ship schemas with one of
    these keywords at the top level (commonly for Pydantic discriminated
    unions). If we land here with those keywords still present after
    nullable-union stripping, drop them and fall back to a plain object
    schema so the tool still validates at the Anthropic boundary.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    from tools.schema_sanitizer import strip_nullable_unions

    normalized = strip_nullable_unions(schema, keep_nullable_hint=False)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    # Strip top-level union keywords that Anthropic's validator rejects.
    banned = {"oneOf", "allOf", "anyOf"}
    if banned & normalized.keys():
        normalized = {k: v for k, v in normalized.items() if k not in banned}
        if "type" not in normalized:
            normalized["type"] = "object"
    if normalized.get("type") == "object" and not isinstance(normalized.get("properties"), dict):
        normalized = {**normalized, "properties": {}}
    return normalized


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    seen_names: set = set()
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        # Defensive dedup: Anthropic rejects requests with duplicate tool
        # names.  Upstream injection paths already dedup, but this guard
        # converts a hard API failure into a warning.  See: #18478
        if name and name in seen_names:
            logger.warning(
                "convert_tools_to_anthropic: duplicate tool name '%s' "
                "— dropping second occurrence",
                name,
            )
            continue
        if name:
            seen_names.add(name)
        anthropic_tool: Dict[str, Any] = {
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": _normalize_tool_input_schema(
                fn.get("parameters", {"type": "object", "properties": {}})
            ),
        }
        # Forward cache_control marker when present on the OpenAI-format
        # tool dict. Anthropic's tools array supports cache_control on the
        # last tool to cache the entire schema cross-session.
        cache_control = t.get("cache_control")
        if isinstance(cache_control, dict):
            anthropic_tool["cache_control"] = dict(cache_control)
        result.append(anthropic_tool)
    return result


def _image_source_from_openai_url(url: str) -> Dict[str, str]:
    """Convert an OpenAI-style image URL/data URL into Anthropic image source."""
    url = str(url or "").strip()
    if not url:
        return {"type": "url", "url": ""}

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    """Convert a single OpenAI-style content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type")

    if ptype == "input_text":
        block: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
    elif ptype == "text":
        # A stored Anthropic text block. Rebuild from whitelisted fields only —
        # SDK response text blocks carry output-only siblings (parsed_output,
        # citations=None) that the Messages INPUT schema rejects with HTTP 400
        # "Extra inputs are not permitted". Do NOT dict(part) it verbatim.
        block = {"type": "text", "text": part.get("text", "")}
        cits = part.get("citations")
        if isinstance(cits, list) and cits:
            block["citations"] = cits
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    if isinstance(part.get("cache_control"), dict) and "cache_control" not in block:
        block["cache_control"] = dict(part["cache_control"])
    return block


def _to_plain_data(value: Any, *, _depth: int = 0, _path: Optional[set] = None) -> Any:
    """Recursively convert SDK objects to plain Python data structures.

    Guards against circular references (``_path`` tracks ``id()`` of objects
    on the *current* recursion path) and runaway depth (capped at 20 levels).
    Uses path-based tracking so shared (but non-cyclic) objects referenced by
    multiple siblings are converted correctly rather than being stringified.
    """
    _MAX_DEPTH = 20
    if _depth > _MAX_DEPTH:
        return str(value)

    if _path is None:
        _path = set()

    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    if hasattr(value, "model_dump"):
        _path.add(obj_id)
        result = _to_plain_data(value.model_dump(), _depth=_depth + 1, _path=_path)
        _path.discard(obj_id)
        return result
    if isinstance(value, dict):
        _path.add(obj_id)
        result = {k: _to_plain_data(v, _depth=_depth + 1, _path=_path) for k, v in value.items()}
        _path.discard(obj_id)
        return result
    if isinstance(value, (list, tuple)):
        _path.add(obj_id)
        result = [_to_plain_data(v, _depth=_depth + 1, _path=_path) for v in value]
        _path.discard(obj_id)
        return result
    if hasattr(value, "__dict__"):
        _path.add(obj_id)
        result = {
            k: _to_plain_data(v, _depth=_depth + 1, _path=_path)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        _path.discard(obj_id)
        return result
    return value


def _extract_preserved_thinking_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return Anthropic thinking blocks previously preserved on the message."""
    raw_details = message.get("reasoning_details")
    if not isinstance(raw_details, list):
        return []

    preserved: List[Dict[str, Any]] = []
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        block_type = str(detail.get("type", "") or "").strip().lower()
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        preserved.append(copy.deepcopy(detail))
    return preserved


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert OpenAI-style multimodal content arrays to Anthropic blocks."""
    if not isinstance(content, list):
        return content

    converted = []
    for part in content:
        block = _convert_content_part_to_anthropic(part)
        if block is not None:
            converted.append(block)
    return converted


def _content_parts_to_anthropic_blocks(parts: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool-message content parts → Anthropic tool_result inner blocks.

    Used for multimodal tool results (e.g. computer_use screenshots). Each
    part is normalized via `_convert_content_part_to_anthropic`, then
    filtered to the block types Anthropic tool_result accepts (text + image).
    """
    if not isinstance(parts, list):
        return []
    out: List[Dict[str, Any]] = []
    for part in parts:
        block = _convert_content_part_to_anthropic(part)
        if not block:
            continue
        btype = block.get("type")
        if btype == "text":
            text_val = block.get("text")
            if isinstance(text_val, str) and text_val:
                out.append({"type": "text", "text": text_val})
        elif btype == "image":
            src = block.get("source")
            if isinstance(src, dict) and src:
                out.append({"type": "image", "source": src})
    return out


def _sanitize_replay_block(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strip output-only fields from a stored Anthropic content block so it is
    valid as REQUEST input on replay.

    The SDK response objects carry output-only attributes that the Messages
    *input* schema forbids ("Extra inputs are not permitted"): text blocks get
    ``parsed_output``/``citations`` (when null), tool_use blocks get ``caller``,
    etc. ``normalize_response`` captured blocks verbatim via ``_to_plain_data``,
    so these leak back as input on the next turn → HTTP 400.

    Whitelist per type (NOT a blacklist) so future SDK output-only fields can't
    reintroduce the bug. Returns a clean block, or None to drop it.
    """
    if not isinstance(b, dict):
        return None
    btype = b.get("type")
    if btype == "text":
        out: Dict[str, Any] = {"type": "text", "text": b.get("text", "")}
        # citations is input-valid ONLY when it's a non-empty list; the SDK
        # emits citations=None on responses, which the input schema rejects.
        cits = b.get("citations")
        if isinstance(cits, list) and cits:
            out["citations"] = cits
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "thinking":
        out = {"type": "thinking", "thinking": b.get("thinking", "")}
        if b.get("signature"):
            out["signature"] = b["signature"]
        return out
    if btype == "redacted_thinking":
        # Only valid with its data payload; drop if missing.
        return {"type": "redacted_thinking", "data": b["data"]} if b.get("data") else None
    if btype == "tool_use":
        out = {
            "type": "tool_use",
            "id": _sanitize_tool_id(b.get("id", "")),
            "name": b.get("name", ""),
            "input": b.get("input", {}),
        }
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "image":
        src = b.get("source")
        return {"type": "image", "source": src} if isinstance(src, dict) else None
    # Unknown/unsupported block type on the input path — drop rather than risk
    # another "Extra inputs are not permitted".
    return None


def _convert_assistant_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an assistant message to Anthropic content blocks.

    Handles thinking blocks, regular content, tool calls, and
    reasoning_content injection for Kimi/DeepSeek endpoints.
    """
    content = m.get("content", "")
    # Anthropic interleaved-thinking fast path: when this turn carries a
    # verbatim, order-preserving block list (set by normalize_response only
    # for turns that interleave SIGNED thinking with tool_use), replay it.
    # Each block is run through _sanitize_replay_block to strip output-only
    # SDK fields (parsed_output, caller, citations=None, …) that the Messages
    # INPUT schema forbids — replaying them verbatim caused HTTP 400 "Extra
    # inputs are not permitted" (text.parsed_output). Block ORDER is preserved
    # (the reason this channel exists); only forbidden sibling fields are
    # dropped, leaving thinking signatures and tool_use id/name/input intact.
    ordered_blocks = m.get("anthropic_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        # Re-source each tool_use input from the stored tool_calls map rather
        # than the captured block. The ordered-blocks list captures tool_use
        # input from the RAW API response (normalize_response), which is NOT
        # credential-redacted; tool_calls[].function.arguments IS redacted at
        # storage time (build_assistant_message, #19798). Replaying the raw
        # block input would resurrect a secret the model inlined into a tool
        # call (e.g. terminal(command="curl -H 'Authorization: Bearer sk-...'")
        # onto the wire, even though the same value is redacted everywhere else
        # in history. Keying by sanitized tool id preserves interleave order
        # (the reason this channel exists) while swapping in the redacted
        # input. Adapted from #36071 (replay-time tool-input re-sourcing).
        redacted_input_by_id: Dict[str, Any] = {}
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, ValueError):
                parsed_args = {}
            redacted_input_by_id[_sanitize_tool_id(tc.get("id", ""))] = parsed_args
        replayed: List[Dict[str, Any]] = []
        for b in ordered_blocks:
            clean = _sanitize_replay_block(b)
            if clean is None:
                continue
            if clean.get("type") == "tool_use":
                # Override raw (un-redacted) input with the redacted copy when
                # we have one for this id; fall back to the sanitized block
                # input only if the tool_call is missing (shape mismatch).
                redacted = redacted_input_by_id.get(clean.get("id", ""))
                if redacted is not None:
                    clean["input"] = redacted
            replayed.append(clean)
        if replayed:
            return {"role": "assistant", "content": replayed}

    blocks = _extract_preserved_thinking_blocks(m)
    if content:
        if isinstance(content, list):
            converted_content = _convert_content_to_anthropic(content)
            if isinstance(converted_content, list):
                blocks.extend(converted_content)
        else:
            blocks.append({"type": "text", "text": str(content)})
    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, ValueError):
            parsed_args = {}
        blocks.append({
            "type": "tool_use",
            "id": _sanitize_tool_id(tc.get("id", "")),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })
    # Kimi's /coding endpoint (Anthropic protocol) requires assistant
    # tool-call messages to carry reasoning_content when thinking is
    # enabled server-side.  Preserve it as a thinking block so Kimi
    # can validate the message history.  See hermes-agent#13848.
    #
    # Accept empty string "" — _copy_reasoning_content_for_api()
    # injects "" as a tier-3 fallback for Kimi tool-call messages
    # that had no reasoning.  Kimi requires the field to exist, even
    # if empty.
    #
    # Prepend (not append): Anthropic protocol requires thinking
    # blocks before text and tool_use blocks.
    #
    # Guard: only add when reasoning_details didn't already contribute
    # thinking blocks.  On native Anthropic, reasoning_details produces
    # signed thinking blocks — adding another unsigned one from
    # reasoning_content would create a duplicate (same text) that gets
    # downgraded to a spurious text block on the last assistant message.
    reasoning_content = m.get("reasoning_content")
    _already_has_thinking = any(
        isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        for b in blocks
    )
    if isinstance(reasoning_content, str) and not _already_has_thinking:
        blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
    # Anthropic rejects empty assistant content
    effective = blocks or content
    if not effective or effective == "":
        effective = [{"type": "text", "text": "(empty)"}]
    return {"role": "assistant", "content": effective}


def _convert_tool_message_to_result(
    result: List[Dict[str, Any]], m: Dict[str, Any]
) -> None:
    """Convert a tool message to an Anthropic tool_result, merging consecutive
    results into one user message.

    Mutates ``result`` in place — either appends a new user message or extends
    the trailing user message's tool_result list.
    """
    content = m.get("content", "")
    multimodal_blocks: Optional[List[Dict[str, Any]]] = None
    if isinstance(content, dict) and content.get("_multimodal"):
        multimodal_blocks = _content_parts_to_anthropic_blocks(
            content.get("content") or []
        )
        # Fallback text if the conversion produced nothing usable.
        if not multimodal_blocks and content.get("text_summary"):
            multimodal_blocks = [
                {"type": "text", "text": str(content["text_summary"])}
            ]
    elif isinstance(content, list):
        converted = _content_parts_to_anthropic_blocks(content)
        if any(b.get("type") == "image" for b in converted):
            multimodal_blocks = converted
    # Back-compat: some callers stash blocks under a private key.
    if multimodal_blocks is None:
        stashed = m.get("_anthropic_content_blocks")
        if isinstance(stashed, list) and stashed:
            text_content = content if isinstance(content, str) and content.strip() else None
            multimodal_blocks = (
                [{"type": "text", "text": text_content}] + stashed
                if text_content else list(stashed)
            )

    if multimodal_blocks:
        result_content: Any = multimodal_blocks
    elif isinstance(content, str):
        result_content = content
    else:
        result_content = json.dumps(content) if content else "(no output)"
    if not result_content:
        result_content = "(no output)"
    tool_result = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": result_content,
    }
    if isinstance(m.get("cache_control"), dict):
        tool_result["cache_control"] = dict(m["cache_control"])
    # Merge consecutive tool results into one user message
    if (
        result
        and result[-1]["role"] == "user"
        and isinstance(result[-1]["content"], list)
        and result[-1]["content"]
        and result[-1]["content"][0].get("type") == "tool_result"
    ):
        result[-1]["content"].append(tool_result)
    else:
        result.append({"role": "user", "content": [tool_result]})


def _convert_user_message(content: Any) -> Dict[str, Any]:
    """Validate and convert a user message to anthropic format."""
    if isinstance(content, list):
        converted_blocks = _convert_content_to_anthropic(content)
        if not converted_blocks or all(
            b.get("text", "").strip() == ""
            for b in converted_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ):
            converted_blocks = [{"type": "text", "text": "(empty message)"}]
        return {"role": "user", "content": converted_blocks}
    else:
        if not content or (isinstance(content, str) and not content.strip()):
            content = "(empty message)"
        return {"role": "user", "content": content}


def _strip_orphaned_tool_blocks(result: List[Dict[str, Any]]) -> None:
    """Strip tool_use blocks with no matching tool_result, and vice versa.

    Context compression or session truncation can remove either side of a
    tool-call pair.  Anthropic rejects both orphans with HTTP 400.

    Mutates ``result`` in place.
    """
    # Strip orphaned tool_use blocks (no matching tool_result follows)
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            kept = [
                b
                for b in m["content"]
                if b.get("type") != "tool_use" or b.get("id") in tool_result_ids
            ]
            # If stripping an orphaned tool_use mutated a turn that also carries a
            # signed thinking block, that block's Anthropic signature was computed
            # against the ORIGINAL (un-stripped) turn content and is now invalid.
            # Anthropic rejects the replayed turn with HTTP 400 "thinking blocks in
            # the latest assistant message cannot be modified".  Flag the turn so
            # _manage_thinking_signatures can demote the dead signature instead of
            # replaying it verbatim.  See hermes-agent: extended-thinking + parallel
            # tool batch interrupted mid-flight → non-retryable 400 crash-loop.
            if len(kept) != len(m["content"]) and any(
                isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
                for b in m["content"]
            ):
                m["_thinking_signature_invalidated"] = True
            m["content"] = kept
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]

    # Strip orphaned tool_result blocks (no matching tool_use precedes them)
    tool_use_ids = set()
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_use":
                    tool_use_ids.add(block.get("id"))
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_result" or b.get("tool_use_id") in tool_use_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool result removed)"}]


def _merge_consecutive_roles(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-role messages to enforce Anthropic alternation.

    Returns a new list (caller must rebind ``result``).
    """
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Propagate the orphan-strip signature-invalidation flag onto the
                # surviving (prev) dict so _manage_thinking_signatures still sees it.
                if m.get("_thinking_signature_invalidated"):
                    fixed[-1]["_thinking_signature_invalidated"] = True
                # Drop thinking blocks from the *second* message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    return fixed


def _manage_thinking_signatures(
    result: List[Dict[str, Any]], base_url: str | None, model: str | None
) -> None:
    """Strip or preserve thinking blocks based on endpoint type.

    Anthropic signs thinking blocks against the full turn content.
    Any upstream mutation (context compression, session truncation, orphan
    stripping, message merging) invalidates the signature, causing HTTP 400
    "Invalid signature in thinking block".

    Signatures are Anthropic-proprietary.  Third-party endpoints (MiniMax,
    Azure AI Foundry, AWS Bedrock, self-hosted proxies) cannot validate them
    and will reject them outright.  Kimi's /coding and DeepSeek's /anthropic
    endpoints speak the Anthropic protocol upstream but require unsigned
    thinking blocks (synthesised from ``reasoning_content``) to round-trip on
    replayed assistant tool-call messages.  See hermes-agent#13848 (Kimi) and
    hermes-agent#16748 (DeepSeek).

    Mutates ``result`` in place.
    """
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    _is_third_party = _is_third_party_anthropic_endpoint(base_url)
    # Kimi / DeepSeek share a contract: strip signed Anthropic blocks
    # (neither upstream can validate Anthropic signatures), preserve unsigned
    # ones synthesised from reasoning_content.  See #13848, #16748.
    _preserve_unsigned_thinking = (
        _is_kimi_family_endpoint(base_url, model)
        or _is_deepseek_anthropic_endpoint(base_url)
    )

    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _preserve_unsigned_thinking:
            # Kimi / DeepSeek: strip signed, preserve unsigned.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    # Signed (or redacted-with-data) — upstream can't validate, strip.
                    continue
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party: strip ALL thinking blocks (signatures are proprietary).
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed, downgrade unsigned
            # to text so the reasoning isn't lost.
            #
            # Exception: if orphan-stripping (or another structural mutation) removed
            # a tool_use block from THIS turn, every thinking signature on it was
            # computed against the original turn content and is now dead.  Anthropic
            # rejects the turn either way — replaying the signed block 400s with
            # "thinking blocks in the latest assistant message cannot be modified",
            # and a bare signed block with no following tool_use is also invalid.
            # Demote ALL thinking blocks on this turn to text so the turn replays
            # cleanly and the model can re-plan from the surviving tool results.
            signature_dead = bool(m.get("_thinking_signature_invalidated"))
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if signature_dead:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
                    continue
                if b.get("type") == "redacted_thinking":
                    # Redacted blocks use 'data' for the signature payload —
                    # drop the block when 'data' is missing (can't be validated).
                    if b.get("data"):
                        new_content.append(b)
                elif b.get("signature"):
                    new_content.append(b)
                else:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking/redacted_thinking
        # blocks — cache markers interfere with signature validation.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

        # Drop the internal bookkeeping flag — it must never reach the API payload.
        m.pop("_thinking_signature_invalidated", None)


def _evict_old_screenshots(result: List[Dict[str, Any]]) -> None:
    """Keep only the most recent ``_MAX_KEEP_IMAGES`` computer-use screenshots.

    Base64 images cost ~1,465 tokens each and accumulate across tool calls.
    Walk backward, keep the most recent N, replace older ones with a placeholder.

    Mutates ``result`` in place.
    """
    _MAX_KEEP_IMAGES = 3
    _image_count = 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            has_image = any(
                isinstance(b, dict) and b.get("type") == "image"
                for b in inner
            )
            if not has_image:
                continue
            _image_count += 1
            if _image_count > _MAX_KEEP_IMAGES:
                block["content"] = [
                    b if b.get("type") != "image"
                    else {"type": "text", "text": "[screenshot removed to save context]"}
                    for b in inner
                ]


def convert_messages_to_anthropic(
    messages: List[Dict],
    base_url: str | None = None,
    model: str | None = None,
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).

    When *base_url* is provided and points to a third-party Anthropic-compatible
    endpoint, all thinking block signatures are stripped.  Signatures are
    Anthropic-proprietary — third-party endpoints cannot validate them and will
    reject them with HTTP 400 "Invalid signature in thinking block".

    When *model* is provided and matches the Kimi / Moonshot family (or
    *base_url* is a Kimi / Moonshot host), unsigned thinking blocks
    synthesised from ``reasoning_content`` are preserved on replayed
    assistant tool-call messages — Kimi requires the field to exist, even
    if empty.
    """
    system = None
    result: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                # Preserve cache_control markers on content blocks
                has_cache = any(
                    p.get("cache_control") for p in content if isinstance(p, dict)
                )
                if has_cache:
                    system = [p for p in content if isinstance(p, dict)]
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            result.append(_convert_assistant_message(m))
            continue

        if role == "tool":
            _convert_tool_message_to_result(result, m)
            continue

        # Regular user message
        result.append(_convert_user_message(content))

    _strip_orphaned_tool_blocks(result)
    result = _merge_consecutive_roles(result)
    _manage_thinking_signatures(result, base_url, model)
    _evict_old_screenshots(result)

    return system, result


def build_anthropic_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    max_tokens: Optional[int],
    reasoning_config: Optional[Dict[str, Any]],
    tool_choice: Optional[str] = None,
    is_oauth: bool = False,
    preserve_dots: bool = False,
    context_length: Optional[int] = None,
    base_url: str | None = None,
    fast_mode: bool = False,
    drop_context_1m_beta: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for anthropic.messages.create().

    Naming note — two distinct concepts, easily confused:
      max_tokens     = OUTPUT token cap for a single response.
                       Anthropic's API calls this "max_tokens" but it only
                       limits the *output*.  Anthropic's own native SDK
                       renamed it "max_output_tokens" for clarity.
      context_length = TOTAL context window (input tokens + output tokens).
                       The API enforces: input_tokens + max_tokens ≤ context_length.
                       Stored on the ContextCompressor; reduced on overflow errors.

    When *max_tokens* is None the model's native output ceiling is used
    (e.g. 128K for Opus 4.6, 64K for Sonnet 4.6).

    When *context_length* is provided and the model's native output ceiling
    exceeds it (e.g. a local endpoint with an 8K window), the output cap is
    clamped to context_length − 1.  This only kicks in for unusually small
    context windows; for full-size models the native output cap is always
    smaller than the context window so no clamping happens.
    NOTE: this clamping does not account for prompt size — if the prompt is
    large, Anthropic may still reject the request.  The caller must detect
    "max_tokens too large given prompt" errors and retry with a smaller cap
    (see parse_available_output_tokens_from_error + _ephemeral_max_output_tokens).

    When *is_oauth* is True, applies Claude Code compatibility transforms:
    system prompt prefix, tool name prefixing, and prompt sanitization.

    When *preserve_dots* is True, model name dots are not converted to hyphens
    (for Alibaba/DashScope anthropic-compatible endpoints: qwen3.5-plus).

    When *base_url* points to a third-party Anthropic-compatible endpoint,
    thinking block signatures are stripped (they are Anthropic-proprietary).

    When *fast_mode* is True, adds ``extra_body["speed"] = "fast"`` and the
    fast-mode beta header for ~2.5x faster output throughput on Opus 4.6.
    Currently only supported on native Anthropic endpoints (not third-party
    compatible ones).
    """
    system, anthropic_messages = convert_messages_to_anthropic(
        messages, base_url=base_url, model=model
    )
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []

    model = normalize_model_name(model, preserve_dots=preserve_dots)
    # effective_max_tokens = output cap for this call (≠ total context window)
    # Use the resolver helper so non-positive values (negative ints,
    # fractional floats, NaN, non-numeric) fail locally with a clear error
    # rather than 400-ing at the Anthropic API. See openclaw/openclaw#66664.
    effective_max_tokens = _resolve_anthropic_messages_max_tokens(
        max_tokens, model, context_length=context_length
    )

    # Clamp output cap to fit inside the total context window.
    # Only matters for small custom endpoints where context_length < native
    # output ceiling.  For standard Anthropic models context_length (e.g.
    # 200K) is always larger than the output ceiling (e.g. 128K), so this
    # branch is not taken.
    if context_length and effective_max_tokens > context_length:
        effective_max_tokens = max(context_length - 1, 1)

    # ── OAuth: Claude Code identity ──────────────────────────────────
    if is_oauth:
        # 1. Prepend Claude Code system prompt identity
        cc_block = {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
        if isinstance(system, list):
            system = [cc_block] + system
        elif isinstance(system, str) and system:
            system = [cc_block, {"type": "text", "text": system}]
        else:
            system = [cc_block]

        # 2. Sanitize system prompt — replace product name references
        #    to avoid Anthropic's server-side content filters.
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                text = text.replace("Hermes Agent", "Claude Code")
                text = text.replace("Hermes agent", "Claude Code")
                text = text.replace("hermes-agent", "claude-code")
                text = text.replace("Nous Research", "Anthropic")
                block["text"] = text

        # 3. Prefix tool names with mcp__ (Claude Code convention)
        #    Skip names that already begin with the marker — native MCP server
        #    tools (from mcp_servers: in config.yaml) are registered under their
        #    full mcp_<server>_<tool> name and would double-prefix otherwise,
        #    breaking round-trip registry lookup in normalize_response. GH-25255.
        #    Check both single-underscore (legacy MCP) and double-underscore
        #    (new prefix) to avoid double-prefixing either format.
        _MCP_SKIP = ("mcp__", "mcp_")
        if anthropic_tools:
            for tool in anthropic_tools:
                if "name" in tool and not tool["name"].startswith(_MCP_SKIP):
                    tool["name"] = _MCP_TOOL_PREFIX + tool["name"]

        # 4. Prefix tool names in message history (tool_use and tool_result blocks)
        for msg in anthropic_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and "name" in block:
                            if not block["name"].startswith(_MCP_SKIP):
                                block["name"] = _MCP_TOOL_PREFIX + block["name"]
                        elif block.get("type") == "tool_result" and "tool_use_id" in block:
                            pass  # tool_result uses ID, not name

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max_tokens,
    }

    if system:
        kwargs["system"] = system

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        # Map OpenAI tool_choice to Anthropic format
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            # Anthropic has no tool_choice "none" — omit tools entirely to prevent use
            kwargs.pop("tools", None)
        elif isinstance(tool_choice, str):
            # Specific tool name
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Map reasoning_config to Anthropic's thinking parameter.
    # Claude 4.6+ models use adaptive thinking + output_config.effort.
    # Older models use manual thinking with budget_tokens.
    # MiniMax Anthropic-compat endpoints support thinking (manual mode only,
    # not adaptive).  Haiku does NOT support extended thinking — skip entirely.
    #
    # Kimi's /coding endpoint speaks the Anthropic Messages protocol but has
    # its own thinking semantics: when ``thinking.enabled`` is sent, Kimi
    # validates the message history and requires every prior assistant
    # tool-call message to carry OpenAI-style ``reasoning_content``.  The
    # Anthropic path never populates that field, and
    # ``convert_messages_to_anthropic`` strips all Anthropic thinking blocks
    # on third-party endpoints — so the request fails with HTTP 400
    # "thinking is enabled but reasoning_content is missing in assistant
    # tool call message at index N".  Kimi's reasoning is driven server-side
    # on the /coding route, so skip Anthropic's thinking parameter entirely
    # for that host.  (Kimi on chat_completions enables thinking via
    # extra_body in the ChatCompletionsTransport — see #13503.)
    #
    # On 4.7+ the `thinking.display` field defaults to "omitted", which
    # silently hides reasoning text that Hermes surfaces in its CLI. We
    # request "summarized" so the reasoning blocks stay populated — matching
    # 4.6 behavior and preserving the activity-feed UX during long tool runs.
    _is_kimi_coding = _is_kimi_family_endpoint(base_url, model)
    if reasoning_config and isinstance(reasoning_config, dict) and not _is_kimi_coding:
        if reasoning_config.get("enabled") is not False and "haiku" not in model.lower():
            effort = str(reasoning_config.get("effort", "medium")).lower()
            budget = THINKING_BUDGET.get(effort, 8000)
            if _supports_adaptive_thinking(model):
                kwargs["thinking"] = {
                    "type": "adaptive",
                    "display": "summarized",
                }
                adaptive_effort = ADAPTIVE_EFFORT_MAP.get(effort, "medium")
                # Downgrade xhigh→max on models that don't list xhigh as a
                # supported level (Opus/Sonnet 4.6). Opus 4.7+ keeps xhigh.
                if adaptive_effort == "xhigh" and not _supports_xhigh_effort(model):
                    adaptive_effort = "max"
                kwargs["output_config"] = {
                    "effort": adaptive_effort,
                }
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # Anthropic requires temperature=1 when thinking is enabled on older models
                kwargs["temperature"] = 1
                kwargs["max_tokens"] = max(effective_max_tokens, budget + 4096)

    # ── Strip sampling params on 4.7+ ─────────────────────────────────
    # Opus 4.7 rejects any non-default temperature/top_p/top_k with a 400.
    # Callers (auxiliary_client, etc.) may set these for older models;
    # drop them here as a safety net so upstream 4.6 → 4.7 migrations
    # don't require coordinated edits everywhere.
    if _forbids_sampling_params(model):
        for _sampling_key in ("temperature", "top_p", "top_k"):
            kwargs.pop(_sampling_key, None)

    # ── Fast mode (Opus 4.6 only) ────────────────────────────────────
    # Adds extra_body.speed="fast" + the fast-mode beta header for ~2.5x
    # output speed. Per Anthropic docs, fast mode is only supported on
    # Opus 4.6 — Opus 4.7 and other models 400 on the speed parameter.
    # Only for native Anthropic endpoints — third-party providers would
    # reject the unknown beta header and speed parameter.
    if (
        fast_mode
        and not _is_third_party_anthropic_endpoint(base_url)
        and _supports_fast_mode(model)
    ):
        kwargs.setdefault("extra_body", {})["speed"] = "fast"
        # Build extra_headers with ALL applicable betas (the per-request
        # extra_headers override the client-level anthropic-beta header).
        betas = list(_common_betas_for_base_url(
            base_url,
            drop_context_1m_beta=drop_context_1m_beta,
        ))
        if is_oauth:
            betas.extend(_OAUTH_ONLY_BETAS)
        betas.append(_FAST_MODE_BETA)
        kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}

    return kwargs


# Keys that belong exclusively to the OpenAI Responses / Codex API shape.
# The Anthropic Messages SDK (``messages.create()`` / ``messages.stream()``)
# raises ``TypeError: ... got an unexpected keyword argument`` on any of them.
_RESPONSES_ONLY_KWARGS = frozenset(
    {"instructions", "input", "store", "parallel_tool_calls"}
)


def sanitize_anthropic_kwargs(api_kwargs: Any, *, log_prefix: str = "") -> Any:
    """Drop Responses-API-only keys before an Anthropic Messages SDK call.

    Defensive boundary guard for #31673: under rare api_mode-flip races
    (e.g. a concurrent auxiliary call mutating a shared agent between the
    kwargs build and the stream dispatch), a Responses-shaped payload
    carrying ``instructions=`` can reach ``messages.stream()`` /
    ``messages.create()``. The Anthropic SDK rejects it with a
    non-retryable ``TypeError`` that nukes the whole turn and propagates
    the entire fallback chain.

    Mutates ``api_kwargs`` in place and returns it. When a foreign key is
    present we log a WARNING so the underlying race stays visible in the
    wild instead of being silently papered over.
    """
    if not isinstance(api_kwargs, dict):
        return api_kwargs
    leaked = _RESPONSES_ONLY_KWARGS.intersection(api_kwargs)
    if leaked:
        for _key in leaked:
            api_kwargs.pop(_key, None)
        logger.warning(
            "%sStripped Responses-only kwarg(s) %s from an Anthropic Messages "
            "call (api_mode flip race — see #31673). The call will proceed; "
            "this breadcrumb means a kwargs build ran under a Responses "
            "api_mode while dispatch ran under anthropic_messages.",
            log_prefix,
            sorted(leaked),
        )
    return api_kwargs
