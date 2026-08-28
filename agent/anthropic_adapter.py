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
from utils import base_url_host_matches, base_url_hostname, normalize_proxy_env_vars
from agent.secret_scope import get_secret as _get_secret

# This module keeps client construction and the Messages API call itself.  The
# three surfaces it used to inline now live next to it:
#
#   agent/anthropic_endpoints.py        base-URL/endpoint-family predicates
#   agent/anthropic_message_convert.py  OpenAI -> Anthropic payload conversion
#   agent/anthropic_credentials.py      credential sources, OAuth, refresh commit
#
# All three are re-exported below so long standing
# ``from agent.anthropic_adapter import resolve_anthropic_token`` (or
# ``convert_messages_to_anthropic``, ...) imports keep resolving.
from agent.anthropic_endpoints import (  # noqa: F401
    _KIMI_FAMILY_EXACT_SLUGS,
    _KIMI_FAMILY_MODEL_PREFIXES,
    _base_url_needs_context_1m_beta,
    _is_azure_anthropic_endpoint,
    _is_deepseek_anthropic_endpoint,
    _is_kimi_coding_endpoint,
    _is_kimi_family_endpoint,
    _is_minimax_anthropic_endpoint,
    _is_nous_portal_endpoint,
    _is_opencode_endpoint,
    _is_third_party_anthropic_endpoint,
    _model_name_is_kimi_family,
    _normalize_base_url_text,
    _requires_bearer_auth,
)
from agent.anthropic_message_convert import (  # noqa: F401
    _EMPTY_TEXT_PLACEHOLDER,
    _apply_assistant_cache_control_to_last_cacheable_block,
    _content_parts_to_anthropic_blocks,
    _convert_assistant_message,
    _convert_content_part_to_anthropic,
    _convert_content_to_anthropic,
    _convert_tool_message_to_result,
    _convert_user_message,
    _ensure_leading_user_turn,
    _evict_old_screenshots,
    _extract_preserved_thinking_blocks,
    _fix_blank_text_blocks_in_list,
    _image_source_from_openai_url,
    _is_bedrock_model_id,
    _manage_thinking_signatures,
    _merge_consecutive_roles,
    _normalize_tool_input_schema,
    _safe_text,
    _sanitize_replay_block,
    _sanitize_tool_id,
    _scrub_blank_text_blocks,
    _strip_orphaned_tool_blocks,
    _to_plain_data,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    normalize_model_name,
)
from agent.anthropic_credentials import (  # noqa: F401
    _OAUTH_CLIENT_ID,
    _OAUTH_REDIRECT_URI,
    _OAUTH_SCOPES,
    _OAUTH_TOKEN_URL,
    _OAUTH_TOKEN_URLS,
    _OAUTH_TOKEN_USER_AGENT,
    CredentialPersistError,
    _generate_pkce,
    _get_hermes_oauth_file,
    _getenv,
    _is_oauth_token,
    _prefer_refreshable_claude_code_token,
    _read_claude_code_credentials_from_file,
    _read_claude_code_credentials_from_keychain,
    _refresh_oauth_token,
    _resolve_anthropic_pool_token,
    _resolve_claude_code_token_from_credentials,
    _write_claude_code_credentials,
    _write_hermes_oauth_credentials,
    claude_code_credentials_path,
    is_claude_code_token_valid,
    is_rotation_consumed_uncommitted,
    mark_rotation_consumed_uncommitted,
    read_claude_code_credentials,
    read_hermes_oauth_credentials,
    refresh_anthropic_oauth_pure,
    resolve_anthropic_token,
    run_hermes_oauth_login_pure,
    run_oauth_setup_token,
)

try:
    import hermes_cli as _hermes_cli

    _HERMES_VERSION = str(_hermes_cli.__version__)
except Exception:
    _HERMES_VERSION = "0.0.0"



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
    "ultra":   "max",
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

# Adaptive Claude families that REJECT a thinking disable — thinking is
# mandatory and ``thinking: {"type": "disabled"}`` answers HTTP 400. The Portal
# catalog flags the same families with ``reasoning.mandatory``.
#
# Unlike the two lists above, the failure here is asymmetric: a missing entry
# 400s the turn, while a spurious one only leaves thinking on. When in doubt,
# add the family.
_MANDATORY_THINKING_CLAUDE_SUBSTRINGS = (
    "claude-fable",
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
    # Claude Sonnet 5
    "claude-sonnet-5":   128_000,
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

    Kimi / Moonshot models are the exception: their Anthropic-compatible
    endpoints implement the adaptive contract (``thinking.type="adaptive"``
    + ``output_config.effort``, including ``xhigh`` and ``display``).
    """
    if _model_name_is_kimi_family(model):
        return True
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


def _accepts_thinking_disable(model: str) -> bool:
    """Return True when *model* accepts an explicit thinking disable.

    Adaptive Claude models default to thinking ON, so "thinking off" only
    takes effect if we actively send ``thinking: {"type": "disabled"}`` —
    omitting the parameter leaves the upstream default in place and the model
    thinks anyway.  Reasoning-mandatory families reject the disable outright
    with an HTTP 400, so they keep the omit-everything behavior.

    Legacy manual-thinking Claude models are excluded because they need no
    disable: thinking is opt-in there via ``budget_tokens``, so not sending
    the block already means off.

    Scoped to Claude deliberately.  Kimi/Moonshot endpoints also speak the
    adaptive contract, but their documented disable behavior is omission
    (#13848) and they are not part of this bug; sending them a new parameter
    on the strength of Claude's contract would be a guess.
    """
    if not _is_claude_model(model):
        return False
    if not _supports_adaptive_thinking(model):
        return False
    m = model.lower()
    return not any(v in m for v in _MANDATORY_THINKING_CLAUDE_SUBSTRINGS)


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
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
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
        # Delegate retry to hermes's outer loop (honors Retry-After); the SDK
        # default max_retries=2 ignores it and double-retries. (#26293)
        "max_retries": 0,
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

    client = _anthropic_sdk.Anthropic(**kwargs)
    # Same env-inference trap as build_anthropic_client: auth_token-only
    # construction would otherwise also send ANTHROPIC_API_KEY as X-Api-Key.
    client.api_key = None
    return client


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
        # Delegate all rate-limit / 5xx retry to hermes's outer conversation
        # loop, which honors Retry-After. The SDK default (max_retries=2) uses
        # its own 1-2s backoff that ignores Retry-After and double-retries
        # inside our loop — burning request slots against a bucket that won't
        # refill for minutes. (#26293)
        "max_retries": 0,
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
        # Kimi's /coding endpoint requires a non-empty User-Agent to be
        # recognized as a valid Coding Agent. Originally we sent
        # ``claude-code/0.1.0`` (the minimum that avoided a 403), but the Kimi
        # team asked us to identify ourselves properly so they can attribute
        # traffic correctly. Send the same attribution header set we send to
        # OpenRouter, Vercel AI Gateway, and Fireworks:
        # HTTP-Referer + X-Title + HermesAgent User-Agent.
        kwargs["api_key"] = api_key
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
            "X-Title": "Hermes Agent",
            "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
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
            "user-agent": f"claude-code/{_get_claude_code_version()} (external, cli)",
            "x-app": "cli",
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    if _is_opencode_endpoint(base_url):
        # OpenCode identifies clients by request headers, like OpenRouter does.
        # The OpenAI-wire paths pick these up from profile.default_headers
        # (plugins/model-providers/opencode-zen), but the Anthropic Messages
        # route builds its client right here and never sees the profile. Merge
        # the same set on top of whatever auth branch ran above.
        headers = dict(kwargs.get("default_headers") or {})
        headers.setdefault("HTTP-Referer", "https://hermes-agent.nousresearch.com")
        headers.setdefault("X-Title", "Hermes Agent")
        headers.setdefault("User-Agent", f"HermesAgent/{_HERMES_VERSION}")
        kwargs["default_headers"] = headers

    client = _anthropic_sdk.Anthropic(**kwargs)
    # Bearer-only construction leaves ``api_key`` unset, so the SDK fills it
    # from ``ANTHROPIC_API_KEY`` (Hermes loads that into the process env from
    # ``~/.hermes/.env``). The result is dual auth —
    # ``X-Api-Key: sk-ant-…`` *and* ``Authorization: Bearer <portal-jwt>`` —
    # on every Portal / MiniMax / OAuth Messages request. Clear the env-filled
    # key whenever we intentionally authenticated via auth_token alone.
    if "auth_token" in kwargs and "api_key" not in kwargs:
        client.api_key = None
    return client


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
        # Delegate retry to hermes's outer loop (honors Retry-After); the SDK
        # default max_retries=2 ignores it and double-retries. (#26293)
        max_retries=0,
        default_headers={"anthropic-beta": ",".join([*_COMMON_BETAS, _CONTEXT_1M_BETA])},
    )




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

    # Nous Portal routes on its own catalog ids (``anthropic/claude-opus-4.8``);
    # normalizing to the bare Anthropic slug would make the model unresolvable
    # there. Skipping the call preserves the prefix AND the dots, so
    # ``preserve_dots`` stays irrelevant for Portal.
    if not _is_nous_portal_endpoint(base_url):
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

        # 3. Normalize tool names so NOTHING goes on the OAuth wire with a
        #    single-underscore ``mcp_`` prefix.  Anthropic's subscription/OAuth
        #    billing classifier treats a single-underscore ``mcp_`` tool name as
        #    a third-party-app fingerprint and rejects the request with HTTP 400
        #    "Third-party apps now draw from extra usage, not plan limits"
        #    (verified empirically: a single ``mcp_foo`` tool flips a request
        #    from plan-billing to the extra-usage lane; ``mcp__foo`` is accepted).
        #
        #    Two cases, both must land on the double-underscore ``mcp__`` form:
        #      a) bare Hermes-native tools (``read_file``)  -> ``mcp__read_file``
        #      b) native MCP server tools registered under their full
        #         single-underscore ``mcp_<server>_<tool>`` name
        #         (``mcp_linear_get_issue``) -> ``mcp__linear_get_issue``
        #    Case (b) is the gap that the bare ``mcp_``->``mcp__`` constant swap
        #    left open: those tools were *skipped* and stayed single-underscore,
        #    so any session with an MCP server configured still tripped the
        #    classifier. normalize_response reverses both forms via registry
        #    lookup so the dispatcher still sees the original name. GH-25255.
        def _to_oauth_wire_name(name: str) -> str:
            if name.startswith("mcp__"):
                return name  # already correct, don't double-prefix
            if name.startswith("mcp_"):
                # single-underscore native MCP tool -> promote to double
                return "mcp__" + name[len("mcp_"):]
            return _MCP_TOOL_PREFIX + name  # bare name -> mcp__<name>

        if anthropic_tools:
            for tool in anthropic_tools:
                if "name" in tool:
                    tool["name"] = _to_oauth_wire_name(tool["name"])

        # 4. Apply the same normalization to tool names in message history
        #    (tool_use blocks) so replayed turns match the wire names above.
        for msg in anthropic_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and "name" in block:
                            block["name"] = _to_oauth_wire_name(block["name"])
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
    # Kimi / Moonshot models also use adaptive thinking: their
    # Anthropic-compatible endpoints (api.moonshot.cn/anthropic,
    # api.kimi.com/coding) accept ``thinking.type="adaptive"`` +
    # ``output_config.effort``, and the replay-validation 400s that
    # originally motivated dropping the parameter (#13848) no longer
    # occur.  (Kimi on chat_completions enables thinking via extra_body
    # in the ChatCompletionsTransport — see #13503.)
    #
    # On 4.7+ the `thinking.display` field defaults to "omitted", which
    # silently hides reasoning text that Hermes surfaces in its CLI. We
    # request "summarized" so the reasoning blocks stay populated — matching
    # 4.6 behavior and preserving the activity-feed UX during long tool runs.
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            # "Thinking off". Adaptive models think by DEFAULT, so omitting the
            # parameter is not a disable — it silently leaves thinking on and
            # the user keeps paying for it. Send the disable explicitly.
            # Mandatory-thinking models reject it with a 400, so they keep the
            # omission: a silently-ignored disable beats a dead turn.
            if _accepts_thinking_disable(model):
                kwargs["thinking"] = {"type": "disabled"}
        elif "haiku" not in model.lower():
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


def _is_stream_unavailable_error(exc: Exception) -> bool:
    """Return True when an Anthropic stream call should fall back to create()."""
    err_lower = str(exc).lower()
    if "stream" in err_lower and "not supported" in err_lower:
        return True
    if "invokemodelwithresponsestream" in err_lower:
        from agent.bedrock_adapter import is_streaming_access_denied_error

        return is_streaming_access_denied_error(exc)
    return False


def create_anthropic_message(
    client: Any,
    api_kwargs: dict,
    *,
    log_prefix: str = "",
    prefer_stream: bool = True,
    on_stream_event=None,
    on_response=None,
) -> Any:
    """Create an Anthropic message, aggregating via stream when available.

    Some Anthropic-compatible gateways are SSE-only: they ignore non-streaming
    requests and return ``text/event-stream`` even for ``messages.create()``.
    The SDK can surface that as raw text, so callers that expect a Message then
    crash on ``.content``.  Prefer ``messages.stream().get_final_message()`` to
    match the main turn path, falling back to ``create()`` only for providers
    that explicitly do not support streaming, such as restricted Bedrock roles.

    ``on_stream_event``: optional callable invoked once per streamed event
    (best-effort, exceptions swallowed). Lets callers report forward progress
    to liveness watchdogs — e.g. the auxiliary compression path ticking its
    progress hook so a slow-but-generating summary model isn't treated as
    hung. Only fires on the streaming path; the ``create()`` fallback has no
    events to report.

    ``on_response``: optional callable invoked once with the underlying httpx
    response before the message is aggregated (best-effort, exceptions
    swallowed). Response *headers* carry out-of-band provider state that the
    parsed ``Message`` drops — Nous Portal's ``x-nous-credits-*`` balance family
    in particular. Only fires on the streaming path, which is the one the main
    turn loop takes.
    """
    sanitize_anthropic_kwargs(api_kwargs, log_prefix=log_prefix)

    messages_api = getattr(client, "messages", None)
    stream_fn = getattr(messages_api, "stream", None)
    if prefer_stream and callable(stream_fn):
        stream_kwargs = dict(api_kwargs)
        stream_kwargs.pop("stream", None)
        try:
            with stream_fn(**stream_kwargs) as stream:
                if callable(on_response):
                    try:
                        on_response(getattr(stream, "response", None))
                    except Exception:
                        logger.debug(
                            "%son_response callback failed",
                            log_prefix, exc_info=True,
                        )
                if callable(on_stream_event):
                    # Consume the event stream manually so each event can
                    # tick the caller's progress callback; get_final_message
                    # then returns the accumulated snapshot.
                    for _event in stream:
                        try:
                            on_stream_event(_event)
                        except Exception:
                            logger.debug(
                                "%son_stream_event callback failed",
                                log_prefix, exc_info=True,
                            )
                return stream.get_final_message()
        except Exception as exc:
            if not _is_stream_unavailable_error(exc):
                raise
            logger.debug(
                "%sAnthropic Messages stream unavailable; falling back to "
                "messages.create(): %s",
                log_prefix,
                exc,
            )

    create_kwargs = dict(api_kwargs)
    create_kwargs.pop("stream", None)
    return messages_api.create(**create_kwargs)
