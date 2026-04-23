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
from pathlib import Path

from hermes_constants import get_hermes_home
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from utils import normalize_proxy_env_vars

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]

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

# Models that accept the "xhigh" output_config.effort level.  Opus 4.7 added
# xhigh as a distinct level between high and max; older adaptive-thinking
# models (4.6) reject it with a 400.  Keep this substring list in sync with
# the Anthropic migration guide as new model families ship.
_XHIGH_EFFORT_SUBSTRINGS = ("4-7", "4.7")

# Models where extended thinking is deprecated/removed (4.6+ behavior: adaptive
# is the only supported mode; 4.7 additionally forbids manual thinking entirely
# and drops temperature/top_p/top_k).
_ADAPTIVE_THINKING_SUBSTRINGS = ("4-6", "4.6", "4-7", "4.7")

# Models where temperature/top_p/top_k return 400 if set to non-default values.
# This is the Opus 4.7 contract; future 4.x+ models are expected to follow it.
_NO_SAMPLING_PARAMS_SUBSTRINGS = ("4-7", "4.7")

# ── Max output token limits per Anthropic model ───────────────────────
# Source: Anthropic docs + Cline model catalog.  Anthropic's API requires
# max_tokens as a mandatory field.  Previously we hardcoded 16384, which
# starves thinking-enabled models (thinking tokens count toward the limit).
_ANTHROPIC_OUTPUT_LIMITS = {
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
    """Return True for Claude 4.6+ models that support adaptive thinking."""
    return any(v in model for v in _ADAPTIVE_THINKING_SUBSTRINGS)


def _supports_xhigh_effort(model: str) -> bool:
    """Return True for models that accept the 'xhigh' adaptive effort level.

    Opus 4.7 introduced xhigh as a distinct level between high and max.
    Pre-4.7 adaptive models (Opus/Sonnet 4.6) only accept low/medium/high/max
    and reject xhigh with an HTTP 400. Callers should downgrade xhigh→max
    when this returns False.
    """
    return any(v in model for v in _XHIGH_EFFORT_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """Return True for models that 400 on any non-default temperature/top_p/top_k.

    Opus 4.7 explicitly rejects sampling parameters; later Claude releases are
    expected to follow suit.  Callers should omit these fields entirely rather
    than passing zero/default values (the API rejects anything non-null).
    """
    return any(v in model for v in _NO_SAMPLING_PARAMS_SUBSTRINGS)


# Beta headers for enhanced features (sent with ALL auth types).
# As of Opus 4.7 (2026-04-16), both of these are GA on Claude 4.6+ — the
# beta headers are still accepted (harmless no-op) but not required. Kept
# here so older Claude (4.5, 4.1) + third-party Anthropic-compat endpoints
# that still gate on the headers continue to get the enhanced features.
# Migration guide: remove these if you no longer support ≤4.5 models.
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]
# MiniMax's Anthropic-compatible endpoints fail tool-use requests when
# the fine-grained tool streaming beta is present.  Omit it so tool calls
# fall back to the provider's default response path.
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"

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
_MCP_TOOL_PREFIX = "mcp_"


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

    Non-Anthropic keys (MiniMax, Alibaba, etc.) don't match either pattern
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

    Third-party proxies (Azure AI Foundry, AWS Bedrock, self-hosted) authenticate
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


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Return True for Anthropic-compatible providers that require Bearer auth.

    Some third-party /anthropic endpoints implement Anthropic's Messages API but
    require Authorization: Bearer *** of Anthropic's native x-api-key header.
    MiniMax's global and China Anthropic-compatible endpoints follow this pattern.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return normalized.startswith(("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic"))


def _common_betas_for_base_url(base_url: str | None) -> list[str]:
    """Return the beta headers that are safe for the configured endpoint.

    MiniMax's Anthropic-compatible endpoints (Bearer-auth) reject requests
    that include Anthropic's ``fine-grained-tool-streaming`` beta — every
    tool-use message triggers a connection error.  Strip that beta for
    Bearer-auth endpoints while keeping all other betas intact.
    """
    if _requires_bearer_auth(base_url):
        return [b for b in _COMMON_BETAS if b != _TOOL_STREAMING_BETA]
    return _COMMON_BETAS


def build_anthropic_client(api_key: str, base_url: str = None, timeout: float = None):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    If *timeout* is provided it overrides the default 900s read timeout.  The
    connect timeout stays at 10s.  Callers pass this from the per-provider /
    per-model ``request_timeout_seconds`` config so Anthropic-native and
    Anthropic-compatible providers respect the same knob as OpenAI-wire
    providers.

    Returns an anthropic.Anthropic instance.
    """
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )

    normalize_proxy_env_vars()

    from httpx import Timeout

    normalized_base_url = _normalize_base_url_text(base_url)
    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    kwargs = {
        "timeout": Timeout(timeout=float(_read_timeout), connect=10.0),
    }
    if normalized_base_url:
        kwargs["base_url"] = normalized_base_url
    common_betas = _common_betas_for_base_url(normalized_base_url)

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
        # Third-party proxies (Azure AI Foundry, AWS Bedrock, etc.) use their
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

    Auth uses the boto3 default credential chain (IAM roles, SSO, env vars).
    """
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
    )


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials from ~/.claude/.credentials.json.

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
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


def read_claude_managed_key() -> Optional[str]:
    """Read Claude's native managed key from ~/.claude.json for diagnostics only."""
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            primary_key = data.get("primaryApiKey", "")
            if isinstance(primary_key, str) and primary_key.strip():
                return primary_key.strip()
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude.json: %s", e)
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
        cred_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        # Restrict permissions (credentials file)
        cred_path.chmod(0o600)
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

    # Run interactively — stdin/stdout/stderr inherited so user can interact
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
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
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
    state = splits[1] if len(splits) > 1 else ""

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": state,
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


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
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


def convert_messages_to_anthropic(
    messages: List[Dict],
    base_url: str | None = None,
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).

    When *base_url* is provided and points to a third-party Anthropic-compatible
    endpoint, all thinking block signatures are stripped.  Signatures are
    Anthropic-proprietary — third-party endpoints cannot validate them and will
    reject them with HTTP 400 "Invalid signature in thinking block".
    """
    system = None
    result = []

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
                isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                for b in blocks
            )
            if isinstance(reasoning_content, str) and not _already_has_thinking:
                blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
            # Anthropic rejects empty assistant content
            effective = blocks or content
            if not effective or effective == "":
                effective = [{"type": "text", "text": "(empty)"}]
            result.append({"role": "assistant", "content": effective})
            continue

        if role == "tool":
            # Sanitize tool_use_id and ensure non-empty content
            result_content = content if isinstance(content, str) else json.dumps(content)
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
            continue

        # Regular user message — validate non-empty content (Anthropic rejects empty)
        if isinstance(content, list):
            converted_blocks = _convert_content_to_anthropic(content)
            # Check if all text blocks are empty
            if not converted_blocks or all(
                b.get("text", "").strip() == ""
                for b in converted_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ):
                converted_blocks = [{"type": "text", "text": "(empty message)"}]
            result.append({"role": "user", "content": converted_blocks})
        else:
            # Validate string content is non-empty
            if not content or (isinstance(content, str) and not content.strip()):
                content = "(empty message)"
            result.append({"role": "user", "content": content})

    # Strip orphaned tool_use blocks (no matching tool_result follows)
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_use" or b.get("id") in tool_result_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]

    # Strip orphaned tool_result blocks (no matching tool_use precedes them).
    # This is the mirror of the above: context compression or session truncation
    # can remove an assistant message containing a tool_use while leaving the
    # subsequent tool_result intact.  Anthropic rejects these with a 400.
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

    # Enforce strict role alternation (Anthropic rejects consecutive same-role messages)
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                # Merge consecutive user messages
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    # Mixed types — wrap string in list
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Drop thinking blocks from the *second* message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    # Mixed types — normalize both to list and merge
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    result = fixed

    # ── Thinking block signature management ──────────────────────────
    # Anthropic signs thinking blocks against the full turn content.
    # Any upstream mutation (context compression, session truncation,
    # orphan stripping, message merging) invalidates the signature,
    # causing HTTP 400 "Invalid signature in thinking block".
    #
    # Signatures are Anthropic-proprietary.  Third-party endpoints
    # (MiniMax, Azure AI Foundry, self-hosted proxies) cannot validate
    # them and will reject them outright.  When targeting a third-party
    # endpoint, strip ALL thinking/redacted_thinking blocks from every
    # assistant message — the third-party will generate its own
    # thinking blocks if it supports extended thinking.
    #
    # For direct Anthropic (strategy following clawdbot/OpenClaw):
    # 1. Strip thinking/redacted_thinking from all assistant messages
    #    EXCEPT the last one — preserves reasoning continuity on the
    #    current tool-use chain while avoiding stale signature errors.
    # 2. Downgrade unsigned thinking blocks (no signature) to text —
    #    Anthropic can't validate them and will reject them.
    # 3. Strip cache_control from thinking/redacted_thinking blocks —
    #    cache markers can interfere with signature validation.
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    _is_third_party = _is_third_party_anthropic_endpoint(base_url)
    _is_kimi = _is_kimi_coding_endpoint(base_url)

    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _is_kimi:
            # Kimi's /coding endpoint enables thinking server-side and
            # requires unsigned thinking blocks on replayed assistant
            # tool-call messages.  Strip signed Anthropic blocks (Kimi
            # can't validate signatures) but preserve the unsigned ones
            # we synthesised from reasoning_content above.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    # Anthropic-signed block — Kimi can't validate, strip
                    continue
                # Unsigned thinking (synthesised from reasoning_content) —
                # keep it: Kimi needs it for message-history validation.
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party endpoint: strip ALL thinking blocks from every
            # assistant message — signatures are Anthropic-proprietary.
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed thinking
            # blocks for reasoning continuity; downgrade unsigned ones to
            # plain text.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("type") == "redacted_thinking":
                    # Redacted blocks use 'data' for the signature payload
                    if b.get("data"):
                        new_content.append(b)
                    # else: drop — no data means it can't be validated
                elif b.get("signature"):
                    # Signed thinking block — keep it
                    new_content.append(b)
                else:
                    # Unsigned thinking — downgrade to text so it's not lost
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking/redacted_thinking
        # blocks — cache markers interfere with signature validation.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

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
    system, anthropic_messages = convert_messages_to_anthropic(messages, base_url=base_url)
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

        # 3. Prefix tool names with mcp_ (Claude Code convention)
        if anthropic_tools:
            for tool in anthropic_tools:
                if "name" in tool:
                    tool["name"] = _MCP_TOOL_PREFIX + tool["name"]

        # 4. Prefix tool names in message history (tool_use and tool_result blocks)
        for msg in anthropic_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and "name" in block:
                            if not block["name"].startswith(_MCP_TOOL_PREFIX):
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
    _is_kimi_coding = _is_kimi_coding_endpoint(base_url)
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
    # Callers (auxiliary_client, flush_memories, etc.) may set these for
    # older models; drop them here as a safety net so upstream 4.6 → 4.7
    # migrations don't require coordinated edits everywhere.
    if _forbids_sampling_params(model):
        for _sampling_key in ("temperature", "top_p", "top_k"):
            kwargs.pop(_sampling_key, None)

    # ── Fast mode (Opus 4.6 only) ────────────────────────────────────
    # Adds extra_body.speed="fast" + the fast-mode beta header for ~2.5x
    # output speed. Only for native Anthropic endpoints — third-party
    # providers would reject the unknown beta header and speed parameter.
    if fast_mode and not _is_third_party_anthropic_endpoint(base_url):
        kwargs.setdefault("extra_body", {})["speed"] = "fast"
        # Build extra_headers with ALL applicable betas (the per-request
        # extra_headers override the client-level anthropic-beta header).
        betas = list(_common_betas_for_base_url(base_url))
        if is_oauth:
            betas.extend(_OAUTH_ONLY_BETAS)
        betas.append(_FAST_MODE_BETA)
        kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}

    return kwargs


def normalize_anthropic_response(
    response,
    strip_tool_prefix: bool = False,
) -> Tuple[SimpleNamespace, str]:
    """Normalize Anthropic response to match the shape expected by AIAgent.

    Returns (assistant_message, finish_reason) where assistant_message has
    .content, .tool_calls, and .reasoning attributes.

    When *strip_tool_prefix* is True, removes the ``mcp_`` prefix that was
    added to tool names for OAuth Claude Code compatibility.
    """
    text_parts = []
    reasoning_parts = []
    reasoning_details = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)
            block_dict = _to_plain_data(block)
            if isinstance(block_dict, dict):
                reasoning_details.append(block_dict)
        elif block.type == "tool_use":
            name = block.name
            if strip_tool_prefix and name.startswith(_MCP_TOOL_PREFIX):
                name = name[len(_MCP_TOOL_PREFIX):]
            tool_calls.append(
                SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    # Map Anthropic stop_reason to OpenAI finish_reason.
    # Newer stop reasons added in Claude 4.5+ / 4.7:
    #   - refusal: the model declined to answer (cyber safeguards, CSAM, etc.)
    #   - model_context_window_exceeded: hit context limit (not max_tokens)
    # Both need distinct handling upstream — a refusal should surface to the
    # user with a clear message, and a context-window overflow should trigger
    # compression/truncation rather than be treated as normal end-of-turn.
    stop_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "length",
    }
    finish_reason = stop_reason_map.get(response.stop_reason, "stop")

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=reasoning_details or None,
        ),
        finish_reason,
    )
