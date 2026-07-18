"""Amazon Bedrock Mantle helpers for Hermes Agent.

Mantle is separate from the native Bedrock Converse provider:

- OSS / third-party models: OpenAI-compatible
  ``https://bedrock-mantle.<region>.api.aws/v1``
- Claude models (optional path): Anthropic Messages-compatible
  ``https://bedrock-mantle.<region>.api.aws/anthropic``

Auth (same bearer family as Bedrock API keys):

1. Explicit ``AWS_BEARER_TOKEN_BEDROCK`` when set
2. Else mint a short-lived bearer from the AWS default credential chain
   via ``aws-bedrock-token-generator`` (optional dependency)

Reference facts from OpenClaw's amazon-bedrock-mantle extension and AWS
Bedrock API-key docs — reimplemented for Hermes, not ported.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PROVIDER_ID = "amazon-bedrock-mantle"
DEFAULT_REGION = "us-east-1"

MANTLE_SUPPORTED_REGIONS = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "ap-northeast-1",
        "ap-south-1",
        "ap-southeast-3",
        "eu-central-1",
        "eu-west-1",
        "eu-west-2",
        "eu-south-1",
        "eu-north-1",
        "sa-east-1",
    }
)

# Curated fallbacks when /v1/models is unreachable. Prefer tool-capable OSS ids.
FALLBACK_OSS_MODELS = (
    "gpt-oss-120b",
    "qwen3-coder-480b-a35b",
    "qwen3-235b-a22b",
    "deepseek-v3.2",
    "kimi-k2.5",
    "glm-4.7",
)

# Claude rows OpenClaw appends after discovery (Anthropic Messages route).
FALLBACK_CLAUDE_MODELS = (
    "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-mythos-5",
    "anthropic.claude-mythos-preview",
)

# IAM token cache: region → (token, expires_at_epoch)
_iam_token_cache: dict[str, tuple[str, float]] = {}
_IAM_TOKEN_TTL_S = 7200.0  # 2h — matches AWS / OpenClaw request lifetime
_IAM_TOKEN_SKEW_S = 120.0  # refresh a bit early

# Discovery cache: region → (model_ids, expires_at_epoch)
_discovery_cache: dict[str, tuple[list[str], float]] = {}
_DISCOVERY_TTL_S = 3600.0


def mantle_endpoint(region: str) -> str:
    """Return Mantle origin for a region (no path suffix)."""
    r = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
    return f"https://bedrock-mantle.{r}.api.aws"


def mantle_openai_base_url(region: str) -> str:
    return f"{mantle_endpoint(region)}/v1"


def mantle_anthropic_base_url(region: str) -> str:
    """Anthropic Messages-compatible Mantle route."""
    return f"{mantle_endpoint(region)}/anthropic"


def resolve_mantle_region(
    *,
    env: Optional[dict[str, str]] = None,
    config_region: str | None = None,
) -> str:
    """Region priority: explicit config → AWS_REGION → AWS_DEFAULT_REGION → us-east-1."""
    if config_region and str(config_region).strip():
        return str(config_region).strip()
    e = env if env is not None else os.environ
    for key in ("AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_MANTLE_REGION"):
        val = (e.get(key) or "").strip()
        if val:
            return val
    try:
        from agent.bedrock_adapter import resolve_bedrock_region

        return resolve_bedrock_region() or DEFAULT_REGION
    except Exception:
        return DEFAULT_REGION


def is_supported_mantle_region(region: str) -> bool:
    return (region or "").strip() in MANTLE_SUPPORTED_REGIONS


def resolve_explicit_bearer_token(env: Optional[dict[str, str]] = None) -> str | None:
    e = env if env is not None else os.environ
    token = (e.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    return token or None


def _get_cached_iam_token(region: str, now: float | None = None) -> str | None:
    now = time.time() if now is None else now
    entry = _iam_token_cache.get(region)
    if not entry:
        return None
    token, expires_at = entry
    if now + _IAM_TOKEN_SKEW_S >= expires_at:
        _iam_token_cache.pop(region, None)
        return None
    return token


def mint_iam_bearer_token(
    region: str,
    *,
    now: float | None = None,
    provider_factory: Callable[..., Any] | None = None,
) -> str | None:
    """Mint a Mantle/Bedrock bearer from the AWS credential chain.

    Uses ``aws-bedrock-token-generator`` when installed. Returns None if the
    package is missing or credentials are unavailable.
    """
    now = time.time() if now is None else now
    cached = _get_cached_iam_token(region, now=now)
    if cached:
        return cached

    try:
        if provider_factory is not None:
            token = provider_factory(region=region)
        else:
            from aws_bedrock_token_generator import provide_token  # type: ignore

            # Official Python helper; region is taken from the env/session.
            # Some versions accept region= — try both call shapes.
            try:
                token = provide_token(region=region)
            except TypeError:
                token = provide_token()
        token = (token or "").strip()
        if not token:
            return None
        _iam_token_cache[region] = (token, now + _IAM_TOKEN_TTL_S)
        return token
    except ImportError:
        logger.debug(
            "aws-bedrock-token-generator not installed; "
            "set AWS_BEARER_TOKEN_BEDROCK or pip install aws-bedrock-token-generator"
        )
        return None
    except Exception as exc:
        logger.debug("Mantle IAM token mint failed for region %s: %s", region, exc)
        return None


def resolve_mantle_bearer_token(
    region: str,
    *,
    env: Optional[dict[str, str]] = None,
    allow_iam_mint: bool = True,
    provider_factory: Callable[..., Any] | None = None,
) -> str | None:
    """Resolve bearer: explicit env first, then optional IAM mint."""
    explicit = resolve_explicit_bearer_token(env)
    if explicit:
        return explicit
    if not allow_iam_mint:
        return None
    return mint_iam_bearer_token(region, provider_factory=provider_factory)


def has_mantle_credentials(env: Optional[dict[str, str]] = None) -> bool:
    """True when an explicit bearer is set or AWS credentials might mint one."""
    if resolve_explicit_bearer_token(env):
        return True
    try:
        from agent.bedrock_adapter import has_aws_credentials

        return bool(has_aws_credentials())
    except Exception:
        e = env if env is not None else os.environ
        return bool(
            (e.get("AWS_ACCESS_KEY_ID") or "").strip()
            and (e.get("AWS_SECRET_ACCESS_KEY") or "").strip()
        )


def is_mantle_claude_model(model: str | None) -> bool:
    """Whether the model should use Mantle's Anthropic Messages route."""
    m = (model or "").strip().lower()
    if not m:
        return False
    # Strip optional provider prefix
    if "/" in m:
        m = m.split("/", 1)[-1]
    if m.startswith("anthropic."):
        return True
    if "claude" in m:
        return True
    return False


def is_mantle_hostname(hostname: str | None) -> bool:
    h = (hostname or "").strip().lower()
    return h.startswith("bedrock-mantle.") and h.endswith(".api.aws")


def discover_mantle_models(
    region: str,
    bearer_token: str,
    *,
    timeout: float = 12.0,
    now: float | None = None,
    fetch_fn: Callable[..., Any] | None = None,
) -> list[str]:
    """GET ``/v1/models`` and return model ids. Cached per region for 1h."""
    now = time.time() if now is None else now
    cached = _discovery_cache.get(region)
    if cached and now < cached[1]:
        return list(cached[0])

    url = f"{mantle_openai_base_url(region)}/models"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "User-Agent": "hermes-cli",
    }

    try:
        if fetch_fn is not None:
            raw = fetch_fn(url, headers=headers, timeout=timeout)
            data = raw if isinstance(raw, dict) else None
        else:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                import json

                data = json.loads(resp.read().decode("utf-8"))
        ids: list[str] = []
        for entry in (data or {}).get("data") or []:
            mid = (entry.get("id") or "").strip()
            if mid:
                ids.append(mid)
        # Always surface Claude Mantle rows when discovery succeeds (OpenClaw parity).
        for claude_id in FALLBACK_CLAUDE_MODELS:
            if claude_id not in ids:
                ids.append(claude_id)
        if ids:
            _discovery_cache[region] = (ids, now + _DISCOVERY_TTL_S)
            return ids
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("Mantle model discovery failed for %s: %s", region, exc)
    except Exception as exc:
        logger.debug("Mantle model discovery error for %s: %s", region, exc)

    # Stale cache on failure
    if cached:
        return list(cached[0])
    return list(FALLBACK_OSS_MODELS) + list(FALLBACK_CLAUDE_MODELS)


def reset_mantle_caches_for_tests() -> None:
    _iam_token_cache.clear()
    _discovery_cache.clear()
