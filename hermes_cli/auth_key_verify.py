"""One live check of an API key before ``hermes auth add`` stores it.

A pasted key that the provider rejects used to land in the pool and fail on the first turn, often
hours later and far from the command that stored it. ``verify_api_key`` makes one cheap
authenticated read (``GET /models``, or the provider's key endpoint where ``/models`` is public)
and classifies the answer; the caller decides what to store.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass

import httpx

from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS
from agent.anthropic_credentials import _is_oauth_token, is_native_anthropic_host
from agent.anthropic_endpoints import _requires_bearer_auth
from agent.auxiliary_openai_url import _to_openai_base_url
from agent.model_metadata_http import resolve_verify
from agent.redact import redact_sensitive_text
from hermes_cli.auth_zai_kimi import _resolve_kimi_base_url
from hermes_cli.config import get_custom_provider_extra_headers
from hermes_cli.version_info import hermes_user_agent
from utils import base_url_host_matches, base_url_hostname

VERIFY_TIMEOUT_S = 10.0

# Hosts whose ``/models`` answers 200 with no key at all: a 200 there proves nothing, so the
# check names the key unverified instead of calling it good. A key endpoint (below) wins.
_PUBLIC_CATALOG_HOSTS = frozenset({
    "inference-api.nousresearch.com", "router.huggingface.co", "integrate.api.nvidia.com", "api.kilo.ai"})
# Hosts whose ``/models`` is public but that expose an authenticated key-info read instead.
_KEY_INFO_URLS = {"openrouter.ai": "https://openrouter.ai/api/v1/key"}

# 400 bodies some providers use for a bad key (Gemini ``API_KEY_INVALID``, xAI "Incorrect API key").
_BAD_KEY_400 = re.compile(r"api_key_invalid|incorrect api key|invalid api key|api key not valid", re.IGNORECASE)
_BILLING_TEXT = re.compile(r"credit|balance|billing|payment|spending limit|quota|out of funds", re.IGNORECASE)


class KeyCheckOutcome(enum.Enum):
    OK = "ok"
    REJECTED = "rejected"  # 401/403 (or a bad-key 400): refuse, store nothing
    BILLING = "billing"  # 402, or a 403 naming credits: store, mark exhausted
    UNVERIFIED = "unverified"  # network error, timeout, unexpected status, or no authenticated read


@dataclass(frozen=True)
class KeyCheck:
    outcome: KeyCheckOutcome
    detail: str = ""  # provider's own message (redacted) or the reason the check was inconclusive
    status_code: int | None = None
    auth_mode: str = ""  # e.g. "Anthropic Console API key" vs "Anthropic OAuth token"


@dataclass(frozen=True)
class _ProbeRequest:
    url: str
    headers: dict[str, str]
    auth_mode: str = ""


def _anthropic_request(base: str, key: str) -> _ProbeRequest:
    headers = {"anthropic-version": "2023-06-01", "User-Agent": hermes_user_agent()}
    native = is_native_anthropic_host(base)
    if native and _is_oauth_token(key):
        headers.update({"Authorization": f"Bearer {key}", "anthropic-beta": ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)})
        mode = "Anthropic OAuth token"
    elif not native and _requires_bearer_auth(base):
        headers["Authorization"] = f"Bearer {key}"
        mode = ""
    else:
        headers["x-api-key"] = key
        mode = "Anthropic Console API key" if native else ""
    root = (base or "https://api.anthropic.com").rstrip("/")
    return _ProbeRequest(root + ("/models" if root.endswith("/v1") else "/v1/models"), headers, mode)


def _openai_request(base: str, key: str) -> _ProbeRequest:
    headers = {"Authorization": f"Bearer {key}", "User-Agent": hermes_user_agent(), "Accept": "application/json"}
    if base_url_host_matches(base, "generativelanguage.googleapis.com"):
        # Google rejects ``Bearer <api-key>`` (reserved for OAuth 2); plain keys ride x-goog-api-key.
        headers.pop("Authorization")
        headers["x-goog-api-key"] = key
    return _ProbeRequest(_KEY_INFO_URLS.get(base_url_hostname(base)) or base.rstrip("/") + "/models", headers)


def build_probe_request(provider: str, key: str, base_url: str) -> _ProbeRequest | None:
    """The one request that proves *key* authenticates at *base_url*; None when no such read exists."""
    base = (base_url or "").strip().rstrip("/")
    if provider.startswith("kimi-coding"):
        base = _resolve_kimi_base_url(key, base, "")  # sk-kimi-* keys live on api.kimi.com/coding
    if base.endswith("/anthropic") or base_url_host_matches(base, "api.kimi.com"):
        base = _to_openai_base_url(base)  # dual-surface hosts list models on their OpenAI sibling
    if provider == "anthropic" or base.endswith("/anthropic"):
        request = _anthropic_request(base, key)
    elif not base or (base_url_hostname(base) in _PUBLIC_CATALOG_HOSTS):
        return None
    else:
        request = _openai_request(base, key)
    request.headers.update(get_custom_provider_extra_headers(base))
    return request


def _provider_message(response) -> str:
    """The provider's own error text, redacted and bounded."""
    text = response.text or ""
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        message = (error.get("message") if isinstance(error, dict) else error) or body.get("message") or body.get("detail")
        if isinstance(message, str) and message.strip():
            text = message
    return redact_sensitive_text(" ".join(text.split()), force=True)[:300]


def _classify(status: int, message: str, auth_mode: str) -> KeyCheck:
    if 200 <= status < 300:
        return KeyCheck(KeyCheckOutcome.OK, status_code=status, auth_mode=auth_mode)
    if status == 402 or (status == 403 and _BILLING_TEXT.search(message)):
        return KeyCheck(KeyCheckOutcome.BILLING, message or f"HTTP {status}", status, auth_mode)
    if status in (401, 403) or (status == 400 and _BAD_KEY_400.search(message)):
        return KeyCheck(KeyCheckOutcome.REJECTED, message or f"HTTP {status}", status, auth_mode)
    return KeyCheck(KeyCheckOutcome.UNVERIFIED, f"HTTP {status}" + (f": {message}" if message else ""), status, auth_mode)


def verify_api_key(provider: str, key: str, base_url: str, *, timeout: float = VERIFY_TIMEOUT_S) -> KeyCheck:
    """One authenticated read of *key* against *base_url*, classified (never raises on transport errors)."""
    request = build_probe_request(provider, key, base_url)
    if request is None:
        return KeyCheck(KeyCheckOutcome.UNVERIFIED, "this endpoint has no authenticated read to check a key against")
    try:
        # No redirects: a key must never follow a 30x to another origin.
        response = httpx.get(request.url, headers=request.headers, timeout=timeout,
                             verify=resolve_verify(base_url), follow_redirects=False)
    except httpx.HTTPError as exc:
        reason = "timed out" if isinstance(exc, httpx.TimeoutException) else type(exc).__name__
        return KeyCheck(KeyCheckOutcome.UNVERIFIED, f"could not reach {base_url_hostname(request.url)} ({reason})",
                        auth_mode=request.auth_mode)
    return _classify(response.status_code, _provider_message(response), request.auth_mode)
