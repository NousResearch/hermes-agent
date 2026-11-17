"""Run the key check under a whole-call deadline and classify the provider's answer."""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass

import httpx

from agent.deadline import run_bounded_sync
from agent.error_classifier import FailoverReason, classify_api_error
from agent.model_metadata_http import resolve_verify
from agent.redact import redact_sensitive_text
from hermes_cli.auth_verify.request import build_probe_request
from utils import base_url_hostname

# Per-socket-op httpx timeout AND the wall-clock cap on the whole exchange: a body trickling in
# under the read timeout must not hold ``hermes auth add`` open.
VERIFY_TIMEOUT_S = 10.0

# 400 bodies some providers use for a bad key (Gemini ``API_KEY_INVALID``, xAI "Incorrect API key").
_BAD_KEY_400 = re.compile(r"api_key_invalid|incorrect api key|invalid api key|api key not valid", re.IGNORECASE)


class KeyCheckOutcome(enum.Enum):
    OK = "ok"
    REJECTED = "rejected"  # the provider refused the key: store nothing
    BILLING = "billing"  # the key authenticates but the account has no usable credit: store benched
    UNVERIFIED = "unverified"  # network error, timeout, unexpected status, or no authenticated read


@dataclass(frozen=True)
class KeyCheck:
    outcome: KeyCheckOutcome
    detail: str = ""  # provider's own message (redacted) or why the check was inconclusive
    status_code: int | None = None
    auth_mode: str = ""


def _provider_message(response: httpx.Response) -> str:
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


def _classify(provider: str, url: str, response: httpx.Response, auth_mode: str) -> KeyCheck:
    status = response.status_code
    if response.is_success:
        return KeyCheck(KeyCheckOutcome.OK, status_code=status, auth_mode=auth_mode)
    message = _provider_message(response)
    detail = message or f"HTTP {status}"
    # The runtime's own verdict decides billing vs auth, so a key ``auth add`` benches is one the
    # turn loop would also bench (OpenRouter "Key limit exceeded" 403, Anthropic credit 400, 402).
    # The message rides in the exception text too: a plain-text body has no JSON for the classifier to read.
    error = httpx.HTTPStatusError(f"HTTP {status}: {message}", request=response.request, response=response)
    verdict = classify_api_error(error,
                                 provider=provider, base_url=url).reason
    if verdict is FailoverReason.billing:
        return KeyCheck(KeyCheckOutcome.BILLING, detail, status, auth_mode)
    if verdict in (FailoverReason.auth, FailoverReason.auth_permanent) or (status == 400 and _BAD_KEY_400.search(message)):
        return KeyCheck(KeyCheckOutcome.REJECTED, detail, status, auth_mode)
    return KeyCheck(KeyCheckOutcome.UNVERIFIED, f"HTTP {status}" + (f": {message}" if message else ""), status, auth_mode)


def verify_api_key(provider: str, key: str, base_url: str, *, api_mode: str = "",
                   timeout: float = VERIFY_TIMEOUT_S) -> KeyCheck:
    """One authenticated read of *key* against *base_url*, classified; transport failures are UNVERIFIED."""
    request = build_probe_request(provider, key, base_url, api_mode)
    if request is None:
        return KeyCheck(KeyCheckOutcome.UNVERIFIED, "this endpoint has no authenticated read to check a key against")
    host = base_url_hostname(request.url)

    def fetch() -> httpx.Response:
        # No redirects: a key must never follow a 30x to another origin.
        return httpx.get(request.url, headers=request.headers, timeout=timeout,
                         verify=resolve_verify(base_url), follow_redirects=False)

    try:
        bounded = run_bounded_sync(fetch, timeout, label=f"auth add key check {host}")
    except httpx.HTTPError as exc:
        reason = "timed out" if isinstance(exc, httpx.TimeoutException) else type(exc).__name__
        return KeyCheck(KeyCheckOutcome.UNVERIFIED, f"could not reach {host} ({reason})", auth_mode=request.auth_mode)
    if bounded.timed_out:
        return KeyCheck(KeyCheckOutcome.UNVERIFIED, f"{host} did not answer within {timeout:g}s",
                        auth_mode=request.auth_mode)
    return _classify(provider, request.url, bounded.value, request.auth_mode)
