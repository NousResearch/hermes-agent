"""How the credential pool benches a row whose OAuth refresh POST failed.

A dead grant (``invalid_grant`` and friends) is handled by each provider branch of
``CredentialPool._recover_failed_refresh`` before this runs. Everything that reaches here is
either a transient token-endpoint failure (5xx, a 429 from the token host, a timeout, a reset
connection, DNS) or an unclassified one. Benching a transient failure like an unknown one put
the sole credential out of rotation for a full cooldown, and the resolver then told the user
"No Anthropic credentials found, run hermes auth add" while the sign-in was fine.

A transient failure gets a short bench stamped ``refresh_unavailable`` instead: long enough that
a burst of resolves does not hammer a struggling token endpoint, short enough that the next turn
retries the refresh. ``refresh_unavailable_error`` reads that stamp back so a resolver that found
no usable row says "temporarily unavailable, your sign-in is kept" instead of "sign in again".
"""

from __future__ import annotations

import contextlib
import logging
import time
import urllib.error
from typing import TYPE_CHECKING, Optional

from hermes_cli.auth_constants import AuthError

if TYPE_CHECKING:  # pragma: no cover
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger(__name__)

REFRESH_UNAVAILABLE_REASON = "refresh_unavailable"
TRANSIENT_REFRESH_BENCH_SECONDS = 30


def _transport_error(exc: BaseException) -> bool:
    """A network-level failure: no HTTP verdict about the grant was ever received."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        return True
    import httpx

    return isinstance(exc, httpx.TransportError)


def transient_refresh_status(exc: BaseException) -> tuple[bool, Optional[int]]:
    """``(transient, http_status)`` for a failed refresh POST.

    Transient = a later POST of the same refresh token can succeed: the token endpoint answered
    5xx (or Anthropic's 429), the request never got an HTTP answer, or the raiser marked its
    ``AuthError`` retryable. Grant-dead and re-login errors are never transient.
    """
    from agent.anthropic_credentials import AnthropicOAuthError

    if isinstance(exc, AnthropicOAuthError):
        status = int(exc.status)
        return (not exc.relogin_required and (status >= 500 or status == 429)), status
    if isinstance(exc, AuthError):
        return (exc.retryable is True and not exc.relogin_required), None
    return _transport_error(exc), None


def mark_refresh_failed(pool: CredentialPool, entry: PooledCredential, exc: BaseException) -> None:
    """Bench *entry* after a refresh failure no provider branch classified as terminal."""
    transient, status = transient_refresh_status(exc)
    if not transient:
        pool._mark_exhausted(entry, None)
        return
    # INFO, not WARNING: nothing is lost and the next resolve retries; a dead grant warns instead.
    logger.info(
        "%s OAuth refresh for %s failed transiently (%s); the sign-in is kept and the refresh "
        "is retried in %ss.", pool.provider, entry.label or entry.id[:8], exc,
        TRANSIENT_REFRESH_BENCH_SECONDS)
    pool._mark_exhausted(entry, status, {
        "reason": REFRESH_UNAVAILABLE_REASON,
        "message": str(exc),
        "reset_at": time.time() + TRANSIENT_REFRESH_BENCH_SECONDS,
    })


def refresh_unavailable_error(provider: str) -> Optional[AuthError]:
    """The retryable error for a pool whose live rows are all benched, at least one of them on a
    transient refresh failure; ``None`` otherwise (the caller keeps its own diagnosis).

    Reads the persisted pool (no seeding, no refresh, no writes), like ``pool_cooldown_message``.
    """
    from agent.credential_pool import STATUS_DEAD, PooledCredential, _exhausted_until
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.auth_constants import TEMPORARILY_UNAVAILABLE_CODE

    entries = []
    with contextlib.suppress(Exception):
        entries = [PooledCredential.from_dict(provider, e)
                   for e in read_credential_pool(provider) if isinstance(e, dict)]
    now = time.time()
    live = [e for e in entries if e.last_status != STATUS_DEAD]
    benched = [e for e in live if (_exhausted_until(e, sole_credential=len(live) == 1) or 0) > now]
    flaky = [e for e in benched if e.last_error_reason == REFRESH_UNAVAILABLE_REASON]
    if not flaky or len(benched) != len(live):
        return None
    code = flaky[0].last_error_code
    detail = f"HTTP {code}" if code else "network error"
    display = provider
    with contextlib.suppress(Exception):
        from hermes_cli.auth import PROVIDER_REGISTRY
        display = PROVIDER_REGISTRY[provider].name
    return AuthError(
        f"{display} sign-in refresh is temporarily unavailable ({detail}); your sign-in is kept.",
        provider=provider, code=TEMPORARILY_UNAVAILABLE_CODE, retryable=True,
        retry_after=float(TRANSIENT_REFRESH_BENCH_SECONDS))
