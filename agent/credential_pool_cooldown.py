"""How long an exhausted pool credential stays benched before it is retried."""

from __future__ import annotations

from typing import Optional

# Cooldowns before retrying an exhausted credential. Transient 401s cool down
# briefly so single-key setups recover; 429/402/other take an hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60
EXHAUSTED_TTL_429_SECONDS = 60 * 60
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60
# When the offending key is the sole non-DEAD entry, an hour-long bench means
# an hour of hard failures. Throttles (429/403/5xx) reset in seconds, so a sole
# credential cools down briefly instead.
EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS = 60

# ``FailoverReason.billing`` as a bare string: the pool persists classified
# failure semantics to JSON and must not import the classifier.
FAILURE_REASON_BILLING = "billing"

# Billing verdict resting on an ambiguous body (#82154): Anthropic's "out of
# extra usage" 400 is returned both for genuine overage and for a server-side
# content-filter rejection, which leaves the credential healthy. Unverified
# billing gets the short transient cooldown; genuine depletion re-latches.
FAILURE_REASON_BILLING_UNVERIFIED = "billing_unverified"


def _exhausted_ttl(
    error_code: Optional[int],
    *,
    sole_credential: bool = False,
    failure_reason: Optional[str] = None,
) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion.

    *sole_credential*: the pool has nothing to rotate to, so transient
    throttles (429 and the catch-all default covering 403/5xx/unknown) are
    capped to a brief cooldown; 401 keeps its own already-short TTL.

    *failure_reason* is the classifier verdict: an OpenRouter ``key limit
    exceeded`` and an xAI spending block both arrive as 403 but are billing,
    and a 60s retry on a spent account just re-fails. Billing keeps the full
    bench regardless of status; 402 is billing by definition.
    Unverified billing (#82154) gets the short cooldown regardless of pool
    size (the credential may be healthy), unless the status is a true 402.
    """
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    base = EXHAUSTED_TTL_429_SECONDS if error_code == 429 else EXHAUSTED_TTL_DEFAULT_SECONDS
    if failure_reason == FAILURE_REASON_BILLING_UNVERIFIED and error_code != 402:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    is_billing = error_code == 402 or failure_reason == FAILURE_REASON_BILLING
    if sole_credential and not is_billing:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    return base
