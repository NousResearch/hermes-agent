"""Single owner for backend identity and failure-scoped skip decisions.

Every fallback / dedup / skip / quarantine decision in Hermes ultimately asks
one question: **"is this candidate the same backend as the one that failed,
along the axis that failure invalidated?"**  Before this module, that
question was re-implemented inline at six call sites across four subsystems,
each comparing whatever string was locally convenient (provider label,
provider+model, base_url+model, ...).  Each incident fixed one site while the
others kept the bug: #22548 (same-shim aliases), #70893 (xai-oauth vs xai —
same host, distinct credential), #59561 (aux chain skipped sibling models),
#72468 (aux main-model safety net, same bug three weeks later), #62984 /
#54250 / #57584 (dedup ignoring base_url strands multi-endpoint pools).

The root insight: "provider" conflates three independent identity axes, and
each failure class invalidates a different one:

* **credential surface** — auth 401 / payment 402 kill everything sharing the
  credential (every model, every host reached with that key/token).
* **endpoint** — DNS failure / connection refused kill everything behind the
  URL, regardless of model or credential.
* **model deployment** — timeout / overload / rate limit / model-incompatible
  kill ONE model's deployment.  A sibling model behind the same URL is an
  independent deployment (real incident: aux ``glm-5.2`` hung and timed out
  while main ``macaron-v1-venti`` on the identical endpoint was serving
  448K-token turns).

Call sites should build :class:`BackendIdentity` values, classify the failure
with :func:`classify_failure_scope`, and ask :func:`should_skip_candidate`.
Do not re-implement any comparison inline — extend THIS module instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FailureScope(Enum):
    """Which identity axis a failure invalidates."""

    #: Timeout, overload/429, connection blip, model-incompatible, invalid
    #: response: evidence against ONE model deployment only.
    MODEL = "model"
    #: Auth 401 / payment 402: evidence against the shared credential —
    #: every model reached with it is equally dead.
    CREDENTIAL = "credential"
    #: DNS / connection-refused / unreachable host: evidence against the
    #: endpoint — every model behind the URL is equally dead.
    ENDPOINT = "endpoint"


#: Reason strings already used by auxiliary_client's except-chain, mapped to
#: scopes. Unknown reasons default to MODEL — the least-invalidating scope —
#: so an unrecognized failure never over-skips viable candidates.
_REASON_SCOPES = {
    "auth error": FailureScope.CREDENTIAL,
    "payment error": FailureScope.CREDENTIAL,
    "rate limit": FailureScope.MODEL,
    "model incompatible with route": FailureScope.MODEL,
    "invalid provider response": FailureScope.MODEL,
    "connection error": FailureScope.MODEL,
    "timeout": FailureScope.MODEL,
}


def classify_failure_scope(reason: Optional[str]) -> FailureScope:
    """Map a human-readable failure reason to the identity axis it kills."""
    return _REASON_SCOPES.get((reason or "").strip().lower(), FailureScope.MODEL)


def _norm_provider(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _norm_model(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _norm_base_url(value: Optional[str]) -> str:
    return (value or "").strip().rstrip("/").lower()


@dataclass(frozen=True)
class BackendIdentity:
    """Normalized identity of one (provider, model, endpoint) deployment.

    Empty fields mean "unknown" — comparisons treat an unknown axis as
    non-distinguishing (it can neither prove sameness nor difference on its
    own; the remaining axes decide).
    """

    provider: str = ""
    model: str = ""
    base_url: str = ""

    @classmethod
    def build(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "BackendIdentity":
        return cls(
            provider=_norm_provider(provider),
            model=_norm_model(model),
            base_url=_norm_base_url(base_url),
        )


def _both_first_class(a: BackendIdentity, b: BackendIdentity) -> bool:
    """True when both providers are distinct registered first-class providers.

    Two different registry providers have distinct credential surfaces even
    when they share an inference host (xai-oauth vs xai, openai-codex vs
    openai-api) — #70893.  Custom/shim aliases are NOT in the registry, so
    two aliases pointing at one URL still count as the same backend (#22548).
    """
    if not a.provider or not b.provider or a.provider == b.provider:
        return False
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        return a.provider in PROVIDER_REGISTRY and b.provider in PROVIDER_REGISTRY
    except Exception:
        return False


def same_credential_surface(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities share the credential a 401/402 just invalidated?

    Conservative on purpose: an unprovable axis must answer "different"
    (try the candidate — worst case one wasted RTT) rather than "same"
    (skip — worst case stranded failover). Two distinct custom labels at
    one URL may carry different per-entry api_keys, so a shared URL alone
    never proves a shared credential; it is only used as a weak signal
    when a provider label is missing entirely.
    """
    if a.provider and b.provider:
        # Same label = same configured credential. Different labels =
        # different credential config (first-class registry providers
        # explicitly so — #70893; custom entries can each carry their own
        # api_key, so sameness is unprovable and we must not skip).
        return a.provider == b.provider
    # Provider unknown on a side: same explicit URL is the best signal left.
    return bool(a.base_url and a.base_url == b.base_url)


def same_endpoint(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities sit behind the endpoint that just went unreachable?"""
    if a.base_url and b.base_url:
        return a.base_url == b.base_url
    # An unknown base_url inherits the provider default → same provider
    # label implies the same default endpoint.
    return bool(a.provider and a.provider == b.provider)


def same_deployment(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Are these the exact same model deployment (the thing a timeout kills)?

    Provider+model must match; the base_url axis distinguishes only when BOTH
    sides carry an explicit URL (#62984: same provider+model on two different
    explicit URLs is two deployments — a pool).  A side with an unknown URL
    inherits the provider default and cannot prove difference.
    """
    if not (a.provider and b.provider and a.provider == b.provider):
        # Same-host different-label shims: same URL + same model IS the same
        # deployment even when the alias labels differ (#22548) — unless both
        # labels are first-class registry providers (#70893).
        if (
            a.base_url
            and a.base_url == b.base_url
            and a.model
            and a.model == b.model
            and not _both_first_class(a, b)
        ):
            return True
        return False
    if not (a.model and b.model and a.model == b.model):
        return False
    if a.base_url and b.base_url and a.base_url != b.base_url:
        return False  # distinct explicit endpoints — a pool, not a dup
    return True


def should_skip_candidate(
    candidate: BackendIdentity,
    failed: BackendIdentity,
    scope: FailureScope = FailureScope.MODEL,
) -> bool:
    """THE skip predicate: would trying ``candidate`` just repeat the failure?

    True when the candidate is the same backend as ``failed`` along the axis
    ``scope`` says the failure invalidated.  Every fallback/dedup/skip site
    must call this instead of comparing labels inline.
    """
    if scope is FailureScope.CREDENTIAL:
        return same_credential_surface(candidate, failed)
    if scope is FailureScope.ENDPOINT:
        return same_endpoint(candidate, failed)
    return same_deployment(candidate, failed)
