"""Single owner for backend identity and failure-scoped skip decisions.

Every fallback / dedup / skip / quarantine decision asks: "is this candidate the same backend
as the one that failed, along the axis that failure invalidated?" Answering inline at each
call site kept reintroducing the same bugs (same-shim aliases treated as distinct, sibling
models skipped for one model's timeout, dedup ignoring ``base_url``). "provider" conflates
three axes — credential surface (401/402), endpoint (DNS/refused), model deployment
(timeout/overload/429). Build :class:`BackendIdentity` values, ask :func:`should_skip_candidate`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hermes_cli.route_identity import normalize_route_base_url

logger = logging.getLogger(__name__)


class FailureScope(Enum):
    """Which identity axis a failure invalidates."""

    #: Timeout, overload/429, connection blip, model-incompatible, invalid response:
    #: evidence against ONE model deployment only.
    MODEL = "model"
    #: Auth 401 / payment 402: evidence against the shared credential.
    CREDENTIAL = "credential"
    #: DNS / connection-refused / unreachable host: evidence against the endpoint.
    ENDPOINT = "endpoint"


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


@dataclass(frozen=True)
class BackendIdentity:
    """Normalized identity of one (provider, model, endpoint) deployment.

    Empty fields mean "unknown" — an unknown axis can neither prove sameness nor difference
    on its own; the remaining axes decide."""

    provider: str = ""
    model: str = ""
    base_url: str = ""

    @classmethod
    def build(
        cls, provider: Optional[str] = None, model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BackendIdentity:
        return cls(
            provider=_norm(provider), model=_norm(model),
            base_url=normalize_route_base_url(base_url),
        )


def _both_first_class(a: BackendIdentity, b: BackendIdentity) -> bool:
    """True when both providers are distinct registered first-class providers.

    Two different registry providers have distinct credential surfaces even when they share an
    inference host (xai-oauth vs xai). Custom/shim aliases are NOT in the registry, so two
    aliases pointing at one URL still count as the same backend."""
    if not a.provider or not b.provider or a.provider == b.provider:
        return False
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        return a.provider in PROVIDER_REGISTRY and b.provider in PROVIDER_REGISTRY
    except Exception:
        return False


def same_credential_surface(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities share the credential a 401/402 just invalidated?

    Conservative: an unprovable axis answers "different" (one wasted RTT) rather than "same"
    (stranded failover). Same label = same configured credential; custom entries can each carry
    their own api_key, so a shared URL alone is only a weak signal when a label is missing."""
    if a.provider and b.provider:
        # Different labels = different credential config (first-class registry providers explicitly so —
        # #70893; custom entries can each carry their own api_key, so sameness is unprovable and we must not
        # skip).
        return a.provider == b.provider
    return bool(a.base_url and a.base_url == b.base_url)


def _url_host(url: Optional[str]) -> str:
    """Bare lowercase host of *url* (scheme/path/port stripped); "" when unusable."""
    candidate = (url or "").strip()
    if "//" in candidate:
        candidate = candidate.split("//", 1)[1]
    candidate = candidate.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].lower()
    return candidate if "." in candidate or candidate == "localhost" else ""


def _host_matches(host: str, known: str) -> bool:
    return bool(host and known) and (
        host == known or host.endswith("." + known) or known.endswith("." + host)
    )


def _explicit_url_provably_foreign(provider: str, url: str) -> bool:
    """True only when *url*'s host provably belongs to a DIFFERENT provider.

    Uses the upstream host->provider map (``agent.model_metadata._infer_provider_from_url``,
    which knows same-company families such as api.z.ai <-> open.bigmodel.cn) with the provider
    registry as a fallback. Fail-open by design: unknown hosts (custom proxies, gateways,
    mirrors, LAN boxes) and any lookup failure answer False — "cannot judge" — which preserves
    the historical same-label-implies-same-endpoint answer.
    """
    host = _url_host(url)
    if not host:
        return False
    try:
        from agent.model_metadata import _infer_provider_from_url

        inferred = _infer_provider_from_url(url)
    except Exception:
        inferred = None
    if inferred:
        return inferred != provider
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        return False
    claimants: set = set()
    try:
        for pid, pconfig in PROVIDER_REGISTRY.items():
            if _host_matches(host, _url_host(getattr(pconfig, "inference_base_url", ""))):
                claimants.add(pid)
    except Exception:
        return False
    return bool(claimants) and provider not in claimants


def same_endpoint(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities sit behind the endpoint that just went unreachable?

    An unknown base_url inherits the provider default, so a shared label implies the same endpoint —
    unless the *known* URL is provably foreign to that provider's endpoint family. A stale
    cross-provider base_url left on the agent by a mid-session /model switch (deepseek label
    carrying open.bigmodel.cn) made every same-provider fallback look like "the same backend" and
    stranded failover exactly when the runtime was already misrouted: the fallback's implicit
    default endpoint is NOT the URL that just failed, so that skip must not apply.
    """
    if a.base_url and b.base_url:
        return a.base_url == b.base_url
    if not (a.provider and a.provider == b.provider):
        return False
    explicit_url = a.base_url or b.base_url
    if not explicit_url:
        return True  # both unknown -> provider default on both sides
    return not _explicit_url_provably_foreign(a.provider, explicit_url)


def same_deployment(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Are these the exact same model deployment (the thing a timeout kills)?

    Provider+model must match; base_url distinguishes only when BOTH sides carry an explicit URL
    (same provider+model on two explicit URLs is a pool, not a dup). Different labels with the
    same URL + model are still one deployment (same-host shim aliases) — unless both labels are
    first-class registry providers."""
    if not (a.provider and b.provider and a.provider == b.provider):
        return bool(
            a.base_url
            # Same-host different-label shims: same URL + same model IS the same deployment even when the
            # alias labels differ (#22548) — unless both labels are first-class registry providers (#70893).
            and a.base_url == b.base_url
            and a.model
            and a.model == b.model
            and not _both_first_class(a, b)
        )
    if not (a.model and b.model and a.model == b.model):
        return False
    return not (a.base_url and b.base_url and a.base_url != b.base_url)


_SCOPE_PREDICATES = {
    FailureScope.CREDENTIAL: same_credential_surface, FailureScope.ENDPOINT: same_endpoint,
    FailureScope.MODEL: same_deployment,
}


def should_skip_candidate(
    candidate: BackendIdentity, failed: BackendIdentity, scope: FailureScope = FailureScope.MODEL
) -> bool:
    """THE skip predicate: would trying ``candidate`` just repeat the failure?
    True when it is the same backend as ``failed`` along the axis ``scope`` invalidated.
    Every fallback/dedup/skip site must call this."""
    return _SCOPE_PREDICATES.get(scope, same_deployment)(candidate, failed)
