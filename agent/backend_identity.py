"""Single owner for backend identity and failure-scoped skip decisions.

Every fallback / dedup / skip / quarantine decision asks: "is this candidate the same backend
as the one that failed, along the axis that failure invalidated?" Answering inline at each
call site kept reintroducing the same bugs (same-shim aliases treated as distinct, sibling
models skipped for one model's timeout, dedup ignoring ``base_url``). "provider" conflates
three axes — credential surface (401/402), endpoint (DNS/refused), model deployment
(timeout/overload/429). Build :class:`BackendIdentity` values, ask :func:`should_skip_candidate`.

A fourth, OPTIONAL axis refines the first two: the fingerprint of a statically configured
``api_key``. Two entries with one URL, one model and two accounts are two quotas, not one
backend (#151321); an unknown fingerprint on either side keeps the historical answer.
"""

from __future__ import annotations

import hashlib
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


# Long enough that two real keys never collide, short enough to be safe in a repr/log line.
_CREDENTIAL_FINGERPRINT_CHARS = 16


def credential_fingerprint(api_key: object) -> str:
    """Non-reversible fingerprint of a configured credential; "" when it is unknown.

    A static key hashes its value. A ``key_cmd`` token source hashes its stable command
    identity (``cache_identity``) WITHOUT minting, so a rotating token never drifts the
    identity; the command namespace is tagged so it cannot collide with a static key. Any
    other callable (Entra ID) resolves per call and is not a key at all: "" — unknown — which
    every predicate below treats as "cannot prove difference".
    """
    identity = getattr(api_key, "cache_identity", None) if callable(api_key) else None
    if isinstance(identity, str) and identity.strip():
        key = "key_cmd\0" + identity.strip()
    elif isinstance(api_key, str) and api_key.strip():
        key = api_key.strip()
    else:
        return ""
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()[:_CREDENTIAL_FINGERPRINT_CHARS]


def route_credential(provider: Optional[str], entry: Optional[dict] = None) -> object:
    """The credential a route would send, for :func:`credential_fingerprint` ONLY (never log it).

    A fallback entry's own ``api_key`` / ``key_env`` wins (what the switch sends); otherwise a
    named ``providers:`` / ``custom_providers:`` entry supplies it with the runtime's precedence
    (``key_cmd`` first, then ``api_key`` / ``key_env``). Built-in providers and unresolvable
    routes return None (unknown), so their dedup is unchanged.
    """
    try:
        from agent.command_token_source import build_command_token_provider
        from hermes_cli.fallback_config import resolve_entry_api_key

        if explicit := resolve_entry_api_key(entry):
            return explicit
        if not (provider or "").strip():
            return None
        from hermes_cli.runtime_provider import _get_named_custom_provider

        named = _get_named_custom_provider(str(provider))
        if not named:
            return None
        return build_command_token_provider(str(named.get("key_cmd") or ""), "named") or named.get("api_key") or None
    except Exception:
        logger.debug("route_credential: could not resolve the credential of %s", provider, exc_info=True)
        return None


@dataclass(frozen=True)
class BackendIdentity:
    """Normalized identity of one (provider, model, endpoint) deployment.

    Empty fields mean "unknown" — an unknown axis can neither prove sameness nor difference
    on its own; the remaining axes decide."""

    provider: str = ""
    model: str = ""
    base_url: str = ""
    #: Fingerprint (never the key) of a statically configured api_key; "" = unknown.
    credential: str = ""

    @classmethod
    def build(
        cls, provider: Optional[str] = None, model: Optional[str] = None,
        base_url: Optional[str] = None, api_key: object = None,
    ) -> BackendIdentity:
        return cls(
            provider=_norm(provider), model=_norm(model),
            base_url=normalize_route_base_url(base_url),
            credential=credential_fingerprint(api_key),
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


def _distinct_known_credentials(a: BackendIdentity, b: BackendIdentity) -> bool:
    """True only when BOTH sides carry a fingerprint and they differ — never guessed (#151321).

    Two entries with one URL and one model but two static keys are two accounts, i.e. two
    quotas: a 401/429 against one says nothing about the other. An unknown fingerprint on
    either side answers False so the historical, conservative comparison still decides.
    """
    return bool(a.credential and b.credential and a.credential != b.credential)


def same_credential_surface(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities share the credential a 401/402 just invalidated?

    Conservative: an unprovable axis answers "different" (one wasted RTT) rather than "same"
    (stranded failover). Same label = same configured credential; custom entries can each carry
    their own api_key, so a shared URL alone is only a weak signal when a label is missing.
    Two KNOWN and different keys under one label are separate credentials (#151321)."""
    if _distinct_known_credentials(a, b):
        return False
    if a.credential and b.credential:
        # One identical known key under two labels (relay aliases, the #22548 family) is one credential:
        # a 401/402 on it holds for both (#91078).
        return True
    if a.provider and b.provider:
        # Different labels = different credential config (first-class registry providers explicitly so —
        # #70893; custom entries can each carry their own api_key, so sameness is unprovable and we must not
        # skip).
        return a.provider == b.provider
    return bool(a.base_url and a.base_url == b.base_url)


def same_endpoint(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities sit behind the endpoint that just went unreachable?
    An unknown base_url inherits the provider default, so a shared label implies the same endpoint."""
    if a.base_url and b.base_url:
        return a.base_url == b.base_url
    return bool(a.provider and a.provider == b.provider)


def same_deployment(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Are these the exact same model deployment (the thing a timeout kills)?

    Provider+model must match; base_url distinguishes only when BOTH sides carry an explicit URL
    (same provider+model on two explicit URLs is a pool, not a dup). Different labels with the
    same URL + model are still one deployment (same-host shim aliases) — unless both labels are
    first-class registry providers. Two KNOWN and different credentials are separate deployments
    even on one URL (#151321): a second account is a second quota, not a duplicate."""
    if _distinct_known_credentials(a, b):
        return False
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
