"""Canonical webhook route, provider, identity, and intake-envelope authority.

This module deliberately has no HTTP-server, secret-storage, session, delivery,
or callback dependencies. It owns the domain facts those layers consume:
which provider a route is bound to, which provider-native identifier is stable
enough for retry deduplication, which event header belongs to that provider, and
the immutable envelope handed across the HTTP/domain boundary.

Legacy request-header inference exists only as an explicit compatibility bridge
for routes that have not declared a provider/signature mode yet. Once a route
is bound, downstream code never re-selects provider authority from headers.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional


class WebhookContractError(ValueError):
    """Webhook configuration cannot be normalized without ambiguity."""


@dataclass(frozen=True)
class WebhookProviderSpec:
    """Wire facts owned by one webhook provider namespace."""

    name: str
    aliases: tuple[str, ...] = ()
    delivery_id_headers: tuple[str, ...] = ()
    event_headers: tuple[str, ...] = ()
    legacy_detection_headers: tuple[str, ...] = ()
    payload_delivery_id_keys: tuple[str, ...] = ()
    signature_modes: tuple[str, ...] = ()
    default_signature_mode: Optional[str] = None


_PROVIDER_SPECS = (
    WebhookProviderSpec(
        name="svix",
        aliases=("agentmail",),
        delivery_id_headers=("svix-id",),
        legacy_detection_headers=("svix-id", "svix-signature", "svix-timestamp"),
        signature_modes=("svix",),
        default_signature_mode="svix",
    ),
    WebhookProviderSpec(
        name="github",
        aliases=("github_hmac_sha256",),
        delivery_id_headers=("X-GitHub-Delivery",),
        event_headers=("X-GitHub-Event",),
        legacy_detection_headers=("X-Hub-Signature-256", "X-GitHub-Delivery"),
        signature_modes=("github", "github_hmac_sha256"),
        default_signature_mode="github",
    ),
    WebhookProviderSpec(
        name="gitlab",
        aliases=("gitlab_token",),
        delivery_id_headers=(
            "X-Gitlab-Event-UUID",
            "X-Gitlab-Webhook-UUID",
            "X-Gitlab-Idempotency-Key",
            "Idempotency-Key",
        ),
        event_headers=("X-GitLab-Event",),
        legacy_detection_headers=(
            "X-Gitlab-Token",
            "X-Gitlab-Event-UUID",
            "X-Gitlab-Webhook-UUID",
            "X-Gitlab-Idempotency-Key",
        ),
        signature_modes=("gitlab", "gitlab_token"),
        default_signature_mode="gitlab",
    ),
    WebhookProviderSpec(
        name="standard_webhooks",
        aliases=("gitlab_standard",),
        delivery_id_headers=("webhook-id", "Idempotency-Key"),
        legacy_detection_headers=("webhook-id", "webhook-signature"),
        signature_modes=("standard_webhooks", "gitlab_standard"),
        default_signature_mode="standard_webhooks",
    ),
    WebhookProviderSpec(
        name="chatwoot",
        delivery_id_headers=("X-Chatwoot-Delivery",),
        legacy_detection_headers=("X-Chatwoot-Delivery",),
        payload_delivery_id_keys=("id",),
        signature_modes=("chatwoot", "generic_v1", "generic_v2"),
        default_signature_mode="generic_v1",
    ),
    WebhookProviderSpec(
        name="linear",
        legacy_detection_headers=("linear-signature",),
        signature_modes=("linear",),
        default_signature_mode="linear",
    ),
    WebhookProviderSpec(
        name="hindsight",
        aliases=("hindsight_hmac_sha256",),
        legacy_detection_headers=("X-Hindsight-Signature",),
        signature_modes=("hindsight", "hindsight_hmac_sha256"),
        default_signature_mode="hindsight",
    ),
    WebhookProviderSpec(
        name="stripe",
        payload_delivery_id_keys=("id",),
        signature_modes=("stripe", "generic_v1", "generic_v2"),
        default_signature_mode="generic_v1",
    ),
    WebhookProviderSpec(
        name="generic",
        aliases=("generic_v1", "generic_v2"),
        delivery_id_headers=("X-Request-ID",),
        legacy_detection_headers=(
            "X-Webhook-Signature-V2",
            "X-Webhook-Signature",
            "X-Request-ID",
        ),
        signature_modes=("generic", "generic_v1", "generic_v2"),
        default_signature_mode="generic_v2",
    ),
)

PROVIDER_REGISTRY: Mapping[str, WebhookProviderSpec] = MappingProxyType(
    {spec.name: spec for spec in _PROVIDER_SPECS}
)
_PROVIDER_ALIASES = MappingProxyType(
    {
        alias: spec.name
        for spec in _PROVIDER_SPECS
        for alias in (spec.name, *spec.aliases)
    }
)
_SIGNATURE_MODE_TO_PROVIDER = MappingProxyType(
    {
        mode: spec.name
        for spec in _PROVIDER_SPECS
        for mode in spec.signature_modes
        if mode not in {"generic_v1", "generic_v2"}
    }
    | {"generic_v1": "generic", "generic_v2": "generic"}
)

# Compatibility inference order intentionally checks provider-specific families
# before generic HMAC. It is used only when a route has no declared provider or
# signature mode; it is never a downstream authorization mechanism.
_LEGACY_PROVIDER_ORDER = (
    "svix",
    "github",
    "gitlab",
    "standard_webhooks",
    "chatwoot",
    "linear",
    "hindsight",
    "generic",
)


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _nonempty_scalar(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (str, int)):
        return None
    normalized = str(value).strip()
    return normalized or None


def _header(headers: Mapping[str, Any], name: str) -> str:
    """Read a case-insensitive HTTP header from real or test mappings."""

    direct = headers.get(name)
    if direct not in (None, ""):
        return str(direct)
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target and value not in (None, ""):
            return str(value)
    return ""


def canonical_provider(value: str) -> str:
    """Return the canonical provider namespace or fail closed."""

    normalized = _normalize_token(value)
    if not normalized:
        raise WebhookContractError("webhook provider must be non-empty")
    provider = _PROVIDER_ALIASES.get(normalized)
    if provider is None:
        raise WebhookContractError(f"unsupported webhook provider {value!r}")
    return provider


def canonical_signature_mode(value: str) -> str:
    """Return one registered verifier mode or fail closed."""

    normalized = _normalize_token(value)
    if not normalized:
        raise WebhookContractError("webhook signature mode must be non-empty")
    if normalized not in _SIGNATURE_MODE_TO_PROVIDER:
        raise WebhookContractError(f"unsupported webhook signature mode {value!r}")
    return normalized


def infer_legacy_provider(headers: Mapping[str, Any]) -> str:
    """Compatibility-only provider inference for undeclared legacy routes."""

    for name in _LEGACY_PROVIDER_ORDER:
        spec = PROVIDER_REGISTRY[name]
        if any(
            _header(headers, header).strip()
            for header in spec.legacy_detection_headers
        ):
            return name
    return "generic"


@dataclass(frozen=True)
class WebhookRouteConfig:
    """Normalized route identity/security binding consumed by intake."""

    name: str
    profile: str
    provider: str
    provider_declared: bool
    signature_mode: str
    enabled: bool
    events: tuple[str, ...]

    @classmethod
    def bind(
        cls,
        name: str,
        route: Mapping[str, Any],
        *,
        headers: Mapping[str, Any],
        request_profile: Optional[str] = None,
    ) -> "WebhookRouteConfig":
        route_name = str(name or "").strip()
        if not route_name:
            raise WebhookContractError("webhook route name must be non-empty")

        if "profile" in route:
            profile_value = route.get("profile")
            if not isinstance(profile_value, str) or not profile_value.strip():
                raise WebhookContractError(
                    f"route {route_name!r} has malformed profile binding"
                )
            configured_profile = profile_value.strip()
        else:
            configured_profile = "default"

        effective_profile = str(request_profile or "default").strip() or "default"
        if configured_profile != effective_profile:
            raise WebhookContractError(
                f"route {route_name!r} is not bound to profile {effective_profile!r}"
            )

        provider_raw = route.get("provider")
        if provider_raw is not None and (
            not isinstance(provider_raw, str) or not provider_raw.strip()
        ):
            raise WebhookContractError(
                f"route {route_name!r} has malformed provider"
            )
        signature_raw = route.get("signature_mode")
        if signature_raw is not None and (
            not isinstance(signature_raw, str) or not signature_raw.strip()
        ):
            raise WebhookContractError(
                f"route {route_name!r} has malformed signature_mode"
            )

        provider_declared = bool(provider_raw or signature_raw)
        if provider_raw:
            provider = canonical_provider(provider_raw)
        elif signature_raw:
            mode = canonical_signature_mode(signature_raw)
            provider = _SIGNATURE_MODE_TO_PROVIDER[mode]
        else:
            provider = infer_legacy_provider(headers)

        spec = PROVIDER_REGISTRY[provider]
        if signature_raw:
            signature_mode = canonical_signature_mode(signature_raw)
            if signature_mode not in spec.signature_modes:
                raise WebhookContractError(
                    f"route {route_name!r} provider {provider!r} does not allow "
                    f"signature mode {signature_mode!r}"
                )
        else:
            signature_mode = spec.default_signature_mode or provider

        events_raw = route.get("events", ())
        if events_raw in (None, ""):
            events: tuple[str, ...] = ()
        elif isinstance(events_raw, (list, tuple, set, frozenset)):
            events = tuple(
                value
                for value in (str(item).strip() for item in events_raw)
                if value
            )
        else:
            raise WebhookContractError(
                f"route {route_name!r} events must be a sequence"
            )

        return cls(
            name=route_name,
            profile=configured_profile,
            provider=provider,
            provider_declared=provider_declared,
            signature_mode=signature_mode,
            enabled=route.get("enabled", True) is not False,
            events=events,
        )

    @property
    def provider_spec(self) -> WebhookProviderSpec:
        return PROVIDER_REGISTRY[self.provider]


@dataclass(frozen=True)
class WebhookDeliveryIdentity:
    provider: str
    value: str

    @property
    def namespaced(self) -> str:
        return f"{self.provider}:{self.value}"


def resolve_delivery_identity(
    route: WebhookRouteConfig,
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Optional[WebhookDeliveryIdentity]:
    """Return one provider-native retry identity, never a synthetic timestamp.

    Payload IDs are accepted only for explicitly declared providers. This
    prevents a generic/legacy route from becoming Stripe or Chatwoot merely
    because an attacker supplied an ``id`` field with a convenient shape.
    """

    spec = route.provider_spec
    for header_name in spec.delivery_id_headers:
        candidate = _nonempty_scalar(_header(headers, header_name))
        if candidate is not None:
            return WebhookDeliveryIdentity(route.provider, candidate)

    if route.provider_declared:
        for key in spec.payload_delivery_id_keys:
            candidate = _nonempty_scalar(payload.get(key))
            if candidate is not None:
                return WebhookDeliveryIdentity(route.provider, candidate)
    return None


def resolve_event_type(
    route: WebhookRouteConfig,
    headers: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """Resolve event type from the already-bound provider namespace."""

    for header_name in route.provider_spec.event_headers:
        value = _nonempty_scalar(_header(headers, header_name))
        if value is not None:
            return value
    for key in ("event_type", "type"):
        value = _nonempty_scalar(payload.get(key))
        if value is not None:
            return value
    return "unknown"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class WebhookAuthProvenance:
    provider: str
    signature_mode: str
    provider_declared: bool

    @property
    def compatibility_inferred(self) -> bool:
        return not self.provider_declared


@dataclass(frozen=True)
class WebhookEnvelope:
    """Immutable domain object handed across the HTTP intake boundary."""

    route: WebhookRouteConfig
    auth: WebhookAuthProvenance
    event_type: str
    delivery_identity: Optional[WebhookDeliveryIdentity]
    trace_id: str
    body_sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def build(
        cls,
        route: WebhookRouteConfig,
        *,
        headers: Mapping[str, Any],
        payload: Mapping[str, Any],
        raw_body: bytes,
        trace_id: Optional[str] = None,
    ) -> "WebhookEnvelope":
        if not isinstance(raw_body, (bytes, bytearray)):
            raise WebhookContractError("raw_body must be bytes")
        if not isinstance(payload, Mapping):
            raise WebhookContractError("webhook payload must be an object")

        delivery_identity = resolve_delivery_identity(route, headers, payload)
        event_type = resolve_event_type(route, headers, payload)
        trace = _nonempty_scalar(trace_id) or str(uuid.uuid4())
        return cls(
            route=route,
            auth=WebhookAuthProvenance(
                provider=route.provider,
                signature_mode=route.signature_mode,
                provider_declared=route.provider_declared,
            ),
            event_type=event_type,
            delivery_identity=delivery_identity,
            trace_id=trace,
            body_sha256=hashlib.sha256(bytes(raw_body)).hexdigest(),
            payload=_freeze_json(payload),
        )

    @property
    def session_identity(self) -> str:
        """Stable provider ID when available, otherwise this request's trace."""

        if self.delivery_identity is not None:
            return self.delivery_identity.value
        return self.trace_id

    @property
    def idempotency_key(self) -> Optional[str]:
        """Profile/route/provider-scoped key, or None when dedupe is unsafe."""

        if self.delivery_identity is None:
            return None
        return ":".join(
            (
                self.route.profile,
                self.route.name,
                self.delivery_identity.provider,
                self.delivery_identity.value,
            )
        )
