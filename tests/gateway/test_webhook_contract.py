"""Contract tests for canonical webhook provider/identity/envelope authority."""

import hashlib
from types import MappingProxyType

import pytest

from gateway.platforms.webhook_contract import (
    PROVIDER_REGISTRY,
    WebhookContractError,
    WebhookEnvelope,
    WebhookRouteConfig,
    canonical_provider,
    canonical_signature_mode,
    infer_legacy_provider,
    resolve_delivery_identity,
    resolve_event_type,
)


def bind(route=None, headers=None, *, profile=None):
    return WebhookRouteConfig.bind(
        "events",
        route or {},
        headers=headers or {},
        request_profile=profile,
    )


def test_registry_is_read_only_and_has_expected_namespaces():
    assert isinstance(PROVIDER_REGISTRY, MappingProxyType)
    assert {
        "github",
        "gitlab",
        "svix",
        "standard_webhooks",
        "chatwoot",
        "linear",
        "hindsight",
        "stripe",
        "generic",
    } <= set(PROVIDER_REGISTRY)
    with pytest.raises(TypeError):
        PROVIDER_REGISTRY["evil"] = PROVIDER_REGISTRY["generic"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("agentmail", "svix"),
        ("github-hmac-sha256", "github"),
        ("gitlab_token", "gitlab"),
        ("gitlab-standard", "standard_webhooks"),
        ("generic_v1", "generic"),
        ("generic-v2", "generic"),
        ("hindsight_hmac_sha256", "hindsight"),
    ],
)
def test_provider_aliases_canonicalize(configured, expected):
    assert canonical_provider(configured) == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("github", "github"),
        ("gitlab-token", "gitlab_token"),
        ("generic-v2", "generic_v2"),
        ("gitlab_standard", "gitlab_standard"),
    ],
)
def test_signature_modes_canonicalize(configured, expected):
    assert canonical_signature_mode(configured) == expected


def test_unknown_explicit_provider_fails_closed():
    with pytest.raises(WebhookContractError, match="unsupported webhook provider"):
        bind({"provider": "made-up"})


def test_unknown_explicit_signature_mode_fails_closed():
    with pytest.raises(WebhookContractError, match="unsupported webhook signature mode"):
        bind({"signature_mode": "made-up"})


def test_provider_and_verifier_mode_cannot_disagree():
    with pytest.raises(WebhookContractError, match="does not allow signature mode"):
        bind({"provider": "github", "signature_mode": "svix"})


def test_provider_can_use_registered_generic_verifier_when_contract_allows_it():
    route = bind({"provider": "chatwoot", "signature_mode": "generic_v2"})
    assert route.provider == "chatwoot"
    assert route.signature_mode == "generic_v2"


def test_explicit_provider_cannot_be_reselected_by_attacker_headers():
    route = bind(
        {"provider": "github"},
        {
            "svix-id": "msg_attacker",
            "svix-signature": "v1,attacker",
            "X-GitHub-Delivery": "gh_123",
        },
    )
    assert route.provider == "github"
    assert route.provider_declared is True
    identity = resolve_delivery_identity(
        route,
        {
            "svix-id": "msg_attacker",
            "X-GitHub-Delivery": "gh_123",
        },
        {},
    )
    assert identity is not None
    assert identity.provider == "github"
    assert identity.value == "gh_123"


def test_explicit_generic_route_ignores_github_event_header():
    route = bind({"provider": "generic"})
    assert (
        resolve_event_type(
            route,
            {"X-GitHub-Event": "push"},
            {"event_type": "generic.tick"},
        )
        == "generic.tick"
    )


def test_legacy_inference_happens_once_with_specific_provider_precedence():
    headers = {
        "svix-id": "msg_1",
        "X-Hub-Signature-256": "sha256=also-present",
        "X-Request-ID": "generic-id",
    }
    assert infer_legacy_provider(headers) == "svix"
    route = bind({}, headers)
    assert route.provider == "svix"
    assert route.provider_declared is False
    identity = resolve_delivery_identity(
        route,
        {"X-GitHub-Delivery": "gh_2", "svix-id": "msg_1"},
        {},
    )
    assert identity is not None
    assert identity.provider == "svix"
    assert identity.value == "msg_1"


def test_signature_mode_can_bind_provider_namespace():
    route = bind({"signature_mode": "generic_v2"})
    assert route.provider == "generic"
    assert route.signature_mode == "generic_v2"
    assert route.provider_declared is True


def test_gitlab_uses_provider_native_identity_order():
    route = bind({"provider": "gitlab"})
    identity = resolve_delivery_identity(
        route,
        {
            "Idempotency-Key": "fallback",
            "X-Gitlab-Webhook-UUID": "webhook-uuid",
            "X-Gitlab-Event-UUID": "event-uuid",
        },
        {},
    )
    assert identity is not None
    assert identity.value == "event-uuid"


def test_no_stable_provider_id_means_no_idempotency_authority():
    route = bind({"provider": "generic"})
    envelope = WebhookEnvelope.build(
        route,
        headers={"X-Webhook-Signature-V2": "sig"},
        payload={"type": "tick"},
        raw_body=b'{"type":"tick"}',
        trace_id="trace-1",
    )
    assert envelope.delivery_identity is None
    assert envelope.idempotency_key is None
    assert envelope.session_identity == "trace-1"


def test_timestamp_headers_are_never_delivery_identity():
    route = bind({"provider": "svix"})
    envelope = WebhookEnvelope.build(
        route,
        headers={"svix-timestamp": "1787248000"},
        payload={},
        raw_body=b"{}",
        trace_id="trace-2",
    )
    assert envelope.delivery_identity is None
    assert envelope.idempotency_key is None
    assert envelope.session_identity == "trace-2"


@pytest.mark.parametrize("provider", ["stripe", "chatwoot"])
def test_payload_id_is_accepted_only_for_explicit_provider(provider):
    explicit = bind({"provider": provider})
    identity = resolve_delivery_identity(explicit, {}, {"id": "evt_123"})
    assert identity is not None
    assert identity.provider == provider
    assert identity.value == "evt_123"

    legacy = bind({}, {})
    assert legacy.provider_declared is False
    assert resolve_delivery_identity(legacy, {}, {"id": "evt_123"}) is None


def test_idempotency_key_is_profile_route_provider_scoped():
    route = WebhookRouteConfig.bind(
        "deploy",
        {"profile": "ops", "provider": "github"},
        headers={},
        request_profile="ops",
    )
    envelope = WebhookEnvelope.build(
        route,
        headers={"X-GitHub-Delivery": "abc"},
        payload={},
        raw_body=b"{}",
        trace_id="trace-3",
    )
    assert envelope.idempotency_key == "ops:deploy:github:abc"


def test_route_profile_mismatch_fails_closed():
    with pytest.raises(WebhookContractError, match="not bound to profile"):
        WebhookRouteConfig.bind(
            "deploy",
            {"profile": "ops", "provider": "github"},
            headers={},
            request_profile="default",
        )


def test_envelope_captures_body_hash_and_auth_provenance():
    raw_body = b'{"action":"opened"}'
    route = bind({"provider": "github", "signature_mode": "github"})
    envelope = WebhookEnvelope.build(
        route,
        headers={
            "X-GitHub-Delivery": "delivery-1",
            "X-GitHub-Event": "pull_request",
        },
        payload={"action": "opened"},
        raw_body=raw_body,
        trace_id="trace-4",
    )
    assert envelope.event_type == "pull_request"
    assert envelope.auth.provider == "github"
    assert envelope.auth.signature_mode == "github"
    assert envelope.auth.compatibility_inferred is False
    assert envelope.body_sha256 == hashlib.sha256(raw_body).hexdigest()


def test_envelope_payload_is_recursively_immutable():
    payload = {"outer": {"items": [1, {"x": 2}]}}
    envelope = WebhookEnvelope.build(
        bind({"provider": "generic"}),
        headers={},
        payload=payload,
        raw_body=b"{}",
        trace_id="trace-5",
    )
    with pytest.raises(TypeError):
        envelope.payload["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        envelope.payload["outer"]["new"] = 1  # type: ignore[index]
    assert envelope.payload["outer"]["items"][1]["x"] == 2


def test_mutating_source_payload_after_build_does_not_mutate_envelope():
    payload = {"nested": {"value": "original"}}
    envelope = WebhookEnvelope.build(
        bind({"provider": "generic"}),
        headers={},
        payload=payload,
        raw_body=b"{}",
        trace_id="trace-6",
    )
    payload["nested"]["value"] = "changed"
    assert envelope.payload["nested"]["value"] == "original"


def test_malformed_route_events_are_rejected():
    with pytest.raises(WebhookContractError, match="events must be a sequence"):
        bind({"provider": "github", "events": "push"})
