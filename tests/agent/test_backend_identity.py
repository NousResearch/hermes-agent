"""Owner-level tests for agent.backend_identity.

This module is the single owner of the "same backend?" question — tests
live HERE, against the predicate, not against each call site. Each test names
the incident whose semantics it pins.
"""

from unittest.mock import patch

from agent.backend_identity import (
    BackendIdentity,
    FailureScope,
    classify_failure_scope,
    same_credential_surface,
    same_deployment,
    same_endpoint,
    should_skip_candidate,
)


def _id(provider="", model="", base_url=""):
    return BackendIdentity.build(provider=provider, model=model, base_url=base_url)


class TestClassifyFailureScope:
    def test_auth_and_payment_are_credential_scoped(self):
        assert classify_failure_scope("auth error") is FailureScope.CREDENTIAL
        assert classify_failure_scope("payment error") is FailureScope.CREDENTIAL

    def test_model_scoped_reasons(self):
        for reason in (
            "rate limit",
            "timeout",
            "connection error",
            "model incompatible with route",
            "invalid provider response",
        ):
            assert classify_failure_scope(reason) is FailureScope.MODEL, reason

    def test_unknown_reason_defaults_to_least_invalidating_scope(self):
        """Never over-skip on a reason string we don't recognize."""
        assert classify_failure_scope("weird new error") is FailureScope.MODEL
        assert classify_failure_scope(None) is FailureScope.MODEL
        assert classify_failure_scope("") is FailureScope.MODEL


class TestSameDeployment:
    def test_incident_59561_72468_sibling_model_same_provider_is_different(self):
        """aux glm-5.2 timing out says nothing about main macaron on the
        same custom endpoint — sibling models are independent deployments."""
        failed = _id("custom", "zai-org/glm-5.2")
        sibling = _id("custom", "mindai/macaron-v1-venti")
        assert not same_deployment(sibling, failed)
        assert not should_skip_candidate(sibling, failed, FailureScope.MODEL)


    def test_incident_62984_same_model_different_explicit_url_is_a_pool(self):
        """Several LM Studio endpoints serving one model = a pool, not dups."""
        a = _id("custom", "qwen-3", "http://box1:1234/v1")
        b = _id("custom", "qwen-3", "http://box2:1234/v1")
        assert not same_deployment(a, b)
        assert not should_skip_candidate(a, b, FailureScope.MODEL)


    def test_incident_22548_shim_aliases_same_url_same_model_are_same(self):
        """Two custom_providers aliases at one shim URL with one model."""
        a = _id("claude-cli", "claude-opus-4.7", "http://127.0.0.1:7891/v1")
        b = _id("claude-cli-alt", "claude-opus-4.7", "http://127.0.0.1:7891/v1")
        assert same_deployment(a, b)

    def test_incident_70893_first_class_pair_same_host_is_not_same(self):
        """xai-oauth vs xai share api.x.ai but are distinct backends."""
        fake_registry = {"xai": object(), "xai-oauth": object()}
        with patch("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry):
            a = _id("xai-oauth", "grok-4.5", "https://api.x.ai/v1")
            b = _id("xai", "grok-4.5", "https://api.x.ai/v1")
            assert not same_deployment(a, b)
            assert not should_skip_candidate(a, b, FailureScope.MODEL)


class TestSameCredentialSurface:
    def test_same_provider_label_shares_credential(self):
        """Auth/payment failure on a provider kills every model on it —
        including the main model (the #59561/#72468 carve-out)."""
        failed = _id("custom", "zai-org/glm-5.2")
        sibling = _id("custom", "mindai/macaron-v1-venti")
        assert same_credential_surface(sibling, failed)
        assert should_skip_candidate(sibling, failed, FailureScope.CREDENTIAL)

    def test_incident_70893_distinct_first_class_providers_differ(self):
        fake_registry = {"xai": object(), "xai-oauth": object()}
        with patch("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry):
            a = _id("xai", "grok-4.5", "https://api.x.ai/v1")
            b = _id("xai-oauth", "grok-4.5", "https://api.x.ai/v1")
            assert not same_credential_surface(a, b)
            assert not should_skip_candidate(a, b, FailureScope.CREDENTIAL)

    def test_distinct_custom_labels_same_url_not_assumed_shared(self):
        """Two custom entries can carry their own api_key each — sameness is
        unprovable, so failover must be allowed (conservative direction)."""
        a = _id("proxy-a", "m", "http://gw:9000/v1")
        b = _id("proxy-b", "m", "http://gw:9000/v1")
        assert not same_credential_surface(a, b)



class TestSameEndpoint:
    def test_same_explicit_url_is_same_endpoint(self):
        a = _id("a", "m1", "http://host:8000/v1/")
        b = _id("b", "m2", "http://HOST:8000/v1")  # trailing slash + case
        assert same_endpoint(a, b)
        assert should_skip_candidate(a, b, FailureScope.ENDPOINT)

    def test_different_urls_are_different_endpoints(self):
        assert not same_endpoint(
            _id("a", "m", "http://h1/v1"), _id("a2", "m", "http://h2/v1")
        )

    def test_unknown_url_falls_back_to_provider_default(self):
        assert same_endpoint(_id("openrouter", "m1"), _id("openrouter", "m2"))
        assert not same_endpoint(_id("openrouter", "m"), _id("nous", "m"))


class TestUnknownAxesNeverStrand:
    """The failure mode that produced this module: over-skipping. An
    unprovable axis must never manufacture a skip."""

    def test_empty_candidate_never_skipped(self):
        failed = _id("custom", "glm", "http://h/v1")
        for scope in FailureScope:
            assert not should_skip_candidate(_id(), failed, scope), scope

    def test_model_scope_requires_model_evidence(self):
        # Failed side has no model recorded → cannot prove the candidate is
        # the same deployment → do not skip.
        failed = _id("custom")
        candidate = _id("custom", "some-model")
        assert not should_skip_candidate(candidate, failed, FailureScope.MODEL)
