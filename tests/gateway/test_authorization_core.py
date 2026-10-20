from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields

import pytest


def test_user_principal_is_frozen_and_has_a_fixed_typed_kind() -> None:
    from gateway.authorization import PrincipalKind, verified_user_principal

    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
        scopes={"agent.use", "rag.read"},
    )

    assert principal.kind is PrincipalKind.USER
    assert principal.scopes == frozenset({"agent.use", "rag.read"})
    with pytest.raises(FrozenInstanceError):
        principal.authz_version = 8  # type: ignore[misc]
    assert {field.name for field in fields(principal)} == {
        "principal_id",
        "kind",
        "authz_version",
        "scopes",
        "initiating_principal_id",
    }


def test_payload_cannot_select_a_principal_kind() -> None:
    from gateway.authorization import PrincipalKind, VerifiedPrincipal

    payload = {
        "principal_id": "model-selected-admin",
        "kind": PrincipalKind.SERVICE,
        "authz_version": 1,
        "scopes": frozenset({"users.manage"}),
    }

    with pytest.raises(TypeError):
        VerifiedPrincipal(**payload)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VerifiedPrincipal()


def test_service_principal_has_a_fixed_service_kind() -> None:
    from gateway.authorization import PrincipalKind, verified_service_principal

    principal = verified_service_principal(
        principal_id="svc.cron.resource_monitor",
        authz_version=3,
        scopes={"metrics.read"},
    )

    assert principal.kind is PrincipalKind.SERVICE
    assert principal.initiating_principal_id is None


def test_on_behalf_of_principal_can_only_attenuate_user_scopes() -> None:
    from gateway.authorization import (
        PrincipalKind,
        attenuated_on_behalf_of_principal,
        verified_user_principal,
    )

    parent = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
        scopes={"agent.use", "rag.read"},
    )
    delegated = attenuated_on_behalf_of_principal(
        parent=parent,
        scopes={"rag.read"},
    )

    assert delegated.kind is PrincipalKind.ON_BEHALF_OF
    assert delegated.principal_id == parent.principal_id
    assert delegated.initiating_principal_id == parent.principal_id
    assert delegated.authz_version == parent.authz_version
    assert delegated.scopes == frozenset({"rag.read"})

    with pytest.raises(ValueError, match="subset"):
        attenuated_on_behalf_of_principal(
            parent=parent,
            scopes={"rag.read", "users.manage"},
        )


def test_authorized_envelope_is_frozen_and_contains_no_message_or_secret() -> None:
    from gateway.authorization import AuthorizedEnvelope, verified_user_principal

    secret = "TOP-SECRET-MESSAGE-BODY"
    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
        scopes={"agent.use"},
    )
    envelope = AuthorizedEnvelope(principal=principal, authz_version=7)

    assert {field.name for field in fields(envelope)} == {
        "principal",
        "authz_version",
    }
    assert secret not in repr(principal)
    assert secret not in repr(envelope)
    with pytest.raises(FrozenInstanceError):
        envelope.authz_version = 8  # type: ignore[misc]
    with pytest.raises(TypeError):
        AuthorizedEnvelope(  # type: ignore[call-arg]
            principal=principal,
            authz_version=7,
            message_body=secret,
        )


def test_authorized_envelope_rejects_a_stale_authz_version() -> None:
    from gateway.authorization import AuthorizedEnvelope, verified_user_principal

    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
    )

    with pytest.raises(ValueError, match="authz_version"):
        AuthorizedEnvelope(principal=principal, authz_version=6)


def test_unavailable_provider_is_not_ready_and_denies_unknown_permissions() -> None:
    from gateway.authorization import (
        AuthorizationProvider,
        PermissionDecision,
        UnavailableAuthorizationProvider,
        verified_user_principal,
    )

    provider = UnavailableAuthorizationProvider()
    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
    )

    assert isinstance(provider, AuthorizationProvider)
    assert asyncio.run(provider.readiness()) is False
    decision = asyncio.run(
        provider.authorize(
            principal=principal,
            permission="future.unknown.permission",
            resource={"resource_id": "safe-metadata"},
        )
    )
    assert decision is PermissionDecision.DENY
    assert {item.value for item in PermissionDecision} == {"allow", "deny"}


def test_authorize_fail_closed_allows_only_explicit_allow() -> None:
    from gateway.authorization import (
        PermissionDecision,
        authorize_fail_closed,
        verified_user_principal,
    )

    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
    )

    class Provider:
        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=principal,
            permission="agent.use",
            known_permissions={"agent.use"},
        )
    )

    assert decision is PermissionDecision.ALLOW


@pytest.mark.parametrize(
    "failure",
    ["not_ready", "readiness_error", "authorize_error", "timeout", "invalid"],
)
def test_authorize_fail_closed_denies_readiness_errors_timeouts_and_non_allow(
    failure: str,
) -> None:
    from gateway.authorization import (
        PermissionDecision,
        authorize_fail_closed,
        verified_user_principal,
    )

    principal = verified_user_principal(
        principal_id="telegram:123",
        authz_version=7,
    )

    class Provider:
        async def readiness(self) -> bool:
            if failure == "readiness_error":
                raise RuntimeError("backend unavailable")
            return failure != "not_ready"

        async def authorize(self, **_kwargs: object) -> object:
            if failure == "authorize_error":
                raise RuntimeError("backend failed")
            if failure == "timeout":
                await asyncio.sleep(0.05)
            if failure == "invalid":
                return "allow"
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),  # type: ignore[arg-type]
            principal=principal,
            permission="agent.use",
            known_permissions={"agent.use"},
            timeout_seconds=0.001,
        )
    )

    assert decision is PermissionDecision.DENY


def test_authorize_fail_closed_denies_unknown_without_backend_call() -> None:
    from gateway.authorization import (
        PermissionDecision,
        authorize_fail_closed,
        verified_user_principal,
    )

    calls: list[str] = []

    class Provider:
        async def readiness(self) -> bool:
            calls.append("readiness")
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            calls.append("authorize")
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=verified_user_principal(
                principal_id="telegram:123",
                authz_version=7,
            ),
            permission="unknown.permission",
            known_permissions={"agent.use"},
        )
    )

    assert decision is PermissionDecision.DENY
    assert calls == []


def test_authorize_fail_closed_does_not_log_message_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.authorization import authorize_fail_closed, verified_user_principal

    secret = "TOP-SECRET-MESSAGE-BODY"

    class FailingProvider:
        async def readiness(self) -> bool:
            raise RuntimeError(secret)

        async def authorize(self, **_kwargs: object) -> object:
            raise AssertionError("authorize must not run")

    asyncio.run(
        authorize_fail_closed(
            FailingProvider(),  # type: ignore[arg-type]
            principal=verified_user_principal(
                principal_id="telegram:123",
                authz_version=7,
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
            resource={"message_body": secret},
        )
    )

    assert secret not in caplog.text
