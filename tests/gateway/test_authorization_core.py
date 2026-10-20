from __future__ import annotations

import asyncio
import gc
import pickle
import warnings
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, fields
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from gateway.authorization import TrustedPrincipalIssuer


def _issuer() -> TrustedPrincipalIssuer:
    from gateway.authorization import _bootstrap_principal_issuer

    return _bootstrap_principal_issuer()


def test_user_principal_is_frozen_and_has_a_fixed_typed_kind() -> None:
    from gateway.authorization import PrincipalKind

    principal = _issuer().issue_user(
        principal_id="telegram:123",
        authz_version=7,
        scopes={"agent.use", "rag.read"},
    )

    assert principal.kind is PrincipalKind.USER
    assert principal.scopes == frozenset({"agent.use", "rag.read"})
    with pytest.raises(FrozenInstanceError):
        setattr(principal, "authz_version", 8)
    assert {field.name for field in fields(principal)} == {
        "principal_id",
        "kind",
        "authz_version",
        "scopes",
        "initiating_principal_id",
    }


def test_payload_cannot_construct_or_select_a_principal_kind() -> None:
    import gateway.authorization as authorization

    from gateway.authorization import PrincipalKind, VerifiedPrincipal

    payload = {
        "principal_id": "model-selected-admin",
        "kind": PrincipalKind.SERVICE,
        "authz_version": 1,
        "scopes": frozenset({"users.manage"}),
    }

    with pytest.raises(TypeError):
        cast(Any, VerifiedPrincipal)(**payload)
    with pytest.raises(TypeError):
        cast(Any, VerifiedPrincipal)()
    assert not hasattr(VerifiedPrincipal, "_create")
    assert not hasattr(authorization, "verified_user_principal")
    assert not hasattr(authorization, "verified_service_principal")
    assert not hasattr(authorization, "_PRINCIPAL_CONSTRUCTION_TOKEN")
    assert not hasattr(_issuer(), "_issue")


def test_service_principal_has_a_fixed_service_kind() -> None:
    from gateway.authorization import PrincipalKind

    principal = _issuer().issue_service(
        principal_id="svc.cron.resource_monitor",
        authz_version=3,
        scopes={"metrics.read"},
    )

    assert principal.kind is PrincipalKind.SERVICE
    assert principal.initiating_principal_id is None


def test_issuer_capability_is_not_publicly_constructible_or_serializable() -> None:
    from gateway.authorization import TrustedPrincipalIssuer

    issuer = _issuer()

    with pytest.raises(TypeError, match="core bootstrap"):
        TrustedPrincipalIssuer()
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(issuer)


@pytest.mark.parametrize("principal_id", [None, 1, True, "", "   ", "x" * 257])
def test_issuer_rejects_invalid_principal_ids(principal_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _issuer().issue_user(
            principal_id=cast(str, principal_id), authz_version=0
        )


@pytest.mark.parametrize("authz_version", [None, True, False, 1.0, "1", -1])
def test_issuer_rejects_invalid_authz_versions(authz_version: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _issuer().issue_user(
            principal_id="telegram:123", authz_version=cast(int, authz_version)
        )


@pytest.mark.parametrize(
    "scopes",
    [
        "agent.use",
        b"agent.use",
        {"agent.use": True},
        [""],
        ["   "],
        [1],
        [True],
        ["x" * 257],
    ],
)
def test_issuer_rejects_invalid_scopes(scopes: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _issuer().issue_user(
            principal_id="telegram:123",
            authz_version=0,
            scopes=cast(Iterable[str], scopes),
        )


def test_on_behalf_of_principal_can_only_attenuate_user_scopes() -> None:
    from gateway.authorization import PrincipalKind

    issuer = _issuer()
    parent = issuer.issue_user(
        principal_id="telegram:123",
        authz_version=7,
        scopes={"agent.use", "rag.read"},
    )
    delegated = issuer.issue_on_behalf_of(
        parent=parent,
        scopes={"rag.read"},
    )

    assert delegated.kind is PrincipalKind.ON_BEHALF_OF
    assert delegated.principal_id == parent.principal_id
    assert delegated.initiating_principal_id == parent.principal_id
    assert delegated.authz_version == parent.authz_version
    assert delegated.scopes == frozenset({"rag.read"})

    with pytest.raises(ValueError, match="subset"):
        issuer.issue_on_behalf_of(
            parent=parent,
            scopes={"rag.read", "users.manage"},
        )


def test_on_behalf_of_issuance_rejects_unverified_or_service_parent() -> None:
    issuer = _issuer()

    with pytest.raises(TypeError, match="VerifiedPrincipal"):
        issuer.issue_on_behalf_of(
            parent=cast(Any, {"principal_id": "telegram:123"}),
            scopes={"agent.use"},
        )
    with pytest.raises(ValueError, match="service principals"):
        issuer.issue_on_behalf_of(
            parent=issuer.issue_service(
                principal_id="svc.scheduler", authz_version=1, scopes={"agent.use"}
            ),
            scopes={"agent.use"},
        )


def test_authorized_envelope_is_frozen_and_contains_no_message_or_secret() -> None:
    from gateway.authorization import AuthorizedEnvelope

    secret = "TOP-SECRET-MESSAGE-BODY"
    principal = _issuer().issue_user(
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
        setattr(envelope, "authz_version", 8)
    with pytest.raises(TypeError):
        cast(Any, AuthorizedEnvelope)(
            principal=principal,
            authz_version=7,
            message_body=secret,
        )


def test_authorized_envelope_rejects_a_stale_authz_version() -> None:
    from gateway.authorization import AuthorizedEnvelope

    principal = _issuer().issue_user(
        principal_id="telegram:123",
        authz_version=7,
    )

    with pytest.raises(ValueError, match="authz_version"):
        AuthorizedEnvelope(principal=principal, authz_version=6)


@pytest.mark.parametrize("authz_version", [True, False, 7.0, "7"])
def test_authorized_envelope_rejects_non_exact_int_version(
    authz_version: object,
) -> None:
    from gateway.authorization import AuthorizedEnvelope

    principal = _issuer().issue_user(
        principal_id="telegram:123", authz_version=7
    )

    with pytest.raises(TypeError, match="exact int"):
        cast(Any, AuthorizedEnvelope)(
            principal=principal, authz_version=authz_version
        )


def test_authorized_envelope_rejects_unverified_principal() -> None:
    from gateway.authorization import AuthorizedEnvelope

    with pytest.raises(TypeError, match="VerifiedPrincipal"):
        cast(Any, AuthorizedEnvelope)(
            principal={"principal_id": "telegram:123", "authz_version": 7},
            authz_version=7,
        )


def test_unavailable_provider_is_not_ready_and_denies_unknown_permissions() -> None:
    from gateway.authorization import (
        AuthorizationProvider,
        PermissionDecision,
        UnavailableAuthorizationProvider,
    )

    provider = UnavailableAuthorizationProvider()
    principal = _issuer().issue_user(
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
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    principal = _issuer().issue_user(
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
        AuthorizationProvider,
        PermissionDecision,
        authorize_fail_closed,
    )

    principal = _issuer().issue_user(
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
            cast(AuthorizationProvider, Provider()),
            principal=principal,
            permission="agent.use",
            known_permissions={"agent.use"},
            timeout_seconds=0.001,
        )
    )

    assert decision is PermissionDecision.DENY


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        None,
        "1",
        True,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        61,
        60.0001,
        pytest.param(10**10000, id="huge-int"),
    ],
)
def test_authorize_fail_closed_rejects_invalid_timeout_before_provider_calls(
    timeout_seconds: object,
) -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    async def scenario() -> None:
        provider_calls: list[str] = []
        unhandled: list[dict[str, object]] = []

        class Provider:
            def __getattribute__(self, name: str) -> Any:
                if name in {"readiness", "authorize"}:
                    provider_calls.append(name)
                return object.__getattribute__(self, name)

            async def readiness(self) -> bool:
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                return PermissionDecision.ALLOW

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        tasks_before = asyncio.all_tasks()
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=cast(int | float, timeout_seconds),
            )
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert decision is PermissionDecision.DENY
        assert provider_calls == []
        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_authorize_fail_closed_accepts_maximum_timeout() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

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
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
            timeout_seconds=60,
        )
    )

    assert decision is PermissionDecision.ALLOW
    assert calls == ["readiness", "authorize"]


@pytest.mark.parametrize("raise_on_factory_call", [1, 2])
@pytest.mark.parametrize("create_task_before_raise", [False, True])
def test_authorize_fail_closed_cleans_up_when_task_factory_raises(
    raise_on_factory_call: int,
    create_task_before_raise: bool,
) -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    caught_warnings: list[warnings.WarningMessage]

    async def scenario() -> None:
        unhandled: list[dict[str, object]] = []
        factory_calls = 0
        unrelated_started = asyncio.Event()

        async def unrelated_operation() -> None:
            unrelated_started.set()
            await asyncio.Event().wait()

        class Provider:
            async def readiness(self) -> bool:
                await asyncio.Event().wait()
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                return PermissionDecision.ALLOW

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        previous_factory = loop.get_task_factory()
        unrelated = asyncio.create_task(unrelated_operation())
        await unrelated_started.wait()
        tasks_before = asyncio.all_tasks()

        def raising_factory(
            task_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **kwargs: Any,
        ) -> asyncio.Task[Any]:
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == raise_on_factory_call:
                if create_task_before_raise:
                    asyncio.Task(coroutine, loop=task_loop, **kwargs)
                raise RuntimeError("task factory failed")
            return asyncio.Task(coroutine, loop=task_loop, **kwargs)

        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        loop.set_task_factory(raising_factory)
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=1,
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            loop.set_task_factory(previous_factory)
            loop.set_exception_handler(previous_handler)

        assert decision is PermissionDecision.DENY
        assert asyncio.all_tasks() == tasks_before
        assert not unrelated.cancelled()
        assert unhandled == []

        unrelated.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unrelated

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        asyncio.run(scenario())
        gc.collect()

    assert [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, RuntimeWarning)
    ] == []


def test_deadline_task_exception_is_retrieved_and_propagated() -> None:
    from gateway.authorization import _await_with_deadline

    async def scenario() -> None:
        factory_calls = 0
        unhandled: list[dict[str, object]] = []

        async def operation() -> None:
            await asyncio.Event().wait()

        async def broken_deadline() -> None:
            raise RuntimeError("deadline task failed")

        def deadline_failing_factory(
            task_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **kwargs: Any,
        ) -> asyncio.Task[Any]:
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 2:
                coroutine.close()
                coroutine = broken_deadline()
            return asyncio.Task(coroutine, loop=task_loop, **kwargs)

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        previous_factory = loop.get_task_factory()
        tasks_before = asyncio.all_tasks()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        loop.set_task_factory(deadline_failing_factory)
        try:
            operation_coroutine = operation()
            with pytest.raises(RuntimeError, match="deadline task failed"):
                await _await_with_deadline(operation_coroutine, 1)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            loop.set_task_factory(previous_factory)
            loop.set_exception_handler(previous_handler)

        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_authorize_fail_closed_denies_deadline_task_exception_without_leak() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    async def scenario() -> None:
        factory_calls = 0
        authorize_calls: list[bool] = []
        unhandled: list[dict[str, object]] = []

        async def broken_deadline() -> None:
            raise RuntimeError("deadline task failed")

        class Provider:
            async def readiness(self) -> bool:
                await asyncio.Event().wait()
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                authorize_calls.append(True)
                return PermissionDecision.ALLOW

        def deadline_failing_factory(
            task_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **kwargs: Any,
        ) -> asyncio.Task[Any]:
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 2:
                coroutine.close()
                coroutine = broken_deadline()
            return asyncio.Task(coroutine, loop=task_loop, **kwargs)

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        previous_factory = loop.get_task_factory()
        tasks_before = asyncio.all_tasks()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        loop.set_task_factory(deadline_failing_factory)
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=1,
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            loop.set_task_factory(previous_factory)
            loop.set_exception_handler(previous_handler)

        assert decision is PermissionDecision.DENY
        assert authorize_calls == []
        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_readiness_timeout_wins_when_provider_suppresses_cancellation() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    authorize_calls: list[bool] = []

    class Provider:
        async def readiness(self) -> bool:
            try:
                await asyncio.sleep(60)
                raise AssertionError("readiness deadline did not fire")
            except asyncio.CancelledError:
                return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            authorize_calls.append(True)
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
            timeout_seconds=0.001,
        )
    )

    assert decision is PermissionDecision.DENY
    assert authorize_calls == []


def test_authorize_timeout_wins_when_provider_returns_allow_after_cancellation() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    class Provider:
        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            try:
                await asyncio.sleep(60)
                raise AssertionError("authorization deadline did not fire")
            except asyncio.CancelledError:
                return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
            timeout_seconds=0.001,
        )
    )

    assert decision is PermissionDecision.DENY


def test_authorize_fail_closed_preserves_external_caller_cancellation() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    async def scenario() -> None:
        started = asyncio.Event()

        class Provider:
            async def readiness(self) -> bool:
                started.set()
                await asyncio.sleep(60)
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                return PermissionDecision.ALLOW

        task = asyncio.create_task(
            authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=60,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_authorize_fail_closed_denies_unknown_without_backend_call() -> None:
    from gateway.authorization import (
        PermissionDecision,
        authorize_fail_closed,
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
            principal=_issuer().issue_user(
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
    from gateway.authorization import AuthorizationProvider, authorize_fail_closed

    secret = "TOP-SECRET-MESSAGE-BODY"

    class FailingProvider:
        async def readiness(self) -> bool:
            raise RuntimeError(secret)

        async def authorize(self, **_kwargs: object) -> object:
            raise AssertionError("authorize must not run")

    asyncio.run(
        authorize_fail_closed(
            cast(AuthorizationProvider, FailingProvider()),
            principal=_issuer().issue_user(
                principal_id="telegram:123",
                authz_version=7,
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
            resource={"message_body": secret},
        )
    )

    assert secret not in caplog.text
