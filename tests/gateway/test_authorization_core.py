from __future__ import annotations

import asyncio
import gc
import pickle
import types
import warnings
from collections.abc import Iterable, Set as AbstractSet
from dataclasses import FrozenInstanceError, fields
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from gateway.authorization import TrustedPrincipalIssuer


class _PermissionSubclass(str):
    pass


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


def test_on_behalf_of_issuance_rejects_verified_principal_subclass() -> None:
    from gateway.authorization import VerifiedPrincipal

    class ForgedPrincipal(VerifiedPrincipal):
        pass

    legitimate = _issuer().issue_user(
        principal_id="telegram:123", authz_version=7, scopes={"agent.use"}
    )
    forged = object.__new__(ForgedPrincipal)
    for field in fields(legitimate):
        object.__setattr__(forged, field.name, getattr(legitimate, field.name))

    with pytest.raises(TypeError, match="VerifiedPrincipal"):
        _issuer().issue_on_behalf_of(parent=forged, scopes={"agent.use"})


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


def test_authorized_envelope_rejects_verified_principal_subclass() -> None:
    from gateway.authorization import AuthorizedEnvelope, VerifiedPrincipal

    class ForgedPrincipal(VerifiedPrincipal):
        pass

    legitimate = _issuer().issue_user(principal_id="telegram:123", authz_version=7)
    forged = object.__new__(ForgedPrincipal)
    for field in fields(legitimate):
        object.__setattr__(forged, field.name, getattr(legitimate, field.name))

    with pytest.raises(TypeError, match="VerifiedPrincipal"):
        AuthorizedEnvelope(principal=forged, authz_version=7)


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


@pytest.mark.parametrize(
    ("principal", "permission", "known_permissions"),
    [
        ({"principal_id": "forged"}, "agent.use", {"agent.use"}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), [], {"agent.use"}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), _PermissionSubclass("agent.use"), {"agent.use"}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "", {""}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "   ", {"   "}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "x" * 257, {"x" * 257}),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "agent.use", ["agent.use"]),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "agent.use", None),
        (_issuer().issue_user(principal_id="telegram:123", authz_version=7), "agent.use", {"agent.use", 7}),
    ],
)
def test_authorize_fail_closed_rejects_malformed_inputs_before_provider_access(
    principal: object,
    permission: object,
    known_permissions: object,
) -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    provider_accesses: list[str] = []

    class Provider:
        def __getattribute__(self, name: str) -> Any:
            if name in {"readiness", "authorize"}:
                provider_accesses.append(name)
            return object.__getattribute__(self, name)

        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=cast(Any, principal),
            permission=cast(Any, permission),
            known_permissions=cast(Any, known_permissions),
        )
    )

    assert decision is PermissionDecision.DENY
    assert provider_accesses == []


def test_authorize_fail_closed_rejects_verified_principal_subclass_before_provider_access() -> None:
    from gateway.authorization import (
        PermissionDecision,
        VerifiedPrincipal,
        authorize_fail_closed,
    )

    class ForgedPrincipal(VerifiedPrincipal):
        pass

    legitimate = _issuer().issue_user(principal_id="telegram:123", authz_version=7)
    forged = object.__new__(ForgedPrincipal)
    for field in fields(legitimate):
        object.__setattr__(forged, field.name, getattr(legitimate, field.name))

    provider_accesses: list[str] = []

    class Provider:
        def __getattribute__(self, name: str) -> Any:
            if name in {"readiness", "authorize"}:
                provider_accesses.append(name)
            return object.__getattribute__(self, name)

        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=forged,
            permission="agent.use",
            known_permissions={"agent.use"},
        )
    )

    assert decision is PermissionDecision.DENY
    assert provider_accesses == []


def test_authorize_fail_closed_denies_known_permission_membership_failure() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    provider_accesses: list[str] = []

    class BrokenPermissions(set[str]):
        def __contains__(self, _permission: object) -> bool:
            raise TypeError("membership unavailable")

    class Provider:
        def __getattribute__(self, name: str) -> Any:
            if name in {"readiness", "authorize"}:
                provider_accesses.append(name)
            return object.__getattribute__(self, name)

        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions=cast(Any, BrokenPermissions({"agent.use"})),
        )
    )

    assert decision is PermissionDecision.DENY
    assert provider_accesses == []


def test_authorize_fail_closed_rejects_custom_set_without_using_it() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    provider_accesses: list[str] = []
    set_accesses: list[str] = []

    class InfinitePermissions(AbstractSet[str]):
        def __contains__(self, _permission: object) -> bool:
            set_accesses.append("contains")
            raise AssertionError("custom membership must not run")

        def __iter__(self) -> Any:
            set_accesses.append("iter")
            raise AssertionError("custom iteration must not run")

        def __len__(self) -> int:
            set_accesses.append("len")
            raise AssertionError("custom length must not run")

    class Provider:
        def __getattribute__(self, name: str) -> Any:
            if name in {"readiness", "authorize"}:
                provider_accesses.append(name)
            return object.__getattribute__(self, name)

        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions=cast(Any, InfinitePermissions()),
        )
    )

    assert decision is PermissionDecision.DENY
    assert set_accesses == []
    assert provider_accesses == []


def test_authorize_fail_closed_rejects_oversized_exact_permission_set() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    provider_accesses: list[str] = []

    class Provider:
        def __getattribute__(self, name: str) -> Any:
            if name in {"readiness", "authorize"}:
                provider_accesses.append(name)
            return object.__getattribute__(self, name)

        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    permissions = {f"permission.{index}" for index in range(513)}
    permissions.add("agent.use")
    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions=permissions,
        )
    )

    assert decision is PermissionDecision.DENY
    assert provider_accesses == []


@pytest.mark.parametrize(
    "known_permissions", [{"agent.use"}, frozenset({"agent.use"})]
)
def test_authorize_fail_closed_accepts_exact_builtin_permission_sets(
    known_permissions: set[str] | frozenset[str],
) -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    class Provider:
        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions=known_permissions,
        )
    )

    assert decision is PermissionDecision.ALLOW


def test_authorize_fail_closed_accepts_maximum_permission_length() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    permission = "p" * 256

    class Provider:
        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    decision = asyncio.run(
        authorize_fail_closed(
            Provider(),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission=permission,
            known_permissions={permission},
        )
    )

    assert decision is PermissionDecision.ALLOW


@pytest.mark.parametrize("protected_invocation", [1, 2])
@pytest.mark.parametrize("factory_style", ["eager", "wrapper"])
def test_authorize_fail_closed_bypasses_custom_task_factories(
    protected_invocation: int,
    factory_style: str,
) -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    async def scenario() -> None:
        factory_calls = 0
        unrelated_started = asyncio.Event()
        release_unrelated = asyncio.Event()

        async def unrelated_operation() -> None:
            unrelated_started.set()
            await release_unrelated.wait()

        async def wrapper(coroutine: Any) -> Any:
            return await coroutine

        def custom_factory(
            task_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **kwargs: Any,
        ) -> asyncio.Future[Any]:
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == protected_invocation:
                raise AssertionError(f"{factory_style} factory touched owned operation")
            if factory_style == "eager":
                try:
                    coroutine.send(None)
                except StopIteration as stopped:
                    completed = task_loop.create_future()
                    completed.set_result(stopped.value)
                    return completed
                return asyncio.Task(coroutine, loop=task_loop, **kwargs)
            return asyncio.Task(wrapper(coroutine), loop=task_loop, **kwargs)

        class Provider:
            async def readiness(self) -> bool:
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                return PermissionDecision.ALLOW

        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        unrelated = asyncio.Task(unrelated_operation(), loop=loop)
        await unrelated_started.wait()
        loop.set_task_factory(custom_factory)
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
            )
        finally:
            loop.set_task_factory(previous_factory)

        assert decision is PermissionDecision.ALLOW
        assert factory_calls == 0
        assert not unrelated.done()
        assert not unrelated.cancelled()
        release_unrelated.set()
        await unrelated

    asyncio.run(scenario())


def test_authorize_fail_closed_rejects_current_loop_future_and_task_without_touching_them() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    async def scenario() -> None:
        release = asyncio.Event()
        current_future = asyncio.get_running_loop().create_future()
        current_task = asyncio.create_task(release.wait())
        authorize_calls = 0

        class Provider:
            def __init__(self, returned: object) -> None:
                self.returned = returned

            def readiness(self) -> object:
                return self.returned

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                nonlocal authorize_calls
                authorize_calls += 1
                return PermissionDecision.ALLOW

        for returned in (current_future, current_task):
            decision = await authorize_fail_closed(
                cast(Any, Provider(returned)),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
            )
            assert decision is PermissionDecision.DENY

        assert authorize_calls == 0
        assert not current_future.done()
        assert not current_task.done()
        assert not current_task.cancelling()
        release.set()
        await current_task
        current_future.cancel()

    asyncio.run(scenario())


def test_authorize_fail_closed_rejects_cross_loop_future_and_task() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    other_loop = asyncio.new_event_loop()

    async def completed() -> bool:
        return True

    cross_loop_future = other_loop.create_future()
    cross_loop_future.set_result(True)
    cross_loop_task = other_loop.create_task(completed())
    other_loop.run_until_complete(cross_loop_task)

    class Provider:
        def __init__(self, returned: object) -> None:
            self.returned = returned

        def readiness(self) -> object:
            return self.returned

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            raise AssertionError("authorize must not run")

    try:
        for returned in (cross_loop_future, cross_loop_task):
            decision = asyncio.run(
                authorize_fail_closed(
                    cast(Any, Provider(returned)),
                    principal=_issuer().issue_user(
                        principal_id="telegram:123", authz_version=7
                    ),
                    permission="agent.use",
                    known_permissions={"agent.use"},
                )
            )
            assert decision is PermissionDecision.DENY
    finally:
        other_loop.close()


def test_authorize_fail_closed_rejects_custom_awaitable_without_touching_methods() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    touched: list[str] = []

    class HostileAwaitable:
        def __await__(self) -> Any:
            touched.append("__await__")
            raise AssertionError("authorization invoked __await__")

        def cancel(self) -> None:
            touched.append("cancel")
            raise AssertionError("authorization invoked cancel")

        def close(self) -> None:
            touched.append("close")
            raise AssertionError("authorization invoked close")

        def result(self) -> object:
            touched.append("result")
            raise AssertionError("authorization invoked result")

        def add_done_callback(self, _callback: object) -> None:
            touched.append("add_done_callback")
            raise AssertionError("authorization invoked add_done_callback")

    class Provider:
        async def readiness(self) -> bool:
            return True

        def authorize(self, **_kwargs: object) -> HostileAwaitable:
            return HostileAwaitable()

    decision = asyncio.run(
        authorize_fail_closed(
            cast(Any, Provider()),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
        )
    )

    assert decision is PermissionDecision.DENY
    assert touched == []


def test_authorize_fail_closed_rejects_generator_based_coroutine_without_starting_or_closing_it() -> None:
    from gateway.authorization import PermissionDecision, authorize_fail_closed

    lifecycle: list[str] = []

    @types.coroutine
    def generator_coroutine() -> Any:
        lifecycle.append("started")
        try:
            yield
        finally:
            lifecycle.append("closed")

    returned = generator_coroutine()

    class Provider:
        def readiness(self) -> object:
            return returned

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            raise AssertionError("authorize must not run")

    decision = asyncio.run(
        authorize_fail_closed(
            cast(Any, Provider()),
            principal=_issuer().issue_user(
                principal_id="telegram:123", authz_version=7
            ),
            permission="agent.use",
            known_permissions={"agent.use"},
        )
    )

    assert decision is PermissionDecision.DENY
    assert lifecycle == []
    cast(Any, returned).close()


def test_authorize_fail_closed_uses_an_exact_owned_task_for_native_coroutines() -> None:
    import gateway.authorization as authorization

    from gateway.authorization import PermissionDecision, authorize_fail_closed

    owned_types: list[type[object]] = []
    original_wait = authorization.asyncio.wait

    async def recording_wait(awaitables: Any, **kwargs: Any) -> Any:
        operation = next(
            awaitable
            for awaitable in awaitables
            if type(awaitable) is asyncio.Task
        )
        owned_types.append(type(operation))
        return await original_wait(awaitables, **kwargs)

    class Provider:
        async def readiness(self) -> bool:
            return True

        async def authorize(self, **_kwargs: object) -> PermissionDecision:
            return PermissionDecision.ALLOW

    setattr(authorization.asyncio, "wait", recording_wait)
    try:
        decision = asyncio.run(
            authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
            )
        )
    finally:
        setattr(authorization.asyncio, "wait", original_wait)

    assert decision is PermissionDecision.ALLOW
    assert owned_types == [asyncio.Task, asyncio.Task]


@pytest.mark.parametrize("phase", ["readiness", "authorize"])
@pytest.mark.parametrize(
    "failure_point", ["create_future", "call_later", "wait", "task"]
)
def test_authorize_fail_closed_closes_provider_coroutine_on_setup_failure(
    phase: str,
    failure_point: str,
) -> None:
    import gateway.authorization as authorization

    from gateway.authorization import PermissionDecision, authorize_fail_closed

    created: list[Any] = []

    class Provider:
        @staticmethod
        async def _result(value: object) -> object:
            return value

        def readiness(self) -> Any:
            coroutine = self._result(True)
            created.append(coroutine)
            return coroutine

        def authorize(self, **_kwargs: object) -> Any:
            coroutine = self._result(PermissionDecision.ALLOW)
            created.append(coroutine)
            return coroutine

    active_phase = ""

    class PhaseProvider(Provider):
        def readiness(self) -> Any:
            nonlocal active_phase
            active_phase = "readiness"
            return super().readiness()

        def authorize(self, **kwargs: object) -> Any:
            nonlocal active_phase
            active_phase = "authorize"
            return super().authorize(**kwargs)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        failure_injected = False

        def should_fail() -> bool:
            nonlocal failure_injected
            if active_phase != phase or failure_injected:
                return False
            failure_injected = True
            return True

        if failure_point == "wait":
            original = authorization.asyncio.wait

            async def injected_wait_failure(*args: Any, **kwargs: Any) -> Any:
                if should_fail():
                    raise RuntimeError("injected wait failure")
                return await original(*args, **kwargs)

            setattr(authorization.asyncio, "wait", injected_wait_failure)
            restore = lambda: setattr(authorization.asyncio, "wait", original)
        elif failure_point == "task":
            original = authorization.asyncio.Task

            def injected_task_failure(*args: Any, **kwargs: Any) -> Any:
                if should_fail():
                    raise RuntimeError("injected Task failure")
                return original(*args, **kwargs)

            setattr(authorization.asyncio, "Task", injected_task_failure)
            restore = lambda: setattr(authorization.asyncio, "Task", original)
        else:
            original = getattr(loop, failure_point)

            def injected_loop_failure(*args: Any, **kwargs: Any) -> Any:
                if should_fail():
                    raise RuntimeError(f"injected {failure_point} failure")
                return original(*args, **kwargs)

            setattr(loop, failure_point, injected_loop_failure)
            restore = lambda: setattr(loop, failure_point, original)

        try:
            decision = await authorize_fail_closed(
                cast(Any, PhaseProvider()),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
            )
            await asyncio.sleep(0)
        finally:
            restore()

        assert decision is PermissionDecision.DENY
        assert failure_injected
        assert len(created) == (1 if phase == "readiness" else 2)
        assert all(coroutine.cr_frame is None for coroutine in created)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(scenario())
        created.clear()
        gc.collect()


def test_zero_deadline_tracks_suppress_then_finish_until_registry_drains() -> None:
    from gateway.authorization import (
        PermissionDecision,
        _quarantined_task_count,
        authorize_fail_closed,
    )

    async def scenario() -> None:
        unhandled: list[dict[str, object]] = []
        suppressed = asyncio.Event()
        finished = asyncio.Event()

        class Provider:
            async def readiness(self) -> bool:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    suppressed.set()
                    await asyncio.sleep(0)
                    finished.set()
                    return True
                raise AssertionError("readiness should only finish after cancellation")

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                raise AssertionError("late readiness ALLOW must be ignored")

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
                timeout_seconds=0,
            )
            assert decision is PermissionDecision.DENY
            await suppressed.wait()
            assert _quarantined_task_count() == 1
            assert len(asyncio.all_tasks() - tasks_before) == 1
            await finished.wait()
            for _ in range(3):
                await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert _quarantined_task_count() == 0
        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_cancel_failure_denies_keeps_task_owned_and_drains_cleanly() -> None:
    import gateway.authorization as authorization

    from gateway.authorization import (
        PermissionDecision,
        _quarantined_task_count,
        authorize_fail_closed,
    )

    class CancelFailure(BaseException):
        pass

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        unhandled: list[dict[str, object]] = []
        cancel_calls = 0

        class OneShotCancelFailureTask(asyncio.Task[Any]):
            def cancel(self, msg: object = None) -> bool:
                nonlocal cancel_calls
                cancel_calls += 1
                if cancel_calls == 1:
                    raise CancelFailure("injected one-shot Task.cancel failure")
                return super().cancel(msg)

        class Provider:
            async def readiness(self) -> bool:
                started.set()
                await release.wait()
                return True

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                raise AssertionError("late readiness must not authorize")

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        original_task_type = authorization.asyncio.Task
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        setattr(authorization.asyncio, "Task", OneShotCancelFailureTask)
        tasks_before = asyncio.all_tasks()
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=0.001,
            )
            assert decision is PermissionDecision.DENY
            assert started.is_set()
            assert cancel_calls == 1
            assert _quarantined_task_count() == 1
            assert len(asyncio.all_tasks() - tasks_before) == 1

            release.set()
            for _ in range(3):
                await asyncio.sleep(0)
        finally:
            setattr(authorization.asyncio, "Task", original_task_type)
            loop.set_exception_handler(previous_handler)

        assert _quarantined_task_count() == 0
        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_callback_registration_failure_denies_reaps_and_drains_cleanly() -> None:
    import gateway.authorization as authorization

    from gateway.authorization import (
        PermissionDecision,
        _quarantined_task_count,
        _reaper_task_count,
        authorize_fail_closed,
    )

    class CallbackRegistrationFailure(BaseException):
        pass

    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        unhandled: list[dict[str, object]] = []
        callback_calls = 0
        registration_failures = 0
        cancel_calls = 0

        class OneShotCallbackFailureTask(asyncio.Task[Any]):
            def add_done_callback(self, callback: Any, *, context: Any = None) -> None:
                nonlocal callback_calls, registration_failures
                callback_calls += 1
                if (
                    callback is authorization._consume_quarantined_task
                    and registration_failures == 0
                ):
                    registration_failures += 1
                    raise CallbackRegistrationFailure(
                        "injected one-shot add_done_callback failure"
                    )
                super().add_done_callback(callback, context=context)

            def cancel(self, msg: object = None) -> bool:
                nonlocal cancel_calls
                cancel_calls += 1
                return super().cancel(msg)

        class Provider:
            async def readiness(self) -> bool:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    finished.set()
                    raise RuntimeError("eventual provider failure")
                raise AssertionError("readiness should only finish after cancellation")

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                raise AssertionError("late readiness must not authorize")

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        original_task_type = authorization.asyncio.Task
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        setattr(authorization.asyncio, "Task", OneShotCallbackFailureTask)
        tasks_before = asyncio.all_tasks()
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=0.001,
            )
            assert decision is PermissionDecision.DENY
            assert started.is_set()
            await cancelled.wait()
            assert registration_failures == 1
            assert callback_calls >= 3
            assert cancel_calls == 1
            assert _quarantined_task_count() == 1
            assert _reaper_task_count() == 1
            assert len(asyncio.all_tasks() - tasks_before) == 2

            release.set()
            await finished.wait()
            for _ in range(3):
                await asyncio.sleep(0)
        finally:
            setattr(authorization.asyncio, "Task", original_task_type)
            loop.set_exception_handler(previous_handler)

        assert _quarantined_task_count() == 0
        assert _reaper_task_count() == 0
        assert asyncio.all_tasks() == tasks_before
        assert unhandled == []

    asyncio.run(scenario())


def test_callback_and_reaper_failures_await_owned_operation_before_deny() -> None:
    import gateway.authorization as authorization

    from gateway.authorization import (
        PermissionDecision,
        _quarantined_task_count,
        _reaper_task_count,
        authorize_fail_closed,
    )

    class CallbackRegistrationFailure(BaseException):
        pass

    class ReaperConstructionFailure(BaseException):
        pass

    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        finished = asyncio.Event()
        unhandled: list[dict[str, object]] = []
        created: list[asyncio.Task[Any]] = []
        consumed: list[asyncio.Task[Any]] = []
        registration_failures = 0
        reaper_failures = 0

        class CallbackFailureTask(asyncio.Task[Any]):
            def __init__(self, coroutine: Any, **kwargs: Any) -> None:
                super().__init__(coroutine, **kwargs)
                created.append(self)

            def add_done_callback(self, callback: Any, *, context: Any = None) -> None:
                nonlocal registration_failures
                if (
                    callback is authorization._consume_quarantined_task
                    and registration_failures == 0
                ):
                    registration_failures += 1
                    raise CallbackRegistrationFailure(
                        "injected add_done_callback failure"
                    )
                super().add_done_callback(callback, context=context)

        def failing_reaper_task(*_args: object, **_kwargs: object) -> Any:
            nonlocal reaper_failures
            reaper_failures += 1
            raise ReaperConstructionFailure("injected reaper Task failure")

        original_consumer = authorization._consume_authorization_task_result

        def recording_consumer(task: asyncio.Task[Any]) -> None:
            consumed.append(task)
            original_consumer(task)

        class Provider:
            async def readiness(self) -> bool:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await asyncio.sleep(0)
                    finished.set()
                    raise RuntimeError("eventual provider failure")
                raise AssertionError("readiness should only finish after cancellation")

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                raise AssertionError("late readiness must not authorize")

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        original_task_type = authorization.asyncio.Task
        original_reaper_task = authorization._NATIVE_ASYNCIO_TASK
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        setattr(authorization.asyncio, "Task", CallbackFailureTask)
        setattr(authorization, "_NATIVE_ASYNCIO_TASK", failing_reaper_task)
        setattr(authorization, "_consume_authorization_task_result", recording_consumer)
        tasks_before = asyncio.all_tasks()
        try:
            decision = await authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=0.001,
            )
            assert decision is PermissionDecision.DENY
            assert started.is_set()
            assert cancelled.is_set()
            assert finished.is_set()
            assert registration_failures == 1
            assert reaper_failures == 1
            assert len(created) == 1
            assert created[0].done()
            assert consumed == [created[0]]
            assert _quarantined_task_count() == 0
            assert _reaper_task_count() == 0
            assert asyncio.all_tasks() == tasks_before
        finally:
            setattr(
                authorization,
                "_consume_authorization_task_result",
                original_consumer,
            )
            setattr(authorization, "_NATIVE_ASYNCIO_TASK", original_reaper_task)
            setattr(authorization.asyncio, "Task", original_task_type)
            loop.set_exception_handler(previous_handler)

        assert unhandled == []

    asyncio.run(scenario())


def test_provider_contract_excludes_indefinite_cancellation_suppression() -> None:
    from gateway.authorization import AuthorizationProvider, authorize_fail_closed

    contract = f"{AuthorizationProvider.__doc__}\n{authorize_fail_closed.__doc__}"

    assert "trusted first-party" in contract
    assert "current event loop" in contract
    assert "nonblocking" in contract
    assert "cancellation-cooperative" in contract
    assert "independently bounded" in contract
    assert "same interpreter" in contract
    assert "availability" in contract
    assert "indefinitely suppress" in contract
    assert "last-resort" in contract
    assert "awaiting the exact owned operation" in contract
    assert "current authorization task" in contract
    assert "selected provider" in contract


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
    from gateway.authorization import (
        PermissionDecision,
        _quarantined_task_count,
        authorize_fail_closed,
    )

    async def scenario() -> None:
        started = asyncio.Event()
        suppressed = asyncio.Event()
        release = asyncio.Event()
        unhandled: list[dict[str, object]] = []

        class Provider:
            async def readiness(self) -> bool:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    suppressed.set()
                    await release.wait()
                    return True
                raise AssertionError("readiness should only finish after cancellation")

            async def authorize(self, **_kwargs: object) -> PermissionDecision:
                return PermissionDecision.ALLOW

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        task = asyncio.Task(
            authorize_fail_closed(
                Provider(),
                principal=_issuer().issue_user(
                    principal_id="telegram:123", authz_version=7
                ),
                permission="agent.use",
                known_permissions={"agent.use"},
                timeout_seconds=60,
            ),
            loop=loop,
        )
        try:
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await suppressed.wait()
            assert _quarantined_task_count() == 1
            release.set()
            for _ in range(3):
                await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert _quarantined_task_count() == 0
        assert unhandled == []

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
