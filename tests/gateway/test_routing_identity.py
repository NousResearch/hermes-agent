from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.routing_identity import (
    DEFAULT_PROFILE,
    RoutingIdentity,
    RoutingIdentityConflict,
    RoutingIdentityRejected,
    attach_identity_to_source,
    canonicalize_routing_identity,
    current_routing_identity,
    identity_from_source,
    persistence_payload_for_source,
    restore_identity_on_source,
    resolve_identity_for_runner_source,
    routing_identity_scope,
    session_namespace,
)


def test_default_keeps_legacy_main_namespace():
    identity = canonicalize_routing_identity()
    assert identity.runtime_profile == DEFAULT_PROFILE
    assert identity.key_namespace == "main"
    assert session_namespace(None) == "main"
    assert session_namespace("default") == "main"
    assert session_namespace("main") == "main"


def test_named_credential_owner_becomes_runtime_without_route():
    identity = canonicalize_routing_identity(
        credential_owner="career-ops",
        active_profile="default",
        served_profiles={"default", "career-ops"},
    )
    assert identity == RoutingIdentity("career-ops", "career-ops", "career-ops")


def test_shared_credential_route_separates_transport_and_runtime():
    identity = canonicalize_routing_identity(
        route_profile="finance",
        credential_owner="default",
        served_profiles={"default", "finance"},
    )
    assert identity.transport_profile == "default"
    assert identity.runtime_profile == "finance"
    assert identity.persistence_profile == "finance"
    assert identity.key_namespace == "finance"


def test_hard_reject_never_degrades_to_default():
    with pytest.raises(RoutingIdentityRejected, match="explicitly rejected"):
        canonicalize_routing_identity(route_rejected=True)


def test_conflicting_route_and_source_reject():
    with pytest.raises(RoutingIdentityConflict, match="conflicts"):
        canonicalize_routing_identity(
            route_profile="finance",
            source_profile="career-ops",
        )


def test_unserved_transport_owner_rejects():
    with pytest.raises(RoutingIdentityRejected, match="transport profile"):
        canonicalize_routing_identity(
            route_profile="finance",
            credential_owner="orphan",
            served_profiles={"default", "finance"},
        )


def test_per_profile_physical_store_invariant_rejects_mismatch():
    with pytest.raises(RoutingIdentityConflict, match="persistence profile"):
        canonicalize_routing_identity(
            source_profile="finance",
            restored_persistence_profile="default",
        )


def test_source_attachment_keeps_transport_owner_wire_private():
    source = SimpleNamespace(profile=None)
    identity = RoutingIdentity("default", "finance", "finance")
    attach_identity_to_source(source, identity)
    assert source.profile == "finance"
    assert source._transport_profile == "default"
    assert source._persistence_profile == "finance"


def test_persistence_round_trip_restores_private_owner():
    live = SimpleNamespace(profile=None)
    identity = RoutingIdentity("default", "finance", "finance")
    attach_identity_to_source(live, identity)
    payload = persistence_payload_for_source(live)

    restored = SimpleNamespace(profile=None)
    restored_identity = restore_identity_on_source(restored, payload)
    assert restored_identity == identity
    assert identity_from_source(restored) == identity
    assert restored.profile == "finance"
    assert restored._transport_profile == "default"


def test_live_owner_mismatch_with_restored_owner_rejects():
    source = SimpleNamespace(
        profile="finance",
        _transport_profile="default",
        _persistence_profile="finance",
        profile_route_rejected=False,
    )
    with pytest.raises(RoutingIdentityConflict, match="live credential owner"):
        identity_from_source(source, credential_owner="career-ops")


@pytest.mark.asyncio
async def test_context_propagates_to_child_task_and_to_thread():
    identity = RoutingIdentity("default", "finance", "finance")

    async def child():
        task_value = current_routing_identity()
        thread_value = await asyncio.to_thread(current_routing_identity)
        return task_value, thread_value

    with routing_identity_scope(identity):
        task_value, thread_value = await asyncio.create_task(child())
        assert current_routing_identity() == identity

    assert task_value == identity
    assert thread_value == identity
    assert current_routing_identity() is None


def test_scope_restores_outer_identity_after_nested_scope():
    outer = RoutingIdentity("default", "default", "default")
    inner = RoutingIdentity("default", "finance", "finance")
    with routing_identity_scope(outer):
        assert current_routing_identity() == outer
        with routing_identity_scope(inner):
            assert current_routing_identity() == inner
        assert current_routing_identity() == outer
    assert current_routing_identity() is None


def test_dataclass_carrier_survives_replace_without_wire_serialization():
    from dataclasses import dataclass, field, replace

    @dataclass
    class Source:
        profile: str | None = None
        routing_identity: dict | None = field(default=None, repr=False, compare=False)

        def to_dict(self):
            return {"profile": self.profile}

    source = Source()
    identity = RoutingIdentity("default", "finance", "finance")
    attach_identity_to_source(source, identity)
    copied = replace(source)

    assert copied.routing_identity == identity.to_persistence_dict()
    assert identity_from_source(copied) == identity
    assert "routing_identity" not in copied.to_dict()


@pytest.mark.asyncio
async def test_entrypoint_resolves_route_before_handler_and_scopes_entire_turn(monkeypatch):
    import contextlib
    import sys
    import types

    from gateway.routing_identity import routing_identity_entrypoint

    entered = []

    @contextlib.contextmanager
    def fake_profile_scope(home):
        entered.append(("enter", home))
        try:
            yield
        finally:
            entered.append(("exit", home))

    fake_run = types.ModuleType("gateway.run")
    fake_run._profile_runtime_scope = fake_profile_scope
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    class Config:
        multiplex_profiles = True

    class Source:
        profile = None
        profile_route_rejected = False
        _transport_profile = "default"
        platform = "telegram"

    class Event:
        source = Source()

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {"finance": {}}

        def _profile_name_for_source(self, source):
            assert source.profile is None
            return "finance"

        def _resolve_profile_home_for_source(self, source):
            assert source.profile == "finance"
            return "/profiles/finance"

        def _registered_transport_adapter(self, source):
            return None

    seen = {}

    @routing_identity_entrypoint
    async def handler(runner, event):
        seen["identity"] = current_routing_identity()
        seen["profile"] = event.source.profile
        return "ok"

    result = await handler(Runner(), Event())
    assert result == "ok"
    assert seen["identity"] == RoutingIdentity("default", "finance", "finance")
    assert seen["profile"] == "finance"
    assert entered == [
        ("enter", "/profiles/finance"),
        ("exit", "/profiles/finance"),
    ]
    assert current_routing_identity() is None


def test_runtime_only_source_is_not_treated_as_trusted_transport_identity():
    source = SimpleNamespace(profile="finance")
    assert persistence_payload_for_source(source) is None


def test_resolved_route_must_match_restored_runtime_identity():
    source = SimpleNamespace(profile=None)
    restore_identity_on_source(
        source, RoutingIdentity("default", "finance", "finance").to_persistence_dict()
    )
    with pytest.raises(RoutingIdentityConflict, match="resolved route"):
        identity_from_source(source, route_profile="career-ops")


@pytest.mark.asyncio
async def test_entrypoint_route_overrides_legacy_credential_owner_stamp(monkeypatch):
    import contextlib
    import sys
    import types

    from gateway.routing_identity import routing_identity_entrypoint

    @contextlib.contextmanager
    def fake_profile_scope(_home):
        yield

    fake_run = types.ModuleType("gateway.run")
    fake_run._profile_runtime_scope = fake_profile_scope
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    owner_adapter = object()

    class Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = ["finance", "career-ops"]

    class Source:
        # Legacy _make_profile_message_handler stamped the credential owner.
        profile = "career-ops"
        profile_route_rejected = False
        platform = "telegram"

    class Event:
        source = Source()

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {"career-ops": {"telegram": owner_adapter}, "finance": {}}

        def _profile_name_for_source(self, _source):
            return "finance"

        def _resolve_profile_home_for_source(self, source):
            return f"/profiles/{source.profile}"

        def _registered_transport_adapter(self, _source):
            return owner_adapter

    seen = {}

    @routing_identity_entrypoint
    async def handler(_runner, event):
        seen["identity"] = current_routing_identity()
        seen["profile"] = event.source.profile
        return "ok"

    assert await handler(Runner(), Event()) == "ok"
    assert seen["identity"] == RoutingIdentity(
        "career-ops", "finance", "finance"
    )
    assert seen["profile"] == "finance"


@pytest.mark.asyncio
async def test_entrypoint_drops_restored_identity_for_unserved_runtime(monkeypatch):
    import contextlib
    import sys
    import types

    from gateway.routing_identity import routing_identity_entrypoint

    @contextlib.contextmanager
    def fake_profile_scope(_home):
        yield

    fake_run = types.ModuleType("gateway.run")
    fake_run._profile_runtime_scope = fake_profile_scope
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    class Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = []

    source = SimpleNamespace(
        profile="retired",
        profile_route_rejected=False,
        platform="telegram",
        routing_identity=RoutingIdentity(
            "default", "retired", "retired"
        ).to_persistence_dict(),
    )
    event = SimpleNamespace(source=source)

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {}

        def _profile_name_for_source(self, _source):
            return None

        def _registered_transport_adapter(self, _source):
            return None

        def _resolve_profile_home_for_source(self, _source):
            raise AssertionError("unserved identity must be rejected first")

    called = False

    @routing_identity_entrypoint
    async def handler(_runner, _event):
        nonlocal called
        called = True

    assert await handler(Runner(), event) is None
    assert called is False
    assert source.profile_route_rejected is True


@pytest.mark.asyncio
async def test_adapter_entrypoint_scopes_pre_ingress_keying_and_child_task(monkeypatch):
    import contextlib
    import sys
    import types

    from gateway.routing_identity import routing_identity_adapter_entrypoint

    entered = []

    @contextlib.contextmanager
    def fake_profile_scope(home):
        entered.append(("enter", home))
        try:
            yield
        finally:
            entered.append(("exit", home))

    fake_run = types.ModuleType("gateway.run")
    fake_run._profile_runtime_scope = fake_profile_scope
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    class Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = ["finance"]

    class Source:
        profile = None
        profile_route_rejected = False
        platform = "telegram"

    source = Source()
    event = SimpleNamespace(source=source)

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {"finance": {}}

        def _profile_name_for_source(self, _source):
            return "finance"

        def _resolve_profile_home_for_source(self, routed_source):
            assert routed_source.profile == "finance"
            return "/profiles/finance"

    runner = Runner()

    class Adapter:
        gateway_runner = runner
        platform = "telegram"

    adapter = Adapter()
    runner.adapters = {"telegram": adapter}
    seen = {}

    @routing_identity_adapter_entrypoint
    async def adapter_handle(_adapter, routed_event):
        seen["entry"] = current_routing_identity()

        async def child():
            return current_routing_identity(), await asyncio.to_thread(
                current_routing_identity
            )

        seen["task"], seen["thread"] = await asyncio.create_task(child())
        seen["profile"] = routed_event.source.profile
        return "accepted"

    assert await adapter_handle(adapter, event) == "accepted"
    expected = RoutingIdentity("default", "finance", "finance")
    assert seen == {
        "entry": expected,
        "task": expected,
        "thread": expected,
        "profile": "finance",
    }
    assert entered == [
        ("enter", "/profiles/finance"),
        ("exit", "/profiles/finance"),
    ]
    assert current_routing_identity() is None


@pytest.mark.asyncio
async def test_adapter_entrypoint_drops_rejected_route_before_local_handler():
    from gateway.routing_identity import routing_identity_adapter_entrypoint

    source = SimpleNamespace(
        profile=None,
        profile_route_rejected=True,
        platform="telegram",
    )
    event = SimpleNamespace(source=source)

    class Config:
        multiplex_profiles = True

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {}

    runner = Runner()

    class Adapter:
        gateway_runner = runner
        platform = "telegram"

    called = False

    @routing_identity_adapter_entrypoint
    async def adapter_handle(_adapter, _event):
        nonlocal called
        called = True

    assert await adapter_handle(Adapter(), event) is None
    assert called is False
    assert source.profile_route_rejected is True


def test_multiplexed_runtime_only_source_fails_closed_without_transport_binding():
    source = SimpleNamespace(
        profile="finance",
        profile_route_rejected=False,
        platform="telegram",
    )

    class Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = ["finance"]

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {"finance": {}}

        def _profile_name_for_source(self, _source):
            return "finance"

        def _registered_transport_adapter(self, _source):
            return None

    with pytest.raises(RoutingIdentityRejected, match="trusted transport binding"):
        resolve_identity_for_runner_source(Runner(), source)


def test_fresh_callback_source_inherits_proven_turn_identity():
    identity = RoutingIdentity("default", "finance", "finance")
    source = SimpleNamespace(
        profile="finance",
        profile_route_rejected=False,
        platform="telegram",
    )

    class Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = ["finance"]

    class Runner:
        config = Config()
        _primary_profile_name = "default"
        adapters = {}
        _profile_adapters = {"finance": {}}

        def _profile_name_for_source(self, _source):
            return "finance"

        def _registered_transport_adapter(self, _source):
            return None

    with routing_identity_scope(identity):
        resolved = resolve_identity_for_runner_source(Runner(), source)

    assert resolved == identity
    assert identity_from_source(source) == identity
