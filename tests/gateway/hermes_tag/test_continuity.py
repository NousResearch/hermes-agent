from __future__ import annotations

import pytest

from gateway.hermes_tag import (
    ContinuityEnvelope,
    ContinuityMode,
    ContinuityStore,
    IdentityConflict,
    ReplayDetected,
    StaleWriteError,
    SurfaceRef,
    arguments_digest,
)


def test_same_surface_returns_existing_binding(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    first = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    second = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    assert first.continuity_id == second.continuity_id


def test_isolated_mode_separates_surfaces(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    other = SurfaceRef("slack", "default", "T_WORKSPACE", "C_OTHER")
    one = store.resolve_or_create(bound_principal, surface, mode=ContinuityMode.ISOLATED)
    two = store.resolve_or_create(bound_principal, other, mode=ContinuityMode.ISOLATED)
    assert one.continuity_id != two.continuity_id


def test_principal_mode_reuses_across_platforms(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    other = SurfaceRef("telegram", "default", "BOT1", "CHAT1")
    one = store.resolve_or_create(bound_principal, surface, mode=ContinuityMode.PRINCIPAL)
    two = store.resolve_or_create(bound_principal, other, mode=ContinuityMode.PRINCIPAL)
    assert one.continuity_id == two.continuity_id


def test_workspace_mode_reuses_inside_workspace(ledger, bound_principal):
    store = ContinuityStore(ledger)
    one = SurfaceRef("slack", "default", "T1", "C1")
    two = SurfaceRef("slack", "default", "T1", "C2")
    a = store.resolve_or_create(bound_principal, one, mode=ContinuityMode.WORKSPACE)
    b = store.resolve_or_create(bound_principal, two, mode=ContinuityMode.WORKSPACE)
    assert a.continuity_id == b.continuity_id


def test_workspace_mode_separates_workspaces(ledger, bound_principal):
    store = ContinuityStore(ledger)
    one = SurfaceRef("slack", "default", "T1", "C1")
    two = SurfaceRef("slack", "default", "T2", "C1")
    a = store.resolve_or_create(bound_principal, one, mode=ContinuityMode.WORKSPACE)
    b = store.resolve_or_create(bound_principal, two, mode=ContinuityMode.WORKSPACE)
    assert a.continuity_id != b.continuity_id


def test_project_mode_reuses_across_platforms(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    other = SurfaceRef("discord", "default", "G1", "C1")
    a = store.resolve_or_create(
        bound_principal,
        surface,
        mode=ContinuityMode.PROJECT,
        project_id="hermes-agent",
    )
    b = store.resolve_or_create(
        bound_principal,
        other,
        mode=ContinuityMode.PROJECT,
        project_id="hermes-agent",
    )
    assert a.continuity_id == b.continuity_id


def test_project_mode_requires_project_id(ledger, bound_principal, surface):
    with pytest.raises(ValueError, match="project"):
        ContinuityStore(ledger).resolve_or_create(
            bound_principal, surface, mode=ContinuityMode.PROJECT
        )


def test_explicit_mode_requires_existing_id(ledger, bound_principal, surface):
    with pytest.raises(ValueError, match="explicit"):
        ContinuityStore(ledger).resolve_or_create(
            bound_principal, surface, mode=ContinuityMode.EXPLICIT
        )


def test_explicit_continuity_cannot_cross_principal(ledger, identity_store, surface):
    store = ContinuityStore(ledger)
    owner = identity_store.create_principal("Owner")
    intruder = identity_store.create_principal("Intruder")
    continuity = store.create(owner, mode=ContinuityMode.PRINCIPAL)
    with pytest.raises(IdentityConflict, match="another principal"):
        store.resolve_or_create(
            intruder,
            surface,
            mode=ContinuityMode.EXPLICIT,
            explicit_id=continuity.continuity_id,
        )


def test_surface_rebind_cannot_cross_principal(
    ledger, identity_store, surface
):
    store = ContinuityStore(ledger)
    one = identity_store.create_principal("One")
    two = identity_store.create_principal("Two")
    a = store.create(one, mode=ContinuityMode.PRINCIPAL)
    b = store.create(two, mode=ContinuityMode.PRINCIPAL)
    store.bind_surface(surface, a.continuity_id, principal_id=one.principal_id)
    with pytest.raises(IdentityConflict):
        store.bind_surface(
            surface,
            b.continuity_id,
            principal_id=two.principal_id,
            allow_rebind=True,
        )


def test_checkpoint_uses_optimistic_version(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    continuity = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    updated = store.update_checkpoint(
        continuity.continuity_id,
        expected_version=1,
        payload={"objective": "finish Slack"},
        objective="Finish Slack",
    )
    assert updated.version == 2
    assert updated.objective == "Finish Slack"
    with pytest.raises(StaleWriteError, match="expected"):
        store.update_checkpoint(
            continuity.continuity_id,
            expected_version=1,
            payload={"stale": True},
        )


def test_envelope_replay_is_rejected(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    continuity = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    envelope = ContinuityEnvelope(
        event_id="event_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        continuity_id=continuity.continuity_id,
        origin=surface,
        payload_digest=arguments_digest({"x": 1}),
    )
    store.accept_envelope(envelope, max_hops=4)
    with pytest.raises(ReplayDetected, match="already"):
        store.accept_envelope(envelope, max_hops=4)


def test_envelope_origin_must_be_bound_to_the_continuity(
    ledger, bound_principal, surface
):
    store = ContinuityStore(ledger)
    continuity = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    unbound_origin = SurfaceRef(
        surface.platform,
        surface.profile,
        surface.scope_id,
        "C_UNBOUND",
    )
    envelope = ContinuityEnvelope(
        event_id="event_dddddddddddddddddddddddddddddddd",
        continuity_id=continuity.continuity_id,
        origin=unbound_origin,
        payload_digest=arguments_digest({"x": 1}),
    )
    with pytest.raises(IdentityConflict, match="origin"):
        store.accept_envelope(envelope, max_hops=4)


def test_envelope_hop_limit_is_enforced(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    continuity = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    envelope = ContinuityEnvelope(
        event_id="event_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        continuity_id=continuity.continuity_id,
        origin=surface,
        payload_digest=arguments_digest({"x": 1}),
        propagation_path=("discord", "telegram"),
        hop_count=2,
    )
    with pytest.raises(ReplayDetected, match="hop limit"):
        store.accept_envelope(envelope, max_hops=1)


def test_envelope_cycle_to_origin_is_rejected(ledger, bound_principal, surface):
    store = ContinuityStore(ledger)
    continuity = store.resolve_or_create(
        bound_principal, surface, mode=ContinuityMode.ISOLATED
    )
    envelope = ContinuityEnvelope(
        event_id="event_cccccccccccccccccccccccccccccccc",
        continuity_id=continuity.continuity_id,
        origin=surface,
        payload_digest=arguments_digest({"x": 1}),
        propagation_path=("slack",),
        hop_count=1,
    )
    with pytest.raises(ReplayDetected, match="cycle"):
        store.accept_envelope(envelope, max_hops=4)
