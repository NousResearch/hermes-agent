from __future__ import annotations

from pathlib import Path

import pytest

from gateway.hermes_tag import (
    ContinuityMode,
    ContinuityStore,
    ExternalIdentity,
    HermesTagLedger,
    IdentityStore,
    SurfaceRef,
    scope_from_surface,
)


@pytest.fixture
def ledger(tmp_path: Path) -> HermesTagLedger:
    value = HermesTagLedger(tmp_path / "hermes-tag.db")
    value.initialize()
    return value


@pytest.fixture
def identity_store(ledger: HermesTagLedger) -> IdentityStore:
    return IdentityStore(ledger)


@pytest.fixture
def principal(identity_store: IdentityStore):
    return identity_store.create_principal("Axl", roles=("admin", "operator"))


@pytest.fixture
def external_identity():
    return ExternalIdentity(
        platform="slack",
        profile="default",
        scope_id="T_WORKSPACE",
        external_id="U_AXL",
        display_name="Axl",
    )


@pytest.fixture
def surface():
    return SurfaceRef(
        platform="slack",
        profile="default",
        scope_id="T_WORKSPACE",
        chat_id="C_ENGINEERING",
        thread_id="1712345.100",
    )


@pytest.fixture
def bound_principal(identity_store, principal, external_identity):
    identity_store.bind_alias(external_identity, principal.principal_id)
    return principal


@pytest.fixture
def continuity(ledger, bound_principal, surface):
    return ContinuityStore(ledger).resolve_or_create(
        bound_principal,
        surface,
        mode=ContinuityMode.WORKSPACE,
    )


@pytest.fixture
def scope(bound_principal, surface, continuity):
    return scope_from_surface(
        surface,
        principal_id=bound_principal.principal_id,
        continuity_id=continuity.continuity_id,
    )
