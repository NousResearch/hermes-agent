"""Behavior contracts for the local specialist capability registry."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest


def _registry_api():
    try:
        from gateway.capability_registry import CapabilityRegistry, CapabilitySignature
    except ImportError as exc:  # RED: the clean replacement starts without this capability.
        pytest.fail(f"specialist capability registry is unavailable: {exc}")
    return CapabilityRegistry, CapabilitySignature


def _kanban_db():
    from hermes_cli import kanban_db

    return kanban_db


def _repository_read_signature():
    _, CapabilitySignature = _registry_api()
    return CapabilitySignature(
        domain="repository-evidence",
        actions=("read", "review"),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read",),
    )


def test_configured_profile_resolves_only_its_exact_active_scope(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(
        db_path=tmp_path / "registry.db",
        configured_profiles={"repository-reviewer": signature},
    )

    registry.register_configured_profile("repository-reviewer")

    resolution = registry.resolve(signature)
    assert resolution.status == "active_match"
    assert resolution.profile == "repository-reviewer"


def test_unconfigured_profile_cannot_use_direct_activation_api(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(db_path=tmp_path / "registry.db")

    with pytest.raises(ValueError, match="configured"):
        registry.register_configured_profile("undeclared")
    with pytest.raises(ValueError, match="direct arbitrary"):
        registry.add_active(profile_id="undeclared", signature=signature)

    assert registry.resolve(signature).status == "no_match"


def test_expired_profile_does_not_resolve(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(
        db_path=tmp_path / "registry.db",
        configured_profiles={"repository-reviewer": signature},
    )
    registry.register_configured_profile(
        "repository-reviewer",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    assert registry.resolve(signature).status == "no_match"


def test_multiple_exact_profiles_fail_closed_as_ambiguous(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(
        db_path=tmp_path / "registry.db",
        configured_profiles={"reviewer-one": signature, "reviewer-two": signature},
    )
    registry.register_configured_profile("reviewer-one")
    registry.register_configured_profile("reviewer-two")

    resolution = registry.resolve(signature)
    assert resolution.status == "ambiguous"
    assert resolution.profile is None


def test_expired_configured_profile_can_append_one_unambiguous_renewal(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(
        db_path=tmp_path / "registry.db",
        configured_profiles={"repository-reviewer": signature},
    )
    registry.register_configured_profile(
        "repository-reviewer",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    registry.register_configured_profile(
        "repository-reviewer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    resolution = registry.resolve(signature)
    assert resolution.status == "active_match"
    assert resolution.profile == "repository-reviewer"


def test_registry_rows_are_append_only_and_revocation_hides_profile(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    registry = CapabilityRegistry(
        db_path=tmp_path / "registry.db",
        configured_profiles={"repository-reviewer": signature},
    )
    declaration_id = registry.register_configured_profile("repository-reviewer")

    with _kanban_db().connect_closing(tmp_path / "registry.db") as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE capability_profiles SET status = 'inactive' "
                "WHERE profile_id = 'repository-reviewer'"
            )

    receipt = registry.revoke(
        declaration_id=declaration_id,
        profile_id="repository-reviewer",
        signature=signature,
        reason_code="operator_revoked",
    )

    assert len(receipt) == 64
    assert registry.resolve(signature).status == "no_match"


def test_trusted_reregistration_after_revocation_creates_one_new_generation(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    db_path = tmp_path / "registry.db"
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": signature},
    )
    first_declaration_id = registry.register_configured_profile("repository-reviewer")
    registry.revoke(
        declaration_id=first_declaration_id,
        profile_id="repository-reviewer",
        signature=signature,
        reason_code="operator_revoked",
    )

    renewed_declaration_id = registry.register_configured_profile("repository-reviewer")
    duplicate_declaration_id = registry.register_configured_profile("repository-reviewer")

    resolution = registry.resolve(signature)
    assert resolution.status == "active_match"
    assert resolution.profile == "repository-reviewer"
    assert renewed_declaration_id != first_declaration_id
    assert duplicate_declaration_id == renewed_declaration_id
    with _kanban_db().connect_closing(db_path) as conn:
        profile_rows = conn.execute(
            "SELECT COUNT(*) FROM capability_profiles WHERE profile_id = ?",
            ("repository-reviewer",),
        ).fetchone()[0]
        revocation_rows = conn.execute(
            "SELECT COUNT(*) FROM specialist_profile_revocations WHERE profile_id = ?",
            ("repository-reviewer",),
        ).fetchone()[0]
    assert profile_rows == 2
    assert revocation_rows == 1


def test_permission_variant_registration_is_not_hidden_by_revoked_version(tmp_path):
    CapabilityRegistry, CapabilitySignature = _registry_api()
    original = _repository_read_signature()
    expanded = CapabilitySignature(
        domain=original.domain,
        actions=original.actions,
        evidence_class=original.evidence_class,
        requested_permissions=(
            "repository-evidence:metadata",
            "repository-evidence:read",
        ),
    )
    db_path = tmp_path / "registry.db"
    original_registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": original},
    )
    original_declaration_id = original_registry.register_configured_profile(
        "repository-reviewer"
    )
    original_registry.revoke(
        declaration_id=original_declaration_id,
        profile_id="repository-reviewer",
        signature=original,
        reason_code="permission_version_revoked",
    )

    renewed_registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": expanded},
    )
    renewed_declaration_id = renewed_registry.register_configured_profile(
        "repository-reviewer"
    )

    assert renewed_registry.resolve(original).status == "no_match"
    resolution = renewed_registry.resolve(expanded)
    assert resolution.status == "active_match"
    assert resolution.profile == "repository-reviewer"
    assert renewed_declaration_id != original_declaration_id


def test_each_revocation_binds_the_latest_unrevoked_generation(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    signature = _repository_read_signature()
    db_path = tmp_path / "registry.db"
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": signature},
    )
    first_declaration_id = registry.register_configured_profile("repository-reviewer")
    first_receipt = registry.revoke(
        declaration_id=first_declaration_id,
        profile_id="repository-reviewer",
        signature=signature,
        reason_code="operator_revoked",
    )
    second_declaration_id = registry.register_configured_profile("repository-reviewer")

    second_receipt = registry.revoke(
        declaration_id=second_declaration_id,
        profile_id="repository-reviewer",
        signature=signature,
        reason_code="operator_revoked",
    )

    assert second_receipt != first_receipt
    assert registry.resolve(signature).status == "no_match"
    with _kanban_db().connect_closing(db_path) as conn:
        declaration_ids = [
            row[0]
            for row in conn.execute(
                "SELECT capability_profile_id FROM specialist_profile_revocations "
                "ORDER BY capability_profile_id"
            ).fetchall()
        ]
    assert len(declaration_ids) == 2
    assert len(set(declaration_ids)) == 2
