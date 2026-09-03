"""Contract tests for the local specialist capability registry."""

from __future__ import annotations

import sqlite3
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature
from hermes_cli import kanban_db as kb


MARKET_DATA = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)


@pytest.fixture
def registry(tmp_path):
    return CapabilityRegistry(db_path=tmp_path / "capabilities.db")


def test_resolve_returns_only_unexpired_exact_scope_active_profile(registry):
    registry.register_fixed_baseline(profile_id="market-data-authority-auditor", signature=MARKET_DATA)

    resolution = registry.resolve(MARKET_DATA)

    assert resolution.status == "active_match"
    assert resolution.profile == "market-data-authority-auditor"


def test_resolve_rejects_expired_or_permission_expanding_profile(registry):
    registry.register_fixed_baseline(
        profile_id="market-data-authority-auditor",
        signature=MARKET_DATA,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="direct arbitrary"):
        registry.add_active(
            profile_id="permission-expanding",
            signature=CapabilitySignature(
                domain=MARKET_DATA.domain,
                actions=MARKET_DATA.actions,
                evidence_class=MARKET_DATA.evidence_class,
                requested_permissions=("market-data:read", "market-data:write"),
            ),
        )

    resolution = registry.resolve(MARKET_DATA)

    assert resolution.status == "no_match"
    assert resolution.profile is None


def test_resolve_returns_ambiguous_instead_of_choosing_between_profiles(registry):
    with kb.connect_closing(registry._db_path) as conn:
        for profile_id in ("one", "two"):
            conn.execute(
                """
                INSERT INTO capability_profiles (
                    profile_id, signature_hash, permissions_hash,
                    model_receipt_hash, verification_receipt_hash,
                    domain, actions_json, evidence_class, requested_permissions_json,
                    expires_at, status, created_at
                ) VALUES (?, ?, ?, '', '', ?, ?, ?, ?, NULL, 'active', ?)
                """,
                (
                    profile_id,
                    MARKET_DATA.signature_hash,
                    MARKET_DATA.permissions_hash,
                    MARKET_DATA.domain,
                    json.dumps(MARKET_DATA.actions),
                    MARKET_DATA.evidence_class,
                    json.dumps(MARKET_DATA.requested_permissions),
                    int(time.time()),
                ),
            )

    resolution = registry.resolve(MARKET_DATA)

    assert resolution.status == "ambiguous"
    assert resolution.profile is None


def test_migration_is_idempotent_and_creates_append_only_registry_table(tmp_path):
    db_path = tmp_path / "capabilities.db"

    kb.init_db(db_path)
    kb.init_db(db_path)

    with kb.connect_closing(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(capability_profiles)")}

    assert {
        "profile_id",
        "signature_hash",
        "permissions_hash",
        "model_receipt_hash",
        "verification_receipt_hash",
        "expires_at",
        "status",
    } <= columns


def test_public_direct_active_registration_cannot_bypass_promotion_gates(registry):
    with pytest.raises(ValueError, match="direct arbitrary"):
        registry.add_active(
            profile_id="generated-bypass",
            signature=MARKET_DATA,
            model_receipt_hash="a" * 64,
            verification_receipt_hash="b" * 64,
        )

    assert registry.resolve(MARKET_DATA).status == "no_match"


def test_capability_profiles_reject_direct_update_and_delete(registry):
    registry.register_fixed_baseline(profile_id="market-data-authority-auditor", signature=MARKET_DATA)

    with kb.connect_closing(registry._db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE capability_profiles SET status = 'inactive' WHERE profile_id = 'market-data-authority-auditor'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM capability_profiles WHERE profile_id = 'market-data-authority-auditor'")

    resolution = registry.resolve(MARKET_DATA)
    assert resolution.status == "active_match"
    assert resolution.profile == "market-data-authority-auditor"


def test_migration_restores_capability_profile_immutability_triggers(tmp_path):
    db_path = tmp_path / "capabilities.db"
    kb.init_db(db_path)

    with kb.connect_closing(db_path) as conn:
        conn.execute("DROP TRIGGER capability_profiles_no_update")
        conn.execute("DROP TRIGGER capability_profiles_no_delete")

    kb.init_db(db_path)

    with kb.connect_closing(db_path) as conn:
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'capability_profiles'"
            )
        }

    assert triggers == {"capability_profiles_no_delete", "capability_profiles_no_update"}


def test_resolve_rejects_malformed_persisted_expiry(registry):
    with kb.connect_closing(registry._db_path) as conn:
        conn.execute(
            """
            INSERT INTO capability_profiles (
                profile_id, signature_hash, permissions_hash,
                model_receipt_hash, verification_receipt_hash,
                domain, actions_json, evidence_class, requested_permissions_json,
                expires_at, status, created_at
            ) VALUES (?, ?, ?, '', '', ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                "malformed-expiry",
                MARKET_DATA.signature_hash,
                MARKET_DATA.permissions_hash,
                MARKET_DATA.domain,
                json.dumps(MARKET_DATA.actions),
                MARKET_DATA.evidence_class,
                json.dumps(MARKET_DATA.requested_permissions),
                "invalid-time",
                int(time.time()),
            ),
        )

    resolution = registry.resolve(MARKET_DATA)

    assert resolution.status == "no_match"
    assert resolution.profile is None


def test_hash_only_promotion_cannot_create_an_active_profile(registry):
    with pytest.raises(ValueError, match="hash-only promotion is disabled"):
        registry.add_active_from_promotion(
            profile_id="forged-promotion",
            signature=MARKET_DATA,
            benchmark_receipt_hash="a" * 64,
            verification_receipt_hash="b" * 64,
        )

    assert registry.resolve(MARKET_DATA).status == "no_match"


def test_durable_looking_generated_promotion_still_fails_without_authenticated_approval_authority(registry):
    with pytest.raises(ValueError, match="authenticated operator approval authority"):
        registry.add_active_from_durable_promotion(
            profile_id="generated-profile",
            signature=MARKET_DATA,
            candidate_id="cpr_1234567890abcdef12345678_12345678",
            promotion_proof_hash="a" * 64,
        )

    assert registry.resolve(MARKET_DATA).status == "no_match"
