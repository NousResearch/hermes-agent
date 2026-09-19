from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from gateway.hermes_tag import (
    BudgetExceeded,
    BudgetLimits,
    ExternalIdentity,
    HermesTagLedger,
    IdentityConflict,
    ReceiptChainError,
    ReplayDetected,
    StorageError,
    UnknownIdentity,
)


def test_ledger_rejects_symlink_database_path(tmp_path: Path):
    target = tmp_path / "outside.db"
    target.write_bytes(b"")
    link = tmp_path / "linked.db"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    with pytest.raises(StorageError, match="symlink"):
        HermesTagLedger(link)


def test_ledger_state_files_are_owner_only_on_posix(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("POSIX mode bits do not apply on Windows")
    ledger = HermesTagLedger(tmp_path / "state" / "tag.db")
    ledger.initialize()
    ledger.append_receipt(
        event_id="event_permissions",
        kind="test.permissions",
        payload={"ok": True},
    )
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(ledger.path.parent.stat().st_mode) == 0o700
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{ledger.path}{suffix}")
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


def test_initialize_is_idempotent(tmp_path: Path):
    ledger = HermesTagLedger(tmp_path / "tag.db")
    ledger.initialize()
    ledger.initialize()
    connection = ledger.connection()
    try:
        assert connection.execute(
            "SELECT value FROM hermes_tag_meta WHERE key='schema_version'"
        ).fetchone()["value"] == "1"
    finally:
        connection.close()


def test_transaction_rolls_back_on_error(ledger):
    with pytest.raises(RuntimeError):
        with ledger.transaction() as connection:
            connection.execute(
                "INSERT INTO hermes_tag_principals VALUES (?, ?, ?, ?, ?)",
                ("principal_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "X", "[]", 0, "now"),
            )
            raise RuntimeError("abort")
    connection = ledger.connection()
    try:
        count = connection.execute("SELECT COUNT(*) FROM hermes_tag_principals").fetchone()[0]
    finally:
        connection.close()
    assert count == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"event_id": "", "kind": "test", "payload": {}}, "event_id"),
        ({"event_id": "x" * 513, "kind": "test", "payload": {}}, "event_id"),
        ({"event_id": "evt", "kind": "x" * 129, "payload": {}}, "kind"),
    ],
)
def test_receipt_identity_fields_are_bounded(ledger, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ledger.append_receipt(**kwargs)


def test_receipt_payload_is_bounded(ledger):
    with pytest.raises(ValueError, match="one megabyte"):
        ledger.append_receipt(
            event_id="evt-large",
            kind="test",
            payload={"value": "x" * 1_000_001},
        )


def test_receipt_append_is_idempotent_for_same_event(ledger):
    first = ledger.append_receipt(event_id="evt-1", kind="test", payload={"x": 1})
    second = ledger.append_receipt(event_id="evt-1", kind="test", payload={"x": 1})
    assert first.receipt_id == second.receipt_id
    assert ledger.verify_receipt_chain()[0] == 1


def test_receipt_event_collision_fails_closed(ledger):
    ledger.append_receipt(event_id="evt-1", kind="test", payload={"x": 1})
    with pytest.raises(ReceiptChainError, match="different"):
        ledger.append_receipt(event_id="evt-1", kind="test", payload={"x": 2})


def test_receipt_chain_detects_payload_tamper(ledger):
    ledger.append_receipt(event_id="evt-1", kind="test", payload={"x": 1})
    connection = sqlite3.connect(ledger.path)
    try:
        connection.execute(
            "UPDATE hermes_tag_receipts SET payload_json='{}' WHERE event_id='evt-1'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ReceiptChainError, match="hash mismatch"):
        ledger.verify_receipt_chain()


def test_replay_fingerprint_is_unique(ledger):
    ledger.register_replay_fingerprint(
        fingerprint="a" * 64,
        event_id="evt-1",
        continuity_id="continuity_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    with pytest.raises(ReplayDetected):
        ledger.register_replay_fingerprint(
            fingerprint="a" * 64,
            event_id="evt-2",
            continuity_id="continuity_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_budget_reserve_and_settle_refunds_unused(ledger):
    limits = BudgetLimits(hourly_tokens=100, daily_tokens=1000, hourly_cost_usd=1)
    reservation = ledger.reserve_budget(
        scope_digest="a" * 64,
        tokens=80,
        cost_usd=0.8,
        limits=limits,
    )
    assert ledger.budget_usage("a" * 64)["hour"]["tokens"] == 80
    settled = ledger.settle_budget(
        reservation.reservation_id,
        actual_tokens=30,
        actual_cost_usd=0.2,
    )
    assert settled.state == "settled"
    usage = ledger.budget_usage("a" * 64)["hour"]
    assert usage == {"tokens": 30, "cost_usd": 0.2}


def test_budget_release_refunds_all(ledger):
    reservation = ledger.reserve_budget(
        scope_digest="b" * 64,
        tokens=50,
        cost_usd=0.5,
        limits=BudgetLimits(hourly_tokens=100, hourly_cost_usd=1),
    )
    released = ledger.release_budget(reservation.reservation_id)
    assert released.state == "released"
    assert ledger.budget_usage("b" * 64)["hour"] == {
        "tokens": 0,
        "cost_usd": 0.0,
    }


def test_budget_rejects_actual_above_reservation(ledger):
    reservation = ledger.reserve_budget(
        scope_digest="c" * 64,
        tokens=10,
        cost_usd=0.1,
        limits=BudgetLimits(hourly_tokens=100),
    )
    with pytest.raises(BudgetExceeded, match="exceeds reserved"):
        ledger.settle_budget(
            reservation.reservation_id,
            actual_tokens=11,
            actual_cost_usd=0.1,
        )


def test_budget_limit_is_atomic_under_concurrency(ledger):
    limits = BudgetLimits(hourly_tokens=100)

    def reserve():
        try:
            ledger.reserve_budget(
                scope_digest="d" * 64,
                tokens=60,
                cost_usd=0,
                limits=limits,
            )
            return "allowed"
        except BudgetExceeded:
            return "denied"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: reserve(), range(2)))
    assert sorted(outcomes) == ["allowed", "denied"]
    assert ledger.budget_usage("d" * 64)["hour"]["tokens"] == 60


def test_unknown_budget_reservation_fails(ledger):
    with pytest.raises(StorageError):
        ledger.release_budget("reservation_missing")


def test_create_and_resolve_principal(identity_store, principal, external_identity):
    identity_store.bind_alias(external_identity, principal.principal_id)
    resolved = identity_store.resolve(external_identity)
    assert resolved.principal_id == principal.principal_id
    assert resolved.roles == ("admin", "operator")


def test_same_user_id_in_two_tenants_is_not_same_alias(identity_store, principal):
    one = ExternalIdentity("slack", "default", "T1", "U1")
    two = ExternalIdentity("slack", "default", "T2", "U1")
    identity_store.bind_alias(one, principal.principal_id)
    with pytest.raises(UnknownIdentity):
        identity_store.resolve(two)


def test_alias_rebind_requires_explicit_authority(identity_store, external_identity):
    a = identity_store.create_principal("A")
    b = identity_store.create_principal("B")
    identity_store.bind_alias(external_identity, a.principal_id)
    with pytest.raises(IdentityConflict):
        identity_store.bind_alias(external_identity, b.principal_id)
    assert identity_store.resolve(external_identity).principal_id == a.principal_id


def test_alias_rebind_preserves_new_authority(identity_store, external_identity):
    a = identity_store.create_principal("A")
    b = identity_store.create_principal("B")
    identity_store.bind_alias(external_identity, a.principal_id)
    identity_store.bind_alias(external_identity, b.principal_id, allow_rebind=True)
    assert identity_store.resolve(external_identity).principal_id == b.principal_id


def test_revoke_alias_removes_resolution(identity_store, principal, external_identity):
    identity_store.bind_alias(external_identity, principal.principal_id)
    identity_store.revoke_alias(external_identity)
    with pytest.raises(UnknownIdentity):
        identity_store.resolve(external_identity)


def test_resolve_or_guest_is_idempotent(identity_store, external_identity):
    first = identity_store.resolve_or_guest(external_identity, allow_guest=True)
    second = identity_store.resolve_or_guest(external_identity, allow_guest=True)
    assert first.principal_id == second.principal_id
    assert first.guest is True


def test_resolve_or_guest_can_fail_closed(identity_store, external_identity):
    with pytest.raises(UnknownIdentity):
        identity_store.resolve_or_guest(external_identity, allow_guest=False)


def test_ledger_parent_permissions_are_private_on_posix(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("POSIX mode assertion")
    ledger = HermesTagLedger(tmp_path / "private" / "tag.db")
    ledger.initialize()
    assert (ledger.path.parent.stat().st_mode & 0o777) == 0o700
