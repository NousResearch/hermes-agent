"""Live PG18 / SQLite receipt-fence conformance for the standalone ledger adapter."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from gateway import delivery_ledger as sqlite
from gateway.delivery_ledger_postgresql import DeliveryLedgerPostgreSQLConfig, PostgreSQLDeliveryLedger
from gateway.delivery_ledger_adapter import open_configured_delivery_ledger
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
@pytest.fixture
def ledger(postgresql_delivery_target: OwnedPostgreSQLTestTarget):
    value = PostgreSQLDeliveryLedger(
        _DSN,
        schema=postgresql_delivery_target.schema,
        settings=DeliveryLedgerPostgreSQLConfig(lease_seconds=1),
    )
    try:
        yield value, postgresql_delivery_target
    finally:
        value.close()


def _record(ledger, oid="o1"):
    return ledger.record_obligation(obligation_id=oid, session_key="s", platform="slack", chat_id="c", thread_id=None, content="body", adapter_profile="metadata-not-tenant")


def test_fresh_catalog_is_dedicated_and_validated(ledger):
    ledger, target = ledger
    assert ledger.debug_rows() == []
    with target.connect() as con, con.cursor() as cur:
        cur.execute(f"SELECT version FROM {target.schema}.delivery_schema_migrations")
        assert cur.fetchall() == [(1,)]
    target.execute(f"DROP INDEX {target.schema}.delivery_claim_idx")
    with pytest.raises(Exception, match="schema drift"):
        PostgreSQLDeliveryLedger(_DSN, schema=target.schema)


def test_idempotent_record_fence_and_ack(ledger):
    ledger, _target = ledger
    first = _record(ledger)
    assert _record(ledger) == first
    claim = ledger.mark_attempting(first)
    assert claim and claim.fence == first.fence + 1
    assert ledger.mark_delivered(claim)
    assert not ledger.mark_failed(claim, "stale")
    assert ledger.debug_rows()[0]["state"] == "delivered"


def test_concurrent_claim_only_one_receipt_wins(ledger):
    ledger, _target = ledger
    receipt = _record(ledger); barrier = Barrier(2)
    def claim(): barrier.wait(); return ledger.mark_attempting(receipt)
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(lambda _: claim(), range(2)))
    assert sum(item is not None for item in results) == 1


def test_expired_lease_steal_rejects_stale_receipt_and_owner_guard(ledger):
    ledger, target = ledger
    short_lease = PostgreSQLDeliveryLedger(
        _DSN, schema=target.schema,
        settings=DeliveryLedgerPostgreSQLConfig(lease_seconds=.01),
    )
    original = _record(short_lease)
    first = short_lease.mark_attempting(original)
    assert first
    import time
    time.sleep(.02)
    thief = PostgreSQLDeliveryLedger(
        _DSN, schema=target.schema,
        settings=DeliveryLedgerPostgreSQLConfig(lease_seconds=1),
    )
    try:
        stolen = thief.sweep_recoverable()
        assert len(stolen) == 1
        assert not short_lease.mark_delivered(first)
        assert thief.mark_delivered(stolen[0]['receipt'])
    finally:
        thief.close()
        short_lease.close()


def test_transaction_rollback_preserves_no_partial_obligation(ledger, monkeypatch):
    ledger, _target = ledger
    original = ledger._validate
    monkeypatch.setattr(ledger, '_validate', lambda cursor: (_ for _ in ()).throw(RuntimeError('inject')))
    with pytest.raises(RuntimeError):
        ledger._migrate(ledger._connect())
    monkeypatch.setattr(ledger, '_validate', original)
    assert ledger.debug_rows() == []


def test_retention_prune_and_adapter_profile_is_not_schema(ledger):
    ledger, target = ledger
    claim=ledger.mark_attempting(_record(ledger)); assert claim and ledger.mark_delivered(claim)
    ledger.prune(); assert ledger.debug_rows()[0]['obligation_id']=='o1'
    assert ledger._schema == target.schema


@pytest.mark.asyncio
async def test_injected_pg_final_consumer_never_opens_state_db_or_sends_without_claim(
    postgresql_delivery_target: OwnedPostgreSQLTestTarget, monkeypatch, tmp_path,
):
    """This is intentionally the sole PG-routed gateway consumer, not gateway activation."""
    import sqlite3
    from unittest.mock import AsyncMock
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    class Adapter(BasePlatformAdapter):
        async def connect(self, *, is_reconnect=False): return True
        async def disconnect(self): return None
        async def get_chat_info(self, chat_id): return {}
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="m")

    config = {"state_store": {"backend": "postgresql", "postgresql": {
        "dsn_env": "OWNED_DELIVERY_DSN", "connect_timeout_seconds": 3, "pool_max_size": 2,
    }}}
    ledger = open_configured_delivery_ledger(
        config, secret_lookup=lambda name: postgresql_delivery_target.dsn,
        schema=postgresql_delivery_target.schema,
    )
    adapter = Adapter(PlatformConfig(enabled=True), Platform.SLACK)
    adapter.delivery_ledger = ledger
    adapter.send = AsyncMock(wraps=adapter.send)
    event = MessageEvent(text="ask", message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel"), message_id="m1")
    try:
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SQLite opened")))
        receipt = await adapter._record_delivery_obligation(event, "agent:main:slack:channel:C1", "answer", adapter, False)
        assert receipt is not None
        assert await adapter._finalize_delivery_obligation(receipt, SendResult(success=True), event, adapter) is None
        assert adapter.send.await_count == 0
        assert not (tmp_path / "state.db").exists()
        assert ledger.debug_rows()[0]["state"] == "delivered"
        class DeniedLedger:
            def record_obligation(self, **kwargs): return object()
            def mark_attempting(self, receipt): return None
            def mark_delivered(self, receipt): raise AssertionError("no ack without claim")
            def mark_failed(self, receipt, error=""): raise AssertionError("no failure without claim")
            def release_runtime_claim(self, receipt, error=""): return False
        adapter.delivery_ledger = DeniedLedger()
        adapter._send_with_retry = AsyncMock()
        denied, _ = await adapter.send_final_ledgered(event, "agent:main:slack:channel:C1", "denied", {}, reply_to=None)
        assert denied.error == "delivery_ledger_claim_denied"
        adapter._send_with_retry.assert_not_awaited()
    finally:
        ledger.close()
