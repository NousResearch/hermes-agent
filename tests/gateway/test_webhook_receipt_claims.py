"""Receipt claims follow what actually happened: an uncertain send keeps its claim, a release is never
undone, a release the store could not write is landed later, and a retired adapter closes its store."""

import sqlite3

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.webhook import _WebhookDeliveryIdentity
from gateway.platforms.webhook_receipts import ACCEPTED, FAILED, WebhookReceipts

from tests.gateway.test_webhook_receipts import _gateway, _post


def _flaky_target(adapter, results):
    sent = []

    async def send(chat_id, content, **_kw):
        sent.append(content)
        return results.pop(0)

    adapter.gateway_runner.adapters[next(iter(adapter.gateway_runner.adapters))].send = send
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [
    SendResult(success=False, error="Timed out"),
    SendResult(success=False, error="chunk 2 failed", continuation_message_ids=("m1",)),
], ids=["timeout", "partial-split"])
async def test_an_uncertain_send_keeps_its_claim(uncertain):
    adapter = _gateway([])
    sent = _flaky_target(adapter, [uncertain, SendResult(success=True)])

    assert await _post(adapter, "d-unsure") == (502, "error")
    # The first attempt may already be in the chat: the provider's retry must not post it again.
    assert await _post(adapter, "d-unsure") == (200, "duplicate")
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unsent", [
    SendResult(success=False, error="httpx.ConnectError: [Errno 111] Connection refused", error_kind="transient"),
    SendResult(success=False, error="Timed out", retryable=True),
], ids=["connect-error", "connect-timeout"])
async def test_a_send_that_never_connected_releases_its_claim(unsent):
    adapter = _gateway([])
    sent = _flaky_target(adapter, [unsent, SendResult(success=True)])

    assert await _post(adapter, "d-unsent") == (502, "error")
    # Nothing reached the chat: once the connection recovers the provider's retry is delivered.
    assert await _post(adapter, "d-unsent") == (200, "delivered")
    assert len(sent) == 2


def _identity(delivery_id="d-1"):
    return _WebhookDeliveryIdentity.from_parts(None, "alerts", delivery_id)


def test_a_release_that_landed_first_is_never_undone(tmp_path):
    store = WebhookReceipts(tmp_path / "r.db")
    row = store.claim(_identity(), "push", 1.0)
    store.release(_identity(), "coalesced batch was not admitted")

    # The request's own (later) filing of its "coalesced" ack must not re-take the claim.
    store.settle(row, _identity(), ACCEPTED, "coalesced", "", keep_claim=True)

    assert store.claim(_identity(), "push", 2.0) is not None


def test_a_release_the_store_could_not_write_lands_on_the_next_claim(tmp_path):
    store = WebhookReceipts(tmp_path / "r.db")
    row = store.claim(_identity(), "push", 1.0)
    real_connect = store._connect

    def broken():
        raise sqlite3.OperationalError("database is locked")

    store._connect = broken
    with pytest.raises(sqlite3.Error):
        store.settle(row, _identity(), FAILED, "target rejected", "", keep_claim=False)
    store._connect = real_connect

    # Nothing ran: once the store is writable again the provider's retry is processed.
    assert store.claim(_identity(), "push", 2.0) is not None


@pytest.mark.asyncio
async def test_disconnect_closes_the_receipts_store():
    adapter = _gateway([])
    assert await _post(adapter, "d-close") == (200, "delivered")
    assert adapter._receipts._conn is not None

    await adapter.disconnect()

    assert adapter._receipts._conn is None
