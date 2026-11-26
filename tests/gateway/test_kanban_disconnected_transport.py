"""A reconnecting secondary credential must never borrow the shared bot (#101335)."""
import asyncio

from gateway.config import Platform
from tests.gateway.test_kanban_routed_transport import (
    RecordingAdapter, collect, completion, deliver, setup_runner, unseen,
)


def test_disconnected_secondary_keeps_events_until_its_own_adapter_recovers(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    runner._profile_failed_platforms = {"yuki": {Platform.DISCORD: {"adapter": RecordingAdapter()}}}
    task = completion()
    assert not collect(runner)
    assert unseen(task)
    assert not primary.sent
    secondary = RecordingAdapter()
    runner._profile_failed_platforms["yuki"].clear()
    runner._profile_adapters["yuki"][Platform.DISCORD] = secondary
    asyncio.run(deliver(runner, collect(runner)))
    assert len(secondary.sent) == 1
    assert not primary.sent
    assert not unseen(task)


def test_successful_delivery_logs_receipt_without_body(tmp_path, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    class ReceiptAdapter(RecordingAdapter):
        async def send(self, chat_id, text, **kwargs):
            await super().send(chat_id, text, **kwargs)
            return SimpleNamespace(success=True, message_id="receipt-42", continuation_message_ids=("receipt-43",))

    runner = setup_runner(tmp_path, monkeypatch)
    runner._profile_adapters["yuki"][Platform.DISCORD] = ReceiptAdapter()
    task = completion(mode="notify")
    with caplog.at_level(logging.INFO):
        asyncio.run(deliver(runner, collect(runner)))
    messages = [record.getMessage() for record in caplog.records if "kanban notifier: delivered" in record.getMessage()]
    assert len(messages) == 1
    assert task in messages[0]
    assert "message_id=receipt-42" in messages[0]
    assert "receipt-43" in messages[0]
    assert "finished" not in messages[0]


def test_disconnect_after_claim_rewinds_instead_of_using_primary(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"][Platform.DISCORD] = secondary
    task = completion()
    rows = collect(runner)
    assert len(rows) == 1
    runner._profile_adapters["yuki"].clear()
    runner._profile_failed_platforms = {"yuki": {Platform.DISCORD: {"adapter": secondary}}}
    asyncio.run(deliver(runner, rows))
    assert unseen(task)
    assert not runner.adapters[Platform.DISCORD].sent
    assert not secondary.sent
