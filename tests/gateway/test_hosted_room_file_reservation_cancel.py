"""Cancellation drains its reservation without releasing another sender's claim."""

import asyncio
import sqlite3
import threading

import pytest

from gateway import hosted_room_file_delivery as delivery
from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway.test_hosted_room_messaging_files import NativeAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed_elsewhere", [False, True])
@pytest.mark.parametrize("pause_after_commit", [False, True])
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_reservation_cancellation_settles_only_owned_work(
    tmp_path, monkeypatch, claimed_elsewhere, pause_after_commit, cancel_count
):
    db = tmp_path / "state.db"
    adapter = NativeAdapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42")
    entered, release, committed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = delivery.reserve_delivery
    if claimed_elsewhere:
        assert original(db, "delivery", "file-scope") == "new"

    def held(*args):
        result = original(*args) if pause_after_commit else None
        entered.set()
        assert release.wait(5)
        result = result if pause_after_commit else original(*args)
        committed.set()
        return result

    monkeypatch.setattr(delivery, "reserve_delivery", held)
    loads = []

    def load(maximum):
        loads.append(maximum)
        return delivery.Document("untouched.txt", b"not sent")

    async def recheck():
        return None

    task = asyncio.create_task(
        delivery.deliver_document(
            db_path=db,
            key="delivery",
            scope="file-scope",
            adapter=adapter,
            source=source,
            load=load,
            recheck=recheck,
            metadata={},
            reply_to=None,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert await asyncio.to_thread(committed.wait, 3)
    finally:
        release.set()
        if not task.done():
            try:
                await asyncio.wait_for(task, 5)
            except asyncio.CancelledError:
                pass
    with sqlite3.connect(db) as conn:
        state = conn.execute(
            "SELECT state FROM hosted_room_file_deliveries WHERE delivery_key='delivery'"
        ).fetchone()[0]
    assert state == ("fetching" if claimed_elsewhere else "failed")
    assert original(db, "next-request", "file-scope") == (
        "busy" if claimed_elsewhere else "new"
    )
    assert loads == [] and adapter.documents == []
