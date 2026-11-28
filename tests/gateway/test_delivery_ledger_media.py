"""Media attachment rows in the delivery ledger (I-018 / #240).

Pins that ``media_path``/``media_kind`` survive record → sweep → redelivery, that
recovery markers ride as file captions (not empty text), and that a vanished file
fails closed without raising.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform
from persistence_assembly import profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.storage_handles import storage_handle


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_scope", lambda: profile_scope(home=home))
    yield


def _holder():
    return raw_holder(storage_handle(dl._scope().profile_dir))


def _execute(sql, params=()):
    _holder()._execute_write(lambda conn: conn.execute(sql, params))


def _orphan(oid):
    _execute(
        "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
        (oid,),
    )


def _row(oid):
    with _holder()._read_ctx() as conn:
        r = conn.execute(
            "SELECT state, last_error FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone()
    return None if r is None else {"state": r[0], "last_error": r[1]}


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    _store = MagicMock()
    _store.clear_resume_pending = AsyncMock()
    _store._store = None
    runner.session_store = None
    runner._async_session_store = _store
    return runner


def _media_adapter(success=True):
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=success, error=""))
    adapter.send_video = AsyncMock(return_value=MagicMock(success=success, error=""))
    adapter.send_document = AsyncMock(return_value=MagicMock(success=success, error=""))
    adapter.send_image_file = AsyncMock(return_value=MagicMock(success=success, error=""))
    adapter.send_voice = AsyncMock(return_value=MagicMock(success=success, error=""))
    return adapter


class TestMediaSweep:
    def test_sweep_claim_includes_media_fields(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"vid")
        dl.record_obligation(
            obligation_id="media-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="",
            media_path=str(video),
            media_kind="video",
        )
        _orphan("media-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        row = claimed[0]
        assert row["media_path"] == str(video)
        assert row["media_kind"] == "video"
        assert "content" in row and row["content"] == ""

    def test_legacy_text_row_has_no_media_keys(self):
        dl.record_obligation(
            obligation_id="text-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="plain reply",
        )
        _orphan("text-1")
        claimed = dl.sweep_recoverable()
        assert claimed and "media_path" not in claimed[0] and "media_kind" not in claimed[0]


class TestMediaRedelivery:
    @pytest.mark.asyncio
    async def test_video_redelivery_uses_send_video_and_caption_marker(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"vid")
        dl.record_obligation(
            obligation_id="v-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            content="",
            media_path=str(video),
            media_kind="video",
        )
        dl.mark_failed("v-1", "send_path_degraded")
        _orphan("v-1")
        adapter = _media_adapter()
        runner = _runner(adapter)

        claimed = await runner._claim_pending_obligations()
        n = await runner._redeliver_claimed_obligations(claimed)

        assert n == 1
        adapter.send.assert_not_awaited()
        adapter.send_video.assert_awaited_once()
        kw = adapter.send_video.await_args.kwargs
        assert kw["video_path"] == str(video)
        assert kw["caption"] == dl.RECOVERED_MARKER
        assert kw["metadata"] == {"thread_id": "171.001"}
        assert _row("v-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_text_redelivery_unchanged(self):
        dl.record_obligation(
            obligation_id="t-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="the final answer",
        )
        _orphan("t-1")
        adapter = _media_adapter()
        runner = _runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        adapter.send_video.assert_not_awaited()
        assert adapter.send.await_args.kwargs["content"] == "the final answer"

    @pytest.mark.asyncio
    async def test_missing_media_file_marks_failed_without_raising(self, tmp_path):
        gone = tmp_path / "missing.mp4"
        dl.record_obligation(
            obligation_id="gone-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="",
            media_path=str(gone),
            media_kind="video",
        )
        _orphan("gone-1")
        adapter = _media_adapter()
        runner = _runner(adapter)

        claimed = await runner._claim_pending_obligations()
        await runner._redeliver_claimed_obligations(claimed)

        assert not os.path.isfile(gone)
        adapter.send_video.assert_not_awaited()
        row = _row("gone-1")
        assert row["state"] == "failed"
        assert row["last_error"] == "media file missing"


class TestCronTextChainUnchanged:
    """Cron text obligations (no media kwargs) still record and sweep."""

    def test_text_record_sweep_without_media_kwargs(self):
        dl.record_obligation(
            obligation_id="cron-1",
            session_key="agent:satellite:telegram:channel:-100",
            platform="telegram",
            chat_id="-100",
            thread_id="42",
            content="the report",
            adapter_profile="satellite",
        )
        dl.mark_failed("cron-1", "send_path_degraded")
        claimed = dl.sweep_failed_for_runtime("telegram", profile="satellite")
        assert [(r["chat_id"], r["thread_id"], r["content"]) for r in claimed] == [("-100", "42", "the report")]
        assert all("media_path" not in r for r in claimed)