"""Installed native overrides, real file handles, and opt-in fallback semantics."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.native_document_guard import (
    NativeDocumentFallback,
    require_native_document,
)
from gateway.platforms.base import BasePlatformAdapter, SendResult


@pytest.fixture(params=["telegram", "discord"])
def native(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Gateway conftest inserts optional-SDK mocks even when SDKs are installed.
    # This file has its own runner subprocess; import the actual distributions.
    for root in ("telegram", "discord"):
        if not isinstance(sys.modules.get(root), ModuleType):
            for name in list(sys.modules):
                if name == root or name.startswith(root + "."):
                    sys.modules.pop(name)
    config = PlatformConfig(enabled=True, token="isolated-test-token")
    if request.param == "telegram":
        import telegram
        from plugins.platforms.telegram.adapter import TelegramAdapter

        assert Path(telegram.__file__).is_file()
        assert isinstance(telegram.__version__, str)
        adapter = TelegramAdapter(config)
    else:
        import discord
        from plugins.platforms.discord.adapter import DiscordAdapter

        assert Path(discord.__file__).is_file()
        assert isinstance(discord.__version__, str)
        adapter = DiscordAdapter(config)
    assert type(adapter).send_document is not BasePlatformAdapter.send_document
    assert type(adapter).send_document.strict_native_document_guard is True
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="notice"))
    path = tmp_path / "private-source.txt"
    path.write_bytes(b"exact native bytes\x00\xff")
    state = SimpleNamespace(
        adapter=adapter,
        path=path,
        platform=request.param,
        calls=[],
        fail=False,
        empty=False,
        entered=None,
        release=None,
    )

    async def upload(**kwargs):
        if state.platform == "telegram":
            state.calls.append((kwargs["document"].read(), kwargs["filename"], kwargs))
        else:
            file = kwargs["files"][0]
            # Keep the real discord.File lifecycle, replacing only network I/O.
            try:
                state.calls.append((file.fp.read(), file.filename, kwargs))
            finally:
                file.close()
        if state.entered is not None:
            state.entered.set()
            await state.release.wait()
        if state.fail:
            raise RuntimeError("upload outcome unavailable")
        return SimpleNamespace(
            message_id=456, id=456, attachments=[] if state.empty else [object()]
        )

    state.upload = AsyncMock(side_effect=upload)
    if state.platform == "telegram":
        adapter._bot = SimpleNamespace(send_document=state.upload)
    else:
        import discord

        channel = SimpleNamespace(type=discord.ChannelType.text, send=state.upload)
        adapter._client = SimpleNamespace(get_channel=lambda _: channel)
    return state


async def send(state):
    return await state.adapter.send_document(
        "123",
        str(state.path),
        caption="Exact file",
        file_name="shared.txt",
        reply_to="42",
        metadata={"thread_id": "7", "group_file_delivery_id": "receipt"},
    )


@pytest.mark.asyncio
async def test_installed_override_native_success_is_preserved(native):
    with require_native_document():
        result = await send(native)
    assert result.success is True and result.message_id == "456"
    assert native.calls[0][:2] == (native.path.read_bytes(), "shared.txt")
    assert (
        native.calls[0][2].get("caption", native.calls[0][2].get("content"))
        == "Exact file"
    )
    if native.platform == "telegram":
        assert native.calls[0][2]["reply_to_message_id"] == 42
        assert native.calls[0][2]["message_thread_id"] == 7
    native.adapter.send.assert_not_awaited()
    assert native.upload.await_count == 1


@pytest.mark.asyncio
async def test_installed_override_exception_cannot_report_text_as_delivery(native):
    native.fail = True
    with require_native_document():
        with pytest.raises(NativeDocumentFallback):
            await send(native)
    native.adapter.send.assert_not_awaited()
    assert native.upload.await_count == 1
    # Existing generic file paths retain their historical text fallback.
    result = await send(native)
    assert result.success and result.message_id == "notice"
    native.adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_ordinary_send_does_not_inherit_strict_context(native):
    native.fail = True
    native.entered, native.release = asyncio.Event(), asyncio.Event()

    async def strict():
        with require_native_document():
            with pytest.raises(NativeDocumentFallback):
                await send(native)

    pending = asyncio.create_task(strict())
    await native.entered.wait()
    ordinary = await BasePlatformAdapter.send_document(
        native.adapter, "123", str(native.path)
    )
    native.release.set()
    await pending
    assert ordinary.success is True
    native.adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_native_file_is_still_a_failure_without_text_fallback(native):
    native.path.unlink()
    with require_native_document():
        result = await send(native)
    assert result.success is False
    native.upload.assert_not_awaited()
    native.adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["discord"], indirect=True)
async def test_discord_zero_attachment_acceptance_is_not_native_success(native):
    native.empty = True
    with require_native_document():
        result = await send(native)
    assert result.success is False and result.message_id == "456"
    assert "no files" in result.error
    native.adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_real_override_delivery_receipt_never_replays_an_ambiguous_send(
    native, failure
):
    from gateway.hosted_room_file_delivery import Document, deliver_document

    native.fail = failure
    recheck = AsyncMock()
    arguments = dict(
        db_path=native.path.parent / "state.db",
        key="request",
        scope="exact-file",
        adapter=native.adapter,
        source=SimpleNamespace(platform=native.platform, chat_id="123"),
        load=lambda maximum: Document("shared.txt", b"exact delivery bytes"),
        recheck=recheck,
        metadata={},
        reply_to="42",
    )
    outcome = await deliver_document(**arguments)
    assert outcome == ("unknown" if failure else "delivered")
    assert await deliver_document(**arguments) == outcome
    assert native.calls[0][:2] == (b"exact delivery bytes", "shared.txt")
    native.adapter.send.assert_not_awaited()
    assert native.upload.await_count == 1
    assert recheck.await_count == 3
    assert list((native.path.parent / "group-file-delivery-tmp").iterdir()) == []
