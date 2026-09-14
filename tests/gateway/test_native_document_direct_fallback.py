"""Native-only document delivery must not accept adapter text fallbacks."""

import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.hosted_room_file_delivery import Document, deliver_document
from gateway.native_document_guard import (
    NativeDocumentFallback,
    require_native_document,
)


class _Native:
    def __init__(self, name, adapter, client, path):
        self.name = name
        self.adapter = adapter
        self.client = client
        self.path = path
        self.uploaded = []
        self.notices = []


@pytest.fixture(params=["slack", "matrix"])
def native(request, tmp_path):
    path = tmp_path / "private-document.bin"
    path.write_bytes(b"exact document bytes\x00\xff")

    if request.param == "slack":
        import slack_sdk

        from plugins.platforms.slack.adapter import SlackAdapter

        assert Path(slack_sdk.__file__).is_file()
        client = slack_sdk.web.async_client.AsyncWebClient(token="isolated-test-token")
        client.api_call = AsyncMock(
            side_effect=AssertionError("unexpected SDK network call")
        )
        client.assistant_threads_setStatus = AsyncMock(return_value={"ok": True})
        adapter = SlackAdapter(
            PlatformConfig(enabled=True, token="isolated-test-token")
        )
        adapter._app = SimpleNamespace(client=client)
    else:
        from plugins.platforms.matrix.adapter import MatrixAdapter

        client = SimpleNamespace()
        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True, token="isolated-test-token", extra={"e2ee": "off"}
            )
        )
        adapter._client = client
        adapter._encryption = False

    yield _Native(request.param, adapter, client, path)
    if request.param == "slack":
        client.api_call.assert_not_awaited()


def _install_success(native):
    if native.name == "slack":

        async def upload(**kwargs):
            native.uploaded.append((Path(kwargs["file"]).read_bytes(), kwargs))
            return {"file": {"id": "F1"}}

        native.client.files_upload_v2 = AsyncMock(side_effect=upload)
        native.client.chat_postMessage = AsyncMock()
    else:

        async def upload(data, **kwargs):
            native.uploaded.append((bytes(data), kwargs))
            return "mxc://isolated/document"

        async def send_event(room_id, event_type, content):
            assert content["msgtype"] == "m.file"
            assert content["url"] == "mxc://isolated/document"
            return "$document"

        native.client.upload_media = AsyncMock(side_effect=upload)
        native.client.send_message_event = AsyncMock(side_effect=send_event)


def _install_failure(native):
    if native.name == "slack":
        native.client.files_upload_v2 = AsyncMock(
            side_effect=RuntimeError("upload_rejected")
        )

        async def text(**kwargs):
            native.notices.append(kwargs)
            return {"ts": "notice"}

        native.client.chat_postMessage = AsyncMock(side_effect=text)
    else:
        native.path.unlink(missing_ok=True)

        async def text(room_id, event_type, content):
            native.notices.append(content)
            return "$notice"

        native.client.send_message_event = AsyncMock(side_effect=text)


async def _send(native, *, path=None):
    return await native.adapter.send_document(
        "C123" if native.name == "slack" else "!room:isolated",
        str(path or native.path),
        caption="Exact document",
        file_name="shared.bin",
        reply_to="42",
        metadata={"thread_id": "42", "group_file_delivery_id": "receipt"},
    )


def receipt(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT state, attempts FROM hosted_room_file_deliveries ORDER BY delivery_key"
        ).fetchall()


@pytest.mark.asyncio
async def test_direct_text_fallback_is_refused_only_in_strict_context(native):
    _install_failure(native)
    with require_native_document():
        with pytest.raises(NativeDocumentFallback):
            await _send(native)
    assert native.notices == []

    result = await _send(native)
    assert result.success is True and result.message_id
    assert len(native.notices) == 1
    rendered = str(native.notices[0])
    assert "Couldn't deliver" in rendered
    assert str(native.path) not in rendered


@pytest.mark.asyncio
async def test_native_success_remains_exact_and_never_sends_a_notice(native):
    _install_success(native)
    with require_native_document():
        result = await _send(native)
    assert result.success is True
    assert native.uploaded[0][0] == native.path.read_bytes()
    assert native.notices == []


@pytest.mark.asyncio
async def test_real_delivery_receipt_never_marks_text_fallback_delivered(
    native, tmp_path
):
    _install_failure(native)
    recheck = AsyncMock()
    if native.name == "matrix":
        # The delivery helper creates its private file before the final recheck.
        async def remove_before_submit():
            if recheck.await_count == 3:
                files = list((tmp_path / "group-file-delivery-tmp").glob("send-*/*"))
                assert len(files) == 1
                files[0].unlink()

        recheck.side_effect = remove_before_submit
        native.path.write_bytes(b"fixture input remains separate")

    args = dict(
        db_path=tmp_path / "state.db",
        key="request",
        scope="exact-file",
        adapter=native.adapter,
        source=SimpleNamespace(
            platform=native.name,
            chat_id="C123" if native.name == "slack" else "!room:isolated",
        ),
        load=lambda maximum: Document("shared.bin", b"exact delivery bytes\x00\xff"),
        recheck=recheck,
        metadata={},
        reply_to="42",
    )
    outcome = await deliver_document(**args)
    assert outcome == "unknown"
    assert await deliver_document(**args) == "unknown"
    assert receipt(args["db_path"]) == [("unknown", 1)]
    assert native.notices == []
    assert recheck.await_count == 3
    assert list((tmp_path / "group-file-delivery-tmp").iterdir()) == []


@pytest.mark.asyncio
async def test_real_delivery_success_records_native_bytes(native, tmp_path):
    _install_success(native)
    recheck = AsyncMock()
    args = dict(
        db_path=tmp_path / "state.db",
        key="request-success",
        scope="exact-file-success",
        adapter=native.adapter,
        source=SimpleNamespace(
            platform=native.name,
            chat_id="C123" if native.name == "slack" else "!room:isolated",
        ),
        load=lambda maximum: Document("shared.bin", b"exact delivery bytes\x00\xff"),
        recheck=recheck,
        metadata={},
        reply_to="42",
    )
    assert await deliver_document(**args) == "delivered"
    assert receipt(args["db_path"]) == [("delivered", 1)]
    assert await deliver_document(**args) == "delivered"
    assert native.uploaded[0][0] == b"exact delivery bytes\x00\xff"
    assert native.notices == []
    assert recheck.await_count == 3
    assert list((tmp_path / "group-file-delivery-tmp").iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_strict_context_does_not_change_later_ordinary_fallback(native):
    entered = asyncio.Event()
    release = asyncio.Event()
    _install_failure(native)
    if native.name == "slack":

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            raise RuntimeError("upload_rejected")

        native.client.files_upload_v2 = AsyncMock(side_effect=blocked)
    else:
        original = native.adapter._send_local_file

        async def blocked(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        native.adapter._send_local_file = blocked

    async def strict():
        with require_native_document():
            await _send(native)

    pending = asyncio.create_task(strict())
    await entered.wait()
    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending

    result = await _send(native)
    assert result.success is True
    assert len(native.notices) == 1


@pytest.mark.asyncio
async def test_same_adapter_ordinary_fallback_remains_available_during_strict_task(
    native,
):
    _install_failure(native)
    entered, release = asyncio.Event(), asyncio.Event()

    async def strict():
        with require_native_document():
            entered.set()
            await release.wait()
            with pytest.raises(NativeDocumentFallback):
                await _send(native)

    pending = asyncio.create_task(strict())
    try:
        await entered.wait()
        ordinary = await _send(native)
        assert ordinary.success is True
    finally:
        release.set()
        await pending
    assert len(native.notices) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 500])
async def test_whatsapp_document_bridge_has_no_direct_text_fallback(tmp_path, status):
    from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter

    adapter = _make_adapter()
    path = tmp_path / "shared.bin"
    path.write_bytes(b"exact bridge bytes\x00\xff")
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value={"messageId": "native-document"})
    response.text = AsyncMock(return_value="native upload rejected")
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))

    with require_native_document():
        result = await adapter.send_document(
            "15551234567", str(path), file_name="shared.bin"
        )
    assert result.success is (status == 200)
    adapter._http_session.post.assert_called_once()
    call = adapter._http_session.post.call_args
    assert call.args[0].endswith("/send-media")
    assert call.kwargs["json"]["mediaType"] == "document"
    assert Path(call.kwargs["json"]["filePath"]).read_bytes() == path.read_bytes()
    assert call.kwargs["json"]["fileName"] == "shared.bin"
