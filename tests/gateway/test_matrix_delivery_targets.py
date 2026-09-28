"""Explicit delivery resolves aliases without sending invalid destinations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "joined",
        "join",
        "lookup_error",
        "missing_room",
        "join_error",
        "wrong_room",
        "mxid",
        "encrypted",
        "state_error",
    ],
)
async def test_explicit_target_resolution_precedes_native_send(case):
    room = "!destination:remote.test"
    alias = "#destination:remote.test"
    target = "@person:remote.test" if case == "mxid" else alias
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"e2ee_mode": "off"}))
    client = SimpleNamespace(
        resolve_room_alias=AsyncMock(
            return_value=SimpleNamespace(room_id=room, servers=["route.test"])
        ),
        join_room=AsyncMock(return_value=room),
        send_message_event=AsyncMock(return_value="$sent"),
        get_state_event=AsyncMock(side_effect=_MissingEncryption()),
        crypto=None,
    )
    adapter._client = client
    adapter._refresh_dm_cache = AsyncMock()
    if case == "joined":
        adapter._joined_rooms.add(room)
    if case == "lookup_error":
        client.resolve_room_alias.side_effect = RuntimeError("directory unavailable")
    if case == "missing_room":
        client.resolve_room_alias.return_value.room_id = ""
    if case == "join_error":
        client.join_room.side_effect = RuntimeError("join forbidden")
    if case == "wrong_room":
        client.join_room.return_value = "!other:remote.test"
    if case == "encrypted":
        client.get_state_event.side_effect = None
        client.get_state_event.return_value = {"algorithm": "m.megolm.v1.aes-sha2"}
    if case == "state_error":
        client.get_state_event.side_effect = RuntimeError("state unavailable")

    result = await adapter.send(
        target, "Scheduled report", metadata={"thread_id": "$root"}
    )

    if case in {"joined", "join"}:
        assert result.success, result.error
        args = client.send_message_event.await_args
        assert str(args.args[0]) == room
        assert args.args[2]["m.relates_to"] == {
            "rel_type": "m.thread",
            "event_id": "$root",
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$root"},
        }
        if case == "join":
            client.join_room.assert_awaited_once_with(
                room, servers=["route.test"], max_retries=0
            )
        else:
            client.join_room.assert_not_awaited()
        return

    assert not result.success
    assert result.error is not None
    assert target in result.error
    expected = {
        "lookup_error": "directory unavailable",
        "missing_room": "room ID",
        "join_error": "join forbidden",
        "wrong_room": "!other:remote.test",
        "mxid": "MXID",
        "encrypted": "encrypted",
        "state_error": "state unavailable",
    }
    assert expected[case] in result.error
    client.send_message_event.assert_not_awaited()
    if case in {"lookup_error", "missing_room", "mxid"}:
        client.join_room.assert_not_awaited()


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["send", "retry", "key_share", "upload", "media_send", "encrypted_upload"],
)
@pytest.mark.parametrize("resolved", [False, True])
async def test_native_send_failures_preserve_the_original_target(
    failure, resolved, tmp_path, monkeypatch
):
    from gateway.platforms.base import SendResult

    alias = "#reports:remote.test"
    room = "!reports:remote.test"
    target = f"{alias}/$root"
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"e2ee_mode": "off"}))
    client = SimpleNamespace(
        resolve_room_alias=AsyncMock(
            return_value=SimpleNamespace(room_id=room, servers=["remote.test"])
        ),
        join_room=AsyncMock(return_value=room),
        get_state_event=AsyncMock(side_effect=_MissingEncryption()),
        send_message_event=AsyncMock(return_value="$sent"),
        upload_media=AsyncMock(return_value="mxc://remote.test/file"),
        crypto=None,
    )
    adapter._client = client
    adapter._refresh_dm_cache = AsyncMock()
    if failure in {"send", "retry", "key_share", "media_send"}:
        client.send_message_event.side_effect = ValueError("power level rejected")
    if failure in {"retry", "key_share", "encrypted_upload"}:
        adapter._encryption = True
        client.crypto = SimpleNamespace(share_keys=AsyncMock())
        if failure == "key_share":
            client.crypto.share_keys.side_effect = ValueError("key share rejected")
    if failure in {"upload", "encrypted_upload"}:
        client.upload_media.side_effect = ValueError("upload rejected")
    if failure == "encrypted_upload":
        import sys

        client.get_state_event.side_effect = None
        client.state_store = SimpleNamespace(is_encrypted=AsyncMock(return_value=True))
        monkeypatch.setitem(
            sys.modules,
            "mautrix.crypto.attachments",
            SimpleNamespace(
                encrypt_attachment=lambda data: (
                    b"encrypted",
                    SimpleNamespace(serialize=lambda: {}),
                ),
            ),
        )
    adapter._is_dm_room = AsyncMock(return_value=False)
    metadata = {"thread_id": "$root"}
    if resolved:
        from gateway.session import SessionSource
        from gateway.config import Platform

        source = await adapter.resolve_delivery_target(
            SessionSource(
                platform=Platform.MATRIX,
                chat_id=alias,
                thread_id="$root",
            )
        )
        chat_id = source.chat_id
        metadata["_original_target"] = target
    else:
        chat_id = target
    if failure in {"send", "retry", "key_share"}:
        result = await adapter.send(chat_id, "Scheduled report", metadata=metadata)
    else:
        attachment = tmp_path / "report.txt"
        attachment.write_text("Scheduled report", encoding="utf-8")
        result = await adapter.send_document(
            chat_id, str(attachment), metadata=metadata
        )
    detail = (
        "key share rejected"
        if failure == "key_share"
        else (
            "upload rejected"
            if failure in {"upload", "encrypted_upload"}
            else "power level rejected"
        )
    )
    assert result == SendResult(
        success=False, error=f"Matrix target '{target}': {detail}",
        error_kind="unknown" if failure == "media_send" else None,
    )
