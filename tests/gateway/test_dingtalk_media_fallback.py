"""DingTalk rich-text candidates must recover through the inbound path."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["standard", "picture", "snake", None])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("failure", ["raise", "empty"])
async def test_rich_media_first_success_reaches_extraction(monkeypatch, caplog, winner, legacy, failure):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter
    import plugins.platforms.dingtalk.adapter as dt

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(dt, "dingtalk_robot_models", SimpleNamespace(
        RobotMessageFileDownloadRequest=Request,
        RobotMessageFileDownloadHeaders=Request,
    ), raising=False)
    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._client_id = "fixture-robot"
    adapter._get_access_token = AsyncMock(return_value="fixture-token")
    adapter._robot_sdk = SimpleNamespace(robot_message_file_download_with_options_async=object())
    calls = []

    async def sdk_call(method, request, headers, token):
        calls.append(request.download_code)
        if request.download_code != winner:
            if failure == "empty":
                return SimpleNamespace(body=None)
            raise RuntimeError("rejected " + request.download_code)
        return SimpleNamespace(body=SimpleNamespace(download_url="https://fixture.invalid/image.png"))

    adapter._sdk_call = sdk_call
    item = {"type": "picture", "downloadCode": "standard",
            "pictureDownloadCode": "picture", "download_code": "snake"}
    message = SimpleNamespace(message_type="richText", message_id="fixture-message",
        conversation_id="fixture-chat", sender_id="fixture-sender", rich_text_content=SimpleNamespace(
        rich_text_list=[{"text": "Please explain this image"}, item]))
    if legacy:
        message.rich_text = message.rich_text_content.rich_text_list
        del message.rich_text_content
    adapter.handle_message = AsyncMock()
    await adapter._on_message(message)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text.startswith("Please explain this image")
    assert ("Attachment unavailable" in event.text) is (winner is None)
    assert event.media_urls == (["https://fixture.invalid/image.png"] if winner else [])
    assert event.raw_message is message
    _, urls, _ = adapter._extract_media(message)
    assert urls == (["https://fixture.invalid/image.png"] if winner else [])
    expected = ["standard", "picture", "snake"]
    assert calls == (expected[:expected.index(winner) + 1] if winner else expected)
    assert adapter._extract_text(message) == "Please explain this image"
    assert not {"downloadCode", "pictureDownloadCode", "download_code"}.intersection(item)
    for code in expected:
        assert code not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["no-token", "no-sdk", "raise", "empty"])
@pytest.mark.parametrize("caption", ["", "keep me"])
@pytest.mark.parametrize("legacy", [False, True])
async def test_unresolved_media_is_delivered_without_raw_codes(monkeypatch, caplog, failure, caption, legacy):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter
    import plugins.platforms.dingtalk.adapter as dt

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(dt, "dingtalk_robot_models", SimpleNamespace(
        RobotMessageFileDownloadRequest=Request,
        RobotMessageFileDownloadHeaders=Request,
    ), raising=False)
    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._get_access_token = AsyncMock(return_value=None if failure == "no-token" else "token")
    adapter._robot_sdk = None if failure == "no-sdk" else SimpleNamespace(
        robot_message_file_download_with_options_async=object())
    adapter._sdk_call = AsyncMock(return_value=SimpleNamespace(body=None))
    if failure == "raise":
        adapter._sdk_call.side_effect = RuntimeError("secret-primary secret-alternate")
    items = [{"text": caption}, {"type": "picture", "downloadCode": "secret-primary",
             "pictureDownloadCode": "secret-alternate"}]
    message = SimpleNamespace(message_type="richText", message_id="fixture-message",
                              conversation_id="fixture-chat", sender_id="fixture-sender")
    if legacy:
        message.rich_text = items
    else:
        message.rich_text_content = SimpleNamespace(rich_text_list=items)
    adapter.handle_message = AsyncMock()
    await adapter._on_message(message)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "attachment" in event.text.lower() and "unavailable" in event.text.lower()
    assert not caption or event.text.startswith(caption)
    assert event.media_urls == []
    for code in ("secret-primary", "secret-alternate"):
        assert code not in event.text + repr(items) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "text", "resolved", "mixed"])
async def test_failure_notice_preserves_other_inbound_content(kind):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._get_access_token = AsyncMock(return_value=None)
    items = [] if kind == "empty" else [{"text": "original"}]
    if kind in {"resolved", "mixed"}:
        items.append({"type": "picture", "downloadCode": "secret",
                      "downloadUrl": "https://fixture.invalid/ready.png"})
    if kind == "mixed":
        items.append({"type": "picture", "downloadCode": "failed-secret"})
    message = SimpleNamespace(message_type="richText", message_id="fixture-message",
        conversation_id="fixture-chat", sender_id="fixture-sender", rich_text=items)
    adapter.handle_message = AsyncMock()
    await adapter._on_message(message)
    if kind == "empty":
        adapter.handle_message.assert_not_awaited()
        return
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text.startswith("original")
    assert ("Attachment unavailable" in event.text) is (kind == "mixed")
    assert event.media_urls == (["https://fixture.invalid/ready.png"] if kind in {"resolved", "mixed"} else [])
    assert "secret" not in event.text + repr(items)


@pytest.mark.asyncio
async def test_missing_token_scrubs_codes_but_preserves_text():
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._get_access_token = AsyncMock(return_value=None)
    item = {"type": "picture", "downloadCode": "raw", "pictureDownloadCode": "other"}
    message = SimpleNamespace(rich_text=[{"text": "keep me"}, item])
    await adapter._resolve_media_codes(message)
    assert adapter._extract_media(message)[1] == []
    assert adapter._extract_text(message) == "keep me"
    assert item == {"type": "picture"}


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_single_image_and_file_keep_writeback_contract(success):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._get_access_token = AsyncMock(return_value="token")
    adapter._resolve_single_code = AsyncMock(return_value="https://fixture.invalid/image.png" if success else None)
    image = SimpleNamespace(download_code="image-code")
    content = {"downloadCode": "file-code", "fileName": "file.png"}
    message = SimpleNamespace(image_content=image, message_type="file", extensions={"content": content}, robot_code="caller-robot")
    await adapter._resolve_media_codes(message)
    assert image.download_code == ("https://fixture.invalid/image.png" if success else "image-code")
    assert content["downloadCode"] == ("https://fixture.invalid/image.png" if success else "file-code")
    assert all(call.args[1:] == ("caller-robot", "token") for call in adapter._resolve_single_code.await_args_list)
