"""Native diagnostic producers keep captions, controls and truthful receipts.

No live platform connections: actual adapters run against recording transports.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.buzz.adapter import BuzzAdapter
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.google_chat.adapter import GoogleChatAdapter
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.wecom.adapter import WeComAdapter


@pytest.fixture(params=[None, False, True, "override"])
def policy(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    setting = request.param
    if setting is not None:
        display = {"suppress_warning_notifications": setting is True}
        if setting == "override":
            display["platforms"] = {
                name: {"suppress_warning_notifications": True}
                for name in ("matrix", "google_chat", "discord", "wecom", "buzz", "mattermost", "feishu", "telegram")
            }
        (tmp_path / "config.yaml").write_text(json.dumps({"display": display}))
    return setting is True or setting == "override"


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", [None, "Error: this is the requested caption"])
async def test_matrix_download_diagnostic_not_caption_is_suppressed(policy, caption, monkeypatch, caplog):
    adapter = MatrixAdapter(PlatformConfig())
    frames = []

    async def record(room, event_type, content):
        frames.append(content)
        return "$sent"

    adapter._client = SimpleNamespace(send_message_event=record)
    adapter._download_external_media_with_cap = AsyncMock(side_effect=RuntimeError("download fixture"))
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    result = await adapter.send_image("!room", "https://example.invalid/image?token=private", caption, "$reply", {"thread_id": "$thread"})
    if policy:
        assert [f["body"] for f in frames] == ([caption] if caption else [])
        assert not result.success and result.message_id is None
        if frames:
            assert frames[0]["m.relates_to"]["event_id"] == "$thread"
    else:
        notice = "I couldn't download and upload the image to Matrix. The source URL was not shown because it may contain private tokens."
        assert [f["body"] for f in frames] == [f"{caption}\n{notice}" if caption else notice]
        assert result.success and result.message_id == "$sent"
    assert "failed to download image" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", [None, "Warning: requested document caption"])
async def test_google_chat_oauth_fallback_only_hides_diagnostic(policy, caption, tmp_path):
    adapter = GoogleChatAdapter(PlatformConfig())
    frames = []

    async def record(chat, body):
        frames.append(body)
        return SendResult(success=True, message_id="spaces/test/messages/sent")

    adapter._create_message = record
    adapter._acquire_user_chat_api = AsyncMock(return_value=(None, None))
    path = tmp_path / "report.pdf"
    path.write_bytes(b"fixture")
    result = await adapter.send_document("spaces/test", str(path), caption=caption, metadata={"thread_id": "spaces/test/threads/thread"})
    assert not result.success and "OAuth" in result.error
    if policy:
        assert [f["text"] for f in frames] == ([caption] if caption else [])
    else:
        assert len(frames) == 1 and "/setup-files" in frames[0]["text"]
        assert str(path) in frames[0]["text"]
        if caption:
            assert frames[0]["text"].startswith(caption + "\n")
    if frames:
        assert frames[0]["thread"]["name"] == "spaces/test/threads/thread"
    # An explicit control request must still use the same transport.
    await adapter.send_clarify("spaces/test", "Continue?", ["yes", "no"], "clarify", "session")
    assert "cardsV2" in frames[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_document", "send_voice", "send_multiple_images"])
@pytest.mark.parametrize("forum", [False, True])
async def test_discord_rejected_media_preserves_caption_without_fake_receipt(policy, method, forum, tmp_path, caplog):
    adapter = DiscordAdapter(PlatformConfig())
    frames = []

    async def record(**payload):
        frames.append(payload)
        return SimpleNamespace(id=123, attachments=[])

    async def create_thread(**payload):
        frames.append(payload)
        return SimpleNamespace(thread=SimpleNamespace(id=101), message=SimpleNamespace(id=102))

    channel = SimpleNamespace(send=record, id=99, create_thread=create_thread)
    adapter._client = SimpleNamespace()
    adapter._resolve_channel = AsyncMock(return_value=channel)
    adapter._is_forum_parent = lambda channel: forum
    adapter._discord_upload_limit_bytes = lambda channel: 1
    path = tmp_path / "large.png"
    path.write_bytes(b"too large")
    caption = "Warning: requested caption"
    if method == "send_multiple_images":
        result = await adapter.send_multiple_images("99", [(path.as_uri(), caption)])
    else:
        result = await getattr(adapter, method)("99", str(path), caption=caption)
    assert not result.success and result.message_id is None
    if policy:
        assert [f["content"] for f in frames] == [caption]
    elif forum:
        assert frames == []  # legacy forum rejection sends no notice
    else:
        assert len(frames) == 1 and "⚠️" in frames[0]["content"]
        assert caption not in frames[0]["content"]  # preserve legacy default
    assert "large" in caplog.text


@pytest.mark.asyncio
async def test_wecom_rejected_media_keeps_caption_and_failed_result(policy):
    adapter = WeComAdapter(PlatformConfig())
    frames = []

    async def record(command, body, **kwargs):
        frames.append(body)
        return {"errcode": 0, "headers": {"req_id": "sent"}}

    adapter._send_request = record
    adapter._prepare_outbound_media = AsyncMock(return_value={"rejected": True, "reject_reason": "fixture rejection"})
    result = await adapter.send_document("chat", "fixture.pdf", caption="Error: requested caption")
    assert not result.success and result.error == "fixture rejection"
    assert [f["markdown"]["content"] for f in frames] == (["Error: requested caption"] if policy else ["⚠️ fixture rejection"])


@pytest.mark.asyncio
async def test_buzz_missing_upload_fallback_and_real_upload_receipts(policy, tmp_path):
    adapter = BuzzAdapter(PlatformConfig(extra={"cli_path": "fixture-buzz"}))
    adapter._mention_pubkeys_for = AsyncMock(return_value=[])
    frames = []

    async def record(args, content, *extra):
        frames.append((args, content))
        return 0, json.dumps({"event_id": "a" * 64, "accepted": True}), ""

    adapter._run_message_send = record
    result = await adapter.send_image_file("chat", str(tmp_path / "missing.png"), caption="requested caption")
    assert frames[0][1] == ("requested caption" if policy else "requested caption\n⚠️ Couldn't deliver the image attachment.")
    assert result.success is (not policy)
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    result = await adapter.send_image_file("chat", str(path), caption="⚠️ user caption")
    assert result.success and result.message_id == "a" * 64
    assert "--file" in frames[-1][0] and frames[-1][1] == "⚠️ user caption"


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["send", "edit"])
async def test_discord_truncation_warning_covers_send_and_final_edit(policy, lane):
    adapter = DiscordAdapter(PlatformConfig())
    frames = []

    async def record(**payload):
        frames.append(payload["content"])
        return SimpleNamespace(id=123, to_reference=lambda **kwargs: None)

    message = SimpleNamespace(id=123, edit=record, to_reference=lambda **kwargs: None)
    channel = SimpleNamespace(send=record, get_partial_message=lambda ident: message, id=99)
    adapter._client = SimpleNamespace()
    adapter._resolve_channel = AsyncMock(return_value=channel)
    adapter._is_forum_parent = lambda channel: False
    adapter._record_response_async = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 80
    adapter.MAX_SPLIT_MESSAGES = 3
    text = "Warning: requested result. " * 40
    if lane == "send":
        result = await adapter.send("99", text)
    else:
        result = await adapter.edit_message("99", "123", text, finalize=True)
    assert result.success
    assert frames[0].startswith("Warning: requested result.")
    assert any("Response truncated" in frame for frame in frames) is (not policy)
    assert len(frames) == (2 if policy else 3)


@pytest.mark.asyncio
async def test_wecom_downgrade_preserves_native_upload_and_caption(policy):
    adapter = WeComAdapter(PlatformConfig())
    frames = []

    async def record(cmd, body, **kwargs):
        frames.append((cmd, body))
        return {"errcode": 0, "body": {"upload_id": "upload", "media_id": "media"}, "headers": {"req_id": "sent"}}

    adapter._send_request = record
    adapter._load_outbound_media = AsyncMock(return_value=(b"fixture", "audio/ogg", "voice.ogg"))
    result = await adapter.send_voice("chat", "voice.ogg", caption="Error: user caption")
    assert result.success
    assert any(body.get("file", {}).get("media_id") == "media" for _, body in frames)
    text = [body["markdown"]["content"] for _, body in frames if "markdown" in body]
    assert text[0] == "Error: user caption"
    assert len(text) == (1 if policy else 2)


@pytest.mark.asyncio
async def test_google_chat_revoked_oauth_fallback_uses_same_policy(policy, tmp_path, monkeypatch, caplog):
    from plugins.platforms.google_chat import adapter as google

    class Revoked(Exception):
        resp = SimpleNamespace(status=403)

    def fail(**kwargs):
        raise Revoked("token revoked")

    adapter = GoogleChatAdapter(PlatformConfig())
    adapter._acquire_user_chat_api = AsyncMock(return_value=(SimpleNamespace(media=lambda: SimpleNamespace(upload=fail)), "owner"))
    adapter._user_creds_by_email["owner"] = object()
    adapter._user_chat_api_by_email["owner"] = object()
    monkeypatch.setattr(google, "HttpError", Revoked)
    monkeypatch.setattr(google, "MediaFileUpload", lambda *args, **kwargs: object())
    frames = []

    async def record(chat_id, body):
        frames.append(body)
        return SendResult(success=True, message_id="sent")

    adapter._create_message = record
    path = tmp_path / "file.pdf"
    path.write_bytes(b"fixture")
    result = await adapter.send_document("chat", str(path), caption="requested caption")
    assert not result.success
    assert "owner" not in adapter._user_creds_by_email
    assert "owner" not in adapter._user_chat_api_by_email
    assert "media.upload auth failure" in caplog.text
    assert frames[0]["text"] == "requested caption" if policy else "/setup-files" in frames[0]["text"]


@pytest.mark.asyncio
async def test_mattermost_thread_fallback_keeps_requested_result(policy, caplog):
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "thread"}))
    frames = []

    async def api(method, path, payload=None):
        frames.append(dict(payload))
        if "root_id" in payload:
            adapter._last_post_status = 404
            adapter._last_post_error = "root_id not found"
            return {}
        return {"id": "sent"}

    adapter._api = api
    adapter._resolve_root_id = AsyncMock(return_value="root")
    result = await adapter.send("chat", "Error: requested result", metadata={"thread_id": "root", "notify": True})
    assert result.success
    assert frames[0]["root_id"] == "root" and "root_id" not in frames[1]
    expected = "Error: requested result"
    if not policy:
        expected = "⚠️ Mattermost thread delivery failed; posting final reply in channel.\n\n" + expected
    assert frames[1]["message"] == expected
    assert "falling back to flat channel" in caplog.text


@pytest.mark.asyncio
async def test_feishu_downgrade_keeps_attachment_and_caption(policy, tmp_path):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    frames = []
    path = tmp_path / "animation.gif"
    path.write_bytes(b"fixture")
    adapter._download_remote_document = AsyncMock(return_value=(str(path), path.name))

    def upload(request):
        return SimpleNamespace(success=lambda: True, code=0, data=SimpleNamespace(file_key="file-key"))

    async def send_raw(**kwargs):
        frames.append(kwargs)
        return SimpleNamespace(success=lambda: True, code=0, data=SimpleNamespace(message_id="sent"))

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(file=SimpleNamespace(create=upload))))
    adapter._send_raw_message = send_raw
    result = await adapter.send_animation("chat", "https://example.invalid/animation.gif", caption="Warning: requested caption")
    assert result.success
    assert "file-key" in frames[0]["payload"]
    assert "Warning: requested caption" in frames[0]["payload"]
    assert ("GIF downgraded to file" in frames[0]["payload"]) is (not policy)


@pytest.mark.asyncio
async def test_telegram_cache_warning_does_not_hide_inbound_context(policy):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig())
    frames = []

    async def reply_text(text):
        frames.append(text)

    from gateway.platforms.event import MessageEvent
    event = MessageEvent(text="Requested content", source=adapter.build_source(chat_id="42", user_id="42"))
    await adapter._surface_media_cache_failure(SimpleNamespace(reply_text=reply_text), event, "photo", RuntimeError("fixture"))
    assert bool(frames) is (not policy)
    assert event.text.startswith("Requested content")
    assert "could not be downloaded" in event.text
    assert ("asked to retry" in event.text) is (not policy)
