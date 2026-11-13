"""A text file sent with a slash-command caption (``/goal summarize this``) keeps the command first.

Adapters inline a small text file's content into the message. Prepended in front of a command
caption it made the command unparseable, so ``/goal`` (or ``/steer``) ran as plain chat with the
literal ``/goal`` inside it. Under a command the content is left out and the file stays attached
(the agent is told to read it), like any file too large to inline.
"""

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("caption, injection, expected", [
    ("look at this", "[Content of a.txt]:\nx", ("[Content of a.txt]:\nx\n\nlook at this", [True])),
    ("", "[Content of a.txt]:\nx", ("[Content of a.txt]:\nx", [True])),
    ("/goal summarize the report", "[Content of a.txt]:\nx", ("/goal summarize the report", [False])),
    ("/home/me/notes.txt is attached", "[Content of a.txt]:\nx",  # a path, not a command
     ("[Content of a.txt]:\nx\n\n/home/me/notes.txt is attached", [True])),
    ("@file:/tmp/a.txt\n/goal go", "[Content of a.txt]:\nx", ("@file:/tmp/a.txt\n/goal go", [False])),
])
def test_inline_document_text_keeps_a_command_caption_first(caption, injection, expected):
    from hermes_agent.gateway.platforms.helpers import inline_document_text  # here, so the adapter cases run on any base

    assert inline_document_text(caption, injection, [True]) == expected


@pytest.mark.asyncio
async def test_whatsapp_cloud_command_caption_on_a_text_file_stays_a_command(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp_cloud import adapter as wac
    from tests.gateway.test_whatsapp_cloud import _make_adapter

    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")  # the dispatch-mechanics opt-in its test file uses
    adapter = _make_adapter(app_secret="key")
    adapter._http_client = MagicMock()
    adapter._http_client.get = AsyncMock(side_effect=[
        MagicMock(status_code=200, json=MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/whatsapp/m/doc", "mime_type": "text/plain"})),
        MagicMock(status_code=200, content=b"quarterly numbers: 42"),
    ])
    raw_message = {
        "from": "1555", "id": "wamid.doc3", "timestamp": "0", "type": "document",
        "document": {"id": "media_doc_abc", "mime_type": "text/plain", "filename": "report.txt",
                     "caption": "/goal summarize the report"},
    }
    with patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
        event = await adapter._build_message_event_from_cloud(raw_message, {"1555": "U"}, {"phone_number_id": "1"})

    assert event is not None
    assert event.get_command() == "goal"
    assert event.get_command_args() == "summarize the report"
    assert len(event.media_urls) == 1 and event.media_text_inlined in ([], [False])


@pytest.mark.asyncio
async def test_feishu_self_mentioned_command_caption_on_a_text_file_stays_a_command():
    """``@Hermes /goal ...`` in a post with a text file: the mention is stripped before the command
    check, so the file body does not become the goal's arguments."""
    import json
    from types import SimpleNamespace
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_bot", "", "Hermes"
    adapter._download_feishu_message_resources = AsyncMock(return_value=(["/cache/report.txt"], ["text/plain"]))
    adapter._maybe_extract_text_document = AsyncMock(return_value="[Content of report.txt]:\nquarterly numbers: 42")
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None})
    adapter._resolve_source_chat_type = MagicMock(return_value="group")
    adapter.build_source = MagicMock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    message = SimpleNamespace(
        content=json.dumps({"title": "", "content": [[
            {"tag": "at", "user_id": "@_user_1", "user_name": "Hermes"},
            {"tag": "text", "text": " /goal summarize the report"},
        ]], "files": [{"file_key": "file_1", "file_name": "report.txt", "is_folder": False}]}),
        message_type="post", message_id="m-goal-file", chat_id="oc_chat", thread_id=None, root_id=None,
        parent_id=None, upper_message_id=None,
        mentions=[SimpleNamespace(key="@_user_1", id=SimpleNamespace(open_id="ou_bot", user_id=""), name="Hermes")],
    )

    await adapter._process_inbound_message(
        data={}, message=message, sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None),
        chat_type="group", message_id="m-goal-file",
    )

    event = adapter._dispatch_inbound_event.await_args.args[0]
    assert event.get_command() == "goal"
    assert "quarterly numbers" not in event.get_command_args()
    assert event.media_urls == ["/cache/report.txt"] and event.media_text_inlined == [False]


async def _telegram_album(tmp_path, monkeypatch, captions):
    from tests.gateway.test_telegram_documents import _make_document, _make_file_obj, _make_message, _make_update
    from tests.waiting import async_wait_until

    from hermes_agent.gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    for attr in ("DOCUMENT_CACHE_DIR", "IMAGE_CACHE_DIR"):
        monkeypatch.setattr(f"hermes_agent.gateway.platforms.base.{attr}", tmp_path / attr)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"media_group_wait_seconds": 0.05}))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    for name, caption in zip(("a.txt", "b.txt", "c.pdf", "d.txt"), captions):
        body = f"SECRETBODY_{name}: 42\n".encode()  # a trailing newline, as real files have
        mime = "application/pdf" if name.endswith(".pdf") else "text/plain"  # a PDF is never inlined
        doc = _make_document(file_name=name, mime_type=mime, file_size=len(body), file_obj=_make_file_obj(body))
        await adapter._handle_media_message(_make_update(_make_message(document=doc, caption=caption, media_group_id="mg")),
                                            MagicMock())
    await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
async def test_telegram_album_with_a_plain_caption_still_inlines_every_file(tmp_path, monkeypatch):
    event = await _telegram_album(tmp_path, monkeypatch, ("compare these", None))
    assert "SECRETBODY_a.txt" in event.text and "SECRETBODY_b.txt" in event.text
    assert event.media_text_inlined == [True, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("captioned", [0, 1])
async def test_telegram_album_with_a_command_caption_keeps_sibling_files_as_attachments(captioned, tmp_path, monkeypatch):
    """Only one album item carries the caption. Its uncaptioned sibling inlined its text, which
    then became the goal's arguments (caption on item 1) or hid the command (caption on item 2)."""
    captions = tuple("/goal summarize both reports" if i == captioned else None for i in range(2))
    event = await _telegram_album(tmp_path, monkeypatch, captions)
    assert event.get_command() == "goal"
    assert event.get_command_args() == "summarize both reports"
    assert "SECRETBODY" not in event.text
    assert len(event.media_urls) == 2 and event.media_text_inlined == [False, False]


@pytest.mark.asyncio
async def test_telegram_album_keeps_a_later_items_caption_after_the_command(tmp_path, monkeypatch):
    """A caption on an item merged after the command one survives later merges."""
    event = await _telegram_album(tmp_path, monkeypatch, (None, "/goal summarize these", "Use only the 2026 figures", None))
    assert event.get_command() == "goal"
    assert "Use only the 2026 figures" in event.get_command_args()
    assert "SECRETBODY" not in event.text and not any(event.media_text_inlined)
