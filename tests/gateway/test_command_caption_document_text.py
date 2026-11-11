"""A text file sent with a slash-command caption (``/goal summarize this``) keeps the command first.

Adapters inline a small text file's content into the message. Prepended in front of a command
caption it made the command unparseable, so ``/goal`` (or ``/steer``) ran as plain chat with the
literal ``/goal`` inside it. Under a command the content is left out and the file stays attached
(the agent is told to read it), like any file too large to inline.
"""

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

import pytest

from gateway.platforms.helpers import inline_document_text


@pytest.mark.parametrize("caption, injection, expected", [
    ("look at this", "[Content of a.txt]:\nx", ("[Content of a.txt]:\nx\n\nlook at this", [True])),
    ("", "[Content of a.txt]:\nx", ("[Content of a.txt]:\nx", [True])),
    ("/goal summarize the report", "[Content of a.txt]:\nx", ("/goal summarize the report", [False])),
    ("/home/me/notes.txt is attached", "[Content of a.txt]:\nx",  # a path, not a command
     ("[Content of a.txt]:\nx\n\n/home/me/notes.txt is attached", [True])),
])
def test_inline_document_text_keeps_a_command_caption_first(caption, injection, expected):
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
