"""LINE text attachments must reach the prompt, not just the cache (#76022)."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

line = load_plugin_adapter("line")


@pytest.mark.parametrize("filename,data", [
    ("report.csv", b"first,value\n" + b"a,1\n" * 22500 + b"last,42\n"),
    ("report.JSON", b'{"answer":42}'),
    ("empty.txt", b""),
    ("limit.txt", b"x" * (100 * 1024)),
], ids=["csv-90kb", "json", "empty", "exact-limit"])
def test_small_text_document_reaches_prompt_and_keeps_original_bytes(filename, data):
    from gateway.run_inbound import GatewayInboundMixin

    adapter = line.LineAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock(fetch_content=AsyncMock(return_value=data))
    adapter.handle_message = AsyncMock()
    asyncio.run(adapter._handle_message_event({
        "source": {"type": "group", "groupId": "Ctest", "userId": "Utest"},
        "message": {"type": "file", "id": "file-1", "fileName": filename},
    }))
    event = adapter.handle_message.await_args.args[0]
    assert Path(event.media_urls[0]).read_bytes() == data
    assert data.decode("utf-8") in event.text
    assert "[Content of " in event.text
    assert event.media_text_inlined == [True]
    prompt = GatewayInboundMixin._prepend_inbound_document_notes(event, event.text)
    assert event.text in prompt
    assert data.decode("utf-8") in prompt


@pytest.mark.parametrize("filename,data", [
    ("large.txt", b"x" * (100 * 1024 + 1)),
    ("invalid.csv", b"\xff\xfe"),
    ("binary.pdf", b"%PDF ascii but not text"),
    ("unreadable.txt", b"cached but unreadable"),
], ids=["over-limit", "invalid-utf8", "binary", "read-error"])
def test_non_inline_document_retains_path_without_promising_content(filename, data, monkeypatch):
    from gateway.run_inbound import GatewayInboundMixin

    if filename == "unreadable.txt":
        def unreadable(*args, **kwargs):
            raise PermissionError("fixture read refused")
        monkeypatch.setattr(line, "open", unreadable, raising=False)

    adapter = line.LineAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock(fetch_content=AsyncMock(return_value=data))
    adapter.handle_message = AsyncMock()
    asyncio.run(adapter._handle_message_event({
        "source": {"type": "group", "groupId": "Ctest", "userId": "Utest"},
        "message": {"type": "file", "id": "file-1", "fileName": filename},
    }))
    event = adapter.handle_message.await_args.args[0]
    assert Path(event.media_urls[0]).read_bytes() == data
    assert event.media_text_inlined == [False]
    assert event.text == "[file]"
    prompt = GatewayInboundMixin._prepend_inbound_document_notes(event, event.text)
    assert "content has been included" not in prompt
    assert "saved at:" in prompt
