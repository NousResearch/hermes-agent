"""A reply over the split cap keeps the flood guard and carries its full text as a Markdown file,
so nothing past the cap is lost in the channel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_split_cap import CAP, _huge_content, _make_adapter
import plugins.platforms.discord.adapter as adapter_module


class _File:
    def __init__(self, stream, filename):
        self.data = stream.read().decode("utf-8")
        self.filename = filename


@pytest.mark.asyncio
async def test_capped_reply_attaches_the_full_text(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module.discord, "File", _File)
    adapter = _make_adapter()
    sends = []

    async def fake_send(*, content, reference=None, file=None):
        sends.append((content, file))
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _cid: channel, fetch_channel=AsyncMock())
    content = _huge_content()

    result = await adapter.send("555", content)

    assert result.success is True
    assert len(sends) == CAP
    assert all(file is None for _, file in sends[:-1])
    notice, attached = sends[-1]
    assert "attached as full-reply.md" in notice
    assert attached.filename == "full-reply.md"
    assert attached.data == content


@pytest.mark.asyncio
async def test_reply_under_the_cap_has_no_attachment(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module.discord, "File", _File)
    adapter = _make_adapter()
    sends = []

    async def fake_send(*, content, reference=None, file=None):
        sends.append(file)
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _cid: channel, fetch_channel=AsyncMock())

    result = await adapter.send("555", "short reply " * 300)

    assert result.success is True
    assert 1 < len(sends) <= CAP
    assert sends == [None] * len(sends)


@pytest.mark.asyncio
async def test_refused_attachment_still_ends_with_the_truncation_notice(monkeypatch, tmp_path):
    """No Attach Files permission (or an upload limit) must not lose the final notice."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module.discord, "File", _File)
    adapter = _make_adapter()
    sends = []

    async def fake_send(*, content, reference=None, file=None):
        if file is not None:
            raise RuntimeError("403 Forbidden (error code: 50013): Missing Permissions")
        sends.append(content)
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _cid: channel, fetch_channel=AsyncMock())

    result = await adapter.send("555", _huge_content())

    assert result.success is True
    assert len(sends) == CAP
    assert "attached" not in sends[-1]
    assert "characters were not delivered" in sends[-1]


@pytest.mark.asyncio
async def test_edit_overflow_attaches_the_full_text_to_the_last_continuation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module.discord, "File", _File)
    adapter = _make_adapter()
    files = []

    async def fake_send(*, content, reference=None, file=None):
        files.append(file)
        return SimpleNamespace(id=7000 + len(files))

    channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
    msg = SimpleNamespace(id=42, edit=AsyncMock())
    content = _huge_content()

    result = await adapter._edit_overflow_split(channel, msg, "42", content)

    assert result.success is True
    assert all(f is None for f in files[:-1])
    assert files[-1].filename == "full-reply.md" and files[-1].data == content
