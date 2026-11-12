"""BlueBubbles paragraph-per-bubble splitting is configurable (#150762)."""
import pytest

from gateway.config import PlatformConfig

BODY = "Daily Status\n\nQuip of the Day\nA line.\n\nDad Joke\nAnother line."


def _adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    adapter = BlueBubblesAdapter(PlatformConfig(
        enabled=True, extra={"server_url": "http://localhost:1234", "password": "secret", **extra}))
    sent = []

    async def fake_resolve(chat_id):
        return "iMessage;-;+15550001111"

    async def fake_api_post(path, payload):
        sent.append(payload["message"])
        return {"status": 200, "data": {"guid": f"g{len(sent)}"}}

    monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)
    monkeypatch.setattr(adapter, "_api_post", fake_api_post)
    return adapter, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, "false"])
async def test_split_paragraphs_off_sends_one_bubble(monkeypatch, value):
    adapter, sent = _adapter(monkeypatch, split_paragraphs=value)
    result = await adapter.send("+15550001111", BODY)
    assert result.success
    assert sent == [BODY]


@pytest.mark.asyncio
async def test_split_paragraphs_default_keeps_one_bubble_per_paragraph(monkeypatch):
    adapter, sent = _adapter(monkeypatch)
    await adapter.send("+15550001111", BODY)
    assert sent == ["Daily Status", "Quip of the Day\nA line.", "Dad Joke\nAnother line."]


@pytest.mark.asyncio
async def test_split_paragraphs_off_still_truncates_oversized_body(monkeypatch):
    adapter, sent = _adapter(monkeypatch, split_paragraphs=False)
    body = "\n\n".join(["x" * 1500] * 4)  # 6,006 chars > MAX_MESSAGE_LENGTH (4000)
    await adapter.send("+15550001111", body)
    assert len(sent) == 2 and all(len(c) <= adapter.MAX_MESSAGE_LENGTH for c in sent)
