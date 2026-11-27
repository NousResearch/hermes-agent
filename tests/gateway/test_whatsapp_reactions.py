"""WhatsApp processing reactions (``whatsapp.reactions`` in config.yaml): 👀 while the agent works, then
✅ / ❌, cleared on cancel; off unless enabled, like Telegram's."""

from contextlib import asynccontextmanager

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource

CHAT, MESSAGE, SENDER = "120363001234567890@g.us", "M1", "900000000000001@lid"


def _adapter(platform_config=None, **extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter(platform_config or PlatformConfig(enabled=True, extra=extra))


def _event():
    source = SessionSource(platform=Platform.WHATSAPP, chat_id=CHAT, chat_type="group", user_id=SENDER,
                           user_name="Alice")
    return MessageEvent(text="hi", source=source, message_id=MESSAGE,
                        raw_message={"chatId": CHAT, "messageId": MESSAGE, "senderId": SENDER})


def _recorded_reactions(adapter, monkeypatch, *, delivered=True):
    sent = []

    async def send_reaction(chat_id, message_id, participant, emoji):
        sent.append((chat_id, message_id, participant, emoji))
        return delivered

    monkeypatch.setattr(adapter, "_send_reaction", send_reaction)
    return sent


async def _process(adapter, outcome):
    event = _event()
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)


@pytest.mark.asyncio
async def test_reactions_are_off_unless_enabled(monkeypatch):
    adapter = _adapter()
    sent = _recorded_reactions(adapter, monkeypatch)
    await _process(adapter, ProcessingOutcome.SUCCESS)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, final", [
    (ProcessingOutcome.SUCCESS, "✅"), (ProcessingOutcome.FAILURE, "❌"), (ProcessingOutcome.CANCELLED, "")])
async def test_the_eyes_are_swapped_for_the_outcome(monkeypatch, outcome, final):
    adapter = _adapter(reactions=True)
    sent = _recorded_reactions(adapter, monkeypatch)
    await _process(adapter, outcome)
    assert sent == [(CHAT, MESSAGE, SENDER, "👀"), (CHAT, MESSAGE, SENDER, final)]


@pytest.mark.asyncio
async def test_no_outcome_reaction_when_the_eyes_never_went_out(monkeypatch):
    adapter = _adapter(reactions=True)
    sent = _recorded_reactions(adapter, monkeypatch, delivered=False)
    await _process(adapter, ProcessingOutcome.SUCCESS)
    assert [reaction[-1] for reaction in sent] == ["👀"]


@pytest.mark.asyncio
async def test_a_reaction_posts_the_bridge_payload(monkeypatch):
    """``/send-reaction`` (``scripts/whatsapp-bridge/bridge.js``) keys a group reaction on the sender."""
    adapter = _adapter(reactions=True)
    posted = []

    async def bridge_ready():
        return False

    @asynccontextmanager
    async def bridge_req(method, path, timeout, **kwargs):
        posted.append((method, path, kwargs["json"]))
        yield type("Response", (), {"status": 200})()

    monkeypatch.setattr(adapter, "_bridge_unavailable", bridge_ready)
    monkeypatch.setattr(adapter, "_bridge_req", bridge_req)
    assert await adapter._send_reaction(CHAT, MESSAGE, SENDER, "👀") is True
    assert posted == [("post", "send-reaction",
                       {"chatId": CHAT, "messageId": MESSAGE, "participant": SENDER, "emoji": "👀"})]


def test_config_yaml_turns_reactions_on(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("whatsapp:\n  enabled: true\n  reactions: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()
    assert _adapter(config.platforms[Platform.WHATSAPP])._reactions_enabled() is True
