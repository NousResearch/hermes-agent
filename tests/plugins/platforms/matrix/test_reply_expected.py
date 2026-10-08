"""Matrix ``reply_expected``.

A bot-authored message that does not address this bot must be answerable with silence:
``MessageEvent.reply_expected is False`` lets a correct ``NO_REPLY`` stand, so a bot-to-bot
exchange ends instead of being rewritten into a reply that demands another reply. A bot that
*does* address us (mention or command) still expects an answer, and human senders keep the
unset (``None``) default so their messages fall back to the gateway's display-kind heuristic.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig

BOT = "@librarian:example.org"
SELF = "@iris:example.org"
HUMAN = "@jon:example.org"


def _adapter(extra=None, env=None, monkeypatch=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    for name in ("MATRIX_BOT_USERS", "MATRIX_USER_ID"):
        monkeypatch.delenv(name, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return MatrixAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def _stub_network(adapter):
    async def fake_identity(_room_id):
        from plugins.platforms.matrix.adapter import MatrixRoomIdentity
        return MatrixRoomIdentity(
            room_id="!r:example.org", room_name="A & B", room_topic=None, canonical_alias=None,
            server_name="example.org", joined_member_count=2, is_direct_account_data=True,
            display_name="A & B", has_explicit_name=True, chat_type="dm", conflict=False)

    async def fake_dm(_room_id):
        return True

    async def fake_name(_room_id, _sender):
        return "Somebody"

    adapter._resolve_room_identity = fake_identity
    adapter._is_dm_room = fake_dm
    adapter._get_display_name = fake_name
    adapter._background_read_receipt = lambda *a, **k: None


def _inbound(adapter, sender, body="hello", content=None):
    """Drive the real inbound path (gate -> MessageEvent) with only network helpers stubbed."""
    _stub_network(adapter)
    return asyncio.run(adapter._build_inbound_event(
        "!r:example.org", sender, "$e1", body, content or {}, {}))


def _bot_adapter(monkeypatch, extra=None):
    cfg = {"user_id": SELF, "bot_users": [BOT]}
    cfg.update(extra or {})
    return _adapter(cfg, {}, monkeypatch)


def test_unmentioned_bot_message_allows_silence(monkeypatch):
    event = _inbound(_bot_adapter(monkeypatch), BOT)
    assert event is not None
    assert event.reply_expected is False


def test_silence_marker_stands_for_unmentioned_bot(monkeypatch):
    """The point of the flag: a correct NO_REPLY is delivered as silence, not rewritten."""
    from gateway.response_filters import display_kind_for_event, silence_allowed

    event = _inbound(_bot_adapter(monkeypatch), BOT)
    assert event is not None
    assert silence_allowed(display_kind_for_event(event), event.reply_expected) is True


def test_silence_marker_still_falls_back_for_human_sender(monkeypatch):
    from gateway.response_filters import display_kind_for_event, silence_allowed

    event = _inbound(_bot_adapter(monkeypatch), HUMAN)
    assert event is not None
    assert silence_allowed(display_kind_for_event(event), event.reply_expected) is False


def test_mentioned_bot_message_expects_a_reply(monkeypatch):
    event = _inbound(_bot_adapter(monkeypatch), BOT, body=f"{SELF} ping")
    assert event is not None
    assert event.reply_expected is True


def test_bot_command_expects_a_reply(monkeypatch):
    event = _inbound(_bot_adapter(monkeypatch), BOT, body="/status")
    assert event is not None
    assert event.reply_expected is True


def test_human_sender_leaves_reply_expected_unset(monkeypatch):
    event = _inbound(_bot_adapter(monkeypatch), HUMAN)
    assert event is not None
    assert event.reply_expected is None


def test_undeclared_sender_leaves_reply_expected_unset(monkeypatch):
    """With no bot declared, nobody is a bot: today's behaviour is unchanged."""
    adapter = _adapter({"user_id": SELF}, {}, monkeypatch)
    event = _inbound(adapter, BOT)
    assert event is not None
    assert event.reply_expected is None


def test_declared_bot_marks_source_as_bot(monkeypatch):
    """The same declaration feeds ``SessionSource.is_bot`` (BotLoopGuard / allow_bots gate)."""
    event = _inbound(_bot_adapter(monkeypatch), BOT)
    assert event is not None
    assert event.source.is_bot is True


@pytest.mark.parametrize("is_bot_sender, mentioned, command, expected", [
    (False, False, False, None),
    (False, True, False, None),
    (True, False, False, False),
    (True, True, False, True),
    (True, False, True, True),
])
def test_reply_expectation_matrix(is_bot_sender, mentioned, command, expected, monkeypatch):
    adapter = _adapter({"user_id": SELF}, {}, monkeypatch)
    assert adapter._matrix_reply_expected(is_bot_sender, mentioned, command) is expected
