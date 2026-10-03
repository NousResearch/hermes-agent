"""Matrix has no client-readable per-user bot flag, so bot-authored senders are declared via
``MATRIX_BOT_USERS`` / ``bot_users`` and must reach ``SessionSource.is_bot``. The gateway's
BotLoopGuard keys off that flag; without it two Hermes bots in one room ping-pong forever.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig


def _adapter(extra=None, env=None, monkeypatch=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    for name in ("MATRIX_BOT_USERS",):
        monkeypatch.delenv(name, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return MatrixAdapter(PlatformConfig(enabled=True, extra=extra or {}))


@pytest.mark.parametrize("extra, env, expected", [
    ({}, {}, set()),
    ({}, {"MATRIX_BOT_USERS": "@iris:example.org,@librarian:example.org"},
     {"@iris:example.org", "@librarian:example.org"}),
    ({"bot_users": ["@forge:example.org"]}, {}, {"@forge:example.org"}),
    # Blank YAML falls through to env, same precedence rule as the other list keys.
    ({"bot_users": "  "}, {"MATRIX_BOT_USERS": "@cos:example.org"}, {"@cos:example.org"}),
])
def test_bot_users_resolution(extra, env, expected, monkeypatch):
    assert _adapter(extra, env, monkeypatch)._bot_users == expected


def _resolve(adapter, sender):
    """Drive the real gating path with only the network-touching helpers stubbed."""
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
    return asyncio.run(adapter._resolve_message_context(
        "!r:example.org", sender, "$e1", "hello", {}, {}))


def test_declared_bot_sender_sets_is_bot(monkeypatch):
    adapter = _adapter({}, {"MATRIX_BOT_USERS": "@librarian:example.org"}, monkeypatch)
    ctx = _resolve(adapter, "@librarian:example.org")
    assert ctx is not None
    assert ctx[5].is_bot is True


def test_human_sender_is_not_a_bot(monkeypatch):
    adapter = _adapter({}, {"MATRIX_BOT_USERS": "@librarian:example.org"}, monkeypatch)
    ctx = _resolve(adapter, "@jon:example.org")
    assert ctx is not None
    assert ctx[5].is_bot is False


def test_unconfigured_list_leaves_is_bot_false(monkeypatch):
    """No list configured => nobody is a bot, i.e. today's behaviour is unchanged."""
    adapter = _adapter({}, {}, monkeypatch)
    ctx = _resolve(adapter, "@librarian:example.org")
    assert ctx is not None
    assert ctx[5].is_bot is False
