"""``/setup-files`` is handled inside the Google Chat adapter, before the runner's allowlist
check. It must only run for a sender that check accepts; anyone else goes on to
``handle_message`` so the runner's normal deny or pairing path answers them."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

ALLOWED = "alice@example.com"


def _adapter(auth_check):
    from plugins.platforms.google_chat.adapter import GoogleChatAdapter

    adapter = GoogleChatAdapter(PlatformConfig(enabled=True, extra={"project_id": "p"}))
    adapter.replies = []
    adapter._create_message = AsyncMock(side_effect=lambda _chat, body: adapter.replies.append(body["text"]))
    adapter.handle_message = AsyncMock()
    adapter.set_authorization_check(auth_check)
    return adapter


def _allowlist(user_id, _chat_type=None, _chat_id=None, **_kwargs):
    return user_id == ALLOWED


def _dm(text, email):
    space = {"name": "spaces/S", "spaceType": "DIRECT_MESSAGE"}
    msg = {
        "name": "spaces/S/messages/M",
        "sender": {"name": "users/1", "email": email, "displayName": "User", "type": "HUMAN"},
        "text": text,
        "argumentText": text,
        "thread": {"name": "spaces/S/threads/T"},
        "space": space,
    }
    return msg, {"space": space}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/setup-files", "/setup-files start", "/setup-files revoke"])
async def test_sender_outside_the_allowlist_is_not_served_by_setup_files(text):
    adapter = _adapter(_allowlist)

    await adapter._dispatch_message(*_dm(text, "mallory@example.com"))

    assert adapter.replies == []
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].source.user_id == "mallory@example.com"


@pytest.mark.asyncio
async def test_allowlisted_sender_still_gets_setup_files():
    adapter = _adapter(_allowlist)

    await adapter._dispatch_message(*_dm("/setup-files", ALLOWED))

    assert len(adapter.replies) == 1
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_files_is_not_served_when_no_authorization_check_is_wired():
    adapter = _adapter(None)

    await adapter._dispatch_message(*_dm("/setup-files", ALLOWED))

    assert adapter.replies == []
    adapter.handle_message.assert_awaited_once()
