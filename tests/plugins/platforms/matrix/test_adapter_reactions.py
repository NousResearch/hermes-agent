"""Inbound ``m.reaction`` events reach the gateway reaction-hook contract.

``_on_reaction`` used to serve only the approval / model-picker / choice-picker prompts, so a
gateway hook (or plugin) registered for ``reaction:added`` never fired on Matrix — the Slack
adapter already forwarded. These tests pin the normalised payload, the ordering (hooks see every
human reaction, including one that a prompt also consumes) and the error containment.

No homeserver connection is made: ``_on_reaction`` is fed a mautrix-shaped event directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from gateway.config import PlatformConfig

_BOT = "@bot:example.org"
_ROOM = "!room:example.org"
_MSG = "$msg1"
_REACTION = "$reaction1"
_THUMBS_UP = "\U0001f44d"


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": _BOT}))


class _ReactionEvent:
    """Minimal stand-in for ``mautrix.event.ReactionEvent``."""

    def __init__(
        self,
        content: dict[str, Any],
        sender: str = "@alice:example.org",
        event_id: str = _REACTION,
        room_id: str = _ROOM,
    ) -> None:
        self.content = content
        self.sender = sender
        self.event_id = event_id
        self.room_id = room_id


def _content(key: str = _THUMBS_UP, target: str = _MSG) -> dict[str, Any]:
    return {"m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": key}}


def _capture(adapter) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def handler(payload: dict[str, Any]) -> None:
        seen.append(payload)

    adapter.set_reaction_handler(handler)
    return seen


@pytest.mark.asyncio
async def test_reaction_is_forwarded_as_reaction_added():
    adapter = _make_adapter()
    seen = _capture(adapter)

    await adapter._on_reaction(_ReactionEvent(_content()))

    assert len(seen) == 1
    payload = seen[0]
    assert payload["platform"] == "matrix"
    assert payload["event_name"] == "reaction:added"
    assert payload["reaction"] == _THUMBS_UP
    assert payload["user_id"] == "@alice:example.org"
    assert payload["item_type"] == "message"
    assert payload["channel_id"] == _ROOM
    assert payload["message_ts"] == _MSG      # the reacted-to message
    assert payload["event_ts"] == _REACTION   # the reaction event itself
    assert payload["raw_event"] == _content()


@pytest.mark.asyncio
async def test_object_style_relates_to_is_forwarded_too():
    """mautrix may hand over a RelatesTo object rather than the raw dict."""

    class _RelatesTo:
        event_id = _MSG
        key = _THUMBS_UP

    class _Content:
        relates_to = _RelatesTo()

    adapter = _make_adapter()
    seen = _capture(adapter)

    await adapter._on_reaction(_ReactionEvent(_Content()))  # type: ignore[arg-type]

    assert [p["reaction"] for p in seen] == [_THUMBS_UP]
    assert seen[0]["message_ts"] == _MSG


@pytest.mark.asyncio
async def test_bot_own_reaction_is_not_forwarded():
    """Our ✅/❌/👀 lifecycle reactions must not loop back as user input."""
    adapter = _make_adapter()
    seen = _capture(adapter)

    await adapter._on_reaction(_ReactionEvent(_content(), sender=_BOT))

    assert seen == []


@pytest.mark.asyncio
async def test_duplicate_reaction_event_is_forwarded_once():
    adapter = _make_adapter()
    seen = _capture(adapter)
    event = _ReactionEvent(_content())

    await adapter._on_reaction(event)
    await adapter._on_reaction(event)

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_reaction_without_target_or_key_is_not_forwarded():
    adapter = _make_adapter()
    seen = _capture(adapter)

    await adapter._on_reaction(_ReactionEvent({"m.relates_to": {"event_id": "", "key": ""}}))
    await adapter._on_reaction(_ReactionEvent({}))

    assert seen == []


@pytest.mark.asyncio
async def test_no_handler_installed_is_harmless():
    adapter = _make_adapter()
    adapter.set_reaction_handler(None)

    await adapter._on_reaction(_ReactionEvent(_content()))  # must not raise


@pytest.mark.asyncio
async def test_hook_error_does_not_block_the_prompt_path():
    """A raising consumer must not swallow the reaction for the approval/picker handlers."""
    adapter = _make_adapter()
    handled: list[tuple] = []

    async def boom(payload: dict[str, Any]) -> None:
        raise RuntimeError("consumer exploded")

    adapter.set_reaction_handler(boom)

    async def fake_approval(room_id, reacts_to, key, sender):
        handled.append((room_id, reacts_to, key, sender))
        return True

    adapter._handle_approval_reaction = fake_approval

    await adapter._on_reaction(_ReactionEvent(_content()))

    assert handled == [(_ROOM, _MSG, _THUMBS_UP, "@alice:example.org")]


@pytest.mark.asyncio
async def test_hook_sees_reaction_before_a_prompt_consumes_it():
    """Hooks fire before the internal gates (Slack contract): an approval ✅ also reaches hooks."""
    adapter = _make_adapter()
    seen = _capture(adapter)
    order: list[str] = []

    async def handler(payload: dict[str, Any]) -> None:
        seen.append(payload)
        order.append("hook")

    adapter.set_reaction_handler(handler)

    async def fake_approval(room_id, reacts_to, key, sender):
        order.append("prompt")
        return True

    adapter._handle_approval_reaction = fake_approval

    await adapter._on_reaction(_ReactionEvent(_content(key="\u2705")))

    assert len(seen) == 1
    assert seen[0]["reaction"] == "\u2705"
    assert order == ["hook", "prompt"]
