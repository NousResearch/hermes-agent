"""Discord component views enforce the prompt owner (#129925 review).

``InteractionOwner.capture`` keeps three independent location anchors (``chat_id``,
``channel_id`` from ``parent_chat_id``, ``thread_id``). These tests drive the real
``ExecApprovalView`` / ``ClarifyChoiceView`` gates with the metadata shape the gateway
produces for Discord threads, so deleting or weakening either gate fails here."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Triggers the shared discord mock from tests/gateway/conftest.py.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    ExecApprovalView,
)

OWNER = "42"
THREAD = "THREAD-9"
PARENT = "CHAN-1"
PROMPT = "PROMPT-7"


def _thread_metadata(*, with_parent: bool) -> dict:
    meta = {"user_id": OWNER, "thread_id": THREAD, "message_id": "SRC-1"}
    if with_parent:
        # source_route_metadata injects this for Discord threads/forums.
        meta["parent_chat_id"] = PARENT
    return meta


def _interaction(*, user_id=OWNER, channel_id=THREAD, parent_id=PARENT, message_id=PROMPT):
    embed = MagicMock()
    embed.color = None
    channel = SimpleNamespace(id=channel_id, parent_id=parent_id)
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="Tester", roles=[]),
        channel_id=channel_id,
        channel=channel,
        message=SimpleNamespace(id=message_id, embeds=[embed], channel=channel),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock(), defer=AsyncMock()),
    )


def _approval_view(metadata):
    view = ExecApprovalView(
        session_key="sk-1", allowed_user_ids={OWNER, "77"},
        owner_chat_id=THREAD, owner_metadata=metadata,
    )
    view._message = SimpleNamespace(id=PROMPT)
    return view


def _clarify_view(metadata):
    view = ClarifyChoiceView(
        choices=["yes", "no"], clarify_id="cid-owner", allowed_user_ids={OWNER, "77"},
        owner_chat_id=THREAD, owner_metadata=metadata,
    )
    view._message = SimpleNamespace(id=PROMPT)
    return view


@pytest.mark.parametrize("make_view", [_approval_view, _clarify_view], ids=["approval", "clarify"])
@pytest.mark.parametrize("with_parent", [False, True], ids=["thread", "thread+parent_chat_id"])
def test_owner_click_inside_their_thread_is_accepted(make_view, with_parent):
    """Case 3 from the review: chat=THREAD, channel=CHAN-1 (parent), thread=THREAD. The
    owner's own click in that thread must pass; it was rejected when one id was compared
    against all three anchors."""
    view = make_view(_thread_metadata(with_parent=with_parent))
    assert view._owner_accepts(_interaction()) is True


@pytest.mark.parametrize("make_view", [_approval_view, _clarify_view], ids=["approval", "clarify"])
@pytest.mark.parametrize(
    "foreign",
    [
        {"user_id": "77"},                      # another allowlisted user
        {"message_id": "PROMPT-OTHER"},        # a different prompt message
        {"channel_id": "THREAD-OTHER"},        # a different thread under the same parent
        {"parent_id": "CHAN-OTHER"},           # the same thread id under another parent
    ],
    ids=["actor", "prompt", "thread", "parent"],
)
def test_foreign_anchor_is_rejected(make_view, foreign):
    view = make_view(_thread_metadata(with_parent=True))
    assert view._owner_accepts(_interaction(**foreign)) is False


@pytest.mark.asyncio
async def test_approval_gate_blocks_another_allowlisted_user_and_lets_the_owner_resolve(monkeypatch):
    resolved = []
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval",
        lambda session_key, choice: resolved.append((session_key, choice)) or 1,
    )
    view = _approval_view(_thread_metadata(with_parent=True))
    view._finalize_embed = AsyncMock()

    intruder = _interaction(user_id="77")
    await view._resolve(intruder, "once", MagicMock(), "platform.discord.approval.resolved_once")
    intruder.response.send_message.assert_awaited_once()
    assert intruder.response.send_message.call_args.kwargs.get("ephemeral") is True
    assert resolved == [] and view.resolved is False

    await view._resolve(_interaction(), "once", MagicMock(), "platform.discord.approval.resolved_once")
    assert resolved == [("sk-1", "once")]


@pytest.mark.asyncio
async def test_clarify_gate_covers_choices_and_the_other_button():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
    cm.register("cid-owner", "sk-owner", "Pick", ["yes", "no"])
    view = _clarify_view(_thread_metadata(with_parent=True))
    view._finish = AsyncMock()

    intruder = _interaction(user_id="77")
    await view._resolve_choice(intruder, index=0, choice="yes")
    await view._on_other(intruder)
    assert intruder.response.send_message.await_count == 2
    assert cm._entries["cid-owner"].awaiting_text is False
    assert not cm._entries["cid-owner"].event.is_set()

    await view._resolve_choice(_interaction(), index=0, choice="yes")
    assert cm._entries["cid-owner"].event.is_set()
