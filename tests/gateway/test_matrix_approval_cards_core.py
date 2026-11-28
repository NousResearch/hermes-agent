"""Matrix approval cards against the real core approval contracts: Deny-only requests and approvers."""

import types
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig


def _adapter(monkeypatch, *, allowed="@user:example.org @other:example.org"):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", allowed.replace(" ", ","))
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = types.SimpleNamespace()
    adapter._user_id = "@bot:example.org"
    adapter._send_reaction = AsyncMock(return_value="$r")
    adapter._redact_bot_approval_reactions = AsyncMock()
    adapter.edit_message = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$edit"))
    return adapter


def _pending(session_key, command, **authority):
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry({"command": command})
    if authority:
        entry.authority = {"requester_id": "", "requester_name": "", "chat_type": "group", "approvers": ["*"],
                           **authority}
    with approval._lock:
        approval._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


def _reaction(target, key, sender="@user:example.org"):
    _reaction.n += 1  # distinct ids: the adapter drops a redelivered reaction event
    return types.SimpleNamespace(sender=sender, event_id=f"$react{_reaction.n}", room_id="!room:example.org",
                                 content={"m.relates_to": {"event_id": target, "key": key}})


_reaction.n = 0


_NO_SUMMARY = patch("plugins.platforms.matrix.approval_cards.load_matrix_approval_summary_config",
                    return_value=types.SimpleNamespace(enabled=False))


@pytest.mark.asyncio
async def test_card_that_shows_whole_command_lifts_deny_only_and_reaction_approves(monkeypatch):
    """Every core request is born Deny-only; the card (base send_exec_approval) lifts it, so ✅ approves."""
    from tools import approval

    adapter = _adapter(monkeypatch)
    adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$card"))
    entry = _pending("sess-lift", "rm -rf /tmp/lift")
    try:
        assert entry.card_deny_only is True
        with _NO_SUMMARY:
            result = await adapter.send_exec_approval(chat_id="!room:example.org", command="rm -rf /tmp/lift",
                                                      session_key="sess-lift", request_id=entry.data["request_id"])
        assert result.success is True
        assert entry.card_deny_only is False
        await adapter._on_reaction(_reaction("$card", "✅"))
        assert (entry.result, entry.event.is_set()) == ("once", True)
        assert "$card" not in adapter._approval_prompts_by_event
    finally:
        approval.clear_session("sess-lift")


@pytest.mark.asyncio
async def test_deny_only_request_gets_a_deny_only_card(monkeypatch):
    """A long command the base cannot send whole ahead of the card (it holds ```) stays Deny-only:
    the card seeds and advertises only Deny, and a hand-added ✅ approves nothing."""
    from tools import approval

    adapter = _adapter(monkeypatch)
    adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$card"))
    command = "echo ```" + " x" * 1500
    entry = _pending("sess-deny-only", command)
    try:
        with _NO_SUMMARY:
            result = await adapter.send_exec_approval(chat_id="!room:example.org", command=command,
                                                      session_key="sess-deny-only", request_id=entry.data["request_id"])
        assert result.success is True
        assert entry.card_deny_only is True
        body = adapter.send.await_args.args[1]
        assert "!approve" not in body
        assert "✅" not in body
        assert [call.args[2] for call in adapter._send_reaction.await_args_list] == ["❌"]
        await adapter._on_reaction(_reaction("$card", "✅"))
        assert (entry.result, entry.event.is_set()) == (None, False)
        assert "$card" in adapter._approval_prompts_by_event  # still live for ❌
        await adapter._on_reaction(_reaction("$card", "❌"))
        assert entry.result == "deny"
    finally:
        approval.clear_session("sess-deny-only")


@pytest.mark.asyncio
async def test_narrowed_approvers_refuse_another_members_reaction(monkeypatch):
    """With approvals.approvers narrowed, an admitted member who is not the requester cannot answer
    the card; the requester still can."""
    from tools import approval

    adapter = _adapter(monkeypatch)
    adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$card"))
    entry = _pending("sess-approvers", "rm -rf /tmp/owned", requester_id="@user:example.org",
                     requester_name="user", approvers=[])
    try:
        with _NO_SUMMARY:
            await adapter.send_exec_approval(chat_id="!room:example.org", command="rm -rf /tmp/owned",
                                             session_key="sess-approvers", request_id=entry.data["request_id"])
        adapter.send.reset_mock()
        await adapter._on_reaction(_reaction("$card", "✅", sender="@other:example.org"))
        assert (entry.result, entry.event.is_set()) == (None, False)
        prompt = adapter._approval_prompts_by_event["$card"]
        assert (prompt.resolved, prompt.terminal_choice) == (False, None)
        adapter.send.assert_awaited()  # the refusal notice
        await adapter._on_reaction(_reaction("$card", "✅"))
        assert entry.result == "once"
    finally:
        approval.clear_session("sess-approvers")
