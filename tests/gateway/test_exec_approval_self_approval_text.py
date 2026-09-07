"""``approvals.forbid_self_approval`` covers the text approval paths too: with the flag on (and no
``allow_admin_from`` policy), the sender whose turn raised the exec prompt may not resolve it with a
bare "yes" (``_route_plaintext_approval_while_busy``), a typed ``/approve`` or a ``/deny``; the prompt
stays pending for someone else, who resolves it with ``decided_by`` recorded. Flag off keeps today's
behavior. Only the Discord buttons enforced this before.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

REQUESTER, OTHER = "222", "111"
SESSION_KEY = "agent:main:discord:channel:chan-1"


def _event(text: str, user_id: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, message_id="m1",
        source=SessionSource(platform=Platform.DISCORD, user_id=user_id, chat_id="chan-1",
                             user_name=f"user-{user_id}", chat_type="channel"),
    )


@pytest.fixture
def flag(monkeypatch):
    import hermes_cli.config

    state = {"forbid_self_approval": True}
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: {"approvals": dict(state)})
    return state


@pytest.fixture
def runner(monkeypatch, flag):
    from gateway.run import GatewayRunner

    r = object.__new__(GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x", extra={})})
    adapter = MagicMock()
    adapter.SUPPORTS_NATIVE_STREAMING = False
    adapter._send_with_retry = AsyncMock(return_value=SimpleNamespace(success=True, message_id="r1"))
    adapter._unwrap_ephemeral = lambda x: (x, 0) if isinstance(x, str) else (None, 0)
    r.adapters = {Platform.DISCORD: adapter}
    r.session_store = None
    r._running_agents = {}
    r._pending_messages = {}
    r._pending_approvals = {}
    monkeypatch.setattr(r, "_session_key_for_source", lambda src: SESSION_KEY, raising=False)
    return r


@pytest.fixture
def queued_entry():
    import tools.approval as approval
    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry({"command": "rm -rf /tmp/x", "requester_user_id": REQUESTER,
                            "requester_platform": "discord"})
    with approval._lock:
        approval._gateway_queues[SESSION_KEY] = [entry]
    try:
        yield entry
    finally:
        with approval._lock:
            approval._gateway_queues.pop(SESSION_KEY, None)


def test_requester_bare_yes_does_not_resolve_own_prompt(runner, queued_entry):
    consumed = asyncio.run(runner._route_plaintext_approval_while_busy(_event("yes", REQUESTER), SESSION_KEY))
    assert consumed is True
    assert queued_entry.result is None, f"requester resolved own prompt via 'yes': decided_by={queued_entry.decided_by}"
    sent = runner.adapters[Platform.DISCORD]._send_with_retry.await_args
    assert sent is not None and "cannot approve their own request" in str(sent)


@pytest.mark.parametrize("text", ["/approve", "/approve all session", "/deny too risky"])
def test_requester_typed_command_does_not_resolve_own_prompt(runner, queued_entry, text):
    handler = runner._handle_approve_command if text.startswith("/approve") else runner._handle_deny_command
    reply = asyncio.run(handler(_event(text, REQUESTER)))
    assert queued_entry.result is None, f"requester resolved own prompt via {text!r}"
    assert "cannot approve their own request" in reply


def test_other_user_still_resolves_with_decided_by(runner, queued_entry):
    consumed = asyncio.run(runner._route_plaintext_approval_while_busy(_event("yes", OTHER), SESSION_KEY))
    assert consumed is True
    assert queued_entry.result == "once"
    assert queued_entry.decided_by == f"discord:{OTHER}"


def test_flag_off_keeps_requester_text_approval(runner, queued_entry, flag):
    flag["forbid_self_approval"] = False
    asyncio.run(runner._handle_approve_command(_event("/approve", REQUESTER)))
    assert queued_entry.result == "once"
    assert queued_entry.decided_by == f"discord:{REQUESTER}"


def test_refused_decision_leaves_queue_untouched():
    """The guard is checked under the queue lock before anything is popped."""
    import tools.approval as approval
    from tools.approval import SelfApprovalForbidden, resolve_gateway_approval
    from tools.approval_gateway_wait import _ApprovalEntry

    own = _ApprovalEntry({"command": "a", "requester_user_id": REQUESTER, "requester_platform": "discord"})
    other = _ApprovalEntry({"command": "b", "requester_user_id": OTHER, "requester_platform": "discord"})
    key = "self-approval-queue"
    with approval._lock:
        approval._gateway_queues[key] = [own, other]
    try:
        with pytest.raises(SelfApprovalForbidden):
            resolve_gateway_approval(key, "once", refuse_requester=("discord", REQUESTER))
        with pytest.raises(SelfApprovalForbidden):  # "all" targets the requester's own entry too
            resolve_gateway_approval(key, "once", resolve_all=True, refuse_requester=("discord", REQUESTER))
        assert approval._gateway_queues[key] == [own, other]
        assert own.result is None and other.result is None
        # A decider who raised none of the targeted prompts resolves normally.
        assert resolve_gateway_approval(key, "once", refuse_requester=("discord", OTHER)) == 1
        assert own.result == "once" and other.result is None
    finally:
        with approval._lock:
            approval._gateway_queues.pop(key, None)
