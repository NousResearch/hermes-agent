"""Approval card contract tests.

Locks down the exec-approval card flow so SDK changes can't silently break
it. SDK ``channel.on("cardAction", ...)`` drives ``_on_sdk_card_action``,
which delegates approval handling to ``_resolve_approval_via_sdk`` and
non-approval card clicks to a synthetic COMMAND through
``_to_command_event_from_card_action`` + ``handle_message``.

Locked behaviors:
- send_exec_approval renders an interactive card with FOUR buttons
  (approve_once / approve_session / approve_always / deny). All four
  share the same approval_id in their value dict.
- _resolve_approval_via_sdk pops _approval_state[approval_id] and calls
  tools.approval.resolve_gateway_approval(session_key, choice).
- _on_sdk_card_action with no hermes_action synthesizes a COMMAND
  MessageEvent that downstream handle_message captures.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("lark_oapi.channel")

from .conftest import _drain_adapter_tasks

from lark_oapi.channel.types import (
    CardActionEvent,
    CardActionPayload,
    EventOperator,
)


def test_send_exec_approval_renders_three_action_buttons(adapter_harness):
    """The card has FOUR buttons (the test name is historical — production
    actually emits approve_once/approve_session/approve_always/deny)."""
    async def _send():
        return await adapter_harness.adapter.send_exec_approval(
            chat_id="oc_test",
            command="rm -rf /",
            session_key="sess_1",
            description="dangerous deletion",
        )
    asyncio.run(_send())
    assert len(adapter_harness.captured_sends) == 1
    sent = adapter_harness.captured_sends[0]
    assert sent.body.get("msg_type") == "interactive"
    # Production calls json.dumps(card) before passing it to the SDK, so
    # the captured `content` is a string here.
    content = sent.body["content"]
    card = json.loads(content) if isinstance(content, str) else content
    actions = card["elements"][1]["actions"]
    # Production emits 4 buttons: approve_once, approve_session,
    # approve_always, deny (see approvals._send_exec_approval_impl).
    assert len(actions) == 4, (
        "Approval card has 4 buttons (allow once / session / always / deny)"
    )
    hermes_actions = [a["value"]["hermes_action"] for a in actions]
    assert hermes_actions == [
        "approve_once", "approve_session", "approve_always", "deny",
    ]
    # All buttons must carry the same approval_id (so any click resolves
    # the same pending approval).
    approval_ids = {a["value"]["approval_id"] for a in actions}
    assert len(approval_ids) == 1


def test_send_update_prompt_renders_buttons_and_targets_thread(adapter_harness):
    """Gateway update prompts use a Feishu card and preserve topic routing."""
    async def _send():
        return await adapter_harness.adapter.send_update_prompt(
            chat_id="oc_test",
            prompt="Restore stashed changes after update?",
            default="y",
            session_key="sess_update",
            metadata={"thread_id": "omt_update_topic"},
        )

    result = asyncio.run(_send())

    assert result.success is True
    assert adapter_harness.adapter._update_prompt_state
    sent = adapter_harness.captured_sends[0]
    assert sent.body["receive_id"] == "omt_update_topic"
    assert sent.extra.get("receive_id_type") == "thread_id"
    assert sent.body["msg_type"] == "interactive"
    card = json.loads(sent.body["content"])
    assert "Restore stashed changes after update?" in card["elements"][0]["content"]
    assert "Default: `y`" in card["elements"][0]["content"]
    actions = card["elements"][1]["actions"]
    assert [a["value"]["hermes_update_prompt_action"] for a in actions] == ["y", "n"]


def test_card_action_with_hermes_action_resolves_approval(adapter_harness):
    """When _resolve_approval_via_sdk runs, it must call
    tools.approval.resolve_gateway_approval(session_key, choice).

    Routed through SDK ``CardActionEvent`` → ``_on_sdk_card_action`` →
    ``_resolve_approval_via_sdk``, which translates ``approve_once`` →
    ``once`` via ``_APPROVAL_CHOICE_MAP`` before calling resolve.
    """
    asyncio.run(adapter_harness.adapter.send_exec_approval(
        chat_id="oc_test", command="ls", session_key="sess_x"
    ))
    assert adapter_harness.adapter._approval_state, (
        "send_exec_approval did not populate _approval_state"
    )
    approval_id = next(iter(adapter_harness.adapter._approval_state.keys()))

    sdk_action = CardActionEvent(
        message_id="om_card_a",
        chat_id="oc_test",
        operator=EventOperator(open_id="ou_admin", name="Admin"),
        action=CardActionPayload(
            value={"hermes_action": "approve_once", "approval_id": approval_id},
            tag="button",
        ),
        raw={},
    )

    with patch("tools.approval.resolve_gateway_approval") as resolve_mock:
        resolve_mock.return_value = 1
        asyncio.run(adapter_harness.adapter._resolve_approval_via_sdk(
            sdk_action, "approve_once",
        ))

    resolve_mock.assert_called_once()
    args = resolve_mock.call_args.args
    assert args[0] == "sess_x", f"session_key must pass through; got args={args}"
    # _resolve_approval_via_sdk translates via _APPROVAL_CHOICE_MAP: approve_once → once
    assert args[1] == "once", f"choice must be _APPROVAL_CHOICE_MAP'd; got args={args}"


def test_card_action_with_update_prompt_writes_response_and_updates_card(
    adapter_harness, tmp_path, monkeypatch,
):
    """Clicking an update prompt button writes .update_response and updates the card."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    asyncio.run(adapter_harness.adapter.send_update_prompt(
        chat_id="oc_test",
        prompt="Continue update?",
        session_key="sess_update",
    ))
    prompt_id = next(iter(adapter_harness.adapter._update_prompt_state.keys()))

    sdk_action = CardActionEvent(
        message_id="om_update_card",
        chat_id="oc_test",
        operator=EventOperator(open_id="ou_admin", name="Admin"),
        action=CardActionPayload(
            value={
                "hermes_update_prompt_action": "y",
                "update_prompt_id": prompt_id,
            },
            tag="button",
        ),
        raw={},
    )

    asyncio.run(adapter_harness.adapter._on_sdk_card_action(sdk_action))

    assert (hermes_home / ".update_response").read_text() == "y"
    assert prompt_id not in adapter_harness.adapter._update_prompt_state
    updates = [
        item for item in adapter_harness.captured_sends
        if item.endpoint == "im.v1.message.card.update"
    ]
    assert updates
    assert updates[-1].body["card"]["header"]["template"] == "green"


@pytest.mark.asyncio
async def test_card_action_without_hermes_action_routes_as_command(adapter_harness):
    """Generic card action (no hermes_action) is delegated to handle_message
    via ``_on_sdk_card_action`` synthesizing a COMMAND MessageEvent."""
    sdk_action = CardActionEvent(
        message_id="om_card_x",
        chat_id="oc_team",
        operator=EventOperator(open_id="ou_alice", user_id="u_alice", name="Alice"),
        action=CardActionPayload(value={"command": "/status"}, tag="button"),
        raw={},
    )
    await adapter_harness.adapter._on_sdk_card_action(sdk_action)
    await _drain_adapter_tasks()

    # _to_command_event_from_card_action emits a synthetic COMMAND with the
    # `/card <tag> <json.dumps(value)>` text shape; handle_message → captured.
    assert adapter_harness.captured_inbound, (
        "Non-approval card actions must synthesize a COMMAND MessageEvent"
    )
    captured_texts = [e.text or "" for e in adapter_harness.captured_inbound]
    assert any("/card" in t for t in captured_texts), (
        f"Expected /card-prefixed COMMAND text; got {captured_texts!r}"
    )


class TestApprovalStatePopOrdering:
    """When resolve_gateway_approval raises, the approval state must NOT be
    popped -- otherwise a transient failure leaves the user with a stuck card
    and no way to retry."""

    @pytest.mark.asyncio
    async def test_state_remains_when_resolve_raises(self, monkeypatch):
        from gateway.platforms.feishu.approvals import _resolve_approval_via_sdk_impl

        approval_id = 42
        adapter = SimpleNamespace(
            _approval_state={approval_id: {
                "session_key": "sess-x",
                "message_id": "om_x",
                "chat_id": "oc_x",
            }},
            _channel=None,
        )

        def raising_resolve(session_key, choice):
            raise RuntimeError("transient failure")

        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval", raising_resolve,
        )

        action = SimpleNamespace(
            action=SimpleNamespace(value={"approval_id": approval_id}),
            operator=SimpleNamespace(open_id="ou_user", name="User"),
            message_id="om_x",
        )

        await _resolve_approval_via_sdk_impl(adapter, action, "approve_once")

        assert approval_id in adapter._approval_state, (
            "approval state must remain when resolve fails, so user/agent can retry"
        )


@pytest.mark.asyncio
async def test_p2p_card_action_refines_source_as_dm(adapter_harness):
    """P2P generic card actions must retain platform and resolve as dm."""
    sdk_action = CardActionEvent(
        message_id="om_card_dm",
        chat_id="p2p_alice",
        operator=EventOperator(open_id="ou_alice", user_id="u_alice", name="Alice"),
        action=CardActionPayload(value={"command": "/status"}, tag="button"),
        raw={},
    )

    await adapter_harness.adapter._on_sdk_card_action(sdk_action)
    await _drain_adapter_tasks()

    assert adapter_harness.captured_inbound, (
        "P2P card action must synthesize a COMMAND MessageEvent"
    )
    event = adapter_harness.captured_inbound[0]
    assert event.source.platform.value == "feishu"
    assert event.source.chat_type == "dm"
