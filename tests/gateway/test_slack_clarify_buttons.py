"""Slack Block Kit rendering and callbacks for clarify choices."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


class _AuthRunner:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.sources = []

    async def handle(self, event):
        return None

    def _is_user_authorized(self, source):
        self.sources.append(source)
        return self.allowed


def _make_adapter(*, allowed=True):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    client1 = AsyncMock()
    client1.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})
    client1.chat_update = AsyncMock()
    client2 = AsyncMock()
    client2.chat_postMessage = AsyncMock(return_value={"ts": "2222.3333"})
    client2.chat_update = AsyncMock()
    adapter._team_clients = {"T1": client1, "T2": client2}
    adapter._team_bot_user_ids = {"T1": "U_BOT", "T2": "U_BOT_2"}
    adapter._channel_team = {"C1": "T1"}
    runner = _AuthRunner(allowed=allowed)
    adapter.set_message_handler(runner.handle)
    return adapter, client1, client2, runner


async def _prime_prompt(adapter, *, team_id="T1", thread_ts="9999.0000"):
    return await adapter.send_clarify(
        chat_id="C1",
        question="What should we do next?",
        choices=["Ship now", "Schedule next"],
        clarify_id="clarify-123",
        session_key="agent:main:slack:dm:C1",
        metadata={"thread_id": thread_ts, "slack_team_id": team_id},
    )


def _button_body(
    *,
    msg_ts="1234.5678",
    team_id="T1",
    channel_id="C1",
    thread_ts="9999.0000",
    user_id="U_RYAN",
    user_name="ryan",
):
    return {
        "team": {"id": team_id},
        "message": {
            "ts": msg_ts,
            "thread_ts": thread_ts,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "❓ What next?"},
                },
                {"type": "actions", "elements": []},
            ],
        },
        "channel": {"id": channel_id},
        "user": {"id": user_id, "name": user_name},
    }


def _choice_action(index=0, clarify_id="clarify-123"):
    return {
        "action_id": "hermes_clarify_choice",
        "value": json.dumps({"id": clarify_id, "index": index}),
    }


def _other_action(clarify_id="clarify-123"):
    return {
        "action_id": "hermes_clarify_choice",
        "value": json.dumps({"id": clarify_id, "other": True}),
    }


@pytest.mark.asyncio
async def test_send_clarify_uses_canonical_indices_and_binds_prompt():
    adapter, client1, _client2, _runner = _make_adapter()

    result = await _prime_prompt(adapter)

    assert result.success is True
    kwargs = client1.chat_postMessage.await_args.kwargs
    assert kwargs["thread_ts"] == "9999.0000"
    assert kwargs["text"] == "What should we do next?"
    elements = kwargs["blocks"][1]["elements"]
    assert [element["text"]["text"] for element in elements] == [
        "Ship now",
        "Schedule next",
        "Other (type answer)",
    ]
    assert [json.loads(element["value"]) for element in elements] == [
        {"id": "clarify-123", "index": 0},
        {"id": "clarify-123", "index": 1},
        {"id": "clarify-123", "other": True},
    ]
    prompt = adapter._clarify_prompts["clarify-123"]
    assert prompt.choices == ("Ship now", "Schedule next")
    assert (prompt.team_id, prompt.channel_id, prompt.thread_ts, prompt.message_ts) == (
        "T1",
        "C1",
        "9999.0000",
        "1234.5678",
    )


@pytest.mark.asyncio
async def test_missing_workspace_fails_closed_without_primary_client_fallback():
    adapter, client1, client2, _runner = _make_adapter()
    adapter._channel_team.clear()
    adapter._app.client = AsyncMock()

    result = await adapter.send_clarify(
        chat_id="UNMAPPED",
        question="Pick",
        choices=["A", "B"],
        clarify_id="missing-team",
        session_key="session",
        metadata=None,
    )

    assert result.success is False
    assert "workspace" in result.error.lower()
    adapter._app.client.chat_postMessage.assert_not_awaited()
    client1.chat_postMessage.assert_not_awaited()
    client2.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_workspace_fails_closed_without_client_fallback():
    adapter, client1, client2, _runner = _make_adapter()

    result = await _prime_prompt(adapter, team_id="UNKNOWN")

    assert result.success is False
    assert "workspace" in result.error.lower()
    client1.chat_postMessage.assert_not_awaited()
    client2.chat_postMessage.assert_not_awaited()
    assert adapter._clarify_prompts == {}


@pytest.mark.asyncio
async def test_send_clarify_selects_metadata_workspace_client():
    adapter, client1, client2, _runner = _make_adapter()

    result = await _prime_prompt(adapter, team_id="T2")

    assert result.message_id == "2222.3333"
    client1.chat_postMessage.assert_not_awaited()
    client2.chat_postMessage.assert_awaited_once()
    assert adapter._clarify_prompts["clarify-123"].team_id == "T2"


@pytest.mark.asyncio
async def test_rendering_respects_section_label_and_value_limits():
    adapter, client1, _client2, _runner = _make_adapter()
    choice = "🚀" * 100

    await adapter.send_clarify(
        chat_id="C1",
        question="Q" * 4000,
        choices=[choice],
        clarify_id="limits",
        session_key="session",
        metadata={"slack_team_id": "T1"},
    )

    blocks = client1.chat_postMessage.await_args.kwargs["blocks"]
    assert len(blocks[0]["text"]["text"]) <= 3000
    button = blocks[1]["elements"][0]
    assert len(button["text"]["text"]) <= 75
    assert len(button["value"]) <= 2000
    assert json.loads(button["value"]) == {"id": "limits", "index": 0}


@pytest.mark.asyncio
async def test_oversized_button_value_falls_back_to_text():
    adapter, client1, _client2, _runner = _make_adapter()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))

    await adapter.send_clarify(
        chat_id="C1",
        question="Pick",
        choices=["A"],
        clarify_id="x" * 2100,
        session_key="session",
        metadata={"slack_team_id": "T1"},
    )

    client1.chat_postMessage.assert_not_awaited()
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_more_than_four_choices_falls_back_to_text():
    adapter, client1, _client2, _runner = _make_adapter()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))

    await adapter.send_clarify(
        chat_id="C1",
        question="Pick",
        choices=["1", "2", "3", "4", "5"],
        clarify_id="too-many",
        session_key="session",
        metadata={"slack_team_id": "T1"},
    )

    client1.chat_postMessage.assert_not_awaited()
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_ended_clarify_uses_text_fallback():
    adapter, client1, _client2, _runner = _make_adapter()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))

    result = await adapter.send_clarify(
        chat_id="C1",
        question="Tell me more",
        choices=None,
        clarify_id="open",
        session_key="session",
        metadata={"thread_id": "9999.0000", "slack_team_id": "T1"},
    )

    assert result.success is True
    client1.chat_postMessage.assert_not_awaited()
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_authorized_choice_resolves_canonical_value_and_updates_correct_workspace():
    adapter, client1, _client2, runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)
    ack = AsyncMock()

    with patch(
        "tools.clarify_gateway.resolve_gateway_choice",
        return_value=(True, "Ship now"),
    ) as resolve:
        await adapter._handle_clarify_action(ack, _button_body(), _choice_action())

    ack.assert_awaited_once()
    resolve.assert_called_once_with("clarify-123", 0)
    assert runner.sources[0].scope_id == "T1"
    update = client1.chat_update.await_args.kwargs
    assert (update["channel"], update["ts"]) == ("C1", "1234.5678")
    assert all(block["type"] != "actions" for block in update["blocks"])
    assert "Ship now" in update["text"] and "ryan" in update["text"]
    assert "clarify-123" not in adapter._clarify_prompts


@pytest.mark.asyncio
async def test_disconnected_workspace_client_fails_before_resolution():
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)
    adapter._team_clients.pop("T1")

    with patch("tools.clarify_gateway.resolve_gateway_choice") as resolve:
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _choice_action()
        )

    resolve.assert_not_called()
    client1.chat_update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        _button_body(team_id="WRONG"),
        _button_body(channel_id="WRONG"),
        _button_body(msg_ts="WRONG"),
        _button_body(thread_ts="WRONG"),
    ],
)
async def test_mismatched_origin_binding_fails_closed(body):
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)

    with patch("tools.clarify_gateway.resolve_gateway_choice") as resolve:
        await adapter._handle_clarify_action(AsyncMock(), body, _choice_action())

    resolve.assert_not_called()
    client1.chat_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_choice_fails_closed_with_workspace_scope():
    adapter, client1, _client2, runner = _make_adapter(allowed=False)
    await _prime_prompt(adapter)

    with patch("tools.clarify_gateway.resolve_gateway_choice") as resolve:
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _choice_action()
        )

    resolve.assert_not_called()
    client1.chat_update.assert_not_awaited()
    assert runner.sources[0].scope_id == "T1"


@pytest.mark.asyncio
async def test_other_switches_to_text_capture_and_keeps_bounded_binding():
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)

    with patch("tools.clarify_gateway.mark_awaiting_text", return_value=True) as mark:
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _other_action()
        )

    mark.assert_called_once_with("clarify-123")
    assert adapter._clarify_prompts["clarify-123"].awaiting_text is True
    update = client1.chat_update.await_args.kwargs
    assert "reply with your answer" in update["text"].lower()
    assert all(block["type"] != "actions" for block in update["blocks"])


@pytest.mark.asyncio
@pytest.mark.parametrize("use_other", [False, True])
async def test_update_failure_posts_workspace_bound_fallback(use_other):
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)
    client1.chat_update.side_effect = RuntimeError("update failed")

    if use_other:
        action = _other_action()
        resolver_patch = patch(
            "tools.clarify_gateway.mark_awaiting_text", return_value=True
        )
    else:
        action = _choice_action()
        resolver_patch = patch(
            "tools.clarify_gateway.resolve_gateway_choice",
            return_value=(True, "Ship now"),
        )

    with resolver_patch:
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), action
        )

    assert client1.chat_postMessage.await_count == 2
    fallback = client1.chat_postMessage.await_args.kwargs
    assert fallback["channel"] == "C1"
    assert fallback["thread_ts"] == "9999.0000"
    if use_other:
        assert "reply with your answer" in fallback["text"].lower()
    else:
        assert "ship now" in fallback["text"].lower()


@pytest.mark.asyncio
async def test_text_wins_race_then_button_is_shown_as_expired():
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)

    with patch(
        "tools.clarify_gateway.resolve_gateway_choice",
        return_value=(False, None),
    ):
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _choice_action()
        )

    assert "already answered" in client1.chat_update.await_args.kwargs["text"].lower()
    assert "clarify-123" not in adapter._clarify_prompts


@pytest.mark.asyncio
async def test_duplicate_or_replayed_click_cannot_resolve_twice():
    adapter, _client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)

    with patch(
        "tools.clarify_gateway.resolve_gateway_choice",
        return_value=(True, "Ship now"),
    ) as resolve:
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _choice_action()
        )
        await adapter._handle_clarify_action(
            AsyncMock(), _button_body(), _choice_action()
        )

    resolve.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,value",
    [
        ({}, json.dumps({"id": "clarify-123", "index": 0})),
        (_button_body(), "not-json"),
        (_button_body(), "{}"),
        (_button_body(), json.dumps({"id": "clarify-123", "index": "0"})),
        (_button_body(), json.dumps({"id": "substituted", "index": 0})),
    ],
)
async def test_malformed_or_substituted_callback_fails_closed(body, value):
    adapter, client1, _client2, _runner = _make_adapter(allowed=True)
    await _prime_prompt(adapter)
    action = {"action_id": "hermes_clarify_choice", "value": value}

    with patch("tools.clarify_gateway.resolve_gateway_choice") as resolve:
        await adapter._handle_clarify_action(AsyncMock(), body, action)

    resolve.assert_not_called()
    client1.chat_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_failure_creates_no_callback_state():
    adapter, client1, _client2, _runner = _make_adapter()
    client1.chat_postMessage.side_effect = RuntimeError("send failed")

    result = await _prime_prompt(adapter)

    assert result.success is False
    assert adapter._clarify_prompts == {}


@pytest.mark.asyncio
async def test_expired_prompt_state_is_pruned():
    adapter, _client1, _client2, _runner = _make_adapter()
    await _prime_prompt(adapter)
    adapter._clarify_prompts["clarify-123"].created_at = time.monotonic() - 999999

    await adapter.send_clarify(
        chat_id="C1",
        question="Another",
        choices=["A"],
        clarify_id="fresh",
        session_key="session",
        metadata={"slack_team_id": "T1"},
    )

    assert "clarify-123" not in adapter._clarify_prompts
    assert "fresh" in adapter._clarify_prompts
