"""Tests for Slack Block Kit interactive choice picker.

Mirrors test_slack_clarify_buttons.py (harness) for the
``send_choice_picker`` override and the ``hermes_cp_<idx>``
action dispatch via ``_handle_choice_picker_action``.

Coverage:
- send_choice_picker: renders Block Kit buttons, stores state, returns msg_ts
- send_choice_picker: empty choices → early failure
- send_choice_picker: not connected → early failure
- send_choice_picker: marks current choice with ✓
- _handle_choice_picker_action: valid tap fires on_choice_selected callback
- _handle_choice_picker_action: unauthorized user → ignored (auth bypass guard)
- _handle_choice_picker_action: double-tap → state already popped, no callback
- _handle_choice_picker_action: malformed action_id → no crash
- _handle_choice_picker_action: out-of-range index → no crash
- _handle_choice_picker_action: expired state (msg_ts not in dict) → no crash
- _handle_choice_picker_action: callback exception → logged, no crash
- _handle_choice_picker_action: chat_update failure → logged, callback still fires
- State isolation: separate pickers don't interfere
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# ---------------------------------------------------------------------------
# Minimal Slack SDK mock (mirrors test_slack_clarify_buttons.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client

_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


def _make_choices():
    return [
        {"value": "fast", "label": "Fast", "is_current": False},
        {"value": "auto", "label": "Auto", "is_current": True},
        {"value": "slow", "label": "Slow", "is_current": False},
    ]


def _make_action_body(action_id="hermes_cp_0", msg_ts="111.222", channel_id="C1",
                      user_id="U1", user_name="alice"):
    return {
        "message": {"ts": msg_ts, "blocks": []},
        "channel": {"id": channel_id},
        "user": {"id": user_id, "name": user_name},
    }


def _make_action(action_id="hermes_cp_0"):
    return {"action_id": action_id, "value": ""}


# ---------------------------------------------------------------------------
# send_choice_picker tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_choice_picker_not_connected():
    adapter = _make_adapter()
    adapter._app = None
    result = await adapter.send_choice_picker(
        chat_id="C1", title="Pick one", choices=_make_choices(),
        session_key="sk", on_choice_selected=AsyncMock(),
    )
    assert not result.success
    assert "Not connected" in (result.error or "")


@pytest.mark.asyncio
async def test_send_choice_picker_empty_choices():
    adapter = _make_adapter()
    adapter._ensure_dm_conversation = AsyncMock(return_value="C1")
    adapter._resolve_thread_ts = MagicMock(return_value=None)
    adapter._get_client = MagicMock(return_value=AsyncMock())
    result = await adapter.send_choice_picker(
        chat_id="C1", title="Pick one", choices=[],
        session_key="sk", on_choice_selected=AsyncMock(),
    )
    assert not result.success
    assert "No choices" in (result.error or "")


@pytest.mark.asyncio
async def test_send_choice_picker_renders_buttons_and_stores_state():
    adapter = _make_adapter()
    adapter._ensure_dm_conversation = AsyncMock(return_value="C1")
    adapter._resolve_thread_ts = MagicMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "111.001"})
    adapter._get_client = MagicMock(return_value=mock_client)

    on_selected = AsyncMock()
    choices = _make_choices()
    result = await adapter.send_choice_picker(
        chat_id="C1", title="Choose mode", choices=choices,
        session_key="sk1", on_choice_selected=on_selected,
    )

    assert result.success
    assert result.message_id == "111.001"
    assert "111.001" in adapter._choice_picker_state
    state = adapter._choice_picker_state["111.001"]
    assert state["session_key"] == "sk1"
    assert state["on_choice_selected"] is on_selected
    assert state["choices"] is choices

    # Verify Block Kit structure
    call_kwargs = mock_client.chat_postMessage.call_args.kwargs
    blocks = call_kwargs["blocks"]
    assert blocks[0]["type"] == "section"
    action_blocks = [b for b in blocks if b["type"] == "actions"]
    assert len(action_blocks) >= 1
    all_elements = [e for b in action_blocks for e in b["elements"]]
    assert len(all_elements) == 3
    assert all_elements[0]["action_id"] == "hermes_cp_0"
    assert all_elements[0]["value"] == "cp:0"


@pytest.mark.asyncio
async def test_send_choice_picker_marks_current_choice():
    adapter = _make_adapter()
    adapter._ensure_dm_conversation = AsyncMock(return_value="C1")
    adapter._resolve_thread_ts = MagicMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "111.002"})
    adapter._get_client = MagicMock(return_value=mock_client)

    choices = _make_choices()  # index 1 is_current=True
    await adapter.send_choice_picker(
        chat_id="C1", title="Pick", choices=choices,
        session_key="sk", on_choice_selected=AsyncMock(),
    )

    call_kwargs = mock_client.chat_postMessage.call_args.kwargs
    action_blocks = [b for b in call_kwargs["blocks"] if b["type"] == "actions"]
    all_elements = [e for b in action_blocks for e in b["elements"]]
    current_label = all_elements[1]["text"]["text"]
    assert current_label.startswith("✓ ")


@pytest.mark.asyncio
async def test_send_choice_picker_with_thread_ts():
    adapter = _make_adapter()
    adapter._ensure_dm_conversation = AsyncMock(return_value="C1")
    adapter._resolve_thread_ts = MagicMock(return_value="100.000")

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "111.003"})
    adapter._get_client = MagicMock(return_value=mock_client)

    await adapter.send_choice_picker(
        chat_id="C1", title="Pick", choices=_make_choices(),
        session_key="sk", on_choice_selected=AsyncMock(),
    )

    call_kwargs = mock_client.chat_postMessage.call_args.kwargs
    assert call_kwargs.get("thread_ts") == "100.000"


# ---------------------------------------------------------------------------
# _handle_choice_picker_action tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_choice_picker_valid_tap_fires_callback():
    adapter = _make_adapter()
    on_selected = AsyncMock()
    adapter._choice_picker_state["111.222"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    mock_client = AsyncMock()
    adapter._get_client = MagicMock(return_value=mock_client)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_0", msg_ts="111.222"),
        action=_make_action("hermes_cp_0"),
    )

    ack.assert_awaited_once()
    on_selected.assert_awaited_once_with("fast")
    assert "111.222" not in adapter._choice_picker_state


@pytest.mark.asyncio
async def test_handle_choice_picker_unauthorized_user_ignored():
    adapter = _make_adapter()
    on_selected = AsyncMock()
    adapter._choice_picker_state["111.333"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=False)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_0", msg_ts="111.333"),
        action=_make_action("hermes_cp_0"),
    )

    ack.assert_awaited_once()
    on_selected.assert_not_called()
    # State must NOT be consumed — authorized user should still be able to pick
    assert "111.333" in adapter._choice_picker_state


@pytest.mark.asyncio
async def test_handle_choice_picker_double_tap_no_second_callback():
    adapter = _make_adapter()
    on_selected = AsyncMock()
    adapter._choice_picker_state["111.444"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    mock_client = AsyncMock()
    adapter._get_client = MagicMock(return_value=mock_client)

    ack = AsyncMock()
    body = _make_action_body(action_id="hermes_cp_1", msg_ts="111.444")
    action = _make_action("hermes_cp_1")

    # First tap
    await adapter._handle_choice_picker_action(ack=ack, body=body, action=action)
    # Second tap — state already popped
    await adapter._handle_choice_picker_action(ack=ack, body=body, action=action)

    assert on_selected.await_count == 1


@pytest.mark.asyncio
async def test_handle_choice_picker_malformed_action_id_no_crash():
    adapter = _make_adapter()
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_notanumber"),
        action=_make_action("hermes_cp_notanumber"),
    )
    ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_choice_picker_out_of_range_index_no_crash():
    adapter = _make_adapter()
    on_selected = AsyncMock()
    adapter._choice_picker_state["111.555"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_99", msg_ts="111.555"),
        action=_make_action("hermes_cp_99"),
    )
    on_selected.assert_not_called()


@pytest.mark.asyncio
async def test_handle_choice_picker_expired_state_no_crash():
    adapter = _make_adapter()
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)

    ack = AsyncMock()
    # No state planted — simulates expired/evicted entry
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_0", msg_ts="999.999"),
        action=_make_action("hermes_cp_0"),
    )
    ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_choice_picker_callback_exception_logged_no_crash():
    adapter = _make_adapter()
    on_selected = AsyncMock(side_effect=RuntimeError("boom"))
    adapter._choice_picker_state["111.666"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    mock_client = AsyncMock()
    adapter._get_client = MagicMock(return_value=mock_client)

    ack = AsyncMock()
    # Should not raise
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_0", msg_ts="111.666"),
        action=_make_action("hermes_cp_0"),
    )
    ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_choice_picker_chat_update_failure_callback_still_fires():
    adapter = _make_adapter()
    on_selected = AsyncMock()
    adapter._choice_picker_state["111.777"] = {
        "choices": _make_choices(),
        "session_key": "sk",
        "on_choice_selected": on_selected,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    mock_client = AsyncMock()
    mock_client.chat_update = AsyncMock(side_effect=Exception("Slack down"))
    adapter._get_client = MagicMock(return_value=mock_client)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_2", msg_ts="111.777"),
        action=_make_action("hermes_cp_2"),
    )
    # Callback must still fire even when chat_update fails
    on_selected.assert_awaited_once_with("slow")


@pytest.mark.asyncio
async def test_choice_picker_state_isolation():
    """Two concurrent pickers must not interfere with each other."""
    adapter = _make_adapter()
    on_selected_a = AsyncMock()
    on_selected_b = AsyncMock()

    adapter._choice_picker_state["ts_a"] = {
        "choices": [{"value": "x", "label": "X", "is_current": False}],
        "session_key": "sk_a",
        "on_choice_selected": on_selected_a,
    }
    adapter._choice_picker_state["ts_b"] = {
        "choices": [{"value": "y", "label": "Y", "is_current": False}],
        "session_key": "sk_b",
        "on_choice_selected": on_selected_b,
    }
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    mock_client = AsyncMock()
    adapter._get_client = MagicMock(return_value=mock_client)

    ack = AsyncMock()
    await adapter._handle_choice_picker_action(
        ack=ack,
        body=_make_action_body(action_id="hermes_cp_0", msg_ts="ts_a"),
        action=_make_action("hermes_cp_0"),
    )

    on_selected_a.assert_awaited_once_with("x")
    on_selected_b.assert_not_called()
    assert "ts_a" not in adapter._choice_picker_state
    assert "ts_b" in adapter._choice_picker_state
