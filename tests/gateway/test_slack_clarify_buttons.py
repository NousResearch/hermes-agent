"""Tests for Slack Block Kit clarify buttons.

Mirrors test_slack_approval_buttons.py for the new ``send_clarify`` and
``hermes_clarify_*`` action dispatch.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported
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


class _AuthRunner:
    def __init__(self, auth_fn=None):
        self._auth_fn = auth_fn or (lambda _source: True)
        self.seen_sources = []

    async def handle(self, event):
        return None

    def _is_user_authorized(self, source):
        self.seen_sources.append(source)
        return self._auth_fn(source)


def _attach_auth_runner(adapter, auth_fn=None):
    runner = _AuthRunner(auth_fn=auth_fn)
    adapter.set_message_handler(runner.handle)
    return runner


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


# ===========================================================================
# send_clarify — Block Kit render
# ===========================================================================

class TestSlackSendClarify:
    """Verify the rendered prompt has buttons or none, and registers state."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_multi_choice_renders_buttons_and_other(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "100.1"})

        result = await adapter.send_clarify(
            chat_id="C1",
            question="Which preset?",
            choices=["Classic", "Golden Hour", "Vibrant", "Blue Tint"],
            clarify_id="cid1",
            session_key="sk1",
        )

        assert result.success is True
        assert result.message_id == "100.1"

        kwargs = mock_client.chat_postMessage.call_args[1]
        assert kwargs["text"] == "❓ Which preset?"
        blocks = kwargs["blocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "section"
        assert "Which preset?" in blocks[0]["text"]["text"]
        assert blocks[1]["type"] == "actions"
        elements = blocks[1]["elements"]
        # 4 choices + 1 Other
        assert len(elements) == 5
        # First four are choice buttons. Each needs a unique action_id
        # (Slack rejects duplicates within a message); idx is encoded in
        # the action_id suffix AND in the value field.
        seen_action_ids = set()
        for idx, choice in enumerate(["Classic", "Golden Hour", "Vibrant", "Blue Tint"]):
            el = elements[idx]
            assert el["action_id"] == f"hermes_clarify_choice_{idx}"
            assert el["action_id"] not in seen_action_ids
            seen_action_ids.add(el["action_id"])
            assert el["value"] == f"cid1|{idx}"
            assert choice in el["text"]["text"]
        # Last is the Other button
        other = elements[-1]
        assert other["action_id"] == "hermes_clarify_other"
        assert other["value"] == "cid1"
        # Double-click guard registered
        assert adapter._clarify_resolved.get("100.1") is False

    @pytest.mark.asyncio
    async def test_open_ended_falls_through_to_base(self):
        """No choices → no Block Kit buttons; the base adapter renders the
        question as plain text and the gateway text-intercept catches the
        reply."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "100.2"})

        result = await adapter.send_clarify(
            chat_id="C1",
            question="What is your name?",
            choices=None,
            clarify_id="cid2",
            session_key="sk2",
        )

        assert result.success is True
        # Base path goes through send() → chat_postMessage with text only,
        # no blocks key.
        kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" not in kwargs
        assert "What is your name?" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._app = None
        result = await adapter.send_clarify(
            chat_id="C1",
            question="?",
            choices=["a"],
            clarify_id="cid3",
            session_key="sk3",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_long_choice_label_truncated(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "100.3"})

        long_choice = "x" * 200
        await adapter.send_clarify(
            chat_id="C1",
            question="?",
            choices=[long_choice],
            clarify_id="cid4",
            session_key="sk4",
        )
        kwargs = mock_client.chat_postMessage.call_args[1]
        label = kwargs["blocks"][1]["elements"][0]["text"]["text"]
        # Numeric prefix + truncated body + ellipsis, well under Slack's 75-char
        # plain_text cap.
        assert len(label) < 80
        assert label.endswith("...")

    @pytest.mark.asyncio
    async def test_long_question_is_budgeted_under_section_cap(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "100.5"})

        await adapter.send_clarify(
            chat_id="C1",
            question="x" * 4000,
            choices=["a", "b"],
            clarify_id="cid-long",
            session_key="sk-long",
        )

        kwargs = mock_client.chat_postMessage.call_args[1]
        section_text = kwargs["blocks"][0]["text"]["text"]
        assert len(section_text) <= 3000
        assert section_text.endswith("...")
        assert kwargs["text"] == section_text

    @pytest.mark.asyncio
    async def test_inherits_thread_ts_from_metadata(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "100.4"})

        await adapter.send_clarify(
            chat_id="C1",
            question="?",
            choices=["a", "b"],
            clarify_id="cid5",
            session_key="sk5",
            metadata={"thread_id": "9000.0"},
        )
        kwargs = mock_client.chat_postMessage.call_args[1]
        assert kwargs.get("thread_ts") == "9000.0"


# ===========================================================================
# _handle_clarify_action — button click handler
# ===========================================================================

class TestSlackClarifyAction:
    """Verify clicking a button resolves the clarify primitive."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_choice_click_resolves_with_choice_text(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        runner = _attach_auth_runner(adapter)
        cm.register("cidA", "sk-cb", "Which preset?", ["Classic", "Golden Hour"])
        adapter._clarify_resolved["1.2"] = False

        ack = AsyncMock()
        body = {
            "message": {
                "ts": "1.2",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Which preset?"}},
                    {"type": "actions", "elements": []},
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "norbert", "id": "U_NORB"},
        }
        action = {
            "action_id": "hermes_clarify_choice_0",
            "value": "cidA|1",  # Golden Hour
        }
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        await adapter._handle_clarify_action(ack, body, action)

        ack.assert_called_once()
        assert runner.seen_sources
        assert runner.seen_sources[0].user_id == "U_NORB"
        # The entry should have been resolved with the choice text.
        with cm._lock:
            entry = cm._entries.get("cidA")
        assert entry is not None
        assert entry.response == "Golden Hour"
        assert entry.event.is_set()
        # Chat message updated to show the decision.
        update_kwargs = mock_client.chat_update.call_args[1]
        assert "Golden Hour" in update_kwargs["text"]
        assert "norbert" in update_kwargs["text"]

    @pytest.mark.asyncio
    async def test_other_click_flips_to_text_mode(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidB", "sk-other", "Pick", ["x", "y"])
        adapter._clarify_resolved["1.3"] = False

        ack = AsyncMock()
        body = {
            "message": {
                "ts": "1.3",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}},
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {"action_id": "hermes_clarify_other", "value": "cidB"}
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        await adapter._handle_clarify_action(ack, body, action)

        # Entry now awaiting text — NOT yet resolved.
        with cm._lock:
            entry = cm._entries.get("cidB")
        assert entry is not None
        assert entry.awaiting_text is True
        assert not entry.event.is_set()
        # User sees the "type your answer" decision text.
        update_kwargs = mock_client.chat_update.call_args[1]
        assert "Other" in update_kwargs["text"]

    @pytest.mark.asyncio
    async def test_double_click_guarded(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidC", "sk-dbl", "Pick", ["x", "y"])
        adapter._clarify_resolved["1.4"] = True  # Already resolved

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.4", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {"action_id": "hermes_clarify_choice_0", "value": "cidC|0"}

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._handle_clarify_action(ack, body, action)

        ack.assert_called_once()
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_gateway_runner_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter, auth_fn=lambda _source: False)
        cm.register("cidD", "sk-auth", "Pick", ["a"])
        adapter._clarify_resolved["1.5"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.5", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "mallory", "id": "U_MAL"},
        }
        action = {"action_id": "hermes_clarify_choice_0", "value": "cidD|0"}

        await adapter._handle_clarify_action(ack, body, action)

        # Entry must remain unresolved.
        with cm._lock:
            entry = cm._entries.get("cidD")
        assert entry is not None
        assert not entry.event.is_set()
        # And the double-click guard must NOT have been consumed —
        # an authorized user re-trying still gets a chance.
        assert adapter._clarify_resolved.get("1.5") is False

    @pytest.mark.asyncio
    async def test_global_gateway_allowlist_authorizes_click(self, monkeypatch):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        monkeypatch.delenv("SLACK_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("SLACK_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_GOOD")
        cm.register("cid-global", "sk-global", "Pick", ["a"])
        adapter._clarify_resolved["1.9"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.9", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "gail", "id": "U_GOOD"},
        }
        action = {"action_id": "hermes_clarify_choice_0", "value": "cid-global|0"}
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        await adapter._handle_clarify_action(ack, body, action)

        with cm._lock:
            entry = cm._entries.get("cid-global")
        assert entry is not None
        assert entry.response == "a"
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_unknown_clarify_id_does_not_crash(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        adapter._clarify_resolved["1.6"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.6", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {"action_id": "hermes_clarify_choice_0", "value": "missing|0"}

        # No registered entry for clarify_id "missing" — must just warn,
        # not raise.
        await adapter._handle_clarify_action(ack, body, action)
        ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_malformed_value_does_not_crash(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        adapter._clarify_resolved["1.7"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.7", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        # No pipe separator at all.
        action = {"action_id": "hermes_clarify_choice_0", "value": "no-pipe-here"}

        await adapter._handle_clarify_action(ack, body, action)
        ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_idx_out_of_range_does_not_crash(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidE", "sk-oob", "Pick", ["only-one"])
        adapter._clarify_resolved["1.8"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.8", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        # idx=5 is out of range for a single-choice entry.
        action = {"action_id": "hermes_clarify_choice_0", "value": "cidE|5"}

        await adapter._handle_clarify_action(ack, body, action)
        with cm._lock:
            entry = cm._entries.get("cidE")
        assert entry is not None
        assert not entry.event.is_set()
