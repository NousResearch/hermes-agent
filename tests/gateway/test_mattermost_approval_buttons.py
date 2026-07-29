"""Tests for Mattermost interactive button prompts (plugin adapter)."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter


def _make_adapter(*, buttons_enabled: bool = True) -> MattermostAdapter:
    """Create a MattermostAdapter with mocked internals for unit testing."""
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = MattermostAdapter(config)
    adapter._base_url = "http://mattermost.local"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter._session = MagicMock()
    adapter._callback_host = "127.0.0.1"
    adapter._callback_port = 18065
    adapter._callback_url = ""
    adapter._runner = object() if buttons_enabled else None
    return adapter


def _make_request(body: dict) -> MagicMock:
    """Build a fake aiohttp Request carrying the given JSON body."""
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


class TestButtonFormat:
    def test_make_button_mattermost_fields(self):
        adapter = _make_adapter()
        btn = adapter._make_button(
            "approveonce", "Allow Once", "action123", {"choice": "once"},
        )
        assert btn["id"] == "approveonce"
        assert btn["name"] == "Allow Once"
        assert "text" not in btn
        assert "value" not in btn
        assert btn["integration"]["url"].endswith("/hermes-callback")
        assert btn["integration"]["context"]["action_id"] == "action123"
        assert btn["integration"]["context"]["choice"] == "once"

    def test_effective_callback_url_override(self):
        adapter = _make_adapter()
        adapter._callback_url = "http://external:9999/hermes-callback"
        assert adapter._effective_callback_url() == "http://external:9999/hermes-callback"

    def test_pop_action_double_click_guard(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok1"] = {"kind": "approval", "session_key": "s1"}
        assert adapter._pop_action("tok1") == {"kind": "approval", "session_key": "s1"}
        assert adapter._pop_action("tok1") is None


class TestCallbackUrlResolution:
    """callback_url must resolve like callback_host/port: config.extra → env."""

    def test_from_config_extra(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_CALLBACK_URL", raising=False)
        config = PlatformConfig(
            enabled=True, token="t",
            extra={"callback_url": "http://cfg:1/hermes-callback"},
        )
        adapter = MattermostAdapter(config)
        assert adapter._callback_url == "http://cfg:1/hermes-callback"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_CALLBACK_URL", "http://env:2/hermes-callback")
        adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t"))
        assert adapter._callback_url == "http://env:2/hermes-callback"

    def test_config_extra_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_CALLBACK_URL", "http://env/hermes-callback")
        config = PlatformConfig(
            enabled=True, token="t",
            extra={"callback_url": "http://cfg/hermes-callback"},
        )
        adapter = MattermostAdapter(config)
        assert adapter._callback_url == "http://cfg/hermes-callback"

    def test_unset_is_empty(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_CALLBACK_URL", raising=False)
        adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t"))
        assert adapter._callback_url == ""


class TestFallbackWhenButtonsDisabled:
    @pytest.mark.asyncio
    async def test_send_exec_approval_disabled(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock()
        result = await adapter.send_exec_approval("ch1", "rm -rf /", "s1")
        assert result.success is False
        adapter._api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_slash_confirm_disabled(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock()
        result = await adapter.send_slash_confirm(
            "ch1", "Title", "body", "s1", "cid1",
        )
        assert result.success is False
        adapter._api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_update_prompt_disabled(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock()
        result = await adapter.send_update_prompt("ch1", "Proceed?")
        assert result.success is False
        adapter._api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_clarify_with_choices_disabled(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock()
        result = await adapter.send_clarify(
            "ch1", "Pick one", ["A", "B"], "cl1", "s1",
        )
        assert result.success is False
        adapter._api_post.assert_not_called()


class TestSendAndRegister:
    @pytest.mark.asyncio
    async def test_send_exec_approval_registers_state(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        result = await adapter.send_exec_approval("ch1", "rm -rf /", "session1")
        assert result.success is True
        assert result.message_id == "p1"
        assert len(adapter._pending_actions) == 1
        payload = next(iter(adapter._pending_actions.values()))
        assert payload["kind"] == "approval"
        assert payload["session_key"] == "session1"

    @pytest.mark.asyncio
    async def test_send_slash_confirm_registers_state(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p2"})
        result = await adapter.send_slash_confirm(
            "ch1", "Confirm", "Are you sure?", "session1", "confirm1",
        )
        assert result.success is True
        payload = next(iter(adapter._pending_actions.values()))
        assert payload["kind"] == "slash"
        assert payload["confirm_id"] == "confirm1"
        assert payload["channel_id"] == "ch1"

    @pytest.mark.asyncio
    async def test_send_update_prompt_registers_state(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p3"})
        result = await adapter.send_update_prompt("ch1", "Update now?")
        assert result.success is True
        payload = next(iter(adapter._pending_actions.values()))
        assert payload["kind"] == "update"

    @pytest.mark.asyncio
    async def test_send_clarify_registers_state(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p4"})
        result = await adapter.send_clarify(
            "ch1", "Pick", ["Alpha", "Beta"], "cl1", "session1",
        )
        assert result.success is True
        payload = next(iter(adapter._pending_actions.values()))
        assert payload["kind"] == "clarify"
        assert payload["choices"] == ["Alpha", "Beta"]

    @pytest.mark.asyncio
    async def test_send_clarify_no_choices_posts_text(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock(return_value={"id": "p5"})
        result = await adapter.send_clarify("ch1", "What?", None, "cl1", "session1")
        assert result.success is True
        adapter._api_post.assert_called_once()


class TestPostFailure:
    @pytest.mark.asyncio
    async def test_exec_approval_no_post_id(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={})
        result = await adapter.send_exec_approval("ch1", "cmd", "s1")
        assert result.success is False
        # Pending entry must NOT be stored when the post failed.
        assert adapter._pending_actions == {}


class TestThreadRouting:
    """Interactive prompts must land inside the active thread when
    ``reply_mode == "thread"`` and the gateway hands us a thread root via
    ``metadata`` — and must stay flat otherwise.

    This is the behavior contract for the "approvals belong in the thread,
    not the parent channel" fix: the gateway already passes
    ``metadata=_status_thread_metadata`` to every prompt send, so the only
    thing the adapter owns is honoring that root.
    """

    @staticmethod
    def _posted_payload(adapter) -> dict:
        # _api_post(path, payload) — payload is the second positional arg.
        return adapter._api_post.call_args.args[1]

    def _thread_adapter(self):
        adapter = _make_adapter()
        adapter._reply_mode = "thread"
        adapter._api_post = AsyncMock(return_value={"id": "posted"})
        # _resolve_root_id GETs the candidate post; an empty body means the
        # candidate is already a thread root, so it is used verbatim.
        adapter._api_get = AsyncMock(return_value={})
        return adapter

    @pytest.mark.asyncio
    async def test_exec_approval_posts_in_thread(self):
        adapter = self._thread_adapter()
        await adapter.send_exec_approval(
            "ch1", "rm -rf /", "s1", metadata={"thread_id": "root123"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root123"

    @pytest.mark.asyncio
    async def test_slash_confirm_posts_in_thread(self):
        adapter = self._thread_adapter()
        await adapter.send_slash_confirm(
            "ch1", "Title", "body", "s1", "cid1", metadata={"thread_id": "root123"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root123"

    @pytest.mark.asyncio
    async def test_update_prompt_posts_in_thread(self):
        adapter = self._thread_adapter()
        await adapter.send_update_prompt(
            "ch1", "Proceed?", metadata={"thread_id": "root123"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root123"

    @pytest.mark.asyncio
    async def test_clarify_buttons_post_in_thread(self):
        adapter = self._thread_adapter()
        await adapter.send_clarify(
            "ch1", "Pick", ["A", "B"], "cl1", "s1", metadata={"thread_id": "root123"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root123"

    @pytest.mark.asyncio
    async def test_clarify_open_ended_posts_in_thread(self):
        # The open-ended (no choices) clarify path posts plain text, not a
        # button attachment — it must still honor the thread root.
        adapter = self._thread_adapter()
        await adapter.send_clarify(
            "ch1", "What?", None, "cl1", "s1", metadata={"thread_id": "root123"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root123"

    @pytest.mark.asyncio
    async def test_reply_to_message_id_used_as_thread_root(self):
        # When the gateway only supplies root_id (e.g. some send paths), it is
        # honored the same as thread_id.
        adapter = self._thread_adapter()
        await adapter.send_exec_approval(
            "ch1", "cmd", "s1", metadata={"root_id": "root456"},
        )
        assert self._posted_payload(adapter)["root_id"] == "root456"

    @pytest.mark.asyncio
    async def test_reply_root_resolved_to_thread_parent(self):
        # If the supplied candidate is itself a reply, Mattermost requires the
        # *parent* root_id — using the reply's own id raises "Invalid RootId".
        adapter = self._thread_adapter()
        adapter._api_get = AsyncMock(return_value={"id": "reply789", "root_id": "parent000"})
        await adapter.send_exec_approval(
            "ch1", "cmd", "s1", metadata={"thread_id": "reply789"},
        )
        assert self._posted_payload(adapter)["root_id"] == "parent000"

    @pytest.mark.asyncio
    async def test_flat_when_reply_mode_off(self):
        # reply_mode defaults to "off": even with a thread root in metadata,
        # no root_id is attached — prompts stay flat in the channel.
        adapter = _make_adapter()
        assert adapter._reply_mode != "thread"
        adapter._api_post = AsyncMock(return_value={"id": "posted"})
        adapter._api_get = AsyncMock(return_value={})
        await adapter.send_exec_approval(
            "ch1", "cmd", "s1", metadata={"thread_id": "root123"},
        )
        assert "root_id" not in self._posted_payload(adapter)

    @pytest.mark.asyncio
    async def test_no_metadata_stays_flat(self):
        # thread mode but no thread root to attach to → flat, no root_id.
        adapter = self._thread_adapter()
        await adapter.send_exec_approval("ch1", "cmd", "s1", metadata=None)
        assert "root_id" not in self._posted_payload(adapter)


class TestPendingActionsCap:
    @pytest.mark.asyncio
    async def test_evicts_oldest_when_cap_reached(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        # Fill to the cap with synthetic entries.
        for i in range(adapter._MAX_PENDING):
            adapter._pending_actions[f"old{i}"] = {"kind": "update"}
        assert len(adapter._pending_actions) == adapter._MAX_PENDING
        await adapter.send_update_prompt("ch1", "Proceed?")
        # Cap must not be exceeded.
        assert len(adapter._pending_actions) <= adapter._MAX_PENDING
        # The oldest synthetic entry must have been evicted.
        assert "old0" not in adapter._pending_actions


class TestClarifyChoiceCap:
    @pytest.mark.asyncio
    async def test_choices_capped_at_ten(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        many_choices = [f"Option {i}" for i in range(25)]
        await adapter.send_clarify("ch1", "Pick one", many_choices, "cl1", "s1")
        stored = next(iter(adapter._pending_actions.values()))
        assert len(stored["choices"]) == 10


class TestSendExecApprovalOptions:
    """allow_permanent / allow_session / smart_denied / approval_id -- the
    kwargs upstream's send_exec_approval contract grew after this PR was
    first opened (mirrors Discord/Slack button-hiding behavior)."""

    @staticmethod
    def _posted_payload(adapter) -> dict:
        return adapter._api_post.call_args.args[1]

    @classmethod
    def _actions(cls, adapter) -> list:
        return cls._posted_payload(adapter)["props"]["attachments"][0]["actions"]

    @pytest.mark.asyncio
    async def test_default_shows_all_four_buttons(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "rm -rf /", "s1")
        labels = [a["name"] for a in self._actions(adapter)]
        assert labels == ["Allow Once", "Allow Session", "Always Allow", "Deny"]

    @pytest.mark.asyncio
    async def test_smart_denied_hides_session_and_always(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "rm -rf /", "s1", smart_denied=True)
        labels = [a["name"] for a in self._actions(adapter)]
        assert labels == ["Allow Once", "Deny"]
        text = self._posted_payload(adapter)["props"]["attachments"][0]["text"]
        assert "Smart DENY" in text

    @pytest.mark.asyncio
    async def test_allow_session_false_hides_session_and_always(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "rm -rf /", "s1", allow_session=False)
        labels = [a["name"] for a in self._actions(adapter)]
        assert labels == ["Allow Once", "Deny"]

    @pytest.mark.asyncio
    async def test_allow_permanent_false_hides_only_always(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "rm -rf /", "s1", allow_permanent=False)
        labels = [a["name"] for a in self._actions(adapter)]
        assert labels == ["Allow Once", "Allow Session", "Deny"]

    @pytest.mark.asyncio
    async def test_approval_id_stored_in_pending(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "rm -rf /", "s1", approval_id="abc123")
        payload = next(iter(adapter._pending_actions.values()))
        assert payload["approval_id"] == "abc123"


class TestMattermostApprovalIdResolution:
    """Exact approval-ID resolution -- the adapter side of the #29373
    review comment (plugins/platforms/mattermost/adapter.py:930 resolved
    by session only, letting a stale button resolve a newer approval)."""

    @pytest.mark.asyncio
    async def test_resolves_by_exact_id_when_present(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok1"] = {
            "kind": "approval", "session_key": "session1", "approval_id": "aid-1",
        }
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval_by_id", return_value=1) as mock_resolve:
            resp = await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok1", "choice": "once"},
            }))

        mock_resolve.assert_called_once_with("session1", "aid-1", "once")
        body = json.loads(resp.text)
        assert "Approved once by alice" in body["update"]["message"]

    @pytest.mark.asyncio
    async def test_falls_back_to_fifo_when_no_approval_id(self):
        """Pending prompts registered before this adapter stamped
        approval_id (e.g. across a gateway restart) must still resolve."""
        adapter = _make_adapter()
        adapter._pending_actions["tok2"] = {"kind": "approval", "session_key": "session1"}
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok2", "choice": "once"},
            }))

        mock_resolve.assert_called_once_with("session1", "once", resolve_all=False)

    @pytest.mark.asyncio
    async def test_stale_approval_id_shows_expired_message(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok3"] = {
            "kind": "approval", "session_key": "session1", "approval_id": "stale-id",
        }
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval_by_id", return_value=0):
            resp = await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok3", "choice": "once"},
            }))

        body = json.loads(resp.text)
        assert "expired" in body["update"]["message"].lower()


class TestClarifyRendering:
    """Telegram-style rendering: full option text in the post body, short
    numeric button labels -- long options no longer get cut off."""

    @pytest.mark.asyncio
    async def test_full_option_text_in_body_not_truncated(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        long_choice = ("A very long clarify option that would have been " * 3).strip()
        await adapter.send_clarify("ch1", "Pick", [long_choice, "short"], "cl1", "s1")
        payload = adapter._api_post.call_args.args[1]
        text = payload["props"]["attachments"][0]["text"]
        assert long_choice in text

    @pytest.mark.asyncio
    async def test_buttons_are_short_numeric_labels(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_clarify("ch1", "Pick", ["Option one", "Option two"], "cl1", "s1")
        payload = adapter._api_post.call_args.args[1]
        actions = payload["props"]["attachments"][0]["actions"]
        labels = [a["name"] for a in actions]
        assert labels[0] == "1"
        assert labels[1] == "2"
        assert "Other" in labels[-1]

    @pytest.mark.asyncio
    async def test_question_included_alongside_options(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_clarify("ch1", "Which env?", ["staging", "prod"], "cl1", "s1")
        payload = adapter._api_post.call_args.args[1]
        text = payload["props"]["attachments"][0]["text"]
        assert "Which env?" in text
        assert "1. staging" in text
        assert "2. prod" in text


class TestMassMentionHardening:
    """Prompt posts must disable Mattermost mentions -- the echoed command
    or clarify question could contain "@channel"/"@all" and unintentionally
    mass-ping (mirrors the upstream send()/edit_message()/file-upload
    hardening applied to every other post path)."""

    @pytest.mark.asyncio
    async def test_exec_approval_disables_mentions(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_exec_approval("ch1", "@channel rm -rf /", "s1")
        payload = adapter._api_post.call_args.args[1]
        assert payload["props"]["disable_mentions"] is True
        # The mention-disable merge must not clobber the button attachment.
        assert "attachments" in payload["props"]

    @pytest.mark.asyncio
    async def test_open_ended_clarify_disables_mentions(self):
        adapter = _make_adapter(buttons_enabled=False)
        adapter._api_post = AsyncMock(return_value={"id": "p1"})
        await adapter.send_clarify("ch1", "@here what?", None, "cl1", "s1")
        payload = adapter._api_post.call_args.args[1]
        assert payload["props"]["disable_mentions"] is True


class TestApprovalResolution:
    @pytest.mark.asyncio
    async def test_handle_callback_resolves_approval(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok1"] = {"kind": "approval", "session_key": "session1"}
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            resp = await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok1", "choice": "once"},
            }))

        mock_resolve.assert_called_once_with("session1", "once", resolve_all=False)
        body = json.loads(resp.text)
        assert "update" in body
        assert adapter._pending_actions == {}

    @pytest.mark.asyncio
    async def test_double_click_guard(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok2"] = {"kind": "approval", "session_key": "session1"}
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval"):
            await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok2", "choice": "once"},
            }))
            resp2 = await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok2", "choice": "once"},
            }))

        body2 = json.loads(resp2.text)
        assert "already been resolved" in body2["ephemeral_text"]


    @pytest.mark.asyncio
    async def test_transient_failure_reinserts_for_retry(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok_err"] = {"kind": "approval", "session_key": "session1"}
        adapter._lookup_username = AsyncMock(return_value="alice")

        with patch("tools.approval.resolve_gateway_approval", side_effect=RuntimeError("transient")):
            resp = await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok_err", "choice": "once"},
            }))

        assert resp.status == 500
        # Entry must be re-registered so the user can retry.
        assert "tok_err" in adapter._pending_actions


class TestAuth:
    @pytest.mark.asyncio
    async def test_unauthorized_click_returns_403(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_ALLOWED_USERS", "alice")
        adapter = _make_adapter()
        adapter._pending_actions["tok3"] = {"kind": "approval", "session_key": "session1"}

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            resp = await adapter._handle_callback(_make_request({
                "user_id": "bob",
                "context": {"action_id": "tok3", "choice": "once"},
            }))

        assert resp.status == 403
        mock_resolve.assert_not_called()
        assert "tok3" in adapter._pending_actions


class TestSlashConfirm:
    @pytest.mark.asyncio
    async def test_handle_callback_resolves_slash(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok4"] = {
            "kind": "slash",
            "session_key": "session1",
            "confirm_id": "confirm1",
            "channel_id": "ch1",
        }
        adapter._api_post = AsyncMock()

        with patch("tools.slash_confirm.resolve", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = None
            await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok4", "choice": "always"},
            }))

        mock_resolve.assert_awaited_once_with("session1", "confirm1", "always")


class TestUpdatePrompt:
    @pytest.mark.asyncio
    async def test_handle_callback_writes_update_response(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        adapter._pending_actions["tok5"] = {"kind": "update"}

        await adapter._handle_callback(_make_request({
            "user_id": "u1",
            "context": {"action_id": "tok5", "choice": "y"},
        }))

        assert (tmp_path / ".update_response").read_text() == "y"


class TestClarify:
    @pytest.mark.asyncio
    async def test_handle_callback_resolves_choice(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok6"] = {
            "kind": "clarify",
            "clarify_id": "cl1",
            "choices": ["Alpha", "Beta"],
        }

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok6", "choice_index": 1},
            }))

        mock_resolve.assert_called_once_with("cl1", "Beta")

    @pytest.mark.asyncio
    async def test_handle_callback_other_marks_awaiting_text(self):
        adapter = _make_adapter()
        adapter._pending_actions["tok7"] = {
            "kind": "clarify",
            "clarify_id": "cl1",
            "choices": ["Alpha"],
        }

        with patch("tools.clarify_gateway.mark_awaiting_text") as mock_mark:
            await adapter._handle_callback(_make_request({
                "user_id": "u1",
                "context": {"action_id": "tok7", "choice_index": -1},
            }))

        mock_mark.assert_called_once_with("cl1")
