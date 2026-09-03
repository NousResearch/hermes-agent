"""Tests for Telegram inline keyboard approval buttons."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    """Create a TelegramAdapter with mocked internals."""
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class _AuthRunner:
    """Minimal runner shim for callback auth tests."""

    def __init__(self, authorized: bool):
        self.authorized = authorized
        self.last_source = None

    async def _handle_message(self, event):
        return None

    def _is_user_authorized(self, source):
        self.last_source = source
        return self.authorized


# ===========================================================================
# send_exec_approval — inline keyboard buttons
# ===========================================================================

class TestTelegramExecApproval:
    """Test the send_exec_approval method sends InlineKeyboard buttons."""

    @pytest.mark.asyncio
    async def test_sends_inline_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /important",
            session_key="agent:main:telegram:group:12345:99",
            description="dangerous deletion",
        )

        assert result.success is True
        assert result.message_id == "42"

        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "rm -rf /important" in kwargs["text"]
        assert "dangerous deletion" in kwargs["text"]
        assert kwargs["reply_markup"] is not None  # InlineKeyboardMarkup


    @pytest.mark.asyncio
    async def test_non_smart_allow_permanent_false_keeps_session(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        buttons = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: buttons.append(text) or text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", lambda rows: rows
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False,
        )

        assert buttons == ["✅ Allow Once", "✅ Session", "❌ Deny"]

    @pytest.mark.asyncio
    async def test_full_approval_keyboard_is_two_by_two(self, monkeypatch):
        """Regression: d48bf743f flattened all buttons into one row (4x1)."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
        )

        assert captured_rows == [
            ["✅ Allow Once", "✅ Session"],
            ["✅ Always", "❌ Deny"],
        ]


    @pytest.mark.asyncio
    async def test_smart_deny_two_buttons_share_one_row(self, monkeypatch):
        """smart_deny yields 2 buttons — they pair into a single readable row."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False, smart_denied=True,
        )

        assert captured_rows == [
            ["✅ Allow Once", "❌ Deny"],
        ]


    @pytest.mark.asyncio
    async def test_send_update_prompt_escapes_dynamic_prompt(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=55)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_update_prompt(
            chat_id="12345",
            prompt="Fix [issue]_1 and verify *markdown*",
            default="alpha_beta",
            metadata={"thread_id": "999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "Fix \\[issue\\]\\_1" in sent["text"]
        assert "alpha\\_beta" in sent["text"]

# _handle_callback_query — approval button clicks
# ===========================================================================

class TestTelegramApprovalCallback:
    """Test the approval callback handling in _handle_callback_query."""


    @pytest.mark.asyncio
    async def test_resume_typing_after_inline_approval(self):
        """Clicking an inline approval button must un-pause the chat's typing.

        Regression for #27853: the text /approve path resumed typing, but the
        ea: callback path did not, so the typing indicator stayed gone for the
        rest of a long-running turn after a button click.
        """
        adapter = _make_adapter()
        adapter._approval_state[5] = "agent:main:telegram:group:12345:99"
        adapter.pause_typing_for_chat("12345")
        assert "12345" in adapter._typing_paused

        query = AsyncMock()
        query.data = "ea:once:5"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        assert "12345" not in adapter._typing_paused


    @pytest.mark.asyncio
    async def test_approval_callback_escapes_dynamic_user_name(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Alice_Bob"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "Alice\\_Bob" in edit_kwargs["text"]
        assert "Approved once" in edit_kwargs["text"]


    @pytest.mark.asyncio
    async def test_update_prompt_callback_not_affected(self, tmp_path):
        """Ensure update prompt callbacks still work."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                # Allow the caller — the new fail-closed allowlist gate
                # (#24457) rejects empty TELEGRAM_ALLOWED_USERS, but this
                # test isn't exercising that gate; it's verifying the
                # update_prompt callback still writes the response.
                with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
                    await adapter._handle_callback_query(update, context)

        # Should NOT have triggered approval resolution
        mock_resolve.assert_not_called()
        assert (tmp_path / ".update_response").read_text() == "y"

    @pytest.mark.asyncio
    async def test_wisdom_update_callback_plans_and_applies_compatible_update(
        self, monkeypatch
    ):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:plan:update:skill-3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.update_plan.return_value = {
            "receipt": "wup_deadbeef",
            "skill_id": "skill-3",
            "slug": "team-runbook",
            "version": 3,
            "compatibility": {"outcome": "compatible"},
            "modified": False,
            "sensitive_expansion": [],
        }
        service.update_apply.return_value = {
            "skill_id": "skill-3",
            "slug": "team-runbook",
            "version": 3,
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        assert service.require_setup.call_count == 2
        service.update_plan.assert_called_once_with("skill-3")
        service.update_apply.assert_called_once_with(
            "wup_deadbeef",
            accept_sensitive=False,
            accept_partial=False,
            preserve_modified=False,
        )
        edit_kwargs = query.edit_message_text.await_args.kwargs
        assert "team-runbook v3 updated" in edit_kwargs["text"]
        assert edit_kwargs["reply_markup"] is None

    @pytest.mark.asyncio
    async def test_wisdom_install_callback_uses_org_default_and_owning_profile(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_adapter()
        adapter.set_owner_profile("customer-b")
        query = AsyncMock()
        query.data = "wi:plan:install:skill-4"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.install_plan.return_value = {
            "receipt": "wip_deadbeef",
            "skill_id": "skill-4",
            "slug": "release-checklist",
            "version": 2,
            "compatibility": {"outcome": "compatible"},
            "allowed": True,
        }
        service.install_apply.return_value = {
            "skill_id": "skill-4",
            "slug": "release-checklist",
            "version": 2,
        }
        entered_profiles = []

        class _ProfileScope:
            def __init__(self, home):
                self.home = home

            def __enter__(self):
                entered_profiles.append(self.home)

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda profile: tmp_path / "profiles" / profile,
        )
        monkeypatch.setattr("gateway.run._profile_runtime_scope", _ProfileScope)

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        service.install_plan.assert_called_once_with("skill-4", update_mode=None)
        service.install_apply.assert_called_once_with(
            "wip_deadbeef", accept_partial=False
        )
        assert entered_profiles == [
            tmp_path / "profiles" / "customer-b",
            tmp_path / "profiles" / "customer-b",
        ]
        assert "release-checklist v2 installed" in (
            query.edit_message_text.await_args.kwargs["text"]
        )

    @pytest.mark.asyncio
    async def test_wisdom_candidate_notification_is_exact_session_and_non_consuming(
        self, tmp_path
    ):
        from hermes_wisdom.store import WisdomStore

        store = WisdomStore(tmp_path / "wisdom")
        store.installation_identity()
        store.verify_installation_identity("org-1")
        store.record_organization_display_name_check("org-1", "Nous Research")
        skill = tmp_path / "telegram-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("# Telegram skill\n", encoding="utf-8")
        skill_id = store.register_skill(
            skill, content_hash="sha256:source", source_kind="local"
        )
        event_id = store.emit_local_event(
            kind="wisdom.candidate",
            skill_id=skill_id,
            content_hash="sha256:source",
            payload={
                "skill_name": "telegram-skill",
                "editorial_name": "Telegram Workflow",
                "editorial_description": "Reuse a reliable Telegram workflow.",
                "local_reasons": {
                    "consecutive_business_days": 7,
                    "business_day_timezone": "Australia/Brisbane",
                },
            },
            session_id="telegram-session",
            task_id="task-1",
            qualification="high_usage",
        )
        assert event_id is not None
        adapter = _make_adapter()

        with patch("hermes_wisdom.store.WisdomStore", return_value=store):
            assert (
                await adapter.send_wisdom_candidate_notifications(
                    "12345", "other-session"
                )
                == 0
            )
            assert (
                await adapter.send_wisdom_candidate_notifications(
                    "12345", "telegram-session"
                )
                == 1
            )

        raw_call = adapter._bot.do_api_request.await_args
        assert raw_call.args == ("sendRichMessage",)
        html = raw_call.kwargs["api_kwargs"]["rich_message"]["html"]
        assert "Telegram Workflow" in html
        assert "Reuse a reliable Telegram workflow." in html
        assert "Your organisation (Nous Research) has enabled Collective Wisdom" in html
        assert "Congratulations! Hermes detected a skill" in html
        assert "Why suggested:" in html
        assert "consistently across consecutive business days" in html
        assert "consecutive_business_days" not in html
        assert "Australia/Brisbane" not in html
        assert "Would you like to share?" in html
        assert "Review first" in html
        assert "Yes" in html
        assert html.index("Not Now") < html.index("Review first") < html.index("Yes")
        assert f"wi:defer:{event_id}" in html
        assert f"wi:draft:{event_id}" in html
        assert f"wi:publish:{event_id}" in html
        assert store.pending_telegram_events(
            kind="wisdom.candidate", session_id="telegram-session"
        ) == []
        assert [
            item["id"]
            for item in store.local_events(
                kind="wisdom.candidate", session_id="telegram-session"
            )
        ] == [event_id]

    def test_wisdom_candidate_returning_copy_and_mobile_keyboard_order(
        self, monkeypatch
    ):
        from hermes_wisdom.notice import qualification_notice

        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, **kwargs: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )
        notice = qualification_notice({"notice_variant": "returning"})
        actions = [
            {"label": "Not Now", "callback_data": "wi:defer:event-2"},
            {"label": "Review first", "callback_data": "wi:draft:event-2"},
            {
                "label": "Yes",
                "callback_data": "wi:publish:event-2",
                "primary": True,
            },
        ]
        html = TelegramAdapter._wisdom_candidate_html(
            skill_name="another-skill",
            qualification_reason="It met the local rules.",
            status=(
                f"{notice}\n\nNothing is shared without your approval.\n\n"
                "Would you like to share?"
            ),
            actions=actions,
        )
        keyboard = TelegramAdapter._wisdom_candidate_keyboard(actions)

        assert "Hermes detected <b>another</b> skill" in html
        assert keyboard is not None
        assert captured_rows == [["Not Now", "Review first", "Yes"]]

    def test_wisdom_candidate_reason_explains_refinement_without_raw_evidence(self):
        reason = TelegramAdapter._wisdom_candidate_qualification_reason("refinement")

        assert "refined this skill repeatedly" in reason
        assert "used it recently" in reason
        assert "remained stable" in reason
        assert "3" not in reason
        assert "7" not in reason

    @pytest.mark.asyncio
    async def test_wisdom_candidate_draft_callback_adds_portal_and_publish_actions(
        self,
    ):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:draft:event-1"
        query.message = MagicMock(chat_id=12345, message_id=77)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.draft_candidate.return_value = {
            "draft_id": "draft-1",
            "skill_name": "telegram-skill",
            "state": "ready",
            "portal_url": "https://portal.test/review/draft-1",
            "created": True,
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        service.draft_candidate.assert_called_once_with("event-1")
        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "Private draft created" in html
        assert "Yes" in html
        assert "https://portal.test/review/draft-1" in html
        assert "wi:defer:event-1" in html
        assert html.index("Not Now") < html.index("View") < html.index("Yes")

    @pytest.mark.asyncio
    async def test_wisdom_candidate_not_now_defers_without_declining(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:defer:event-1"
        query.message = MagicMock(chat_id=12345, message_id=79)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.defer_candidate_prompt.return_value = {
            "skill_name": "telegram-skill",
            "qualification": "high_usage",
            "state": "deferred",
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        service.defer_candidate_prompt.assert_called_once_with(
            "event-1", surface="telegram"
        )
        service.decline_candidate.assert_not_called()
        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "Not sharing right now" in html
        assert "Collective Wisdom" in html

    @pytest.mark.asyncio
    async def test_wisdom_candidate_publish_callback_reports_moderation_state(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:publish:event-1"
        query.message = MagicMock(chat_id=12345, message_id=78)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.approve_candidate.return_value = {
            "skill_name": "telegram-skill",
            "publication_state": "pending_moderation",
            "portal_url": "https://portal.test/review/draft-1",
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        service.approve_candidate.assert_called_once_with("event-1")
        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "collective administrator" in html
        assert "https://portal.test/review/draft-1" in html

    @pytest.mark.asyncio
    async def test_wisdom_candidate_stale_publish_button_reports_portal_winner(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:publish:event-1"
        query.message = MagicMock(chat_id=12345, message_id=78)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.approve_candidate.return_value = {
            "skill_name": "telegram-skill",
            "publication_state": "published",
            "already_advanced": True,
            "portal_url": "https://portal.test/review/draft-1",
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "already published to your collective" in html
        assert "https://portal.test/review/draft-1" in html
        assert "Approve &amp; publish" not in html
        assert "wi:publish:event-1" not in html

    @pytest.mark.asyncio
    async def test_wisdom_candidate_changes_requested_removes_publish_action(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:publish:event-1"
        query.message = MagicMock(chat_id=12345, message_id=78)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.approve_candidate.return_value = {
            "skill_name": "telegram-skill",
            "publication_state": "changes_requested",
            "already_advanced": True,
            "portal_url": "https://portal.test/review/draft-1",
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "requested changes" in html
        assert "Approve &amp; publish" not in html
        assert "https://portal.test/review/draft-1" in html

    @pytest.mark.asyncio
    async def test_wisdom_candidate_transient_failure_preserves_retry_controls(self):
        adapter = _make_adapter()
        adapter._bot.do_api_request.reset_mock()
        query = AsyncMock()
        query.data = "wi:publish:event-1"
        query.message = MagicMock(chat_id=12345, message_id=78)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.approve_candidate.side_effect = TimeoutError("Gateway timed out")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        adapter._bot.do_api_request.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wisdom_candidate_decline_callback_suppresses_exact_bytes(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:decline:event-1"
        query.message = MagicMock(chat_id=12345, message_id=79)
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        service = MagicMock()
        service.decline_candidate.return_value = {
            "skill_name": "telegram-skill",
            "state": "declined",
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query), MagicMock()
                )

        service.decline_candidate.assert_called_once_with("event-1")
        html = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"][
            "rich_message"
        ]["html"]
        assert "exact bytes will not be suggested again" in html

    @pytest.mark.asyncio
    async def test_wisdom_install_preserves_rich_notification_and_disables_action(
        self,
    ):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:plan:install:skill-4"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 77
        query.message.rich_message = None
        query.message.reply_markup = None
        query.message.api_kwargs = {
            "rich_message": {
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": [
                            "New skill from your team\nrelease-checklist · v2\n",
                            {
                                "type": "button",
                                "button": {
                                    "text": "Install",
                                    "style": "primary",
                                    "callback_data": "wi:plan:install:skill-4",
                                },
                            },
                            " ",
                            {
                                "type": "button",
                                "button": {
                                    "text": "View ↗",
                                    "url": "https://portal.test/release-checklist",
                                },
                            },
                        ],
                    },
                    {
                        "type": "paragraph",
                        "text": [
                            "Update available\nteam-runbook · v3\n",
                            {
                                "type": "button",
                                "button": {
                                    "text": "Update",
                                    "style": "primary",
                                    "callback_data": "wi:plan:update:skill-3",
                                },
                            },
                        ],
                    },
                ]
            }
        }
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.install_plan.return_value = {
            "receipt": "wip_deadbeef",
            "skill_id": "skill-4",
            "slug": "release-checklist",
            "version": 2,
            "compatibility": {"outcome": "compatible"},
            "allowed": True,
        }
        service.install_apply.return_value = {
            "skill_id": "skill-4",
            "slug": "release-checklist",
            "version": 2,
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        adapter._bot.do_api_request.assert_awaited_once()
        raw_call = adapter._bot.do_api_request.await_args
        assert raw_call.args == ("editMessageText",)
        payload = raw_call.kwargs["api_kwargs"]
        assert payload["chat_id"] == 12345
        assert payload["message_id"] == 77
        rich_message = payload["rich_message"]
        install_button = rich_message["blocks"][0]["text"][1]["button"]
        assert install_button == {
            "text": "✓ Installed",
            "style": "success",
            "disabled": {},
        }
        assert rich_message["blocks"][0]["text"][3]["button"] == {
            "text": "View ↗",
            "url": "https://portal.test/release-checklist",
        }
        assert rich_message["blocks"][1]["text"][1]["button"] == {
            "text": "Update",
            "style": "primary",
            "callback_data": "wi:plan:update:skill-3",
        }
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wisdom_update_preserves_fallback_keyboard_and_disables_action(
        self,
    ):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:plan:update:skill-3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 78
        query.message.rich_message = None
        query.message.api_kwargs = {}
        query.message.reply_markup.to_dict.return_value = {
            "inline_keyboard": [
                [
                    {
                        "text": "Update",
                        "callback_data": "wi:plan:update:skill-3",
                        "style": "primary",
                    },
                    {"text": "View ↗", "url": "https://portal.test/team-runbook"},
                ],
                [
                    {
                        "text": "Install",
                        "callback_data": "wi:plan:install:skill-4",
                    }
                ],
            ]
        }
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.update_plan.return_value = {
            "receipt": "wup_deadbeef",
            "skill_id": "skill-3",
            "slug": "team-runbook",
            "version": 3,
            "compatibility": {"outcome": "compatible"},
            "modified": False,
            "sensitive_expansion": [],
        }
        service.update_apply.return_value = {
            "skill_id": "skill-3",
            "slug": "team-runbook",
            "version": 3,
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        adapter._bot.do_api_request.assert_awaited_once()
        raw_call = adapter._bot.do_api_request.await_args
        assert raw_call.args == ("editMessageReplyMarkup",)
        keyboard = raw_call.kwargs["api_kwargs"]["reply_markup"]["inline_keyboard"]
        assert keyboard[0][0] == {
            "text": "✓ Updated",
            "style": "success",
            "disabled": {},
        }
        assert keyboard[0][1] == {
            "text": "View ↗",
            "url": "https://portal.test/team-runbook",
        }
        assert keyboard[1][0] == {
            "text": "Install",
            "callback_data": "wi:plan:install:skill-4",
        }
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wisdom_callback_does_not_apply_when_full_review_is_required(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:plan:update:skill-3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.update_plan.return_value = {
            "receipt": "wup_deadbeef",
            "skill_id": "skill-3",
            "slug": "team-runbook",
            "version": 3,
            "compatibility": {"outcome": "partial"},
            "modified": False,
            "sensitive_expansion": [],
        }

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        service.update_apply.assert_not_called()
        assert "needs a full review" in (
            query.edit_message_text.await_args.kwargs["text"]
        )

    @pytest.mark.asyncio
    async def test_wisdom_update_confirmation_applies_exact_receipt(self):
        adapter = _make_adapter()
        query = AsyncMock()
        query.data = "wi:confirm:update:wup_deadbeef"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock(id="12345", first_name="Shannon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        service = MagicMock()
        service.update_apply.return_value = {"skill_id": "skill-3", "version": 3}

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_wisdom.service.WisdomService", return_value=service):
                await adapter._handle_callback_query(update, MagicMock())

        service.update_apply.assert_called_once_with(
            "wup_deadbeef",
            accept_sensitive=False,
            accept_partial=False,
            preserve_modified=False,
        )
        assert "Skill updated v3" in query.edit_message_text.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_unauthorized_user(self, tmp_path):
        """Update prompt buttons should honor TELEGRAM_ALLOWED_USERS."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_user_blocked_by_global_allowlist(self, tmp_path):
        adapter = _make_adapter()
        runner = _AuthRunner(authorized=False)
        adapter._message_handler = runner._handle_message

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.from_user.first_name = "Mallory"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()
        assert runner.last_source is not None
        assert runner.last_source.platform == Platform.TELEGRAM
        assert runner.last_source.user_id == "222"
