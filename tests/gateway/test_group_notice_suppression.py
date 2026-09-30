"""Gateway/system notices must never land in a shared WhatsApp/Telegram/etc group or channel.

Evidence (t_e0761f44): a group chat was spammed with the busy-interrupt ack's "First-time tip"
suffix and would equally receive first-contact onboarding / no-home-channel setup prompts, since
none of those one-time notices previously checked ``source.chat_type``. All three are gated on
``gateway.session.is_group_notice_source`` — this file locks that gate in place per code path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, is_group_notice_source
from gateway.turn_context import TurnContext


def _source(chat_type: str, platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(platform=platform, chat_id="c1", user_id="u1", chat_type=chat_type)


def _event(chat_type: str, platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=_source(chat_type, platform), message_id="m1",
    )


# ---------------------------------------------------------------------------
# is_group_notice_source
# ---------------------------------------------------------------------------


class TestIsGroupNoticeSource:
    @pytest.mark.parametrize("chat_type", ["group", "channel"])
    def test_group_and_channel_are_group_sources(self, chat_type):
        assert is_group_notice_source(_source(chat_type)) is True

    @pytest.mark.parametrize("chat_type", ["dm", "thread", ""])
    def test_dm_and_other_types_are_not_group_sources(self, chat_type):
        assert is_group_notice_source(_source(chat_type)) is False

    def test_missing_chat_type_defaults_to_not_group(self):
        assert is_group_notice_source(SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# _compose_busy_ack_message — first-time /busy tip
# ---------------------------------------------------------------------------


def _busy_ack_kwargs():
    return dict(
        is_steer_mode=False, is_queue_mode=True, is_redirect_mode=False,
        demoted_for_subagents=False, demoted_for_compression=False,
    )


class TestBusyAckGroupSuppression:
    def test_busy_tip_suppressed_in_group(self, tmp_path, monkeypatch):
        from gateway.config import GatewayConfig

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = GatewayRunner(config=GatewayConfig())
        message = runner._compose_busy_ack_message(
            _event("group"), 0.0, None, SimpleNamespace(), **_busy_ack_kwargs()
        )
        assert "First-time tip" not in message
        assert "tip only shows once" not in message
        # The onboarding flag must NOT be consumed — a later DM should still get the tip.
        assert not (tmp_path / "config.yaml").exists()

    def test_busy_tip_still_fires_in_dm(self, tmp_path, monkeypatch):
        from gateway.config import GatewayConfig

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = GatewayRunner(config=GatewayConfig())
        message = runner._compose_busy_ack_message(
            _event("dm"), 0.0, None, SimpleNamespace(), **_busy_ack_kwargs()
        )
        assert "First-time tip" in message


# ---------------------------------------------------------------------------
# _hmwa_first_contact_notes — intro / profile-build / no-home-channel prompt
# ---------------------------------------------------------------------------


class _FakeFirstContactSelf:
    def __init__(self, has_any_sessions: bool):
        self.async_session_store = SimpleNamespace(has_any_sessions=AsyncMock(return_value=has_any_sessions))
        self.config = SimpleNamespace(get_home_channel=MagicMock(return_value=None))
        self._deliver_platform_notice = AsyncMock()


@pytest.mark.asyncio
class TestFirstContactNotesGroupSuppression:
    async def test_no_notices_for_group_first_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", "")
        fake_self = _FakeFirstContactSelf(has_any_sessions=False)
        notes: list = []

        await GatewayTurnMixin._hmwa_first_contact_notes(fake_self, _source("group"), [], notes)

        assert notes == []
        fake_self._deliver_platform_notice.assert_not_awaited()

    async def test_notices_still_fire_for_dm_first_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", "")
        fake_self = _FakeFirstContactSelf(has_any_sessions=False)
        notes: list = []

        await GatewayTurnMixin._hmwa_first_contact_notes(fake_self, _source("dm"), [], notes)

        assert notes  # intro / profile-build note appended
        fake_self._deliver_platform_notice.assert_awaited_once()
        assert "No home channel" in fake_self._deliver_platform_notice.await_args.args[1]

    async def test_group_with_existing_history_is_a_noop_either_way(self):
        fake_self = _FakeFirstContactSelf(has_any_sessions=False)
        notes: list = []

        await GatewayTurnMixin._hmwa_first_contact_notes(fake_self, _source("group"), ["prior"], notes)

        assert notes == []
        fake_self._deliver_platform_notice.assert_not_awaited()
        fake_self.async_session_store.has_any_sessions.assert_not_awaited()


# ---------------------------------------------------------------------------
# _progress_onboarding_hint — tool-progress "/verbose" tip
# ---------------------------------------------------------------------------


class TestToolProgressHintGroupSuppression:
    def test_hint_suppressed_in_group(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {"display": {"tool_progress_command": True}},
        )
        ctx = TurnContext(
            source=_source("group"), progress_mode="all", progress_queue=MagicMock(),
        )
        runner = TurnRunner(SimpleNamespace(), ctx)

        runner._progress_onboarding_hint({"duration": 999})

        ctx.progress_queue.put.assert_not_called()
        assert ctx.long_tool_hint_fired[0] is False

    def test_hint_still_fires_in_dm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {"display": {"tool_progress_command": True}},
        )
        ctx = TurnContext(
            source=_source("dm"), progress_mode="all", progress_queue=MagicMock(),
        )
        runner = TurnRunner(SimpleNamespace(), ctx)

        runner._progress_onboarding_hint({"duration": 999})

        ctx.progress_queue.put.assert_called_once()
        assert ctx.long_tool_hint_fired[0] is True
