"""Actual per-turn wiring must retain observers before presentation scope binds."""
from types import SimpleNamespace as NS
from unittest.mock import patch

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.config import Platform
from gateway.session import SessionSource




def test_concrete_gateway_sinks_hide_all_freeform_muted_turn_output():
    for muted in (False, True, False):
        scheduled = []
        ctx = TurnContext(source=SessionSource(platform=Platform.TELEGRAM, chat_id="offline"),
            user_config={}, mute_notification_reply=muted)
        holder = NS(_ctx=ctx, _status_live=lambda: True,
            _schedule=lambda coro, *args: scheduled.append(coro),
            _runner=NS(_deliver_platform_notice=lambda *args: "notice-send"))
        with patch("gateway.run._prepare_gateway_status_message", lambda *args: "prepared"), \
             patch("gateway.run._send_or_update_status_coro", lambda *args: "status-send"), \
             patch("gateway.run.render_notice_line", lambda notice: notice.text):
            TurnRunner._status_callback_sync(holder, "lifecycle", "ordinary progress")
            TurnRunner._status_callback_sync(holder, "warn", "warning")
            TurnRunner._notice_callback_sync(holder, NS(level="info", text="info"))
            TurnRunner._notice_callback_sync(holder, NS(level="warn", text="warning"))
        assert len(scheduled) == (0 if muted else 4)
