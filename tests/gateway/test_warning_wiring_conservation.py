"""Actual per-turn wiring must retain observers before presentation scope binds."""
from types import SimpleNamespace as NS
from unittest.mock import patch

from agent.status_output import StatusOutputMixin
from agent.notification_presentation import notification_turn
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.config import Platform
from gateway.session import SessionSource


def test_actual_wiring_retains_observers_and_controls_across_muted_turn():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline", chat_type="dm")
    for muted in (False, True, False):
        observed = []
        ctx = TurnContext(source=source, user_config={"display": {"suppress_warning_notifications": True}},
            mute_notification_reply=muted, _hooks_ref=NS(loaded_hooks=[]),
            _status_callback_sync=lambda *a: observed.append(("status", a)))
        holder = NS(_ctx=ctx, _runner=NS(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda _: []),
            _make_bg_review_callbacks=lambda: (lambda _: None, lambda: None),
            _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
            _clarify_callback_sync=lambda *a: "yes",
            _notice_callback_sync=lambda *a: observed.append(("notice", a)),
            _attach_session_title_callback=lambda *a: None)
        agent = StatusOutputMixin()
        agent.suppress_status_output = True
        TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
        with notification_turn(agent, muted=muted, session_id="offline"):
            agent._emit_warning("real warning source")
            agent._emit_notice(NS(level="warn", text="structured diagnostic"))
            assert agent.clarify_callback("Continue?") == "yes"
        assert len(observed) == 2
        assert callable(agent.status_callback) and callable(agent.notice_callback)


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
