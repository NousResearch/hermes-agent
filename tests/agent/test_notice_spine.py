"""Regression tests for the notice-spine (AgentNotice + emitter callbacks).

Covers:
  A. _emit_notice / _emit_notice_clear emitter behaviour (bare AIAgent via
     object.__new__ — same pattern as test_steer.py and test_file_mutation_verifier.py).
  B. Constructor / init_agent signature threading.
  C. TUI _agent_cbs notice binding — mirrors the status_callback tests already
     in tests/tui_gateway/test_tui_gateway_server.py.
"""
from __future__ import annotations

from unittest.mock import patch


from agent.credits_tracker import AgentNotice
from run_agent import AIAgent


# ── A. Emitter behaviour ─────────────────────────────────────────────────────


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__ (no heavy init required).

    Only the two callback slots used by _emit_notice / _emit_notice_clear are
    installed — mirrors the pattern in test_steer.py.
    """
    agent = object.__new__(AIAgent)
    agent.notice_callback = None
    agent.notice_clear_callback = None
    return agent









# ── B. Constructor / init_agent signature threading ─────────────────────────







# ── C. TUI _agent_cbs binding ────────────────────────────────────────────────


class TestAgentCbsNoticeBinding:
    """Mirror test_status_callback_emits_kind_and_text from test_tui_gateway_server.py."""

    def test_notice_callback_emits_notification_show(self):
        from tui_gateway import server

        with patch("tui_gateway.server._emit") as mock_emit:
            cbs = server._agent_cbs("sid123")
            notice = AgentNotice(
                text="credits 90% used",
                level="warn",
                kind="sticky",
                ttl_ms=None,
                key="credits.warn90",
                id="n1",
            )
            cbs["notice_callback"](notice)

        mock_emit.assert_called_once_with(
            "notification.show",
            "sid123",
            {
                "text": "credits 90% used",
                "level": "warn",
                "kind": "sticky",
                "ttl_ms": None,
                "key": "credits.warn90",
                "id": "n1",
            },
        )




    def test_notice_clear_callback_event_type_is_notification_clear(self):
        from tui_gateway import server

        captured = []
        with patch("tui_gateway.server._emit", side_effect=lambda *a: captured.append(a)):
            cbs = server._agent_cbs("sid123")
            cbs["notice_clear_callback"]("some.key")

        assert captured[0][0] == "notification.clear"
        assert captured[0][1] == "sid123"
        assert captured[0][2] == {"key": "some.key"}
