"""Regression tests for the notice-spine (AgentNotice + emitter callbacks).

Covers:
  A. _emit_notice / _emit_notice_clear emitter behaviour (bare AIAgent via
     object.__new__ — same pattern as test_steer.py and test_file_mutation_verifier.py).
  B. Constructor / init_agent signature threading.
  C. TUI _agent_cbs notice binding — mirrors the status_callback tests already
     in tests/test_tui_gateway_server.py.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

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


class TestEmitNotice:
    def test_emit_notice_calls_callback_with_exact_notice(self):
        agent = _bare_agent()
        received = []
        notice = AgentNotice(
            text="credits 90% used",
            level="warn",
            kind="sticky",
            ttl_ms=None,
            key="credits.warn90",
            id="n1",
        )
        agent.notice_callback = received.append
        agent._emit_notice(notice)
        assert received == [notice]

    def test_emit_notice_clear_calls_callback_with_exact_key(self):
        agent = _bare_agent()
        received = []
        agent.notice_clear_callback = received.append
        agent._emit_notice_clear("credits.depleted")
        assert received == ["credits.depleted"]

    def test_emit_notice_swallows_callback_exception(self):
        agent = _bare_agent()

        def _boom(n):
            raise RuntimeError("renderer exploded")

        agent.notice_callback = _boom
        # Must not raise.
        agent._emit_notice(AgentNotice(text="x"))

    def test_emit_notice_clear_swallows_callback_exception(self):
        agent = _bare_agent()

        def _boom(key):
            raise ValueError("clear renderer exploded")

        agent.notice_clear_callback = _boom
        # Must not raise.
        agent._emit_notice_clear("some.key")

    def test_emit_notice_no_op_when_callback_is_none(self):
        agent = _bare_agent()
        agent.notice_callback = None
        # Should not raise AttributeError or anything else.
        agent._emit_notice(AgentNotice(text="x"))

    def test_emit_notice_clear_no_op_when_callback_is_none(self):
        agent = _bare_agent()
        agent.notice_clear_callback = None
        # Should not raise.
        agent._emit_notice_clear("any.key")


# ── B. Constructor / init_agent signature threading ─────────────────────────


class TestSignatureThreading:
    def test_agent_init_exposes_notice_callback(self):
        sig = inspect.signature(AIAgent.__init__)
        assert "notice_callback" in sig.parameters

    def test_agent_init_exposes_notice_clear_callback(self):
        sig = inspect.signature(AIAgent.__init__)
        assert "notice_clear_callback" in sig.parameters

    def test_init_agent_exposes_notice_callback(self):
        from agent.agent_init import init_agent
        sig = inspect.signature(init_agent)
        assert "notice_callback" in sig.parameters

    def test_init_agent_exposes_notice_clear_callback(self):
        from agent.agent_init import init_agent
        sig = inspect.signature(init_agent)
        assert "notice_clear_callback" in sig.parameters


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

    def test_notice_callback_payload_is_full_snake_case_dict(self):
        """All six snake_case fields must be present in the payload — no extras,
        no camelCase variants."""
        from tui_gateway import server

        captured = []
        with patch("tui_gateway.server._emit", side_effect=lambda *a: captured.append(a)):
            cbs = server._agent_cbs("sid123")
            cbs["notice_callback"](
                AgentNotice(
                    text="credits 90% used",
                    level="warn",
                    kind="sticky",
                    ttl_ms=None,
                    key="credits.warn90",
                    id="n1",
                )
            )

        assert len(captured) == 1
        _event_type, _sid, payload = captured[0]
        assert set(payload.keys()) == {"text", "level", "kind", "ttl_ms", "key", "id"}

    def test_notice_clear_callback_emits_notification_clear(self):
        from tui_gateway import server

        with patch("tui_gateway.server._emit") as mock_emit:
            cbs = server._agent_cbs("sid123")
            cbs["notice_clear_callback"]("credits.depleted")

        mock_emit.assert_called_once_with(
            "notification.clear",
            "sid123",
            {"key": "credits.depleted"},
        )

    def test_notice_callback_event_type_is_notification_show(self):
        from tui_gateway import server

        captured = []
        with patch("tui_gateway.server._emit", side_effect=lambda *a: captured.append(a)):
            cbs = server._agent_cbs("sid123")
            cbs["notice_callback"](AgentNotice(text="any"))

        assert captured[0][0] == "notification.show"

    def test_notice_clear_callback_event_type_is_notification_clear(self):
        from tui_gateway import server

        captured = []
        with patch("tui_gateway.server._emit", side_effect=lambda *a: captured.append(a)):
            cbs = server._agent_cbs("sid123")
            cbs["notice_clear_callback"]("some.key")

        assert captured[0][0] == "notification.clear"
        assert captured[0][1] == "sid123"
        assert captured[0][2] == {"key": "some.key"}
