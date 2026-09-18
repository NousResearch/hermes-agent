"""Composition-only Preview runs must not participate in Kanban worker lifecycle."""

from types import SimpleNamespace
from unittest.mock import Mock


def test_composition_only_blocks_kanban_runtime_side_effects(monkeypatch):
    from agent.activity_tracking import ActivityTrackingMixin
    from agent.turn_finalizer import _resolve_budget_fallback
    from agent.turn_stop_gates import _kanban_stop_nudge
    from tools import kanban_tools
    import agent.turn_finalizer as finalizer

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_preview")
    heartbeat = Mock()
    comments = Mock()
    record_timeout = Mock()
    monkeypatch.setattr(kanban_tools, "heartbeat_current_worker_from_env", heartbeat)
    monkeypatch.setattr(kanban_tools, "inject_new_comments_from_env", comments)
    monkeypatch.setattr(finalizer, "_record_kanban_budget_exhausted", record_timeout)

    class PreviewAgent(ActivityTrackingMixin):
        _composition_only = True
        max_iterations = 1
        iteration_budget = SimpleNamespace(remaining=0)
        quiet_mode = True

        def _persist_session_activity_if_due(self):
            pass

    agent = PreviewAgent()
    ActivityTrackingMixin._touch_activity(agent, "preview composition")

    assert _kanban_stop_nudge(agent, []) is None
    _resolve_budget_fallback(
        agent,
        final_response="preview complete",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[],
        _turn_exit_reason="budget_exhausted",
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
        logger=Mock(),
    )

    heartbeat.assert_not_called()
    comments.assert_not_called()
    record_timeout.assert_not_called()