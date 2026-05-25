"""Cursor SDK runtime injects the Hermes system prompt on the first turn."""

from __future__ import annotations

from types import SimpleNamespace

from agent.cursor_runtime import compose_cursor_user_input, run_cursor_sdk_turn
from agent.transports.cursor_sdk_session import TurnResult


def test_compose_cursor_user_input_injects_on_first_turn_only():
    agent = SimpleNamespace(
        _cached_system_prompt="SYSTEM RULES",
        ephemeral_system_prompt=None,
    )
    first = compose_cursor_user_input(
        agent,
        "work kanban task t_1",
        inject_system=True,
    )
    assert "SYSTEM RULES" in first
    assert "operating instructions" in first
    assert "work kanban task t_1" in first
    assert first.index("SYSTEM RULES") < first.index("work kanban task t_1")

    second = compose_cursor_user_input(
        agent,
        "follow up",
        inject_system=False,
    )
    assert second == "follow up"
    assert "SYSTEM RULES" not in second


def test_compose_cursor_user_input_includes_ephemeral_overlay():
    agent = SimpleNamespace(
        _cached_system_prompt="BASE",
        ephemeral_system_prompt="EPHEMERAL",
    )
    text = compose_cursor_user_input(agent, "hi", inject_system=True)
    assert "BASE" in text
    assert "EPHEMERAL" in text


def test_run_cursor_sdk_turn_injects_kanban_guidance_first_turn_only(
    monkeypatch,
):
    """Kanban workers must see KANBAN_GUIDANCE on turn 0, not on turn 1."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    captured: list[str] = []

    def _fake_run_turn(self, user_input, **kwargs):
        captured.append(user_input)
        return TurnResult(final_text="done")

    monkeypatch.setattr(
        "agent.transports.cursor_sdk_session.preflight_cursor_sdk",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "agent.transports.cursor_sdk_session.CursorSDKSession.run_turn",
        _fake_run_turn,
    )

    agent = SimpleNamespace(
        api_mode="cursor_sdk_runtime",
        session_id="sess-1",
        model="composer-2.5",
        api_key="test-key",
        session_cwd=".",
        quiet_mode=True,
        _cached_system_prompt=f"Base prompt\n\n{KANBAN_GUIDANCE}",
        ephemeral_system_prompt=None,
        _cursor_session=None,
        thinking_callback=None,
        tool_progress_callback=None,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names={"kanban_show"},
        _interrupt_requested=False,
        _sync_external_memory_for_turn=lambda **kwargs: None,
        _spawn_background_review=lambda **kwargs: None,
    )

    messages: list = []
    run_cursor_sdk_turn(
        agent,
        user_message="work kanban task t_fake",
        original_user_message="work kanban task t_fake",
        messages=messages,
        effective_task_id="task-1",
    )
    run_cursor_sdk_turn(
        agent,
        user_message="continue please",
        original_user_message="continue please",
        messages=messages,
        effective_task_id="task-1",
    )

    assert len(captured) == 2
    assert "Kanban task execution protocol" in captured[0]
    assert "Review" in captured[0]
    assert "status = review" in captured[0]
    assert "work kanban task t_fake" in captured[0]
    assert captured[1] == "continue please"
    assert "Kanban task execution protocol" not in captured[1]
