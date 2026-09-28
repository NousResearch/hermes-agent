"""Plain-language Slack progress: card steps, collapsed history, and the 3-minute check-in."""

import time

from gateway import progress_text
from gateway.progress_text import friendly_step, heartbeat_text
from gateway.run_turn_runner import TurnRunner


def test_friendly_step_hides_commands_and_paths():
    assert friendly_step("terminal", "cd /home/aya && rm -rf x") == "Running"
    assert friendly_step("read_file", "/home/aya/secret.env") == "Reading"
    assert friendly_step("web_search", "pump pressure switch") == "Searching the web for pump pressure switch"
    assert friendly_step("mcp__highlevel_busybee__search_contacts", "Jane Doe") == "Using Highlevel Busybee"
    assert friendly_step("custom_tool", "anything") == "Using custom tool"


def test_heartbeat_text_is_plain_and_includes_extra_lines():
    text = heartbeat_text(6, "terminal, read_file", ["🔀 Subagents: 1 of 2 done"])
    assert text == "⏳ Still working · 6 min\nNow: Running\n🔀 Subagents: 1 of 2 done"
    assert heartbeat_text(3, "_thinking") == "⏳ Still working · 3 min"


def test_progress_line_providers_are_isolated():
    calls = []

    def good(**turn):
        calls.append(turn)
        return ["line A"]

    def bad(**turn):
        raise RuntimeError("boom")

    progress_text.register_progress_line_provider(good)
    progress_text.register_progress_line_provider(bad)
    try:
        assert progress_text.extra_progress_lines(chat_id="C1") == ["line A"]
        assert calls == [{"chat_id": "C1"}]
    finally:
        progress_text._providers[:] = [p for p in progress_text._providers if p not in (good, bad)]


def test_task_card_uses_friendly_steps_and_collapses_history():
    st = TurnRunner._TaskCardState(adapter=None)
    for i in range(7):
        st.apply_event({"type": "tool.started", "tool_call_id": f"c{i}", "tool_name": "terminal", "preview": "cd /x"})
        st.apply_event({"type": "tool.completed", "tool_call_id": f"c{i}", "is_error": i == 0})
    st.apply_event({"type": "tool.started", "tool_call_id": "w", "tool_name": "web_search", "preview": "pumps"})
    tasks = st.visible_tasks()
    # the first step failed but later steps succeeded: recovered, no card-wide warning
    assert tasks[0] == {"id": "earlier_steps", "title": "3 earlier steps (1 recovered)", "status": "complete"}
    assert [t["title"] for t in tasks[1:]] == ["Running"] * 4 + ["Searching the web for pumps"]
    assert "cd /x" not in str(tasks)
    assert st.title() == "Working"
    st.started = time.monotonic() - 125
    assert st.title() == "Working · 2 min"
    assert st.fallback_text().startswith("Working · 2 min\n- 3 earlier steps (1 recovered) - done")


def test_error_clears_once_agent_moves_past_it():
    st = TurnRunner._TaskCardState(adapter=None)
    st.apply_event({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal"})
    st.apply_event({"type": "tool.completed", "tool_call_id": "a", "is_error": True})
    assert st.tasks["a"]["status"] == "error"          # still failing: warning shows
    st.apply_event({"type": "tool.started", "tool_call_id": "b", "tool_name": "terminal"})
    assert st.tasks["a"]["status"] == "error"          # retry running: not yet recovered
    st.apply_event({"type": "tool.completed", "tool_call_id": "b"})
    assert st.tasks["a"]["status"] == "complete"       # moved past it: warning clears
    assert st.tasks["a"]["title"] == "Running (error, recovered)"
    assert all(t["status"] != "error" for t in st.visible_tasks())


def test_run_that_ends_on_failure_keeps_the_error():
    st = TurnRunner._TaskCardState(adapter=None)
    st.apply_event({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal"})
    st.apply_event({"type": "tool.completed", "tool_call_id": "a"})
    st.apply_event({"type": "tool.started", "tool_call_id": "b", "tool_name": "terminal"})
    st.apply_event({"type": "tool.completed", "tool_call_id": "b", "is_error": True})
    assert st.tasks["b"]["status"] == "error"
    assert st.tasks["a"]["status"] == "complete" and "recovered" not in st.tasks["a"]


def test_recovered_errors_fold_into_earlier_steps_without_warning():
    st = TurnRunner._TaskCardState(adapter=None)
    st.apply_event({"type": "tool.started", "tool_call_id": "x", "tool_name": "terminal"})
    st.apply_event({"type": "tool.completed", "tool_call_id": "x", "is_error": True})
    for i in range(6):
        st.apply_event({"type": "tool.started", "tool_call_id": f"c{i}", "tool_name": "read_file"})
        st.apply_event({"type": "tool.completed", "tool_call_id": f"c{i}"})
    head = st.visible_tasks()[0]
    assert head == {"id": "earlier_steps", "title": "2 earlier steps (1 recovered)", "status": "complete"}
