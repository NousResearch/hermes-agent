from types import SimpleNamespace

from agent.conversation_loop import _relay_available_reasoning


def test_available_reasoning_relay_preserves_full_content():
    events = []
    reasoning = "reasoning " * 80
    agent = SimpleNamespace(
        _delegate_depth=0,
        tool_progress_callback=lambda *args: events.append(args),
    )

    _relay_available_reasoning(
        agent,
        f"<REASONING_SCRATCHPAD>{reasoning}</REASONING_SCRATCHPAD>",
    )

    assert events == [("reasoning.available", "_thinking", reasoning.strip(), None)]
    assert len(events[0][2]) > 500


def test_available_reasoning_relay_keeps_subagent_preview_bounded():
    events = []
    agent = SimpleNamespace(
        _delegate_depth=1,
        tool_progress_callback=lambda *args: events.append(args),
    )

    _relay_available_reasoning(agent, "first line " * 20 + "\nsecond line")

    assert events == [("_thinking", ("first line " * 20)[:80])]
