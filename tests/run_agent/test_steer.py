"""Tests for AIAgent.steer() — mid-run user message injection.

/steer lets the user add a note to the agent's next tool result without
interrupting the current tool call. The agent sees the note inline with
tool output on its next iteration, preserving message-role alternation
and prompt-cache integrity.
"""
from __future__ import annotations

import threading

import pytest

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__, then install the steer
    state manually — matches the existing object.__new__ stub pattern
    used elsewhere in the test suite.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    return agent


class TestSteerAcceptance:
    def test_accepts_non_empty_text(self):
        agent = _bare_agent()
        assert agent.steer("go ahead and check the logs") is True
        assert agent._pending_steer == "go ahead and check the logs"

    def test_rejects_empty_string(self):
        agent = _bare_agent()
        assert agent.steer("") is False
        assert agent._pending_steer is None

    def test_rejects_whitespace_only(self):
        agent = _bare_agent()
        assert agent.steer("   \n\t  ") is False
        assert agent._pending_steer is None

    def test_rejects_none(self):
        agent = _bare_agent()
        assert agent.steer(None) is False  # type: ignore[arg-type]
        assert agent._pending_steer is None

    def test_strips_surrounding_whitespace(self):
        agent = _bare_agent()
        assert agent.steer("  hello world  \n") is True
        assert agent._pending_steer == "hello world"

    def test_concatenates_multiple_steers_with_newlines(self):
        agent = _bare_agent()
        agent.steer("first note")
        agent.steer("second note")
        agent.steer("third note")
        assert agent._pending_steer == "first note\nsecond note\nthird note"


class TestSteerDrain:
    def test_drain_returns_and_clears(self):
        agent = _bare_agent()
        agent.steer("hello")
        assert agent._drain_pending_steer() == "hello"
        assert agent._pending_steer is None

    def test_drain_on_empty_returns_none(self):
        agent = _bare_agent()
        assert agent._drain_pending_steer() is None


class TestSteerInjection:
    def test_appends_user_turn_after_tool_batch(self):
        agent = _bare_agent()
        agent.steer("please also check auth.log")
        messages = [
            {"role": "user", "content": "what's in /var/log?"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ls output A", "tool_call_id": "a"},
            {"role": "tool", "content": "ls output B", "tool_call_id": "b"},
        ]
        agent._deliver_pending_steer_as_user_turn(messages)
        # Tool results are untouched.
        assert messages[2]["content"] == "ls output A"
        assert messages[3]["content"] == "ls output B"
        # A new user-role message is appended carrying the steer.
        assert messages[-1]["role"] == "user"
        assert "please also check auth.log" in messages[-1]["content"]
        # No tool message contains the steer text (the whole point).
        assert all(
            "please also check auth.log" not in (m.get("content") or "")
            for m in messages if m.get("role") == "tool"
        )
        assert agent._pending_steer is None

    def test_no_op_when_no_steer_pending(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._deliver_pending_steer_as_user_turn(messages)
        assert len(messages) == 2
        assert messages[-1]["content"] == "output"

    def test_restashed_when_tail_is_not_tool(self):
        """If the trailing message is not a tool result (e.g. all tools were
        skipped after an interrupt, or first iteration), inserting a user
        message would be unsafe/wrong — restash for later delivery."""
        agent = _bare_agent()
        agent.steer("ping")
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
        agent._deliver_pending_steer_as_user_turn(messages)
        assert len(messages) == 2
        assert messages[-1]["content"] == "y"
        assert agent._pending_steer == "ping"

    def test_steer_text_carries_provenance_prefix(self):
        agent = _bare_agent()
        agent.steer("stop after next step")
        messages = [{"role": "tool", "content": "x", "tool_call_id": "1"}]
        agent._deliver_pending_steer_as_user_turn(messages)
        assert messages[-1]["role"] == "user"
        assert "/steer" in messages[-1]["content"]
        assert "stop after next step" in messages[-1]["content"]


class TestSteerThreadSafety:
    def test_concurrent_steer_calls_preserve_all_text(self):
        agent = _bare_agent()
        N = 200

        def worker(idx: int) -> None:
            agent.steer(f"note-{idx}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        text = agent._drain_pending_steer()
        assert text is not None
        # Every single note must be preserved — none dropped by the lock.
        lines = text.split("\n")
        assert len(lines) == N
        assert set(lines) == {f"note-{i}" for i in range(N)}


class TestSteerClearedOnInterrupt:
    def test_clear_interrupt_drops_pending_steer(self):
        """A hard interrupt supersedes any pending steer — the agent's
        next tool iteration won't happen, so delivering the steer later
        would be surprising."""
        agent = _bare_agent()
        # Minimal surface needed by clear_interrupt()
        agent._interrupt_requested = True
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None

        agent.steer("will be dropped")
        assert agent._pending_steer == "will be dropped"

        agent.clear_interrupt()
        assert agent._pending_steer is None


class TestPreApiCallSteerDrain:
    """A steer arriving while the model was thinking must land on THIS
    iteration. With messages ending in a tool result, it is delivered as a
    trailing user turn; with no completed tool run, it stays pending."""

    def test_pre_api_drain_appends_user_turn(self):
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "output here", "tool_call_id": "tc1"},
        ]
        agent.steer("focus on error handling")
        agent._deliver_pending_steer_as_user_turn(messages)
        assert messages[-1]["role"] == "user"
        assert "focus on error handling" in messages[-1]["content"]
        assert agent._pending_steer is None

    def test_pre_api_drain_restashes_when_no_tool_message(self):
        agent = _bare_agent()
        messages = [{"role": "user", "content": "hello"}]
        agent.steer("early steer")
        agent._deliver_pending_steer_as_user_turn(messages)
        assert len(messages) == 1
        assert agent._pending_steer == "early steer"


class TestSteerCommandRegistry:
    def test_steer_in_command_registry(self):
        """The /steer slash command must be registered so it reaches all
        platforms (CLI, gateway, TUI autocomplete, Telegram/Slack menus).
        """
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("steer")
        assert cmd is not None
        assert cmd.name == "steer"
        assert cmd.category == "Session"
        assert cmd.args_hint == "<prompt>"

    def test_steer_in_bypass_set(self):
        """When the agent is running, /steer MUST bypass the Level-1
        base-adapter queue so it reaches the gateway runner's /steer
        handler. Otherwise it would be queued as user text and only
        delivered at turn end — defeating the whole point.
        """
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, should_bypass_active_session

        assert "steer" in ACTIVE_SESSION_BYPASS_COMMANDS
        assert should_bypass_active_session("steer") is True


class TestSteerCallSites:
    """Source-level guard: tool_executor delivers the steer only at the
    batch boundary (one aggregate call per execution path), and the old
    per-tool drain / old method name are gone. Per-tool insertion is
    incompatible with safe user-turn delivery (would orphan later tool
    results in repair_message_sequence)."""

    def test_tool_executor_uses_renamed_aggregate_only(self):
        import inspect

        import agent.tool_executor as te

        src = inspect.getsource(te)
        assert "_apply_pending_steer_to_tool_results" not in src
        # Exactly two delivery calls: parallel-path aggregate + sequential-path
        # aggregate. No per-tool drains.
        assert src.count("_deliver_pending_steer_as_user_turn") == 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
