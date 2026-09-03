"""Tests for agent/side_question.py — the /btw context-aware side question engine."""

from unittest.mock import patch

from agent.side_question import (
    SIDE_QUESTION_TASK,
    answer_side_question,
    render_history_for_side_question,
    trim_snapshot_for_fork,
)


class TestRenderHistory:
    def test_empty_history(self):
        assert render_history_for_side_question([]) == "(no prior conversation)"
        assert render_history_for_side_question(None) == "(no prior conversation)"

    def test_basic_roles(self):
        history = [
            {"role": "system", "content": "SYSTEM PROMPT — must not appear"},
            {"role": "user", "content": "fix the bug in foo.py"},
            {
                "role": "assistant",
                "content": "Looking now.",
                "tool_calls": [
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "patch"}},
                ],
            },
            {"role": "tool", "content": "Traceback: ValueError in foo.py line 3"},
            {"role": "assistant", "content": "Fixed it."},
        ]
        out = render_history_for_side_question(history)
        assert "SYSTEM PROMPT" not in out
        assert "USER: fix the bug in foo.py" in out
        assert "ASSISTANT [called tools: read_file, patch]" in out
        assert "TOOL RESULT: Traceback: ValueError in foo.py line 3" in out
        assert "ASSISTANT: Fixed it." in out

    def test_structured_content_blocks(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "hello there"}]},
        ]
        out = render_history_for_side_question(history)
        assert "USER: hello there" in out

    def test_newest_biased_truncation(self):
        history = [
            {"role": "user", "content": f"message number {i} " + "x" * 400}
            for i in range(200)
        ]
        out = render_history_for_side_question(history, char_budget=3000)
        # Newest messages survive; oldest are dropped with a marker.
        assert "message number 199" in out
        assert "message number 0 " not in out
        assert out.startswith("[...older conversation omitted...]")
        assert len(out) < 4000

    def test_non_dict_entries_ignored(self):
        out = render_history_for_side_question(["garbage", None, 42, {"role": "user", "content": "hi"}])
        assert "USER: hi" in out


class TestAnswerSideQuestion:
    def test_empty_question_raises(self):
        try:
            answer_side_question("   ", [])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for empty question")

    def test_calls_oneshot_with_snapshot_and_task(self):
        captured = {}

        def fake_run_oneshot(**kwargs):
            captured.update(kwargs)
            return "the error was in foo.py"

        history = [{"role": "user", "content": "run the tests"}]
        runtime = {"model": "m", "provider": "p", "base_url": "u", "api_key": "k", "api_mode": "chat_completions"}
        with patch("agent.oneshot.run_oneshot", side_effect=fake_run_oneshot):
            answer = answer_side_question(
                "which file had the error?", history, main_runtime=runtime
            )

        assert answer == "the error was in foo.py"
        assert captured["task"] == SIDE_QUESTION_TASK
        assert captured["main_runtime"] is runtime
        assert "USER: run the tests" in captured["user_input"]
        assert "Side question: which file had the error?" in captured["user_input"]
        # The instructions steer the model to answer only the side question.
        assert "side" in captured["instructions"].lower()


class TestTrimSnapshotForFork:
    def test_trims_unresolved_tool_loop_tail(self):
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "done first task"},
            {"role": "user", "content": "u2 (in-flight)"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result"},
        ]
        trimmed = trim_snapshot_for_fork(history)
        assert trimmed[-1] == {"role": "assistant", "content": "done first task"}
        assert len(trimmed) == 2

    def test_keeps_completed_history(self):
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert trim_snapshot_for_fork(history) == history

    def test_empty_when_no_completed_assistant(self):
        history = [{"role": "user", "content": "first message, turn running"}]
        assert trim_snapshot_for_fork(history) == []


class TestForkPath:
    def test_prefers_fork_when_parent_agent_given(self):
        seen = {}

        def fake_fork(parent, question, history):
            seen["parent"] = parent
            seen["question"] = question
            return "fork answer"

        parent = object()
        with patch("agent.side_question._answer_via_fork", side_effect=fake_fork), \
             patch("agent.side_question._answer_via_oneshot") as oneshot:
            out = answer_side_question("q?", [], parent_agent=parent)
        assert out == "fork answer"
        assert seen["parent"] is parent
        oneshot.assert_not_called()

    def test_falls_back_to_oneshot_when_fork_fails(self):
        with patch(
            "agent.side_question._answer_via_fork",
            side_effect=RuntimeError("boom"),
        ), patch(
            "agent.side_question._answer_via_oneshot", return_value="digest answer"
        ) as oneshot:
            out = answer_side_question("q?", [], parent_agent=object())
        assert out == "digest answer"
        oneshot.assert_called_once()

    def test_no_parent_agent_uses_oneshot(self):
        with patch("agent.side_question._answer_via_fork") as fork, patch(
            "agent.side_question._answer_via_oneshot", return_value="digest"
        ):
            out = answer_side_question("q?", [])
        assert out == "digest"
        fork.assert_not_called()

    def test_fork_denies_tools_and_replays_snapshot(self):
        """_answer_via_fork wires the empty whitelist, replays the trimmed
        snapshot, runs the fork, attributes usage, and tears down."""
        from agent.side_question import _answer_via_fork

        calls = {}

        class FakeFork:
            def run_conversation(self, user_message, conversation_history):
                calls["user_message"] = user_message
                calls["history"] = conversation_history
                return {"final_response": "it was foo.py"}

            def shutdown_memory_provider(self):
                calls["shutdown"] = True

            def close(self):
                calls["closed"] = True

        def fake_build(parent, task_cfg, *, max_iterations, write_origin):
            calls["write_origin"] = write_origin
            return FakeFork(), {"model": "m"}, False

        whitelists = []

        history = [
            {"role": "user", "content": "fix foo.py"},
            {"role": "assistant", "content": "fixed"},
        ]
        with patch("agent.background_review.build_cache_parity_fork", fake_build), \
             patch("hermes_cli.plugins.set_thread_tool_whitelist",
                   side_effect=lambda allowed, **kw: whitelists.append(allowed)), \
             patch("hermes_cli.plugins.clear_thread_tool_whitelist"), \
             patch("agent.background_review._snapshot_review_usage", return_value={}), \
             patch("agent.background_review._record_review_usage_to_parent"):
            answer = _answer_via_fork(object(), "which file?", history)

        assert answer == "it was foo.py"
        assert whitelists == [set()]  # every tool denied at dispatch
        assert calls["history"] == history  # full snapshot replayed verbatim
        assert "which file?" in calls["user_message"]
        assert calls["write_origin"] == "side_question"
        assert calls.get("shutdown") and calls.get("closed")
