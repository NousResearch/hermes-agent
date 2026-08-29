"""Tests for agent/side_question.py — the /btw context-aware side question engine."""

from unittest.mock import patch

from agent.side_question import (
    SIDE_QUESTION_TASK,
    answer_side_question,
    render_history_for_side_question,
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
