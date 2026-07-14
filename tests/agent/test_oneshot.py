"""Tests for agent.oneshot — shared one-off (stateless) LLM requests."""

from unittest.mock import MagicMock, patch

import pytest

from agent.oneshot import (
    PROMPT_TEMPLATES,
    render_template,
    run_oneshot,
    _strip_code_fence,
    _truncate,
)


class TestRenderTemplate:
    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_template("does-not-exist", {})

    def test_commit_message_template_is_registered(self):
        assert "commit_message" in PROMPT_TEMPLATES

    def test_commit_message_includes_diff_and_recent(self):
        instructions, user = render_template(
            "commit_message",
            {"diff": "diff --git a/x b/x\n+new", "recent_commits": "feat: a\nfix: b"},
        )
        # Instructions describe the contract (conventional commits), not a snapshot.
        assert "Conventional Commits" in instructions
        assert "diff --git a/x b/x" in user
        assert "feat: a" in user

    def test_commit_message_diff_with_braces_passes_through(self):
        # Templates must not use str.format — code payloads carry literal { }.
        _, user = render_template("commit_message", {"diff": "x = {a: 1}"})
        assert "x = {a: 1}" in user

    def test_commit_message_handles_missing_variables(self):
        instructions, user = render_template("commit_message", {})
        assert instructions
        assert "no textual diff available" in user

    def test_commit_message_avoid_forces_new_message(self):
        # Passing the previous message must instruct the model not to repeat it,
        # so "regenerate" yields a different result even on greedy models.
        _, plain = render_template("commit_message", {"diff": "d"})
        _, regen = render_template("commit_message", {"diff": "d", "avoid": "feat: prior"})
        assert "feat: prior" in regen
        assert "do not repeat" in regen
        assert "feat: prior" not in plain


class TestRunOneshot:
    def _mock_response(self, content):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.choices[0].message.reasoning = None
        resp.choices[0].message.reasoning_content = None
        resp.choices[0].message.reasoning_details = None
        return resp

    def test_template_path_calls_llm_with_rendered_prompt(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("feat: add thing"),
        ) as llm:
            out = run_oneshot(template="commit_message", variables={"diff": "d"})

        assert out == "feat: add thing"
        messages = llm.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_explicit_instructions_path(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("hello"),
        ) as llm:
            out = run_oneshot(instructions="be brief", user_input="say hi")

        assert out == "hello"
        messages = llm.call_args.kwargs["messages"]
        assert messages[0]["content"] == "be brief"
        assert messages[1]["content"] == "say hi"

    def test_requires_template_or_prompt(self):
        with pytest.raises(ValueError):
            run_oneshot()

    def test_strips_wrapping_code_fence(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("```\nfix: bug\n```"),
        ):
            assert run_oneshot(instructions="x", user_input="y") == "fix: bug"


class TestHelpers:
    def test_truncate_under_limit_unchanged(self):
        assert _truncate("short", 100) == "short"

    def test_truncate_over_limit_marks_truncation(self):
        out = _truncate("x" * 200, 50)
        assert out.endswith("…(truncated)")
        assert len(out) < 200

    def test_strip_code_fence_without_fence_is_noop(self):
        assert _strip_code_fence("plain text") == "plain text"
