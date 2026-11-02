"""Tests for agent.oneshot — shared one-off (stateless) LLM requests."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.oneshot import (
    client_task,
    oneshot_reply,
    render_template,
    run_oneshot,
    _truncate,
)

class TestRenderTemplate:

    def test_commit_message_includes_diff_and_recent(self):
        instructions, user = render_template(
            "commit_message",
            {"diff": "diff --git a/x b/x\n+new", "recent_commits": "feat: a\nfix: b"},
        )
        assert instructions
        assert "diff --git a/x b/x" in user
        assert "feat: a" in user

    def test_commit_message_avoid_forces_new_message(self):
        # Passing the previous message must instruct the model not to repeat it,
        # so "regenerate" yields a different result even on greedy models.
        _, plain = render_template("commit_message", {"diff": "d"})
        _, regen = render_template("commit_message", {"diff": "d", "avoid": "feat: prior"})
        assert "feat: prior" in regen
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

    def test_strips_wrapping_code_fence(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("```\nfix: bug\n```"),
        ):
            assert run_oneshot(instructions="x", user_input="y") == "fix: bug"

    def test_a_client_runs_a_plugin_task_and_gets_schema_checked_json(self, monkeypatch):
        """#141976: the task is a built-in or plugin-registered slot, never an unknown key; a schema
        reply comes back parsed and a prose one is an error; ``redact`` masks the input first."""
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_auxiliary_tasks",
                            lambda: [{"key": "tern_charts", "display_name": "Charts", "description": "d"}])
        assert client_task("tern_charts") == "tern_charts"
        with pytest.raises(KeyError, match="tern_chart"):
            client_task("tern_chart")
        schema = {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]}
        secret = "sk-proj-" + "A1b2C3d4" * 5
        replies = [self._mock_response('{"kind": "line"}'), self._mock_response("a line chart")]
        with patch("agent.oneshot.call_llm", side_effect=replies) as llm:
            assert oneshot_reply(instructions="sys", user_input=f"key {secret}", task="tern_charts", schema=schema,
                                 redact=True) == {"text": '{"kind": "line"}', "parsed": {"kind": "line"}}
            with pytest.raises(ValueError, match="JSON"):
                oneshot_reply(instructions="sys", user_input="x", task="tern_charts", schema=schema)
        sent = llm.call_args_list[0].kwargs
        assert sent["task"] == "tern_charts" and sent["extra_body"]["response_format"]["type"] == "json_schema"
        assert secret not in json.dumps(sent["messages"])
        monkeypatch.setitem(sys.modules, "jsonschema", None)  # a core install has no jsonschema
        with patch("agent.oneshot.call_llm", return_value=self._mock_response('{"kind": 1}')) as llm:
            with pytest.raises(ValueError, match="jsonschema"):
                oneshot_reply(instructions="sys", user_input="x", task="tern_charts", schema=schema)
        llm.assert_not_called()

class TestHelpers:

    def test_truncate_over_limit_marks_truncation(self):
        out = _truncate("x" * 200, 50)
        assert out.endswith("…(truncated)")
        assert len(out) < 200
