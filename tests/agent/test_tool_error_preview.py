"""The `returned error` log preview must keep the failure classification fields.

A raw ``text[:200]`` slice cut terminal JSON results mid-``output`` and dropped
``exit_code`` / ``error`` (issue #125238)."""

from __future__ import annotations

import json

from agent.tool_executor import _truncate_error_preview


class TestTruncateErrorPreview:
    def test_short_text_untouched(self):
        assert _truncate_error_preview("boom") == "boom"

    def test_plain_long_text_still_truncated(self):
        text = "x" * 5000
        out = _truncate_error_preview(text)
        assert len(out) <= 200
        assert out == text[:200]

    def test_terminal_json_keeps_exit_code_and_error(self):
        payload = json.dumps({
            "output": "E" * 4000 + "\nFAILED tests/x.py::test_y - assert 1 == 2",
            "exit_code": 1,
            "error": "command failed",
        })
        out = _truncate_error_preview(payload)
        assert len(out) <= 260
        data = json.loads(out)  # stays valid JSON
        assert data["exit_code"] == 1
        assert data["error"] == "command failed"
        assert data["output"].endswith("...[truncated]")

    def test_error_only_payload_keeps_error(self):
        payload = json.dumps({"output": "z" * 4000, "error": "killed by signal 9" * 10})
        out = _truncate_error_preview(payload)
        data = json.loads(out)
        assert data["error"].startswith("killed by signal 9")

    def test_non_json_object_truncated_plainly(self):
        text = "{" + "a" * 4000
        out = _truncate_error_preview(text)
        assert out == text[:200]
