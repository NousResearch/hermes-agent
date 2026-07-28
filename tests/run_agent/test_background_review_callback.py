"""Test the background_review_callback hook.

Verifies:
1. _build_improvement_records_from_review_messages — walks the review_agent
   message buffer and produces a list of dict records.
2. _invoke_callback_with_timeout swallows exceptions.
3. Timeout aborts a hung callback without crashing the review thread.
4. Empty messages emit no records.
5. Failed writes are captured.
"""

import json
import time

from agent.background_review_callback import (
    _build_improvement_records_from_review_messages,
    _invoke_callback_with_timeout,
)


def test_build_records_extracts_one_record_per_skill_write():
    messages = [
        {"role": "assistant", "content": "I'll refactor the receipt matcher."},
        {"role": "tool_use", "name": "skill_manage", "input": {
            "action": "write_file",
            "skill": "physical-receipt-ingestion",
            "file": "SKILL.md",
            "content": "new body",
        }, "tool_call_id": "t1"},
        {"role": "tool_result", "tool_call_id": "t1", "content": json.dumps({
            "success": True, "path": "/opt/data/skills/grocery/physical-receipt-ingestion/SKILL.md",
            "diff": "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n",
        })},
    ]
    records = _build_improvement_records_from_review_messages(
        messages, session_id="sess-1", ts="2026-06-17T00:00:00Z",
    )
    assert len(records) == 1
    r = records[0]
    assert r["schema_version"] == 1
    assert r["session_id"] == "sess-1"
    assert r["origin"] == "background_review"
    assert "refactor the receipt matcher" in r["plan"]
    assert len(r["writes"]) == 1
    w = r["writes"][0]
    assert w["op"] == "write_file"
    assert w["path"].endswith("SKILL.md")
    assert w["success"] is True
    assert w["diff"].startswith("--- a")
    assert w["post_content"] == "new body"


def test_build_records_captures_failed_writes():
    messages = [
        {"role": "tool_use", "name": "skill_manage", "input": {
            "action": "write_file", "skill": "x", "file": "f.py", "content": "..."
        }, "tool_call_id": "t1"},
        {"role": "tool_result", "tool_call_id": "t1", "content": json.dumps({
            "success": False, "error": "Permission denied",
            "path": "/opt/data/skills/x/f.py",
        })},
    ]
    records = _build_improvement_records_from_review_messages(
        messages, session_id="s", ts="t",
    )
    assert len(records) == 1
    w = records[0]["writes"][0]
    assert w["success"] is False
    assert w["error_preview"] == "Permission denied"
    assert w["post_content"] is None
    assert w["diff"] is None


def test_callback_exception_does_not_propagate():
    called = []
    def boom(record):
        called.append(record)
        raise RuntimeError("kaboom")
    # Should swallow.
    _invoke_callback_with_timeout(boom, {"x": 1}, timeout=1.0)
    assert called == [{"x": 1}]  # callback was invoked and exception swallowed


def test_callback_timeout_does_not_crash():
    def hang(record):
        time.sleep(10)
    start = time.time()
    _invoke_callback_with_timeout(hang, {"x": 1}, timeout=0.2)
    assert time.time() - start < 2.0


def test_empty_review_emits_no_records():
    records = _build_improvement_records_from_review_messages(
        [], session_id="s", ts="t",
    )
    assert records == []
