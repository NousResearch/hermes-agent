from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from agent.delegation_context import delegated_child_context
from agent.delegation_receipts import (
    append_receipt_marker,
    commit_runtime_receipt,
    parent_visible_receipts,
    prepare_runtime_receipt,
)
from agent.tool_executor import _ToolCallRef, _commit_tool_result
from tools.budget_config import DEFAULT_BUDGET


class _FakeAgent:
    def __init__(self, *, flush_ok: bool = True):
        self.session_id = "child-session"
        self._subagent_id = "sa-0-test"
        self._delegate_purpose = "research_evidence"
        self._current_tool = "read_file"
        self._subdirectory_hints = SimpleNamespace(check_tool_call=lambda *_: "")
        self._flush_messages_to_session_db = lambda _messages: flush_ok
        self._tool_result_content_for_active_model = lambda _name, result: result
        self.tool_progress_callback = None
        self.activities = []

    def _touch_activity(self, text):
        self.activities.append(text)


def _commit(agent: _FakeAgent, messages: list):
    ref = _ToolCallRef(
        "read_file", {"path": "C:/work/item.txt", "secret": "not-recorded"},
        "task-1", "call-1", [],
    )
    with patch("agent.tool_executor.maybe_persist_tool_result", side_effect=lambda **kwargs: kwargs["content"]):
        return _commit_tool_result(
            agent, messages, ref, "file bytes", budget=DEFAULT_BUDGET,
            tool_duration=0.1, is_error=False, blocked=False,
            effect_disposition="none",
        )


def test_receipt_is_minted_only_in_delegated_child_context():
    agent = _FakeAgent()
    assert prepare_runtime_receipt(
        agent, tool_name="read_file", tool_call_id="call-1",
        arguments={"path": "C:/work/item.txt"}, result="bytes", status="ok",
        effect_disposition="none",
    ) is None

    with delegated_child_context("child-session"):
        receipt = prepare_runtime_receipt(
            agent, tool_name="read_file", tool_call_id="call-1",
            arguments={"path": "C:/work/item.txt", "token": "secret"},
            result="bytes", status="ok", effect_disposition="none",
        )
    assert receipt is not None
    assert receipt["receipt_id"].startswith("dr_")
    assert receipt["input_summary"] == {
        "argument_keys": ["path", "token"],
        "targets": {"path": "C:/work/item.txt"},
    }
    assert "secret" not in str(receipt)


def test_durable_tool_result_commits_runtime_receipt_and_marker():
    agent = _FakeAgent(flush_ok=True)
    messages = []
    with delegated_child_context("child-session"):
        committed = _commit(agent, messages)

    assert committed is not None
    assert len(agent._delegate_runtime_receipts) == 1
    receipt_id = agent._delegate_runtime_receipts[0]["receipt_id"]
    assert agent._delegate_runtime_receipts[0]["purpose"] == "research_evidence"
    assert messages[0]["role"] == "tool"
    assert f"[Runtime receipt: {receipt_id}]" in messages[0]["content"]

    summary = f"Read the file and verified its contents. [{receipt_id}]"
    visible = parent_visible_receipts(agent, summary)
    assert visible["provenance_status"] == "verified_citations"
    assert visible["cited_receipt_ids"] == [receipt_id]
    assert visible["fabricated_receipt_ids"] == []


def test_receipt_digest_covers_full_pre_spill_output():
    agent = _FakeAgent(flush_ok=True)
    messages = []
    full_output = "full-output-that-will-be-spilled"
    ref = _ToolCallRef("read_file", {"path": "C:/work/item.txt"}, "task-1", "call-spill", [])
    with delegated_child_context("child-session"), patch(
        "agent.tool_executor.maybe_persist_tool_result", return_value="<persisted-output>stub</persisted-output>"
    ):
        committed = _commit_tool_result(
            agent, messages, ref, full_output, budget=DEFAULT_BUDGET,
            tool_duration=0.1, is_error=False, blocked=False, effect_disposition="none",
        )
    assert committed is not None
    assert agent._delegate_runtime_receipts[0]["output_sha256"] == hashlib.sha256(full_output.encode()).hexdigest()
    assert "<persisted-output>stub</persisted-output>" in messages[0]["content"]


def test_flush_failure_never_commits_pending_receipt():
    agent = _FakeAgent(flush_ok=False)
    messages = []
    with delegated_child_context("child-session"):
        assert _commit(agent, messages) is None

    assert not hasattr(agent, "_delegate_runtime_receipts")
    pending_id = messages[0]["content"].split("dr_", 1)[1].split("]", 1)[0]
    visible = parent_visible_receipts(agent, f"Claim cites dr_{pending_id}")
    assert visible["provenance_status"] == "no_runtime_evidence"
    assert visible["cited_receipt_ids"] == []
    assert visible["fabricated_receipt_ids"] == [f"dr_{pending_id}"]


def test_non_child_commit_preserves_tool_content_byte_for_byte():
    agent = _FakeAgent(flush_ok=True)
    messages = []
    assert _commit(agent, messages) is not None
    assert messages[0]["content"] == "file bytes"
    assert not hasattr(agent, "_delegate_runtime_receipts")


def test_fabricated_summary_receipt_is_not_promoted():
    agent = _FakeAgent()
    with delegated_child_context("child-session"):
        receipt = prepare_runtime_receipt(
            agent, tool_name="terminal", tool_call_id="call-2", arguments={"cwd": "C:/work"},
            result="ok", status="ok", effect_disposition="none",
        )
    commit_runtime_receipt(agent, receipt)
    fake = "dr_000000000000000000000000"
    visible = parent_visible_receipts(agent, f"Verified with {fake}")
    assert visible["provenance_status"] == "missing_citation"
    assert visible["fabricated_receipt_ids"] == [fake]
    assert visible["cited_receipt_ids"] == []


def test_multimodal_marker_keeps_existing_blocks():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]
    stamped = append_receipt_marker(content, "dr_000000000000000000000000")
    assert stamped[0] == content[0]
    assert stamped[-1] == {"type": "text", "text": "[Runtime receipt: dr_000000000000000000000000]"}
