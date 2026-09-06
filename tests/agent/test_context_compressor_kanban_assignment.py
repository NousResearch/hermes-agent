import json

import pytest

from agent.context_compressor import ContextCompressor
from agent.llm_egress_runtime import _project_bound_kanban_show


@pytest.mark.parametrize("protected", [False, True])
def test_compression_keeps_only_latest_assignment_and_protected_egress(protected):
    task_id = "t_12345678"
    receipt = "--feedback-id 3942545980 --receipt-head-sha " + "a" * 40
    body = "Complete only after verified publication: " + receipt
    task = {"id": task_id, "title": "Repair current feedback", "body": body, "status": "running"}
    payload = {"task": task, "comments": [{"body": "obsolete private history"}],
               "runs": [{"summary": "old run must not return"}], "unrelated": "private-extra"}
    if protected:
        payload["protected_task_spec"] = {"version": "v1", "title": task["title"], "body": body}
    messages = [{"role": "tool", "tool_call_id": call, "content": json.dumps(payload)}
                for call in ("old", "new")]
    calls = {call: ("kanban_show", json.dumps({"task_id": task_id})) for call in ("old", "new")}
    for index in range(2):
        assert ContextCompressor._demote_tool_result_at(messages, index, calls, 0)
    assert receipt not in messages[0]["content"]
    latest = json.loads(messages[1]["content"])
    assert latest["task"]["body"] == body
    assert "obsolete private history" not in messages[1]["content"]
    assert "old run must not return" not in messages[1]["content"]
    assert "private-extra" not in messages[1]["content"]
    assert ("protected_task_spec" in latest) is protected
    if protected:
        assert receipt in _project_bound_kanban_show(messages[1]["content"]).text
    assert not ContextCompressor._demote_tool_result_at(messages, 1, calls, 0)


@pytest.mark.parametrize("case", ["long", "malformed", "foreign", "forged_spec", "surrogate"])
def test_assignment_summary_is_bounded_and_does_not_promote_unbound_text(case):
    body = "receipt-first " + "界" * 9000
    payload = {"task": {"id": "t_12345678", "title": "Bounded assignment", "body": body}}
    if case == "foreign":
        payload["task"]["id"] = "t_87654321"
    if case == "forged_spec":
        payload["protected_task_spec"] = {"version": "unknown", "body": "do not promote"}
    if case == "surrogate":
        payload["task"]["body"] = "receipt-first \ud800"
    content = "not JSON" if case == "malformed" else json.dumps(payload)
    messages = [{"role": "tool", "tool_call_id": "show", "content": content}]
    calls = {"show": ("kanban_show", json.dumps({"task_id": "t_12345678"}))}
    ContextCompressor._demote_tool_result_at(messages, 0, calls, 0)
    summary = messages[0]["content"]
    if case in {"malformed", "foreign"}:
        assert summary.startswith("[kanban_show]")
        assert body not in summary
    else:
        parsed = json.loads(summary)
        assert len(parsed["task"]["body"].encode("utf-8")) <= 8 * 1024
        assert parsed["task"]["body"].startswith("receipt-first ")
        assert "protected_task_spec" not in parsed


def test_pressure_demotes_noncurrent_kanban_projection():
    task_id = "t_12345678"
    body = "assignment details " + ("x" * 2000)
    payload = {"task": {"id": task_id, "title": "Referenced card", "body": body}}
    messages = [{"role": "tool", "tool_call_id": "show", "content": json.dumps(payload)}]
    calls = {"show": ("kanban_show", json.dumps({"task_id": task_id}))}

    assert ContextCompressor._demote_tool_result_at(messages, 0, calls, 0, pressure=True)
    assert messages[0]["content"].startswith("[kanban_show]")
    assert body not in messages[0]["content"]


def test_lean_tail_keeps_newest_current_assignment_projection(monkeypatch):
    task_id = "t_12345678"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    assignment_body = "current assignment " + ("x" * 2000)
    assignment = json.dumps({"task": {"id": task_id, "title": "Current card", "body": assignment_body}})
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "assignment", "function": {
            "name": "kanban_show", "arguments": json.dumps({"task_id": task_id}),
        }}]},
        {"role": "tool", "tool_call_id": "assignment", "content": assignment},
    ]
    for index in range(7):
        call_id = f"newer-{index}"
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": call_id, "function": {
                "name": "other_tool", "arguments": "{}",
            }}]},
            {"role": "tool", "tool_call_id": call_id, "content": f"newer result {index} " + ("y" * 2000)},
        ])

    compressor = ContextCompressor("test-model", quiet_mode=True)
    result = compressor._demote_stale_tail_tools(messages, 0)

    assert json.loads(result[1]["content"])["task"]["body"] == assignment_body
    assert result[3]["content"] != messages[3]["content"]


def test_full_compress_keeps_current_assignment_from_summarized_window(monkeypatch):
    task_id = "t_12345678"
    receipt = "--feedback-id 3943941351 --receipt-head-sha " + "a" * 40
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    assignment = json.dumps({
        "task": {"id": task_id, "title": "Current card", "body": f"Finish repair: {receipt}"},
    })
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "tool_calls": [{"id": "assignment", "function": {
            "name": "kanban_show", "arguments": json.dumps({"task_id": task_id}),
        }}]},
        {"role": "tool", "tool_call_id": "assignment", "content": assignment},
        {"role": "assistant", "content": "tail context"},
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "new response"},
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "latest response"},
        {"role": "user", "content": "final request"},
    ]
    compressor = ContextCompressor("test-model", quiet_mode=True)
    monkeypatch.setattr(compressor, "_compress_window", lambda _messages: (2, 6))
    monkeypatch.setattr(compressor, "_generate_summary", lambda _messages, **_kwargs: "summary without receipt")

    result = compressor.compress(messages, current_tokens=999_999, force=True)

    assert len(result) < len(messages)
    assert receipt in json.dumps(result)
