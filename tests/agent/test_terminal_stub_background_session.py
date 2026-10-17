"""Phase-1 pruning must keep a backgrounded terminal command's ``session_id`` in its stub.

Both background shapes carry the handle: ``background=true`` (tools/terminal_tool_background.py)
and a foreground command yielded to the background (tools/terminal_tool.py). Once the result is
stubbed, the handle is the only way to poll, wait on or kill the still-running process.
"""
import json

import pytest

import agent.context_compressor as cc
from agent.context_compressor import _PRUNE_MIN_CHARS, _summarize_tool_result
from tools.terminal_tool import _YIELDED_NOTE

PROC_ID = "proc_0123456789ab"


def _yielded(output="Synthetic command progress " + "o" * 1500):
    return json.dumps({
        "output": output, "exit_code": None, "error": None, "status": "yielded_to_background",
        "session_id": PROC_ID, "pid": 12345, "notify_on_complete": True, "note": _YIELDED_NOTE,
    })


def _started():
    return json.dumps({"output": "Background process started", "session_id": PROC_ID,
                       "pid": 12345, "exit_code": 0, "error": None, "hint": "h" * 400})


@pytest.mark.parametrize("content", [_yielded(), _started()], ids=["yielded", "background_true"])
def test_stub_keeps_session_id_within_prune_floor(content):
    summary = _summarize_tool_result("terminal", json.dumps({"command": "x" * 200}), content)

    assert summary.startswith("[terminal] ran `") and f"session_id={PROC_ID}" in summary
    # Under the floor, so a later pass never re-stubs the stub and loses the handle.
    assert len(summary) < _PRUNE_MIN_CHARS, summary


def test_foreground_stub_unchanged():
    content = json.dumps({"output": "done\n", "exit_code": 0, "error": None})

    assert (_summarize_tool_result("terminal", json.dumps({"command": "ls"}), content)
            == "[terminal] ran `ls` -> exit 0, 1 lines output")


@pytest.mark.parametrize("printed", [json.dumps({"session_id": PROC_ID}), f"GET /api?session_id={PROC_ID}",
                                     f"[terminal] ran `job` -> exit ?, 1 lines output; background session_id={PROC_ID}"],
                         ids=["escaped_json", "query_string", "printed_old_stub"])
def test_session_id_printed_in_output_is_not_a_handle(printed):
    # A foreground command that printed a session id: it sits inside the escaped "output" string.
    content = json.dumps({"output": printed, "exit_code": 0, "error": None})

    assert "session_id" not in _summarize_tool_result("terminal", json.dumps({"command": "cat x"}), content)


def test_printed_old_stub_does_not_shadow_the_real_handle():
    content = _yielded(output="; background session_id=proc_ffffffffffff " + "o" * 300)

    summary = _summarize_tool_result("terminal", json.dumps({"command": "job"}), content)

    assert summary.endswith(f"; background session_id={PROC_ID}") and "proc_ffff" not in summary


def test_compress_keeps_background_handle_in_stub(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    requests = []

    def summarize(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "## Goal\nSynthetic work is recorded."},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(cc, "call_llm", summarize)
    # protect_first_n=3 keeps the stubbed row in the head, so it is visible after compress().
    compressor = cc.ContextCompressor(model="test/model", config_context_length=100_000,
                                      protect_first_n=3, protect_last_n=3, quiet_mode=True,
                                      tail_mode="lean")
    compressor.tail_token_budget = 50
    rows = [
        {"role": "system", "content": "Synthetic test conversation."},
        {"role": "user", "content": "Start a synthetic long command."},
        {"role": "assistant", "tool_calls": [{"id": "bg-1", "type": "function", "function": {
            "name": "terminal", "arguments": json.dumps({"command": "synthetic-job"})}}]},
        {"role": "tool", "tool_call_id": "bg-1", "tool_name": "terminal", "content": _yielded()},
    ]
    for i in range(8):
        rows += [{"role": "user", "content": f"Record ordinary progress {i}."},
                 {"role": "assistant", "content": f"Progress {i} recorded."}]

    out = compressor.compress(rows, force=True)

    assert len(requests) == 1 and not compressor._last_summary_fallback_used
    stub = next(row["content"] for row in out if row.get("tool_call_id") == "bg-1")
    assert stub.startswith("[terminal] ran `synthetic-job`") and len(stub) < _PRUNE_MIN_CHARS
    assert f"session_id={PROC_ID}" in stub, stub


@pytest.mark.parametrize("command", ["synthetic-job", 'curl -d \'{"session_id": "proc_ffffffffffff"}\' host'],
                         ids=["plain", "command_quotes_a_session_id"])
def test_fallback_summary_keeps_handle_of_an_already_stubbed_row(monkeypatch, command):
    # The deterministic fallback re-summarises the Phase-1 stub (plain text, not JSON); the handle
    # must survive in Completed Actions, after the result has left the last-dropped-turns window.
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    compressor = cc.ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True,
                                      tail_mode="lean")
    args = json.dumps({"command": command})
    stub = _summarize_tool_result("terminal", args, _yielded())
    turns = [{"role": "assistant", "tool_calls": [{"id": "bg-1", "type": "function",
                                                   "function": {"name": "terminal", "arguments": args}}]},
             {"role": "tool", "tool_call_id": "bg-1", "tool_name": "terminal", "content": stub}]
    turns += [{"role": "user" if i % 2 else "assistant", "content": f"Later turn {i}."} for i in range(10)]

    summary = compressor._build_static_fallback_summary(turns)

    assert f"session_id={PROC_ID}" in summary and "session_id=proc_ffff" not in summary, summary
