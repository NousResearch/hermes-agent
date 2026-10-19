"""execute_code result outcomes must survive deterministic compaction and replay."""

import json
import time

import pytest

from agent.context_compressor import ContextCompressor, _summarize_tool_result
from hermes_state import SessionDB
from tools.code_execution_tool import _error_result, _finish_remote_kernel_result


@pytest.mark.parametrize("payload, expected", [
    ({"status": "error", "error": "ValueError: invalid schema", "exit_code": 1},
     "FAILED: ValueError: invalid schema"),
    ({"status": "error"}, "FAILED"),
    ({"exit_code": 1}, "FAILED: exit code 1"),
    ({"error": "Traceback (most recent call last):\n  File 'cell.py', line 1\nValueError: invalid schema"},
     "FAILED: ValueError: invalid schema"),
    ({"status": "timeout", "error": "Cell exceeded its deadline", "kernel": {"mode": "session"}},
     "TIMEOUT; kernel state lost: Cell exceeded its deadline"),
    ({"status": "interrupted", "error": "Stopped", "kernel": {"mode": "session"}},
     "INTERRUPTED; kernel state lost: Stopped"),
    ({"status": "error", "kernel": {"state_lost": True}}, "FAILED; kernel state lost"),
    ({"status": "success", "exit_code": 0, "tool_errors": [{"tool": "patch", "error": "old_string not found"}]},
     "inner tool failed: patch: old_string not found"),
    ({"status": "error", "error": "ValueError: bad input", "tool_errors": [{"tool": "patch", "error": "not found"}]},
     "FAILED; inner tool failed: ValueError: bad input"),
    ({"status": "error", "error": "ValueError: " + "long detail\n" * 50,
      "tool_errors": [{"tool": "patch", "error": "not found"}], "kernel": {"state_lost": True}},
     "FAILED; inner tool failed; kernel state lost"),
    ({"status": "success", "kernel": {"state_reset": True}}, "kernel state reset"),
    ({"status": "success", "kernel": {"ended": True}}, "kernel ended"),
    ({"status": "blocked", "error": "BLOCKED: not approved"}, "BLOCKED, not run"),
    ({"status": "pending_approval"}, "awaiting the user's approval, not run"),
    ({"status": "success", "exit_code": 0, "output": '{"error":"ordinary output data"}',
      "tool_errors": [], "kernel": {"reused": True, "state_reset": False}}, None),
    ({"status": "success", "exit_code": 0, "tool_errors": [{"tool": "patch", "error": ""}]}, None),
    ({"status": ["error"], "exit_code": False, "tool_errors": [None], "kernel": []}, None),
    (None, None),
    ([], None),
])
@pytest.mark.parametrize("appended", ["", "\n\n[Subdirectory context discovered: ops/AGENTS.md]\nUse the ops venv."])
def test_execute_code_stub_preserves_outcome_without_reclassifying_output(payload, expected, appended):
    content = json.dumps(payload) + appended
    stub = _summarize_tool_result("execute_code", '{"code":"run_validation()"}', content)
    if expected is None:
        lines = content.count("\n") + 1
        assert stub == f"[execute_code] `run_validation()` ({lines} lines output)"
    else:
        assert expected in stub
        assert "\n" not in stub
        assert len(stub) < 200
        long_stub = _summarize_tool_result("execute_code", json.dumps({"code": "x" * 100}), content)
        assert expected in long_stub and len(long_stub) < 200
        assert _summarize_tool_result("execute_code", '{"code":"run_validation()"}', long_stub) == long_stub


@pytest.mark.parametrize("path", ["prune", "compress", "reopen"])
def test_execute_code_failure_reaches_summary_input_and_durable_history(tmp_path, monkeypatch, path):
    # Use the real error producer and remote-result normalizer; no code/model/remote API is run.
    failure = (_error_result("ValueError: invalid schema " + "detail " * 3000) if path == "reopen" else
               _finish_remote_kernel_result({"status": "error", "kernel": {
                   "state_lost": True, "note": "Remote protocol failed; state lost. " + "transport detail " * 20}},
                   timeout=10, exec_start=time.monotonic()))
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "execute-1", "type": "function",
            "function": {"name": "execute_code", "arguments": '{"code":"run_validation()"}'}}]}
    messages = [{"role": "user", "content": "Start"}, {"role": "assistant", "content": "Ready"},
                {"role": "user", "content": "Run validation"}, call,
                {"role": "tool", "name": "execute_code", "tool_name": "execute_code",
                 "tool_call_id": "execute-1", "content": failure}]
    for i in range(40 if path == "compress" else 12):
        messages.extend([{"role": "user", "content": f"Follow-up {i}: " + "u" * 3500},
                         {"role": "assistant", "content": f"Reply {i}: " + "a" * 3500}])
    compressor = ContextCompressor(model="test", config_context_length=100000, quiet_mode=True)
    if path == "compress":
        captured = []
        def summarize(window, **kwargs):
            captured.append(compressor._serialize_for_summary(window))
            return "## Goal\nContinue validation.\n\n## Active State\nValidation failed; inspect the error."
        monkeypatch.setattr(compressor, "_generate_summary", summarize)
        compressor.compress(messages, current_tokens=80000, force=True)
        assert captured and "FAILED; kernel state lost" in captured[0]
        return
    if path == "prune":
        pruned, _ = compressor._prune_old_tool_results(messages, protect_tail_count=2)
        stub = next(m["content"] for m in pruned if m.get("tool_call_id") == "execute-1")
        assert "FAILED; kernel state lost" in stub
        repeated, _ = compressor._prune_old_tool_results(pruned, protect_tail_count=2)
        assert next(m["content"] for m in repeated if m.get("tool_call_id") == "execute-1") == stub
        # Exercise re-summarization, independently of the fallback's overall history-size cap.
        assert "FAILED; kernel state lost" in compressor._build_static_fallback_summary(pruned[:5])
        return
    db_path = tmp_path / "state.db"
    with SessionDB(db_path=db_path) as db:
        db.create_session("compaction-outcome", source="test")
        db.append_messages_batch("compaction-outcome", messages)
        compressor._session_db, compressor._session_id = db, "compaction-outcome"
        compressor.proactive_prune_tokens = 1
        pruned, count = compressor.prune_tool_results_only(
            db.get_messages_as_conversation("compaction-outcome", include_row_ids=True), current_tokens=80000)
        assert count == 1
        stub = next(m["content"] for m in pruned if m.get("tool_call_id") == "execute-1")
        assert "FAILED: ValueError: invalid schema" in stub
        assert any(row["content"] == failure and not row["active"]
                   for row in db.get_messages("compaction-outcome", include_inactive=True))
        assert db.search_messages('"invalid schema"', role_filter=["tool"])
    with SessionDB(db_path=db_path) as reopened:
        restored = reopened.get_messages_as_conversation("compaction-outcome")
        assert next(m["content"] for m in restored if m.get("tool_call_id") == "execute-1") == stub
