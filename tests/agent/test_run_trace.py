"""Integration tests for the per-run trace lifecycle (task t_053fa848).

Acceptance: runs through each of the five outcome classes (completed, blocked,
crashed, timeout, review_requested) emit exactly one schema-valid JSONL record
with all required v1 fields populated (null where N/A); a forced crash still
yields ``outcome=crashed``. Emit-once is asserted under a double-close.

Records are validated against the normative ``specs/trace-v1.schema.json``
(the JSON Schema wins over prose per spec §7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml  # noqa: F401  (import-parity with the sibling redactor suite)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator  # noqa: E402

import agent.run_trace as rt  # noqa: E402
from agent.run_trace import RunTraceContext  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "specs" / "trace-v1.schema.json"
VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))

FIXED_HARNESS = {
    "prompt_builder_sha": "a" * 40,
    "config_sha256": "b" * 64,
}

AGENT_MESSAGES = [
    {"role": "system", "content": "You are a worker."},
    {"role": "user", "content": "Do the task."},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
    {"role": "tool", "content": "result text", "tool_call_id": "call_1", "name": "read_file"},
]


@pytest.fixture()
def trace_root(tmp_path):
    return str(tmp_path / "traces")


@pytest.fixture(autouse=True)
def _no_active_trace(monkeypatch):
    rt._ACTIVE = None
    # The redactor's default config lookup walks ~/.hermes — the suite's
    # home_io_guard refuses real-home I/O. Pin the packaged default config.
    monkeypatch.setenv(
        "HERMES_TRACE_REDACTION_CONFIG", str(REPO_ROOT / "specs" / "trace_redaction_default.yaml")
    )
    import agent.trace_redactor as red_mod

    red_mod._SHARED = None
    yield
    red_mod._SHARED = None
    rt._ACTIVE = None


def _records(trace_root: str) -> list[dict]:
    root = Path(trace_root)
    out = []
    for f in sorted(root.glob("traces-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _open_ctx(trace_root: str, **kw) -> RunTraceContext:
    params: dict = dict(
        run_id=42,
        profile_slug="viiy-coder",
        model="kimi-k3",
        provider="ollama-cloud",
        board_slug="specialized",
        task_id="t_test1234",
        harness_version=FIXED_HARNESS,
        trace_root=trace_root,
    )
    params.update(kw)
    return RunTraceContext(**params)


def _assert_valid(record: dict, outcome: str) -> None:
    errors = sorted(VALIDATOR.iter_errors(record), key=lambda e: list(e.path))
    assert not errors, f"schema errors: {[e.message for e in errors]}"
    assert record["outcome"] == outcome
    assert record["schema_version"] == "1.0.0"
    assert record["run_id"] == 42
    assert record["session_id"]
    assert record["profile_slug"] == "viiy-coder"
    assert record["board_slug"] == "specialized"
    assert record["task_id"] == "t_test1234"
    assert record["messages"] and record["messages"][0]["role"] == "system"
    assert record["wall_clock_seconds"] == record["ended_at"] - record["started_at"]
    tc = record["token_counts"]
    assert tc["total"] == tc["input"] + tc["output"] + tc["cache_read"] + tc["cache_write"]


def _prime(ctx: RunTraceContext) -> None:
    ctx.set_identity(session_id="sess-1")
    ctx.record_usage(
        {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 5, "cache_write_tokens": 3},
        0.0012,
    )
    ctx.record_tool_call("read_file", {"path": "a.py"}, "file body", call_id="call_1", duration_ms=12)


# ----------------------------------------------------------------------
# Five outcome classes
# ----------------------------------------------------------------------


def test_outcome_completed_via_terminal_call(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(
        outcome="completed",
        terminal_call="kanban_complete",
        exit_code=0,
        agent_messages=AGENT_MESSAGES,
    )
    records = _records(trace_root)
    assert len(records) == 1
    rec = records[0]
    _assert_valid(rec, "completed")
    assert rec["terminal_call"] == "kanban_complete"
    assert rec["exit_code"] == 0
    assert rec["error_class"] is None
    assert rec["token_counts"] == {"input": 100, "output": 20, "cache_read": 5, "cache_write": 3, "total": 128}
    assert rec["cost_usd"] == pytest.approx(0.0012)
    assert rec["token_counts_complete"] is True
    terminal = [tc for tc in rec["tool_calls"] if tc["is_terminal"]]
    # kanban_complete was never dispatched through the executor in this fixture:
    # no recorded call carries the name, so none is flagged terminal.
    assert terminal == []
    tool = rec["tool_calls"][0]
    assert tool["name"] == "read_file"
    assert tool["arguments"] == {"path": "a.py"}
    assert tool["result"] == "file body"
    assert tool["duration_ms"] == 12
    assert tool["exit_code"] is None
    assert rec["messages"][3]["role"] == "tool"
    assert rec["messages"][3]["tool_call_id"] == "call_1"
    assert rec["messages"][3]["name"] == "read_file"


def test_outcome_blocked(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(outcome="blocked", terminal_call="kanban_block", exit_code=0, agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    _assert_valid(rec, "blocked")
    assert rec["terminal_call"] == "kanban_block"


def test_outcome_review_requested(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(
        outcome="review_requested",
        terminal_call="kanban_request_review",
        exit_code=0,
        agent_messages=AGENT_MESSAGES,
    )
    rec = _records(trace_root)[0]
    _assert_valid(rec, "review_requested")
    assert rec["terminal_call"] == "kanban_request_review"


def test_outcome_timeout(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(outcome="timeout", error_class="TimeoutError", exit_code=None, agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    _assert_valid(rec, "timeout")
    assert rec["terminal_call"] is None
    assert rec["error_class"] == "TimeoutError"


def test_forced_crash_still_emits_record(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(outcome="crashed", error_class="KeyboardInterrupt", exit_code=130, agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    _assert_valid(rec, "crashed")
    assert rec["terminal_call"] is None
    assert rec["error_class"] == "KeyboardInterrupt"
    assert rec["exit_code"] == 130


def test_crash_before_any_provider_call(trace_root):
    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="sess-1")
    assert ctx.close(outcome="crashed", error_class="SIGTERM", exit_code=None, agent_messages=None)
    rec = _records(trace_root)[0]
    _assert_valid(rec, "crashed")
    # Usage was never observable: not zero, unknowable.
    assert rec["token_counts_complete"] is False
    assert rec["token_counts"]["total"] == 0
    assert rec["cost_usd"] is None
    # No buffered messages: the schema's minimum-one placeholder synthesizes.
    assert len(rec["messages"]) == 1
    assert rec["messages"][0]["role"] == "system"


# ----------------------------------------------------------------------
# Exactly-once
# ----------------------------------------------------------------------


def test_emit_exactly_once_under_double_close(trace_root):
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    assert ctx.close(outcome="completed", terminal_call="kanban_complete", agent_messages=AGENT_MESSAGES)
    # The run-level try/finally fires after the kanban handler already closed:
    # a second close on the same context must not append.
    assert not ctx.close(outcome="crashed", error_class="RuntimeError", agent_messages=AGENT_MESSAGES)
    assert len(_records(trace_root)) == 1


# ----------------------------------------------------------------------
# Module-level API: identity late-binding, redaction hook, fail-closed
# ----------------------------------------------------------------------


def test_module_lifecycle_completed(trace_root, monkeypatch):
    monkeypatch.setattr(rt, "prompt_builder_sha", lambda: "a" * 40)
    monkeypatch.setattr(rt, "config_sha256", lambda: "b" * 64)
    ctx = rt.start_run_trace(
        run_id=7,
        profile_slug="viiy-coder",
        board_slug="specialized",
        task_id="t_abc",
        trace_root=trace_root,
    )
    assert ctx is rt.active_trace()

    class _Agent:
        session_id = "sess-9"
        model = "kimi-k3"
        provider = "ollama-cloud"
        messages = AGENT_MESSAGES

    rt.record_provider_usage(
        _Agent(), {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0, "cache_write_tokens": 0}, 0.0
    )
    rt.record_tool_call("kanban_complete", {"summary": "done"}, "{}", call_id="call_k", duration_ms=3)
    assert rt.close_run_trace("completed", agent=_Agent(), terminal_call="kanban_complete", exit_code=0)
    rec = _records(trace_root)[0]
    assert rec["run_id"] == 7
    assert rec["model"] == "kimi-k3"
    assert rec["provider"] == "ollama-cloud"
    terminal = [tc for tc in rec["tool_calls"] if tc["is_terminal"]]
    assert len(terminal) == 1 and terminal[0]["name"] == "kanban_complete"
    # Emit-once at module level too.
    assert not rt.close_run_trace("crashed", error=RuntimeError("late"))
    assert rt.active_trace() is None


def test_tool_call_arguments_redacted_before_buffer(trace_root):
    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="s")
    secret = "sk-ant-" + "x" * 30
    ctx.record_tool_call("terminal", {"command": f"curl -H 'Authorization: {secret}'"}, "ok", call_id="c1")
    ctx.close(outcome="completed", agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    blob = json.dumps(rec)
    assert secret not in blob
    assert rec["tool_calls"][0]["name"] == "terminal"
    assert rec.get("redaction_log"), "a credential marker should land in redaction_log"
    assert any(e["reason"] == "credential" for e in rec["redaction_log"])


def test_fail_closed_when_redactor_raises(trace_root, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("detector blew up")

    import agent.trace_redactor as red_mod

    monkeypatch.setattr(red_mod, "redact", _boom)
    # run_trace resolves trace_redactor lazily at call time, so the patched
    # module attribute is what record_tool_call invokes.

    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="s")
    ctx.record_tool_call("read_file", {"path": "x"}, "y", call_id="c2")
    # Redactor raised -> both fields carry the fail-closed marker.
    entry = ctx.tool_calls[0]
    assert entry["arguments"] == {"redacted": True, "reason": "redactor_error", "sha256": None}
    assert entry["result"] == {"redacted": True, "reason": "redactor_error", "sha256": None}
    assert ctx.close(outcome="completed", agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    _assert_valid(rec, "completed")


def test_safety_tier_keeps_highest(trace_root):
    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="s")
    ctx.note_safety_tier("green")
    ctx.record_tool_call("read_file", {}, "ok", call_id="a")
    ctx.note_safety_tier("red")
    ctx.note_safety_tier("yellow")
    ctx.close(outcome="completed", agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    assert rec["safety_tier_hit"] == "red"
    _assert_valid(rec, "completed")


def test_cost_aggregates_across_calls(trace_root):
    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="s")
    ctx.record_usage({"input_tokens": 5, "output_tokens": 5, "cache_read_tokens": 0, "cache_write_tokens": 0}, 0.5)
    ctx.record_usage({"input_tokens": 7, "output_tokens": 3, "cache_read_tokens": 2, "cache_write_tokens": 1}, None)
    ctx.close(outcome="completed", agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    assert rec["token_counts"] == {"input": 12, "output": 8, "cache_read": 2, "cache_write": 1, "total": 23}
    assert rec["cost_usd"] == pytest.approx(0.5)
    # The second call returned usage, so the run's accounting stays complete.
    assert rec["token_counts_complete"] is True


# ----------------------------------------------------------------------
# Regression: close() must not deadlock on redactable message content
# (QA audit t_0b40eccc, finding 8 — Lock re-acquired at old :425 under the
# lock held from close's entry).
# ----------------------------------------------------------------------


def test_close_no_deadlock_on_redactable_message_content(trace_root):
    import threading

    secret = "sk-ant-" + "x" * 30
    ctx = _open_ctx(trace_root)
    _prime(ctx)
    msgs = AGENT_MESSAGES + [
        {"role": "user", "content": f"the key is {secret} and the reply address is a@b.co"}
    ]
    done: list[bool] = []

    def _close() -> None:
        done.append(ctx.close(outcome="completed", agent_messages=msgs, exit_code=0))

    t = threading.Thread(target=_close)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "close() deadlocked on redactable message content"
    assert done == [True]
    rec = _records(trace_root)[0]
    _assert_valid(rec, "completed")
    blob = json.dumps(rec)
    assert secret not in blob
    assert "a@b.co" not in blob
    user_msgs = [m for m in rec["messages"] if m["role"] == "user"]
    assert user_msgs and isinstance(user_msgs[-1]["content"], str)
    assert user_msgs[-1]["content"] != f"the key is {secret} and the reply address is a@b.co"
    # Message-content redaction events land in the record-level log, keyed by path.
    assert any(e["path"].startswith("messages[") for e in rec["redaction_log"])
    assert any(e["reason"] == "credential" for e in rec["redaction_log"])


# ----------------------------------------------------------------------
# tool_executor hook: exit_code extracted from the tool result payload
# (QA audit t_0b40eccc minor — the hook never passed it, so
# tool_calls[].exit_code was always null; follow-up t_4f637fdb).
# ----------------------------------------------------------------------


def test_extract_tool_exit_code_shapes():
    from agent.tool_executor import _extract_tool_exit_code

    assert _extract_tool_exit_code({"output": "x", "exit_code": 0, "error": None}) == 0
    assert _extract_tool_exit_code('{"output": "x", "exit_code": 2, "error": null}') == 2
    assert _extract_tool_exit_code('{"output": "", "exit_code": -1, "error": "boom"}') == -1
    assert _extract_tool_exit_code({"exit_code": None}) is None
    assert _extract_tool_exit_code("plain text result") is None
    assert _extract_tool_exit_code('{"status": "ok"}') is None
    assert _extract_tool_exit_code('{"exit_code": "1"}') is None  # str is not a code
    assert _extract_tool_exit_code({"exit_code": True}) is None  # bool is not a code
    assert _extract_tool_exit_code("not json {") is None
    assert _extract_tool_exit_code(None) is None


def test_record_tool_call_populates_exit_code(trace_root):
    ctx = _open_ctx(trace_root)
    ctx.set_identity(session_id="s")
    ctx.record_tool_call(
        "terminal",
        {"command": "false"},
        '{"output": "", "exit_code": 1, "error": null}',
        call_id="call_rc",
        duration_ms=4,
        exit_code=1,
    )
    ctx.close(outcome="completed", agent_messages=AGENT_MESSAGES)
    rec = _records(trace_root)[0]
    tool = rec["tool_calls"][0]
    assert tool["exit_code"] == 1
    _assert_valid(rec, "completed")
