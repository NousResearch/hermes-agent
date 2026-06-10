"""Action Runtime contract + adapter tests.

The load-bearing guarantee is the byte-compat round-trip:
``shell_to_wire(shell_to_result(...))`` must equal the exact dict the
shell.exec handler returns today, so routing the handler through the adapter
cannot change the wire. (See docs/architecture/central-brain-openclaw.md §12.)
"""

from action_runtime import (
    ErrorType,
    ExecutionResult,
    ExecutionTask,
    Status,
    cli_to_result,
    cli_to_wire,
    plugin_to_result,
    plugin_to_wire,
    result_to_wire_rich,
    shell_to_result,
    shell_to_wire,
    slash_to_result,
    slash_to_wire,
)


class _Effect:
    """Stand-in for tui_gateway.server._SlashSideEffect (duck-typed)."""

    def __init__(self, kind="", message=""):
        self.kind = kind
        self.message = message


def test_shell_success_round_trip_is_byte_identical():
    r = shell_to_result(stdout="hi", stderr="", code=0)
    assert r.status is Status.SUCCEEDED
    assert r.ok
    assert r.error is None
    # Byte-identical to the shell.exec _ok payload.
    assert shell_to_wire(r) == {"stdout": "hi", "stderr": "", "code": 0}


def test_shell_nonzero_exit_is_failed_but_wire_is_unchanged():
    r = shell_to_result(stdout="", stderr="boom", code=3)
    assert r.status is Status.FAILED
    assert not r.ok
    assert r.error is not None
    assert r.error.type is ErrorType.NONZERO_EXIT
    assert r.error.retryable is False
    # The honest FAILED status lives in the structured result; the wire dict a
    # client receives is the SAME shape as success — failure is read off `code`.
    assert shell_to_wire(r) == {"stdout": "", "stderr": "boom", "code": 3}


def test_shell_wire_preserves_already_truncated_strings_verbatim():
    # The handler truncates (stdout[-4000:]/stderr[-2000:]) BEFORE calling the
    # adapter; the adapter must echo whatever it is handed, never re-slice.
    big_out = "x" * 4000
    big_err = "y" * 2000
    r = shell_to_result(stdout=big_out, stderr=big_err, code=0)
    wire = shell_to_wire(r)
    assert wire["stdout"] == big_out
    assert wire["stderr"] == big_err
    assert len(wire["stdout"]) == 4000 and len(wire["stderr"]) == 2000


def test_execution_result_defaults():
    r = ExecutionResult(task_id=None, status=Status.SUCCEEDED)
    assert r.outputs == {}
    assert r.side_effects == []
    assert r.needs_input is None
    assert r.error is None


def test_cli_success_round_trip_is_byte_identical():
    r = cli_to_result(blocked=False, code=0, output="ok")
    assert r.status is Status.SUCCEEDED
    assert cli_to_wire(r) == {"blocked": False, "code": 0, "output": "ok"}


def test_cli_nonzero_exit_is_failed_but_wire_unchanged():
    r = cli_to_result(blocked=False, code=2, output="boom")
    assert r.status is Status.FAILED
    assert r.error is not None and r.error.type is ErrorType.NONZERO_EXIT
    assert cli_to_wire(r) == {"blocked": False, "code": 2, "output": "boom"}


def test_cli_blocked_round_trip_preserves_hint_and_shape():
    r = cli_to_result(blocked=True, code=-1, output="", hint="needs a terminal")
    assert r.status is Status.BLOCKED
    # Byte-identical to the blocked-gate _ok payload.
    assert cli_to_wire(r) == {
        "blocked": True,
        "hint": "needs a terminal",
        "code": -1,
        "output": "",
    }


# ── plugin path ──────────────────────────────────────────────────────


def test_plugin_success_round_trip():
    r = plugin_to_result("hi there")
    assert r.status is Status.SUCCEEDED
    assert plugin_to_wire(r) == {"output": "hi there"}


def test_plugin_failure_sets_output_and_error_identically():
    r = plugin_to_result(exc=RuntimeError("boom"))
    assert r.status is Status.FAILED
    assert r.error is not None and r.error.type is ErrorType.PROVIDER_ERROR
    # Both output and error equal "Plugin command error: boom" (Phase 1a test).
    assert plugin_to_wire(r) == {
        "output": "Plugin command error: boom",
        "error": "Plugin command error: boom",
    }


# ── slash worker path (live-agent side effects) ──────────────────────


def test_slash_clean_has_no_warning_or_error():
    r = slash_to_result("done", _Effect(kind="", message=""))
    assert r.status is Status.SUCCEEDED
    assert r.error is None
    assert slash_to_wire(r, warning="") == {"output": "done"}


def test_slash_benign_warning_has_warning_but_no_error():
    eff = _Effect(kind="warning", message="model not in catalog — proceeding")
    r = slash_to_result("ok", eff)
    assert r.status is Status.SUCCEEDED
    assert r.error is None
    assert r.side_effects[0].applied is True
    # Successful switch carries an advisory warning, never an error.
    assert slash_to_wire(r, warning=eff.message) == {
        "output": "ok",
        "warning": "model not in catalog — proceeding",
    }


def test_slash_failure_promotes_warning_to_error():
    eff = _Effect(kind="failure", message="'bad/model' unavailable")
    rendered = "live session sync failed: 'bad/model' unavailable"
    r = slash_to_result("  ✗ not available", eff)
    assert r.status is Status.FAILED
    assert r.error is not None and r.error.type is ErrorType.PROVIDER_ERROR
    assert r.side_effects[0].applied is False
    # error mirrors the rendered warning (the desktop picker rolls back on it).
    assert slash_to_wire(r, warning=rendered) == {
        "output": "  ✗ not available",
        "warning": rendered,
        "error": rendered,
    }


def test_slash_busy_is_failed_and_retryable():
    eff = _Effect(kind="busy", message="/interrupt the current turn first")
    rendered = "session busy — /interrupt the current turn first"
    r = slash_to_result("busy", eff)
    assert r.status is Status.FAILED
    assert r.error is not None and r.error.type is ErrorType.TRANSPORT
    assert r.error.retryable is True
    assert slash_to_wire(r, warning=rendered) == {
        "output": "busy",
        "warning": rendered,
        "error": rendered,
    }


# ── Phase 4: additive task_id echo on the legacy wire ────────────────


def test_slash_wire_echoes_task_id_additively():
    r = slash_to_result("done", _Effect(kind="", message=""), task_id="task-1")
    assert slash_to_wire(r, warning="") == {"output": "done", "task_id": "task-1"}


def test_slash_wire_without_task_id_is_unchanged():
    r = slash_to_result("done", _Effect(kind="", message=""))
    assert slash_to_wire(r, warning="") == {"output": "done"}


def test_plugin_wire_echoes_task_id_additively():
    r = plugin_to_result("hi", task_id="task-2")
    assert plugin_to_wire(r) == {"output": "hi", "task_id": "task-2"}


def test_plugin_failure_wire_echoes_task_id_additively():
    r = plugin_to_result(exc=RuntimeError("boom"), task_id="task-3")
    assert plugin_to_wire(r) == {
        "output": "Plugin command error: boom",
        "error": "Plugin command error: boom",
        "task_id": "task-3",
    }


# ── Phase 4: rich wire renderer (task.submit) ────────────────────────


def test_rich_wire_success_renders_every_key():
    r = slash_to_result("done", _Effect(kind="", message=""), task_id="task-4")
    assert result_to_wire_rich(r) == {
        "task_id": "task-4",
        "status": "succeeded",
        "outputs": {"output": "done"},
        "error": None,
        "side_effects": [],
    }


def test_rich_wire_failure_renders_structured_error_and_side_effects():
    eff = _Effect(kind="failure", message="'bad/model' unavailable")
    r = slash_to_result("  ✗ not available", eff, task_id="task-5")
    assert result_to_wire_rich(r) == {
        "task_id": "task-5",
        "status": "failed",
        "outputs": {"output": "  ✗ not available"},
        "error": {
            "type": "provider_error",
            "retryable": False,
            "message": "'bad/model' unavailable",
        },
        "side_effects": [
            {
                "kind": "slash_sync",
                "detail": "'bad/model' unavailable",
                "applied": False,
                "target": None,
            }
        ],
    }


def test_rich_wire_busy_is_retryable_transport_error():
    eff = _Effect(kind="busy", message="/interrupt the current turn first")
    r = slash_to_result("busy", eff, task_id="task-6")
    rich = result_to_wire_rich(r)
    assert rich["status"] == "failed"
    assert rich["error"] == {
        "type": "transport",
        "retryable": True,
        "message": "/interrupt the current turn first",
    }
    assert rich["side_effects"] == [
        {
            "kind": "slash_sync",
            "detail": "/interrupt the current turn first",
            "applied": False,
            "target": None,
        }
    ]


def test_rich_wire_blocked_cli_keeps_enum_value_strings():
    # status/error.type must be the ENUM VALUE strings, not Enum reprs.
    r = cli_to_result(blocked=True, code=-1, output="", hint="needs a terminal")
    rich = result_to_wire_rich(r)
    assert rich["status"] == "blocked"
    assert isinstance(rich["status"], str)
    assert rich["error"]["type"] == "denied"
    assert rich["task_id"] is None  # never omitted, even when unset


# ── Phase 5 Task 2.1: trace_id plumbing (§12 observability) ──────────


def test_contract_trace_id_round_trip():
    """trace_id rides both contract dataclasses; default None marks a
    pre-trace caller (the Runtime never synthesizes one)."""
    t = ExecutionTask(task_id="t-1", trace_id="trace-1")
    assert t.trace_id == "trace-1"
    r = ExecutionResult(task_id="t-1", status=Status.SUCCEEDED, trace_id=t.trace_id)
    assert r.trace_id == "trace-1"


def test_contract_trace_id_defaults_to_none():
    assert ExecutionTask(task_id="t-1").trace_id is None
    assert ExecutionResult(task_id="t-1", status=Status.SUCCEEDED).trace_id is None


def test_rich_wire_omits_trace_id_when_unset():
    """Byte-compat pin: a pre-trace result must render the EXACT pre-trace
    rich dict — no trace_id key at all (absent-when-None, same additive
    pattern as snapshot()'s session_id)."""
    r = slash_to_result("done", _Effect(kind="", message=""), task_id="task-7")
    assert result_to_wire_rich(r) == {
        "task_id": "task-7",
        "status": "succeeded",
        "outputs": {"output": "done"},
        "error": None,
        "side_effects": [],
    }


def test_rich_wire_carries_trace_id_when_set():
    r = slash_to_result("done", _Effect(kind="", message=""), task_id="task-8")
    r.trace_id = "trace-8"
    assert result_to_wire_rich(r) == {
        "task_id": "task-8",
        "status": "succeeded",
        "outputs": {"output": "done"},
        "error": None,
        "side_effects": [],
        "trace_id": "trace-8",
    }
