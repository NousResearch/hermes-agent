"""Exact terminal timeout retries are recorded and blocked."""

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.tool_timeout_circuit as circuit
from agent.tool_executor import _DispatchCommit, _run_agent_tool_execution_middleware
from tools.terminal_tool import _ExecPlan, _run_foreground


def _agent():
    return SimpleNamespace(
        session_id="session-a",
        quiet_mode=True,
        verbose_logging=False,
        log_prefix="",
        log_prefix_chars=120,
        tool_progress_mode="off",
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _subagent_id=None,
        _current_turn_id="turn",
        _current_api_request_id="request",
        _touch_activity=MagicMock(),
        _tool_guardrails=SimpleNamespace(
            before_call=lambda *_args: SimpleNamespace(allows_execution=True)
        ),
    )


def _reset_memory(monkeypatch, tmp_path) -> None:
    circuit._memory_entries.clear()
    monkeypatch.setattr(circuit, "_ephemeral_key", b"m" * 32)
    monkeypatch.setattr(circuit, "_ledger_path", lambda: tmp_path / "cache" / "tool-timeout-circuit.json")


def test_timeout_circuit_private_exact_and_safe_storage_fallbacks(tmp_path, monkeypatch) -> None:
    _reset_memory(monkeypatch, tmp_path)
    ledger = circuit._ledger_path()
    args = {"command": "SECRET_PAYLOAD", "timeout": 1}

    circuit.record_tool_timeout("terminal", args, "session-a")

    assert circuit.is_tool_timeout_blocked(
        "terminal", {**args, "timeout": 9, "force": True}, "session-a"
    )
    assert not circuit.is_tool_timeout_blocked("terminal", args, "session-b")
    assert "SECRET_PAYLOAD" not in ledger.read_text(encoding="utf-8")
    assert ledger.stat().st_mode & 0o777 == 0o600

    @contextmanager
    def unlocked():
        yield False

    # Lock failure must never perform an unlocked shared write; the process-local
    # circuit still blocks the immediate retry.
    fallback_ledger = tmp_path / "unlocked" / "tool-timeout-circuit.json"
    monkeypatch.setattr(circuit, "_ledger_path", lambda: fallback_ledger)
    monkeypatch.setattr(circuit, "_process_ledger_lock", unlocked)
    fallback_args = {"command": "fallback-only"}
    circuit.record_tool_timeout("terminal", fallback_args, "session-a")
    assert circuit.is_tool_timeout_blocked("terminal", fallback_args, "session-a")
    assert not fallback_ledger.exists()

    # Platforms without enforceable owner-only permissions also stay in memory.
    unsupported_ledger = tmp_path / "unsupported" / "tool-timeout-circuit.json"
    monkeypatch.setattr(circuit, "_ledger_path", lambda: unsupported_ledger)
    monkeypatch.setattr(circuit, "_persistent_storage_supported", lambda: False)
    platform_args = {"command": "platform-fallback"}
    circuit.record_tool_timeout("terminal", platform_args, "session-a")
    assert circuit.is_tool_timeout_blocked("terminal", platform_args, "session-a")
    assert not unsupported_ledger.exists()

    # An unrelated backend exception containing the word timeout is not an
    # explicit execution timeout and therefore cannot poison the circuit.
    env = SimpleNamespace(execute=MagicMock(side_effect=RuntimeError("connection timeout negotiating TLS")))
    plan = _ExecPlan(
        config={},
        env_type="local",
        effective_task_id="task",
        image="",
        cwd=str(tmp_path),
        host_cwd=None,
        effective_timeout=3,
    )
    with patch("tools.terminal_tool._resolve_command_cwd", return_value=str(tmp_path)), patch(
        "tools.terminal_tool._yield_kwargs", return_value={}
    ):
        result = json.loads(
            _run_foreground(
                "printf safe",
                env,
                plan,
                task_id="task",
                session_id="session-a",
                session_key="session-a",
                workdir=None,
                approval_note=None,
                clear_interrupt=False,
            )
        )
    assert result["exit_code"] == 124
    assert result.get("status") != "timeout"
    assert result.get("error_type") != "terminal_timeout"


def test_dispatch_boundary_linearizes_timeout_record_and_abandonment(tmp_path, monkeypatch) -> None:
    _reset_memory(monkeypatch, tmp_path)
    agent = _agent()
    final_args = {"command": "sleep 9", "timeout": 1}

    # Native structured timeouts record the hook-modified final arguments and
    # the immediate exact retry is blocked before execute.
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, final_args)):
        first = _run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "original"},
            effective_task_id="task",
            tool_call_id="call-1",
            execute=lambda _args: json.dumps(
                {"status": "timeout", "error_type": "terminal_timeout", "exit_code": 124}
            ),
        )
        exact_dispatch = MagicMock(return_value="must not run")
        second = _run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args=final_args,
            effective_task_id="task",
            tool_call_id="call-2",
            execute=exact_dispatch,
        )
    assert "terminal_timeout" in first.result
    assert second.blocked is True
    exact_dispatch.assert_not_called()

    # Deterministic race: a second call has passed policy and is parked at the
    # start gate when another worker records the first timeout. Admission is
    # rechecked at dispatch, so the parked call never executes.
    raced_args = {"command": "sleep 10", "timeout": 1}
    parked = threading.Event()
    release = threading.Event()
    raced_dispatch = MagicMock(return_value="must not run")
    outcome = []

    def delayed_begin(callback):
        parked.set()
        assert release.wait(5)
        callback()

    def run_raced_call():
        outcome.append(
            _run_agent_tool_execution_middleware(
                agent,
                function_name="terminal",
                function_args=raced_args,
                effective_task_id="task",
                tool_call_id="call-race",
                execute=raced_dispatch,
                begin_execution=delayed_begin,
            )
        )

    worker = threading.Thread(target=run_raced_call)
    worker.start()
    assert parked.wait(5)
    circuit.record_tool_timeout("terminal", raced_args, "session-a")
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert outcome[0].blocked is True
    raced_dispatch.assert_not_called()

    # The per-call lifecycle has one linearization point: abandonment before
    # commit prevents dispatch; commitment before abandonment yields a snapshot.
    abandoned = _DispatchCommit()
    assert abandoned.abandon_and_snapshot() is None
    assert abandoned.commit({"command": "late"}) is False
    committed = _DispatchCommit()
    nested_args = {"command": "started", "notify": ["ready"]}
    assert committed.commit(nested_args) is True
    snapshot = committed.abandon_and_snapshot()
    nested_args["notify"].append("mutated-after-commit")
    assert snapshot == {"command": "started", "notify": ["ready"]}
