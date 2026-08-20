from __future__ import annotations

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import kanban_db
from hermes_cli.lifecycle import invoke_hook
from agent.run_usage_ledger import process_ledger, reset_process_ledger_cache


def test_lifecycle_hooks_record_direct_and_card_receipts(tmp_path, monkeypatch):
    token = set_hermes_home_override(tmp_path)
    reset_process_ledger_cache()
    monkeypatch.setenv("HERMES_RUN_ID", "run-hook")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-hook")
    try:
        invoke_hook(
            "on_session_start",
            session_id="session-hook",
            model="model-hook",
            provider="provider-hook",
            platform="cli",
            task_id="task-hook",
        )
        invoke_hook(
            "post_api_request",
            session_id="session-hook",
            task_id="task-hook",
            turn_id="turn-hook",
            api_request_id="api-hook",
            model="model-hook",
            provider="provider-hook",
            usage={"input_tokens": 100, "output_tokens": 30},
            cost_usd=0.25,
        )
        invoke_hook(
            "api_request_error",
            session_id="session-hook",
            api_request_id="api-retry",
            retry_count=1,
            retryable=True,
            error={"type": "RateLimitError"},
        )
        invoke_hook(
            "post_tool_call",
            session_id="session-hook",
            task_id="task-hook",
            tool_call_id="tool-hook",
        )
        invoke_hook(
            "on_session_finalize",
            session_id="session-hook",
            completed=True,
            platform="cli",
        )

        receipt = process_ledger().get_run("run-hook")
        assert receipt["task_id"] == "task-hook"
        assert receipt["input_tokens"] == 100
        assert receipt["output_tokens"] == 30
        assert receipt["cost_usd"] == 0.25
        assert receipt["turn_count"] == 1
        assert receipt["tool_call_count"] == 1
        assert receipt["retry_count"] == 1
        assert receipt["outcome"] == "completed"
        assert receipt["ended_at"] is not None
    finally:
        reset_hermes_home_override(token)
        reset_process_ledger_cache()


def test_dispatcher_style_lifecycle_links_exact_task_run(tmp_path, monkeypatch):
    board = tmp_path / "kanban.db"
    kanban_db.init_db(board)
    with kanban_db.connect_closing(board) as connection:
        task_id = kanban_db.create_task(connection, title="worker", assignee="default")
        claimed = kanban_db.claim_task(connection, task_id, claimer="dispatcher:test")
        assert claimed is not None
        task_run_id = connection.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0]

    token = set_hermes_home_override(tmp_path / "profile")
    reset_process_ledger_cache()
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    try:
        invoke_hook("on_session_start", session_id="s", model="m", provider="p")
        invoke_hook(
            "post_api_request", session_id="s", turn_id="t", api_request_id="api",
            model="m", provider="p", usage={"input_tokens": 7, "output_tokens": 8}, cost_usd=0.9,
        )
        invoke_hook("on_session_finalize", session_id="s", completed=True)
        with kanban_db.connect_closing(board) as connection:
            row = connection.execute(
                "SELECT task_run_id, usage_run_id, input_tokens, output_tokens, cost_usd FROM task_run_usage"
            ).fetchone()
        assert tuple(row) == (task_run_id, f"task-run:{task_run_id}", 7, 8, 0.9)
    finally:
        reset_hermes_home_override(token)
        reset_process_ledger_cache()


def test_production_emitters_handle_tool_retry_interrupt_and_no_finalize(tmp_path, monkeypatch):
    token = set_hermes_home_override(tmp_path)
    reset_process_ledger_cache()
    monkeypatch.setenv("HERMES_RUN_ID", "run-emitter")
    try:
        from model_tools import _emit_post_tool_call_hook

        invoke_hook("on_session_start", session_id="s", model="m", provider="p")
        _emit_post_tool_call_hook(
            function_name="local_tool", function_args={}, result="ok",
            session_id="s", tool_call_id="tool-stable", turn_id="t",
        )
        _emit_post_tool_call_hook(function_name="local_tool", function_args={}, result="ok", session_id="s")
        invoke_hook(
            "api_request_error", session_id="s", api_request_id="api-stable",
            model="m", provider="p", retry_count=1, retryable=True,
            error={"type": "TransportError"},
        )
        invoke_hook("on_session_finalize", session_id="s", interrupted=True, reason="interrupt")
        receipt = process_ledger().get_run("run-emitter")
        assert receipt["tool_call_count"] == 1
        assert receipt["retry_count"] == 1
        assert receipt["outcome"] == "interrupted"

        monkeypatch.setenv("HERMES_RUN_ID", "run-no-finalize")
        invoke_hook("on_session_start", session_id="s2", model="m", provider="p")
        invoke_hook(
            "post_api_request", session_id="s2", turn_id="t2", api_request_id="api-2",
            model="m", provider="p", usage={"input_tokens": 1, "output_tokens": 1},
        )
        process_ledger().shutdown()
        assert process_ledger().get_run("run-no-finalize")["ended_at"] is None
    finally:
        reset_hermes_home_override(token)
        reset_process_ledger_cache()
