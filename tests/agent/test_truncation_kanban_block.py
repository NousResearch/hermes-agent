"""Regression tests for #115999: when a dispatcher-owned Kanban worker exhausts its
truncated-tool-call retries, the give-up site records a ``transient`` block on the
worker's behalf. The worker cannot emit ANY tool call (that is exactly the failure),
so it cannot file ``kanban_block`` itself and its exit would otherwise be booked as a
protocol violation against a diagnostic the worker had no way to satisfy.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

# Dummy fixture credential (concatenated so no literal key-shaped string appears).
_AGENT_KEY = "unit" + "-test-key"


def _make_agent(tool_call_fragment='{"path":'):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key=_AGENT_KEY,
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    # The transport's normalized view of the truncated response: a tool call whose
    # arguments were cut off mid-JSON (truthy tool_calls routes to the retry site).
    agent._get_transport = MagicMock(return_value=MagicMock(
        normalize_response=MagicMock(return_value=SimpleNamespace(
            content=None,
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": tool_call_fragment}}],
        )),
    ))
    return agent


def _truncated_response(response_id="chatcmpl-trunc"):
    return SimpleNamespace(
        id=response_id,
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(
                role="assistant", content=None,
                tool_calls=[{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": "{"}}],
                reasoning_content=None),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _run_recovery(agent, response=None):
    from agent.turn_truncation import recover_from_truncation

    messages = [{"role": "user", "content": "go"}]
    return recover_from_truncation(
        agent, response or _truncated_response(), FINISH_REASON_LENGTH, MagicMock(),
        messages=messages, conversation_history=None, api_kwargs={},
        api_call_count=1, effective_task_id=None, current_turn_user_idx=None,
        length_continue_retries=0, truncated_response_parts=[],
        # Retries already spent: the give-up branch runs on the first visit.
        truncated_tool_call_retries=4, retry_count=0, compression_attempts=0,
    )


def _patch_board(monkeypatch, *, block_result=True, block_exc=None):
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_connect as kbc

    conn = MagicMock(name="conn")
    block = MagicMock(name="block_task", return_value=block_result,
                      side_effect=block_exc)
    monkeypatch.setattr(kbc, "connect", MagicMock(name="connect", return_value=conn))
    monkeypatch.setattr(kb, "block_task", block)
    return conn, block


def _patch_worker_env(monkeypatch, *, delegated=False, owner=True):
    import agent.delegation_context as dctx

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_115999")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setattr(dctx, "is_delegated_child_context", lambda: delegated)
    monkeypatch.setattr(dctx, "is_dispatcher_owned_worker_context", lambda: owner)


class TestTruncationKanbanBlock:
    def test_exhausted_retries_block_on_worker_behalf(self, monkeypatch):
        conn, block = _patch_board(monkeypatch)
        _patch_worker_env(monkeypatch)

        verdict = _run_recovery(_make_agent())

        block.assert_called_once()
        assert block.call_args.args == (conn, "t_115999")
        kwargs = block.call_args.kwargs
        assert kwargs["kind"] == "transient"
        assert kwargs["expected_run_id"] == 7
        assert "truncated" in kwargs["reason"]
        assert "behalf" in kwargs["reason"]
        conn.close.assert_called_once()
        # The board bookkeeping never displaces the truncation verdict itself.
        assert verdict.action == "return"
        result = verdict.result or {}
        assert result.get("partial") is True
        assert result.get("failure_reason") == "truncated"

    def test_stream_stall_reason_names_the_stream(self, monkeypatch):
        _, block = _patch_board(monkeypatch)
        _patch_worker_env(monkeypatch)

        _run_recovery(_make_agent(), response=_truncated_response(PARTIAL_STREAM_STUB_ID))

        assert "stream" in block.call_args.kwargs["reason"]

    def test_no_task_env_is_noop(self, monkeypatch):
        _, block = _patch_board(monkeypatch)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        _run_recovery(_make_agent())

        block.assert_not_called()

    def test_delegated_child_never_blocks_parent_task(self, monkeypatch):
        _, block = _patch_board(monkeypatch)
        _patch_worker_env(monkeypatch, delegated=True)

        _run_recovery(_make_agent())

        block.assert_not_called()

    def test_board_failure_does_not_mask_truncation(self, monkeypatch):
        _, block = _patch_board(monkeypatch, block_exc=RuntimeError("board offline"))
        _patch_worker_env(monkeypatch)

        verdict = _run_recovery(_make_agent())

        block.assert_called_once()
        assert verdict.action == "return"
        assert (verdict.result or {}).get("failure_reason") == "truncated"

    def test_cleanup_failure_still_files_the_block(self, monkeypatch):
        # The block is filed BEFORE _cleanup_task_resources: a cleanup blow-up on the
        # way out must not cost the worker the only durable record of the cause.
        _, block = _patch_board(monkeypatch)
        _patch_worker_env(monkeypatch)
        agent = _make_agent()
        monkeypatch.setattr(
            agent, "_cleanup_task_resources",
            MagicMock(side_effect=RuntimeError("cleanup exploded")),
        )

        with pytest.raises(RuntimeError, match="cleanup exploded"):
            _run_recovery(agent)

        block.assert_called_once()
