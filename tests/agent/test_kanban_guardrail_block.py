"""Tests for the Kanban worker's guardrail-halt terminal block.

Regression for the 2026-11-03 00:00-00:09 UTC incident on card ``t_d8d6c78f``:
a headless Kanban worker's turn was halted by the tool guardrail
(``idempotent_no_progress_block`` on ``search_files``). The halt produced a normal
final response, so the worker exited rc=0 with no terminal Kanban call; the
dispatcher booked a protocol violation, re-spawned the card into the SAME
guardrail, and gave up after three violations.

The fix records the halt as a terminal ``kanban_block`` so the run reaches a
terminal state and the card becomes a human-input problem instead of a respawn
loop.

Placement mirrors the source tree: ``tests/agent/test_kanban_guardrail_block.py``.
"""

from __future__ import annotations

import pytest


def _halt_result(**overrides):
    result = {
        "turn_exit_reason": "guardrail_halt",
        "failed": False,
        "partial": False,
        "completed": True,
        "final_response": "I stopped retrying because I kept running search_files 5 times.",
        "guardrail": {
            "action": "block",
            "code": "idempotent_no_progress_block",
            "tool_name": "search_files",
            "count": 5,
        },
    }
    result.update(overrides)
    return result


def test_is_guardrail_halt_detects_the_halt():
    from agent.kanban_guardrail_block import is_guardrail_halt

    assert is_guardrail_halt(_halt_result()) is True
    assert is_guardrail_halt(_halt_result(turn_exit_reason="text_response")) is False
    assert is_guardrail_halt({"failed": True}) is False
    assert is_guardrail_halt(None) is False
    assert is_guardrail_halt("not a dict") is False


def test_block_reason_names_the_guardrail_and_says_it_is_headless():
    from agent.kanban_guardrail_block import block_reason

    reason = block_reason(_halt_result())
    assert "idempotent_no_progress_block" in reason
    assert "search_files" in reason
    # The guardrail's own advice assumes an interactive user; the reason must say
    # that no such user exists here, or a human will look for one.
    assert "headless" in reason
    assert "continue" in reason


def test_block_reason_survives_a_result_without_guardrail_metadata():
    from agent.kanban_guardrail_block import block_reason

    reason = block_reason({"turn_exit_reason": "guardrail_halt"})
    assert reason
    assert "guardrail" in reason


def test_non_worker_never_blocks(monkeypatch):
    """A human at a terminal keeps the normal 'how would you like to proceed?' reply."""
    from agent import kanban_guardrail_block as mod

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert mod.block_guardrail_halt(_halt_result()) is False


def test_a_non_halt_turn_never_blocks(monkeypatch):
    from agent import kanban_guardrail_block as mod

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_card")
    assert mod.block_guardrail_halt(_halt_result(turn_exit_reason="text_response")) is False
    assert mod.block_guardrail_halt(None) is False


def test_guardrail_halt_blocks_the_card(monkeypatch):
    """The worker's own card must end in a terminal blocked state."""
    from agent import kanban_guardrail_block as mod

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_card")

    calls: list[dict] = []

    class _FakeTask:
        block_kind = "capability"

    class _FakeKB:
        @staticmethod
        def block_task(conn, task_id, *, reason, kind=None, expected_run_id=None):
            calls.append({
                "task_id": task_id,
                "reason": reason,
                "kind": kind,
                "expected_run_id": expected_run_id,
            })
            return True

        @staticmethod
        def get_task(conn, task_id):
            return _FakeTask()

    class _FakeConn:
        def close(self):
            return None

    class _FakeConnect:
        @staticmethod
        def connect(board=None):
            return _FakeConn()

    import hermes_cli.kanban_db as kb_mod
    import hermes_cli.kanban_db_connect as kbc_mod

    monkeypatch.setattr(kb_mod, "block_task", _FakeKB.block_task)
    monkeypatch.setattr(kbc_mod, "connect", _FakeConnect.connect)

    emitted: list[str] = []
    assert mod.block_guardrail_halt(_halt_result(), emit=emitted.append) is True

    assert len(calls) == 1
    assert calls[0]["task_id"] == "t_some_card"
    assert calls[0]["kind"] == "capability"
    assert "idempotent_no_progress_block" in calls[0]["reason"]
    assert emitted and "block" in emitted[0]


def test_block_failure_is_reported_not_raised(monkeypatch):
    """A board that cannot be written must not raise: the dispatcher's protocol
    violation is the correct fallback, and an exception here would mask the
    original halt."""
    from agent import kanban_guardrail_block as mod

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_card")

    class _FakeKB:
        @staticmethod
        def block_task(conn, task_id, *, reason, kind=None, expected_run_id=None):
            raise RuntimeError("board is locked")

    class _FakeConn:
        def close(self):
            return None

    class _FakeConnect:
        @staticmethod
        def connect(board=None):
            return _FakeConn()

    import hermes_cli.kanban_db as kb_mod
    import hermes_cli.kanban_db_connect as kbc_mod

    monkeypatch.setattr(kb_mod, "block_task", _FakeKB.block_task)
    monkeypatch.setattr(kbc_mod, "connect", _FakeConnect.connect)

    assert mod.block_guardrail_halt(_halt_result()) is False


def test_block_lands_as_a_terminal_state_not_a_respawn(monkeypatch):
    """The blocked card must not be requeued for a respawn: a guardrail halt
    reproduces unchanged, so the card has to wait for a human."""
    from agent import kanban_guardrail_block as mod

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_card")

    calls: list[dict] = []

    class _FakeKB:
        @staticmethod
        def block_task(conn, task_id, *, reason, kind=None, expected_run_id=None):
            calls.append({"kind": kind})
            return True

        @staticmethod
        def get_task(conn, task_id):
            return None

    class _FakeConn:
        def close(self):
            return None

    class _FakeConnect:
        @staticmethod
        def connect(board=None):
            return _FakeConn()

    import hermes_cli.kanban_db as kb_mod
    import hermes_cli.kanban_db_connect as kbc_mod

    monkeypatch.setattr(kb_mod, "block_task", _FakeKB.block_task)
    monkeypatch.setattr(kbc_mod, "connect", _FakeConnect.connect)

    assert mod.block_guardrail_halt(_halt_result()) is True
    # A capability block is sticky until a human unblocks it; a transient block
    # would be respawned into the same halt.
    assert calls[0]["kind"] == mod.BLOCK_KIND
    assert mod.BLOCK_KIND == "capability"


@pytest.mark.parametrize("missing", ["HERMES_KANBAN_TASK"])
def test_worker_without_a_task_env_never_blocks(monkeypatch, missing):
    from agent import kanban_guardrail_block as mod

    monkeypatch.delenv(missing, raising=False)
    assert mod.block_guardrail_halt(_halt_result()) is False
