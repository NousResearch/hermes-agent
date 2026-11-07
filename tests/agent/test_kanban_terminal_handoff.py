"""A kanban worker that lands a terminal board call must end its turn.

``agent/kanban_stop.py`` guards the MIRROR direction — a worker that exits with NO
terminal call gets nudged to make one. This file pins the opposite direction: once a
terminal call has landed, the card belongs to whoever owns it next, and the worker's own
process must not keep iterating on it. Measured 2026-09-23 on ops/t_4ff2c95c: run 2666
called ``kanban_block``, then its still-live process performed the card's entire
implementation and filed a child card while the respawn already held the card under a
new run id.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
from agent.kanban_stop import _TERMINAL_KANBAN_TOOLS, terminal_handoff_status
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.turn_tool_round import run_tool_round
from agent.turn_tool_validation import ToolValidationVerdict
from tools.kanban_tools import _ok
from tools.registry import tool_error

# Where each terminal tool lands the card, as the handlers report it in their payload.
LANDED = {
    "kanban_complete": "done",
    "kanban_block": "blocked",
    "kanban_request_review": "review",
    "kanban_request_changes": "ready",
}


class _FakeAgent:
    """The agent surface ``run_tool_round`` touches after tool execution.

    ``_execute_tool_calls`` is stood in for (real execution needs the tool registry and a
    live board) but it appends rows built by the production constructor, so the payload
    shape the guard reads is the shape production writes.
    """

    def __init__(self, results: dict[str, str]):
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix = ""
        self.valid_tool_names = set(_TERMINAL_KANBAN_TOOLS) | {"kanban_comment", "kanban_heartbeat"}
        self.stream_delta_callback = None
        self.session_id = "sess-test"
        self._incremental_persistence_failed = False
        self._tool_guardrail_halt_decision = None
        self.iteration_budget = SimpleNamespace(refund=lambda: None)
        self._session_messages: list = []
        self.results = results
        self.printed: list[str] = []

    def _flush_messages_to_session_db(self, messages, conversation_history):
        return True

    def _deduplicate_tool_calls(self, calls):
        return calls

    def _cap_delegate_task_calls(self, calls):
        return calls

    def _emit_interim_assistant_message(self, message):
        pass

    def _safe_print(self, text):
        self.printed.append(text)

    def _touch_activity(self, note):
        pass

    def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count):
        for tc in assistant_message.tool_calls:
            messages.append(
                make_tool_result_message(tc.function.name, self.results[tc.id], tc.id)
            )


def _round(monkeypatch, calls, results, *, messages=None):
    """One tool round through ``run_tool_round``.

    The neighbours with their own tests are patched at this module's seam: call
    validation, the assistant-row staging, and post-tool compression. Everything the
    handoff guard itself depends on (the executed rows, the scope gate, the exit reason
    and its failure classification) is real.
    """
    agent = _FakeAgent(results)
    monkeypatch.setattr(
        "agent.turn_tool_round.validate_tool_calls",
        lambda *a, **kw: ToolValidationVerdict(action="ok", result=None, mixed_invalid_batch=False),
    )

    def _stage(agent_, *, assistant_message, finish_reason, messages):
        row = {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_message.tool_calls
            ],
        }
        messages.append(row)
        return row, False

    monkeypatch.setattr("agent.turn_tool_round.stage_tool_call_message", _stage)
    monkeypatch.setattr(
        "agent.turn_tool_round.compress_after_tool_results",
        lambda agent_, **kw: SimpleNamespace(
            messages=kw["messages"],
            active_system_prompt=kw["active_system_prompt"],
            conversation_history=kw["conversation_history"],
            compression_attempts=kw["compression_attempts"],
            final_response=kw["final_response"],
            turn_exit_reason=kw["turn_exit_reason"],
            current_turn_user_idx=kw["current_turn_user_idx"],
            end_turn=False,
        ),
    )
    msgs = list(messages) if messages is not None else [
        {"role": "user", "content": "work kanban task t_handoff"}
    ]
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id=call_id, function=SimpleNamespace(name=name, arguments="{}")
            )
            for call_id, name in calls
        ],
    )
    verdict = run_tool_round(
        agent,
        assistant_message=assistant_message,
        finish_reason="tool_calls",
        messages=msgs,
        conversation_history=[],
        api_call_count=1,
        effective_task_id="t_handoff",
        user_message="work kanban task t_handoff",
        system_message="sys",
        active_system_prompt="sys",
        compression_attempts=0,
        max_compression_attempts=3,
        final_response="",
        failed=False,
        _turn_exit_reason=None,
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
    )
    return agent, msgs, verdict


@pytest.fixture
def clear_kanban_env(monkeypatch):
    """Kanban identity, the way the runner hands it to a worker: TASK present, no nudge
    override, and no inherited delegated-child marker (that marker turns every kanban
    identity gate off, including this one)."""
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def worker_env(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    return clear_kanban_env


def test_handoff_exit_reason_is_not_a_failure():
    """The turn must finalize as completed, not failed: the dispatcher reads a clean
    handoff, and a failure classification here would mark a correct worker red."""
    from agent.turn_failure_copy import exit_reason_failure

    assert exit_reason_failure("kanban_terminal_handoff") is None


@pytest.mark.parametrize("tool_name", sorted(_TERMINAL_KANBAN_TOOLS))
def test_successful_terminal_call_breaks_the_turn(worker_env, tool_name):
    agent, messages, verdict = _round(
        worker_env, [("c0", tool_name)], {"c0": _ok(status=LANDED[tool_name])}
    )

    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "kanban_terminal_handoff"
    assert verdict.failed is False
    assert verdict.final_response
    assert LANDED[tool_name] in verdict.final_response
    # The turn's own words are persisted as the closing row (append_message stamps it).
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == verdict.final_response
    assert agent.printed == [f"\n{verdict.final_response}\n"]


@pytest.mark.parametrize("tool_name", sorted(_TERMINAL_KANBAN_TOOLS))
def test_refused_terminal_call_keeps_the_worker_alive(worker_env, tool_name):
    """A tool_error means the call did NOT land — the worker must retry, not die."""
    _, _, verdict = _round(
        worker_env, [("c0", tool_name)], {"c0": tool_error("ownership claim refused")}
    )

    assert verdict.action == "continue"


def test_non_terminal_kanban_tool_keeps_the_worker_alive(worker_env):
    _, _, verdict = _round(worker_env, [("c0", "kanban_comment")], {"c0": _ok()})

    assert verdict.action == "continue"


def test_no_owned_task_means_no_break(monkeypatch):
    """Interactive CLI users and orchestrators with no owned card are out of scope."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)

    _, _, verdict = _round(monkeypatch, [("c0", "kanban_complete")], {"c0": _ok(status="done")})

    assert verdict.action == "continue"


def test_delegated_child_never_breaks(worker_env):
    """A delegate_task child inherits the env var but owns no board card."""
    with delegated_child_context():
        _, _, verdict = _round(
            worker_env, [("c0", "kanban_complete")], {"c0": _ok(status="done")}
        )

    assert verdict.action == "continue"


def test_only_this_rounds_result_hands_off(worker_env):
    """Success is matched to the calls THIS round executed: an earlier terminal result in
    history never stands in for the call that just ran and failed."""
    history = [
        {"role": "user", "content": "work kanban task t_handoff"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        make_tool_result_message("kanban_complete", _ok(status="done"), "old"),
    ]

    _, _, verdict = _round(
        worker_env, [("new", "kanban_complete")], {"new": tool_error("refused")}, messages=history
    )

    assert verdict.action == "continue"


def test_helper_reports_the_landed_status():
    calls = [
        {
            "id": "c0",
            "type": "function",
            "function": {"name": "kanban_block", "arguments": "{}"},
        }
    ]
    messages = [
        {"role": "assistant", "content": "", "tool_calls": calls},
        make_tool_result_message("kanban_block", _ok(status="blocked"), "c0"),
    ]

    assert terminal_handoff_status(messages=messages, tool_calls=calls) == "blocked"


def test_helper_ignores_a_result_it_cannot_parse():
    """A non-payload result (e.g. a spilled/stubbed row) is not evidence of a handoff."""
    calls = [
        {
            "id": "c0",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {"role": "assistant", "content": "", "tool_calls": calls},
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "c0",
            "content": "[Tool result persisted to file: /tmp/result.txt]",
        },
    ]

    assert terminal_handoff_status(messages=messages, tool_calls=calls) is None
