"""A guardrail halt must never look like a clean exit while the card is still ``running``.

Runtime probes for the halt branch of ``agent.turn_tool_round``: on a dispatcher-owned
board worker, a guardrail halt delivers the shipped kanban stop-nudge (``agent.kanban_stop``
via ``turn_stop_gates``) and the turn continues, or — once that nudge budget is spent — the
run exits non-zero with a reason naming the class. Both legs drive a real ``AIAgent`` with a
mocked provider so the assertion lands on the result dict the worker entrypoint turns into an
exit code (``hermes_cli.turn_exit.turn_exit_code``), not on a constant.

The probe is self-contained (no sibling-test imports) and imports nothing new at module level,
so it collects on an unpatched tree and reports per-test FAILED there. The same bytes run
upstream and inside the shadow replay, where the bake prunes ``tests/``.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

TASK = "t-haltprobe"
BOOM = json.dumps({"error": "boom"})
HALT_MARK = "stopped retrying"
NUDGE_MARK = "has not been handed off"


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(message: SimpleNamespace, finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model", usage=None,
    )


def _calling(call_id: str, name: str = "web_search", arguments: str = '{"query": "same"}'):
    """An assistant turn whose only tool call is ``name`` with ``arguments``."""
    return _response(
        SimpleNamespace(content="", tool_calls=[_tool_call(name, arguments, call_id)]),
        "tool_calls",
    )


def _saying(content: str):
    return _response(SimpleNamespace(content=content, tool_calls=None), "stop")


def _hard_stop_config() -> dict:
    # The halt classes t_0d64c45e measured on the paid lanes, at their lowest settings:
    # two identical failures arm the block, so the third identical call halts.
    return {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }


def _make_agent(tmp_path, monkeypatch, *tool_names: str, kanban_task: str | None) -> AIAgent:
    """A worker-shaped agent: the dispatcher spawns workers with ``platform='cli'`` and the
    board identity in the environment (``gateway/session_kanban.py``), so identity is env here."""
    if kanban_task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    # A dispatched worker is not a delegate child: this marker is what makes
    # ``owned_kanban_task()`` answer "" for an in-process sub-agent that inherited the
    # env. A harness running inside a worker must clear it to model the real one.
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    # Hermetic session store: the interim answer row really is flushed, just not into a
    # profile's live state.db.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    (tmp_path / "hermes-home").mkdir(parents=True, exist_ok=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=_hard_stop_config()),
        patch("hermes_cli.config.load_config_readonly", return_value=_hard_stop_config()),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    # The mocked responses are not iterable stream chunks: force the non-streaming path so
    # the halt branch is reached without the streaming machinery (#30770's harness).
    agent._disable_streaming = True
    return agent


def _run(agent: AIAgent, responses: list):
    def dispatch(function_name, function_args, *args, **kwargs):
        if function_name == "kanban_complete":
            return json.dumps({"ok": True})
        return BOOM

    agent.client.chat.completions.create.side_effect = responses
    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("search for the thing")


def _rows(result: dict, role: str) -> list[dict]:
    return [m for m in result["messages"] if isinstance(m, dict) and m.get("role") == role]


def _matching(rows: list[dict], mark: str) -> list[dict]:
    return [m for m in rows if mark in str(m.get("content") or "")]


def test_guardrail_halt_on_a_board_worker_continues_under_the_stop_nudge(tmp_path, monkeypatch):
    """The halt must not end the run: the worker gets the shipped nudge and hands off.

    Red on the unpatched branch, which breaks at the halt: the model is never asked again,
    no nudge row exists, and the card is still `running` on a `guardrail_halt` rc=0 exit.
    """
    agent = _make_agent(tmp_path, monkeypatch, "web_search", "kanban_complete", kanban_task=TASK)
    result = _run(agent, [
        _calling("c1"),                       # fails 1
        _calling("c2"),                       # fails 2 -> identical-failure block armed
        _calling("c3"),                       # blocked before execution -> halt
        _calling("c4", name="kanban_complete", arguments='{"summary": "done"}'),  # after the nudge
        _saying("handed off"),
    ])

    nudges = _matching(_rows(result, "user"), NUDGE_MARK)
    assert nudges, (
        "the halt exited without the kanban stop-nudge: a headless worker ended its run on "
        f"copy written for a human. rows: {[m.get('content') for m in _rows(result, 'user')]!r}"
    )
    assert all(m.get("_kanban_stop_synthetic") is True for m in nudges), (
        "the nudge row must stay ephemeral scaffolding or a resumed session replays it"
    )
    halts = _matching(_rows(result, "assistant"), HALT_MARK)
    assert halts, f"the halt explanation never reached the transcript: {result['messages']!r}"
    assert all("_kanban_stop_synthetic" not in m for m in halts), (
        "the halt explanation is real content: it must survive into the durable transcript"
    )
    # Answer row first, nudge after it — the same role-alternation contract
    # ``apply_stop_gates`` keeps, so the model never sees a nudge answering nothing.
    first_halt = next(i for i, m in enumerate(result["messages"]) if m in halts)
    first_nudge = next(i for i, m in enumerate(result["messages"]) if m in nudges)
    assert first_halt < first_nudge, (
        f"halt row {first_halt} must precede the nudge row {first_nudge}: "
        f"{[m.get('role') for m in result['messages']]!r}"
    )
    assert result["turn_exit_reason"].startswith("text_response"), (
        f"the turn should have gone on to hand the card off, ended on {result['turn_exit_reason']!r}"
    )
    assert result["final_response"] == "handed off"
    # It handed the card off, so this exit is honestly clean.
    from hermes_cli.turn_exit import turn_exit_code

    assert turn_exit_code(result, kanban_worker=True) == 0


def test_guardrail_halt_with_the_nudge_budget_spent_exits_nonzero(tmp_path, monkeypatch):
    """A worker that burns both nudges must fail the run, not violate the protocol.

    Red on the unpatched branch, which breaks at the first halt and exits rc=0 with the card
    `running` — the dispatcher books that as ``worker exited cleanly (rc=0) without
    kanban_complete, kanban_block or kanban_request_review``.
    """
    agent = _make_agent(tmp_path, monkeypatch, "web_search", kanban_task=TASK)
    result = _run(agent, [
        _calling("c1"), _calling("c2"), _calling("c3"),   # halt 1 -> nudge 1
        _calling("c4"),                                   # blocked replay -> halt 2 -> nudge 2
        _calling("c5"),                                   # blocked replay -> halt 3, budget spent
    ])

    from agent.turn_failure_copy import exit_reason_failure
    from hermes_cli.turn_exit import turn_exit_code

    assert result["turn_exit_reason"] == "guardrail_halt_unhanded", (
        f"a terminal halt must name itself; got {result['turn_exit_reason']!r}"
    )
    verdict = exit_reason_failure(result["turn_exit_reason"])
    assert verdict is not None and verdict.fails_turn is True, (
        "the minted reason carries no failure verdict, so the finalizer would leave the turn "
        "unfailed"
    )
    assert result["failed"] is True and result["completed"] is False
    assert TASK in str(result["final_response"]), (
        "the terminal halt must name the card it is abandoning: "
        f"{result['final_response']!r}"
    )
    assert turn_exit_code(result, kanban_worker=True) == 1, (
        "a terminal halt on a board worker must not be booked as a clean exit"
    )


def test_guardrail_halt_on_an_attended_session_still_breaks_clean(tmp_path, monkeypatch):
    """No board task, no behaviour change: the human can answer ``continue``.

    Green on both legs by design — this is the blast-radius pin, the reason the seam is
    gated on owned-task identity rather than on the halt itself.
    """
    agent = _make_agent(tmp_path, monkeypatch, "web_search", kanban_task=None)
    result = _run(agent, [_calling("c1"), _calling("c2"), _calling("c3"), _saying("unused")])

    from hermes_cli.turn_exit import turn_exit_code

    assert result["turn_exit_reason"] == "guardrail_halt"
    assert not _matching(_rows(result, "user"), NUDGE_MARK)
    assert result["failed"] is not True
    assert HALT_MARK in str(result["final_response"])
    assert turn_exit_code(result, kanban_worker=False) == 0


def test_the_unhanded_predicate_follows_board_identity(tmp_path, monkeypatch):
    """The seam fires on an un-handed-off board task and nowhere else.

    Red on the unpatched tree, which has no such helper (``turn_stop_gates`` only asks the
    nudge builder, whose ``None`` cannot tell "attended" from "budget spent").
    """
    try:
        from agent.turn_stop_gates import unhanded_kanban_task
    except ImportError:  # pragma: no cover - the red leg
        pytest.fail("agent.turn_stop_gates.unhanded_kanban_task is missing: the halt branch "
                    "cannot tell a terminal board-worker halt from an attended stop")

    def called(name: str) -> list[dict]:
        return [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": name, "arguments": "{}"}}]}]

    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    assert unhanded_kanban_task([]) == TASK
    assert unhanded_kanban_task([{"role": "user", "content": "go"}]) == TASK
    assert unhanded_kanban_task(called("kanban_request_review")) == "", (
        "a worker that already handed the card off must not be failed by a later halt"
    )
    assert unhanded_kanban_task(called("web_search")) == TASK
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert unhanded_kanban_task([]) == "", "attended sessions keep the interactive halt copy"
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    # An in-process delegate child inherits the env but owns no card: its halt stays
    # interactive copy, exactly as before this seam existed.
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    assert unhanded_kanban_task([]) == ""
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")
    monkeypatch.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert unhanded_kanban_task([]) == "", "an operator who disabled the guard keeps rc=0"
