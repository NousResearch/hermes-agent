"""Keep the saved /plan file discoverable when its writing turn is compacted."""

import copy
import json
from itertools import pairwise

import pytest

from agent.context_compressor import is_compaction_summary_message
from agent.conversation_compression import _is_real_user_message, _message_text
from agent.plan_prompt import build_plan_prompt
from hermes_state import SessionDB
from tests.agent.test_compression_rotation_state import _build_agent_with_db
from tests.agent.test_prepared_compaction import _compressor, _messages, _prepare


PATH = ".hermes/plans/2026-10-26_120000-task.md"
HEADER = "[Active plan from /plan: "


def _plan_turn(path=PATH, *, bridged=False):
    args = {"path": path, "content": "# Implementation plan"}
    name = "write_file"
    if bridged:
        name, args = "tool_call", {"calls": [{"name": name, "arguments": args}]}
    return [
        {"role": "user", "content": build_plan_prompt("implement the task")},
        {"role": "assistant", "tool_calls": [{
            "id": "plan-write", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]},
        {"role": "tool", "tool_call_id": "plan-write", "content": "written"},
        {"role": "assistant", "content": "Plan saved."},
    ]


@pytest.fixture
def compact(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("plan-session", source="cli")
    agent = _build_agent_with_db(db, "plan-session", platform="cli")

    def run(messages, prepared=False):
        cc = _compressor(enabled=prepared, session_id=agent.session_id)
        agent.context_compressor = cc
        if prepared:
            entry = _prepare(cc, messages)
            assert entry.compress_end > 8  # the plan writing turn is summarized
        result, _ = agent._compress_context(messages, "sys", approx_tokens=21_000)
        assert result != messages, "must exercise an admitted compaction"
        assert [call["worker"] for call in cc.calls] == [prepared]
        return result

    yield agent, run
    db.close()


def _assert_pointer(messages, path):
    text = "\n".join(_message_text(row) for row in messages)
    assert text.count(HEADER) == 1
    pointer = f"{HEADER}{path}. Re-read it before continuing the planned work.]"
    assert pointer in text
    carrier = next(row for row in messages if pointer in _message_text(row))
    assert is_compaction_summary_message(carrier)
    assert not _is_real_user_message(carrier)
    assert not any(a["role"] == b["role"] == "user" for a, b in pairwise(messages))


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("bridged", [False, True])
def test_active_plan_survives_repeated_compaction_and_newer_plan_replaces_it(compact, prepared, bridged):
    agent, compress = compact
    agent._todo_store.write([{"id": "t1", "content": "implement the task", "status": "pending"}])
    history = _messages(30)
    history[5:5] = _plan_turn(bridged=bridged)
    first = compress(copy.deepcopy(history), prepared)
    _assert_pointer(first, PATH)

    # A later ordinary write must not become the active plan. Preserve the pointer
    # from the first handoff even when a new summarizer omits it entirely.
    reference = _plan_turn(".hermes/plans/reference.md")[1:]
    second = compress(first + _messages(30)[1:] + reference, prepared)
    _assert_pointer(second, PATH)

    newer = ".hermes/plans/2026-10-26_130000-next.md"
    third = compress(second + _plan_turn(newer, bridged=bridged) + _messages(30)[1:], prepared)
    _assert_pointer(third, newer)
    assert PATH not in "\n".join(_message_text(row) for row in third)


@pytest.mark.parametrize("scenario", ["no_prompt", "outside_plans", "no_write"])
def test_without_an_active_plan_compaction_adds_no_pointer(compact, scenario):
    _, compress = compact
    turn = _plan_turn("notes/plan.md" if scenario == "outside_plans" else PATH)
    if scenario == "no_prompt":
        turn = turn[1:]
    elif scenario == "no_write":
        turn = turn[:1]
    history = _messages(30)
    history[5:5] = turn
    result = compress(history)
    assert HEADER not in "\n".join(_message_text(row) for row in result)
