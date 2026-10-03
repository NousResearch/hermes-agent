"""Provider-minted parallel tool-call ids are normalized once, at mint time (#130363)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _assistant_tool_call_dict
from run_agent import AIAgent


def _tool_call(raw_id, n, **extra):
    return SimpleNamespace(
        id=raw_id, type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path": "%s"}' % n), **extra,
    )


def _agent():
    # Real AIAgent methods (duplicate repair, assistant serialization), no init side effects.
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "nous", "m", "s1", []
    agent.valid_tool_names = {"read_file"}
    agent.log_prefix = ""
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    return agent


def _mint(tool_calls):
    """Run the real mint path and return the rows that would be stored and replayed."""
    from agent.turn_tool_validation import validate_tool_calls

    agent = _agent()
    verdict = validate_tool_calls(
        agent, SimpleNamespace(content="", tool_calls=tool_calls), "tool_calls",
        messages=[], conversation_history=[], api_call_count=1, effective_task_id="t1",
    )
    assert verdict.action == "ok"
    stored = [_assistant_tool_call_dict(agent, tc, i) for i, tc in enumerate(tool_calls)]
    # Tool results take their id from the same object, so call and result agree.
    assert [row["id"] for row in stored] == [AIAgent._get_tool_call_id_static(tc) for tc in tool_calls]
    return stored


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *a, **k: None)


def test_parallel_provider_ids_are_rewritten_stably():
    raw = ["chatcmpl-tool-aaa", "chatcmpl-tool-bbb"]
    ids = [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])]

    assert all(i.startswith("call_") for i in ids)
    assert len(set(ids)) == 2
    # Byte-stable across a re-mint of the same turn.
    assert [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])] == ids


@pytest.mark.parametrize("raw", [
    ["chatcmpl-tool-aaa"],
    ["chatcmpl-tool-aaa", "call_1"],
    ["call_0", "call_1"],
])
def test_single_mixed_and_ordinary_batches_are_untouched(raw):
    assert [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])] == raw


@pytest.mark.parametrize("keys", [("a", "b"), ("a", "a")], ids=["distinct", "duplicate"])
def test_response_item_half_survives_when_only_id_is_composite(keys):
    calls = [
        _tool_call(f"chatcmpl-tool-{k}|fc_original_{n}", n, call_id=f"chatcmpl-tool-{k}")
        for n, k in enumerate(keys)
    ]
    stored = _mint(calls)

    assert [row["response_item_id"] for row in stored] == ["fc_original_0", "fc_original_1"]
    assert all(row["id"].startswith("call_") for row in stored)
    assert len({row["id"] for row in stored}) == 2


def _padded(calls):
    calls[0].id = " chatcmpl-tool-a"


def _blank_call_id(calls):
    calls[0].call_id = " "


def _mixed_until_dedup(calls):
    # Mixed at validation; dedup then drops call_c, leaving an all-provider batch.
    calls.append(_tool_call("call_c", "a"))


@pytest.mark.parametrize("shape", [None, _padded, _blank_call_id, _mixed_until_dedup],
                         ids=["plain", "padded", "blank_call_id", "mixed_until_dedup"])
def test_final_staged_batch_never_reaches_wire_all_provider_prefixed(shape, monkeypatch):
    import agent.turn_tool_round as round_module
    from agent.transports.chat_completions import ChatCompletionsTransport

    class _Staged(Exception):
        pass

    staged = []

    def _capture(agent, *, assistant_message, **kwargs):
        staged.extend(assistant_message.tool_calls)
        raise _Staged

    monkeypatch.setattr(round_module, "stage_tool_call_message", _capture)
    agent = _agent()
    agent.quiet_mode, agent.verbose_logging = True, False
    calls = [_tool_call("chatcmpl-tool-a", "a"), _tool_call("chatcmpl-tool-b", "b")]
    if shape:
        shape(calls)
    with pytest.raises(_Staged):
        round_module.run_tool_round(
            agent, assistant_message=SimpleNamespace(content="", tool_calls=calls), finish_reason="tool_calls",
            messages=[], conversation_history=[], api_call_count=1, effective_task_id="t", user_message="read",
            system_message="", active_system_prompt="", compression_attempts=0, max_compression_attempts=1,
            final_response=None, failed=False, _turn_exit_reason=None, truncated_tool_call_retries=0,
            current_turn_user_idx=0,
        )
    rows = [_assistant_tool_call_dict(agent, tc, i) for i, tc in enumerate(staged)]
    wire = ChatCompletionsTransport().convert_messages([
        {"role": "assistant", "content": "", "tool_calls": rows},
        *[{"role": "tool", "tool_call_id": AIAgent._get_tool_call_id_static(tc), "content": "ok"} for tc in staged],
    ])
    ids = [c["id"] for c in wire[0]["tool_calls"]]

    assert len(ids) == 2
    assert ids == [m["tool_call_id"] for m in wire[1:]]
    assert not any(i.startswith("chatcmpl-tool-") for i in ids), ids


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_real_transport_tool_calls_normalize_without_crashing(api):
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport
    from agent.turn_tool_validation import validate_tool_calls

    if api == "responses":
        # Production ToolCall exposes call_id as a read-only view of provider_data.
        response = SimpleNamespace(status="completed", output=[
            SimpleNamespace(type="function_call", id=f"fc_{n}", call_id=f"chatcmpl-tool-{n}",
                            name="read_file", arguments="{}") for n in range(2)])
        message = ResponsesApiTransport().normalize_response(response)
    else:
        calls = [SimpleNamespace(id=f"chatcmpl-tool-{n}", function=SimpleNamespace(name="read_file", arguments="{}"))
                 for n in range(2)]
        message = ChatCompletionsTransport().normalize_response(SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls), finish_reason="tool_calls")]))
    agent = _agent()
    verdict = validate_tool_calls(agent, message, "tool_calls", messages=[], conversation_history=[],
                                  api_call_count=1, effective_task_id="t")
    assert verdict.action == "ok"
    rows = [_assistant_tool_call_dict(agent, tc, i) for i, tc in enumerate(message.tool_calls)]

    assert [r["id"] for r in rows] == [AIAgent._get_tool_call_id_static(tc) for tc in message.tool_calls]
    assert all(r["id"].startswith("call_") for r in rows) and len({r["id"] for r in rows}) == 2
    if api == "responses":
        assert [r["response_item_id"] for r in rows] == ["fc_0", "fc_1"]
        assert [tc.call_id for tc in message.tool_calls] == [r["id"] for r in rows]


def test_unencodable_provider_ids_do_not_crash_and_stay_distinct():
    raw = ["chatcmpl-tool-\ud800", "chatcmpl-tool-?"]
    ids = [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])]

    assert all(i.startswith("call_") for i in ids)
    assert len(set(ids)) == 2
