"""A call dropped by the turn guardrails (dedup, delegate cap) must leave the native replay
sidecars too: Converse replays ``bedrock_content_blocks`` / ``anthropic_content_blocks`` ahead of
``tool_calls`` and rejects a ``toolUse`` with no ``toolResult``."""

from __future__ import annotations

import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from run_agent import AIAgent

_SIGNED = {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-1"}}}


class _HistoryAgent:
    """Just what ``build_assistant_message`` reads."""
    verbose_logging = False
    reasoning_callback = stream_delta_callback = _stream_callback = None
    _extract_reasoning = staticmethod(lambda message: getattr(message, "reasoning_content", None))
    _strip_think_blocks = staticmethod(lambda text: text)
    _needs_thinking_reasoning_pad = staticmethod(lambda: False)
    _split_responses_tool_id = staticmethod(lambda value: (value, None))
    _deterministic_call_id = staticmethod(lambda name, args, index: f"call-{index}")
    _derive_responses_function_call_id = staticmethod(lambda value, response_item_id=None: value)


def _agent():
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "bedrock", "m", "s1", []
    agent.valid_tool_names = {"read_file", "delegate_task"}
    agent.log_prefix = ""
    agent.quiet_mode, agent.verbose_logging = True, False
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    return agent


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *a, **k: None)


def _converse(*blocks):
    """A real Converse response through the production Bedrock transport."""
    from agent.transports.bedrock import BedrockTransport

    return BedrockTransport().normalize_response({
        "output": {"message": {"role": "assistant", "content": list(blocks)}}, "stopReason": "tool_use"})


def _tool_use(tool_id, name="read_file", **args):
    return {"toolUse": {"toolUseId": tool_id, "name": name, "input": args}}


def _run_round(monkeypatch, message):
    """Run the real tool round up to staging; return the staged history row."""
    import agent.turn_tool_round as round_module
    from agent.chat_completion_helpers import build_assistant_message

    class _Staged(Exception):
        pass

    staged = {}

    def _capture(agent, *, assistant_message, finish_reason, **kwargs):
        staged["row"] = build_assistant_message(_HistoryAgent(), assistant_message, finish_reason)
        raise _Staged

    sys.modules.setdefault("requests", ModuleType("requests"))
    monkeypatch.setattr(round_module, "stage_tool_call_message", _capture)
    with pytest.raises(_Staged):
        round_module.run_tool_round(
            _agent(), assistant_message=message, finish_reason="tool_calls",
            messages=[], conversation_history=[], api_call_count=1, effective_task_id="t", user_message="go",
            system_message="", active_system_prompt="", compression_attempts=0, max_compression_attempts=1,
            final_response=None, failed=False, _turn_exit_reason=None, truncated_tool_call_retries=0,
            current_turn_user_idx=0,
        )
    return staged["row"]


def _assert_every_tool_use_answered(row):
    """Build the next Converse request the way the agent would: one result per kept call."""
    from agent.bedrock_adapter import convert_messages_to_converse

    results = [{"role": "tool", "tool_call_id": tc["id"], "content": "ok"} for tc in row["tool_calls"]]
    _system, wire = convert_messages_to_converse([{"role": "user", "content": "go"}, row, *results])
    uses = [b["toolUse"]["toolUseId"] for b in wire[1]["content"] if "toolUse" in b]
    answered = [b["toolResult"]["toolUseId"] for b in wire[2]["content"] if "toolResult" in b]
    assert uses == answered == [tc["id"] for tc in row["tool_calls"]]
    return wire[1]["content"]


def test_duplicate_call_in_parallel_batch_leaves_bedrock_sidecar(monkeypatch):
    message = _converse(_SIGNED, {"text": "Reading both."}, _tool_use("tu_a", path="a"),
                        _tool_use("tu_dup", path="a"), _tool_use("tu_b", path="b"))
    captured = copy.deepcopy(message.provider_data["bedrock_content_blocks"])

    row = _run_round(monkeypatch, message)

    assert [tc["id"] for tc in row["tool_calls"]] == ["tu_a", "tu_b"]
    blocks = _assert_every_tool_use_answered(row)  # the Converse ValidationException shape
    # Exactly the dropped call's block goes; reasoning/text/order survive byte-for-byte.
    assert row["bedrock_content_blocks"] == [captured[0], captured[1], captured[2], captured[4]]
    assert blocks[0] == {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-1"}}}


def test_capped_delegate_calls_leave_bedrock_sidecar(monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 1)
    message = _converse(_SIGNED, _tool_use("tu_d1", "delegate_task", goal="x"),
                        _tool_use("tu_r", path="r"), _tool_use("tu_d2", "delegate_task", goal="y"))

    row = _run_round(monkeypatch, message)

    assert [tc["id"] for tc in row["tool_calls"]] == ["tu_d1", "tu_r"]
    assert [next(iter(b)) for b in row["bedrock_content_blocks"]] == ["reasoningContent", "toolUse", "toolUse"]
    _assert_every_tool_use_answered(row)


def test_no_drop_leaves_bedrock_sidecar_identical(monkeypatch):
    message = _converse(_SIGNED, _tool_use("tu_a", path="a"), _tool_use("tu_b", path="b"))
    sidecar = message.provider_data["bedrock_content_blocks"]
    before = copy.deepcopy(sidecar)

    row = _run_round(monkeypatch, message)

    assert message.provider_data["bedrock_content_blocks"] is sidecar
    assert row["bedrock_content_blocks"] == before
    _assert_every_tool_use_answered(row)


def test_mixed_invalid_batch_keeps_the_invalid_calls_block(monkeypatch):
    # The invalid call stays in tool_calls and gets an error result, so its block must stay too.
    message = _converse(_SIGNED, _tool_use("tu_a", path="a"), _tool_use("tu_bad", "no_such_tool"),
                        _tool_use("tu_dup", path="a"))

    row = _run_round(monkeypatch, message)

    assert [tc["id"] for tc in row["tool_calls"]] == ["tu_a", "tu_bad"]
    assert [b["toolUse"]["toolUseId"] for b in row["bedrock_content_blocks"] if "toolUse" in b] == ["tu_a", "tu_bad"]
    _assert_every_tool_use_answered(row)


def test_duplicate_call_leaves_anthropic_sidecar(monkeypatch):
    from agent.transports.anthropic import AnthropicTransport

    def _block(**fields):
        return SimpleNamespace(model_dump=lambda **_: dict(fields), **fields)

    response = SimpleNamespace(stop_reason="tool_use", content=[
        _block(type="thinking", thinking="plan", signature="sig-1"),
        _block(type="tool_use", id="toolu_a", name="read_file", input={"path": "a"}),
        _block(type="tool_use", id="toolu_dup", name="read_file", input={"path": "a"}),
    ])
    message = AnthropicTransport().normalize_response(response)
    assert len(message.provider_data["anthropic_content_blocks"]) == 3

    row = _run_round(monkeypatch, message)

    assert [tc["id"] for tc in row["tool_calls"]] == ["toolu_a"]
    assert [(b["type"], b.get("id")) for b in row["anthropic_content_blocks"]] == [
        ("thinking", None), ("tool_use", "toolu_a")]
    assert row["anthropic_content_blocks"][0]["signature"] == "sig-1"
    # The AnthropicBedrock -> Converse fallback replays this sidecar for Claude.
    blocks = _assert_every_tool_use_answered(row)
    assert json.dumps(blocks[0]["reasoningContent"]["reasoningText"]["signature"]) == '"sig-1"'
