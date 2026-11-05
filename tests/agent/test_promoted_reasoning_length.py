"""Oversized promoted reasoning cannot silently complete delegated work (#134028)."""

import random
import string
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.message_sanitization import apply_reasoning_content_policy
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


def _reasoning(size):
    tail = "\nAll observations have been recorded."
    return "".join(random.Random(53).choices(string.ascii_letters + " ", k=size - len(tail))) + tail


def _response(content="", reasoning=None, source="sdk"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        setattr(msg, "reasoning_content" if source == "sdk" else "reasoning", reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           model="test/model", usage=None)


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("hermes_cli.plugins.discover_plugins"),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:8000/v1",
                        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", provider="vllm",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=10)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Keep the system prompt stable."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._stall_guards = True
    return agent


def _run(agent, responses):
    agent.client.chat.completions.create.side_effect = list(responses)
    with (patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
          patch.object(agent, "_cleanup_task_resources")):
        return agent.run_conversation("Report the result of the task.")


@pytest.mark.parametrize("source", ["sdk", "stream"])
@pytest.mark.parametrize("size,repeats,offered", [
    (3999, 1, True), (4000, 1, True), (14070, 1, True),
    (4000, 3, True), (4000, 1, False), (14070, 3, False),
])
def test_length_gate_is_bounded_and_elides_every_replay_carrier(loop_agent, caplog, source, size, repeats, offered):
    text = _reasoning(size)
    loop_agent.valid_tool_names = {"terminal"} if offered else set()
    responses = [_response(reasoning=text, source=source) for _ in range(repeats)]
    result = _run(loop_agent, responses + [_response(content="The task is complete.")])
    expected_calls = 1 if size < 4000 else min(repeats + 1, 3)
    assert result["api_calls"] == expected_calls
    exhausted = size >= 4000 and repeats == 3
    assert result["final_response"] == (text if size < 4000 or exhausted else "The task is complete.")
    if exhausted:
        assert result["turn_exit_reason"] == "text_response(reasoning_only_mangled)"
        assert any("not a final answer" in record.message for record in caplog.records)
    assistant_rows = [m for m in result["messages"] if m.get("role") == "assistant"]
    for row in assistant_rows[:-1]:
        assert not row.get("content")
        carriers = [row[key] for key in ("api_content", "reasoning", "reasoning_content")]
        assert all(isinstance(value, str) and len(value) <= 500 for value in carriers)
        assert carriers[0] == carriers[1] == carriers[2]
        wire = {"role": "assistant", "content": row["api_content"]}
        apply_reasoning_content_policy(row, wire, needs_thinking_pad=True)
        assert wire["reasoning_content"] == carriers[0]
        apply_reasoning_content_policy(row, wire, needs_thinking_pad=False)
        assert "reasoning_content" not in wire
    requests = [call.kwargs["messages"] for call in loop_agent.client.chat.completions.create.call_args_list]
    assert all(request[0]["content"] == requests[0][0]["content"] for request in requests)
    if size >= 4000:
        nudges = [m["content"] for m in result["messages"] if m.get("role") == "user" and "response text" in m.get("content", "")]
        assert len(nudges) == min(repeats, 2)


@pytest.mark.parametrize("mode", ["short", "recovered", "exhausted", "budget", "shared", "visible"])
def test_delegation_preserves_honest_completion_and_budget_verdicts(loop_agent, mode):
    loop_agent.valid_tool_names = {"terminal"}
    long = _reasoning(4000)
    responses = {
        "short": [_response(reasoning="The answer is 42.")],
        "recovered": [_response(reasoning=long), _response(content="The task is complete.")],
        "visible": [_response(content=long)],
        "shared": [_response(reasoning="Let me read the file."), _response(reasoning=long), _response(reasoning=long)],
        "exhausted": [_response(reasoning=long)] * 3,
        "budget": [_response(reasoning=long)] * 3,
    }[mode]
    if mode == "budget":
        loop_agent.max_iterations = 1
        responses += [_response(content="Budget-limited task summary.")] * 3
    result = _run(loop_agent, responses)
    entry = _build_result_entry(loop_agent, result, 0, 0.5, _SchemaOutcome(None, None, [], 0))
    if mode in {"exhausted", "shared"}:
        assert result["api_calls"] == 3
        assert entry["status"] == "failed" and entry["exit_reason"] == "mangled"
        assert long in entry["summary"]
        assert entry["summary"].startswith("[Unreliable reasoning-only output")
        assert not entry["truncated"]
    elif mode == "budget":
        assert not result["completed"]
        assert entry["exit_reason"] == "max_iterations" and entry["truncated"]
    else:
        assert entry["status"] == "completed" and entry["exit_reason"] == "completed"
        assert not entry["summary"].startswith("[Unreliable reasoning-only output")
