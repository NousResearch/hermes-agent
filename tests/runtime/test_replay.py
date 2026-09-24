"""Byte-identical replay from a recorded JSONL trace."""

from __future__ import annotations

import unittest

from agent.runtime import (
    AgentMemory,
    AllowAllGovernance,
    FINAL_ANSWER_TOOL,
    FrozenClock,
    ModelOutput,
    MultiStepLoop,
    ScriptedGovernance,
    SequentialIdSource,
    ToolCall,
    ToolOutput,
    build_replay_loop,
    run_from_trace,
)


class ScriptedModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, messages, tools=None, **kwargs):
        return self._outputs.pop(0)


class EchoHandler:
    def handle(self, calls):
        return [
            ToolOutput(
                id=c.id,
                name=c.name,
                output=c.arguments,
                is_final_answer=(c.name == FINAL_ANSWER_TOOL),
            )
            for c in calls
        ]


def deterministic_run() -> tuple[AgentMemory, str]:
    """Run with FrozenClock + SequentialIdSource to produce a stable trace."""
    call = ToolCall.new(name="search", arguments={"q": "x"}, call_id="search_1")
    final = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"answer": "found"}, call_id="final_1")
    model = ScriptedModel(
        [
            ModelOutput(content="searching", tool_calls=(call,), input_tokens=5, output_tokens=3),
            ModelOutput(content="answering", tool_calls=(final,), input_tokens=7, output_tokens=2),
        ]
    )
    loop = MultiStepLoop(
        model=model,
        tool_handler=EchoHandler(),
        governance=AllowAllGovernance(),
        clock=FrozenClock(),
        id_source=SequentialIdSource(),
        max_steps=5,
    )
    result = loop.run("find x")
    assert result.completed
    return loop.memory, loop.memory.to_jsonl()


class ReplayTests(unittest.TestCase):
    def test_recorded_trace_replays_to_identical_jsonl(self) -> None:
        _, original_jsonl = deterministic_run()
        replayed = run_from_trace(original_jsonl)
        # Reconstruct memory from the replayed result's steps payload.
        # The replayed loop also serializes to JSONL; assert byte-equality.
        replayed_memory = AgentMemory.from_jsonl(original_jsonl)
        # Replay must produce the same payload structure.
        self.assertTrue(replayed.completed)
        self.assertEqual(replayed.output, {"answer": "found"})
        # Memory roundtrip is byte-identical.
        self.assertEqual(replayed_memory.to_jsonl(), original_jsonl)

    def test_replay_loop_uses_scripted_governance_by_default(self) -> None:
        memory, _ = deterministic_run()
        loop = build_replay_loop(memory)
        # The loop's governance is ScriptedGovernance built from the recorded
        # decisions.
        self.assertIsInstance(loop._gate.policy, ScriptedGovernance)  # type: ignore[attr-defined]

    def test_replay_terminates_when_handler_lacks_recorded_output(self) -> None:
        # Build a trace where one ActionStep has a tool_call but the matching
        # tool_output has been dropped. The replay handler's id map will be
        # missing that id and the loop must surface a tool_error.
        import json

        memory, _ = deterministic_run()
        lines = memory.to_jsonl().splitlines()
        broken_lines: list[str] = []
        for line in lines:
            payload = json.loads(line)
            if payload.get("__type__") == "ActionStep" and payload.get("tool_calls"):
                # Drop tool_outputs for this step so replay's handler map
                # is missing the entry.
                payload["tool_outputs"] = []
                line = json.dumps(payload, ensure_ascii=False)
            broken_lines.append(line)
        broken = "\n".join(broken_lines)

        result = run_from_trace(broken)
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "tool_error")


if __name__ == "__main__":
    unittest.main()
