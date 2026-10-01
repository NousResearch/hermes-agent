"""AgentMemory contract."""

from __future__ import annotations

import unittest

from agent.runtime import ActionStep, AgentMemory, FinalAnswerStep, TaskStep, ToolCall, ToolOutput


class MemoryTests(unittest.TestCase):
    def test_append_and_iterate(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="t"))
        memory.append(ActionStep(step_number=1, model_output="thinking"))
        self.assertEqual(len(memory), 2)
        self.assertEqual([type(s).__name__ for s in memory], ["TaskStep", "ActionStep"])

    def test_append_rejects_non_memorystep(self) -> None:
        memory = AgentMemory()
        with self.assertRaises(TypeError):
            memory.append("not a step")  # type: ignore[arg-type]

    def test_steps_property_is_immutable_snapshot(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="t"))
        snapshot = memory.steps
        # snapshot is a tuple — cannot mutate
        with self.assertRaises(AttributeError):
            snapshot.append(TaskStep(step_number=99, task="x"))  # type: ignore[attr-defined]
        # Appending to memory does not change the prior snapshot
        memory.append(ActionStep(step_number=1))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(memory.steps), 2)

    def test_to_messages_projects_task_and_action(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="hello"))
        call = ToolCall.new(name="echo", arguments={"x": 1}, call_id="c1")
        out = ToolOutput(id="c1", name="echo", output={"x": 1})
        memory.append(
            ActionStep(
                step_number=1,
                model_output="I'll echo.",
                tool_calls=(call,),
                tool_outputs=(out,),
            )
        )
        messages = memory.to_messages()
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["user", "assistant", "assistant", "tool"])
        self.assertEqual(messages[0]["content"], "hello")
        self.assertEqual(messages[1]["content"], "I'll echo.")
        self.assertEqual(messages[2]["tool_calls"][0]["name"], "echo")
        self.assertEqual(messages[3]["tool_call_id"], "c1")

    def test_final_answer_lookup(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="t"))
        memory.append(FinalAnswerStep(step_number=1, output={"answer": 7}))
        self.assertEqual(memory.final_answer(), {"answer": 7})

    def test_final_answer_returns_none_when_absent(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="t"))
        self.assertIsNone(memory.final_answer())

    def test_jsonl_roundtrip(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="task"))
        memory.append(
            ActionStep(
                step_number=1,
                model_output="m",
                tool_calls=(ToolCall.new(name="t", arguments={"a": 1}, call_id="c1"),),
                tool_outputs=(ToolOutput(id="c1", name="t", output="ok"),),
                input_tokens=5,
                output_tokens=3,
            )
        )
        memory.append(FinalAnswerStep(step_number=2, output="done"))

        blob = memory.to_jsonl()
        restored = AgentMemory.from_jsonl(blob)
        self.assertEqual(len(restored), 3)
        action = restored.steps[1]
        self.assertIsInstance(action, ActionStep)
        self.assertEqual(action.tool_calls[0].name, "t")  # type: ignore[union-attr]
        self.assertEqual(action.tool_outputs[0].output, "ok")  # type: ignore[union-attr]
        self.assertEqual(restored.final_answer(), "done")

    def test_replay_includes_step_headers(self) -> None:
        memory = AgentMemory()
        memory.append(TaskStep(step_number=0, task="t"))
        memory.append(FinalAnswerStep(step_number=1, output="a"))
        replay = memory.replay()
        self.assertIn("Step   0 · TaskStep", replay)
        self.assertIn("Step   1 · FinalAnswerStep", replay)


if __name__ == "__main__":
    unittest.main()
