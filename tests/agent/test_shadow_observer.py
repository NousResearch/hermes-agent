from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent.shadow_observer import observe_completed_response


class ShadowObserverTests(unittest.TestCase):
    def test_disabled_is_noop(self):
        result = {"completed": True, "final_response": "private answer", "api_calls": 1}
        observation = observe_completed_response(result, {})
        self.assertEqual(observation["status"], "disabled")
        self.assertFalse(observation["invoked"])

    def test_enabled_invokes_adapter_and_keeps_result_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.jsonl"
            command = [
                sys.executable,
                "-c",
                (
                    "import json,sys; json.load(sys.stdin); "
                    "open(sys.argv[-1], 'w', encoding='utf-8').write('{\\\"ok\\\":true}\\n'); "
                    "print(json.dumps({'status':'passed','evidence_persisted':True,"
                    "'delivery_attempted':False,'side_effect_status':'not_attempted'}))"
                ),
            ]
            result = {"completed": True, "final_response": "private answer", "api_calls": 2, "tool_calls": 1}
            snapshot = dict(result)
            observation = observe_completed_response(
                result,
                {"evaluation": {"shadow": {
                    "enabled": True,
                    "adapter_command": command,
                    "evidence_output": str(output),
                }}},
            )
            self.assertEqual(result, snapshot)
            self.assertEqual(observation["status"], "passed")
            self.assertTrue(observation["invoked"])
            self.assertTrue(observation["persisted"])
            self.assertFalse(observation["delivery_attempted"])
            self.assertTrue(output.exists())

    def test_adapter_failure_is_inconclusive_and_does_not_raise(self):
        result = {"completed": True, "final_response": "private answer"}
        observation = observe_completed_response(
            result,
            {"evaluation": {"shadow": {
                "enabled": True,
                "adapter_command": [sys.executable, "-c", "import sys; sys.exit(7)"],
                "evidence_output": "/tmp/shadow-observer-test.jsonl",
            }}},
        )
        self.assertEqual(observation["status"], "inconclusive")
        self.assertTrue(observation["invoked"])
        self.assertFalse(observation["persisted"])


if __name__ == "__main__":
    unittest.main()
