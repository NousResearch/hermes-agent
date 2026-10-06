"""Acceptance: non-goals are absent.

The cognition plugin must NOT ship a swarm board, and Dream must have no
autonomous write-back path: the read-only digest tool shares no code with any
enqueue/commit call, and the tool surface exposes exactly the three read and
status tools (search, status, dream) — no writer.
"""

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import COG_DIR, load  # noqa: E402

provider_mod = load("provider")


class NonGoalsTest(unittest.TestCase):
    def test_no_swarm_board_anywhere_in_plugin(self):
        for root, dirnames, files in os.walk(COG_DIR):
            # Acceptance tests describe the non-goals in their own prose;
            # only shipped plugin surface counts.
            dirnames[:] = [d for d in dirnames if d != "tests"]
            for filename in files:
                if filename.endswith((".py", ".yaml", ".md")):
                    with open(os.path.join(root, filename), encoding="utf-8") as handle:
                        body = handle.read().lower()
                    self.assertNotIn("swarm", body, filename)
                    self.assertNotIn("board", body, filename)

    def test_dream_has_no_write_back_path(self):
        source = inspect.getsource(provider_mod.CognitionMemoryProvider._tool_dream)
        for writer in ("enqueue", "fetch_first_push", "_apply_locked", "ack(", "fail("):
            self.assertNotIn(writer, source)
        # The tool surface is exactly search + status + dream: no writer tool.
        provider = provider_mod.CognitionMemoryProvider()
        names = sorted(s["name"] for s in provider.get_tool_schemas())
        self.assertEqual(names,
                         ["cognition_dream", "cognition_search", "cognition_status"])

    def test_no_autonomous_write_hooks(self):
        source = inspect.getsource(provider_mod.CognitionMemoryProvider)
        self.assertNotIn("autonomous", source.lower())
        # Session-end is the only writer; per-turn sync persists nothing.
        self.assertIn("def on_session_end", source)
        sync_source = inspect.getsource(
            provider_mod.CognitionMemoryProvider.sync_turn)
        self.assertNotIn("enqueue", sync_source)


if __name__ == "__main__":
    unittest.main()
