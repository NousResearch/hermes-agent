import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tui_gateway import server


class BlockLogicGatewayIntegrationTests(unittest.TestCase):
    def _agent(self, *, mode="observe_only", callback=None, log_path=None):
        return types.SimpleNamespace(
            model="gpt-5.6-sol",
            session_total_tokens=82_000,
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=82_000,
                context_length=100_000,
                compression_count=0,
            ),
            _block_logic_threshold_config={
                "enabled": True,
                "mode": mode,
                "log_path": str(log_path) if log_path else "",
            },
            notice_callback=callback,
        )

    @staticmethod
    def _session():
        return {
            "created_at": time.time() - 18 * 3600,
            "history": [{"role": "user", "content": "x"}] * 130,
            "block_logic_metrics": {
                "decision_count": 16,
                "changed_files": 11,
                "model_handoffs": 2,
            },
        }

    def test_observe_only_adds_decision_logs_transition_and_emits_no_notice(self):
        notices = []
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "decisions.jsonl"
            agent = self._agent(callback=notices.append, log_path=log_path)
            session = self._session()

            first = server._get_usage(agent, session)
            second = server._get_usage(agent, session)

            self.assertEqual(first["block_logic"]["level"], "block_logic_recommended")
            self.assertEqual(first["block_logic"]["mode"], "observe_only")
            self.assertFalse(first["block_logic"]["canonical_write_allowed"])
            self.assertEqual(notices, [])
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["level"], "block_logic_recommended")
            self.assertEqual(second["block_logic"], first["block_logic"])

    def test_prompt_mode_emits_one_notice_per_level_transition(self):
        notices = []
        agent = self._agent(mode="prompt", callback=notices.append)
        session = self._session()

        server._get_usage(agent, session)
        server._get_usage(agent, session)

        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].key, "block_logic.threshold")
        self.assertEqual(notices[0].level, "warn")

    def test_live_config_is_loaded_once_when_agent_has_no_cached_config(self):
        agent = self._agent()
        del agent._block_logic_threshold_config
        session = self._session()
        config = {
            "block_logic_threshold": {
                "enabled": True,
                "mode": "observe_only",
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config) as load:
            usage = server._get_usage(agent, session)
            server._get_usage(agent, session)

        self.assertEqual(usage["block_logic"]["level"], "block_logic_recommended")
        self.assertEqual(load.call_count, 1)
        self.assertEqual(agent._block_logic_threshold_config["mode"], "observe_only")

    def test_live_git_and_history_metrics_enable_power_sprint_override(self):
        agent = self._agent()
        agent.context_compressor.last_prompt_tokens = 78_000
        session = {
            "created_at": time.time() - 36 * 3600,
            "cwd": "C:/repo",
            "history": [
                {"role": "user", "content": "ordinary work"}
                for _ in range(148)
            ]
            + [
                {
                    "role": "user",
                    "content": "[System: The active model for this chat has changed to model-a.]",
                },
                {
                    "role": "user",
                    "content": "[System: The active model for this chat has changed to model-b.]",
                },
            ],
        }
        changed = "\n".join(f" M file-{index}.py" for index in range(10))

        with patch.object(server, "_git", return_value=changed) as git:
            usage = server._get_usage(agent, session)
            server._get_usage(agent, session)

        self.assertEqual(usage["block_logic"]["level"], "block_logic_recommended")
        self.assertIn(
            "power-sprint continuity override triggered",
            usage["block_logic"]["reasons"],
        )
        self.assertIn("model handoffs total 2", usage["block_logic"]["reasons"])
        self.assertEqual(git.call_count, 1)

    def test_unknown_post_compression_context_emits_no_decision_or_receipt(self):
        notices = []
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "decisions.jsonl"
            agent = self._agent(callback=notices.append, log_path=log_path)
            agent.context_compressor.last_prompt_tokens = -1
            session = self._session()

            usage = server._get_usage(agent, session)

            self.assertNotIn("context_used", usage)
            self.assertNotIn("block_logic", usage)
            self.assertFalse(log_path.exists())
            self.assertEqual(notices, [])

    def test_existing_usage_fields_are_preserved_when_decision_is_attached(self):
        agent = self._agent()
        session = self._session()

        usage = server._get_usage(agent, session)

        self.assertEqual(usage["context_used"], 82_000)
        self.assertEqual(usage["context_max"], 100_000)
        self.assertEqual(usage["context_percent"], 82)
        self.assertEqual(usage["compressions"], 0)
        self.assertIn("block_logic", usage)


if __name__ == "__main__":
    unittest.main()
