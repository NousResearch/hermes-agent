from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PROFILE_HOME = Path(
    os.environ.get(
        "HERMES_PROFILE_HOME",
        "/Users/subbota/HERMES_RUNTIMES/main-gateway-v2/runtime-home/profiles/main-gateway-v2",
    )
)
PROD_SCRIPT = PROFILE_HOME / "scripts/auth_expired_401_watchdog.py"


class AuthWatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.profile = Path(self.tmp.name)
        self.alerts = self.profile / "alerts"
        self.scripts = self.profile / "scripts"
        self.alerts.mkdir(mode=0o700)
        self.scripts.mkdir(mode=0o700)
        self.script = self.scripts / "auth_expired_401_watchdog.py"
        shutil.copy2(PROD_SCRIPT, self.script)
        os.chmod(self.script, 0o700)
        self.log = self.alerts / "auth_watchdog.log"
        self.state = self.alerts / "auth_watchdog_state.json"
        self.log.write_text("", encoding="utf-8")
        self.state.write_text('{"schema_version":1,"processed_event_ids":[],"updated_at":null}\n', encoding="utf-8")
        os.chmod(self.log, 0o600)
        os.chmod(self.state, 0o600)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_script(self):
        return subprocess.run(
            [sys.executable, str(self.script)],
            cwd=str(REPO),
            text=True,
            capture_output=True,
            timeout=10,
        )

    def append_event(self, event: dict) -> None:
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def valid_event(self, event_id: str = "evt-1", **extra):
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "event_type": "auth_expired_401",
            "occurred_at": "2026-07-11T00:00:00Z",
        }
        event.update(extra)
        return event

    def test_no_event_no_output(self):
        proc = self.run_script()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_one_valid_event_one_fixed_output(self):
        self.append_event(self.valid_event("evt-abc"))
        proc = self.run_script()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(
            proc.stdout.strip(),
            "⚠️ Hermes: auth_expired_401 — перевір авторизацію провайдера. Event: evt-abc",
        )

    def test_duplicate_event_no_second_output(self):
        self.append_event(self.valid_event("evt-dup"))
        first = self.run_script()
        second = self.run_script()
        self.assertIn("evt-dup", first.stdout)
        self.assertEqual(second.stdout, "")

    def test_unrelated_event_type_ignored(self):
        self.append_event(self.valid_event("evt-other", event_type="other"))
        proc = self.run_script()
        self.assertEqual(proc.stdout, "")

    def test_malformed_json_fails_closed(self):
        self.log.write_text("{not json\n", encoding="utf-8")
        proc = self.run_script()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("auth_watchdog_error: JSONDecodeError", proc.stderr)

    def test_malformed_state_fails_closed(self):
        self.state.write_text("{bad state\n", encoding="utf-8")
        self.append_event(self.valid_event("evt-state"))
        proc = self.run_script()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("auth_watchdog_error: JSONDecodeError", proc.stderr)

    def test_state_update_atomic_and_records_event(self):
        self.append_event(self.valid_event("evt-atomic"))
        proc = self.run_script()
        self.assertIn("evt-atomic", proc.stdout)
        data = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertIn("evt-atomic", data["processed_event_ids"])
        leftovers = list(self.alerts.glob(".auth_watchdog_state.*.tmp"))
        self.assertEqual(leftovers, [])

    def test_no_raw_event_private_content_in_output(self):
        self.append_event(self.valid_event("evt-safe", private_text="SECRET_PRIVATE_CONTENT"))
        proc = self.run_script()
        self.assertIn("evt-safe", proc.stdout)
        self.assertNotIn("SECRET_PRIVATE_CONTENT", proc.stdout)
        self.assertNotIn("private_text", proc.stdout)

    def test_script_reads_only_profile_paths_statically(self):
        text = PROD_SCRIPT.read_text(encoding="utf-8")
        forbidden = ["/Users/subbota/.hermes/logs", "~/.hermes/logs", "gateway.log", "TELEGRAM_BOT_TOKEN"]
        for snippet in forbidden:
            self.assertNotIn(snippet, text)
        self.assertIn("Path(__file__).resolve().parents[1]", text)

    def test_no_network_or_llm_calls_statically(self):
        text = PROD_SCRIPT.read_text(encoding="utf-8")
        forbidden = ["requests", "httpx", "telegram", "openai", "anthropic", "subprocess", "socket"]
        for snippet in forbidden:
            self.assertNotIn(snippet, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
