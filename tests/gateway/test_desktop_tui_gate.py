"""W1 cycle 1d-r2: the TUI-gateway turn gate (seam 4).

Empirical trace proved the Desktop app's chats run through the serve's
in-process ``tui_gateway`` (``tui turn finished`` log lines, agent
turns via ``agent.conversation_loop``), with zero admission rows in any
of 925 scanned databases - so neither the gateway submit seam, the CLI
chat seam, nor the API admission door sees them. ``tui_turn_gate``
screens the turn text at ``_run_prompt_submit`` (the ownership/liveness
choke point every TUI turn source crosses) BEFORE ``_admit_prompt_turn``.
"""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from gateway import desktop_screen


class EngineStubHandler(BaseHTTPRequestHandler):
    tier: str = "allow"
    block_pattern: str | None = None
    screen_calls = 0
    last_texts: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/screen"):
            self.send_error(404)
            return
        EngineStubHandler.screen_calls += 1
        text = str(body.get("text", ""))
        EngineStubHandler.last_texts.append(text)
        if not text.strip():
            self.send_error(400, "needs a non-empty 'text'")
            return
        tier = "block" if (EngineStubHandler.block_pattern
                           and EngineStubHandler.block_pattern in text) \
            else EngineStubHandler.tier
        reasons = ["jailbreak", "injection"] if tier == "block" else []
        payload = json.dumps({"tier": tier, "reasons": reasons,
                              "request_id": "req-stub-1",
                              "served_by": {"provider": "stub"},
                              "scores": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class _Agent:
    def __init__(self, session_id, db=None):
        self.session_id = session_id
        self._session_db = db


class TuiTurnGateTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        desktop_screen._DISABLED_LOGGED = False
        self.tmp = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmp, "tui-gate.log")
        self.state_db = os.path.join(self.tmp, "state.db")
        conn = sqlite3.connect(self.state_db)
        conn.execute("CREATE TABLE sessions "
                     "(id TEXT, source TEXT, created_source TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                         [("s-tui", "desktop", "desktop")])
        conn.commit()
        conn.close()
        self._path_patch = mock.patch.object(
            desktop_screen, "_state_db_path", lambda: Path(self.state_db))
        self._path_patch.start()
        self.addCleanup(self._path_patch.stop)
        self.env = {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT":
                f"http://127.0.0.1:{self.server.server_port}/v1/screen",
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": self.log_path,
        }
        self._saved = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self._desktop = os.environ.get("HERMES_DESKTOP")
        os.environ["HERMES_DESKTOP"] = "1"
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._desktop is None:
            os.environ.pop("HERMES_DESKTOP", None)
        else:
            os.environ["HERMES_DESKTOP"] = self._desktop
        self.server.shutdown()

    def session(self, session_id="s-tui", rows=None):
        if rows is not None:
            conn = sqlite3.connect(self.state_db)
            conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)", rows)
            conn.commit()
            conn.close()
        return {"session_key": session_id, "agent": _Agent(session_id)}

    def gate(self, text, session=None):
        session = session or self.session()
        return desktop_screen.tui_turn_gate("ui-1", session, text)

    def last_record(self):
        with open(self.log_path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        return json.loads(lines[-1]) if lines else {}


class TestTuiTurnGate(TuiTurnGateTestCase):
    def test_allow_returns_none_and_screens_once(self):
        self.assertIsNone(self.gate("hello from the desktop app"))
        self.assertEqual(EngineStubHandler.screen_calls, 1)
        self.assertEqual(self.last_record().get("subject"), "desktop-tui")
        self.assertEqual(self.last_record().get("tier"), "allow")

    def test_block_returns_hold_notice(self):
        EngineStubHandler.block_pattern = "ATTACK"
        notice = self.gate("please ATTACK the system")
        self.assertIsInstance(notice, str)
        self.assertIn("Held by the desktop intake screen", notice)
        self.assertEqual(self.last_record().get("tier"), "block")

    def test_env_off_inert_logged_once(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self.assertIsNone(self.gate("hello"))
        self.assertIsNone(self.gate("second"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"), "skipped_disabled")
        with open(self.log_path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().strip().splitlines()), 1)

    def test_cli_source_out_of_scope(self):
        self.assertIsNone(self.gate("hello", self.session(
            "s-cli", rows=[("s-cli", "cli", "cli")])))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"),
                         "skipped_out_of_scope")

    def test_rebound_local_desktop_rescues(self):
        self.assertIsNone(self.gate("hello", self.session(
            "s-rb", rows=[("s-rb", "local", "desktop")])))
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_no_session_id_still_screens_under_process_scope(self):
        # The agent's session id is unavailable, but this process IS the
        # Desktop backend (HERMES_DESKTOP=1): fail-closed, screen anyway
        # (the UI sid rides along as the session reference).
        EngineStubHandler.block_pattern = "ATTACK"
        notice = self.gate("ATTACK now", {"agent": _Agent(None, None)})
        self.assertIsInstance(notice, str)
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_no_text_skips(self):
        self.assertIsNone(self.gate("   "))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"), "skipped_no_text")

    def test_multimodal_parts_screened(self):
        EngineStubHandler.block_pattern = "ATTACK"
        notice = self.gate([{"type": "text", "text": "see this"},
                            {"type": "text", "text": "ATTACK now"}])
        self.assertIsInstance(notice, str)
        self.assertEqual(len(EngineStubHandler.last_texts), 2)

    def test_unknown_row_process_scope_screens(self):
        # No row, no snapshot - but this IS the Desktop backend.
        EngineStubHandler.block_pattern = "ATTACK"
        notice = self.gate("ATTACK now", self.session("unknown-1"))
        self.assertIsInstance(notice, str)

    def test_unknown_row_outside_desktop_skips(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": ""}):
            self.assertIsNone(self.gate("hello", self.session("unknown-2")))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"),
                         "skipped_out_of_scope")

    def test_hostile_session_never_raises(self):
        class Hostile(dict):
            def get(self, key, default=None):
                raise RuntimeError("session refused")

        self.assertIsNone(desktop_screen.tui_turn_gate(
            "ui-1", Hostile(), "hello"))
        self.assertEqual(self.last_record().get("tier"), "skipped_error")

    def test_skip_records_carry_no_text(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self.gate("sk-live-SECRET")
        record = self.last_record()
        self.assertNotIn("preview", record)
        self.assertNotIn("SECRET", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
