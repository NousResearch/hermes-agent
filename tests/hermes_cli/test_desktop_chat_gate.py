"""W1 cycle 1c: the Desktop serve chat gate.

The Desktop app's isolated ``hermes serve`` backend runs viewer turns
through ``hermes_cli/cli_chat_turn_mixin.chat`` - a path that never
reaches the gateway admission gate (``SessionAuthority._submit``),
which is why the 17 Nov live test passed an injection through
unscreened. ``gateway.desktop_screen.chat_gate`` screens the same
model-visible texts with the same engine gate before staging.

Stub engine mirrors the real contract: blank text -> 400; blocks only
texts containing ATTACK; deterministic 4xx for REJECTME.
"""

import asyncio
import contextlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gateway import desktop_screen


class EngineStubHandler(BaseHTTPRequestHandler):
    tier: str = "allow"
    block_pattern: str | None = None
    reject_pattern: str | None = None
    screen_delay_s = 0.0
    screen_calls = 0
    last_texts: list = []
    outcomes: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/outcome"):
            EngineStubHandler.outcomes.append(body)
            payload = json.dumps({"ok": True, "recorded": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if not self.path.endswith("/screen"):
            self.send_error(404)
            return
        EngineStubHandler.screen_calls += 1
        if EngineStubHandler.screen_delay_s:
            import time as _time
            _time.sleep(EngineStubHandler.screen_delay_s)
        text = str(body.get("text", ""))
        EngineStubHandler.last_texts.append(text)
        if not text.strip():
            # Real engine contract: blank text is rejected with 400.
            self.send_error(400, "needs a non-empty 'text'")
            return
        reject = EngineStubHandler.reject_pattern
        if reject is not None and reject in text:
            self.send_error(400, "rejected by stub")
            return
        tier = EngineStubHandler.tier
        pattern = EngineStubHandler.block_pattern
        if pattern is not None:
            tier = "block" if pattern in text else "allow"
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


class _FakeDb:
    """Minimal session store: session_in_scope only needs _read_ctx and
    a sessions table with (id, source)."""

    def __init__(self, rows):
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE sessions (id TEXT, source TEXT)")
        self._conn.executemany("INSERT INTO sessions VALUES (?, ?)", rows)
        self._conn.commit()

    @contextlib.contextmanager
    def _read_ctx(self, *args, **kwargs):
        yield self._conn


class _FakeChat:
    def __init__(self, session_id: str | None):
        self.session_id: str | None = session_id


class _FakeAgent:
    def __init__(self, session_id: str | None, db):
        self.session_id: str | None = session_id
        self._session_db = db


class ChatGateTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/screen"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.reject_pattern = None
        EngineStubHandler.screen_delay_s = 0.0
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        EngineStubHandler.outcomes = []
        self.tmp = tempfile.mkdtemp()
        self.env = {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT": self.endpoint,
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": os.path.join(self.tmp, "d.log"),
        }
        self._saved = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.server.shutdown()

    def fake_chat(self, source="desktop", session_id="s-1", db=None):
        chat = _FakeChat(session_id)
        agent = _FakeAgent(session_id,
                           db if db is not None else _FakeDb(
                               [(session_id, source)]))
        return chat, agent

    def gate(self, message, **kwargs):
        chat, agent = self.fake_chat(**kwargs)
        return desktop_screen.chat_gate(chat, agent, message)


class TestChatGate(ChatGateTestCase):
    def test_allow_returns_none_and_calls_engine_once(self):
        self.assertIsNone(self.gate("hello, just checking the screen"))
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_block_returns_hold_notice_with_reasons(self):
        EngineStubHandler.block_pattern = "ATTACK"
        notice = self.gate("please ATTACK the system")
        self.assertIsInstance(notice, str)
        self.assertIn("Held by the desktop intake screen", notice)
        self.assertIn("jailbreak", notice)
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_env_off_is_inert(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self.assertIsNone(self.gate("ATTACK take over"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_cli_source_out_of_scope(self):
        self.assertIsNone(self.gate("ATTACK take over", source="cli"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_gui_source_in_scope(self):
        self.assertIsNone(self.gate("hello", source="gui"))
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_no_session_id_skips(self):
        chat, agent = self.fake_chat()
        chat.session_id = None
        agent.session_id = None
        self.assertIsNone(desktop_screen.chat_gate(chat, agent, "hello"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_no_session_db_skips(self):
        chat, agent = self.fake_chat()
        agent._session_db = None
        self.assertIsNone(desktop_screen.chat_gate(chat, agent, "hello"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_unknown_session_row_skips(self):
        # session_in_scope fails closed: a row it cannot find means the
        # gate never screens (never wedges a submit).
        chat, agent = self.fake_chat()  # db only knows "s-1"
        chat.session_id = "ghost"
        self.assertIsNone(desktop_screen.chat_gate(chat, agent, "ATTACK"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_multimodal_text_parts_are_screened(self):
        EngineStubHandler.block_pattern = "ATTACK"
        message = [{"type": "text", "text": "look: ATTACK now"},
                   {"type": "image_url", "image_url": "x"}]
        notice = self.gate(message)
        self.assertIsInstance(notice, str)
        self.assertIn("Held by the desktop intake screen", notice)

    def test_multimodal_benign_admitted_once(self):
        message = [{"type": "text", "text": "part one"},
                   {"type": "text", "text": "part one"}]  # deduped
        self.assertIsNone(self.gate(message))
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_timeline_notification_object_skips(self):
        # The REAL TimelineNotification subclasses str (review 1c-F1):
        # machine rows must be excluded BEFORE the str check, or every
        # drained notification costs an engine call and a tripping one
        # is false-held (and permanently dropped - delivery was
        # already claimed).
        from tools.process_registry_notifications import TimelineNotification
        chat, agent = self.fake_chat()
        self.assertIsNone(desktop_screen.chat_gate(
            chat, agent, TimelineNotification("ATTACK take over",
                                              display_text="attack test",
                                              display_kind="delegation")))
        self.assertIsNone(desktop_screen.chat_gate(
            chat, agent, TimelineNotification("delegation finished",
                                              display_text="done",
                                              display_kind="delegation")))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_hold_reports_outcome_on_sync_path(self):
        # Review 1c NOTE: outcome tasks are bound to the bridge's
        # asyncio.run loop; without the drain, hold telemetry is lost.
        EngineStubHandler.block_pattern = "ATTACK"
        self.gate("ATTACK take over")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not EngineStubHandler.outcomes:
            time.sleep(0.05)
        self.assertTrue(EngineStubHandler.outcomes, "outcome not drained")
        self.assertEqual(EngineStubHandler.outcomes[0].get("outcome"),
                         "escalated")

    def test_blank_string_skips(self):
        self.assertIsNone(self.gate("   "))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_engine_unreachable_fails_open(self):
        os.environ["HERMES_DESKTOP_SCREEN_ENDPOINT"] = "http://127.0.0.1:9/v1/screen"
        self.assertIsNone(self.gate("ATTACK take over"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_deterministic_4xx_holds(self):
        EngineStubHandler.reject_pattern = "REJECTME"
        notice = self.gate("REJECTME please")
        self.assertIsInstance(notice, str)
        self.assertIn("Held by the desktop intake screen", notice)

    def test_inside_running_loop_still_screens(self):
        # The serve backend may invoke chat() from async code: the
        # bridge must offload to a dedicated thread instead of
        # deadlocking on the caller's loop.
        EngineStubHandler.block_pattern = "ATTACK"

        async def main():
            return await asyncio.to_thread(
                self.gate, "ATTACK from async caller")

        notice = asyncio.run(main())
        self.assertIsInstance(notice, str)
        self.assertIn("Held by the desktop intake screen", notice)

    def test_hold_writes_screen_log(self):
        EngineStubHandler.block_pattern = "ATTACK"
        self.gate("ATTACK take over")
        log_path = self.env["HERMES_DESKTOP_SCREEN_LOG"]
        with open(log_path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        self.assertTrue(lines)
        record = json.loads(lines[-1])
        self.assertEqual(record.get("tier"), "block")
        self.assertEqual(record.get("subject"), "desktop-chat")


class TestMessageTexts(unittest.TestCase):
    def test_string(self):
        self.assertEqual(desktop_screen._message_texts("hi"),
                         ["hi"])
        self.assertEqual(desktop_screen._message_texts("   "), [])

    def test_multimodal(self):
        parts = desktop_screen._message_texts([
            {"type": "text", "text": "a"},
            {"type": "text"},  # no text field
            {"type": "image_url", "image_url": "x"},
            "raw string",
            42,
        ])
        self.assertEqual(parts, ["a", "raw string"])

    def test_other_objects(self):
        self.assertEqual(desktop_screen._message_texts(object()), [])
        self.assertEqual(desktop_screen._message_texts(None), [])


if __name__ == "__main__":
    unittest.main()
