"""Replay of Go2Stone/freightman#2463 through the real capture path.

2026-10-17 → 10-19: after a bridge reconnect, sendMessage kept resolving but
WhatsApp never acked anything, and the capture hook stored every Selim reply
as delivered. Here the hook's handle() runs against a real state.db and a
bridge stand-in that uses the real ack watch (fake_bridge.mjs).

Run: python3 -m unittest discover -s ops/profiles/selim/hooks/whatsapp-capture/tests
"""

import asyncio
import importlib.util
import json
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "971556003130@s.whatsapp.net"


def _load_handler(home: str):
    os.environ["HERMES_HOME"] = home
    spec = importlib.util.spec_from_file_location("wa_capture_handler", os.path.join(HERE, "..", "handler.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_db(home: str, created_at: float) -> None:
    conn = sqlite3.connect(os.path.join(home, "state.db"))
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, session_key TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, content TEXT);
        CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, session_key TEXT, chat_id TEXT,
            content TEXT, created_at REAL);
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 'agent:main:whatsapp:dm:971556003130')")
    conn.execute("INSERT INTO messages VALUES (1, 's1', 'user', ?, 'Any update on the quote?')", (created_at - 5,))
    conn.execute(
        "INSERT INTO delivery_obligations VALUES ('ob1', 'agent:main:whatsapp:dm:971556003130', ?, ?, ?)",
        (CHAT, "Hi Mohammed, the quote is attached.", created_at),
    )
    conn.commit()
    conn.close()


class CaptureAckReplay(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.bridge = subprocess.Popen(
            ["node", os.path.join(HERE, "fake_bridge.mjs")], stdout=subprocess.PIPE, text=True,
            env={**os.environ, "STALE_MS": "120000"},
        )
        self.base = f"http://127.0.0.1:{self.bridge.stdout.readline().strip()}"
        self.handler = _load_handler(self.home)
        self.handler.BRIDGE_URL = self.base
        self.handler.ACK_WAIT_SECONDS = 1.5
        self.handler.ACK_POLL_SECONDS = 0.1
        self.handler._await_obligation = lambda *a, **k: None
        self.stores = []
        self.handler._store_email = lambda args: self.stores.append(dict(args)) or True

    def tearDown(self):
        self.bridge.kill()
        self.bridge.wait()
        self.bridge.stdout.close()

    def _bridge(self, path):
        urllib.request.urlopen(urllib.request.Request(f"{self.base}{path}", method="POST")).read()

    def _health(self):
        return json.load(urllib.request.urlopen(f"{self.base}/health"))

    def _capture(self):
        asyncio.run(self.handler.handle("agent:end", {
            "platform": "whatsapp", "message": "Any update on the quote?",
            "response": "Hi Mohammed, the quote is attached.", "user_id": CHAT, "chat_id": CHAT,
            "session_id": "s1",
        }))

    def _out_statuses(self):
        return [s.get("deliveryStatus") for s in self.stores if s["direction"] == "OUTBOUND"]

    def test_lost_send_is_never_stored_as_delivered(self):
        created = time.time()
        _state_db(self.home, created)
        self._bridge(f"/_sent?id=WA-1&chatId={CHAT}")  # sendMessage resolved; no ack ever comes

        self._capture()
        time.sleep(2.0)  # the background check gives up

        self.assertEqual(self._out_statuses(), ["UNKNOWN"])
        self.assertNotIn("PROVIDER_ACCEPTED", self._out_statuses())

    def test_reply_not_sent_yet_is_not_accepted(self):
        # The old check read "nothing pending" as "all acked".
        _state_db(self.home, time.time())
        self._capture()
        time.sleep(2.0)
        self.assertEqual(self._out_statuses(), ["UNKNOWN"])

    def test_ack_after_the_row_exists_upgrades_it(self):
        _state_db(self.home, time.time())
        self._bridge(f"/_sent?id=WA-2&chatId={CHAT}")
        self._capture()
        self.assertEqual(self._out_statuses(), ["UNKNOWN"])

        self._bridge("/_ack?id=WA-2")
        time.sleep(0.5)

        self.assertEqual(self._out_statuses(), ["UNKNOWN", "PROVIDER_ACCEPTED"])
        self.assertEqual(self.stores[-1]["whatsappMsgId"], "wa-do-ob1")

    def test_acked_send_is_accepted_at_once(self):
        _state_db(self.home, time.time())
        self._bridge(f"/_sent?id=WA-3&chatId={CHAT}")
        self._bridge("/_ack?id=WA-3")
        self._capture()
        self.assertEqual(self._out_statuses(), ["PROVIDER_ACCEPTED"])

    def test_an_older_acked_send_to_the_chat_does_not_count(self):
        self._bridge(f"/_sent?id=WA-OLD&chatId={CHAT}")
        self._bridge("/_ack?id=WA-OLD")
        time.sleep(0.05)
        _state_db(self.home, time.time())
        self._bridge(f"/_sent?id=WA-4&chatId={CHAT}")
        self._capture()
        self.assertEqual(self._out_statuses()[0], "UNKNOWN")

    def test_bridge_without_ack_tracking_stores_like_before(self):
        _state_db(self.home, time.time())
        self.handler.BRIDGE_URL = "http://127.0.0.1:9"  # nothing listening
        self._capture()
        self.assertEqual(self._out_statuses(), [None])

    def test_health_turns_unhealthy_within_the_window(self):
        # The same bridge state, seen by the gateway's health check.
        bridge = subprocess.Popen(
            ["node", os.path.join(HERE, "fake_bridge.mjs")], stdout=subprocess.PIPE, text=True,
            env={**os.environ, "STALE_MS": "300"},
        )
        try:
            base = f"http://127.0.0.1:{bridge.stdout.readline().strip()}"
            urllib.request.urlopen(urllib.request.Request(f"{base}/_sent?id=WA-5&chatId={CHAT}", method="POST")).read()
            self.assertEqual(json.load(urllib.request.urlopen(f"{base}/health"))["status"], "connected")
            time.sleep(0.5)
            health = json.load(urllib.request.urlopen(f"{base}/health"))
            self.assertEqual(health["status"], "sends_unacked")
            self.assertEqual(health["unackedSends"], 1)
        finally:
            bridge.kill()
            bridge.wait()
            bridge.stdout.close()


if __name__ == "__main__":
    unittest.main()
