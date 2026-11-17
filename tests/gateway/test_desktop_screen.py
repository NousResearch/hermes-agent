"""W1 desktop intake screen tests (owner order 16 Nov 2026; CBA
section 4 - placement class BLOCKING, purpose id desktop-screen).

Covers the intake-hold contract against a real loopback HTTP stub
speaking the engine's wire format:

- default-inert config; enabled flag spellings
- allow / review / block / chain_down tiers and the outcome loop
- unreachable / malformed engine responses fail open, never raise
- purpose + caller attribution headers (the W1 per-surface tagging)
- session launch-source scoping (desktop surfaces only by default)
- the hold refusal shape the control plane returns to the client
- the three wiring points consult the gate before any state changes:
  Controls.submit, prompt.steer_queued, prompt.replace_queued
- a slow /outcome endpoint never delays the intake decision (the
  Phase 2 review B4 rule, carried over)
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gateway import desktop_screen
from gateway.desktop_screen import screen, session_in_scope


class EngineStubHandler(BaseHTTPRequestHandler):
    tier = "allow"
    block_pattern = None  # when set: block only texts containing this
    outcome_delay_s = 0.0
    outcomes = []
    last_caller = None
    last_purpose = None
    last_text = None
    last_texts = []
    screen_calls = 0
    malformed = False

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/screen"):
            EngineStubHandler.screen_calls += 1
            EngineStubHandler.last_caller = self.headers.get("X-Judge-Caller", "")
            EngineStubHandler.last_purpose = self.headers.get("X-Judge-Purpose", "")
            EngineStubHandler.last_text = body.get("text")
            EngineStubHandler.last_texts.append(body.get("text"))
            if EngineStubHandler.malformed:
                payload = json.dumps(["not", "a", "dict"]).encode()
            else:
                tier = EngineStubHandler.tier
                pattern = EngineStubHandler.block_pattern
                if pattern is not None:
                    tier = "block" if pattern in str(body.get("text", "")) else "allow"
                response = {"tier": tier,
                            "reasons": (["jailbreak", "injection"]
                                        if tier == "block"
                                        else ([] if tier == "allow"
                                              else ["borderline"])),
                            "request_id": "req-stub-1",
                            "served_by": {"provider": "stub"},
                            "scores": {}}
                payload = json.dumps(response).encode()
        elif self.path.endswith("/outcome"):
            if EngineStubHandler.outcome_delay_s:
                time.sleep(EngineStubHandler.outcome_delay_s)
            EngineStubHandler.outcomes.append(body)
            payload = json.dumps({"ok": True, "recorded": True}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class DesktopScreenTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Persistent loop: production's loop outlives one screen call,
        # so fire-and-forget outcome tasks must survive it too.
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/screen"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.outcome_delay_s = 0.0
        EngineStubHandler.outcomes = []
        EngineStubHandler.last_caller = None
        EngineStubHandler.last_purpose = None
        EngineStubHandler.last_text = None
        EngineStubHandler.last_texts = []
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.malformed = False
        self.env = {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT": self.endpoint,
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": os.path.join(self.tmp, "d.log"),
        }
        self.patcher = mock.patch.dict(os.environ, self.env)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.server.shutdown)
        # LIFO: drains BEFORE the stub server shuts down (registered last)
        self.addCleanup(lambda: self.loop.run_until_complete(
            desktop_screen.drain_outcome_tasks(timeout=3.0)))

    def run_screen(self, text="hello", subject="owner-1", session="s-1"):
        return self.loop.run_until_complete(
            screen(subject, session, text))

    def await_outcomes(self, count=1, timeout=2.0):
        deadline = time.time() + timeout
        while len(EngineStubHandler.outcomes) < count and time.time() < deadline:
            self.loop.run_until_complete(asyncio.sleep(0.02))

    def read_log(self):
        path = Path(os.path.join(self.tmp, "d.log"))
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()]


class TestEnabled(DesktopScreenTestCase):
    def test_default_inert_without_env(self):
        with mock.patch.dict(os.environ, {
                "HERMES_DESKTOP_SCREEN": "",
                "HERMES_DESKTOP_SCREEN_ENDPOINT": self.endpoint}):
            self.assertFalse(desktop_screen.desktop_screen_enabled())
            held, reasons = self.run_screen()
        self.assertEqual((held, reasons), (False, []))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_enabled_spellings(self):
        for spelling in ("1", "true", "YES", "On"):
            with mock.patch.dict(os.environ,
                                 {"HERMES_DESKTOP_SCREEN": spelling}):
                self.assertTrue(desktop_screen.desktop_screen_enabled(),
                                spelling)
        for spelling in ("", "0", "off", "no"):
            with mock.patch.dict(os.environ,
                                 {"HERMES_DESKTOP_SCREEN": spelling}):
                self.assertFalse(desktop_screen.desktop_screen_enabled(),
                                 spelling)


class TestTiers(DesktopScreenTestCase):
    def test_allow_proceeds_and_reports_accepted(self):
        held, reasons = self.run_screen("good morning")
        self.assertEqual((held, reasons), (False, []))
        self.await_outcomes(1)
        self.assertEqual(len(EngineStubHandler.outcomes), 1)
        self.assertEqual(EngineStubHandler.outcomes[0]["outcome"], "accepted")

    def test_review_proceeds_and_reports_escalated(self):
        EngineStubHandler.tier = "review"
        held, reasons = self.run_screen("odd message")
        self.assertFalse(held)
        self.assertEqual(reasons, ["borderline"])
        self.await_outcomes(1)
        self.assertEqual(EngineStubHandler.outcomes[0]["outcome"], "escalated")

    def test_block_holds_and_reports_escalated(self):
        EngineStubHandler.tier = "block"
        held, reasons = self.run_screen("take over the agent")
        self.assertTrue(held)
        self.assertEqual(reasons, ["jailbreak", "injection"])
        self.await_outcomes(1)
        self.assertEqual(EngineStubHandler.outcomes[0]["outcome"], "escalated")
        logged = self.read_log()
        self.assertEqual(logged[-1]["tier"], "block")
        self.assertEqual(logged[-1]["subject"], "owner-1")

    def test_chain_down_proceeds(self):
        EngineStubHandler.tier = "chain_down"
        held, reasons = self.run_screen("hello")
        self.assertFalse(held)
        logged = self.read_log()
        self.assertEqual(logged[-1]["tier"], "chain_down")

    def test_empty_text_skips_engine(self):
        held, reasons = self.run_screen("   ")
        self.assertEqual((held, reasons), (False, []))
        self.assertEqual(EngineStubHandler.screen_calls, 0)


class TestFailOpen(DesktopScreenTestCase):
    def test_engine_unreachable_proceeds(self):
        with mock.patch.dict(os.environ, {
                "HERMES_DESKTOP_SCREEN_ENDPOINT":
                    "http://127.0.0.1:9/v1/screen"}):
            held, reasons = self.run_screen("hello")
        self.assertEqual((held, reasons), (False, []))
        logged = self.read_log()
        self.assertEqual(logged[-1]["tier"], "engine_unreachable_or_malformed")

    def test_malformed_response_proceeds(self):
        EngineStubHandler.malformed = True
        held, reasons = self.run_screen("hello")
        self.assertEqual((held, reasons), (False, []))
        logged = self.read_log()
        self.assertEqual(logged[-1]["tier"], "engine_unreachable_or_malformed")

    def test_garbage_numeric_env_never_raises(self):
        for bad in ("abc", "nan", "-3", "inf", "0"):
            with mock.patch.dict(os.environ, {
                    "HERMES_DESKTOP_SCREEN_TIMEOUT_S": bad}):
                held, _ = self.run_screen("hello")
                self.assertFalse(held, bad)

    def test_screen_never_raises_on_any_input(self):
        for text in (None, "", 12345, b"bytes", {"dict": 1}):
            held, reasons = self.loop.run_until_complete(
                screen("s", "sess", text))
            self.assertEqual((held, reasons), (False, []), repr(text))

    def test_slow_outcome_never_delays_the_decision(self):
        EngineStubHandler.outcome_delay_s = 5.0
        started = time.monotonic()
        held, _ = self.run_screen("hello")
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse(held)


class TestAttribution(DesktopScreenTestCase):
    def test_default_caller_and_purpose_headers(self):
        self.run_screen()
        self.assertEqual(EngineStubHandler.last_caller, "hermes-desktop")
        self.assertEqual(EngineStubHandler.last_purpose, "desktop-screen")

    def test_env_overrides_caller_and_purpose(self):
        with mock.patch.dict(os.environ, {
                "HERMES_DESKTOP_SCREEN_CALLER": "hermes-desktop-test",
                "HERMES_DESKTOP_SCREEN_PURPOSE": "desktop-screen-v2"}):
            self.run_screen()
        self.assertEqual(EngineStubHandler.last_caller, "hermes-desktop-test")
        self.assertEqual(EngineStubHandler.last_purpose, "desktop-screen-v2")


class FakeAuthority:
    """launch_sources + a read ctx + authorize shaped like the real rules."""

    def __init__(self, source=None, raise_on_read=False, allowed=True):
        self.launch_sources = {}
        self.source = source
        self.raise_on_read = raise_on_read
        self.allowed = allowed
        self.authorize_calls = []
        self.db = self

    def authorize(self, actor, ref, scope):
        self.authorize_calls.append(scope)
        if not self.allowed:
            from hermes_state_runtime import RuntimeStoreError
            raise RuntimeStoreError("permission_denied")

    def _read_ctx(self):
        if self.raise_on_read:
            raise RuntimeError("db unavailable")
        return FakeReadCtx(self.source)


class FakeReadCtx:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        return self  # sqlite3 cursor shape: execute() returns the cursor

    def fetchone(self):
        if self.source is None:
            return None
        return (self.source,)


class TestScope(DesktopScreenTestCase):
    def test_desktop_and_gui_in_scope_by_default(self):
        for source in ("desktop", "gui"):
            self.assertTrue(session_in_scope(FakeAuthority(source), "s1"),
                            source)

    def test_other_interactive_sources_out_of_scope_by_default(self):
        for source in ("cli", "tui", "acp", "cron", "kanban"):
            self.assertFalse(session_in_scope(FakeAuthority(source), "s1"),
                             source)

    def test_env_widens_to_all_interactive(self):
        with mock.patch.dict(os.environ,
                             {"HERMES_DESKTOP_SCREEN_SOURCES": "all"}):
            for source in ("cli", "tui", "acp", "desktop", "gui"):
                self.assertTrue(session_in_scope(FakeAuthority(source), "s1"),
                                source)

    def test_env_narrows_to_one_label(self):
        with mock.patch.dict(os.environ,
                             {"HERMES_DESKTOP_SCREEN_SOURCES": "desktop"}):
            self.assertTrue(session_in_scope(FakeAuthority("desktop"), "s1"))
            self.assertFalse(session_in_scope(FakeAuthority("gui"), "s1"))

    def test_env_keeps_only_known_labels(self):
        with mock.patch.dict(os.environ,
                             {"HERMES_DESKTOP_SCREEN_SOURCES": "desktop, nonsense"}):
            self.assertTrue(session_in_scope(FakeAuthority("desktop"), "s1"))
            self.assertFalse(session_in_scope(FakeAuthority("nonsense"), "s1"))

    def test_missing_session_row_not_in_scope(self):
        self.assertFalse(session_in_scope(FakeAuthority(None), "s1"))

    def test_launch_sources_cache_write_through(self):
        authority = FakeAuthority("desktop")
        authority.launch_sources["s1"] = "desktop"
        self.assertTrue(session_in_scope(authority, "s1"))

    def test_fresh_db_row_beats_stale_cache(self):
        # Review F4: a stale/poisoned cache entry must fail in the
        # SCREENED direction - the row is authoritative.
        authority = FakeAuthority("desktop")
        authority.launch_sources["s1"] = "cli"  # poisoned
        self.assertTrue(session_in_scope(authority, "s1"))
        self.assertEqual(authority.launch_sources["s1"], "desktop")

    def test_poisoned_cache_cannot_disable_screening(self):
        authority = FakeAuthority("cli")
        authority.launch_sources["s1"] = "desktop"  # stale row is cli
        self.assertFalse(session_in_scope(authority, "s1"))

    def test_db_failure_means_not_in_scope_never_raises(self):
        self.assertFalse(session_in_scope(FakeAuthority(raise_on_read=True), "s1"))

    def test_authority_without_launch_sources_not_in_scope(self):
        self.assertFalse(session_in_scope(SimpleNamespace(), "s1"))


class TestSubmitTexts(unittest.TestCase):
    """The model-visible text resolver (reviews F1 + N1: BOTH the
    api_text override (this turn's model input) and the persisted text
    (the row later turns load as model history) are screened -
    screening either alone moves the blind spot to the other)."""

    def test_both_fields_returned_api_text_first(self):
        self.assertEqual(
            desktop_screen.submit_texts({"text": "persisted words",
                                         "api_text": "voice prefix"}),
            ["voice prefix", "persisted words"])

    def test_identical_values_screened_once(self):
        self.assertEqual(
            desktop_screen.submit_texts({"text": "same",
                                         "api_text": "same"}),
            ["same"])

    def test_text_alone(self):
        self.assertEqual(desktop_screen.submit_texts({"text": "hello"}),
                         ["hello"])

    def test_blank_api_text_falls_back_to_text(self):
        self.assertEqual(
            desktop_screen.submit_texts({"text": "hello", "api_text": "  "}),
            ["hello"])

    def test_nothing_to_screen(self):
        self.assertEqual(desktop_screen.submit_texts({}), [])
        self.assertEqual(desktop_screen.submit_texts({"text": "  "}), [])
        self.assertEqual(desktop_screen.submit_texts(None), [])
        self.assertEqual(desktop_screen.submit_texts({"text": 12345}), [])


class TestHoldError(DesktopScreenTestCase):
    def test_hold_error_is_store_refusal_with_visible_notice(self):
        from hermes_state_runtime import RuntimeStoreError
        err = desktop_screen.hold_error(["jailbreak", "injection"])
        self.assertIsInstance(err, RuntimeStoreError)
        self.assertEqual(err.reason, "desktop_screen_hold")
        self.assertEqual(err.wire_code, 4001)
        self.assertIn("desktop intake screen", err.wire_message)
        self.assertEqual(err.reasons, ["jailbreak", "injection"])

    def test_hold_error_reasons_capped_and_tolerated(self):
        err = desktop_screen.hold_error([f"r{i}" for i in range(20)])
        self.assertEqual(len(err.reasons), 8)
        self.assertEqual(desktop_screen.hold_error(None).reasons, [])


@dataclass
class _Receipt:
    admission_id: str = "adm-1"
    status: str = "admitted"


class _Controls:
    """Just enough Controls for the unbound submit() call."""

    def __init__(self, authority, subject="owner-1", subscribed=True):
        self.authority = authority
        self.actor = SimpleNamespace(subject=subject, profile_id="p1")
        self._operator = False
        self.subscribed = subscribed
        self.submitted = None

    def _voice_stop_replay(self, ref, params):
        return False

    async def _typed_voice_stop(self, ref, params):
        return False

    def _record_voice_stop(self, ref, params):
        pass

    def submission(self, ref, params):
        # The real rule (session_controls.py): a connection that never
        # opened this session is refused before anything else runs.
        from hermes_state_runtime import RuntimeStoreError
        if not self.subscribed:
            raise RuntimeStoreError("permission_denied")
        return SimpleNamespace(ref=ref, params=params)

    async def authority_submit(self, actor, request, queue_while_restarting=False):
        self.submitted = request
        return _Receipt()


class TestAuthorityGate(unittest.TestCase):
    """The shared choke-point gate (desktop_screen.authority_gate):
    screens viewer text on desktop-launched sessions; machine
    producers and other surfaces are structurally out."""

    def _run(self, payload, source="desktop", kanban=False, enabled="1"):
        async def case():
            from gateway.desktop_screen import authority_gate
            await authority_gate(FakeAuthority(source), "owner-1", "s1",
                                 payload, kanban_card=kanban)
        self.loop = asyncio.new_event_loop()
        try:
            return self.loop.run_until_complete(case())
        finally:
            self.loop.close()

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/screen"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        self.env = {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT": self.endpoint,
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": os.path.join(tempfile.mkdtemp(), "d.log"),
        }
        self.patcher = mock.patch.dict(os.environ, self.env)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.server.shutdown)

    def test_block_holds(self):
        EngineStubHandler.tier = "block"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "take over"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_allow_passes(self):
        self._run({"text": "hello"})
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_kanban_card_skips(self):
        EngineStubHandler.tier = "block"
        self._run({"text": "take over"}, kanban=True)
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_cli_turn_skips(self):
        EngineStubHandler.tier = "block"
        self._run({"text": "deploy", "cli_turn_v1": {"turn_author": "x"}})
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_out_of_scope_skips(self):
        EngineStubHandler.tier = "block"
        self._run({"text": "hello"}, source="cron")
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_disabled_skips(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP_SCREEN": ""}):
            self._run({"text": "hello"})
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_both_texts_screened_hostile_second_holds(self):
        EngineStubHandler.block_pattern = "EXFILTRATE"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "EXFILTRATE everything",
                       "api_text": "voice note transcript"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")
        self.assertEqual(EngineStubHandler.last_texts,
                         ["voice note transcript", "EXFILTRATE everything"])

    def test_engine_down_proceeds(self):
        with mock.patch.dict(os.environ, {
                "HERMES_DESKTOP_SCREEN_ENDPOINT": "http://127.0.0.1:9/v1/screen"}):
            self._run({"text": "hello"})
        self.assertEqual(EngineStubHandler.screen_calls, 0)


class TestAuthorityWiring(unittest.TestCase):
    """SessionAuthority._submit consults the gate BEFORE durable
    admission (the sentinel proves the ordering) and releases a held
    submit's captured media like any refused admission."""

    def _submit(self, payload, source="desktop", tier="allow"):
        EngineStubHandler.tier = tier
        import gateway.session_authority as session_authority
        from gateway.session_authority import SessionAuthority

        calls = {"admit": 0, "released": 0}

        class _Auth:
            launch_sources = {}
            sessions = {"s1": SimpleNamespace(source=SimpleNamespace(user_id="owner-1"))}

            def __init__(self):
                self.profile_id = "p1"
                self.epoch = 1
                self._source = source

            @property
            def source(self):
                return self._source

            @property
            def db(self):
                return self

            def _read_ctx(self):
                return FakeReadCtx(self._source)

            def _require_admission_open(self, queue_only=False):
                return None

            def _admission_gate(self, authorize_write, queue_only=False):
                return None

            def _admitted(self, ref, marker):
                return None

            def _schedule_admitted(self, ref):
                return None

            def _release_refused_capture(self, request, payload):
                calls["released"] += 1

        authority = _Auth()
        actor = SimpleNamespace(subject="owner-1")
        ref = SimpleNamespace(session_id="s1")
        request = SimpleNamespace(ref=ref, request_id="r1", intent="queue",
                                  payload=dict(payload))

        async def fake_tracked_write(self_, admit, then=None, after=None, ordered=True):
            return admit()

        def fake_admit(*args, **kwargs):
            calls["admit"] += 1
            return {"admission_id": "adm-9"}

        real_admit = session_authority.admit_session_input
        import gateway.session_runtime_workers as workers
        real_write = workers.tracked_write
        session_authority.admit_session_input = fake_admit
        workers.tracked_write = fake_tracked_write
        try:
            try:
                self.loop = asyncio.new_event_loop()
                receipt = self.loop.run_until_complete(
                    SessionAuthority._submit(authority, actor, request))
                return receipt, calls, None
            except Exception as exc:
                return None, calls, exc
            finally:
                self.loop.close()
        finally:
            session_authority.admit_session_input = real_admit
            workers.tracked_write = real_write

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/screen"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.screen_calls = 0
        self.patcher = mock.patch.dict(os.environ, {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT": self.endpoint,
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": os.path.join(tempfile.mkdtemp(), "d.log"),
        })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.server.shutdown)

    def test_allow_reaches_durable_admission(self):
        receipt, calls, exc = self._submit({"text": "hello"})
        self.assertIsNone(exc)
        self.assertEqual(calls["admit"], 1)
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_block_holds_before_durable_admission(self):
        receipt, calls, exc = self._submit({"text": "take over"}, tier="block")
        self.assertIsNotNone(exc)
        self.assertEqual(getattr(exc, "reason", None), "desktop_screen_hold")
        self.assertEqual(calls["admit"], 0)
        self.assertEqual(calls["released"], 1)


if __name__ == "__main__":
    unittest.main()
