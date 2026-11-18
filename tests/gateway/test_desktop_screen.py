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
- SessionAuthority._submit (the single shared choke point - r4/r5
  after the owner's 17 Nov live test proved the Desktop app's own
  `hermes serve` backend submits straight to the authority): the
  gate fires BEFORE durable admission, viewer text is screened even
  when canonical params (api_text / turn_author / model_override /
  clarify_batch / cli_turn_v1 branding) are present, machine
  producers (kanban cards) skip, and a hold releases captured media
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
    tier: str = "allow"
    block_pattern: str | None = None  # when set: block only texts containing this
    reject_pattern: str | None = None  # when set: 400 for texts containing this
    rate_limit_after: int | None = None  # when set: error after N screen calls
    rate_limit_code: int = 429  # which code that error uses (408/429/...)  
    blank_bodies = 0  # blank texts posted to the engine (must stay 0)
    screen_delay_s = 0.0  # simulate a slow engine (budget tests)
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
            if EngineStubHandler.screen_delay_s:
                time.sleep(EngineStubHandler.screen_delay_s)
            EngineStubHandler.last_caller = self.headers.get("X-Judge-Caller", "")
            EngineStubHandler.last_purpose = self.headers.get("X-Judge-Purpose", "")
            EngineStubHandler.last_text = body.get("text")
            EngineStubHandler.last_texts.append(body.get("text"))
            text = str(body.get("text", ""))
            if not text.strip():
                # Mirror the real engine contract (engine/api.py rejects
                # blank text with 400). Without this, the unit suite
                # cannot see a blank-chunk regression at all - that is
                # exactly how review r8-HIGH escaped the tests.
                EngineStubHandler.blank_bodies += 1
                self.send_error(400, "needs a non-empty 'text'")
                return
            reject = EngineStubHandler.reject_pattern
            if reject is not None and reject in text:
                self.send_error(400, "rejected by stub")
                return
            rate = EngineStubHandler.rate_limit_after
            if rate is not None and EngineStubHandler.screen_calls > rate:
                self.send_error(EngineStubHandler.rate_limit_code,
                                "transient error by stub")
                return
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
        EngineStubHandler.reject_pattern = None
        EngineStubHandler.rate_limit_after = None
        EngineStubHandler.rate_limit_code = 429
        EngineStubHandler.blank_bodies = 0
        EngineStubHandler.screen_delay_s = 0.0
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
        held, _reasons = self.run_screen("hello")
        self.assertFalse(held)
        logged = self.read_log()
        self.assertEqual(logged[-1]["tier"], "chain_down")

    def test_empty_text_skips_engine(self):
        held, reasons = self.run_screen("   ")
        self.assertEqual((held, reasons), (False, []))
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_every_chunk_outcome_reported(self):
        # Review r7-4: each judged chunk's allow verdict is reported
        # to the outcome join, not only the last one.
        self.run_screen("D" * 70000)
        self.await_outcomes(3)
        self.assertEqual(len(EngineStubHandler.outcomes), 3)


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
        # Outside the Desktop backend an unknowable source is out of
        # scope (review 1c-r4: the backend itself screens instead).
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": ""}):
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
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": ""}):
            self.assertFalse(session_in_scope(FakeAuthority(raise_on_read=True), "s1"))

    def test_authority_without_launch_sources_not_in_scope(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": ""}):
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

    def test_committed_payload_nested_api_text_is_screened(self):
        # Review r5-2: at the authority choke point the committed
        # payload nests api_text under cli_turn_v1 - the resolver must
        # read that shape too, or the hostile model-facing text rides
        # in behind a benign persisted text.
        self.assertEqual(
            desktop_screen.submit_texts({
                "text": "what is the weather today",
                "cli_turn_v1": {"api_text": "ATTACK take over"}}),
            ["ATTACK take over", "what is the weather today"])

    def test_nested_and_top_level_dedupe(self):
        self.assertEqual(
            desktop_screen.submit_texts({
                "text": "same words",
                "cli_turn_v1": {"api_text": "same words"}}),
            ["same words"])


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
        EngineStubHandler.reject_pattern = None
        EngineStubHandler.rate_limit_after = None
        EngineStubHandler.rate_limit_code = 429
        EngineStubHandler.blank_bodies = 0
        EngineStubHandler.screen_delay_s = 0.0
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

    def test_cli_branded_payload_is_screened_anyway(self):
        # Review r5-1: cli_turn_v1 is committed from ORDINARY canonical
        # params (api_text / turn_author / model_override / clarify_batch)
        # that the desktop app's own backend sends in normal flows - the
        # branding is client-asserted and must never exempt a submit.
        EngineStubHandler.tier = "block"
        from hermes_state_runtime import RuntimeStoreError
        for extra in ({"cli_turn_v1": {"turn_author": "x"}},
                      {"clarify_batch": True},
                      {"api_text": "ATTACK model-facing"},
                      {"turn_author": {"kind": "voice"}},
                      {"model_override": {"model": "m"}}):
            EngineStubHandler.screen_calls = 0
            with self.assertRaises(RuntimeStoreError) as raised:
                self._run(dict({"text": "ATTACK plain"}, **extra))
            self.assertEqual(raised.exception.reason, "desktop_screen_hold",
                             extra)
            self.assertEqual(EngineStubHandler.screen_calls, 1, extra)

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

    def test_tail_beyond_the_old_cap_is_screened(self):
        # Review r6-1 repro: 4050 benign chars then the attack - the
        # old 4000-char truncation admitted this; segmentation does not.
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "B" * 4050 + "ATTACK outside the boundary"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_tail_in_nested_api_text_is_screened(self):
        # Review r6-1, api_text side: the committed payload can carry a
        # 1M-char api_text - its tail must be judged too.
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "benign opener",
                       "cli_turn_v1": {"api_text":
                                       "C" * 5000 + "ATTACK api tail"}})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_all_chunks_judged(self):
        # 70k benign chars = 3 chunks against MAX_TEXT_CHARS=32000;
        # every chunk must reach the engine.
        self._run({"text": "D" * 70000})
        self.assertEqual(EngineStubHandler.screen_calls, 3)

    def test_budget_expiry_holds_for_coverage(self):
        # Coverage invariant: if the blocking budget runs out with
        # chunks unjudged, the submit is HELD (fail-closed for
        # coverage), never silently allowed.
        from hermes_state_runtime import RuntimeStoreError
        with mock.patch.object(desktop_screen, "_SCREEN_BUDGET_S", 0.0):
            with self.assertRaises(RuntimeStoreError) as raised:
                self._run({"text": "E" * 70000})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_seam_split_attack_is_held(self):
        # Review r7-3: a payload spanning a chunk boundary must not
        # dissolve into two harmless fragments - the overlap window
        # keeps every seam visible to at least one chunk.
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "B" * 31998 + "ATTACK take over the agent"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_oversized_text_is_held_not_truncated(self):
        # Review r7-1: no silent tail drop - text above the judging
        # ceiling is HELD with zero engine calls.
        from hermes_state_runtime import RuntimeStoreError
        with mock.patch.object(desktop_screen, "_MAX_SCREEN_CHARS", 100):
            with self.assertRaises(RuntimeStoreError) as raised:
                self._run({"text": "F" * 200})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def test_budget_clipped_timeout_holds_not_failopen(self):
        # Review r7-2: a call that times out AT the budget boundary is
        # coverage exhaustion -> HELD; only a genuine mid-budget engine
        # failure may fail open.
        from hermes_state_runtime import RuntimeStoreError
        EngineStubHandler.screen_delay_s = 0.9
        with mock.patch.object(desktop_screen, "_SCREEN_BUDGET_S", 0.6):
            with self.assertRaises(RuntimeStoreError) as raised:
                self._run({"text": "hello"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_shared_deadline_expired_holds(self):
        import time as _time
        self.loop = asyncio.new_event_loop()
        try:
            held, reasons = self.loop.run_until_complete(
                desktop_screen.screen("s", "s1", "hello",
                                      deadline=_time.monotonic() - 1))
        finally:
            self.loop.close()
        self.assertTrue(held)
        self.assertEqual(reasons, ["screening_incomplete_budget"])
        self.assertEqual(EngineStubHandler.screen_calls, 0)

    def _blank_middle_text(self, tail: str) -> str:
        # A whitespace run long enough that one chunk is ENTIRELY blank,
        # derived from the constants (round-11 NOTE: a hard-coded 70,000
        # went blind if MAX_TEXT_CHARS changed). Chunk k starts at
        # k*(MAX_TEXT_CHARS-_SEAM_OVERLAP); chunk 1 must lie fully inside
        # the run: prefix + run >= step + MAX_TEXT_CHARS.
        step = (desktop_screen.MAX_TEXT_CHARS
                - desktop_screen._SEAM_OVERLAP)
        run = step + desktop_screen.MAX_TEXT_CHARS  # 62,976 at 32k/1k
        prefix = "hello"
        text = prefix + " " * (run - len(prefix)) + tail
        blank = [c for c in desktop_screen.segments(text) if not c.strip()]
        self.assertTrue(blank, "fixture must contain an all-whitespace chunk")
        return text

    def test_whitespace_middle_chunk_does_not_bypass(self):
        # Review r8-HIGH repro: 'hello' + long spaces + attack used to
        # send a blank chunk, earn a deterministic engine 400, and
        # fail-open the tail. Blank chunks are skipped; the attack
        # chunk is still judged and held.
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": self._blank_middle_text(
                "ATTACK take over the agent")})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_benign_blank_middle_is_admitted_and_posts_no_blank(self):
        # Pins the blank-skip: the fixture guarantees an all-whitespace
        # chunk, and with the stub enforcing the real engine's
        # blank->400 contract, deleting the skip turns this benign
        # submit into a hold.
        self._run({"text": self._blank_middle_text("how are you")})
        self.assertEqual(EngineStubHandler.blank_bodies, 0)

    def test_transient_408_and_429_fail_open(self):
        # Review r9-NOTE 2 + round-10 mutation (e): a rate-limited (429)
        # or proxy-timeout (408) engine is ALIVE but busy - fail open,
        # never lock the owner out. Both codes are pinned separately;
        # dropping either from the transient set must fail this test.
        for code in (408, 429):
            with self.subTest(code=code):
                EngineStubHandler.rate_limit_after = 1
                EngineStubHandler.rate_limit_code = code
                EngineStubHandler.screen_calls = 0
                self._run({"text": "E" * 70000})
                self.assertEqual(EngineStubHandler.screen_calls, 2)
                EngineStubHandler.rate_limit_after = None

    def test_engine_4xx_holds_not_failopen(self):
        # A deterministic client rejection means the engine cannot
        # judge that chunk - coverage hole -> HOLD, never fail-open.
        EngineStubHandler.reject_pattern = "REJECTME"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "REJECTME harmless"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")

    def test_seam_sentinel_longer_than_old_overlap_is_held(self):
        # Review r8-LOW repro: an INDIVISIBLE 400-char sentinel
        # straddling the old 256-char overlap was never fully inside
        # one chunk (max ~360 chars per chunk) and was admitted on r8.
        # The 1 KB overlap contains it whole. The marker is
        # non-repetitive on purpose: a periodic marker would match on a
        # fragment and make this test pass on the old code too.
        marker = ("GHIJK" + "".join(chr(97 + (i * 7) % 26) for i in range(390))
                  + "ZYXWV")
        self.assertEqual(len(marker), 400)
        EngineStubHandler.block_pattern = marker
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self._run({"text": "B" * 31700 + marker + "B" * 2000})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")


class TestAuthorityWiring(unittest.TestCase):
    """SessionAuthority._submit consults the gate BEFORE durable
    admission (the sentinel proves the ordering) and releases a held
    submit's captured media like any refused admission."""

    def _submit(self, payload, source="desktop", tier="allow"):
        EngineStubHandler.tier = tier
        from gateway import session_authority
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
        EngineStubHandler.block_pattern = None
        EngineStubHandler.reject_pattern = None
        EngineStubHandler.rate_limit_after = None
        EngineStubHandler.rate_limit_code = 429
        EngineStubHandler.blank_bodies = 0
        EngineStubHandler.screen_delay_s = 0.0
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
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
        _receipt, calls, exc = self._submit({"text": "hello"})
        self.assertIsNone(exc)
        self.assertEqual(calls["admit"], 1)
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_block_holds_before_durable_admission(self):
        _receipt, calls, exc = self._submit({"text": "take over"}, tier="block")
        self.assertIsNotNone(exc)
        self.assertEqual(getattr(exc, "reason", None), "desktop_screen_hold")
        self.assertEqual(calls["admit"], 0)
        self.assertEqual(calls["released"], 1)

    def test_nested_hostile_api_text_held_before_admission(self):
        # Review r5-2 repro at the real call site: benign persisted
        # text + hostile model-facing api_text nested under cli_turn_v1
        # (the committed shape) must be HELD, engine judging BOTH.
        EngineStubHandler.block_pattern = "ATTACK"
        EngineStubHandler.last_texts = []
        _receipt, calls, exc = self._submit({
            "text": "what is the weather today",
            "api_text": "ATTACK take over the agent"}, tier="allow")
        self.assertIsNotNone(exc)
        self.assertEqual(getattr(exc, "reason", None), "desktop_screen_hold")
        self.assertEqual(calls["admit"], 0)
        # Hold-on-first-block short-circuits: the engine judged the
        # hostile model-facing text and the submit was refused before
        # anything was admitted (screening the remaining text of an
        # already-refused submit would add nothing).
        self.assertEqual(EngineStubHandler.last_texts,
                         ["ATTACK take over the agent"])

    def test_canonical_param_matrix_all_screened(self):
        # The independent reviewers' full bypass matrix (r4 P3/P8,
        # r5 round-1/r2), pinned as permanent regression coverage: NO
        # combination of canonical viewer params may carry hostile
        # text past the gate on a desktop-source session. Each shape
        # must be REFUSED with zero admissions and the engine must
        # have judged the hostile string.
        EngineStubHandler.block_pattern = "ATTACK"
        shapes = [
            {"text": "ATTACK plain"},
            {"text": "benign words", "api_text": "ATTACK model-facing"},
            {"text": "ATTACK persisted", "api_text": "benign model-facing"},
            {"text": "ATTACK with author", "turn_author": {"name": "voice-prefix"}},
            {"text": "ATTACK with override", "model_override": {"model": "m"}},
            {"text": "ATTACK with batch", "clarify_batch": True},
            {"text": "ATTACK all fields", "api_text": "ATTACK api",
             "turn_author": {"name": "voice-prefix"},
             "model_override": {"model": "m"}, "clarify_batch": True},
        ]
        for shape in shapes:
            EngineStubHandler.last_texts = []
            _receipt, calls, exc = self._submit(shape, tier="allow")
            self.assertIsNotNone(exc, shape)
            self.assertEqual(getattr(exc, "reason", None),
                             "desktop_screen_hold", shape)
            self.assertEqual(calls["admit"], 0, shape)
            self.assertTrue(
                any("ATTACK" in str(t) for t in EngineStubHandler.last_texts),
                {"shape": shape, "engine_saw": EngineStubHandler.last_texts})


if __name__ == "__main__":
    unittest.main()
