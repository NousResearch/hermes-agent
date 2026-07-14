"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it.
"""

import asyncio

import pytest

from gateway import code_skew


@pytest.fixture(autouse=True)
def _reset_boot_fingerprint(monkeypatch):
    """Each test starts with no recorded boot fingerprint."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)


def _pending_intent() -> code_skew.PendingModelSwitch:
    return code_skew.PendingModelSwitch(
        source={
            "platform": "telegram",
            "chat_id": "-1003418564280",
            "chat_type": "group",
            "user_id": "29264781",
            "thread_id": "3006",
            "message_id": "987",
            "profile": "default",
            "api_key": "must-not-persist",
        },
        model_input="openai/gpt-5.6",
        provider="openai-codex",
        persist_global=False,
    )


class TestDetectCodeSkew:
    def test_no_boot_fingerprint_means_no_skew(self, monkeypatch):
        # Nothing recorded (e.g. non-git install) -> never a false positive.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def456")
        assert code_skew.detect_code_skew() is None

    def test_unchanged_checkout_is_not_skew(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        assert code_skew.detect_code_skew() is None

    def test_drift_is_detected_with_short_revs(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "def4567890")

    def test_unreadable_current_rev_does_not_false_positive(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: None)
        assert code_skew.detect_code_skew() is None

    def test_record_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:first")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:second")
        code_skew.record_boot_fingerprint()  # must not overwrite the boot snapshot
        assert code_skew._boot_fingerprint == "git:refs/heads/main:first"


class TestShort:
    def test_shortens_long_sha(self):
        assert code_skew._short("git:refs/heads/main:abcdef0123456789") == "abcdef0123"

    def test_keeps_unresolved_marker(self):
        assert code_skew._short("git:refs/heads/main:unresolved") == "unresolved"

    def test_passes_short_sha_through_untruncated(self):
        assert code_skew._short("git:HEAD:abc1234") == "abc1234"


class TestModelSwitchSkewGuard:
    def test_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert slash_commands._model_switch_skew_guard() is None

    def test_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "hermes gateway restart" in msg

    def test_guard_queues_typed_intent_when_runner_provided(self, monkeypatch):
        """Stale code queues the switch instead of executing it or cutting work."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        intent = _pending_intent()
        calls = []

        class FakeRunner:
            def defer_code_skew_model_switch(self, queued_intent):
                calls.append(queued_intent)
                return False  # unrelated topic is still active

        msg = slash_commands._model_switch_skew_guard(
            runner=FakeRunner(), continuation=intent
        )
        assert msg is not None
        assert "safely queued" in msg
        assert "after active work completes" in msg
        assert calls == [intent]

    def test_replay_guard_does_not_enqueue_duplicate_intent_on_fresh_drift(self, monkeypatch):
        """The original durable intent owns retries across a second checkout drift."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        calls = []

        class FakeRunner:
            def defer_code_skew_model_switch(self, queued_intent):
                calls.append(queued_intent)
                return False

        intent = code_skew.PendingModelSwitch(
            source=_pending_intent().source,
            model_input="openai/gpt-5.6",
            provider="openai-codex",
            persist_global=False,
            replay=True,
        )
        msg = slash_commands._model_switch_skew_guard(runner=FakeRunner(), continuation=intent)

        assert msg is not None
        assert "remains queued" in msg
        assert calls == []

    def test_guard_falls_back_to_manual_when_queueing_fails(self, monkeypatch):
        """If durable queueing fails, preserve the original safe manual guard."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        class FakeRunner:
            def defer_code_skew_model_switch(self, queued_intent):
                raise OSError("state unavailable")

        msg = slash_commands._model_switch_skew_guard(
            runner=FakeRunner(), continuation=_pending_intent()
        )
        assert msg is not None
        assert "hermes gateway restart" in msg

    def test_guard_falls_back_to_manual_without_runner(self, monkeypatch):
        """Without a runner, the guard returns the manual restart message
        (preserving backwards compatibility for any caller not yet passing one)."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "hermes gateway restart" in msg

    def test_runner_restarts_only_after_last_active_topic_finishes(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._running_agents = {"telegram:-1003418564280:1270": object()}
        runner._running_agents_ts = {}
        runner._active_session_leases = {}
        runner._busy_ack_ts = {}
        runner._restart_task_started = False
        runner._restart_requested = False
        runner._draining = False
        runner._persist_active_agents = lambda: None
        queued = []
        calls = []
        monkeypatch.setattr(code_skew, "enqueue_pending_model_switch", queued.append)

        def request_restart(**kwargs):
            calls.append(kwargs)
            runner._restart_requested = True
            return True

        monkeypatch.setattr(runner, "request_restart", request_restart)
        monkeypatch.setenv("XPC_SERVICE_NAME", "com.apple.xpc.launchd")

        assert runner.defer_code_skew_model_switch(_pending_intent()) is False
        assert runner._new_turn_drain_message() is None
        assert queued
        assert calls == []

        runner._release_running_agent_state("telegram:-1003418564280:1270")
        assert runner._draining is True
        assert "restarting" in runner._new_turn_drain_message()
        assert calls == [{"detached": False, "via_service": True}]

    @pytest.mark.asyncio
    async def test_post_lookup_gate_blocks_turn_when_last_agent_finishes_during_yield(self, monkeypatch):
        """The final pre-claim await must not admit work after a code-skew drain latches."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        session_key = "telegram:-1003418564280:1270"
        runner._running_agents = {session_key: object()}
        runner._running_agents_ts = {}
        runner._active_session_leases = {}
        runner._busy_ack_ts = {}
        runner._restart_task_started = False
        runner._restart_requested = False
        runner._draining = False
        runner._code_skew_restart_when_idle = True
        runner._persist_active_agents = lambda: None
        restart_calls = []

        def request_restart(**kwargs):
            restart_calls.append(kwargs)
            runner._restart_requested = True
            return True

        runner.request_restart = request_restart
        monkeypatch.setenv("XPC_SERVICE_NAME", "com.apple.xpc.launchd")

        def finish_last_agent_during_lookup(_source):
            runner._release_running_agent_state(session_key)
            return False

        runner._is_telegram_topic_root_lobby = finish_last_agent_during_lookup
        handled, post_yield_drain_message = await runner._topic_root_lobby_or_drain(None)

        assert handled is False
        assert post_yield_drain_message is not None
        assert "restarting" in post_yield_drain_message
        assert runner._running_agents == {}
        assert restart_calls == [{"detached": False, "via_service": True}]


class TestPurgeStalePycache:
    """Tests for purge_stale_pycache — purging __pycache__ when the checkout
    advanced between the previous gateway boot and the current one."""

    def test_no_purge_when_previous_fingerprint_missing(self, monkeypatch, tmp_path):
        """No persisted fingerprint file → nothing to compare → no purge."""
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: tmp_path / "nonexistent")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False

    def test_no_purge_when_fingerprints_match(self, monkeypatch, tmp_path):
        """Same checkout as previous boot → no drift → no purge."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False

    def test_purges_pyc_files_on_drift(self, monkeypatch, tmp_path):
        """Drift detected → .pyc files inside __pycache__ are deleted."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:oldhash123")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:newhash456")
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)

        # Create a fake __pycache__ with .pyc files
        pycache = tmp_path / "gateway" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "slash_commands.cpython-311.pyc").write_bytes(b"stale bytecode")
        (pycache / "run.cpython-311.pyc").write_bytes(b"stale bytecode")

        assert code_skew.purge_stale_pycache() is True
        assert not (pycache / "slash_commands.cpython-311.pyc").exists()
        assert not (pycache / "run.cpython-311.pyc").exists()

    def test_skips_venv_pycache(self, monkeypatch, tmp_path):
        """.pyc inside .venv are not purged (they're managed by pip)."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:oldhash123")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:newhash456")
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)

        venv_pycache = tmp_path / ".venv" / "lib" / "__pycache__"
        venv_pycache.mkdir(parents=True)
        (venv_pycache / "site.cpython-311.pyc").write_bytes(b"should not be touched")

        code_skew.purge_stale_pycache()
        assert (venv_pycache / "site.cpython-311.pyc").exists()

    def test_no_purge_when_fingerprint_unreadable(self, monkeypatch, tmp_path):
        """If _fingerprint returns None (non-git), no purge."""
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: None)
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: tmp_path / "x")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False


class TestPersistBootFingerprint:
    def test_persists_fingerprint_to_file(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        code_skew._persist_boot_fingerprint("git:refs/heads/main:abcdef1234")
        assert fp_file.read_text() == "git:refs/heads/main:abcdef1234"

    def test_does_not_persist_none(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        code_skew._persist_boot_fingerprint(None)
        assert not fp_file.exists()


class TestPendingModelSwitch:
    def test_queue_preserves_topic_and_never_serializes_secrets(self, monkeypatch, tmp_path):
        state_file = tmp_path / ".pending_model_switches.json"
        monkeypatch.setattr(code_skew, "_pending_model_switches_file", lambda: state_file)
        intent = code_skew.PendingModelSwitch(
            source=_pending_intent().source,
            model_input="openai/gpt-5.6",
            provider="openai-codex",
            persist_global=False,
            replay=True,
        )

        code_skew.enqueue_pending_model_switch(intent)

        stored = state_file.read_text(encoding="utf-8")
        assert "api_key" not in stored
        assert "must-not-persist" not in stored
        assert '"replay"' not in stored
        pending = code_skew.peek_pending_model_switch()
        assert pending is not None
        assert pending.model_input == intent.model_input
        assert pending.provider == intent.provider
        assert pending.persist_global is intent.persist_global
        assert pending.replay is False
        assert pending.source["thread_id"] == "3006"
        assert "api_key" not in pending.source
        code_skew.ack_pending_model_switch(pending.intent_id)
        assert code_skew.peek_pending_model_switch() is None

    @pytest.mark.asyncio
    async def test_fresh_gateway_replays_switch_in_original_topic(self, monkeypatch, tmp_path):
        from gateway.run import GatewayRunner

        state_file = tmp_path / ".pending_model_switches.json"
        monkeypatch.setattr(code_skew, "_pending_model_switches_file", lambda: state_file)
        intent = _pending_intent()
        code_skew.enqueue_pending_model_switch(intent)
        events = []

        class FakeAdapter:
            pass

        adapter = FakeAdapter()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._adapter_for_source = lambda source: adapter
        runner._normalize_source_for_session_key = lambda source: source
        runner._session_key_for_source = lambda source: "telegram:-1003418564280:3006"
        runner._session_model_overrides = {}
        completed = asyncio.Event()

        async def handle_model(event):
            events.append(event)
            await completed.wait()
            runner._session_model_overrides["telegram:-1003418564280:3006"] = {
                "model": intent.model_input,
                "provider": intent.provider,
            }
            return None

        runner._handle_model_command = handle_model
        replay_task = asyncio.create_task(runner._replay_pending_code_skew_model_switches())
        duplicate_replay_task = asyncio.create_task(runner._replay_pending_code_skew_model_switches())
        await asyncio.sleep(0)
        assert code_skew.peek_pending_model_switch() is not None
        assert not replay_task.done()
        assert not duplicate_replay_task.done()
        completed.set()
        await asyncio.gather(replay_task, duplicate_replay_task)

        assert len(events) == 1
        event = events[0]
        assert event.source.thread_id == "3006"
        assert event.source.chat_id == "-1003418564280"
        assert event.internal is True
        assert event.text == "/model openai/gpt-5.6 --provider openai-codex --session"
        assert code_skew.peek_pending_model_switch() is None

    def test_peek_keeps_request_until_explicit_ack(self, monkeypatch, tmp_path):
        state_file = tmp_path / ".pending_model_switches.json"
        monkeypatch.setattr(code_skew, "_pending_model_switches_file", lambda: state_file)
        intent = _pending_intent()
        code_skew.enqueue_pending_model_switch(intent)

        assert code_skew.peek_pending_model_switch() == intent
        assert code_skew.peek_pending_model_switch() == intent
        code_skew.ack_pending_model_switch(intent.intent_id)
        assert code_skew.peek_pending_model_switch() is None

    def test_rejects_nested_source_metadata(self):
        raw = _pending_intent().to_dict()
        raw["source"]["chat_name"] = {"api_key": "must-not-persist"}
        assert code_skew.PendingModelSwitch.from_dict(raw) is None

    @pytest.mark.asyncio
    async def test_replay_retains_request_when_model_handler_does_not_commit(self, monkeypatch, tmp_path):
        from gateway.run import GatewayRunner

        state_file = tmp_path / ".pending_model_switches.json"
        monkeypatch.setattr(code_skew, "_pending_model_switches_file", lambda: state_file)
        intent = _pending_intent()
        code_skew.enqueue_pending_model_switch(intent)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._adapter_for_source = lambda source: object()
        runner._normalize_source_for_session_key = lambda source: source
        runner._session_key_for_source = lambda source: "telegram:-1003418564280:3006"
        runner._session_model_overrides = {}

        async def handle_model(event):
            return None  # Mirrors an adapter-delivered error path with no commit.

        runner._handle_model_command = handle_model
        await runner._replay_pending_code_skew_model_switches()

        assert code_skew.peek_pending_model_switch() == intent

    @pytest.mark.asyncio
    async def test_replay_keeps_request_when_source_adapter_is_unavailable(self, monkeypatch, tmp_path):
        from gateway.run import GatewayRunner

        state_file = tmp_path / ".pending_model_switches.json"
        monkeypatch.setattr(code_skew, "_pending_model_switches_file", lambda: state_file)
        intent = _pending_intent()
        code_skew.enqueue_pending_model_switch(intent)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._adapter_for_source = lambda source: None
        await runner._replay_pending_code_skew_model_switches()

        assert code_skew.peek_pending_model_switch() == intent
