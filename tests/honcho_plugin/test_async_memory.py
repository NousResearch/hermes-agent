"""Tests for the async-memory Honcho improvements.

Covers:
  - write_frequency parsing (async / turn / session / int)
  - resolve_session_name with session_title
  - HonchoSessionManager.save() routing per write_frequency
  - async writer thread lifecycle and retry
  - flush_all() drains pending messages
  - shutdown() joins the thread
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch


from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    _ASYNC_SHUTDOWN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(**kwargs) -> HonchoSession:
    return HonchoSession(
        key=kwargs.get("key", "cli:test"),
        user_peer_id=kwargs.get("user_peer_id", "eri"),
        assistant_peer_id=kwargs.get("assistant_peer_id", "hermes"),
        honcho_session_id=kwargs.get("honcho_session_id", "cli-test"),
        messages=kwargs.get("messages", []),
    )


def _make_manager(write_frequency="turn") -> HonchoSessionManager:
    cfg = HonchoClientConfig(
        write_frequency=write_frequency,
        api_key="test-key",
        enabled=True,
    )
    mgr = HonchoSessionManager(config=cfg)
    mgr._honcho = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# write_frequency parsing from config file
# ---------------------------------------------------------------------------

class TestWriteFrequencyParsing:
    def test_string_async(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "async"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "async"

    def test_string_turn(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "turn"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "turn"

    def test_string_session(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "session"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "session"

    def test_integer_frequency(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": 5}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == 5

    def test_integer_string_coerced(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "3"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == 3

    def test_host_block_overrides_root(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "writeFrequency": "turn",
            "hosts": {"hermes": {"writeFrequency": "session"}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "session"

    def test_defaults_to_async(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "async"


# ---------------------------------------------------------------------------
# resolve_session_name with session_title
# ---------------------------------------------------------------------------

class TestResolveSessionNameTitle:
    def test_manual_override_beats_title(self):
        cfg = HonchoClientConfig(sessions={"/my/project": "manual-name"})
        result = cfg.resolve_session_name("/my/project", session_title="the-title")
        assert result == "manual-name"

    def test_title_beats_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="my-project")
        assert result == "my-project"

    def test_title_with_peer_prefix(self):
        cfg = HonchoClientConfig(peer_name="eri", session_peer_prefix=True)
        result = cfg.resolve_session_name("/some/dir", session_title="aeris")
        assert result == "eri-aeris"

    def test_title_sanitized(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="my project/name!")
        # trailing dashes stripped by .strip('-')
        assert result == "my-project-name"

    def test_title_all_invalid_chars_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="!!! ###")
        # sanitized to empty → falls back to dirname
        assert result == "dir"

    def test_none_title_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title=None)
        assert result == "dir"

    def test_empty_title_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="")
        assert result == "dir"

    def test_per_session_uses_session_id(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_per_session_with_peer_prefix(self):
        cfg = HonchoClientConfig(session_strategy="per-session", peer_name="eri", session_peer_prefix=True)
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "eri-20260309_175514_9797dd"

    def test_per_session_no_id_falls_back_to_dirname(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_id=None)
        assert result == "dir"

    def test_per_session_id_beats_title(self):
        # per-session: the run's session_id is authoritative; an (auto-)generated
        # title must NOT remap a live conversation onto a second Honcho session.
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_title="my-title", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_per_session_id_beats_manual_map(self):
        # per-session: session_id also wins over a stale cwd map entry (e.g. the
        # desktop launching from a mapped home dir).
        cfg = HonchoClientConfig(session_strategy="per-session", sessions={"/some/dir": "pinned"})
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_title_still_applies_for_non_per_session(self):
        # Outside per-session, /title still names the Honcho session.
        cfg = HonchoClientConfig(session_strategy="per-directory")
        result = cfg.resolve_session_name("/some/dir", session_title="my-title", session_id="20260309_175514_9797dd")
        assert result == "my-title"

    def test_gateway_key_beats_per_session_id(self):
        # Gateways keep per-chat isolation even in per-session.
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", gateway_session_key="agent:main:telegram:dm:42", session_id="20260309_175514_9797dd")
        assert result == "agent-main-telegram-dm-42"

    def test_global_strategy_returns_workspace(self):
        cfg = HonchoClientConfig(session_strategy="global", workspace_id="my-workspace")
        result = cfg.resolve_session_name("/some/dir")
        assert result == "my-workspace"


# ---------------------------------------------------------------------------
# save() routing per write_frequency
# ---------------------------------------------------------------------------

class TestSaveRouting:
    def _make_session_with_message(self, mgr=None):
        sess = _make_session()
        sess.add_message("user", "hello")
        sess.add_message("assistant", "hi")
        if mgr:
            mgr._cache[sess.key] = sess
        return sess

    def test_turn_flushes_immediately(self):
        mgr = _make_manager(write_frequency="turn")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_called_once_with(sess)

    def test_session_mode_does_not_flush(self):
        mgr = _make_manager(write_frequency="session")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_not_called()

    def test_async_mode_enqueues(self):
        mgr = _make_manager(write_frequency="async")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            # flush_session should NOT be called synchronously
            mock_flush.assert_not_called()
        assert not mgr._async_queue.empty()

    def test_int_frequency_flushes_on_nth_turn(self):
        mgr = _make_manager(write_frequency=3)
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)  # turn 1
            mgr.save(sess)  # turn 2
            assert mock_flush.call_count == 0
            mgr.save(sess)  # turn 3
            assert mock_flush.call_count == 1

    def test_int_frequency_skips_other_turns(self):
        mgr = _make_manager(write_frequency=5)
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            for _ in range(4):
                mgr.save(sess)
            assert mock_flush.call_count == 0
            mgr.save(sess)  # turn 5
            assert mock_flush.call_count == 1


# ---------------------------------------------------------------------------
# flush_all()
# ---------------------------------------------------------------------------

class TestFlushAll:
    def test_flushes_all_cached_sessions(self):
        mgr = _make_manager(write_frequency="session")
        s1 = _make_session(key="s1", honcho_session_id="s1")
        s2 = _make_session(key="s2", honcho_session_id="s2")
        s1.add_message("user", "a")
        s2.add_message("user", "b")
        mgr._cache = {"s1": s1, "s2": s2}

        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.flush_all()
            assert mock_flush.call_count == 2

    def test_flush_all_drains_async_queue(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "pending")

        with patch.object(mgr, "_flush_session") as mock_flush:
            # Put the item AFTER the mock is installed so the background
            # writer thread (if it dequeues before flush_all) still hits
            # the mock rather than the real _flush_session.
            mgr._async_queue.put(sess)
            mgr.flush_all()
            # Called at least once for the queued item
            assert mock_flush.call_count >= 1

    def test_flush_all_tolerates_errors(self):
        mgr = _make_manager(write_frequency="session")
        sess = _make_session()
        mgr._cache = {"key": sess}
        with patch.object(mgr, "_flush_session", side_effect=RuntimeError("oops")):
            # Should not raise
            mgr.flush_all()


# ---------------------------------------------------------------------------
# async writer thread lifecycle
# ---------------------------------------------------------------------------

class TestAsyncWriterThread:
    def test_thread_started_on_async_mode(self):
        mgr = _make_manager(write_frequency="async")
        assert mgr._async_thread is not None
        assert mgr._async_thread.is_alive()
        mgr.shutdown()

    def test_no_thread_for_turn_mode(self):
        mgr = _make_manager(write_frequency="turn")
        assert mgr._async_thread is None
        assert mgr._async_queue is None

    def test_shutdown_joins_thread(self):
        mgr = _make_manager(write_frequency="async")
        assert mgr._async_thread.is_alive()
        mgr.shutdown()
        assert not mgr._async_thread.is_alive()

    def test_shutdown_twice_is_idempotent(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session(key="shutdown-idempotent")
        sess.add_message("user", "before shutdown")
        mgr.save(sess)

        mgr.shutdown()
        assert mgr._shutdown_requested.is_set()
        assert mgr._shutdown_complete.is_set()
        assert mgr._async_queue is not None
        assert mgr._async_queue.unfinished_tasks == 0
        assert not mgr._async_thread.is_alive()

        mgr.shutdown()
        assert mgr._shutdown_complete.is_set()
        assert mgr._async_queue.unfinished_tasks == 0
        assert not mgr._async_thread.is_alive()

    def test_concurrent_shutdown_calls_enqueue_single_sentinel(self):
        mgr = _make_manager(write_frequency="async")

        sentinels_put = []
        put_lock = threading.Lock()

        original_put = mgr._async_queue.put

        def tracking_put(item, *args, **kwargs):
            if item is _ASYNC_SHUTDOWN:
                with put_lock:
                    sentinels_put.append(item)
            return original_put(item, *args, **kwargs)

        thread_done: list[threading.Event] = [threading.Event(), threading.Event()]

        def _run(idx: int):
            mgr.shutdown()
            thread_done[idx].set()

        with patch.object(mgr._async_queue, "put", side_effect=tracking_put):
            t1 = threading.Thread(target=_run, args=(0,))
            t2 = threading.Thread(target=_run, args=(1,))
            t1.start()
            t2.start()

            assert thread_done[0].wait(timeout=2.0)
            assert thread_done[1].wait(timeout=2.0)

            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

        assert len(sentinels_put) == 1
        assert not mgr._async_thread.is_alive()
        assert mgr._async_queue.unfinished_tasks == 0

    def test_async_writer_calls_flush(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "async msg")

        flushed = threading.Event()

        def capture(s):
            assert s is sess
            flushed.set()
            return True

        mgr._flush_session = capture
        mgr._async_queue.put(sess)
        assert flushed.wait(timeout=2.0)

        mgr.shutdown()
        assert flushed.is_set()

    def test_shutdown_sentinel_stops_loop(self):
        mgr = _make_manager(write_frequency="async")
        thread = mgr._async_thread
        mgr.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()

    def test_racing_shutdown_and_save(self):
        mgr = _make_manager(write_frequency="async")

        pre_shutdown_session = _make_session(key="pre-shutdown")
        pre_shutdown_session.add_message("user", "before shutdown")
        post_shutdown_session = _make_session(key="post-shutdown")
        post_shutdown_session.add_message("user", "after shutdown")

        pre_put = threading.Event()
        pre_flush_started = threading.Event()
        pre_flush_done = threading.Event()
        allow_flush = threading.Event()
        post_put = []

        original_flush = mgr._flush_session
        original_put = mgr._async_queue.put

        def tracking_flush(session: HonchoSession):
            if session is pre_shutdown_session:
                pre_flush_started.set()
                allow_flush.wait()
                pre_flush_done.set()
            return original_flush(session)

        def tracking_put(item, *args, **kwargs):
            if item is pre_shutdown_session:
                pre_put.set()
            if item is post_shutdown_session:
                post_put.append(item)
            return original_put(item, *args, **kwargs)

        mgr._flush_session = tracking_flush

        with patch.object(mgr._async_queue, "put", side_effect=tracking_put):
            pre_thread = threading.Thread(target=mgr.save, args=(pre_shutdown_session,))
            pre_thread.start()
            assert pre_put.wait(timeout=1.0)
            pre_thread.join(timeout=1.0)

            shutdown_thread = threading.Thread(target=mgr.shutdown)
            shutdown_thread.start()

            assert pre_flush_started.wait(timeout=1.0)
            post_thread = threading.Thread(target=mgr.save, args=(post_shutdown_session,))
            post_thread.start()

            allow_flush.set()
            shutdown_thread.join(timeout=2.0)
            post_thread.join(timeout=2.0)

        assert pre_flush_done.is_set()
        assert not post_put
        assert mgr._shutdown_requested.is_set()
        assert mgr._async_queue.unfinished_tasks == 0
        assert not mgr._async_thread.is_alive()

    def test_async_writer_and_flush_all_do_not_double_submit_unsynced_message(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "async duplicate test")
        mgr._cache[sess.key] = sess

        call_count = {"count": 0}
        call_count_lock = threading.Lock()
        first_submit_started = threading.Event()
        second_submit_seen = threading.Event()
        release_submit = threading.Event()
        flush_all_done = threading.Event()

        class _FakePeer:
            def message(self, content):
                return {"content": content}

        class _FakeHonchoSession:
            def add_messages(self, _messages):
                with call_count_lock:
                    call_count["count"] += 1
                    nth = call_count["count"]

                if nth == 1:
                    # Hold the first submission while the main test thread
                    # invokes flush_all() to create the race.
                    first_submit_started.set()
                    release_submit.wait()
                else:
                    # If a second concurrent submission enters, mark it and
                    # let both writers continue.
                    second_submit_seen.set()
                    release_submit.set()

        mgr._peers_cache[sess.user_peer_id] = _FakePeer()
        mgr._peers_cache[sess.assistant_peer_id] = _FakePeer()
        mgr._sessions_cache[sess.honcho_session_id] = _FakeHonchoSession()

        mgr._async_queue.put(sess)
        assert first_submit_started.wait(timeout=1.0)

        def _run_flush_all():
            mgr.flush_all()
            flush_all_done.set()

        flush_thread = threading.Thread(target=_run_flush_all, daemon=True)
        flush_thread.start()

        assert not second_submit_seen.wait(timeout=1.0)
        release_submit.set()
        assert flush_all_done.wait(timeout=1.0)

        flush_thread.join(timeout=2.0)
        mgr.shutdown()

        assert call_count["count"] == 1

    def test_flush_all_waits_for_queue_completion_when_retries_fail(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "pending")

        call_count = {"count": 0}
        failures = threading.Event()

        def always_fail(_session):
            call_count["count"] += 1
            if call_count["count"] >= 2:
                failures.set()
            raise RuntimeError("broken")

        mgr._flush_session = always_fail

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            flush_thread = threading.Thread(
                target=lambda: (mgr.flush_all()),
                name="flush-all-once",
                daemon=True,
            )
            flush_thread.start()
            assert failures.wait(timeout=1.5)
            flush_thread.join(timeout=1.5)

        assert call_count["count"] == 2
        mgr.shutdown()


class TestShutdownFlushForNonAsync:
    def test_shutdown_flushes_session_mode(self):
        mgr = _make_manager(write_frequency="session")
        sess = _make_session(key="non-async-session")
        sess.add_message("user", "shutdown persistence")
        mgr._cache = {"non-async-session": sess}

        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.shutdown()
            mock_flush.assert_called_once_with(sess)

    def test_shutdown_flushes_integer_cadence_before_threshold(self):
        mgr = _make_manager(write_frequency=3)
        sess = _make_session(key="non-async-int")
        sess.add_message("user", "shutdown pending")
        mgr._cache = {"non-async-int": sess}

        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)  # turn 1
            mgr.save(sess)  # turn 2
            assert mock_flush.call_count == 0

            mgr.shutdown()
            assert mock_flush.call_count == 1


# ---------------------------------------------------------------------------
# async retry on failure
# ---------------------------------------------------------------------------

class TestAsyncWriterRetry:
    def test_retries_once_on_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        second_call = threading.Event()

        def flaky_flush(s):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network blip")
            second_call.set()
            # second call succeeds silently
            return True

        mgr._flush_session = flaky_flush

        with patch("time.sleep"):  # skip the 2s sleep in retry
            mgr._async_queue.put(sess)
            assert second_call.wait(timeout=1.0)

        mgr.shutdown()
        assert call_count[0] == 2

    def test_drops_after_two_failures(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        second_attempt = threading.Event()

        def always_fail(s):
            call_count[0] += 1
            if call_count[0] >= 2:
                second_attempt.set()
            raise RuntimeError("always broken")

        mgr._flush_session = always_fail

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert second_attempt.wait(timeout=1.0)

        mgr.shutdown()
        # Should have tried exactly twice (initial + one retry) and not crashed
        assert call_count[0] == 2
        assert not mgr._async_thread.is_alive()

    def test_retries_when_flush_reports_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        second_call = threading.Event()

        def fail_then_succeed(_session):
            call_count[0] += 1
            if call_count[0] > 1:
                second_call.set()
            return call_count[0] > 1

        mgr._flush_session = fail_then_succeed

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert second_call.wait(timeout=1.0)

        mgr.shutdown()
        assert call_count[0] == 2


class TestProviderShutdownLifecycle:
    def test_provider_shutdown_stops_async_writer_and_flushes_pending_work(self):
        provider = HonchoMemoryProvider()
        provider._manager = _make_manager(write_frequency="async")

        sess = _make_session()
        sess.add_message("user", "queued by provider shutdown")
        provider._manager._cache[sess.key] = sess

        class _FakePeer:
            def message(self, content):
                return {"content": content}

        class _FakeHonchoSession:
            def __init__(self):
                self.added = threading.Event()

            def add_messages(self, _messages):
                self.added.set()

        fake_honcho_session = _FakeHonchoSession()

        provider._manager._peers_cache[sess.user_peer_id] = _FakePeer()
        provider._manager._peers_cache[sess.assistant_peer_id] = _FakePeer()
        provider._manager._sessions_cache[sess.honcho_session_id] = fake_honcho_session

        assert provider._manager._async_thread is not None and provider._manager._async_thread.is_alive()
        provider._manager._async_queue.put(sess)

        provider.shutdown()

        assert fake_honcho_session.added.wait(timeout=1.0)
        assert provider._manager._shutdown_requested.is_set()
        assert not provider._manager._async_thread.is_alive()
        assert sess.messages[0].get("_synced") is True


class TestMemoryFileMigrationTargets:
    def test_soul_upload_targets_ai_peer(self, tmp_path):
        mgr = _make_manager(write_frequency="turn")
        session = _make_session(
            key="cli:test",
            user_peer_id="custom-user",
            assistant_peer_id="custom-ai",
            honcho_session_id="cli-test",
        )
        mgr._cache[session.key] = session

        user_peer = MagicMock(name="user-peer")
        ai_peer = MagicMock(name="ai-peer")
        mgr._peers_cache[session.user_peer_id] = user_peer
        mgr._peers_cache[session.assistant_peer_id] = ai_peer

        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")
        (tmp_path / "USER.md").write_text("user profile", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("ai identity", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is True
        assert honcho_session.upload_file.call_count == 3

        peer_by_upload_name = {}
        for call_args in honcho_session.upload_file.call_args_list:
            payload = call_args.kwargs["file"]
            peer_by_upload_name[payload[0]] = call_args.kwargs["peer"]

        assert peer_by_upload_name["consolidated_memory.md"] is user_peer
        assert peer_by_upload_name["user_profile.md"] is user_peer
        assert peer_by_upload_name["agent_soul.md"] is ai_peer


# ---------------------------------------------------------------------------
# HonchoClientConfig dataclass defaults for new fields
# ---------------------------------------------------------------------------

class TestNewConfigFieldDefaults:
    def test_write_frequency_default(self):
        cfg = HonchoClientConfig()
        assert cfg.write_frequency == "async"

    def test_write_frequency_set(self):
        cfg = HonchoClientConfig(write_frequency="turn")
        assert cfg.write_frequency == "turn"


class TestPrefetchCacheAccessors:
    def test_set_and_pop_context_result(self):
        mgr = _make_manager(write_frequency="turn")
        payload = {"representation": "Known user", "card": "prefers concise replies"}

        mgr.set_context_result("cli:test", payload)

        assert mgr.pop_context_result("cli:test") == payload
        assert mgr.pop_context_result("cli:test") == {}
