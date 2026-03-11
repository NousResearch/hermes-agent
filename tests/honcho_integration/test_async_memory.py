"""Tests for the async-memory Honcho improvements.

Covers:
  - write_frequency parsing (async / turn / session / int)
  - memory_mode parsing
  - resolve_session_name with session_title
  - HonchoSessionManager.save() routing per write_frequency
  - async writer thread lifecycle and retry
  - flush_all() drains pending messages
  - shutdown() joins the thread
  - memory_mode gating helpers (unit-level)
"""

import json
import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from honcho_integration.client import HonchoClientConfig
from honcho_integration.session import (
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


def _make_manager(write_frequency="turn", memory_mode="hybrid") -> HonchoSessionManager:
    cfg = HonchoClientConfig(
        write_frequency=write_frequency,
        memory_mode=memory_mode,
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
# memory_mode parsing from config file
# ---------------------------------------------------------------------------

class TestMemoryModeParsing:
    def test_hybrid(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "memoryMode": "hybrid"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "hybrid"

    def test_honcho_only(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "memoryMode": "honcho"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "honcho"

    def test_local_only(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "memoryMode": "local"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "local"

    def test_defaults_to_hybrid(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "hybrid"

    def test_host_block_overrides_root(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "memoryMode": "hybrid",
            "hosts": {"hermes": {"memoryMode": "honcho"}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "honcho"

    def test_object_form_sets_default_and_overrides(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {"hermes": {"memoryMode": {
                "default": "hybrid",
                "hermes": "honcho",
                "sentinel": "local",
            }}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "hybrid"
        assert cfg.peer_memory_mode("hermes") == "honcho"
        assert cfg.peer_memory_mode("sentinel") == "local"
        assert cfg.peer_memory_mode("unknown") == "hybrid"  # falls through to default

    def test_object_form_no_default_falls_back_to_hybrid(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {"hermes": {"memoryMode": {"hermes": "honcho"}}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "hybrid"
        assert cfg.peer_memory_mode("hermes") == "honcho"
        assert cfg.peer_memory_mode("other") == "hybrid"

    def test_global_string_host_object_override(self, tmp_path):
        """Host object form overrides global string."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "memoryMode": "local",
            "hosts": {"hermes": {"memoryMode": {"default": "hybrid", "hermes": "honcho"}}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.memory_mode == "hybrid"  # host default wins over global "local"
        assert cfg.peer_memory_mode("hermes") == "honcho"


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

    def test_title_beats_session_id(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_title="my-title", session_id="20260309_175514_9797dd")
        assert result == "my-title"

    def test_manual_beats_session_id(self):
        cfg = HonchoClientConfig(session_strategy="per-session", sessions={"/some/dir": "pinned"})
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "pinned"

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
        mgr._async_queue.put(sess)

        with patch.object(mgr, "_flush_session") as mock_flush:
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

    def test_async_writer_calls_flush(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "async msg")

        flushed = []

        def capture(s):
            flushed.append(s)
            return True

        mgr._flush_session = capture
        mgr._async_queue.put(sess)
        # Give the daemon thread time to process
        deadline = time.time() + 2.0
        while not flushed and time.time() < deadline:
            time.sleep(0.05)

        mgr.shutdown()
        assert len(flushed) == 1
        assert flushed[0] is sess

    def test_shutdown_sentinel_stops_loop(self):
        mgr = _make_manager(write_frequency="async")
        thread = mgr._async_thread
        mgr.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# async retry on failure
# ---------------------------------------------------------------------------

class TestAsyncWriterRetry:
    def test_retries_once_on_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]

        def flaky_flush(s):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network blip")
            # second call succeeds silently

        mgr._flush_session = flaky_flush

        with patch("time.sleep"):  # skip the 2s sleep in retry
            mgr._async_queue.put(sess)
            deadline = time.time() + 3.0
            while call_count[0] < 2 and time.time() < deadline:
                time.sleep(0.05)

        mgr.shutdown()
        assert call_count[0] == 2

    def test_drops_after_two_failures(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]

        def always_fail(s):
            call_count[0] += 1
            raise RuntimeError("always broken")

        mgr._flush_session = always_fail

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            deadline = time.time() + 3.0
            while call_count[0] < 2 and time.time() < deadline:
                time.sleep(0.05)

        mgr.shutdown()
        # Should have tried exactly twice (initial + one retry) and not crashed
        assert call_count[0] == 2
        assert not mgr._async_thread.is_alive()

    def test_retries_when_flush_reports_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]

        def fail_then_succeed(_session):
            call_count[0] += 1
            return call_count[0] > 1

        mgr._flush_session = fail_then_succeed

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            deadline = time.time() + 3.0
            while call_count[0] < 2 and time.time() < deadline:
                time.sleep(0.05)

        mgr.shutdown()
        assert call_count[0] == 2


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

    def test_memory_mode_default(self):
        cfg = HonchoClientConfig()
        assert cfg.memory_mode == "hybrid"

    def test_write_frequency_set(self):
        cfg = HonchoClientConfig(write_frequency="turn")
        assert cfg.write_frequency == "turn"

    def test_memory_mode_set(self):
        cfg = HonchoClientConfig(memory_mode="honcho")
        assert cfg.memory_mode == "honcho"

    def test_peer_memory_mode_falls_back_to_global(self):
        cfg = HonchoClientConfig(memory_mode="honcho")
        assert cfg.peer_memory_mode("any-peer") == "honcho"

    def test_peer_memory_mode_override(self):
        cfg = HonchoClientConfig(memory_mode="hybrid", peer_memory_modes={"hermes": "local"})
        assert cfg.peer_memory_mode("hermes") == "local"
        assert cfg.peer_memory_mode("other") == "hybrid"


class TestPrefetchCacheAccessors:
    def test_set_and_pop_context_result(self):
        mgr = _make_manager(write_frequency="turn")
        payload = {"representation": "Known user", "card": "prefers concise replies"}

        mgr.set_context_result("cli:test", payload)

        assert mgr.pop_context_result("cli:test") == payload
        assert mgr.pop_context_result("cli:test") == {}

    def test_set_and_pop_dialectic_result(self):
        mgr = _make_manager(write_frequency="turn")

        mgr.set_dialectic_result("cli:test", "Resume with toolset cleanup")

        assert mgr.pop_dialectic_result("cli:test") == "Resume with toolset cleanup"
        assert mgr.pop_dialectic_result("cli:test") == ""
