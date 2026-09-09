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
from unittest.mock import MagicMock, patch

import pytest


from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
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


# B8: managers are built ONLY through the make_manager fixture below. The old
# helper constructed the manager first and swapped in a MagicMock afterwards -
# the honcho property refreshes the client via get_honcho_client() on every
# access, so the late mock never protected flush paths and test messages were
# written to a live local Honcho (production incident, session cli-test).


@pytest.fixture
def make_manager(monkeypatch):
    """Factory: fake client is injected BEFORE the constructor, shutdown is
    guaranteed for every created manager (even on assertion failure)."""
    from plugins.memory.honcho import session as session_module

    client = MagicMock()
    monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: client)
    created = []

    def _make(
        write_frequency="turn",
        *,
        runtime_user_peer_name=None,
        **cfg_kwargs,
    ) -> HonchoSessionManager:
        cfg = HonchoClientConfig(
            write_frequency=write_frequency,
            api_key="test-key",
            enabled=True,
            **cfg_kwargs,
        )
        mgr = HonchoSessionManager(
            honcho=client,
            config=cfg,
            runtime_user_peer_name=runtime_user_peer_name,
        )
        created.append(mgr)
        return mgr

    _make.client = client
    yield _make
    for mgr in created:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# write_frequency parsing from config file
# ---------------------------------------------------------------------------

class TestWriteFrequencyParsing:
    def test_string_async(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "async"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "async"


    def test_integer_frequency(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": 5}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == 5


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


    def test_title_sanitized(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="my project/name!")
        # trailing dashes stripped by .strip('-')
        assert result == "my-project-name"


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

    def test_turn_flushes_immediately(self, make_manager):
        mgr = make_manager(write_frequency="turn")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_called_once_with(sess)

    def test_session_mode_does_not_flush(self, make_manager):
        mgr = make_manager(write_frequency="session")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_not_called()

    def test_async_mode_enqueues(self, make_manager):
        mgr = make_manager(write_frequency="async")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            # flush_session should NOT be called synchronously
            mock_flush.assert_not_called()
        assert not mgr._async_queue.empty()

    def test_int_frequency_flushes_on_nth_turn(self, make_manager):
        mgr = make_manager(write_frequency=3)
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)  # turn 1
            mgr.save(sess)  # turn 2
            assert mock_flush.call_count == 0
            mgr.save(sess)  # turn 3
            assert mock_flush.call_count == 1

    def test_int_frequency_skips_other_turns(self, make_manager):
        mgr = make_manager(write_frequency=5)
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
    def test_flushes_all_cached_sessions(self, make_manager):
        mgr = make_manager(write_frequency="session")
        s1 = _make_session(key="s1", honcho_session_id="s1")
        s2 = _make_session(key="s2", honcho_session_id="s2")
        s1.add_message("user", "a")
        s2.add_message("user", "b")
        mgr._cache = {"s1": s1, "s2": s2}

        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.flush_all()
            assert mock_flush.call_count == 2

    def test_flush_all_drains_async_queue(self, make_manager):
        mgr = make_manager(write_frequency="async")
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

    def test_flush_all_tolerates_errors(self, make_manager):
        mgr = make_manager(write_frequency="session")
        sess = _make_session()
        mgr._cache = {"key": sess}
        with patch.object(mgr, "_flush_session", side_effect=RuntimeError("oops")):
            # Should not raise
            mgr.flush_all()


# ---------------------------------------------------------------------------
# async writer thread lifecycle
# ---------------------------------------------------------------------------

class TestAsyncWriterThread:
    def test_thread_starts_lazily_on_first_enqueue(self, make_manager):
        # B8: constructing a manager must not spawn background work
        mgr = make_manager(write_frequency="async")
        assert mgr._async_queue is not None
        assert mgr._async_thread is None
        mgr.save(_make_session())
        assert mgr._async_thread is not None
        assert mgr._async_thread.is_alive()
        mgr.shutdown()

    def test_no_thread_for_turn_mode(self, make_manager):
        mgr = make_manager(write_frequency="turn")
        assert mgr._async_thread is None
        assert mgr._async_queue is None

    def test_shutdown_joins_thread(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        assert mgr._async_thread.is_alive()
        mgr.shutdown()
        assert not mgr._async_thread.is_alive()

    def test_async_writer_calls_flush(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        sess = _make_session()
        sess.add_message("user", "async msg")

        flushed = []
        flushed_event = threading.Event()

        def capture(session):
            flushed.append(session)
            flushed_event.set()
            return True

        mgr._flush_session = capture
        mgr._async_queue.put(sess)
        assert flushed_event.wait(timeout=10), "async writer never flushed"

        mgr.shutdown()
        assert len(flushed) == 1
        assert flushed[0] is sess

    def test_shutdown_sentinel_stops_loop(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        thread = mgr._async_thread
        mgr.shutdown()
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_shutdown_without_started_thread_is_noop(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr.shutdown()
        assert mgr._async_thread is None

    def test_stop_async_writer_joins_thread_without_flushing(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        sess = _make_session()
        sess.add_message("user", "must not be written")
        with mgr._cache_lock:
            mgr._cache[sess.key] = sess

        flushed = []
        mgr._flush_session = lambda session: flushed.append(session) or True

        thread = mgr._async_thread
        mgr.stop_async_writer()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert flushed == []

    def test_stop_async_writer_without_started_thread_is_noop(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr.stop_async_writer()
        assert mgr._async_thread is None


# ---------------------------------------------------------------------------
# async retry on failure
# ---------------------------------------------------------------------------

class TestAsyncWriterRetry:
    def test_retries_once_on_failure(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def flaky_flush(session):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network blip")
            retry_done.set()
            return True

        mgr._flush_session = flaky_flush

        with patch("time.sleep"):  # skip the 2s sleep in retry
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        assert call_count[0] == 2

    def test_drops_after_two_failures(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def always_fail(session):
            call_count[0] += 1
            if call_count[0] >= 2:
                retry_done.set()
            raise RuntimeError("always broken")

        mgr._flush_session = always_fail

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        # Should have tried exactly twice (initial + one retry) and not crashed
        assert call_count[0] == 2
        assert not mgr._async_thread.is_alive()

    def test_retries_when_flush_reports_failure(self, make_manager):
        mgr = make_manager(write_frequency="async")
        mgr._ensure_async_writer()
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def fail_then_succeed(session):
            call_count[0] += 1
            if call_count[0] >= 2:
                retry_done.set()
            return call_count[0] > 1

        mgr._flush_session = fail_then_succeed

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        assert call_count[0] == 2


def _prime_migration_session(mgr, key, honcho_session_id, ai_peer_id="custom-ai"):
    """Cache a session whose user peer is what the REAL resolver returns for
    this manager — exactly what get_or_create stores — so the owner gate is
    tested against reachable states, not hand-picked peer ids."""
    session = _make_session(
        key=key,
        user_peer_id=mgr._resolve_user_peer_id(key),
        assistant_peer_id=ai_peer_id,
        honcho_session_id=honcho_session_id,
    )
    mgr._cache[session.key] = session
    honcho_session = MagicMock()
    mgr._sessions_cache[session.honcho_session_id] = honcho_session
    return session, honcho_session


class TestMemoryFileMigrationTargets:
    def test_soul_upload_targets_ai_peer(self, tmp_path, make_manager):
        # peerName declares the owner; no runtime identity, so the session
        # resolves to the owner peer and migration proceeds.
        mgr = make_manager(write_frequency="turn", peer_name="custom-user")
        session, honcho_session = _prime_migration_session(mgr, "cli:test", "cli-test")
        assert session.user_peer_id == "custom-user"

        user_peer = MagicMock(name="user-peer")
        ai_peer = MagicMock(name="ai-peer")
        mgr._peers_cache[session.user_peer_id] = user_peer
        mgr._peers_cache[session.assistant_peer_id] = ai_peer

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


class TestMemoryFileMigrationOwnerGate:
    def test_non_owner_gateway_user_is_skipped(self, tmp_path, make_manager):
        """The shared-channel scenario: a declared owner exists, but the
        session was triggered by someone else's platform identity. The old
        gate (re-resolving the session's own peer) passed here."""
        mgr = make_manager(
            write_frequency="turn",
            peer_name="owner-user",
            runtime_user_peer_name="some-other-human",
        )
        session, honcho_session = _prime_migration_session(
            mgr, "discord:shared", "shared-chan"
        )
        assert session.user_peer_id == "some-other-human"

        (tmp_path / "MEMORY.md").write_text("owner facts", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is False
        assert honcho_session.upload_file.call_count == 0

    def test_no_declared_owner_with_gateway_identity_is_skipped(
            self, tmp_path, make_manager):
        """Without peerName nobody messaging through a gateway can be proven
        to be the owner — migration must not run."""
        mgr = make_manager(
            write_frequency="turn",
            runtime_user_peer_name="discord-123",
        )
        session, honcho_session = _prime_migration_session(
            mgr, "discord:shared", "shared-chan"
        )

        (tmp_path / "MEMORY.md").write_text("owner facts", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is False
        assert honcho_session.upload_file.call_count == 0

    def test_no_declared_owner_single_operator_migrates(self, tmp_path, make_manager):
        """No peerName and no runtime identity is the plain CLI install —
        the only person who exists is the operator the files describe."""
        mgr = make_manager(write_frequency="turn")
        session, honcho_session = _prime_migration_session(mgr, "cli:test", "cli-test")
        mgr._peers_cache[session.user_peer_id] = MagicMock()
        mgr._peers_cache[session.assistant_peer_id] = MagicMock()

        (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is True
        assert honcho_session.upload_file.call_count == 1

    def test_aliased_owner_identity_migrates(self, tmp_path, make_manager):
        """An alias mapping the owner's platform ID onto peerName makes that
        gateway identity the owner."""
        mgr = make_manager(
            write_frequency="turn",
            peer_name="owner-user",
            user_peer_aliases={"discord-999": "owner-user"},
            runtime_user_peer_name="discord-999",
        )
        session, honcho_session = _prime_migration_session(
            mgr, "discord:dm", "discord-dm"
        )
        assert session.user_peer_id == "owner-user"
        mgr._peers_cache[session.user_peer_id] = MagicMock()
        mgr._peers_cache[session.assistant_peer_id] = MagicMock()

        (tmp_path / "USER.md").write_text("user profile", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is True
        assert honcho_session.upload_file.call_count == 1

    def test_pinned_peer_name_migrates(self, tmp_path, make_manager):
        """pinPeerName collapses every identity onto the owner peer by
        explicit config, so the files land on the peer they describe."""
        mgr = make_manager(
            write_frequency="turn",
            peer_name="owner-user",
            pin_peer_name=True,
            runtime_user_peer_name="anyone-at-all",
        )
        session, honcho_session = _prime_migration_session(
            mgr, "discord:shared", "shared-chan"
        )
        assert session.user_peer_id == "owner-user"
        mgr._peers_cache[session.user_peer_id] = MagicMock()
        mgr._peers_cache[session.assistant_peer_id] = MagicMock()

        (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is True
        assert honcho_session.upload_file.call_count == 1


# ---------------------------------------------------------------------------
# HonchoClientConfig dataclass defaults for new fields
# ---------------------------------------------------------------------------

class TestNewConfigFieldDefaults:
    def test_write_frequency_default(self):
        cfg = HonchoClientConfig()
        assert cfg.write_frequency == "async"


class TestPrefetchCacheAccessors:
    def test_set_and_pop_context_result(self, make_manager):
        mgr = make_manager(write_frequency="turn")
        payload = {"representation": "Known user", "card": "prefers concise replies"}

        mgr.set_context_result("cli:test", payload)

        assert mgr.pop_context_result("cli:test") == payload
        assert mgr.pop_context_result("cli:test") == {}

