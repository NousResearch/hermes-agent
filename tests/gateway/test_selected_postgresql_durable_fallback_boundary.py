"""Selected PostgreSQL refuses legacy durable mirror, recovery, and directory paths."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from state_store_runtime_readiness import PostgreSQLRuntimeActivationError, trap_state_db_opens


@pytest.fixture
def selected_pg_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


def test_selected_pg_mirror_public_and_direct_paths_refuse_before_legacy_artifacts(selected_pg_home, monkeypatch):
    from gateway import mirror

    sessions_path = selected_pg_home / "sessions" / "sessions.json"
    sessions_path.parent.mkdir()
    original = json.dumps({"telegram": {"session_id": "sid", "origin": {"platform": "telegram", "chat_id": "1"}}})
    sessions_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(mirror, "_SESSIONS_DIR", sessions_path.parent)
    monkeypatch.setattr(mirror, "_SESSIONS_INDEX", sessions_path)

    entrypoints = (
        lambda: mirror.mirror_to_session("telegram", "1", "must not mirror"),
        lambda: mirror._find_session_id("telegram", "1"),
        lambda: mirror._append_to_sqlite("sid", {"role": "assistant", "content": "must not append"}),
    )
    for call in entrypoints:
        with trap_state_db_opens(selected_pg_home) as events:
            with pytest.raises(PostgreSQLRuntimeActivationError):
                call()
        assert events == []
    assert sessions_path.read_text(encoding="utf-8") == original
    assert not (selected_pg_home / "state.db").exists()


def test_selected_pg_shutdown_flush_public_and_direct_paths_refuse_before_spool_or_replay(selected_pg_home):
    from gateway import shutdown_flush

    spool_dir = selected_pg_home / "pending_messages"
    spool_dir.mkdir()
    pending = spool_dir / "pending-existing.json"
    original = json.dumps({"session_key": "session", "data": {"text": "preserve", "session_id": "sid"}})
    pending.write_text(original, encoding="utf-8")
    replayed = []
    entrypoints = (
        lambda: shutdown_flush._get_flush_dir(),
        lambda: shutdown_flush._write_payload(spool_dir, {"data": {}}),
        lambda: shutdown_flush._flush_value(spool_dir, "pending", "session", "message"),
        lambda: shutdown_flush.flush_pending_to_file({"session": "message"}),
        lambda: shutdown_flush.flush_overflow_to_file({"session": ["message"]}),
        lambda: shutdown_flush.spool_dropped_transcript_message("sid", {"role": "user", "content": "message"}),
        lambda: shutdown_flush.drain_transcript_spool("sid", replayed.append),
        lambda: shutdown_flush.recover_pending_to_db(),
        lambda: shutdown_flush._recover_one_payload(object(), pending, {}),
        lambda: shutdown_flush.flush_agent_history_to_file("sid", [{"role": "user", "content": "message"}]),
    )
    for call in entrypoints:
        with trap_state_db_opens(selected_pg_home) as events:
            with pytest.raises(PostgreSQLRuntimeActivationError):
                call()
        assert events == []
    assert list(spool_dir.iterdir()) == [pending]
    assert pending.read_text(encoding="utf-8") == original
    assert replayed == []
    assert not (selected_pg_home / "state.db").exists()


def test_selected_pg_channel_directory_paths_refuse_before_cache_session_or_adapter_activity(selected_pg_home, monkeypatch):
    from gateway import channel_directory
    from gateway.config import Platform

    sessions_path = selected_pg_home / "sessions" / "sessions.json"
    sessions_path.parent.mkdir()
    sessions_path.write_text(json.dumps({"session": {"origin": {"platform": "telegram", "chat_id": "1"}}}), encoding="utf-8")
    directory_path = selected_pg_home / "channel_directory.json"
    directory_original = json.dumps({"updated_at": "before", "platforms": {"telegram": [{"id": "1", "name": "one"}]}})
    directory_path.write_text(directory_original, encoding="utf-8")
    monkeypatch.setattr(channel_directory, "DIRECTORY_PATH", directory_path)
    adapter_calls = []

    class Adapter:
        async def list_channels(self):
            adapter_calls.append("list")
            return [{"id": "1", "name": "one"}]

    entrypoints = (
        lambda: asyncio.run(channel_directory.build_channel_directory({Platform.TELEGRAM: Adapter()})),
        lambda: channel_directory._build_from_sessions("telegram"),
        lambda: channel_directory._build_from_sessions_db("telegram"),
        lambda: channel_directory._build_from_sessions_json("telegram"),
        lambda: channel_directory._load_json_dict(directory_path),
        lambda: channel_directory._read_json(directory_path),
        lambda: channel_directory.load_directory(),
    )
    for call in entrypoints:
        with trap_state_db_opens(selected_pg_home) as events:
            with pytest.raises(PostgreSQLRuntimeActivationError):
                call()
        assert events == []
    assert adapter_calls == []
    assert directory_path.read_text(encoding="utf-8") == directory_original
    assert sessions_path.exists()
    assert not (selected_pg_home / "state.db").exists()


def test_selected_pg_cron_seed_refuses_before_session_creation(selected_pg_home):
    from cron.scheduler_delivery import _seed_cron_session

    class SessionStore:
        def get_or_create_session(self, _source):
            raise AssertionError("selected PostgreSQL must not create a legacy cron session")

    class Adapter:
        _session_store = SessionStore()

    with trap_state_db_opens(selected_pg_home) as events:
        with pytest.raises(PostgreSQLRuntimeActivationError):
            _seed_cron_session(
                {"id": "job"}, Adapter(), "telegram", "1", "must not seed", thread_id=None,
                chat_type="dm", user_id=None, chat_name=None, scope_id=None,
            )
    assert events == []
    assert not (selected_pg_home / "state.db").exists()
