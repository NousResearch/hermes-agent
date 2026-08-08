from __future__ import annotations

import argparse
import sqlite3
from argparse import Namespace
from types import SimpleNamespace

import hermes_cli.memory_reset as memory_reset_module
from hermes_cli.memory_reset import cmd_memory_reset
from hermes_cli.subcommands.memory import build_memory_parser
from hermes_state import SessionDB

_SEARCH_NEEDLE = "memoryresetneedle"
_TELEGRAM_CHAT_ID = "208214988"
_TELEGRAM_USER_ID = "208214988"
_TELEGRAM_THREAD_ID = "17585"


def _fts_schema_objects(home) -> set[tuple[str, str]]:
    with sqlite3.connect(home / "state.db") as conn:
        return {
            (row[0], row[1])
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name LIKE 'messages_fts%' ORDER BY type, name"
            )
        }


def _seed_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    sessions = home / "sessions"
    memories.mkdir(parents=True)
    sessions.mkdir()
    (memories / "MEMORY.md").write_text("remember this", encoding="utf-8")
    (memories / "USER.md").write_text("user profile", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    db = SessionDB(home / "state.db")
    db.create_session("session-1", "cli")
    db.append_message("session-1", "user", f"hello {_SEARCH_NEEDLE}")
    db.create_session("session-2", "telegram", user_id=_TELEGRAM_USER_ID)
    db.append_message("session-2", "assistant", "world")
    db.set_session_archived("session-2", True)
    db.create_session("session-child", "tool", parent_session_id="session-1")
    db.append_message("session-child", "assistant", "delegated work")
    db.set_meta("memory-reset-preservation", "keep")
    db.save_gateway_routing_entry(
        "route-1", '{"session_id":"session-1"}', scope="test-scope"
    )
    db.enable_telegram_topic_mode(chat_id=_TELEGRAM_CHAT_ID, user_id=_TELEGRAM_USER_ID)
    db.bind_telegram_topic(
        chat_id=_TELEGRAM_CHAT_ID,
        thread_id=_TELEGRAM_THREAD_ID,
        user_id=_TELEGRAM_USER_ID,
        session_key="telegram:dm:208214988:topic:17585",
        session_id="session-2",
    )
    assert db.search_messages(_SEARCH_NEEDLE)
    db.close()

    (sessions / "session-1.jsonl").write_text("transcript", encoding="utf-8")
    (sessions / "session-child.json").write_text("transcript", encoding="utf-8")
    (sessions / "request_dump_session-2_001.json").write_text(
        "request", encoding="utf-8"
    )
    (sessions / "unrelated.jsonl").write_text("keep", encoding="utf-8")
    return home, _fts_schema_objects(home)


def _assert_seed_still_present(home):
    db = SessionDB(home / "state.db")
    try:
        assert db.session_count(include_archived=True) == 3
        assert db.message_count() == 3
        assert db.get_meta("memory-reset-preservation") == "keep"
        assert db.search_messages(_SEARCH_NEEDLE)
        assert db.is_telegram_topic_mode_enabled(
            chat_id=_TELEGRAM_CHAT_ID, user_id=_TELEGRAM_USER_ID
        )
        assert (
            db.get_telegram_topic_binding(
                chat_id=_TELEGRAM_CHAT_ID, thread_id=_TELEGRAM_THREAD_ID
            )
            is not None
        )
    finally:
        db.close()


def _assert_conversations_cleared_and_state_preserved(home, fts_objects):
    db = SessionDB(home / "state.db")
    try:
        assert db.session_count(include_archived=True) == 0
        assert db.message_count() == 0
        assert not db.search_messages(_SEARCH_NEEDLE)
        assert db.get_meta("memory-reset-preservation") == "keep"
        assert db.load_gateway_routing_entries(scope="test-scope") == {
            "route-1": '{"session_id":"session-1"}'
        }
        assert db.is_telegram_topic_mode_enabled(
            chat_id=_TELEGRAM_CHAT_ID, user_id=_TELEGRAM_USER_ID
        )
        assert (
            db.get_telegram_topic_binding(
                chat_id=_TELEGRAM_CHAT_ID, thread_id=_TELEGRAM_THREAD_ID
            )
            is None
        )
    finally:
        db.close()

    assert _fts_schema_objects(home) == fts_objects
    assert not (home / "sessions" / "session-1.jsonl").exists()
    assert not (home / "sessions" / "session-child.json").exists()
    assert not (home / "sessions" / "request_dump_session-2_001.json").exists()
    assert (home / "sessions" / "unrelated.jsonl").read_text(encoding="utf-8") == "keep"


def test_memory_reset_parser_invokes_exactly_one_handler(monkeypatch):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    legacy_calls: list[str] = []
    conversation_calls: list[str] = []

    def legacy_handler(args):
        legacy_calls.append(args.target)
        return 0

    def conversation_handler(args):
        conversation_calls.append(args.target)
        return 0

    monkeypatch.setattr(memory_reset_module, "cmd_memory_reset", conversation_handler)
    build_memory_parser(subparsers, cmd_memory=legacy_handler)

    legacy_args = parser.parse_args(["memory", "reset", "--target", "all"])
    conversation_args = parser.parse_args([
        "memory",
        "reset",
        "--target",
        "conversations",
        "--yes",
    ])

    assert legacy_args.func(legacy_args) == 0
    assert legacy_calls == ["all"]
    assert conversation_calls == []
    assert conversation_args.func(conversation_args) == 0
    assert legacy_calls == ["all"]
    assert conversation_calls == ["conversations"]


def test_conversations_target_clears_history_and_preserves_other_state(
    tmp_path, monkeypatch
):
    home, fts_objects = _seed_home(tmp_path, monkeypatch)

    assert cmd_memory_reset(Namespace(target="conversations", yes=True)) == 0

    assert (home / "memories" / "MEMORY.md").is_file()
    assert (home / "memories" / "USER.md").is_file()
    _assert_conversations_cleared_and_state_preserved(home, fts_objects)


def test_conversation_reset_uses_bounded_delete_batches(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(memory_reset_module, "_SESSION_DELETE_BATCH", 2)

    db = SessionDB(home / "state.db")
    for index in range(5):
        session_id = f"batch-session-{index}"
        db.create_session(session_id, "cli")
        db.append_message(session_id, "user", f"message {index}")
    db.close()

    assert cmd_memory_reset(Namespace(target="conversations", yes=True)) == 0

    db = SessionDB(home / "state.db")
    try:
        assert db.session_count(include_archived=True) == 0
        assert db.message_count() == 0
    finally:
        db.close()


def test_gateway_probe_checks_profile_and_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    probed_homes = []

    def fake_liveness(*, profile_dir, use_cache):
        assert use_cache is False
        probed_homes.append(profile_dir)
        return SimpleNamespace(
            running=profile_dir == root.resolve(),
            pid=4242 if profile_dir == root.resolve() else None,
            probe_error=False,
        )

    monkeypatch.setattr("gateway.status.resolve_gateway_liveness", fake_liveness)

    assert memory_reset_module._get_running_gateway_pid(profile) == 4242
    assert probed_homes == [profile.resolve(), root.resolve()]


def test_running_gateway_blocks_reset_without_touching_state(tmp_path, monkeypatch):
    home, _fts_objects = _seed_home(tmp_path, monkeypatch)

    monkeypatch.setattr(
        memory_reset_module, "_get_running_gateway_pid", lambda _home: 4242
    )

    assert cmd_memory_reset(Namespace(target="conversations", yes=True)) == 1
    assert (home / "memories" / "MEMORY.md").is_file()
    assert (home / "memories" / "USER.md").is_file()
    _assert_seed_still_present(home)


def test_confirmation_denied_leaves_state_untouched(tmp_path, monkeypatch):
    home, _fts_objects = _seed_home(tmp_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    assert cmd_memory_reset(Namespace(target="conversations", yes=False)) == 0
    _assert_seed_still_present(home)


def test_conversations_target_does_not_create_empty_database(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert cmd_memory_reset(Namespace(target="conversations", yes=True)) == 0
    assert not (home / "state.db").exists()
