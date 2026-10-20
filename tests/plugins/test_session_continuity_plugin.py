from pathlib import Path

import yaml

from hermes_state import SessionDB
from plugins.session_continuity import (
    build_continuity_context,
    continuity_status,
    pre_llm_call,
)


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _seed_session(db: SessionDB, sid: str, source: str = "cli", user_id: str = "") -> None:
    kwargs = {"user_id": user_id} if user_id else {}
    db.create_session(session_id=sid, source=source, model="test-model", **kwargs)


def test_returns_no_context_when_not_first_turn(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "prior")
        db.append_message("prior", "user", "remember this")

        assert build_continuity_context(
            session_db=db,
            current_session_id="current",
            is_first_turn=False,
        ) is None
    finally:
        db.close()


def test_returns_no_context_for_hidden_or_current_only_sessions(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "current")
        db.append_message("current", "user", "new turn")

        _seed_session(db, "archived")
        db.append_message("archived", "user", "hidden fact")
        db.set_session_archived("archived", True)

        _seed_session(db, "cron-run", source="cron")
        db.append_message("cron-run", "user", "cron fact")

        assert build_continuity_context(
            session_db=db,
            current_session_id="current",
            is_first_turn=True,
            conversation_history=[{"role": "user", "content": "new turn"}],
            user_message="new turn",
        ) is None
    finally:
        db.close()


def test_builds_bounded_block_from_previous_visible_session(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "older-session")
        db.append_message("older-session", "user", "older user detail")
        db.append_message("older-session", "assistant", "older assistant detail")

        _seed_session(db, "prior-session")
        db.set_session_title("prior-session", "Project Alpha")
        db.append_message("prior-session", "user", "first user fact")
        db.append_message("prior-session", "tool", "tool output secret", tool_name="terminal")
        db.append_message("prior-session", "assistant", "assistant answer")
        db.append_message(
            "prior-session",
            "assistant",
            "",
            tool_calls=[{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
        )
        db.append_message("prior-session", "user", "latest user fact " + ("x" * 500))

        _seed_session(db, "current")
        db.append_message("current", "user", "hello")

        block = build_continuity_context(
            session_db=db,
            current_session_id="current",
            is_first_turn=True,
            conversation_history=[{"role": "user", "content": "hello"}],
            user_message="hello",
            max_chars=700,
            message_limit=4,
        )

        assert block is not None
        assert len(block) <= 700
        assert "Session continuity" in block
        assert "prior-se" in block
        assert "Project Alpha" in block
        assert "first user fact" in block
        assert "assistant answer" in block
        assert "latest user fact" in block
        assert "tool output secret" not in block
        assert "older user detail" not in block
        assert "current" not in block
    finally:
        db.close()


def test_skips_when_history_already_has_prior_dialog(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "prior")
        db.append_message("prior", "user", "remember this")

        block = build_continuity_context(
            session_db=db,
            current_session_id="current",
            is_first_turn=True,
            conversation_history=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "new"},
            ],
            user_message="new",
        )

        assert block is None
    finally:
        db.close()


def test_scopes_recall_to_same_source_and_user_when_available(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "cli-session", source="cli", user_id="phil")
        db.append_message("cli-session", "user", "private cli detail")

        _seed_session(db, "telegram-other-user", source="telegram", user_id="other")
        db.append_message("telegram-other-user", "user", "other user's telegram detail")

        _seed_session(db, "telegram-same-user", source="telegram", user_id="phil")
        db.append_message("telegram-same-user", "user", "same user telegram detail")

        block = build_continuity_context(
            session_db=db,
            current_session_id="current",
            is_first_turn=True,
            conversation_history=[{"role": "user", "content": "hi"}],
            user_message="hi",
            platform="telegram",
            sender_id="phil",
        )

        assert block is not None
        assert "same user telegram detail" in block
        assert "private cli detail" not in block
        assert "other user's telegram detail" not in block
    finally:
        db.close()


def test_hook_return_is_user_message_context_only(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "prior")
        db.append_message("prior", "user", "continuity detail")

        result = pre_llm_call(
            session_id="current",
            is_first_turn=True,
            conversation_history=[{"role": "user", "content": "hi"}],
            user_message="hi",
            session_db=db,
        )

        assert result is not None
        assert set(result) == {"context"}
        assert "continuity detail" in result["context"]
    finally:
        db.close()


def test_status_helper_reports_candidate(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_session(db, "prior-status")
        db.set_session_title("prior-status", "Status Title")
        db.append_message("prior-status", "user", "hello")

        status = continuity_status(session_db=db)

        assert "enabled" in status
        assert "prior-st" in status
        assert "Status Title" in status
    finally:
        db.close()


def test_plugin_manager_loads_bundled_plugin_when_enabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["session_continuity"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.plugins import PluginManager

    mgr = PluginManager()
    mgr.discover_and_load()

    loaded = mgr._plugins.get("session_continuity")
    assert loaded is not None
    assert loaded.enabled
    assert "pre_llm_call" in loaded.hooks_registered
    assert "continuity" in loaded.commands_registered
