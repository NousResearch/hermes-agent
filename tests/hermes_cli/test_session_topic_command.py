"""CLI routing and lifecycle tests for /session-topic."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command
from hermes_state import SessionDB


class _TestCLI(CLICommandsMixin):
    def __init__(self, db: SessionDB):
        self._session_db = db
        self.session_id = "session"
        self.agent = SimpleNamespace(
            _ensure_db_session=lambda: None,
            _session_init_model_config={},
            _topic_segmentation_enabled=False,
            _active_topic_id=None,
            context_compressor=SimpleNamespace(),
        )


def _cli(db: SessionDB) -> _TestCLI:
    return _TestCLI(db)


def test_registry_keeps_telegram_topic_and_adds_cli_only_session_topic():
    telegram = resolve_command("topic")
    session_topic = resolve_command("session-topic")

    assert telegram is not None and telegram.gateway_only is True
    assert session_topic is not None and session_topic.cli_only is True
    assert resolve_command("session-topics") is session_topic


def test_cli_new_switch_off_persist_override_without_renaming(tmp_path, capsys):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session", source="cli", model="test")
        db.set_session_title("session", "User title")
        git = db.ensure_session_topic("session", "git")
        cli = _cli(db)

        cli._handle_session_topic_command("/session-topic new cooking")
        cooking = db.get_active_topic("session")
        assert cooking is not None and cooking["title"] == "cooking"
        assert cli.agent._topic_segmentation_enabled is True
        assert cli.agent.context_compressor._active_topic_id == cooking["id"]

        cli._handle_session_topic_command(f"/session-topic switch {git['id']}")
        active = db.get_active_topic("session")
        assert active is not None and active["id"] == git["id"]

        cli._handle_session_topic_command("/session-topic off")
        assert cli.agent._topic_segmentation_enabled is False
        assert cli.agent._active_topic_id is None
        assert db.get_session_model_config_value(
            "session", "topic_segmentation_enabled"
        ) is False
        assert db.get_session_title("session") == "User title"
        assert "Session topic segmentation disabled" in capsys.readouterr().out
    finally:
        db.close()


def test_cli_invalid_switch_preserves_active_topic(tmp_path, capsys):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session", source="cli", model="test")
        topic = db.ensure_session_topic("session", "git")
        cli = _cli(db)

        cli._handle_session_topic_command("/session-topic switch 999999")

        active = db.get_active_topic("session")
        assert active is not None and active["id"] == topic["id"]
        assert "does not belong" in capsys.readouterr().out
    finally:
        db.close()
