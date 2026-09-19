"""Behavior contracts for age-gated tool-payload retention.

The retention sweep must only replace old tool output, preserve every other
message row, and remain disabled unless explicitly configured.
"""

import time

from hermes_state import SessionDB


_TOMBSTONE = "[elided: tool output older than 30d - retention policy]"


def _message_snapshot(db):
    return [
        dict(row)
        for row in db._read_all(
            "SELECT id, role, content, tool_name, tool_calls, timestamp "
            "FROM messages ORDER BY id"
        )
    ]


def test_elide_old_tool_payloads_preserves_recent_and_non_tool_rows(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        now = 2_000_000_000.0
        cutoff = now - 30 * 86400
        old_tool = db.append_message(
            "s1",
            role="tool",
            content="sensitive old terminal output",
            tool_name="terminal",
            timestamp=cutoff - 1,
        )
        recent_tool = db.append_message(
            "s1",
            role="tool",
            content="recent terminal output",
            tool_name="terminal",
            timestamp=cutoff + 1,
        )
        user = db.append_message(
            "s1", role="user", content="old user prompt", timestamp=cutoff - 1
        )
        assistant = db.append_message(
            "s1", role="assistant", content="old assistant reply", timestamp=cutoff - 1
        )
        already_elided = db.append_message(
            "s1", role="tool", content=_TOMBSTONE, timestamp=cutoff - 1
        )

        changed = db.elide_old_tool_payloads(
            retention_days=30, now=now, batch_size=2
        )

        assert changed == 1
        rows = {row["id"]: row for row in _message_snapshot(db)}
        assert rows[old_tool]["content"] == _TOMBSTONE
        assert rows[recent_tool]["content"] == "recent terminal output"
        assert rows[user]["content"] == "old user prompt"
        assert rows[assistant]["content"] == "old assistant reply"
        assert rows[already_elided]["content"] == _TOMBSTONE
        assert rows[old_tool]["role"] == "tool"
        assert rows[old_tool]["tool_name"] == "terminal"
        assert rows[old_tool]["timestamp"] == cutoff - 1

        # A second pass is a no-op: tombstones are stable and idempotent.
        assert db.elide_old_tool_payloads(retention_days=30, now=now) == 0
    finally:
        db.close()


def test_auto_maintenance_tool_payload_retention_is_opt_in(tmp_path, monkeypatch):
    import cli
    import hermes_cli.config
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "home")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        old_timestamp = time.time() - 31 * 86400
        db.append_message(
            "s1", role="tool", content="must remain by default", timestamp=old_timestamp
        )

        base_sessions = {
            "auto_prune": True,
            "retention_days": 90,
            "vacuum_after_prune": False,
            "min_interval_hours": 24,
            "min_vacuum_interval_days": 30,
            "tool_payload_retention_days": 0,
        }
        monkeypatch.setattr(
            hermes_cli.config,
            "load_config",
            lambda: {"sessions": dict(base_sessions)},
        )
        cli._run_state_db_auto_maintenance(db)
        assert db.get_messages("s1")[0]["content"] == "must remain by default"

        db.set_meta("last_auto_prune", "0")
        configured = dict(base_sessions, tool_payload_retention_days=30)
        monkeypatch.setattr(
            hermes_cli.config,
            "load_config",
            lambda: {"sessions": configured},
        )
        cli._run_state_db_auto_maintenance(db)
        assert db.get_messages("s1")[0]["content"] == (
            "[elided: tool output older than 30d - retention policy]"
        )
    finally:
        db.close()
