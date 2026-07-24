from __future__ import annotations

from pathlib import Path

from hegi.config import load_config, validate_config
from hegi.state import StateStore


def test_config_rejects_automatic_memory_writes(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_path = tmp_path / "hegi.yaml"
    config_path.write_text(
        """
enabled: true
telegram:
  chat_id: "-1001"
agents:
  - name: "헤헤"
    db_path: "source.db"
memory:
  auto_commit: true
  auto_draft: true
  require_professor_approval: false
""",
        encoding="utf-8",
    )
    errors = validate_config(load_config(config_path))
    assert any("auto_commit" in error for error in errors)
    assert any("auto_draft" in error for error in errors)
    assert any("require_professor_approval" in error for error in errors)


def test_state_delivery_is_idempotent_and_dead_letter_is_durable(tmp_path):
    store = StateStore(tmp_path / "state.db")
    store.record_delivery("meeting", 1, "hash", status="sent", platform_message_id="7")
    store.record_delivery("meeting", 1, "hash", status="sent", platform_message_id="7")
    store.add_dead_letter("telegram", {"part": 2}, "network", "meeting")
    assert store.delivered_parts("meeting") == {1}
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM report_delivery").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0] == 1
