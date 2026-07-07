from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb


def test_task_round_trips_provider_override(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="provider routed task",
            assignee="reviewer",
            provider_override="openai-codex",
            model_override="gpt-5.5",
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.provider_override == "openai-codex"
    assert task.model_override == "gpt-5.5"
