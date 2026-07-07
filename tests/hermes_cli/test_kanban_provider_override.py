from __future__ import annotations

from argparse import Namespace
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


def test_task_to_dict_includes_provider_and_model_overrides() -> None:
    from hermes_cli import kanban as kc

    task = kb.Task(
        id="t_provider",
        title="provider routed",
        body=None,
        assignee="reviewer",
        status="todo",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        provider_override="openai-codex",
        model_override="gpt-5.5",
    )

    payload = kc._task_to_dict(task)
    assert payload["provider_override"] == "openai-codex"
    assert payload["model_override"] == "gpt-5.5"


def test_show_prints_provider_and_model_overrides(tmp_path, monkeypatch, capsys) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    from hermes_cli import kanban as kc

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="provider routed task",
            assignee="reviewer",
            provider_override="openai-codex",
            model_override="gpt-5.5",
        )

    rc = kc._cmd_show(
        Namespace(task_id=task_id, json=False, state_type=None, state_name=None)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "  provider:  openai-codex" in out
    assert "  model:     gpt-5.5" in out
