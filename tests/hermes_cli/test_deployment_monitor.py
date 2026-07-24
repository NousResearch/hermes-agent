"""Tests for the deployment-monitor command helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_monitor_stores_indexed_kanban_card(kanban_home: Path) -> None:
    from hermes_cli import deployment_monitor as dm

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="push preview")
        task_id = dm.create_monitor(
            conn,
            url="https://preview.example.com/page",
            markers=["Ready text", "schema marker"],
            parent=parent,
            deadline_seconds=900,
            now=1000,
        )
        task = kb.get_task(conn, task_id)
        cfg = dm.parse_monitor_config(task.body or "")

    assert task.workflow_template_id == dm.TEMPLATE_ID
    assert task.current_step_key == "watching"
    assert task.status == "todo"
    assert task.assignee is None
    assert cfg.url == "https://preview.example.com/page"
    assert cfg.markers == ["Ready text", "schema marker"]
    assert cfg.deadline_at == 1900
    assert cfg.parent == parent


def test_tick_completes_monitor_when_url_contains_all_markers(
    kanban_home: Path,
) -> None:
    from hermes_cli import deployment_monitor as dm

    def fetcher(url: str) -> dm.FetchResult:
        assert url == "https://preview.example.com/page"
        return dm.FetchResult(
            status_code=200,
            body="<h1>Ready text</h1><script>schema marker</script>",
            final_url=url,
        )

    with kb.connect_closing() as conn:
        task_id = dm.create_monitor(
            conn,
            url="https://preview.example.com/page",
            markers=["Ready text", "schema marker"],
            deadline_seconds=900,
            now=1000,
        )

        result = dm.tick(conn, now=1010, fetcher=fetcher)

        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert result.completed == [task_id]
    assert result.blocked == []
    assert result.pending == []
    assert task.status == "done"
    assert run.status == "completed"
    assert run.outcome == "completed"
    assert "https://preview.example.com/page" in (run.summary or "")
    assert run.metadata["status_code"] == 200
    assert run.metadata["markers"] == ["Ready text", "schema marker"]


def test_tick_leaves_monitor_pending_before_deadline(
    kanban_home: Path,
) -> None:
    from hermes_cli import deployment_monitor as dm

    def fetcher(url: str) -> dm.FetchResult:
        return dm.FetchResult(
            status_code=404,
            body="Nothing is here yet",
            final_url=url,
        )

    with kb.connect_closing() as conn:
        task_id = dm.create_monitor(
            conn,
            url="https://preview.example.com/page",
            markers=["Ready text"],
            deadline_seconds=900,
            now=1000,
        )

        result = dm.tick(conn, now=1200, fetcher=fetcher)

        task = kb.get_task(conn, task_id)

    assert result.completed == []
    assert result.blocked == []
    assert result.pending == [task_id]
    assert task.status == "ready"


def test_tick_blocks_monitor_after_deadline(
    kanban_home: Path,
) -> None:
    from hermes_cli import deployment_monitor as dm

    def fetcher(url: str) -> dm.FetchResult:
        return dm.FetchResult(
            status_code=404,
            body="Nothing is here yet",
            final_url=url,
        )

    with kb.connect_closing() as conn:
        task_id = dm.create_monitor(
            conn,
            url="https://preview.example.com/page",
            markers=["Ready text"],
            deadline_seconds=60,
            now=1000,
        )

        result = dm.tick(conn, now=1061, fetcher=fetcher)

        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert result.completed == []
    assert result.blocked == [task_id]
    assert result.pending == []
    assert task.status == "blocked"
    assert task.block_kind == "transient"
    assert "timed out" in (run.summary or "")
    assert "status=404" in (run.summary or "")
