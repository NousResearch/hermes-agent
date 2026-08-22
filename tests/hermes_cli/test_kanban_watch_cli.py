"""Behavior tests for task-scoped, bounded Kanban watch mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb.init_db()
    return home


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = parser.parse_args(["kanban", *argv])
    return kc.kanban_command(args)


def test_watch_until_status_returns_immediately_for_done_task(
    kanban_home, capsys
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="already done",
            initial_status="running",
        )
        kb.complete_task(conn, task_id, result="ok")

    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "done,blocked,archived",
            "--timeout",
            "1",
        ]
    )

    assert rc == 0
    assert f"{task_id} reached status done" in capsys.readouterr().out


def test_watch_until_status_exits_after_task_transitions_to_done(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="finishes later",
            initial_status="running",
        )

    sleep_calls = 0

    def complete_on_first_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            with kb.connect_closing() as conn:
                kb.complete_task(conn, task_id, result="ok")
            return
        raise AssertionError("watch did not exit after the task reached done")

    monkeypatch.setattr(kc.time, "sleep", complete_on_first_sleep)
    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "done,blocked,archived",
            "--timeout",
            "2",
            "--interval",
            "0.05",
        ]
    )

    assert rc == 0
    assert sleep_calls == 1
    assert f"{task_id} reached status done" in capsys.readouterr().out


def test_watch_timeout_is_bounded_and_does_not_mutate_task(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="keeps running",
            initial_status="running",
        )
        status_before_watch = kb.get_task(conn, task_id).status

    def unexpected_sleep(_seconds: float) -> None:
        raise AssertionError("watch slept after its deadline")

    monkeypatch.setattr(kc.time, "sleep", unexpected_sleep)
    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "done,blocked,archived",
            "--timeout",
            "0",
        ]
    )

    assert rc == 124
    assert f"timed out waiting for {task_id}" in capsys.readouterr().err
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == status_before_watch


def test_watch_unknown_until_status_is_rejected(kanban_home, capsys) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="never finishes",
            initial_status="running",
        )

    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "nope,also_nope",
            "--timeout",
            "0",
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown status" in err
    assert "nope" in err


def test_watch_negative_timeout_is_rejected(kanban_home, capsys) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="never finishes",
            initial_status="running",
        )

    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "done",
            "--timeout",
            "-5",
        ]
    )

    assert rc == 2
    assert "--timeout must be >= 0" in capsys.readouterr().err


def test_watch_missing_task_is_rejected(kanban_home, capsys) -> None:
    rc = _run(
        [
            "watch",
            "--task",
            "t_does_not_exist",
            "--until-status",
            "done",
            "--timeout",
            "0",
        ]
    )

    assert rc == 1
    assert "no such task" in capsys.readouterr().err


def test_watch_is_allowed_in_delegated_child_context(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="observer only",
            initial_status="running",
        )
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    rc = _run(
        [
            "watch",
            "--task",
            task_id,
            "--until-status",
            "done,blocked,archived",
            "--timeout",
            "0",
        ]
    )

    # Policy §3.1: delegated children MAY use read-only K0 verbs (watch is one
    # of them). The verb is served over a genuine mode=ro connection, bypassing
    # init_db entirely so the in-flight backfill's write_txn never runs.
    assert rc == 124
    assert "timed out" in capsys.readouterr().err


def test_delegated_child_k0_list_is_allowed(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="visible to child", initial_status="running")
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    rc = _run(["list"])

    assert rc == 0
    assert "visible to child" in capsys.readouterr().out


def test_delegated_child_k0_show_is_allowed(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="readable task",
            initial_status="running",
        )
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    rc = _run(["show", task_id])

    assert rc == 0
    assert "readable task" in capsys.readouterr().out


def test_delegated_child_k0_connection_is_genuinely_readonly(kanban_home) -> None:
    conn = kc._k0_readonly_connect()
    try:
        with pytest.raises(Exception):
            conn.execute("INSERT INTO tasks (id, title) VALUES ('t_x', 'nope')")
    finally:
        conn.close()


def test_delegated_child_cannot_complete_a_task(
    kanban_home, capsys, monkeypatch
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="must not be mutated",
            initial_status="running",
        )
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    rc = _run(["complete", task_id, "--result", "forbidden"])

    assert rc == 1
    assert "delegate_task" in capsys.readouterr().err
