"""End-to-end tests for the Kanban dead-letter notification recovery CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _dead_letter_notification(
    conn, *, task_id: str, subscription_id: str, event_id: int,
) -> None:
    assert kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=event_id,
        actions=[
            {
                "event_id": event_id,
                "action": "message",
                "payload": {"task_id": task_id, "kind": "completed"},
            },
        ],
    ) == 1
    for _ in range(kb.NOTIFICATION_MAX_ATTEMPTS):
        leased = kb.lease_notification_outbox(conn)
        assert len(leased) == 1
        item = leased[0]
        state = kb.fail_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=event_id,
            action="message",
            lease_token=item["lease_token"],
            error="test delivery failure",
        )
    assert state == "dead_letter"


def _outbox_state(conn, subscription_id: str) -> str:
    row = conn.execute(
        "SELECT state FROM kanban_notification_outbox WHERE subscription_id = ?",
        (subscription_id,),
    ).fetchone()
    assert row is not None
    return str(row["state"])


def test_notify_retry_requeues_only_requested_task_dead_letters(kanban_home, capsys):
    with kb.connect_closing() as conn:
        retry_task_id = kb.create_task(conn, title="retry this notification")
        retry_subscription_id = kb.add_notify_sub(
            conn,
            task_id=retry_task_id,
            platform="telegram",
            chat_id="retry-chat",
        )
        _dead_letter_notification(
            conn,
            task_id=retry_task_id,
            subscription_id=retry_subscription_id,
            event_id=1,
        )
        _dead_letter_notification(
            conn,
            task_id=retry_task_id,
            subscription_id=retry_subscription_id,
            event_id=2,
        )

        untouched_task_id = kb.create_task(conn, title="leave this notification dead")
        untouched_subscription_id = kb.add_notify_sub(
            conn,
            task_id=untouched_task_id,
            platform="telegram",
            chat_id="untouched-chat",
        )
        _dead_letter_notification(
            conn,
            task_id=untouched_task_id,
            subscription_id=untouched_subscription_id,
            event_id=1,
        )

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = parser.parse_args(["kanban", "notify-retry", retry_task_id])

    assert kc.kanban_command(args) == 0
    assert capsys.readouterr().out == (
        f"Requeued 2 dead-letter notification deliveries for task {retry_task_id}.\n"
    )

    with kb.connect_closing() as conn:
        assert _outbox_state(conn, retry_subscription_id) == "pending"
        assert _outbox_state(conn, untouched_subscription_id) == "dead_letter"


def test_notify_retry_localizes_output_and_help_for_simplified_chinese(
    kanban_home, monkeypatch,
):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="localized recovery")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="localized-chat",
        )
        _dead_letter_notification(
            conn,
            task_id=task_id,
            subscription_id=subscription_id,
            event_id=1,
        )

    monkeypatch.setenv("HERMES_LANGUAGE", "zh-Hans")

    output = kc.run_slash(f"notify-retry {task_id}")
    help_output = kc.run_slash("notify-retry --help")

    assert output == f"已为任务 {task_id} 重新排队 1 条死信通知投递。"
    assert "重新排队任务的死信通知投递" in help_output
    assert "要恢复的任务 ID" in help_output
