"""A kanban subscriber whose profile the multiplexer no longer serves (#154610).

A parked profile may be served again, so its subscription is rewound without spending the failure
budget; a deleted one counts each refusal and is dropped at ``MAX_SEND_FAILURES``.
"""

import asyncio
import shutil

import pytest

from gateway import kanban_watchers_notifier as notifier
from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification
from hermes_cli import kanban_db_connect as kbc
from tests.gateway.test_kanban_routed_transport import RecordingAdapter, collect, completion, setup_runner


def _subscribed(task):
    with kbc.connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (task,)).fetchone()[0]


@pytest.mark.parametrize("gone", ["parked", "deleted"])
def test_unserved_subscriber_is_rewound_when_parked_and_dropped_when_deleted(tmp_path, monkeypatch, gone):
    runner = setup_runner(tmp_path, monkeypatch)
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"] = {Platform.DISCORD: secondary}
    task = completion(mode="wake")
    yuki = tmp_path / ".hermes" / "profiles" / "yuki"
    if gone == "parked":
        (yuki / "gateway.parked").write_text("", encoding="utf-8")
    fail_counts = {}
    for _ in range(notifier.MAX_SEND_FAILURES):
        rows = collect(runner)
        assert [row["task"].id for row in rows] == [task]
        if gone == "deleted":
            shutil.rmtree(yuki, ignore_errors=True)

        async def tick():
            for row in rows:
                await _KanbanNotification(runner, row, platform_cls=Platform, sub_fail_counts=fail_counts).deliver()

        asyncio.run(tick())

    assert secondary.sent == secondary.handled == []
    if gone == "parked":
        assert fail_counts == {} and _subscribed(task) == 1
    else:
        assert _subscribed(task) == 0
