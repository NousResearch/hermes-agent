"""A completion receipt may opt into a silent continuation without muting later human work."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.notification_presentation import event_presentation_muted, notification_turn
from agent.session_persistence import _db_flush_collect
from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.platforms.event import MessageEvent
from gateway.warning_notifications import diagnostic_metadata, diagnostic_turn_muted
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_kanban_wake_acceptance import drain, setup_route


@pytest.mark.asyncio
@pytest.mark.parametrize("silent", [False, True])
@pytest.mark.parametrize("outcome", ["completed", "blocked", "mixed", "mixed_hidden_diagnostic"])
async def test_checkpointed_completion_continues_once_without_repeat_presentation(tmp_path, monkeypatch, silent, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    hidden_diagnostic = outcome == "mixed_hidden_diagnostic"
    (tmp_path / "config.yaml").write_text(
        f"kanban: {{silent_completion_continuation: {str(silent).lower()}}}\n"
        f"display: {{suppress_warning_notifications: {str(hidden_diagnostic).lower()}}}\n"
    )
    runner, adapter, source, key = setup_route()
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title="work", assignee="worker", session_id=key)
        kbn.add_notify_sub(conn, task_id=task, platform="telegram", chat_id="42", user_id="42",
                          chat_type="dm", delivery_mode="notify+wake")
        if outcome != "completed":
            kb.block_task(conn, task, reason="Blocked", kind="transient" if hidden_diagnostic else "needs_input")
        if outcome != "blocked":
            if outcome != "completed":
                kb.unblock_task(conn, task)
            kb.complete_task(conn, task, summary="handoff")
    muted = silent and outcome == "completed"
    passive_count = 2 if outcome == "mixed" else 1

    async def tick():
        deliveries = await asyncio.to_thread(_notifier_collect, runner, kb, notifier_profile=None,
                                             gc_due=False, gc_retention_days=30)
        for delivery in deliveries:
            await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()

    work, observed = [], {}
    async def handler(event):
        if not event.internal:
            return "Independent subsequent result"
        work.append(event)
        metadata = diagnostic_metadata(event)
        observed["muted"] = diagnostic_turn_muted(metadata, Platform.TELEGRAM)
        observed["category"] = event.metadata["notification_category"]
        agent = SimpleNamespace(session_id="fixture", _last_flushed_db_idx=0)
        prompt = {"role": "user", "content": "completion receipt", "display_metadata": metadata}
        with notification_turn(agent, muted=observed["muted"], session_id="fixture"):
            observed["freeform_muted"] = event_presentation_muted("message.delta", "fixture")
            observed["controls_muted"] = [event_presentation_muted(control, "fixture")
                for control in ("approval.request", "clarify.request", "connection.request")]
            observed["rows"], _ = _db_flush_collect(agent,
                [prompt, {"role": "assistant", "content": "repeat"}], [])
        observed["still_muted"] = event_presentation_muted("message.delta", "fixture")
        observed["next_rows"], _ = _db_flush_collect(agent,
            [{"role": "assistant", "content": "next result"}], [])
        return "Repeated completion"

    await adapter.connect()
    try:
        await tick()  # Passive send succeeds, but no handler exists yet: wake must be retried.
        assert len(adapter.wire) == passive_count
        with kbc.connect_closing() as conn:
            row = kbn.list_notify_subs(conn, task)[0]
            assert row["last_ping_event_id"] > row["last_event_id"]
        adapter.set_message_handler(handler)
        await tick()
        await drain(adapter)
        await tick()
        expected_category = "delivered" if muted else "result"
        assert [ev.metadata["notification_category"] for ev in work] == (
            ["diagnostic", expected_category] if hidden_diagnostic else [expected_category]
        )
        assert observed["muted"] is muted
        assert observed["freeform_muted"] is muted
        assert observed["controls_muted"] == [False, False, False]
        if muted:
            assert observed["rows"][0]["display_metadata"]["notification_category"] == "delivered"
        assert not observed["still_muted"]
        assert observed["next_rows"][0]["display_kind"] is None
        from persistence.conversation.messages import NewMessage
        from persistence_assembly import open_persistence, profile_scope
        from tests.fixtures.storage_read import transcript_rows
        from tests.fixtures.storage_seed import seed_session
        with open_persistence(profile_scope(home=tmp_path)) as persistence:
            seed_session(persistence, "fixture", "telegram")
            persistence.messages.append_batch("fixture", [NewMessage.from_live(row) for row in observed["rows"]])
            history = transcript_rows(persistence, "fixture")
        assert runner._crash_left_reply(history, 0, source) == ("" if muted else "repeat")
        assert len(adapter.wire) == passive_count + (0 if muted else 1)
        await adapter._process_message_background(MessageEvent(text="next request", source=source), key)
        assert adapter.wire[-1] == "Independent subsequent result"
    finally:
        await adapter.disconnect()
