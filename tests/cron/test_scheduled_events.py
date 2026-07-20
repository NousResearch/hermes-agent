from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from cron.scheduled_events import ScheduledEventStore, dispatch_due_scheduled_events
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _event(*, event_id: str = "deadline-1", due_at: datetime | None = None) -> dict:
    return {
        "event_id": event_id,
        "subject_id": "user:local",
        "due_at": (due_at or datetime.now(timezone.utc)).isoformat(),
        "event_type": "idle_reconsideration",
        "correlation_id": "turn-123",
    }


def test_upsert_is_idempotent_and_snapshot_survives_restart(tmp_path):
    path = tmp_path / "scheduled_events.json"
    store = ScheduledEventStore(path)
    event = _event()

    first = store.upsert(**event)
    second = store.upsert(**event)
    restarted = ScheduledEventStore(path)

    assert first["generation"] == second["generation"]
    assert restarted.snapshot("user:local") == [second]


def test_changed_upsert_replaces_event_and_advances_generation(tmp_path):
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    first = store.upsert(**_event())
    changed = store.upsert(
        **_event(due_at=datetime.now(timezone.utc) + timedelta(hours=1))
    )

    assert changed["generation"] == first["generation"] + 1
    assert store.snapshot("user:local") == [changed]


def test_concurrent_due_claim_has_one_winner(tmp_path):
    path = tmp_path / "scheduled_events.json"
    ScheduledEventStore(path).upsert(**_event())

    def claim():
        return ScheduledEventStore(path).claim_due(limit=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))

    winners = [events for events in results if events]
    assert len(winners) == 1
    assert winners[0][0]["event_id"] == "deadline-1"


def test_cancelled_event_never_becomes_due(tmp_path):
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event())

    assert store.cancel("deadline-1") is True
    assert store.claim_due() == []
    assert store.cancel("deadline-1") is False


def test_claim_is_durable_and_overdue_event_is_not_replayed(tmp_path):
    path = tmp_path / "scheduled_events.json"
    ScheduledEventStore(path).upsert(**_event())

    claimed = ScheduledEventStore(path).claim_due()
    after_restart = ScheduledEventStore(path)

    assert len(claimed) == 1
    assert after_restart.claim_due() == []
    assert after_restart.snapshot("user:local")[0]["status"] == "claimed"


def test_plugin_context_exposes_host_neutral_scheduled_event_contract(tmp_path):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="consumer", source="test"), manager)
    store_path = tmp_path / "scheduled_events.json"
    context._scheduled_event_store_path = store_path

    saved = context.upsert_scheduled_event(**_event())
    capabilities = context.get_host_capabilities()

    assert context.scheduled_event_snapshot("user:local") == [saved]
    assert context.cancel_scheduled_event("deadline-1") is True
    assert capabilities["scheduled_events"] == {
        "available": True,
        "durable": True,
        "cancel": True,
        "dedupe": True,
    }
    assert "prompt" not in saved


def test_due_event_waits_for_a_registered_consumer(tmp_path, monkeypatch):
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: False)

    assert dispatch_due_scheduled_events(store=store) == 0
    assert store.snapshot("user:local")[0]["status"] == "pending"


def test_due_event_invokes_typed_hook_once(tmp_path, monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="consumer", source="test"), manager)
    received = []
    context.register_hook("scheduled_event_due", lambda **payload: received.append(payload))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event())

    assert dispatch_due_scheduled_events(store=store) == 1
    assert dispatch_due_scheduled_events(store=store) == 0
    assert len(received) == 1
    assert received[0]["event_id"] == "deadline-1"
    assert received[0]["subject_id"] == "user:local"
    assert received[0]["event_type"] == "idle_reconsideration"
    assert received[0]["correlation_id"] == "turn-123"
    assert received[0]["generation"] == 1
    assert "prompt" not in received[0]


def test_scheduler_tick_dispatches_events_without_cron_jobs(tmp_path, monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [])
    monkeypatch.setattr(scheduler, "dispatch_due_scheduled_events", lambda **_kwargs: 1)
    monkeypatch.setattr(scheduler, "_get_lock_paths", lambda: (tmp_path, tmp_path / "tick.lock"))

    assert scheduler.tick(verbose=False) == 1


def test_typed_outreach_request_is_delivered_and_receipted(tmp_path, monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="consumer", source="test"), manager)
    receipts = []
    context.register_hook(
        "scheduled_event_due",
        lambda **event: {
            "action_id": "outreach-1",
            "delivery_attempt_id": "attempt-1",
            "subject_id": event["subject_id"],
            "content": "A bounded proactive message.",
        },
    )
    context.register_hook(
        "scheduled_outreach_delivery", lambda **payload: receipts.append(payload)
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event())
    requests = []

    def deliver(event, request):
        requests.append((event, request))
        return {"status": "delivered", "targets": [{"platform": "telegram"}]}

    assert dispatch_due_scheduled_events(store=store, deliver=deliver) == 1
    assert requests[0][1]["content"] == "A bounded proactive message."
    assert receipts == [
        {
            "event_id": "deadline-1",
            "subject_id": "user:local",
            "generation": 1,
            "claim_id": requests[0][0]["claim_id"],
            "action_id": "outreach-1",
            "delivery_attempt_id": "attempt-1",
            "status": "delivered",
            "targets": [{"platform": "telegram"}],
            "error": None,
            "telemetry_schema_version": "hermes.observer.v1",
        }
    ]


def test_invalid_outreach_request_is_not_delivered(tmp_path, monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="consumer", source="test"), manager)
    context.register_hook(
        "scheduled_event_due",
        lambda **_event: {"subject_id": "other-user", "content": "unsafe"},
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event())
    delivered = []

    dispatch_due_scheduled_events(
        store=store, deliver=lambda event, request: delivered.append((event, request))
    )

    assert delivered == []


def test_due_hook_receives_same_subject_batch_without_cross_subject_data(tmp_path, monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="consumer", source="test"), manager)
    received = []
    context.register_hook("scheduled_event_due", lambda **event: received.append(event))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    store = ScheduledEventStore(tmp_path / "scheduled_events.json")
    store.upsert(**_event(event_id="a"))
    store.upsert(**_event(event_id="b"))
    other = _event(event_id="other")
    other["subject_id"] = "user:other"
    store.upsert(**other)

    assert dispatch_due_scheduled_events(store=store) == 3
    local = next(event for event in received if event["event_id"] == "a")

    assert {event["event_id"] for event in local["due_events"]} == {"a", "b"}
    assert {event["subject_id"] for event in local["due_events"]} == {"user:local"}
