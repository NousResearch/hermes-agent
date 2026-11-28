"""Specialized Bot Chat migration and producer admission contracts."""
import json
from unittest.mock import Mock

import pytest
from cron import scheduler as sched, scheduler_delivery as delivery, delivery_queue as queue
from cron import bot_chat_legacy as legacy, delivery_outcome
from tools import bot_live_delivery as live


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(sched, "load_config", dict)
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    monkeypatch.setattr(live, "read_delivery_result", lambda *a: None)
    return tmp_path


def envelope(home):
    return {"id": "job", "name": "Daily", "execution_id": "actual-execution", "_bot_chat_pending": str(home)}


def test_old_minimal_envelope_uses_original_digest_not_generic_route(store, monkeypatch):
    job = dict(envelope(store), _failure_incident_id="incident")
    key = "a" * 64  # preserve persisted identity, even when not freshly derived
    queue.enqueue(key, job, "report", for_failure=True)
    calls = []
    def admit(home, owner, message, **kwargs):
        calls.append((home, message, kwargs))
        return dict(status="queued", message=message, notification_category="diagnostic")
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    generic = Mock(return_value=None)
    monkeypatch.setattr(sched, "_deliver_result", generic)
    alerted = Mock()
    monkeypatch.setattr(sched, "_mark_incident_alerted", alerted)
    settled = Mock()
    monkeypatch.setattr(delivery_outcome, "settle_quietly", settled)
    assert sched.drain_delivery_queue({}, None) == 1
    assert len(calls) == 1
    assert calls[0][2] == {"delivery_id": key, "notification_category": "diagnostic"}
    assert queue.get_status(key)["status"] == "pending"  # parked handoff, never eligible again
    generic.assert_not_called()
    alerted.assert_called_once_with("incident")
    settled.assert_not_called()  # digest must never be used as an execution ledger id


@pytest.mark.parametrize("origin", ["wrong", "", None, 42])
def test_wrong_origin_stays_pending_without_transport_or_scrub(store, monkeypatch, origin):
    key = "a" * 64
    queue.enqueue(key, envelope(store / "other"), "report")
    with queue._transaction() as conn:
        malformed = envelope(store / "other")
        malformed["_bot_chat_pending"] = origin
        conn.execute("UPDATE deliveries SET job_json=?", (json.dumps(malformed),))
    admit = Mock()
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    assert sched.drain_delivery_queue({}, None) == 1
    row = queue.get_status(key)
    assert row["status"] == "pending"
    assert row["content"] == "report"
    assert "profile mismatch" in row["error"]
    admit.assert_not_called()


@pytest.mark.parametrize("route", ["bot-chat", "bot-chat:research"])
def test_agent_queue_first_persistence_is_minimal_redacted_pinned(store, monkeypatch, route):
    from cron import jobs, executions
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: store / "profiles" / name)
    monkeypatch.setattr(delivery, "_redact_cron_payload", lambda text, what: text.replace("SECRET", "[redacted]"))
    monkeypatch.setattr(sched, "run_job", lambda job, **kw: (True, "raw", "body SECRET", None))
    job = jobs.create_job(prompt="private prompt", schedule="every 1h", deliver=route,
                          name="Daily SECRET")
    sched.run_one_job(job)
    execution = executions.latest_execution(job["id"])
    row = queue.get_status(execution["id"])
    persisted = json.loads(row["job_json"])
    assert "SECRET" not in row["content"] + row["job_json"]
    assert "prompt" not in persisted and "schedule" not in persisted
    assert persisted["execution_id"] == execution["id"]
    assert persisted["_bot_chat_pending"] == str(store)
    assert persisted["_bot_chat_destination"]
    assert row["payload_hash"]


def test_json_first_persistence_redacts_name_and_content(store, monkeypatch):
    monkeypatch.setattr(delivery, "_redact_cron_payload", lambda text, what: text.replace("SECRET", "[redacted]"))
    monkeypatch.setattr(live, "find_canonical_live_owner", Mock(side_effect=ValueError("not ready")))
    job = dict(id="job", name="Daily SECRET", execution_id="run", prompt="private prompt")
    delivery._deliver_to_bot_chat(job, "body SECRET", "")
    key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
    record = legacy.read_pending(key)
    assert "SECRET" not in json.dumps(record)
    assert set(record["job"]) == {"id", "name", "execution_id"}
    assert record["origin_home"] == str(store)


@pytest.mark.parametrize("owner", ["queued", "claimed", "settled", "ambiguous", "suppressed", "tombstone"])
def test_existing_owner_fences_direct_admission_and_payload_collision(store, monkeypatch, owner):
    job = dict(id="job", name="Daily", execution_id="run")
    key = delivery._bot_chat_delivery_key(job, store, store)
    if owner == "tombstone":
        queue.enqueue(key, envelope(store), "body", payload_hash=legacy.payload_hash(job, "body", store))
        queue.claim_next()
        monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
        queue._finish(key, error=None)
    else:
        record = legacy.defer(key, job, "body", store, for_failure=False)
        record["status"] = owner
        legacy.atomic_json_write(legacy._root() / (key + ".json"), record)
    admit = Mock()
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    delivery._deliver_to_bot_chat(dict(job), "body", "")
    admit.assert_not_called()
    error = delivery._deliver_to_bot_chat(dict(job, name="Renamed"), "body", "")
    assert error and "different payload" in error
    admit.assert_not_called()


@pytest.mark.parametrize("queued", [False, True])
def test_suppression_is_durable_across_policy_change(store, monkeypatch, queued):
    job = dict(id="job", name="Daily", execution_id="run")
    key = delivery._bot_chat_delivery_key(job, store, store)
    if queued:
        legacy.defer(key, job, "warning", store, for_failure=True)
    from gateway import warning_notifications
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", lambda *a: False)
    assert delivery._deliver_to_bot_chat(dict(job), "warning", "", for_failure=True) is None
    record = legacy.read_pending(key)
    assert record and record["status"] == "suppressed"
    assert record["payload_hash"] and "content" not in record and "job" not in record
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", lambda *a: True)
    admit = Mock()
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    assert delivery._deliver_to_bot_chat(dict(job), "warning", "", for_failure=True) is None
    admit.assert_not_called()


def test_new_direct_owner_is_durable_before_transport_without_holding_reservation(store, monkeypatch):
    job = dict(id="job", name="Daily", execution_id="run")
    key = delivery._bot_chat_delivery_key(job, store, store)
    def admit(home, owner, message, **kw):
        record = legacy.read_pending(key)
        assert record and record["status"] == "claimed"
        # A competing admission can acquire the reservation and sees exactly one owner.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            decision = pool.submit(legacy.arbitrate, key, record["payload_hash"]).result(timeout=3)
        assert decision["owner"] == "legacy"
        return dict(status="settled", message=message)
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    assert delivery._deliver_to_bot_chat(job, "body", "") is None
    assert legacy.read_pending(key)["status"] == "settled"


@pytest.mark.parametrize("case,retry", [("predial", True), ("draining", True), ("timeout", False),
    ("prose", False), ("held_hint", False), ("malformed", False), ("wrong_code", False)])
def test_typed_refusal_survives_dispatch_but_uncertainty_never_replays(store, monkeypatch, case, retry):
    from hermes_cli.gateway_client import GatewayRPCError
    errors = {
        "predial": live.PreDialError("not sent"),
        "draining": GatewayRPCError("runtime_draining", code=4001, data={"reason": "runtime_draining"}),
        "timeout": TimeoutError("not admitted"),
        "prose": ValueError("runtime_draining"),
        "held_hint": GatewayRPCError("runtime_coordination_required", code=4001,
            data={"reason": "runtime_coordination_required", "field": "session_writer", "admitted": False}),
        "malformed": GatewayRPCError("runtime_draining", code=4001, data={"reason": "other"}),
        "wrong_code": GatewayRPCError("runtime_draining", code=999, data={"reason": "runtime_draining"}),
    }
    key = "a" * 64
    queue.enqueue(key, envelope(store), "body")
    admit = Mock(side_effect=errors[case])
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    assert sched.drain_delivery_queue({}, None) == 1
    row = queue.get_status(key)
    assert row["status"] == ("pending" if retry else "unknown")
    assert row["retry_count"] == (1 if retry else 0)
    if retry:
        monkeypatch.setattr(queue.time, "time", lambda: row["next_attempt_at"])
        admit.side_effect = None
        admit.return_value = dict(status="settled", message=delivery.bot_chat_message(envelope(store), "body"))
        assert sched.drain_delivery_queue({}, None) == 1
        assert queue.get_status(key)["status"] == "delivered"
    else:
        assert sched.drain_delivery_queue({}, None) == 0
        admit.assert_called_once()


def test_direct_predial_refusal_keeps_visible_owner_retryable(store, monkeypatch):
    job = dict(id="job", execution_id="run")
    key = delivery._bot_chat_delivery_key(job, store, store)
    monkeypatch.setattr(live, "deliver_to_live_owner", Mock(side_effect=live.PreDialError("no dial")))
    result = delivery._deliver_to_bot_chat(job, "body", "")
    assert isinstance(result, queue.KnownNotAdmitted)
    assert legacy.read_pending(key)["status"] == "queued"


def test_canonical_sql_owner_fences_direct_and_drain_does_not_reenqueue(store, monkeypatch):
    job = dict(id="job", name="Daily", execution_id="execution")
    pinned, body = delivery._bot_chat_envelope(job, "body", "")
    queue.enqueue("execution", pinned, body)
    admit = Mock(return_value=dict(status="settled", message=delivery.bot_chat_message(job, body)))
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    delivery._deliver_to_bot_chat(dict(job), body, "")
    admit.assert_not_called()
    assert sched.drain_delivery_queue({}, None) == 1
    assert admit.call_args.kwargs["delivery_id"] == pinned["_bot_chat_delivery_id"]
    assert legacy.read_pending(pinned["_bot_chat_delivery_id"]) is None
    with queue._transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1
    delivery._deliver_to_bot_chat(dict(job), body, "")
    admit.assert_called_once()


def test_canonical_sql_producer_respects_legacy_owner(store):
    job = dict(id="job", name="Daily", execution_id="execution")
    pinned, body = delivery._bot_chat_envelope(job, "body", "")
    key = pinned["_bot_chat_delivery_id"]
    legacy.defer(key, job, body, store, for_failure=False, claimed=True)
    row = queue.enqueue("execution", pinned, body)
    assert row["status"] == "claimed"
    assert queue.get_status("execution") is None
    assert legacy.read_pending(key)["status"] == "claimed"


def test_mixed_route_keeps_required_fields_and_refusal_cannot_retry_whole_row(store, monkeypatch):
    job = dict(id="job", execution_id="execution", deliver="bot-chat,telegram:fixture", prompt="needed elsewhere")
    pinned, body = delivery._prepare_bot_chat_queue(job, "body")
    assert pinned == job and "_bot_chat_pending" not in pinned
    queue.enqueue("execution", pinned, body)
    calls = []
    def mixed(job, body, **kw):
        calls.append("external send")
        return queue.KnownNotAdmitted("Bot Chat refused")
    monkeypatch.setattr(sched, "_deliver_result", mixed)
    sched.drain_delivery_queue({}, None)
    assert queue.get_status("execution")["status"] == "failed"
    assert sched.drain_delivery_queue({}, None) == 0
    assert calls == ["external send"]


def test_canonical_handoff_is_not_inference_completion_after_bookkeeping(store, monkeypatch):
    job = dict(id="job", execution_id="execution")
    pinned, body = delivery._bot_chat_envelope(job, "body", "")
    queue.enqueue("execution", pinned, body)
    message = delivery.bot_chat_message(job, body)
    monkeypatch.setattr(live, "deliver_to_live_owner", lambda *a, **kw: dict(status="queued", message=message))
    assert sched.drain_delivery_queue({}, None) == 1
    assert queue.get_status("execution")["status"] == "pending"
    assert delivery_outcome.settled_outcome("execution") is None
    monkeypatch.setattr(live, "read_delivery_result", lambda *a: dict(status="settled", message=message))
    assert delivery_outcome.settled_outcome("execution") == "delivered"


def test_abandoned_digest_owner_never_settles_digest_as_execution(store, monkeypatch):
    key = "a" * 64
    queue.enqueue(key, envelope(store), "body")
    queue.claim_next()
    queue._ACTIVE_DELIVERIES.discard(key)
    settled = Mock()
    monkeypatch.setattr(delivery_outcome, "settle_quietly", settled)
    assert queue.recover_abandoned() == 1
    # Recovery must settle the retained original run, never the digest key.
    settled.assert_called_once_with("job", "actual-execution")


def test_sql_owner_not_ready_stays_pending_without_nested_json(store, monkeypatch):
    key = "a" * 64
    queue.enqueue(key, envelope(store), "body")
    monkeypatch.setattr(live, "find_canonical_live_owner", Mock(side_effect=ValueError("profile authority is not ready")))
    assert sched.drain_delivery_queue({}, None) == 1
    assert queue.get_status(key)["status"] == "pending"
    assert legacy.read_pending(key) is None


def test_legacy_drain_rereads_suppressed_owner_before_transport(store, monkeypatch):
    (store / "state.db").touch()
    job = dict(id="job", execution_id="run")
    key = delivery._bot_chat_delivery_key(job, store, store)
    legacy.defer(key, job, "body", store, for_failure=True)
    real_scan = legacy._queued
    def stale(root):
        snapshot = real_scan(root)
        legacy.defer(key, job, "body", store, for_failure=True, suppressed=True)
        return snapshot
    monkeypatch.setattr(legacy, "_queued", stale)
    admit = Mock()
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    legacy.drain_legacy_pending()
    admit.assert_not_called()
    assert legacy.read_pending(key)["status"] == "suppressed"


@pytest.mark.parametrize("typed", [False, True])
def test_legacy_transport_claim_and_strict_refusal(store, monkeypatch, typed):
    (store / "state.db").touch()
    key = "a" * 64
    legacy.defer(key, {"id": "job"}, "body", store, for_failure=False)
    seen = []
    def admit(*a, **kw):
        seen.append(legacy.read_pending(key)["status"])
        raise live.PreDialError("no dial") if typed else TimeoutError("uncertain")
    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    legacy.drain_legacy_pending()
    assert seen == ["claimed"]
    assert legacy.read_pending(key)["status"] == ("queued" if typed else "ambiguous")
    if not typed:
        legacy.drain_legacy_pending()
        assert seen == ["claimed"]
